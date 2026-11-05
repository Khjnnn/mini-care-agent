"""The agent loop: input gate → (LLM ↔ guarded tools)* → answer.

Design notes
* Tool failures, validation errors and safety denials are returned to the model as
  tool results instead of raising, so the model can recover (pick another tool,
  fix arguments, or explain to the user). Only the harness decides to stop.
* The loop is bounded by ``max_steps``; hitting it is a distinct, observable outcome.
* Everything the loop does is appended to ``trace``.
* The run's patient/approver live in a ``RunContext`` context variable so tools that
  start nested work (sub-agents) inherit the same patient scope without the model
  being able to choose it.
"""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from care_agent.agent.prompts import SYSTEM_PROMPT
from care_agent.agent.trace import TraceEvent
from care_agent.llm.base import LLM, Message, ToolCall
from care_agent.memory.store import MemoryStore
from care_agent.safety.gates import EMERGENCY_RESPONSE, Approver, EmergencyGate, ToolGuard
from care_agent.safety.output_gate import OutputGate
from care_agent.tools.registry import ToolRegistry

StopReason = Literal["final", "max_steps", "emergency"]

MAX_STEPS_ANSWER = "요청을 처리하는 중 단계 한도에 도달해 답변을 완성하지 못했습니다. 질문을 조금 더 구체적으로 해 주세요."
MAX_TOOL_RESULT_CHARS = 6000


@dataclass
class RunContext:
    agent: str
    patient_id: str
    approver: Approver | None
    depth: int
    trace: list[TraceEvent]
    step: int = 0


_current_run: ContextVar[RunContext | None] = ContextVar("care_agent_run", default=None)


def current_run() -> RunContext:
    ctx = _current_run.get()
    if ctx is None:
        raise RuntimeError("no agent run in progress")
    return ctx


@dataclass
class AgentResult:
    answer: str
    stop_reason: StopReason
    steps: int
    trace: list[TraceEvent] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[str]:
        return [e.data["name"] for e in self.trace if e.type == "tool_call"]


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        *,
        name: str = "agent",
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 6,
        emergency_gate: EmergencyGate | None = None,
        output_gate: OutputGate | None = None,
        memory: MemoryStore | None = None,
        history_turns: int = 10,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.name = name
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.emergency_gate = emergency_gate or EmergencyGate()
        self.output_gate = output_gate or OutputGate()
        self.memory = memory
        self.history_turns = history_turns

    async def run(
        self,
        question: str,
        *,
        patient_id: str,
        approver: Approver | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        trace: list[TraceEvent] = []
        if self.memory and session_id:
            self.memory.open_session(session_id, patient_id)  # raises on patient mismatch

        result = await self._run(question, patient_id, approver, session_id, trace)

        if self.memory and session_id:
            self.memory.add_turn(session_id, "user", question)
            self.memory.add_turn(session_id, "assistant", result.answer)
        return result

    async def _run(
        self,
        question: str,
        patient_id: str,
        approver: Approver | None,
        session_id: str | None,
        trace: list[TraceEvent],
    ) -> AgentResult:
        gate = self.emergency_gate.check(question)
        trace.append(
            TraceEvent("input_gate", {"allowed": gate.allowed, "matched": gate.matched}, step=0)
        )
        if not gate.allowed:
            return AgentResult(EMERGENCY_RESPONSE, "emergency", 0, trace)

        parent = _current_run.get()
        ctx = RunContext(
            agent=self.name,
            patient_id=patient_id,
            approver=approver,
            depth=parent.depth + 1 if parent else 0,
            trace=trace,
        )
        token = _current_run.set(ctx)
        try:
            return await self._loop(question, ctx, session_id)
        finally:
            _current_run.reset(token)

    async def _loop(self, question: str, ctx: RunContext, session_id: str | None) -> AgentResult:
        trace = ctx.trace
        guard = ToolGuard(ctx.patient_id, ctx.approver)
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt.format(patient_id=ctx.patient_id)}
        ]
        if self.memory and session_id:
            messages += [
                {"role": t.role, "content": t.content}
                for t in self.memory.recent_turns(session_id, self.history_turns)
            ]
        messages.append({"role": "user", "content": question})
        tool_specs = self.tools.to_openai()

        for step in range(1, self.max_steps + 1):
            ctx.step = step
            t0 = time.perf_counter()
            resp = await self.llm.complete(messages, tool_specs)
            trace.append(
                TraceEvent(
                    "llm",
                    {
                        "model": resp.model,
                        "usage": resp.usage,
                        "content": resp.content,
                        "tool_calls": [tc.name for tc in resp.tool_calls],
                        **({"routing": resp.meta["routing"]} if "routing" in resp.meta else {}),
                    },
                    step=step,
                    duration_ms=_ms(t0),
                )
            )
            messages.append(resp.as_message())

            if not resp.tool_calls:
                checked = self.output_gate.check(resp.content or "")
                if checked.modified:
                    trace.append(
                        TraceEvent(
                            "output_gate",
                            {"masked": checked.masked, "prescriptive": checked.prescriptive},
                            step=step,
                        )
                    )
                trace.append(TraceEvent("final", {"answer": checked.text}, step=step))
                return AgentResult(checked.text, "final", step, trace, messages)

            for call in resp.tool_calls:
                result = await self._execute(call, guard, trace, step)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": _to_text(result)}
                )

        return AgentResult(MAX_STEPS_ANSWER, "max_steps", self.max_steps, trace, messages)

    async def _execute(
        self, call: ToolCall, guard: ToolGuard, trace: list[TraceEvent], step: int
    ) -> Any:
        tool = self.tools.get(call.name)
        if tool is None:
            err = f"unknown tool {call.name!r}. available: {self.tools.names()}"
            trace.append(TraceEvent("tool_error", {"name": call.name, "error": err}, step))
            return {"error": err}
        if call.parse_error:
            trace.append(
                TraceEvent("tool_error", {"name": call.name, "error": call.parse_error}, step)
            )
            return {"error": call.parse_error}

        decision = await guard.check(tool, call.arguments)
        if not decision.allowed:
            trace.append(
                TraceEvent(
                    "tool_denied",
                    {"name": call.name, "arguments": call.arguments, "reason": decision.reason},
                    step,
                )
            )
            return {"error": f"denied by safety gate: {decision.reason}"}

        t0 = time.perf_counter()
        try:
            result = await tool.invoke(call.arguments)
        except ValidationError as e:
            err = f"invalid arguments: {e.errors(include_url=False)}"
            trace.append(TraceEvent("tool_error", {"name": call.name, "error": err}, step))
            return {"error": err}
        except Exception as e:  # tool failures are reported to the model, not raised
            err = f"{type(e).__name__}: {e}"
            trace.append(TraceEvent("tool_error", {"name": call.name, "error": err}, step))
            return {"error": err}

        trace.append(
            TraceEvent(
                "tool_call",
                {"name": call.name, "arguments": call.arguments, "result": result},
                step,
                duration_ms=_ms(t0),
            )
        )
        return result


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def _to_text(result: Any) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + " …(truncated; narrow the query)"
    return text
