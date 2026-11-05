"""The agent loop: input gate → (LLM ↔ guarded tools)* → answer.

Design notes
* Tool failures, validation errors and safety denials are returned to the model as
  tool results instead of raising, so the model can recover (pick another tool,
  fix arguments, or explain to the user). Only the harness decides to stop.
* The loop is bounded by ``max_steps``; hitting it is a distinct, observable outcome.
* Everything the loop does is appended to ``trace``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from care_agent.agent.prompts import SYSTEM_PROMPT
from care_agent.agent.trace import TraceEvent
from care_agent.llm.base import LLM, Message, ToolCall
from care_agent.safety.gates import EMERGENCY_RESPONSE, Approver, EmergencyGate, ToolGuard
from care_agent.tools.registry import ToolRegistry

StopReason = Literal["final", "max_steps", "emergency"]

MAX_STEPS_ANSWER = "요청을 처리하는 중 단계 한도에 도달해 답변을 완성하지 못했습니다. 질문을 조금 더 구체적으로 해 주세요."
MAX_TOOL_RESULT_CHARS = 6000


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
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 6,
        emergency_gate: EmergencyGate | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.emergency_gate = emergency_gate or EmergencyGate()

    async def run(
        self, question: str, *, patient_id: str, approver: Approver | None = None
    ) -> AgentResult:
        trace: list[TraceEvent] = []

        gate = self.emergency_gate.check(question)
        trace.append(
            TraceEvent("input_gate", {"allowed": gate.allowed, "matched": gate.matched}, step=0)
        )
        if not gate.allowed:
            return AgentResult(EMERGENCY_RESPONSE, "emergency", 0, trace)

        guard = ToolGuard(patient_id, approver)
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt.format(patient_id=patient_id)},
            {"role": "user", "content": question},
        ]
        tool_specs = self.tools.to_openai()

        for step in range(1, self.max_steps + 1):
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
                    },
                    step=step,
                    duration_ms=_ms(t0),
                )
            )
            messages.append(resp.as_message())

            if not resp.tool_calls:
                answer = resp.content or ""
                trace.append(TraceEvent("final", {"answer": answer}, step=step))
                return AgentResult(answer, "final", step, trace, messages)

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
