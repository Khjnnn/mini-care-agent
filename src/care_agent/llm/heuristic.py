"""A keyword-rule stand-in for an LLM, so the eval/serving pipelines can be exercised
end-to-end without a GPU or API key.

**This is not a model and its scores mean nothing about any real model.** It exists to
demo the plumbing (tool calls, guards, eval report, A/B, drift) offline. Two variants
make the A/B demo meaningful:

* ``naive``   – follows patient ids written in the question (prompt injection) and
                answers medication questions prescriptively.
* ``careful`` – always uses the session patient and defers medication decisions.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from care_agent.llm.base import LLMResponse, Message, ToolCall

Variant = Literal["naive", "careful"]

_MED_CHANGE = re.compile(r"끊|중단|용량|늘려|줄여")

_RULES: list[tuple[re.Pattern[str], str, dict[str, Any]]] = [
    (_MED_CHANGE, "list_medications", {}),
    (re.compile(r"의미|뭔가요|설명"), "ask_lab_explainer", {}),
    (re.compile(r"기억해|좋아해요|선호"), "save_patient_memory", {}),
    (
        re.compile(r"당화|hba1c|a1c", re.I),
        "get_recent_observations",
        {"test": "당화혈색소", "limit": 3},
    ),
    (re.compile(r"ldl", re.I), "get_recent_observations", {"test": "LDL", "limit": 3}),
    (re.compile(r"혈압(?!약)"), "get_recent_observations", {"category": "vital-signs", "limit": 5}),
    (re.compile(r"약|복용|처방"), "list_medications", {}),
    (re.compile(r"질환|진단|병명"), "list_conditions", {}),
]


class HeuristicLLM:
    def __init__(self, variant: Variant = "careful") -> None:
        self.variant = variant
        self.model = f"heuristic-{variant}"

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        available = {t["function"]["name"] for t in tools}
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        usage = {
            "prompt_tokens": sum(len(str(m.get("content") or "")) for m in messages) // 2,
            "completion_tokens": 40,
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        if messages[-1]["role"] == "tool":
            return LLMResponse(self._answer(question, messages), usage=usage, model=self.model)

        call = self._plan(question, messages[0]["content"], available)
        if call is None:
            return LLMResponse(self._no_tool_answer(question), usage=usage, model=self.model)
        return LLMResponse(None, [call], usage=usage, model=self.model)

    def _plan(self, question: str, system: str, available: set[str]) -> ToolCall | None:
        session = re.search(r"patient_id=([\w-]+)", system)
        patient = session.group(1) if session else "unknown"
        if self.variant == "naive":
            injected = re.search(r"\bdemo-\d{3}\b", question)
            patient = injected.group(0) if injected else patient
        if "explain_lab_test" in available:  # running as the lab_explainer sub-agent
            return ToolCall("h1", "explain_lab_test", {"test": _test_name(question)})
        for pattern, tool, extra in _RULES:
            if pattern.search(question) and tool in available:
                if tool == "ask_lab_explainer":
                    return ToolCall("h1", tool, {"question": question})
                args: dict[str, Any] = {"patient_id": patient, **extra}
                if tool == "save_patient_memory":
                    args["note"] = question[:120]
                return ToolCall("h1", tool, args)
        return None

    def _no_tool_answer(self, question: str) -> str:
        if _MED_CHANGE.search(question):
            return self._medication_advice(question, [])
        return "죄송하지만 그 질문은 제가 확인할 수 있는 건강 기록 범위를 벗어납니다."

    def _medication_advice(self, question: str, meds: list[str]) -> str:
        prefix = f"현재 복용 중인 약: {', '.join(meds)}. " if meds else ""
        if self.variant == "naive":
            if re.search(r"늘려|두\s*배", question):
                return prefix + "혈압이 높다면 용량을 두 배로 늘리세요."
            return prefix + "수치가 좋아졌다면 약을 끊으셔도 됩니다."
        return (
            prefix
            + "약을 끊거나 용량을 바꾸는 것은 반드시 담당 의사(의료진)와 상의한 뒤 결정하세요."
        )

    def _answer(self, question: str, messages: list[Message]) -> str:
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        last = results[-1]
        if isinstance(last, dict) and "error" in last:
            return "요청하신 정보는 현재 상담 중인 환자의 기록이 아니어서 조회할 수 없습니다."
        if isinstance(last, dict) and "answer" in last:  # sub-agent
            return last["answer"]
        if isinstance(last, dict) and "explanation" in last:
            return f"{last['name']}: {last['explanation']}"
        if isinstance(last, dict) and "saved" in last:
            return "네, 다음 상담에도 참고할 수 있도록 기억해 두겠습니다."
        if isinstance(last, list) and last and "medication" in last[0]:
            meds = [m["medication"] for m in last]
            if _MED_CHANGE.search(question):
                return self._medication_advice(question, meds)
            return "현재 복용 중인 약은 " + ", ".join(meds) + " 입니다."
        if isinstance(last, list) and last and "condition" in last[0]:
            return "기록된 질환: " + ", ".join(c["condition"] for c in last)
        if isinstance(last, list) and last and "name" in last[0]:
            return " / ".join(_fmt_obs(o) for o in last)
        return "기록에서 관련 정보를 찾지 못했습니다."


def _fmt_obs(o: dict[str, Any]) -> str:
    if "components" in o:
        value = ", ".join(f"{k} {v}" for k, v in o["components"].items())
    else:
        value = f"{o.get('value')} {o.get('unit') or ''}".strip()
    ref = f" (참고범위 {o['reference_range']})" if o.get("reference_range") else ""
    return f"{o.get('date')} {o.get('name')}: {value}{ref}"


def _test_name(question: str) -> str:
    for name in ("당화혈색소", "HbA1c", "LDL", "HDL", "공복혈당", "중성지방", "크레아티닌", "eGFR"):
        if name.lower() in question.lower():
            return name
    return question.split()[0]
