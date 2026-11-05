"""Safety gates that sit between the LLM and the world.

Week-1 scope (deterministic, no model calls):

* ``EmergencyGate``   – input gate. Possible emergencies short-circuit the agent with a
                        fixed "call 119" message instead of an LLM-generated answer.
* ``ToolGuard``       – tool-call gate. Enforces (1) the session's patient scope, so the
                        model cannot read another patient's record even if a prompt
                        injection asks it to, and (2) permission tiers: WRITE tools need
                        an explicit human approval callback, and are denied by default.

Later: PII masking on output, LLM-based policy classifier, definitive-diagnosis filter.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from care_agent.tools.registry import PATIENT_ARG, Permission, Tool

Approver = Callable[[Tool, dict[str, Any]], Awaitable[bool]]

EMERGENCY_RESPONSE = (
    "말씀하신 증상은 응급 상황일 수 있습니다. 지금 바로 119에 연락하거나 가까운 응급실을 방문하세요. "
    "자살·자해 생각이 드신다면 자살예방상담전화 109(24시간)로 연락해 주세요. "
    "이 상담 서비스는 응급 상황에 대응할 수 없습니다."
)

# Deliberately conservative: a false positive costs one redirect, a false negative can cost a life.
_EMERGENCY_PATTERNS = [
    r"가슴.{0,6}(통증|아프|아파|조이|답답|짓눌)",
    r"흉통",
    r"숨.{0,4}(안\s*쉬어|못\s*쉬|막히|차)",
    r"호흡\s*곤란",
    r"의식.{0,4}(없|잃)",
    r"쓰러졌",
    r"(한쪽|반쪽).{0,6}(마비|힘이\s*없)",
    r"말이\s*어눌",
    r"피를\s*토",
    r"토혈|객혈",
    r"경련",
    r"자살|자해|죽고\s*싶",
    r"(약|수면제).{0,8}(한꺼번에|많이|통째로)\s*(먹|삼)",
    r"\bchest pain\b",
    r"\bcan'?t breathe\b",
    r"\bsuicid",
    r"\boverdose\b",
]
_EMERGENCY_RE = re.compile("|".join(_EMERGENCY_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str | None = None
    matched: str | None = None


class EmergencyGate:
    def check(self, text: str) -> GateResult:
        m = _EMERGENCY_RE.search(text)
        if m:
            return GateResult(False, "possible_emergency", m.group(0))
        return GateResult(True)


class ToolGuard:
    """Per-session guard. Create one per (user request, patient)."""

    def __init__(self, patient_id: str, approver: Approver | None = None) -> None:
        self.patient_id = patient_id
        self.approver = approver

    async def check(self, tool: Tool, arguments: dict[str, Any]) -> GateResult:
        if tool.patient_scoped:
            requested = arguments.get(PATIENT_ARG)
            if requested is not None and str(requested) != self.patient_id:
                return GateResult(
                    False,
                    f"patient_scope_violation: this session may only access patient "
                    f"{self.patient_id!r}",
                )
        if tool.permission is Permission.WRITE:
            if self.approver is None:
                return GateResult(False, "write_requires_approval: no approver configured")
            if not await self.approver(tool, arguments):
                return GateResult(False, "write_rejected_by_human")
        return GateResult(True)
