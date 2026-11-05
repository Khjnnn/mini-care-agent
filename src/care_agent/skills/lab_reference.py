"""Example skill: general (non-patient) knowledge about lab tests.

Also serves as the reference for how a team ships a skill as a plugin module: expose
``register(registry, platform)`` and list the module under ``plugins`` in the platform
config. See ``docs/adding-a-skill.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from care_agent.knowledge.lab_catalog import CATALOG, by_loinc, lookup
from care_agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from care_agent.platform.core import Platform


def register(registry: ToolRegistry, platform: Platform | None = None) -> None:
    @registry.register
    async def explain_lab_test(
        test: Annotated[
            str,
            Field(description="검사 이름(한글/영문) 또는 LOINC 코드. 예: 당화혈색소, LDL, 4548-4"),
        ],
    ) -> dict[str, Any]:
        """검사 항목의 의미와 일반적인 참고범위를 설명한다. 환자 기록과 무관한 일반 의학 정보다."""
        info = by_loinc(test) or lookup(test)
        if info is None:
            return {"error": f"unknown test {test!r}", "known": [t.name_ko for t in CATALOG]}
        return {
            "name": f"{info.name_ko} ({info.name_en})",
            "loinc": info.loinc,
            "unit": info.unit,
            "typical_reference_range": _range(info.ref_low, info.ref_high),
            "explanation": info.explanation,
            "note": "실제 판정은 검사기관이 결과지에 표기한 참고범위를 우선합니다.",
        }


def _range(low: float | None, high: float | None) -> str:
    if low is not None and high is not None:
        return f"{low}-{high}"
    if high is not None:
        return f"< {high}"
    return f">= {low}"
