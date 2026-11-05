"""Read-only EHR tools backed by a FHIR server.

Every tool takes ``patient_id`` so the safety layer can pin a session to one patient,
and every tool returns *summaries* (see ``fhir.summaries``), never raw resources.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import Field

from care_agent.fhir.client import FHIRClient
from care_agent.fhir.summaries import (
    summarize_condition,
    summarize_medication_request,
    summarize_observation,
    summarize_patient,
)
from care_agent.knowledge.lab_catalog import lookup, normalize_name
from care_agent.tools.registry import ToolRegistry

PatientId = Annotated[str, Field(description="FHIR Patient resource id", min_length=1)]

# Vital signs are not in the lab catalogue; common names → LOINC.
_VITALS = {
    "혈압": "85354-9",
    "bloodpressure": "85354-9",
    "bp": "85354-9",
    "체중": "29463-7",
    "몸무게": "29463-7",
    "bodyweight": "29463-7",
    "weight": "29463-7",
    "bmi": "39156-5",
    "체질량지수": "39156-5",
}
TEST_HINT = "검사 이름(예: 당화혈색소, LDL, 공복혈당, 혈압, 체중) 또는 LOINC 코드를 쓰거나, test 없이 전체를 조회하세요."


def resolve_test_code(test: str) -> str | None:
    """Free-text test name or LOINC code → LOINC code (the model need not know codes)."""
    t = test.strip()
    if re.fullmatch(r"\d{1,6}-\d", t):
        return t
    key = normalize_name(t)
    if key in _VITALS:
        return _VITALS[key]
    info = lookup(t)
    return info.loinc if info else None


def build_fhir_tools(client: FHIRClient, registry: ToolRegistry | None = None) -> ToolRegistry:
    registry = registry if registry is not None else ToolRegistry()

    @registry.register
    async def get_patient_summary(patient_id: PatientId) -> dict[str, Any]:
        """환자의 기본 정보(성별, 나이, 출생연도)를 조회한다. 이름·주소·연락처 등 식별정보는 반환하지 않는다."""
        return summarize_patient(await client.read("Patient", patient_id))

    @registry.register
    async def list_conditions(
        patient_id: PatientId,
        active_only: Annotated[bool, Field(description="true면 현재 활성 상태인 진단만")] = True,
    ) -> list[dict[str, Any]]:
        """환자의 진단(질환) 목록을 조회한다. 진단명, SNOMED 코드, 상태, 발병일을 반환한다."""
        params: dict[str, str | int] = {"patient": patient_id, "_count": 100}
        if active_only:
            params["clinical-status"] = "active"
        return [summarize_condition(c) for c in await client.search("Condition", params)]

    @registry.register
    async def get_recent_observations(
        patient_id: PatientId,
        category: Annotated[
            Literal["laboratory", "vital-signs"] | None,
            Field(description="laboratory=검사결과, vital-signs=활력징후. 생략 시 전체"),
        ] = None,
        test: Annotated[
            str | None,
            Field(
                description="특정 항목만 조회할 때: 검사 이름(한글/영문) 또는 LOINC 코드. "
                "예: '당화혈색소', 'LDL', '혈압', '4548-4'. 생략 시 전체"
            ),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="최대 개수 (최신순)")] = 10,
    ) -> Any:
        """환자의 검사결과·활력징후를 최신순으로 조회한다. 수치, 단위, 참고범위, 판정(High/Low)을 반환한다."""
        params: dict[str, str | int] = {"patient": patient_id, "_sort": "-date", "_count": limit}
        if category:
            params["category"] = category
        if test:
            code = resolve_test_code(test)
            if code is None:
                return {"error": f"알 수 없는 검사 이름: {test!r}", "hint": TEST_HINT}
            params["code"] = code
        results = [summarize_observation(o) for o in await client.search("Observation", params)]
        if results or not test:
            return results
        # Empty after filtering: tell the model what *is* on record, instead of a bare []
        # that small models read as "no record at all" (seen with qwen2.5:7b on LDL).
        recent = await client.search(
            "Observation", {"patient": patient_id, "_sort": "-date", "_count": 50}
        )
        on_record = sorted({summarize_observation(o)["name"] for o in recent} - {None})
        return {"results": [], "note": f"'{test}' 기록이 없습니다.", "tests_on_record": on_record}

    @registry.register
    async def list_medications(
        patient_id: PatientId,
        active_only: Annotated[bool, Field(description="true면 현재 복용 중인 약만")] = True,
    ) -> list[dict[str, Any]]:
        """환자의 처방 약 목록을 조회한다. 약 이름, 상태, 처방일, 용법, 처방 사유를 반환한다."""
        params: dict[str, str | int] = {"patient": patient_id, "_count": 100}
        if active_only:
            params["status"] = "active"
        return [
            summarize_medication_request(m)
            for m in await client.search("MedicationRequest", params)
        ]

    return registry
