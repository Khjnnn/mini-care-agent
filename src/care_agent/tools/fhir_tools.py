"""Read-only EHR tools backed by a FHIR server.

Every tool takes ``patient_id`` so the safety layer can pin a session to one patient,
and every tool returns *summaries* (see ``fhir.summaries``), never raw resources.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from care_agent.fhir.client import FHIRClient
from care_agent.fhir.summaries import (
    summarize_condition,
    summarize_medication_request,
    summarize_observation,
    summarize_patient,
)
from care_agent.tools.registry import ToolRegistry

PatientId = Annotated[str, Field(description="FHIR Patient resource id", min_length=1)]


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
        loinc_code: Annotated[
            str | None,
            Field(description="특정 항목만 조회할 LOINC 코드 (예: 4548-4 = HbA1c)"),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="최대 개수 (최신순)")] = 10,
    ) -> list[dict[str, Any]]:
        """환자의 검사결과·활력징후를 최신순으로 조회한다. 수치, 단위, 참고범위, 판정(High/Low)을 반환한다."""
        params: dict[str, str | int] = {"patient": patient_id, "_sort": "-date", "_count": limit}
        if category:
            params["category"] = category
        if loinc_code:
            params["code"] = loinc_code
        return [summarize_observation(o) for o in await client.search("Observation", params)]

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
