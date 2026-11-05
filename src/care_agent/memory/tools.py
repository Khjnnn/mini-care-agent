"""Memory exposed to the model as tools (patient-scoped like every EHR tool)."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from care_agent.memory.store import MemoryStore
from care_agent.tools.registry import Permission, ToolRegistry

PatientId = Annotated[str, Field(description="FHIR Patient resource id", min_length=1)]


def build_memory_tools(store: MemoryStore, registry: ToolRegistry | None = None) -> ToolRegistry:
    registry = registry if registry is not None else ToolRegistry()

    @registry.register
    async def search_patient_memory(
        patient_id: PatientId,
        query: Annotated[
            str | None, Field(description="검색어(공백으로 구분). 생략 시 최근 메모")
        ] = None,
    ) -> list[dict[str, Any]]:
        """이전 상담에서 저장해 둔 환자 메모(선호, 목표, 생활 맥락 등)를 검색한다. EHR 기록이 아니다."""
        return [
            {"note": n.note, "saved_at": int(n.ts)} for n in store.search_notes(patient_id, query)
        ]

    @registry.register(permission=Permission.MEMORY)
    async def save_patient_memory(
        patient_id: PatientId,
        note: Annotated[
            str, Field(min_length=2, max_length=300, description="기억할 내용 한 문장")
        ],
    ) -> dict[str, Any]:
        """다음 상담에도 참고할 환자의 선호·목표·생활 맥락을 한 문장으로 저장한다. 의학적 판단이나 검사 수치는 저장하지 않는다."""
        return {"saved": store.add_note(patient_id, note, source="agent")}

    return registry
