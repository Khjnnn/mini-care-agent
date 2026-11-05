"""HTTP API for running agents and discovering tools/agents.

Approvals: WRITE tools run only if the caller lists them in ``approved_tools`` for that
request (pre-approval by a clinician UI). Anything else is denied and reported in
``denied_tools`` so the UI can ask the human and retry.

Authentication / patient-level authorization is out of scope for the demo: in a real
deployment the caller's token must be checked against ``patient_id`` before ``run``.
"""

from __future__ import annotations

import base64
import binascii
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from care_agent.documents.pipeline import ReviewRequired, outcome_to_dict
from care_agent.documents.schema import StructuredLabReport
from care_agent.fhir.client import FHIRError
from care_agent.memory.store import SessionPatientMismatch
from care_agent.platform.core import Platform
from care_agent.tools.registry import Tool


class ExtractRequest(BaseModel):
    text: str | None = Field(default=None, max_length=50_000)
    image_base64: str | None = None
    mime: str = "image/png"


class CommitRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    report: StructuredLabReport
    reviewed_by: str = Field(min_length=1, description="검토한 사람 (감사 로그용)")
    acknowledge_issues: bool = False


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    patient_id: str = Field(min_length=1)
    session_id: str | None = Field(default=None, max_length=128)
    approved_tools: list[str] = Field(default_factory=list)
    include_trace: bool = False


class RunResponse(BaseModel):
    agent: str
    trace_id: str
    answer: str
    stop_reason: str
    steps: int
    tool_calls: list[str]
    denied_tools: list[dict[str, Any]]
    flags: list[str]
    trace: list[dict[str, Any]] | None = None


def create_app(platform: Platform) -> FastAPI:
    app = FastAPI(title="mini-care-agent", version="0.2.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/tools")
    async def list_tools() -> list[dict[str, Any]]:
        return [_tool_info(t) for t in platform.tools]

    @app.get("/v1/agents")
    async def list_agents() -> list[dict[str, Any]]:
        return [
            s.model_dump(
                include={"name", "description", "tools", "subagents", "max_steps", "model"}
            )
            for s in platform.specs.values()
        ]

    @app.post(
        "/v1/agents/{name}/runs", response_model=RunResponse, response_model_exclude_none=True
    )
    async def run_agent(name: str, req: RunRequest, background: BackgroundTasks) -> RunResponse:
        if name not in platform.specs:
            raise HTTPException(404, f"unknown agent {name!r}")
        approved = set(req.approved_tools)

        async def approver(tool: Tool, _args: dict[str, Any]) -> bool:
            return tool.name in approved

        agent = platform.agent(name)
        try:
            result = await agent.run(
                req.question,
                patient_id=req.patient_id,
                approver=approver,
                session_id=req.session_id,
            )
        except SessionPatientMismatch as e:
            raise HTTPException(409, str(e)) from e

        trace_id = str(uuid.uuid4())
        if platform.exporter is not None:  # after the response is sent; never blocks the user
            background.add_task(
                platform.exporter.export,
                result,
                trace_id=trace_id,
                name=f"agent:{name}",
                question=req.question,
                session_id=req.session_id,
                metadata={"patient_id": req.patient_id},
            )

        denied = [e.data for e in result.trace if e.type == "tool_denied"]
        flags = []
        if result.stop_reason == "emergency":
            flags.append("emergency")
        if any(e.type == "output_gate" for e in result.trace):
            flags.append("output_modified")
        if denied:
            flags.append("tool_denied")
        return RunResponse(
            agent=name,
            trace_id=trace_id,
            answer=result.answer,
            stop_reason=result.stop_reason,
            steps=result.steps,
            tool_calls=result.tool_calls,
            denied_tools=[{"name": d["name"], "reason": d["reason"]} for d in denied],
            flags=flags,
            trace=[e.to_dict() for e in result.trace] if req.include_trace else None,
        )

    @app.post("/v1/documents/lab-report/extract")
    async def extract_lab_report(req: ExtractRequest) -> dict[str, Any]:
        """Image or OCR text → structured report for human review. Writes nothing."""
        if (req.text is None) == (req.image_base64 is None):
            raise HTTPException(422, "pass exactly one of text or image_base64")
        image = None
        if req.image_base64 is not None:
            try:
                image = base64.b64decode(req.image_base64, validate=True)
            except binascii.Error as e:
                raise HTTPException(422, "image_base64 is not valid base64") from e
        outcome = await platform.lab_reports.extract(text=req.text, image=image, mime=req.mime)
        return outcome_to_dict(outcome)

    @app.post("/v1/documents/lab-report/commit")
    async def commit_lab_report(req: CommitRequest) -> dict[str, Any]:
        """Human-reviewed report → FHIR Observations (one transaction)."""
        try:
            out = await platform.lab_reports.commit(
                req.report,
                req.patient_id,
                reviewed_by=req.reviewed_by,
                acknowledge_issues=req.acknowledge_issues,
            )
        except ReviewRequired as e:
            raise HTTPException(
                422, {"message": str(e), "issues": [i.model_dump() for i in e.issues]}
            ) from e
        except FHIRError as e:
            raise HTTPException(502 if e.status_code >= 500 else e.status_code, str(e)) from e
        return {
            "created": out.created,
            "acknowledged_issues": [i.model_dump() for i in out.acknowledged_issues],
            "reviewed_by": req.reviewed_by,
        }

    @app.get("/v1/sessions/{session_id}/turns")
    async def session_turns(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return [t.__dict__ for t in platform.memory.recent_turns(session_id, limit)]

    return app


def _tool_info(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "permission": tool.permission.value,
        "patient_scoped": tool.patient_scoped,
        "input_schema": tool.json_schema(),
    }
