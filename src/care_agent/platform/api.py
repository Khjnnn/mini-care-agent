"""HTTP API for running agents and discovering tools/agents.

Approvals: WRITE tools run only if the caller lists them in ``approved_tools`` for that
request (pre-approval by a clinician UI). Anything else is denied and reported in
``denied_tools`` so the UI can ask the human and retry.

Authentication / patient-level authorization is out of scope for the demo: in a real
deployment the caller's token must be checked against ``patient_id`` before ``run``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from care_agent.memory.store import SessionPatientMismatch
from care_agent.platform.core import Platform
from care_agent.tools.registry import Tool


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    patient_id: str = Field(min_length=1)
    session_id: str | None = Field(default=None, max_length=128)
    approved_tools: list[str] = Field(default_factory=list)
    include_trace: bool = False


class RunResponse(BaseModel):
    agent: str
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
    async def run_agent(name: str, req: RunRequest) -> RunResponse:
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
            answer=result.answer,
            stop_reason=result.stop_reason,
            steps=result.steps,
            tool_calls=result.tool_calls,
            denied_tools=[{"name": d["name"], "reason": d["reason"]} for d in denied],
            flags=flags,
            trace=[e.to_dict() for e in result.trace] if req.include_trace else None,
        )

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
