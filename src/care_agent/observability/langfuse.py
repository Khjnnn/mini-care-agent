"""Export agent traces (and eval scores) to Langfuse via its public ingestion API.

Uses the HTTP API directly (``POST /api/public/ingestion``) instead of the SDK so the
mapping is explicit and version-stable:

    agent run       → trace
    llm step        → generation (model, token usage, latency)
    tool call       → span (ERROR level when denied/failed)
    gates/subagent  → event
    judge scores    → score

Patient data: tool results are de-identified summaries, but they are still health data.
Point this at a self-hosted Langfuse inside the same trust boundary, or set
``redact_tool_results=True`` to export only tool names, arguments and status.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from care_agent.agent.loop import AgentResult
from care_agent.agent.trace import TraceEvent

log = logging.getLogger(__name__)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def _envelope(kind: str, body: dict[str, Any], ts: float) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "timestamp": _iso(ts), "type": kind, "body": body}


class LangfuseExporter:
    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        *,
        redact_tool_results: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.redact = redact_tool_results
        self._http = httpx.AsyncClient(
            base_url=host.rstrip("/"),
            auth=(public_key, secret_key),
            timeout=10,
            transport=transport,
        )

    def build_batch(
        self,
        result: AgentResult,
        *,
        trace_id: str,
        name: str,
        question: str,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        events = result.trace
        start = min((e.ts - (e.duration_ms or 0) / 1000 for e in events), default=0.0)
        batch = [
            _envelope(
                "trace-create",
                {
                    "id": trace_id,
                    "name": name,
                    "sessionId": session_id,
                    "input": question,
                    "output": result.answer,
                    "tags": [*(tags or []), f"stop:{result.stop_reason}"],
                    "metadata": {**(metadata or {}), "steps": result.steps},
                },
                start,
            )
        ]
        for e in events:
            batch.append(self._observation(e, trace_id))
        return batch

    def _observation(self, e: TraceEvent, trace_id: str) -> dict[str, Any]:
        end = e.ts
        begin = end - (e.duration_ms or 0) / 1000
        common = {
            "id": str(uuid.uuid4()),
            "traceId": trace_id,
            "startTime": _iso(begin),
            "metadata": {"step": e.step},
        }
        if e.type == "llm":
            usage = e.data.get("usage") or {}
            return _envelope(
                "generation-create",
                {
                    **common,
                    "name": f"llm-step-{e.step}",
                    "endTime": _iso(end),
                    "model": e.data.get("model"),
                    "output": {
                        "content": e.data.get("content"),
                        "tool_calls": e.data.get("tool_calls"),
                    },
                    "usage": {
                        "input": usage.get("prompt_tokens"),
                        "output": usage.get("completion_tokens"),
                        "unit": "TOKENS",
                    },
                },
                begin,
            )
        if e.type in ("tool_call", "tool_denied", "tool_error"):
            output: Any = (
                e.data.get("result")
                if e.type == "tool_call"
                else e.data.get("reason") or e.data.get("error")
            )
            if self.redact and e.type == "tool_call":
                output = "[redacted]"
            return _envelope(
                "span-create",
                {
                    **common,
                    "name": f"tool:{e.data.get('name')}",
                    "endTime": _iso(end),
                    "input": e.data.get("arguments"),
                    "output": output,
                    "level": "DEFAULT"
                    if e.type == "tool_call"
                    else ("WARNING" if e.type == "tool_denied" else "ERROR"),
                    "statusMessage": None if e.type == "tool_call" else e.type,
                },
                begin,
            )
        data = e.data
        if e.type == "subagent":
            data = {k: v for k, v in data.items() if k != "trace"}
        return _envelope(
            "event-create",
            {
                **common,
                "name": e.type,
                "output": data,
                "level": "WARNING" if e.type == "output_gate" else "DEFAULT",
            },
            begin,
        )

    @staticmethod
    def score_events(
        trace_id: str, scores: dict[str, float], comment: str | None = None
    ) -> list[dict[str, Any]]:
        now = datetime.now(UTC).timestamp()
        return [
            _envelope(
                "score-create",
                {
                    "id": str(uuid.uuid4()),
                    "traceId": trace_id,
                    "name": k,
                    "value": v,
                    "comment": comment,
                },
                now,
            )
            for k, v in scores.items()
        ]

    async def send(self, batch: list[dict[str, Any]]) -> bool:
        """Best-effort: observability must never break the request path."""
        try:
            resp = await self._http.post("/api/public/ingestion", json={"batch": batch})
            if resp.status_code >= 400:
                log.warning("langfuse ingestion failed: %s %s", resp.status_code, resp.text[:200])
                return False
            return True
        except httpx.HTTPError as e:
            log.warning("langfuse ingestion error: %s", e)
            return False

    async def export(self, result: AgentResult, **kwargs: Any) -> str:
        trace_id = kwargs.pop("trace_id", None) or str(uuid.uuid4())
        await self.send(self.build_batch(result, trace_id=trace_id, **kwargs))
        return trace_id

    async def aclose(self) -> None:
        await self._http.aclose()
