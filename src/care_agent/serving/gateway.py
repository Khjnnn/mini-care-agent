"""OpenAI-compatible gateway in front of the router.

Other services keep using any OpenAI client and point ``base_url`` here; routing,
PHI policy, failover and accounting happen centrally.

* ``POST /v1/chat/completions`` – non-streaming chat completions (tools supported).
  Header ``x-data-classification: phi`` forces PHI routing (it can only tighten the
  policy). The chosen backend is returned in ``x-routed-backend``.
* ``GET /v1/backends`` – breaker state, latency EWMA, request/failure counts, cost.
* ``GET /metrics`` – Prometheus text format.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict

from care_agent.llm.base import LLMResponse
from care_agent.serving.router import NoBackendAvailable, RoutingLLM


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # temperature etc. are accepted and ignored

    model: str | None = None  # treated as a tier hint: "small" / "large" / anything else = any
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    stream: bool = False


def to_openai(resp: LLMResponse, request_id: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": resp.content}
    if resp.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in resp.tool_calls
        ]
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if resp.tool_calls else "stop",
            }
        ],
        "usage": resp.usage,
    }


def create_gateway(router: RoutingLLM) -> FastAPI:
    app = FastAPI(title="mini-care-agent LLM gateway", version="0.5.0")
    outcomes: Counter[tuple[str, str]] = Counter()
    latency_sum: Counter[str] = Counter()

    @app.post("/v1/chat/completions")
    async def chat(
        req: ChatRequest,
        response: Response,
        x_data_classification: Literal["phi", "public"] | None = Header(default=None),
    ) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(400, "streaming is not supported by the gateway yet")
        tiers = {b.config.tier for b in router.backends}
        view = router.with_hint(req.model if req.model in tiers else None)
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()
        try:
            resp = await view.complete(
                req.messages, req.tools or [], classification=x_data_classification
            )
        except NoBackendAvailable as e:
            outcomes[("none", "unavailable")] += 1
            raise HTTPException(503, str(e)) from e
        decision = resp.meta["routing"]
        outcomes[(decision["backend"], "ok")] += 1
        latency_sum[decision["backend"]] += time.perf_counter() - t0
        response.headers["x-routed-backend"] = decision["backend"]
        response.headers["x-data-classification"] = decision["classification"]
        return to_openai(resp, rid)

    @app.get("/v1/backends")
    async def backends() -> list[dict[str, Any]]:
        return router.status()

    @app.get("/metrics")
    async def metrics() -> Response:
        lines = [
            "# HELP gateway_requests_total Chat completion requests by backend and outcome.",
            "# TYPE gateway_requests_total counter",
        ]
        lines += [
            f'gateway_requests_total{{backend="{b}",outcome="{o}"}} {n}'
            for (b, o), n in sorted(outcomes.items())
        ]
        lines += ["# TYPE gateway_latency_seconds_sum counter"]
        lines += [
            f'gateway_latency_seconds_sum{{backend="{b}"}} {v:.4f}'
            for b, v in sorted(latency_sum.items())
        ]
        lines += ["# TYPE backend_circuit_open gauge"]
        lines += [
            f'backend_circuit_open{{backend="{s["name"]}"}} {int(s["breaker"] == "open")}'
            for s in router.status()
        ]
        lines += ["# TYPE backend_cost_usd_total counter"]
        lines += [
            f'backend_cost_usd_total{{backend="{s["name"]}"}} {s["cost_usd"]}'
            for s in router.status()
        ]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app
