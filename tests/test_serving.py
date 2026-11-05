import json

import httpx
import pytest

from care_agent.llm.base import LLMResponse
from care_agent.llm.heuristic import HeuristicLLM
from care_agent.serving.bench import markdown_table, run_bench
from care_agent.serving.gateway import create_gateway
from care_agent.serving.router import (
    CircuitBreaker,
    NoBackendAvailable,
    RouterConfig,
    RoutingLLM,
    classify,
)

USER = [{"role": "user", "content": "LDL이 뭐예요?"}]


class Fake:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, 0

    async def complete(self, messages, tools):
        self.calls += 1
        if self.fail:
            raise ConnectionError(f"{self.name} down")
        return LLMResponse(
            f"from {self.name}", usage={"prompt_tokens": 1000, "completion_tokens": 500}
        )


def config(**overrides):
    backends = [
        {
            "name": "onprem",
            "base_url": "http://a",
            "model": "m",
            "location": "onprem",
            "phi_allowed": True,
            "tier": "large",
            "priority": 10,
        },
        {
            "name": "gke",
            "base_url": "http://b",
            "model": "m",
            "location": "private_cloud",
            "phi_allowed": True,
            "tier": "small",
            "priority": 10,
            "cost_per_1k_input": 0.001,
        },
        {
            "name": "public",
            "base_url": "http://c",
            "model": "m",
            "location": "public_cloud",
            "phi_allowed": False,
            "tier": "small",
            "priority": 5,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
        },
    ]
    return RouterConfig.model_validate(
        {"backends": backends, "policy": {"failure_threshold": 2, "cooldown_s": 30}, **overrides}
    )


def router(fail=()):
    fakes = {}

    def factory(b):
        fakes[b.name] = Fake(b.name, fail=b.name in fail)
        return fakes[b.name]

    return RoutingLLM.from_config(config(), llm_factory=factory), fakes


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (USER, "public"),
        ([{"role": "system", "content": "patient_id=demo-001"}, *USER], "phi"),
        ([*USER, {"role": "tool", "tool_call_id": "1", "content": "{}"}], "phi"),
        ([{"role": "user", "content": "제 번호는 010-1234-5678 이에요"}], "phi"),
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}],
            "phi",
        ),
    ],
)
def test_classify(messages, expected):
    assert classify(messages) == expected


async def test_public_traffic_goes_to_highest_priority():
    r, _ = router()
    resp = await r.complete(USER, [])
    assert resp.meta["routing"]["backend"] == "public" and resp.model == "public/m"


async def test_phi_never_reaches_public_even_when_everything_else_is_down():
    r, fakes = router(fail={"onprem", "gke"})
    phi = [{"role": "system", "content": "patient_id=demo-001"}, *USER]
    with pytest.raises(NoBackendAvailable):
        await r.complete(phi, [])
    assert fakes["public"].calls == 0


async def test_explicit_label_can_only_tighten():
    r, _ = router()
    resp = await r.complete(USER, [], classification="phi")
    assert resp.meta["routing"]["backend"] != "public"
    phi = [{"role": "tool", "tool_call_id": "1", "content": "{}"}]
    resp = await r.complete(phi, [], classification="public")
    assert resp.meta["routing"]["classification"] == "phi"


async def test_tier_hint_and_failover_with_attempts():
    r, _ = router(fail={"gke"})
    small = r.with_hint("small")
    phi = [{"role": "system", "content": "patient_id=x"}]
    with pytest.raises(NoBackendAvailable):  # only gke is phi+small; tier pool is respected
        await small.complete(phi, [])
    resp = await r.with_hint("large").complete(phi, [])
    assert resp.meta["routing"]["backend"] == "onprem"


async def test_failover_records_attempts_and_cost():
    r, fakes = router(fail={"public"})
    resp = await r.complete(USER, [])
    routing = resp.meta["routing"]
    assert routing["backend"] in {"onprem", "gke"}
    assert (
        routing["attempts"][0]["backend"] == "public" and "down" in routing["attempts"][0]["error"]
    )
    status = {s["name"]: s for s in r.status()}
    assert status["public"]["failures"] == 1


def test_circuit_breaker_states():
    now = [0.0]
    b = CircuitBreaker(threshold=2, cooldown_s=30, clock=lambda: now[0])
    b.failure()
    assert b.state == "closed"
    b.failure()
    assert b.state == "open" and not b.allow()
    now[0] = 31
    assert b.state == "half_open" and b.allow()
    b.failure()  # failed probe → open again
    assert b.state == "open"
    now[0] = 62
    b.success()
    assert b.state == "closed"


async def test_open_breaker_is_skipped():
    r, fakes = router(fail={"public"})
    for _ in range(2):
        await r.complete(USER, [])
    calls = fakes["public"].calls
    resp = await r.complete(USER, [])
    assert fakes["public"].calls == calls  # skipped, not retried
    assert resp.meta["routing"]["attempts"][0]["skipped"] == "circuit_open"


async def test_platform_routes_by_agent_data(make_platform):
    """care_chat (patient data) stays on-prem; the patient-free sub-agent may use the public API."""
    r = RoutingLLM.from_config(config(), llm_factory=lambda b: HeuristicLLM("careful"))
    platform = make_platform(r)
    result = await platform.agent("care_chat").run(
        "당화혈색소가 뭔가요? 의미를 설명해 주세요", patient_id="demo-001"
    )
    parent = [e.data["routing"]["backend"] for e in result.trace if e.type == "llm"]
    sub = next(e for e in result.trace if e.type == "subagent")
    child = [e["data"]["routing"]["backend"] for e in sub.data["trace"] if e["type"] == "llm"]
    assert set(parent) == {"onprem"}
    assert child[0] == "public"  # first call carries no patient data
    assert "public" not in child[1:]  # after a tool result the conversation counts as PHI


async def test_gateway():
    r, _ = router(fail={"onprem"})
    app = create_gateway(r)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        ok = await c.post("/v1/chat/completions", json={"model": "small", "messages": USER})
        assert ok.status_code == 200 and ok.headers["x-routed-backend"] == "public"
        assert ok.json()["choices"][0]["message"]["content"] == "from public"

        forced = await c.post(
            "/v1/chat/completions",
            json={"messages": USER},
            headers={"x-data-classification": "phi"},
        )
        assert forced.headers["x-routed-backend"] == "gke"

        down = await c.post(
            "/v1/chat/completions",
            json={"model": "large", "messages": USER},
            headers={"x-data-classification": "phi"},
        )
        assert down.status_code == 503

        metrics = (await c.get("/metrics")).text
        assert 'gateway_requests_total{backend="public",outcome="ok"} 1' in metrics
        assert (await c.get("/v1/backends")).json()[0]["name"] == "onprem"


async def test_bench_against_mock_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        chunks = [{"choices": [{"delta": {"content": t}}]} for t in ["안", "녕", "하", "세요"]]
        chunks.append({"choices": [], "usage": {"completion_tokens": 4}})
        sse = (
            "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks)
            + "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    result = await run_bench(
        "http://vllm/v1",
        "m",
        concurrency_levels=[1, 4],
        requests=8,
        transport=httpx.MockTransport(handler),
    )
    lv = result["levels"][1]
    assert lv["concurrency"] == 4 and lv["requests"] == 8 and lv["success_rate"] == 1.0
    assert lv["ttft_ms_p50"] is not None and lv["output_tok_per_s"] > 0
    assert "| 4 | 1.0 |" in markdown_table(result)


async def test_bench_records_http_errors():
    result = await run_bench(
        "http://vllm/v1",
        "m",
        concurrency_levels=[2],
        requests=2,
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="busy")),
    )
    lv = result["levels"][0]
    assert lv["success_rate"] == 0.0 and lv["errors"][0].startswith("503")
