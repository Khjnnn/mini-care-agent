"""Load benchmark for OpenAI-compatible endpoints (vLLM, the gateway, cloud APIs).

Streams every request (``stream: true``) to measure what users feel:

* **TTFT** – time to first token
* **ITL / TPOT** – mean inter-token latency (time per output token after the first)
* **E2E** – request latency
* **throughput** – requests/s and output tokens/s at each concurrency level

    care-agent bench --base-url http://dgx:8000/v1 --model Qwen/Qwen2.5-72B-Instruct \
        --concurrency 1 4 16 64 --requests 64 --out bench/dgx-72b.json

Prompts default to short Korean health-counselling questions (output ~150-300 tokens),
which is the traffic shape of the chat use case; pass ``--prompts file.jsonl`` for others.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_PROMPTS = [
    "당화혈색소 7.8%는 어떤 의미인가요? 생활습관에서 주의할 점을 알려주세요.",
    "공복혈당이 130 정도 나오는데 식사는 어떻게 조절하는 게 좋을까요?",
    "LDL 콜레스테롤이 160이면 높은 편인가요? 운동은 얼마나 해야 하나요?",
    "고혈압 약을 먹고 있는데 아침에 혈압을 재면 140/90 정도 나옵니다. 괜찮은가요?",
    "eGFR 수치가 58이라고 하는데 콩팥 건강을 위해 무엇을 해야 하나요?",
    "건강검진 결과에서 간수치(ALT)가 52로 나왔어요. 원인이 뭘까요?",
]


@dataclass
class RequestStat:
    ok: bool
    ttft_ms: float | None
    e2e_ms: float
    output_tokens: int
    itl_ms: float | None
    error: str | None = None


async def one_request(
    client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int
) -> RequestStat:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first: float | None = None
    stamps: list[float] = []
    usage_tokens: int | None = None
    try:
        async with client.stream("POST", "chat/completions", json=body) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors="replace")[:200]
                return RequestStat(
                    False,
                    None,
                    (time.perf_counter() - t0) * 1000,
                    0,
                    None,
                    f"{resp.status_code} {text}",
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens")
                for choice in chunk.get("choices") or []:
                    if (choice.get("delta") or {}).get("content"):
                        now = time.perf_counter()
                        first = first or now
                        stamps.append(now)
    except httpx.HTTPError as e:
        return RequestStat(
            False, None, (time.perf_counter() - t0) * 1000, 0, None, f"{type(e).__name__}: {e}"
        )
    end = time.perf_counter()
    tokens = usage_tokens if usage_tokens is not None else len(stamps)
    itl = (
        (stamps[-1] - stamps[0]) * 1000 / (tokens - 1)
        if first and tokens > 1 and len(stamps) > 1
        else None
    )
    return RequestStat(
        ok=first is not None,
        ttft_ms=(first - t0) * 1000 if first else None,
        e2e_ms=(end - t0) * 1000,
        output_tokens=tokens,
        itl_ms=itl,
        error=None if first else "no tokens received",
    )


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


def summarize_level(concurrency: int, stats: list[RequestStat], wall_s: float) -> dict[str, Any]:
    ok = [s for s in stats if s.ok]
    ttft = [s.ttft_ms for s in ok if s.ttft_ms is not None]
    itl = [s.itl_ms for s in ok if s.itl_ms is not None]
    e2e = [s.e2e_ms for s in ok]
    out_tokens = sum(s.output_tokens for s in ok)
    return {
        "concurrency": concurrency,
        "requests": len(stats),
        "success_rate": round(len(ok) / len(stats), 3) if stats else None,
        "ttft_ms_p50": _p(ttft, 0.5),
        "ttft_ms_p95": _p(ttft, 0.95),
        "itl_ms_mean": round(statistics.mean(itl), 2) if itl else None,
        "e2e_ms_p50": _p(e2e, 0.5),
        "e2e_ms_p95": _p(e2e, 0.95),
        "req_per_s": round(len(ok) / wall_s, 2) if wall_s else None,
        "output_tok_per_s": round(out_tokens / wall_s, 1) if wall_s else None,
        "errors": [s.error for s in stats if not s.ok][:3],
    }


async def run_level(
    client: httpx.AsyncClient,
    model: str,
    prompts: list[str],
    *,
    concurrency: int,
    requests: int,
    max_tokens: int,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)

    async def task(i: int) -> RequestStat:
        async with sem:
            return await one_request(client, model, prompts[i % len(prompts)], max_tokens)

    t0 = time.perf_counter()
    stats = await asyncio.gather(*(task(i) for i in range(requests)))
    return summarize_level(concurrency, list(stats), time.perf_counter() - t0)


async def run_bench(
    base_url: str,
    model: str,
    *,
    concurrency_levels: list[int],
    requests: int,
    max_tokens: int = 256,
    prompts: list[str] | None = None,
    api_key: str = "EMPTY",
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    prompts = prompts or DEFAULT_PROMPTS
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(300, connect=10),
        limits=httpx.Limits(max_connections=max(concurrency_levels) + 4),
        transport=transport,
    ) as client:
        levels = [
            await run_level(
                client,
                model,
                prompts,
                concurrency=c,
                requests=max(requests, c),
                max_tokens=max_tokens,
            )
            for c in concurrency_levels
        ]
    return {"base_url": base_url, "model": model, "max_tokens": max_tokens, "levels": levels}


def markdown_table(result: dict[str, Any]) -> str:
    head = "| conc. | ok | TTFT p50/p95 ms | ITL ms | E2E p50/p95 ms | req/s | out tok/s |\n|---|---|---|---|---|---|---|"
    rows = [
        f"| {lv['concurrency']} | {lv['success_rate']} | {lv['ttft_ms_p50']} / {lv['ttft_ms_p95']} | "
        f"{lv['itl_ms_mean']} | {lv['e2e_ms_p50']} / {lv['e2e_ms_p95']} | {lv['req_per_s']} | {lv['output_tok_per_s']} |"
        for lv in result["levels"]
    ]
    return (
        f"**{result['model']}** @ `{result['base_url']}` (max_tokens={result['max_tokens']})\n\n"
        + "\n".join([head, *rows])
        + "\n"
    )
