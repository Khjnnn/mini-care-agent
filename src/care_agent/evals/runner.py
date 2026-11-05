"""Bulk evaluation: a bounded worker queue over eval cases.

* ``concurrency`` workers pull from an ``asyncio.Queue`` (the LLM backend, not the
  harness, is the bottleneck, so this is the knob that matters for throughput).
* Per-case timeout and retry on infrastructure errors; a case that still fails is
  recorded as ``error`` instead of aborting the run.
* Output is a self-describing run file (JSON) that ``compare`` / ``drift`` consume.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from care_agent.agent.loop import Agent, AgentResult
from care_agent.evals.analysis import failure_tags
from care_agent.evals.cases import EvalCase, run_checks
from care_agent.evals.judge import LLMJudge

AgentFactory = Callable[[EvalCase], Agent]
#: Called after every case with (case, result, trace_id); e.g. Langfuse export.
OnResult = Callable[[EvalCase, AgentResult], Awaitable[None]]


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    answer: str
    stop_reason: str
    tool_calls: list[str]
    checks: list[dict[str, Any]]
    passed: bool
    latency_ms: float
    tokens: int
    judge: dict[str, int] | None = None
    judge_rationale: str | None = None
    judge_error: str | None = None
    failure_tags: list[str] = field(default_factory=list)
    error: str | None = None
    #: Deterministic checks only (comparable across runs with and without a judge).
    checks_passed: bool | None = None
    #: Every tool call with its arguments and outcome — needed to debug wrong answers
    #: (e.g. which `test` the model asked for when it said "no LDL on record").
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    nudges: list[str] = field(default_factory=list)


def _tool_trace(result: AgentResult) -> list[dict[str, Any]]:
    out = []
    for e in result.trace:
        if e.type not in ("tool_call", "tool_denied", "tool_error"):
            continue
        item: dict[str, Any] = {
            "step": e.step,
            "name": e.data.get("name"),
            "arguments": e.data.get("arguments"),
            "status": {"tool_call": "ok", "tool_denied": "denied", "tool_error": "error"}[e.type],
        }
        if e.type == "tool_call":
            res = e.data.get("result")
            item["result_items"] = len(res) if isinstance(res, list) else None
            if isinstance(res, dict) and "note" in res:
                item["note"] = res["note"]
        else:
            item["detail"] = str(e.data.get("reason") or e.data.get("error"))[:200]
        out.append(item)
    return out


@dataclass
class RunFile:
    meta: dict[str, Any]
    results: list[CaseResult]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {"meta": self.meta, "results": [asdict(r) for r in self.results]},
                ensure_ascii=False,
                indent=1,
            ),
            "utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> RunFile:
        data = json.loads(Path(path).read_text("utf-8"))
        return cls(data["meta"], [CaseResult(**r) for r in data["results"]])


class EvalRunner:
    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        judge: LLMJudge | None = None,
        concurrency: int = 4,
        timeout_s: float = 120.0,
        retries: int = 1,
        on_result: OnResult | None = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.judge = judge
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.retries = retries
        self.on_result = on_result

    async def run(
        self, cases: list[EvalCase], *, label: str = "run", meta: dict[str, Any] | None = None
    ) -> RunFile:
        queue: asyncio.Queue[EvalCase] = asyncio.Queue()
        for case in cases:
            queue.put_nowait(case)
        results: dict[str, CaseResult] = {}

        async def worker() -> None:
            while True:
                try:
                    case = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results[case.id] = await self._one(case)

        started = time.time()
        await asyncio.gather(*(worker() for _ in range(max(1, self.concurrency))))
        ordered = [results[c.id] for c in cases]
        return RunFile(
            meta={
                "label": label,
                "started_at": started,
                "duration_s": round(time.time() - started, 2),
                "cases": len(cases),
                "judge": self.judge is not None,
                **(meta or {}),
                "summary": summarize(ordered),
            },
            results=ordered,
        )

    async def _one(self, case: EvalCase) -> CaseResult:
        last_error = ""
        for _ in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    self.agent_factory(case).run(case.question, patient_id=case.patient_id),
                    self.timeout_s,
                )
            except Exception as e:  # infra failure: retry, then record
                last_error = f"{type(e).__name__}: {e}"
                continue
            latency = (time.perf_counter() - t0) * 1000
            return await self._score(case, result, latency)
        return CaseResult(
            case.id,
            case.category,
            case.question,
            "",
            "error",
            [],
            [],
            False,
            0.0,
            0,
            failure_tags=["infra_error"],
            error=last_error,
        )

    async def _score(self, case: EvalCase, result: AgentResult, latency: float) -> CaseResult:
        checks = run_checks(case, result)
        verdict = None
        judge_error = None
        if self.judge and result.stop_reason == "final":
            try:
                verdict = await self.judge.score(case.question, result, note=case.judge_note)
            except Exception as e:  # a broken judge must be visible, not silently "no score"
                judge_error = f"{type(e).__name__}: {e}"[:300]
        tokens = sum(
            (e.data.get("usage") or {}).get("total_tokens", 0)
            for e in result.trace
            if e.type == "llm"
        )
        scores = verdict.scores.model_dump() if verdict else None
        record = CaseResult(
            case_id=case.id,
            category=case.category,
            question=case.question,
            answer=result.answer,
            stop_reason=result.stop_reason,
            tool_calls=result.tool_calls,
            checks=[asdict(c) for c in checks],
            passed=all(c.passed for c in checks) and (scores is None or min(scores.values()) >= 3),
            latency_ms=round(latency, 1),
            tokens=tokens,
            judge=scores,
            judge_rationale=verdict.rationale if verdict else None,
            judge_error=judge_error,
            checks_passed=all(c.passed for c in checks),
            tool_trace=_tool_trace(result),
            nudges=[e.data["kind"] for e in result.trace if e.type == "nudge"],
        )
        record.failure_tags = failure_tags(record, result)
        if self.on_result:
            await self.on_result(case, result)
        return record


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 1)


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r.passed)
    judged = [r.judge for r in results if r.judge]
    latencies = [r.latency_ms for r in results if r.error is None]
    checked = [r.checks_passed for r in results if r.checks_passed is not None]
    return {
        "pass_rate": round(sum(r.passed for r in results) / len(results), 3) if results else None,
        "checks_pass_rate": round(sum(checked) / len(checked), 3) if checked else None,
        "nudges": dict(Counter(n for r in results for n in r.nudges)),
        "pass_rate_by_category": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())},
        "judge_mean": {
            k: round(statistics.mean(j[k] for j in judged), 2)
            for k in (judged[0] if judged else {})
        },
        "latency_p50_ms": _pct(latencies, 0.5),
        "latency_p95_ms": _pct(latencies, 0.95),
        "tool_usage": dict(Counter(t for r in results for t in r.tool_calls).most_common()),
        "stop_reasons": dict(Counter(r.stop_reason for r in results)),
        "failure_tags": dict(Counter(t for r in results for t in r.failure_tags).most_common()),
        "errors": sum(r.error is not None for r in results),
        "judge_failures": sum(r.judge_error is not None for r in results),
    }
