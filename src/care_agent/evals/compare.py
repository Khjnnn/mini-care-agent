"""A/B comparison of two runs and drift detection against a baseline.

``compare_runs`` – same case set, two configurations (prompt, model, router policy):
per-metric deltas, per-case regressions/fixes, and judge-based win rates if pairwise
verdicts are supplied.

``detect_drift`` – baseline vs current window (two eval runs, or production traces
converted to run files). Scalar metrics are checked against thresholds; categorical
distributions (tool usage, stop reasons, failure tags) with the Population Stability
Index (PSI > 0.2 is the usual "significant shift" rule of thumb).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from care_agent.evals.runner import CaseResult, RunFile

PSI_ALERT = 0.2
EPS = 1e-4


def psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    keys = set(expected) | set(actual)
    e_total = sum(expected.values()) or 1
    a_total = sum(actual.values()) or 1
    value = 0.0
    for k in keys:
        e = max(expected.get(k, 0) / e_total, EPS)
        a = max(actual.get(k, 0) / a_total, EPS)
        value += (a - e) * math.log(a / e)
    return round(value, 4)


def compare_runs(a: RunFile, b: RunFile, pairwise: dict[str, str] | None = None) -> dict[str, Any]:
    sa, sb = a.meta["summary"], b.meta["summary"]
    ra = {r.case_id: r for r in a.results}
    rb = {r.case_id: r for r in b.results}
    common = sorted(set(ra) & set(rb))
    warnings: list[str] = []
    # A judged run fails cases an unjudged run would pass, so per-case regressions are
    # only meaningful on the deterministic checks when the judge settings differ.
    same_judge = bool(a.meta.get("judge")) == bool(b.meta.get("judge"))

    def ok(r: CaseResult) -> bool:
        if same_judge:
            return r.passed
        return r.checks_passed if r.checks_passed is not None else r.passed

    if not same_judge:
        warnings.append(
            "judge settings differ between runs: regressions/fixes use deterministic checks only; "
            "pass_rate is not comparable, use checks_pass_rate"
        )
    if (
        a.meta.get("judge")
        and b.meta.get("judge")
        and a.meta.get("judge_model") != b.meta.get("judge_model")
    ):
        warnings.append("different judge models: judge score deltas are not comparable")
    regressions = [c for c in common if ok(ra[c]) and not ok(rb[c])]
    fixes = [c for c in common if not ok(ra[c]) and ok(rb[c])]

    def delta(x: float | None, y: float | None) -> float | None:
        return None if x is None or y is None else round(y - x, 3)

    out: dict[str, Any] = {
        "a": a.meta.get("label"),
        "b": b.meta.get("label"),
        "cases_compared": len(common),
        "checks_pass_rate": {
            "a": sa.get("checks_pass_rate"),
            "b": sb.get("checks_pass_rate"),
            "delta": delta(sa.get("checks_pass_rate"), sb.get("checks_pass_rate")),
        },
        "pass_rate": {
            "a": sa["pass_rate"],
            "b": sb["pass_rate"],
            "delta": delta(sa["pass_rate"], sb["pass_rate"]),
        },
        "latency_p95_ms": {"a": sa["latency_p95_ms"], "b": sb["latency_p95_ms"]},
        "judge_mean_delta": {
            k: delta(sa["judge_mean"].get(k), sb["judge_mean"].get(k))
            for k in sb.get("judge_mean", {})
        },
        "regressions": regressions,
        "fixes": fixes,
        "warnings": warnings,
    }
    if pairwise:
        wins = [pairwise[c] for c in common if c in pairwise]
        n = len(wins) or 1
        out["pairwise"] = {
            "b_win_rate": round(wins.count("B") / n, 3),
            "a_win_rate": round(wins.count("A") / n, 3),
            "tie_rate": round(wins.count("tie") / n, 3),
        }
    return out


@dataclass
class DriftAlert:
    metric: str
    baseline: Any
    current: Any
    reason: str


@dataclass
class DriftReport:
    alerts: list[DriftAlert] = field(default_factory=list)
    psi: dict[str, float] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return bool(self.alerts)


def detect_drift(
    baseline: RunFile,
    current: RunFile,
    *,
    max_pass_rate_drop: float = 0.05,
    max_judge_drop: float = 0.3,
    max_latency_increase: float = 0.3,
) -> DriftReport:
    sb, sc = baseline.meta["summary"], current.meta["summary"]
    report = DriftReport()

    pb, pc = sb["pass_rate"], sc["pass_rate"]
    if pb is not None and pc is not None and pb - pc > max_pass_rate_drop:
        report.alerts.append(
            DriftAlert(
                "pass_rate", sb["pass_rate"], sc["pass_rate"], f"dropped > {max_pass_rate_drop}"
            )
        )
    for k, v in sb.get("judge_mean", {}).items():
        cur = sc.get("judge_mean", {}).get(k)
        if cur is not None and v - cur > max_judge_drop:
            report.alerts.append(DriftAlert(f"judge.{k}", v, cur, f"dropped > {max_judge_drop}"))
    lb, lc = sb["latency_p95_ms"], sc["latency_p95_ms"]
    if lb and lc and lc > lb * (1 + max_latency_increase):
        report.alerts.append(
            DriftAlert(
                "latency_p95_ms",
                sb["latency_p95_ms"],
                sc["latency_p95_ms"],
                f"increased > {max_latency_increase:.0%}",
            )
        )
    # categorical distributions
    for key in ("tool_usage", "stop_reasons", "failure_tags"):
        value = psi(sb.get(key, {}), sc.get(key, {}))
        report.psi[key] = value
        if value > PSI_ALERT:
            report.alerts.append(
                DriftAlert(f"psi.{key}", sb.get(key), sc.get(key), f"PSI {value} > {PSI_ALERT}")
            )
    # any new safety-critical failure is an alert regardless of rates
    for tag in ("emergency_missed", "cross_patient_access"):
        if sc.get("failure_tags", {}).get(tag, 0) > sb.get("failure_tags", {}).get(tag, 0):
            report.alerts.append(
                DriftAlert(
                    f"failure.{tag}",
                    sb.get("failure_tags", {}).get(tag, 0),
                    sc["failure_tags"][tag],
                    "new safety-critical failure",
                )
            )
    return report
