import base64
import json
from pathlib import Path

import httpx
import pytest

from care_agent.agent.loop import AgentResult
from care_agent.agent.trace import TraceEvent
from care_agent.evals.analysis import markdown_report
from care_agent.evals.cases import EvalCase, load_cases, run_checks
from care_agent.evals.compare import compare_runs, detect_drift, psi
from care_agent.evals.judge import LLMJudge
from care_agent.evals.runner import EvalRunner, RunFile
from care_agent.llm.heuristic import HeuristicLLM
from care_agent.llm.scripted import ScriptedLLM
from care_agent.observability.langfuse import LangfuseExporter
from care_agent.safety.output_gate import OutputGate
from tests.helpers import final

CASES = Path(__file__).parents[1] / "datasets" / "agent_eval" / "cases.yaml"


@pytest.fixture
def cases():
    return load_cases(CASES)


async def run_variant(make_platform, cases, variant):
    platform = make_platform(HeuristicLLM(variant))
    runner = EvalRunner(lambda _c: platform.agent("care_chat"), concurrency=3)
    return await runner.run(cases, label=variant)


async def test_suite_end_to_end_and_ab(make_platform, cases):
    careful = await run_variant(make_platform, cases, "careful")
    naive = await run_variant(make_platform, cases, "naive")

    assert careful.meta["summary"]["pass_rate"] == 1.0
    assert naive.meta["summary"]["pass_rate_by_category"]["prescriptive"] == 0.0
    # the naive model *tried* to read demo-002, but the guard kept the injection cases green
    inj = next(r for r in naive.results if r.case_id == "inject-ignore")
    assert inj.passed and inj.tool_calls == []

    diff = compare_runs(careful, naive)
    assert set(diff["regressions"]) == {"rx-stop-metformin", "rx-double-dose"}
    assert diff["pass_rate"]["delta"] < 0

    drift = detect_drift(careful, naive)
    assert drift.drifted
    assert {a.metric for a in drift.alerts} >= {"pass_rate", "psi.failure_tags"}
    assert not detect_drift(careful, careful).drifted

    report = markdown_report(naive)
    assert "`prescriptive_output`" in report and "rx-stop-metformin" in report


def test_run_file_roundtrip(tmp_path, cases):
    run = RunFile({"label": "x", "cases": 0, "summary": {}}, [])
    run.save(tmp_path / "r.json")
    assert RunFile.load(tmp_path / "r.json").meta["label"] == "x"


def test_checks_detect_cross_patient_read():
    case = EvalCase(id="c", category="injection", question="q", forbidden_patient="p2")
    leaked = AgentResult(
        "ok",
        "final",
        1,
        [
            TraceEvent(
                "tool_call", {"name": "list_medications", "arguments": {"patient_id": "p2"}}, 1
            )
        ],
    )
    failed = [c.name for c in run_checks(case, leaked) if not c.passed]
    assert failed == ["no_cross_patient_access"]


def test_case_file_validation(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text(
        "cases:\n  - {id: a, category: factual, question: q}\n  - {id: a, category: factual, question: q}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(bad)


class Boom:
    async def run(self, *a, **k):
        raise TimeoutError("backend down")


async def test_infra_errors_are_recorded_not_raised(cases):
    runner = EvalRunner(lambda _c: Boom(), retries=1)
    run = await runner.run(cases[:2])
    assert all(r.error and r.failure_tags == ["infra_error"] for r in run.results)
    assert run.meta["summary"]["errors"] == 2


def _result(answer="HbA1c 7.8%"):
    return AgentResult(
        answer, "final", 2, [TraceEvent("tool_call", {"name": "t", "result": {"v": 7.8}}, 1)]
    )


async def test_judge_scores_with_retry():
    llm = ScriptedLLM(
        [
            final("점수: 5점"),
            final(
                '{"scores": {"groundedness": 5, "safety": 4, "helpfulness": 4, "clarity": 5}, "rationale": "ok"}'
            ),
        ]
    )
    verdict = await LLMJudge(llm).score("q", _result())
    assert verdict.scores.safety == 4
    assert '"v": 7.8' in llm.calls[0][1]["content"]  # evidence is shown to the judge


async def test_judge_low_score_fails_case(make_platform, cases):
    platform = make_platform(HeuristicLLM("careful"))
    low = '{"scores": {"groundedness": 1, "safety": 5, "helpfulness": 4, "clarity": 4}, "rationale": "made up"}'
    judge = LLMJudge(ScriptedLLM([final(low)]))
    case = next(c for c in cases if c.id == "fact-a1c")
    run = await EvalRunner(lambda _c: platform.agent("care_chat"), judge=judge).run([case])
    r = run.results[0]
    assert not r.passed and "low_groundedness" in r.failure_tags
    assert run.meta["summary"]["judge_mean"]["groundedness"] == 1


async def test_pairwise_position_bias_becomes_tie():
    consistent = ScriptedLLM([final('{"winner": "A"}'), final('{"winner": "B"}')])
    assert (await LLMJudge(consistent).compare("q", _result(), _result())).winner == "A"
    biased = ScriptedLLM([final('{"winner": "A"}'), final('{"winner": "A"}')])
    assert (await LLMJudge(biased).compare("q", _result(), _result())).winner == "tie"


def test_psi():
    assert psi({"a": 10, "b": 10}, {"a": 10, "b": 10}) == 0
    assert psi({"a": 10, "b": 10}, {"a": 19, "b": 1}) > 0.2


def test_output_gate_does_not_flag_safe_deferral():
    safe = "약을 끊거나 용량을 바꾸는 것은 반드시 담당 의사와 상의한 뒤 결정하세요."
    assert not OutputGate().check(safe).modified


async def test_langfuse_batch_and_send():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(207, json={"successes": [], "errors": []})

    exporter = LangfuseExporter(
        "https://lf.test",
        "pk",
        "sk",
        redact_tool_results=True,
        transport=httpx.MockTransport(handler),
    )
    result = AgentResult(
        "answer",
        "final",
        2,
        [
            TraceEvent("input_gate", {"allowed": True}, 0),
            TraceEvent(
                "llm",
                {"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                1,
                12.0,
            ),
            TraceEvent(
                "tool_call",
                {"name": "list_medications", "arguments": {"patient_id": "p"}, "result": ["x"]},
                1,
                3.0,
            ),
            TraceEvent(
                "tool_denied", {"name": "list_medications", "arguments": {}, "reason": "scope"}, 2
            ),
        ],
    )
    trace_id = await exporter.export(result, name="agent:care_chat", question="q", tags=["t"])
    await exporter.aclose()

    assert seen["auth"] == "Basic " + base64.b64encode(b"pk:sk").decode()
    batch = seen["body"]["batch"]
    kinds = [e["type"] for e in batch]
    assert kinds == [
        "trace-create",
        "event-create",
        "generation-create",
        "span-create",
        "span-create",
    ]
    assert batch[0]["body"]["id"] == trace_id and "stop:final" in batch[0]["body"]["tags"]
    assert batch[2]["body"]["usage"]["input"] == 10
    assert batch[3]["body"]["output"] == "[redacted]"
    assert batch[4]["body"]["level"] == "WARNING"


async def test_langfuse_failure_is_swallowed():
    exporter = LangfuseExporter(
        "https://lf.test",
        "pk",
        "sk",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="down")),
    )
    assert await exporter.send([]) is False


async def test_broken_judge_is_reported(make_platform, cases):
    platform = make_platform(HeuristicLLM("careful"))
    judge = LLMJudge(ScriptedLLM([final("not json"), final("still not json")]))
    case = next(c for c in cases if c.id == "fact-a1c")
    run = await EvalRunner(lambda _c: platform.agent("care_chat"), judge=judge).run([case])
    assert run.results[0].judge is None and "judge output invalid" in run.results[0].judge_error
    assert run.meta["summary"]["judge_failures"] == 1


def test_old_run_files_still_load(tmp_path):
    baseline = Path(__file__).parents[1] / "evals" / "baselines" / "heuristic-careful.json"
    assert RunFile.load(baseline).results[0].judge_error is None
