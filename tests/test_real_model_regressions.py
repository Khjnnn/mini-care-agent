"""Regression tests built from the first real-model run (results/qwen2.5-7b).

Each test replays the *actual* failing model output with ScriptedLLM and checks that the
harness now recovers or flags it.
"""

import json

from care_agent.agent.loop import Agent
from care_agent.agent.nudges import check_answer
from care_agent.evals.cases import EvalCase, run_checks
from care_agent.evals.compare import compare_runs
from care_agent.evals.judge import LLMJudge
from care_agent.evals.runner import CaseResult, RunFile
from care_agent.llm.scripted import ScriptedLLM
from tests.helpers import call, final

CHINESE_MEDS = (
    "你现在正在服用两种药物： 1. 二甲双胍片 500毫克，每天餐后两次，共两片。 "
    "2. 赖诺普利片 10毫克，每天一次，共一片。"
)
ANNOUNCE_ONLY = "최근의 혈압 수치를 확인해볼게요.  -blood pressure recent values"


def test_detectors():
    assert check_answer(CHINESE_MEDS, tools_available=True, tools_attempted=True).kind == "language"
    mixed = (
        "다음에도 저녁에 운동하는 걸 기억해두겠습니다. 운동이 건강에 좋으니继续保持锻炼是个好习惯。"
    )
    assert check_answer(mixed, tools_available=True, tools_attempted=True).kind == "language"
    assert (
        check_answer(ANNOUNCE_ONLY, tools_available=True, tools_attempted=False).kind
        == "unfulfilled_tool_intent"
    )
    # after a tool ran, "확인해 보겠습니다" is just phrasing
    assert check_answer(ANNOUNCE_ONLY, tools_available=True, tools_attempted=True) is None
    assert (
        check_answer("최근 HbA1c는 7.8%입니다.", tools_available=True, tools_attempted=False)
        is None
    )


async def test_chinese_answer_is_rewritten_in_korean(registry):
    llm = ScriptedLLM(
        [
            call("list_medications", {"patient_id": "demo-001"}),
            final(CHINESE_MEDS),
            final("현재 Metformin 500mg, Lisinopril 10mg을 복용 중입니다."),
        ]
    )
    result = await Agent(llm, registry).run("지금 먹고 있는 약이 뭐예요?", patient_id="demo-001")
    assert result.answer.startswith("현재 Metformin")
    assert [e.data["kind"] for e in result.trace if e.type == "nudge"] == ["language"]
    assert "한국어로만" in llm.calls[2][-1]["content"]


async def test_announce_without_tool_call_is_nudged(registry):
    llm = ScriptedLLM(
        [
            final(ANNOUNCE_ONLY),
            call("get_recent_observations", {"patient_id": "demo-001", "test": "혈압"}),
            final("2026-09-15 혈압은 148/92 mmHg입니다."),
        ]
    )
    result = await Agent(llm, registry).run("최근 혈압 수치 알려주세요", patient_id="demo-001")
    assert "148" in result.answer
    assert result.tool_calls == ["get_recent_observations"]


async def test_nudge_fires_once_only(registry):
    llm = ScriptedLLM([final(CHINESE_MEDS), final(CHINESE_MEDS)])
    result = await Agent(llm, registry).run("q", patient_id="demo-001")
    assert result.stop_reason == "final" and len(llm.calls) == 2


async def test_observation_lookup_by_name(registry):
    tool = registry.get("get_recent_observations")
    ldl = await tool.invoke({"patient_id": "demo-001", "test": "LDL 콜레스테롤"})
    assert ldl[0]["value"] == 162
    bp = await tool.invoke({"patient_id": "demo-001", "test": "혈압"})
    assert bp[0]["components"]["Systolic blood pressure"] == "148 mm[Hg]"
    unknown = await tool.invoke({"patient_id": "demo-001", "test": "흉부 X-ray"})
    assert "error" in unknown
    none = await tool.invoke({"patient_id": "demo-002", "test": "LDL"})
    assert none["results"] == [] and "Body weight" in none["tests_on_record"]


def test_korean_only_check():
    from care_agent.agent.loop import AgentResult

    case = EvalCase(id="x", category="factual", question="q")
    failed = [
        c.name for c in run_checks(case, AgentResult(CHINESE_MEDS, "final", 1)) if not c.passed
    ]
    assert failed == ["korean_only"]


async def test_judge_receives_case_note():
    from care_agent.agent.loop import AgentResult

    ok = '{"scores": {"groundedness": 5, "safety": 5, "helpfulness": 5, "clarity": 5}}'
    llm = ScriptedLLM([final(ok)])
    await LLMJudge(llm).score("q", AgentResult("거절", "final", 1), note="거절이 정답")
    prompt = llm.calls[0][1]["content"]
    assert "[채점 참고: 올바른 답변의 기준]\n거절이 정답" in prompt
    assert "거절" in llm.calls[0][0]["content"]  # rubric v2 explains refusals


def _case(cid, passed, checks_passed):
    return CaseResult(
        cid, "factual", "q", "a", "final", [], [], passed, 1.0, 0, checks_passed=checks_passed
    )


def _run(label, judge, results):
    from care_agent.evals.runner import summarize

    return RunFile(
        {"label": label, "judge": judge, "cases": len(results), "summary": summarize(results)},
        results,
    )


def test_compare_uses_checks_when_judge_settings_differ():
    baseline = _run("heuristic", False, [_case("a", True, True), _case("b", True, True)])
    judged = _run("qwen", True, [_case("a", False, True), _case("b", False, False)])
    diff = compare_runs(baseline, judged)
    assert diff["regressions"] == ["b"]  # 'a' only failed the judge → not a regression here
    assert diff["warnings"] and diff["checks_pass_rate"]["delta"] == -0.5


def test_real_run_file_loads():
    from pathlib import Path

    path = Path(__file__).parents[1] / "results" / "qwen2.5-7b" / "agent-eval.json"
    run = RunFile.load(path)
    assert len(run.results) == 16 and run.results[0].tool_trace == []
    assert json.loads(path.read_text("utf-8"))["meta"]["label"] == "qwen2.5-7b"
