import pytest

from care_agent.agent.loop import Agent
from care_agent.llm.scripted import ScriptedLLM
from care_agent.memory.store import MemoryStore, SessionPatientMismatch
from care_agent.safety.output_gate import CAUTION, OutputGate
from tests.helpers import call, final


def test_session_bound_to_patient():
    store = MemoryStore()
    store.open_session("s1", "p1")
    store.open_session("s1", "p1")  # idempotent
    with pytest.raises(SessionPatientMismatch):
        store.open_session("s1", "p2")


def test_notes_search():
    store = MemoryStore()
    store.add_note("p1", "저녁에 걷기 운동을 선호함")
    store.add_note("p1", "목표: 체중 5kg 감량")
    store.add_note("p2", "저녁 운동 싫어함")
    assert [n.note for n in store.search_notes("p1")] == [
        "목표: 체중 5kg 감량",
        "저녁에 걷기 운동을 선호함",
    ]
    assert [n.note for n in store.search_notes("p1", "운동")] == ["저녁에 걷기 운동을 선호함"]


async def test_history_is_replayed_in_next_turn(registry):
    store = MemoryStore()
    llm = ScriptedLLM([final("HbA1c는 7.8%입니다."), final("네, 3월보다 올랐습니다.")])
    agent = Agent(llm, registry, memory=store)
    await agent.run("당화혈색소?", patient_id="demo-001", session_id="s")
    await agent.run("지난번보다 올랐어?", patient_id="demo-001", session_id="s")
    second = llm.calls[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[2]["content"] == "HbA1c는 7.8%입니다."
    with pytest.raises(SessionPatientMismatch):
        await agent.run("q", patient_id="demo-002", session_id="s")


async def test_memory_tools_are_patient_scoped(make_platform):
    llm = ScriptedLLM(
        [
            call("save_patient_memory", {"patient_id": "demo-001", "note": "저녁 운동 선호"}),
            call("search_patient_memory", {"patient_id": "demo-002"}, "c2"),
            final("ok"),
        ]
    )
    platform = make_platform(llm)
    result = await platform.agent("care_chat").run("기억해줘", patient_id="demo-001")
    assert [e.type for e in result.trace if e.type.startswith("tool")] == [
        "tool_call",
        "tool_denied",
    ]
    assert platform.memory.search_notes("demo-001")[0].note == "저녁 운동 선호"


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("주민번호 900101-1234567 확인", ["rrn"]),
        ("연락처 010-1234-5678 / a.b@test.com", ["phone", "email"]),
        ("HbA1c 7.8% (2026-09-15)", []),
    ],
)
def test_pii_masking(text, masked):
    out = OutputGate().check(text)
    assert out.masked == masked
    assert "1234567" not in out.text and "5678" not in out.text and "@" not in out.text


@pytest.mark.parametrize(
    "text",
    [
        "메트포르민 복용을 중단하세요.",
        "약을 바로 끊으셔도 됩니다.",
        "당뇨병으로 확진입니다.",
        "용량을 두 배로 늘리세요.",
    ],
)
def test_prescriptive_language_flagged_with_caution(text):
    out = OutputGate().check(text)
    assert out.prescriptive and out.text.endswith(CAUTION)


def test_safe_advice_not_flagged():
    out = OutputGate().check("약 복용에 대해서는 담당 의사와 상의해 보세요.")
    assert not out.modified


async def test_output_gate_applies_to_final_answer(registry):
    llm = ScriptedLLM([final("혈압약 복용을 중단하세요. 문의: 010-0000-1111")])
    result = await Agent(llm, registry).run("q", patient_id="demo-001")
    assert "[전화번호]" in result.answer and result.answer.endswith(CAUTION)
    assert any(e.type == "output_gate" for e in result.trace)
