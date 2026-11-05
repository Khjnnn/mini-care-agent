import pytest

from care_agent.safety.gates import EmergencyGate, ToolGuard
from care_agent.tools.registry import Permission, ToolRegistry


@pytest.mark.parametrize(
    "text",
    [
        "갑자기 가슴이 조이듯 아파요",
        "숨이 안 쉬어져요",
        "엄마가 쓰러졌어요",
        "요즘 죽고 싶다는 생각이 들어요",
        "수면제를 한꺼번에 먹었어요",
        "I have chest pain",
    ],
)
def test_emergency_detected(text):
    assert not EmergencyGate().check(text).allowed


@pytest.mark.parametrize(
    "text", ["최근 당화혈색소 수치가 어때요?", "혈압약 언제 먹어요?", "운동은 얼마나 해야 하나요?"]
)
def test_routine_questions_pass(text):
    assert EmergencyGate().check(text).allowed


@pytest.fixture
def tools():
    reg = ToolRegistry()

    @reg.register
    async def read_x(patient_id: str):
        """read"""

    @reg.register(permission=Permission.WRITE)
    async def write_x(patient_id: str, text: str):
        """write"""

    return reg


async def test_patient_scope(tools):
    guard = ToolGuard("p1")
    assert (await guard.check(tools.get("read_x"), {"patient_id": "p1"})).allowed
    denied = await guard.check(tools.get("read_x"), {"patient_id": "p2"})
    assert not denied.allowed and denied.reason.startswith("patient_scope_violation")


async def test_write_requires_approval(tools):
    tool, args = tools.get("write_x"), {"patient_id": "p1", "text": "hi"}

    assert not (await ToolGuard("p1").check(tool, args)).allowed

    async def yes(_tool, _args):
        return True

    async def no(_tool, _args):
        return False

    assert (await ToolGuard("p1", yes).check(tool, args)).allowed
    assert (await ToolGuard("p1", no).check(tool, args)).reason == "write_rejected_by_human"
    # scope is checked before asking a human
    assert not (await ToolGuard("p1", yes).check(tool, {**args, "patient_id": "p2"})).allowed
