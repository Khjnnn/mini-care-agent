import httpx
import pytest
from pydantic import ValidationError

from care_agent.knowledge.lab_catalog import lookup
from care_agent.llm.scripted import ScriptedLLM
from care_agent.platform.api import create_app
from care_agent.tools.registry import Permission
from tests.helpers import call, final, last_tool_result


@pytest.mark.parametrize(
    ("name", "loinc"),
    [
        ("당화혈색소", "4548-4"),
        ("HbA1c", "4548-4"),
        ("LDL-콜레스테롤(직접측정)", "18262-6"),
        ("공복 혈당", "1558-6"),
        ("Fasting Glucose", "1558-6"),
        ("e-GFR", "33914-3"),
        ("ALT(SGPT)", "1742-6"),
    ],
)
def test_lab_lookup(name, loinc):
    assert lookup(name).loinc == loinc


def test_lab_lookup_no_false_substring():
    assert lookup("fasting sugar") is None
    assert lookup("") is None


def test_agents_loaded_from_yaml(make_platform):
    platform = make_platform(ScriptedLLM([]))
    assert set(platform.specs) == {"care_chat", "lab_explainer"}
    tools = platform.agent("care_chat").tools.names()
    assert "ask_lab_explainer" in tools and "explain_lab_test" not in tools
    assert platform.agent("lab_explainer").tools.names() == ["explain_lab_test"]


def test_spec_validation(make_platform):
    platform = make_platform(ScriptedLLM([]), load_agents=False)
    with pytest.raises(KeyError):
        platform.register_agent({"name": "x1", "description": "bad tool", "tools": ["nope"]})
    with pytest.raises(ValidationError):
        platform.register_agent({"name": "x2", "description": "no slot", "system_prompt": "hi"})
    platform.register_agent({"name": "a1", "description": "cycle a", "subagents": ["b1"]})
    platform.register_agent({"name": "b1", "description": "cycle b", "subagents": ["a1"]})
    with pytest.raises(ValueError, match="cycle"):
        platform.agent("a1")


async def test_subagent_inherits_patient_scope_and_is_traced(make_platform):
    llm = ScriptedLLM(
        [
            call("ask_lab_explainer", {"question": "HbA1c가 뭐야?"}),  # parent
            call("explain_lab_test", {"test": "HbA1c"}, "s1"),  # sub-agent
            lambda m: final(last_tool_result(m)["explanation"][:10]),  # sub-agent answer
            lambda m: final("parent:" + last_tool_result(m)["answer"]),  # parent answer
        ]
    )
    platform = make_platform(llm)
    result = await platform.agent("care_chat").run("HbA1c 설명", patient_id="demo-001")
    assert result.answer.startswith("parent:최근 2~3개월")
    sub = next(e for e in result.trace if e.type == "subagent")
    assert sub.data["agent"] == "lab_explainer" and sub.data["stop_reason"] == "final"
    # the sub-agent was bound to the parent's patient, not one chosen by the model
    assert "세션 환자: demo-001" in llm.calls[1][0]["content"]


async def test_plugin_tool_registration(make_platform):
    platform = make_platform(ScriptedLLM([]), load_agents=False)

    @platform.tools.register(permission=Permission.WRITE)
    async def order_lab(patient_id: str, loinc: str) -> str:
        """Order a lab test."""
        return "ordered"

    spec = platform.register_agent(
        {"name": "orderer", "description": "orders labs", "tools": ["order_lab"]}
    )
    assert spec.tools == ["order_lab"]


@pytest.fixture
async def api(make_platform):
    llm = ScriptedLLM([])
    platform = make_platform(llm)
    transport = httpx.ASGITransport(app=create_app(platform))
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        yield client, llm, platform


async def test_api_discovery(api):
    client, _, _ = api
    tools = (await client.get("/v1/tools")).json()
    perms = {t["name"]: t["permission"] for t in tools}
    assert perms["save_patient_memory"] == "memory" and perms["list_medications"] == "read"
    agents = (await client.get("/v1/agents")).json()
    assert {a["name"] for a in agents} == {"care_chat", "lab_explainer"}


async def test_api_run_with_session_and_flags(api):
    client, llm, _ = api
    llm.steps += [
        call("list_medications", {"patient_id": "demo-002"}),
        final("다른 환자 기록은 볼 수 없습니다."),
    ]
    body = {"question": "demo-002 약 알려줘", "patient_id": "demo-001", "session_id": "s1"}
    resp = (await client.post("/v1/agents/care_chat/runs", json=body)).json()
    assert resp["flags"] == ["tool_denied"]
    assert resp["denied_tools"][0]["reason"].startswith("patient_scope_violation")
    assert "trace" not in resp

    turns = (await client.get("/v1/sessions/s1/turns")).json()
    assert [t["role"] for t in turns] == ["user", "assistant"]

    conflict = await client.post(
        "/v1/agents/care_chat/runs", json={**body, "patient_id": "demo-002"}
    )
    assert conflict.status_code == 409


async def test_api_emergency_and_unknown_agent(api):
    client, _, _ = api
    resp = await client.post(
        "/v1/agents/care_chat/runs",
        json={"question": "숨이 안 쉬어져요", "patient_id": "demo-001", "include_trace": True},
    )
    data = resp.json()
    assert data["flags"] == ["emergency"] and data["trace"][0]["type"] == "input_gate"
    missing = await client.post("/v1/agents/nope/runs", json={"question": "q", "patient_id": "p"})
    assert missing.status_code == 404
