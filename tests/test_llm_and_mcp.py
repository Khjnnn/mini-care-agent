import json

import httpx

from care_agent.llm.openai_compat import OpenAICompatLLM, parse_chat_completion
from care_agent.mcp_server import build_mcp_server
from care_agent.tools.registry import Permission


def test_parse_tool_calls_including_bad_json():
    resp = parse_chat_completion(
        {
            "model": "m",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "a", "function": {"name": "t", "arguments": '{"x": 1}'}},
                            {"id": "b", "function": {"name": "t", "arguments": "{oops"}},
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 3},
        }
    )
    assert resp.tool_calls[0].arguments == {"x": 1} and resp.tool_calls[0].parse_error is None
    assert resp.tool_calls[1].parse_error.startswith("invalid JSON")
    assert resp.usage == {"prompt_tokens": 3}
    assert json.loads(resp.as_message()["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


async def test_openai_compat_request_shape(registry):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    llm = OpenAICompatLLM("http://vllm:8000/v1", "qwen", transport=httpx.MockTransport(handler))
    resp = await llm.complete([{"role": "user", "content": "x"}], registry.to_openai())
    await llm.aclose()
    assert seen["url"] == "http://vllm:8000/v1/chat/completions"
    assert seen["body"]["tool_choice"] == "auto" and len(seen["body"]["tools"]) == len(registry)
    assert resp.content == "hi"


async def test_mcp_exports_read_tools_only(registry):
    @registry.register(permission=Permission.WRITE)
    async def add_note(patient_id: str, text: str) -> str:
        """Add a note."""
        return "ok"

    server = build_mcp_server(registry)
    names = {t.name for t in await server.list_tools()}
    assert names == {
        "get_patient_summary",
        "list_conditions",
        "get_recent_observations",
        "list_medications",
    }
    result = await server.call_tool("get_patient_summary", {"patient_id": "demo-001"})
    text = json.dumps(result.model_dump(), ensure_ascii=False)
    assert "1968" in text and "Synthetic" not in text
