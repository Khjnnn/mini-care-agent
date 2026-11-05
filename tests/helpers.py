import json

from care_agent.llm.base import LLMResponse, ToolCall


def call(name, args, i="c1", parse_error=None):
    return LLMResponse(None, [ToolCall(i, name, args, parse_error)])


def final(text):
    return LLMResponse(text)


def last_tool_result(messages):
    return json.loads(next(m for m in reversed(messages) if m["role"] == "tool")["content"])
