"""Provider-neutral LLM interface (OpenAI chat-completions message format)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    #: Set when the model emitted arguments that are not valid JSON.
    parse_error: str | None = None


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None
    #: Free-form metadata from wrappers, e.g. the router's decision.
    meta: dict[str, Any] = field(default_factory=dict)

    def as_message(self) -> Message:
        msg: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


class LLM(Protocol):
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...
