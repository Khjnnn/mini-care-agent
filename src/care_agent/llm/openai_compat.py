"""Client for any OpenAI-compatible ``/chat/completions`` endpoint.

Works with vLLM (``vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes``),
Ollama, LiteLLM, OpenAI, etc. — which is what lets the router (week 5) treat on-prem
GPUs and cloud APIs as interchangeable backends.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from care_agent.llm.base import LLMResponse, Message, ToolCall


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        *,
        temperature: float = 0.0,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        resp = await self._http.post("chat/completions", json=body)
        resp.raise_for_status()
        return parse_chat_completion(resp.json())

    async def aclose(self) -> None:
        await self._http.aclose()


def parse_chat_completion(data: dict[str, Any]) -> LLMResponse:
    msg = data["choices"][0]["message"]
    calls = []
    for raw in msg.get("tool_calls") or []:
        fn = raw["function"]
        args_text = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_text)
            error = None if isinstance(args, dict) else "arguments must be a JSON object"
        except json.JSONDecodeError as e:
            args, error = {}, f"invalid JSON arguments: {e}"
        calls.append(
            ToolCall(
                id=raw.get("id") or f"call_{len(calls)}",
                name=fn["name"],
                arguments=args if isinstance(args, dict) else {},
                parse_error=error,
            )
        )
    return LLMResponse(
        content=msg.get("content"),
        tool_calls=calls,
        usage=data.get("usage") or {},
        model=data.get("model"),
    )
