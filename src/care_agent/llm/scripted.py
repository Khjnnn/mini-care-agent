"""Deterministic fake LLM for tests and offline demos."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from care_agent.llm.base import LLMResponse, Message

Step = LLMResponse | Callable[[list[Message]], LLMResponse]


class ScriptedLLM:
    """Replays a fixed list of responses. A step may be a callable that inspects history."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append(copy.deepcopy(messages))
        if not self.steps:
            raise RuntimeError("ScriptedLLM ran out of scripted steps")
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step
