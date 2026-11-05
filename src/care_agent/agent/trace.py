"""Structured trace events.

Kept as plain data so an exporter (Langfuse in week 4) can be bolted on without
touching the loop.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EventType = Literal[
    "input_gate",
    "llm",
    "tool_call",
    "tool_denied",
    "tool_error",
    "subagent",
    "nudge",
    "output_gate",
    "final",
]


@dataclass
class TraceEvent:
    type: EventType
    data: dict[str, Any]
    step: int
    duration_ms: float | None = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
