"""Tool registry.

A tool is a plain typed ``async def``. The registry derives a strict pydantic input
model (and therefore a JSON schema) from the function signature, so the same
definition can be served to an OpenAI-style tool-calling LLM, exposed over MCP, and
checked by the safety layer — without writing the schema twice.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

ToolFn = Callable[..., Awaitable[Any]]

#: Argument name that marks a tool as operating on a single patient's record.
PATIENT_ARG = "patient_id"


class Permission(StrEnum):
    """How much a tool can change the world. Used by the safety gate."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    permission: Permission
    fn: ToolFn
    input_model: type[BaseModel]

    @property
    def patient_scoped(self) -> bool:
        return PATIENT_ARG in self.input_model.model_fields

    def json_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def validate(self, arguments: dict[str, Any]) -> BaseModel:
        """Raise ``pydantic.ValidationError`` if the arguments do not match the schema."""
        return self.input_model.model_validate(arguments)

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        args = self.validate(arguments)
        return await self.fn(**dict(args))


def _input_model(fn: ToolFn, tool_name: str) -> type[BaseModel]:
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"tool {tool_name!r}: *args/**kwargs are not allowed")
        if param.name not in hints:
            raise TypeError(f"tool {tool_name!r}: parameter {param.name!r} needs a type hint")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (hints[param.name], default)
    model_name = "".join(part.title() for part in tool_name.split("_")) + "Input"
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)


class ToolRegistry:
    """Name → Tool mapping with a decorator for registration."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        fn: ToolFn | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        permission: Permission = Permission.READ,
    ) -> Any:
        def decorator(f: ToolFn) -> ToolFn:
            if not inspect.iscoroutinefunction(f):
                raise TypeError(f"tool {f.__name__!r} must be an async function")
            tool_name = name or f.__name__
            if tool_name in self._tools:
                raise ValueError(f"tool {tool_name!r} is already registered")
            desc = description or inspect.getdoc(f)
            if not desc:
                raise ValueError(f"tool {tool_name!r} needs a description or docstring")
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=desc,
                permission=permission,
                fn=f,
                input_model=_input_model(f, tool_name),
            )
            return f

        return decorator(fn) if fn is not None else decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def to_openai(self) -> list[dict[str, Any]]:
        return [t.to_openai() for t in self._tools.values()]

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
