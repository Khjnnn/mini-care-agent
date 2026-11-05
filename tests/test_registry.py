from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from care_agent.tools.registry import Permission, ToolRegistry


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    async def echo(patient_id: str, n: Annotated[int, Field(ge=1, description="count")] = 1):
        """Echo things."""
        return {"patient_id": patient_id, "n": n}

    @reg.register(permission=Permission.WRITE, name="write_note")
    async def _write(text: str) -> str:
        """Write a note."""
        return text

    return reg


def test_schema_is_derived_from_signature():
    tool = make_registry().get("echo")
    schema = tool.json_schema()
    assert schema["required"] == ["patient_id"]
    assert schema["properties"]["n"]["minimum"] == 1
    assert schema["properties"]["n"]["description"] == "count"
    assert schema["additionalProperties"] is False
    assert tool.description == "Echo things."
    assert tool.patient_scoped


def test_openai_format_and_permissions():
    reg = make_registry()
    specs = reg.to_openai()
    assert [s["function"]["name"] for s in specs] == ["echo", "write_note"]
    assert reg.get("write_note").permission is Permission.WRITE
    assert not reg.get("write_note").patient_scoped


async def test_invoke_validates_arguments():
    tool = make_registry().get("echo")
    assert await tool.invoke({"patient_id": "p1", "n": 3}) == {"patient_id": "p1", "n": 3}
    with pytest.raises(ValidationError):
        await tool.invoke({"patient_id": "p1", "n": 0})
    with pytest.raises(ValidationError):
        await tool.invoke({"patient_id": "p1", "unexpected": True})


def test_registration_errors():
    reg = ToolRegistry()

    with pytest.raises(TypeError):

        @reg.register
        def not_async(x: int):
            """sync"""

    with pytest.raises(ValueError):

        @reg.register
        async def no_doc(x: int): ...

    @reg.register
    async def dup(x: int):
        """d"""

    with pytest.raises(ValueError):
        reg.register(dup)
