"""Tests for ``nu_ai.utils.validation``."""

from __future__ import annotations

from typing import Any

import pytest
from nu_ai.types import Tool, ToolCall
from nu_ai.utils.validation import validate_tool_arguments, validate_tool_call


def _tool(name: str = "t", *, parameters: dict[str, Any] | None = None) -> Tool:
    return Tool(
        name=name,
        description="",
        parameters=parameters
        or {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    )


def _call(name: str = "t", *, arguments: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments={"x": 1} if arguments is None else arguments)


def test_validate_tool_call_unknown_tool() -> None:
    with pytest.raises(ValueError, match="not found"):
        validate_tool_call([], _call())


def test_validate_tool_call_passthrough() -> None:
    assert validate_tool_call([_tool()], _call()) == {"x": 1}


def test_coerce_string_to_integer() -> None:
    args = validate_tool_arguments(_tool(), _call(arguments={"x": "42"}))
    assert args == {"x": 42}


def test_coerce_string_to_number() -> None:
    tool = _tool(parameters={"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]})
    args = validate_tool_arguments(tool, _call(arguments={"x": "3.5"}))
    assert args == {"x": 3.5}


def test_coerce_invalid_integer_passes_through() -> None:
    # Non-numeric string can't be coerced; validation then fails.
    with pytest.raises(ValueError, match="x"):
        validate_tool_arguments(_tool(), _call(arguments={"x": "not a number"}))


def test_coerce_string_to_boolean() -> None:
    tool = _tool(parameters={"type": "object", "properties": {"x": {"type": "boolean"}}, "required": ["x"]})
    assert validate_tool_arguments(tool, _call(arguments={"x": "true"})) == {"x": True}
    assert validate_tool_arguments(tool, _call(arguments={"x": "false"})) == {"x": False}


def test_coerce_array_items() -> None:
    tool = _tool(
        parameters={
            "type": "object",
            "properties": {"xs": {"type": "array", "items": {"type": "integer"}}},
            "required": ["xs"],
        }
    )
    args = validate_tool_arguments(tool, _call(arguments={"xs": ["1", "2", "3"]}))
    assert args == {"xs": [1, 2, 3]}


def test_coerce_nested_object() -> None:
    tool = _tool(
        parameters={
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "required": ["y"],
                },
            },
            "required": ["outer"],
        }
    )
    args = validate_tool_arguments(tool, _call(arguments={"outer": {"y": "9"}}))
    assert args == {"outer": {"y": 9}}


def test_required_field_missing() -> None:
    with pytest.raises(ValueError, match="Validation failed"):
        validate_tool_arguments(_tool(), _call(arguments={}))


def test_validation_error_includes_received_args() -> None:
    with pytest.raises(ValueError, match="Received arguments"):
        validate_tool_arguments(_tool(), _call(arguments={}))


def test_coerce_returns_value_when_schema_not_dict() -> None:
    # Hits the early _coerce return when schema is not a dict (degenerate case
    # — exercised via a Tool whose parameters don't match the upstream shape).
    tool = Tool(name="t", description="", parameters={"type": "object", "properties": {"x": True}})
    call = ToolCall(id="c1", name="t", arguments={"x": "anything"})
    args = validate_tool_arguments(tool, call)
    assert args == {"x": "anything"}
