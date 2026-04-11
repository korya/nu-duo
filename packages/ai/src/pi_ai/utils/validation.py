"""Tool-call argument validation.

Port of ``packages/ai/src/utils/validation.ts``. Replaces AJV with the
``jsonschema`` package; type coercion is implemented manually because
``jsonschema`` does not coerce by default.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import jsonschema
from jsonschema.validators import Draft7Validator

if TYPE_CHECKING:
    from pi_ai.types import Tool, ToolCall


def validate_tool_call(tools: list[Tool], tool_call: ToolCall) -> dict[str, Any]:
    """Locate ``tool_call.name`` in ``tools`` and validate its arguments.

    Raises :class:`ValueError` when the tool is unknown or validation fails.
    """
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> dict[str, Any]:
    """Validate ``tool_call.arguments`` against ``tool.parameters``.

    Mirrors the upstream behaviour of coercing simple scalar types (string ↔
    number/boolean) before reporting failure, so LLMs that return stringified
    numbers still satisfy integer schemas.
    """
    args = copy.deepcopy(tool_call.arguments)
    schema = tool.parameters
    _coerce(args, schema)

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(args), key=lambda e: list(e.path))
    if not errors:
        return args

    formatted_errors: list[str] = []
    for err in errors:
        if err.path:
            path = ".".join(str(p) for p in err.path)
        elif err.validator == "required":
            path = err.message.split("'")[1] if "'" in err.message else "root"
        else:
            path = "root"
        formatted_errors.append(f"  - {path}: {err.message}")

    received = json.dumps(tool_call.arguments, indent=2)
    raise ValueError(
        f'Validation failed for tool "{tool_call.name}":\n'
        + "\n".join(formatted_errors)
        + f"\n\nReceived arguments:\n{received}"
    )


# ---------------------------------------------------------------------------
# Minimal type coercion — AJV-compatible subset.
# ---------------------------------------------------------------------------


def _coerce(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value
    schema_type = schema.get("type")

    if schema_type == "object" and isinstance(value, dict):
        props = schema.get("properties") or {}
        for key, child_schema in props.items():
            if key in value:
                value[key] = _coerce(value[key], child_schema)
        return value

    if schema_type == "array" and isinstance(value, list):
        items_schema = schema.get("items")
        if items_schema is not None:
            for i, item in enumerate(value):
                value[i] = _coerce(item, items_schema)
        return value

    if schema_type in {"integer", "number"} and isinstance(value, str):
        try:
            return int(value) if schema_type == "integer" else float(value)
        except ValueError:
            return value

    if schema_type == "boolean" and isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

    return value


__all__ = ["validate_tool_arguments", "validate_tool_call"]


# Keep ``jsonschema`` referenced for static analyzers that strip unused imports.
_ = jsonschema
