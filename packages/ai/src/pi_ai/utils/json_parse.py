"""Tolerant JSON parser for streaming tool-call arguments.

Port of ``packages/ai/src/utils/json-parse.ts``. Tries ``json.loads`` first
(fastest for complete documents), then falls back to
``partial_json_parser.loads`` (the Python port of the ``partial-json`` npm
package the TS version depends on). Always returns a dict/list/value — never
raises.
"""

from __future__ import annotations

import json
from typing import Any

from partial_json_parser import loads as partial_loads  # type: ignore[import-untyped]


def parse_streaming_json(partial_json: str | None) -> Any:
    """Attempt to parse ``partial_json``, tolerating truncation.

    Returns an empty ``dict`` when the input is ``None``, empty, whitespace,
    or cannot be parsed at all.
    """
    if not partial_json or not partial_json.strip():
        return {}
    try:
        return json.loads(partial_json)
    except ValueError:
        pass
    try:
        result = partial_loads(partial_json)
        return result if result is not None else {}
    except (ValueError, Exception):
        return {}


__all__ = ["parse_streaming_json"]
