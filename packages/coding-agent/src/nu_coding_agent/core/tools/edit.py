"""``edit`` tool — exact-text replacement with optional fuzzy fallback.

Direct port of ``packages/coding-agent/src/core/tools/edit.ts``. Each
``edit`` call carries one or more ``{old_text, new_text}`` pairs that
get matched against the *original* file content (not against each
other's results) and applied in a single atomic write.

The TS version supports a legacy ``{oldText, newText}`` flat-form input;
the Python port keeps that compatibility shim via ``prepare_arguments``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

from nu_coding_agent.core.tools.edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from nu_coding_agent.core.tools.path_utils import resolve_to_cwd

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EditToolDetails:
    diff: str
    first_changed_line: int | None = None


@dataclass(slots=True)
class EditOperations:
    """Pluggable I/O hooks for ``edit``."""

    read_file: Callable[[str], Awaitable[bytes]] | None = None
    write_file: Callable[[str, str], Awaitable[None]] | None = None
    access: Callable[[str], Awaitable[None]] | None = None


# ---------------------------------------------------------------------------
# Default operations
# ---------------------------------------------------------------------------


def _read_bytes_sync(absolute_path: str) -> bytes:
    return Path(absolute_path).read_bytes()


def _write_text_sync(absolute_path: str, content: str) -> None:
    Path(absolute_path).write_text(content, encoding="utf-8")


def _check_readable_sync(absolute_path: str) -> None:
    p = Path(absolute_path)
    if not p.exists():
        raise FileNotFoundError(absolute_path)
    if not p.is_file():
        raise IsADirectoryError(absolute_path)


async def _default_read_file(absolute_path: str) -> bytes:
    return await asyncio.to_thread(_read_bytes_sync, absolute_path)


async def _default_write_file(absolute_path: str, content: str) -> None:
    await asyncio.to_thread(_write_text_sync, absolute_path, content)


async def _default_access(absolute_path: str) -> None:
    await asyncio.to_thread(_check_readable_sync, absolute_path)


def _default_operations() -> EditOperations:
    return EditOperations(
        read_file=_default_read_file,
        write_file=_default_write_file,
        access=_default_access,
    )


# ---------------------------------------------------------------------------
# Argument prep / validation
# ---------------------------------------------------------------------------


def _prepare_edit_arguments(args: Any) -> dict[str, Any]:
    """Promote legacy flat-form ``{oldText, newText}`` into ``edits[]``.

    Mirrors the upstream ``prepareEditArguments`` shim that lets older
    extensions emit the single-edit shape without breaking the schema.
    """
    if not isinstance(args, dict):
        return args  # type: ignore[unreachable]
    old_text = args.get("oldText")
    new_text = args.get("newText")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return args
    edits = list(args.get("edits") or [])
    edits.append({"oldText": old_text, "newText": new_text})
    rest = {k: v for k, v in args.items() if k not in {"oldText", "newText"}}
    return {**rest, "edits": edits}


def _validate_edit_input(input_: dict[str, Any]) -> tuple[str, list[Edit]]:
    edits = input_.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
    parsed: list[Edit] = []
    for raw in edits:
        # Both ``oldText``/``newText`` (TS-style camelCase from the wire) and
        # ``old_text``/``new_text`` (Python-style) are accepted.
        old_text = raw.get("oldText", raw.get("old_text"))
        new_text = raw.get("newText", raw.get("new_text"))
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ValueError("Each edit must have string oldText and newText fields.")
        parsed.append(Edit(old_text=old_text, new_text=new_text))
    return input_["path"], parsed


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


_REPLACE_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "oldText": {
            "type": "string",
            "description": (
                "Exact text for one targeted replacement. It must be unique in the original file "
                "and must not overlap with any other edits[].oldText in the same call."
            ),
        },
        "newText": {
            "type": "string",
            "description": "Replacement text for this targeted edit.",
        },
    },
    "required": ["oldText", "newText"],
    "additionalProperties": False,
}


_EDIT_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to edit (relative or absolute)",
        },
        "edits": {
            "type": "array",
            "items": _REPLACE_EDIT_SCHEMA,
            "description": (
                "One or more targeted replacements. Each edit is matched against the original "
                "file, not incrementally. Do not include overlapping or nested edits. If two "
                "changes touch the same block or nearby lines, merge them into one edit instead."
            ),
        },
    },
    "required": ["path", "edits"],
    "additionalProperties": False,
}


_EDIT_DESCRIPTION = (
    "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
    "non-overlapping region of the original file. If two changes affect the same block or nearby "
    "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
    "unchanged regions just to connect distant changes."
)


def create_edit_tool(
    cwd: str,
    *,
    operations: EditOperations | None = None,
) -> AgentTool[dict[str, Any], EditToolDetails | None]:
    """Build the ``edit`` :class:`AgentTool` rooted at ``cwd``."""
    ops = operations or _default_operations()
    read_file = ops.read_file or _default_read_file
    write_file = ops.write_file or _default_write_file
    access = ops.access or _default_access

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[EditToolDetails | None]:
        prepared = _prepare_edit_arguments(params)
        path, edits = _validate_edit_input(prepared)
        absolute_path = resolve_to_cwd(path, cwd)

        try:
            await access(absolute_path)
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise ValueError(f"File not found: {path}") from exc

        buffer = await read_file(absolute_path)
        raw_content = buffer.decode("utf-8")
        bom, content = strip_bom(raw_content)
        original_ending = detect_line_ending(content)
        normalized_content = normalize_to_lf(content)
        applied = apply_edits_to_normalized_content(normalized_content, edits, path)

        final_content = bom + restore_line_endings(applied.new_content, original_ending)
        await write_file(absolute_path, final_content)

        diff_result = generate_diff_string(applied.base_content, applied.new_content)
        return AgentToolResult(
            content=[
                TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}."),
            ],
            details=EditToolDetails(
                diff=diff_result.diff,
                first_changed_line=diff_result.first_changed_line,
            ),
        )

    return AgentTool[dict[str, Any], EditToolDetails | None](
        name="edit",
        description=_EDIT_DESCRIPTION,
        parameters=_EDIT_PARAMETERS,
        label="edit",
        execute=execute,
        prepare_arguments=_prepare_edit_arguments,
    )


__all__ = ["EditOperations", "EditToolDetails", "create_edit_tool"]
