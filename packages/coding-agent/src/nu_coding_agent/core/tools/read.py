"""``read`` tool — file reader with offset/limit/truncation.

Direct port of ``packages/coding-agent/src/core/tools/read.ts``. The TS
version supports image files (with optional auto-resize via the photon
WASM library); the Python port handles text reads end-to-end and exposes
an extension hook for image MIME detection so an image-aware variant can
be plugged in later (Pillow-based) without changing the call site.

The :class:`nu_agent_core.AgentTool` produced by :func:`create_read_tool`
plugs straight into the agent loop's tool runtime — no UI rendering is
attached, so the tool runs identically in interactive, print, and
RPC modes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

from nu_coding_agent.core.tools.path_utils import resolve_read_path
from nu_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nu_ai.types import ImageContent


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReadToolDetails:
    """Optional details payload returned alongside the read result."""

    truncation: TruncationResult | None = None


@dataclass(slots=True)
class ReadOperations:
    """Pluggable I/O hooks for ``read``.

    Override these to delegate file access to a remote backend (e.g. SSH).
    Defaults: local filesystem via :class:`pathlib.Path`.
    """

    read_file: Callable[[str], Awaitable[bytes]] | None = None
    access: Callable[[str], Awaitable[None]] | None = None
    detect_image_mime_type: Callable[[str], Awaitable[str | None]] | None = None


# ---------------------------------------------------------------------------
# Default operations
# ---------------------------------------------------------------------------


async def _default_read_file(absolute_path: str) -> bytes:
    return await asyncio.to_thread(Path(absolute_path).read_bytes)


def _check_readable_sync(absolute_path: str) -> None:
    p = Path(absolute_path)
    if not p.exists():
        raise FileNotFoundError(absolute_path)
    if not p.is_file():
        raise IsADirectoryError(absolute_path)


async def _default_access(absolute_path: str) -> None:
    """Raise if the file isn't readable.

    Mirrors ``fs.access(path, R_OK)`` from the upstream — the existence of
    the path and the read permission are validated together. Read-permission
    check happens implicitly on read; no portable stdlib API distinguishes
    "denied" from "missing", so we leave that to the read.
    """
    await asyncio.to_thread(_check_readable_sync, absolute_path)


def _default_operations() -> ReadOperations:
    return ReadOperations(
        read_file=_default_read_file,
        access=_default_access,
        detect_image_mime_type=None,  # Image support deferred to interactive mode.
    )


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


_READ_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read (relative or absolute)",
        },
        "offset": {
            "type": "number",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {
            "type": "number",
            "description": "Maximum number of lines to read",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


def _build_description() -> str:
    return (
        f"Read the contents of a file. Supports text files (image support is provided "
        f"via an optional extension hook). Output is truncated to {DEFAULT_MAX_LINES} "
        f"lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use "
        f"offset/limit for large files. When you need the full file, continue with "
        f"offset until complete."
    )


def create_read_tool(
    cwd: str,
    *,
    operations: ReadOperations | None = None,
) -> AgentTool[dict[str, Any], ReadToolDetails | None]:
    """Build the ``read`` :class:`AgentTool` rooted at ``cwd``."""
    ops = operations or _default_operations()
    read_file = ops.read_file or _default_read_file
    access = ops.access or _default_access
    detect_image_mime_type = ops.detect_image_mime_type

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[ReadToolDetails | None]:
        path: str = params["path"]
        offset: int | None = params.get("offset")
        limit: int | None = params.get("limit")
        absolute_path = resolve_read_path(path, cwd)

        await access(absolute_path)

        mime_type: str | None = None
        if detect_image_mime_type is not None:
            mime_type = await detect_image_mime_type(absolute_path)

        if mime_type is not None:
            buffer = await read_file(absolute_path)
            import base64  # noqa: PLC0415 — only needed in the rare image branch

            data = base64.b64encode(buffer).decode("ascii")
            content: list[TextContent | ImageContent] = [
                TextContent(text=f"Read image file [{mime_type}]"),
            ]
            from nu_ai.types import ImageContent as _ImageContent  # noqa: PLC0415

            content.append(_ImageContent(data=data, mime_type=mime_type))
            return AgentToolResult(content=content, details=None)

        buffer = await read_file(absolute_path)
        text_content = buffer.decode("utf-8")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, (offset - 1) if offset else 0)
        start_line_display = start_line + 1

        if start_line >= len(all_lines):
            raise ValueError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")

        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected_content = "\n".join(all_lines[start_line:end_line])
            user_limited_lines: int | None = end_line - start_line
        else:
            selected_content = "\n".join(all_lines[start_line:])
            user_limited_lines = None

        truncation = truncate_head(selected_content)
        details: ReadToolDetails | None = None

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds "
                f"{format_size(DEFAULT_MAX_BYTES)} limit. Use bash: sed -n "
                f"'{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            details = ReadToolDetails(truncation=truncation)
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                    f"{total_file_lines}. Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                    f"{total_file_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
            details = ReadToolDetails(truncation=truncation)
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = (
                f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )
        else:
            output_text = truncation.content

        return AgentToolResult(
            content=[TextContent(text=output_text)],
            details=details,
        )

    return AgentTool[dict[str, Any], ReadToolDetails | None](
        name="read",
        description=_build_description(),
        parameters=_READ_PARAMETERS,
        label="read",
        execute=execute,
    )


__all__ = ["ReadOperations", "ReadToolDetails", "create_read_tool"]
