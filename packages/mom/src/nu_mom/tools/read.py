"""Read tool — sandbox-aware file reading with truncation.

Port of ``packages/mom/src/tools/read.ts``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

from nu_mom.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
)

if TYPE_CHECKING:
    from nu_mom.sandbox import Executor

__all__ = ["create_read_tool"]

_IMAGE_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Brief description of what you're reading and why"},
        "path": {"type": "string", "description": "Path to the file (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum lines to read"},
    },
    "required": ["label", "path"],
}


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _image_mime(path: str) -> str | None:
    import os

    ext = os.path.splitext(path)[1].lower()
    return _IMAGE_MIME.get(ext)


def create_read_tool(executor: Executor) -> AgentTool:  # type: ignore[type-arg]
    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        from nu_mom.sandbox import ExecOptions

        path: str = params["path"]
        offset: int | None = params.get("offset")
        limit: int | None = params.get("limit")
        opts = ExecOptions()

        mime = _image_mime(path)
        if mime:
            result = await executor.exec(f"base64 < {_shell_escape(path)}", opts)
            if result.code != 0:
                raise RuntimeError(result.stderr or f"Failed to read file: {path}")
            b64 = result.stdout.replace("\n", "").replace("\r", "")
            return AgentToolResult(
                content=[
                    TextContent(type="text", text=f"Read image file [{mime}]"),
                    {"type": "image", "data": b64, "mimeType": mime},  # type: ignore[dict-item]
                ],
                details=None,
            )

        # Get total line count
        count_result = await executor.exec(f"wc -l < {_shell_escape(path)}", opts)
        if count_result.code != 0:
            raise RuntimeError(count_result.stderr or f"Failed to read file: {path}")
        total_file_lines = int(count_result.stdout.strip()) + 1

        start_line = max(1, offset) if offset else 1
        if start_line > total_file_lines:
            raise RuntimeError(f"Offset {offset} is beyond end of file ({total_file_lines} lines total)")

        cmd = f"cat {_shell_escape(path)}" if start_line == 1 else f"tail -n +{start_line} {_shell_escape(path)}"

        result = await executor.exec(cmd, opts)
        if result.code != 0:
            raise RuntimeError(result.stderr or f"Failed to read file: {path}")

        selected = result.stdout
        user_limited_lines: int | None = None

        if limit is not None:
            lines = selected.split("\n")
            end_idx = min(limit, len(lines))
            selected = "\n".join(lines[:end_idx])
            user_limited_lines = end_idx

        truncation = truncate_head(selected)

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len((selected.split("\n") or [""])[0].encode("utf-8")))
            output_text = (
                f"[Line {start_line} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
        elif truncation.truncated:
            end_line = start_line + truncation.output_lines - 1
            next_offset = end_line + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += f"\n\n[Showing lines {start_line}-{end_line} of {total_file_lines}. Use offset={next_offset} to continue]"
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue]"
                )
        elif user_limited_lines is not None:
            lines_from_start = start_line - 1 + user_limited_lines
            if lines_from_start < total_file_lines:
                remaining = total_file_lines - lines_from_start
                next_offset = start_line + user_limited_lines
                output_text = (
                    truncation.content + f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue]"
                )
            else:
                output_text = truncation.content
        else:
            output_text = truncation.content

        return AgentToolResult(
            content=[TextContent(type="text", text=output_text)],
            details={"truncation": truncation},
        )

    return AgentTool(
        name="read",
        label="read",
        description=(
            f"Read a file. Images are sent as attachments. Text output truncated to "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. Use offset/limit for large files."
        ),
        parameters=_SCHEMA,
        execute=execute,
    )
