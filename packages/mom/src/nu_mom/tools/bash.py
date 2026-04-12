"""Bash tool — sandbox-aware shell command execution.

Port of ``packages/mom/src/tools/bash.ts``.
"""

from __future__ import annotations

import secrets
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

from nu_mom.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_tail,
)

if TYPE_CHECKING:
    from nu_mom.sandbox import Executor

__all__ = ["create_bash_tool"]

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Brief description of what this command does (shown to user)"},
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
    },
    "required": ["label", "command"],
}


def _temp_path() -> str:
    return str(Path(tempfile.gettempdir()) / f"mom-bash-{secrets.token_hex(8)}.log")


def create_bash_tool(executor: Executor) -> AgentTool:  # type: ignore[type-arg]
    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        from nu_mom.sandbox import ExecOptions

        command: str = params["command"]
        timeout: float | None = params.get("timeout")

        result = await executor.exec(command, ExecOptions(timeout=timeout))
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        temp_file_path: str | None = None
        if len(output.encode("utf-8")) > DEFAULT_MAX_BYTES:
            temp_file_path = _temp_path()
            Path(temp_file_path).write_text(output, encoding="utf-8")

        truncation = truncate_tail(output)
        output_text = truncation.content or "(no output)"

        if truncation.truncated:
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                last_line_size = format_size(len((output.split("\n") or [""])[-1].encode("utf-8")))
                output_text += f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line} (line is {last_line_size}). Full output: {temp_file_path}]"
            elif truncation.truncated_by == "lines":
                output_text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. Full output: {temp_file_path}]"
            else:
                output_text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp_file_path}]"

        if result.code != 0:
            raise RuntimeError(f"{output_text}\n\nCommand exited with code {result.code}".strip())

        return AgentToolResult(
            content=[TextContent(type="text", text=output_text)],
            details={"truncation": truncation, "full_output_path": temp_file_path},
        )

    return AgentTool(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command. Output is truncated to last {DEFAULT_MAX_LINES} lines "
            f"or {DEFAULT_MAX_BYTES // 1024}KB. If truncated, full output is saved to a temp file."
        ),
        parameters=_SCHEMA,
        execute=execute,
    )
