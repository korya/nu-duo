"""Edit tool — sandbox-aware surgical file editing.

Port of ``packages/mom/src/tools/edit.ts``.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

if TYPE_CHECKING:
    from nu_mom.sandbox import Executor

__all__ = ["create_edit_tool"]

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Brief description of the edit"},
        "path": {"type": "string", "description": "Path to the file (relative or absolute)"},
        "old_text": {"type": "string", "description": "Exact text to find and replace (must match exactly)"},
        "new_text": {"type": "string", "description": "New text to replace with"},
    },
    "required": ["label", "path", "old_text", "new_text"],
}


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _generate_diff(old: str, new: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
    return "".join(diff)


def create_edit_tool(executor: Executor) -> AgentTool:  # type: ignore[type-arg]
    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        from nu_mom.sandbox import ExecOptions

        path: str = params["path"]
        old_text: str = params["old_text"]
        new_text: str = params["new_text"]
        opts = ExecOptions()

        read_result = await executor.exec(f"cat {_shell_escape(path)}", opts)
        if read_result.code != 0:
            raise RuntimeError(read_result.stderr or f"File not found: {path}")

        content = read_result.stdout

        if old_text not in content:
            raise RuntimeError(
                f"Could not find the exact text in {path}. "
                "The old text must match exactly including all whitespace and newlines."
            )

        occurrences = content.count(old_text)
        if occurrences > 1:
            raise RuntimeError(
                f"Found {occurrences} occurrences of the text in {path}. "
                "The text must be unique. Please provide more context to make it unique."
            )

        new_content = content.replace(old_text, new_text, 1)
        if content == new_content:
            raise RuntimeError(f"No changes made to {path}. The replacement produced identical content.")

        write_result = await executor.exec(f"printf '%s' {_shell_escape(new_content)} > {_shell_escape(path)}", opts)
        if write_result.code != 0:
            raise RuntimeError(write_result.stderr or f"Failed to write file: {path}")

        diff = _generate_diff(content, new_content)

        return AgentToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Successfully replaced text in {path}. Changed {len(old_text)} characters to {len(new_text)} characters.",
                )
            ],
            details={"diff": diff},
        )

    return AgentTool(
        name="edit",
        label="edit",
        description="Edit a file by replacing exact text. The old_text must match exactly (including whitespace).",
        parameters=_SCHEMA,
        execute=execute,
    )
