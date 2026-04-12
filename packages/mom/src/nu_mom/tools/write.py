"""Write tool — sandbox-aware file writing.

Port of ``packages/mom/src/tools/write.ts``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

if TYPE_CHECKING:
    from nu_mom.sandbox import Executor

__all__ = ["create_write_tool"]

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Brief description of what you're writing"},
        "path": {"type": "string", "description": "Path to the file (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["label", "path", "content"],
}


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def create_write_tool(executor: Executor) -> AgentTool:  # type: ignore[type-arg]
    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        from nu_mom.sandbox import ExecOptions

        path: str = params["path"]
        content: str = params["content"]

        # Determine parent directory
        parent = path.rsplit("/", 1)[0] if "/" in path else "."

        cmd = f"mkdir -p {_shell_escape(parent)} && printf '%s' {_shell_escape(content)} > {_shell_escape(path)}"
        result = await executor.exec(cmd, ExecOptions())
        if result.code != 0:
            raise RuntimeError(result.stderr or f"Failed to write file: {path}")

        return AgentToolResult(
            content=[TextContent(type="text", text=f"Successfully wrote {len(content)} bytes to {path}")],
            details=None,
        )

    return AgentTool(
        name="write",
        label="write",
        description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories automatically.",
        parameters=_SCHEMA,
        execute=execute,
    )
