"""Attach tool — upload files to Slack from the sandbox.

Port of ``packages/mom/src/tools/attach.ts``.  The upload function is
supplied by the :class:`~nu_mom.agent.AgentRunner` that created this tool
instance, so the tool remains stateless beyond that reference.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

if TYPE_CHECKING:
    from nu_mom.agent import AgentRunner

__all__ = ["create_attach_tool"]

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Brief description of what you're sharing"},
        "path": {"type": "string", "description": "Path to the file to attach"},
        "title": {"type": "string", "description": "Title for the file (defaults to filename)"},
    },
    "required": ["label", "path"],
}


def create_attach_tool(runner: AgentRunner | None) -> AgentTool:  # type: ignore[type-arg]
    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        if runner is None or runner._upload_fn is None:
            raise RuntimeError("Upload function not configured")

        path: str = params["path"]
        title: str | None = params.get("title")
        file_name = title or os.path.basename(os.path.realpath(path))

        await runner._upload_fn(path, file_name)

        return AgentToolResult(
            content=[TextContent(type="text", text=f"Attached file: {file_name}")],
            details=None,
        )

    return AgentTool(
        name="attach",
        label="attach",
        description="Attach a file to your response. Only files from the workspace can be attached.",
        parameters=_SCHEMA,
        execute=execute,
    )
