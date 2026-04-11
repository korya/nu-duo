"""``write`` tool — atomic file write with mkdir.

Direct port of ``packages/coding-agent/src/core/tools/write.ts`` (logic
only — UI rendering deferred). Creates parent directories on demand and
overwrites existing files.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

from nu_coding_agent.core.tools.file_mutation_queue import with_file_mutation_queue
from nu_coding_agent.core.tools.path_utils import resolve_to_cwd

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(slots=True)
class WriteOperations:
    """Pluggable I/O hooks for ``write``."""

    write_file: Callable[[str, str], Awaitable[None]] | None = None
    mkdir: Callable[[str], Awaitable[None]] | None = None


def _write_text_sync(absolute_path: str, content: str) -> None:
    Path(absolute_path).write_text(content, encoding="utf-8")


def _mkdir_sync(directory: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)


async def _default_write_file(absolute_path: str, content: str) -> None:
    await asyncio.to_thread(_write_text_sync, absolute_path, content)


async def _default_mkdir(directory: str) -> None:
    await asyncio.to_thread(_mkdir_sync, directory)


def _default_operations() -> WriteOperations:
    return WriteOperations(write_file=_default_write_file, mkdir=_default_mkdir)


_WRITE_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to write (relative or absolute)",
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def create_write_tool(
    cwd: str,
    *,
    operations: WriteOperations | None = None,
) -> AgentTool[dict[str, Any], None]:
    """Build the ``write`` :class:`AgentTool` rooted at ``cwd``."""
    ops = operations or _default_operations()
    write_file = ops.write_file or _default_write_file
    mkdir = ops.mkdir or _default_mkdir

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[None]:
        path: str = params["path"]
        content: str = params["content"]
        absolute_path = resolve_to_cwd(path, cwd)

        async def _do_write() -> AgentToolResult[None]:
            await mkdir(str(Path(absolute_path).parent))
            await write_file(absolute_path, content)
            return AgentToolResult(
                content=[
                    TextContent(
                        text=f"Successfully wrote {len(content.encode('utf-8'))} bytes to {path}",
                    )
                ],
                details=None,
            )

        return await with_file_mutation_queue(absolute_path, _do_write)

    return AgentTool[dict[str, Any], None](
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        parameters=_WRITE_PARAMETERS,
        label="write",
        execute=execute,
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=["Use write only for new files or complete rewrites."],
    )


__all__ = ["WriteOperations", "create_write_tool"]
