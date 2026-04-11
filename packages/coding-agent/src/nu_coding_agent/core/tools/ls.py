"""``ls`` tool — directory listing.

Direct port of ``packages/coding-agent/src/core/tools/ls.ts``. Lists
directory contents sorted alphabetically (case-insensitive), suffixes
directories with ``/``, includes dotfiles. Output is capped to a
configurable entry count and the standard byte limit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

from nu_coding_agent.core.tools.path_utils import resolve_to_cwd
from nu_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    TruncationResult,
    format_size,
    truncate_head,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


_DEFAULT_LIMIT = 500


@dataclass(slots=True)
class LsToolDetails:
    truncation: TruncationResult | None = None
    entry_limit_reached: int | None = None


@dataclass(slots=True)
class LsOperations:
    """Pluggable I/O hooks for ``ls``."""

    exists: Callable[[str], Awaitable[bool]] | None = None
    is_directory: Callable[[str], Awaitable[bool]] | None = None
    readdir: Callable[[str], Awaitable[list[str]]] | None = None
    is_dir_entry: Callable[[str], Awaitable[bool]] | None = None


def _exists_sync(path: str) -> bool:
    return Path(path).exists()


def _is_directory_sync(path: str) -> bool:
    return Path(path).is_dir()


def _readdir_sync(path: str) -> list[str]:
    return [entry.name for entry in Path(path).iterdir()]


async def _default_exists(path: str) -> bool:
    return await asyncio.to_thread(_exists_sync, path)


async def _default_is_directory(path: str) -> bool:
    return await asyncio.to_thread(_is_directory_sync, path)


async def _default_readdir(path: str) -> list[str]:
    return await asyncio.to_thread(_readdir_sync, path)


async def _default_is_dir_entry(path: str) -> bool:
    return await asyncio.to_thread(_is_directory_sync, path)


def _default_operations() -> LsOperations:
    return LsOperations(
        exists=_default_exists,
        is_directory=_default_is_directory,
        readdir=_default_readdir,
        is_dir_entry=_default_is_dir_entry,
    )


_LS_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to list (default: current directory)",
        },
        "limit": {
            "type": "number",
            "description": f"Maximum number of entries to return (default: {_DEFAULT_LIMIT})",
        },
    },
    "additionalProperties": False,
}


def _build_description() -> str:
    return (
        f"List directory contents. Returns entries sorted alphabetically, with '/' "
        f"suffix for directories. Includes dotfiles. Output is truncated to "
        f"{_DEFAULT_LIMIT} entries or {DEFAULT_MAX_BYTES // 1024}KB (whichever is "
        f"hit first)."
    )


def create_ls_tool(
    cwd: str,
    *,
    operations: LsOperations | None = None,
) -> AgentTool[dict[str, Any], LsToolDetails | None]:
    """Build the ``ls`` :class:`AgentTool` rooted at ``cwd``."""
    ops = operations or _default_operations()
    exists = ops.exists or _default_exists
    is_directory = ops.is_directory or _default_is_directory
    readdir = ops.readdir or _default_readdir
    is_dir_entry = ops.is_dir_entry or _default_is_dir_entry

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[LsToolDetails | None]:
        path: str = params.get("path") or "."
        limit: int = params.get("limit") or _DEFAULT_LIMIT
        dir_path = resolve_to_cwd(path, cwd)

        if not await exists(dir_path):
            raise ValueError(f"Path not found: {dir_path}")
        if not await is_directory(dir_path):
            raise ValueError(f"Not a directory: {dir_path}")

        entries = await readdir(dir_path)
        entries.sort(key=lambda s: s.lower())

        results: list[str] = []
        entry_limit_reached = False
        for entry in entries:
            if len(results) >= limit:
                entry_limit_reached = True
                break
            full_path = str(Path(dir_path) / entry)
            try:
                is_dir = await is_dir_entry(full_path)
            except OSError:
                continue
            results.append(f"{entry}/" if is_dir else entry)

        if not results:
            return AgentToolResult(
                content=[TextContent(text="(empty directory)")],
                details=None,
            )

        raw_output = "\n".join(results)
        # No line limit because the entry count already caps line growth.
        truncation = truncate_head(raw_output, max_lines=2**31 - 1)
        output = truncation.content
        details = LsToolDetails()
        notices: list[str] = []
        if entry_limit_reached:
            notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
            details.entry_limit_reached = limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        has_details = details.entry_limit_reached is not None or details.truncation is not None
        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details if has_details else None,
        )

    return AgentTool[dict[str, Any], LsToolDetails | None](
        name="ls",
        description=_build_description(),
        parameters=_LS_PARAMETERS,
        label="ls",
        execute=execute,
    )


__all__ = ["LsOperations", "LsToolDetails", "create_ls_tool"]
