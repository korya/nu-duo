"""``grep`` tool — file content search by regex.

Direct port of ``packages/coding-agent/src/core/tools/grep.ts``. The TS
version shells out to ``ripgrep`` and downloads it on demand; the
Python port uses :mod:`re` plus the same gitignore-aware walker as
:mod:`nu_coding_agent.core.tools.find`, so no external binary is
required.

Behaviour parity:

* ``pattern`` is treated as a regex by default; pass ``literal=True``
  to escape it.
* ``ignore_case`` toggles ``re.IGNORECASE``.
* ``glob`` filters which files are scanned (e.g. ``"*.py"``).
* ``context`` adds N lines before/after each match (matches ``rg -C``).
* Output format: ``relative/path:line_no:line_text``.
* Match limit defaults to 100; long lines truncate to
  :data:`nu_coding_agent.core.tools.truncate.GREP_MAX_LINE_LENGTH` chars.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

from nu_coding_agent.core.tools.find import load_gitignore_spec
from nu_coding_agent.core.tools.path_utils import resolve_to_cwd
from nu_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


_DEFAULT_LIMIT = 100
_ALWAYS_IGNORE = ("node_modules", ".git")


@dataclass(slots=True)
class GrepToolDetails:
    truncation: TruncationResult | None = None
    match_limit_reached: int | None = None
    lines_truncated: bool = False


@dataclass(slots=True)
class GrepOperations:
    """Pluggable I/O hooks for ``grep``."""

    is_directory: Callable[[str], Awaitable[bool]] | None = None
    read_file: Callable[[str], Awaitable[str]] | None = None


def _is_directory_sync(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p.is_dir()


def _read_file_sync(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


async def _default_is_directory(path: str) -> bool:
    return await asyncio.to_thread(_is_directory_sync, path)


async def _default_read_file(path: str) -> str:
    return await asyncio.to_thread(_read_file_sync, path)


def _default_operations() -> GrepOperations:
    return GrepOperations(is_directory=_default_is_directory, read_file=_default_read_file)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


_GREP_PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Search pattern (regex or literal string)",
        },
        "path": {
            "type": "string",
            "description": "Directory or file to search (default: current directory)",
        },
        "glob": {
            "type": "string",
            "description": "Filter files by glob pattern, e.g. '*.py' or '**/*.test.py'",
        },
        "ignoreCase": {
            "type": "boolean",
            "description": "Case-insensitive search (default: false)",
        },
        "literal": {
            "type": "boolean",
            "description": "Treat pattern as literal string instead of regex (default: false)",
        },
        "context": {
            "type": "number",
            "description": "Number of lines to show before and after each match (default: 0)",
        },
        "limit": {
            "type": "number",
            "description": f"Maximum number of matches to return (default: {_DEFAULT_LIMIT})",
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


def _build_description() -> str:
    return (
        f"Search file contents for a pattern. Returns matching lines with file "
        f"paths and line numbers. Respects .gitignore. Output is truncated to "
        f"{_DEFAULT_LIMIT} matches or {DEFAULT_MAX_BYTES // 1024}KB (whichever is "
        f"hit first). Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
    )


def _walk_files(
    search_root: Path,
    *,
    file_glob: str | None,
) -> list[Path]:
    """Walk ``search_root`` and return files matching ``file_glob`` (or all)."""
    spec = load_gitignore_spec(search_root)

    iterator = search_root.rglob(file_glob) if file_glob else search_root.rglob("*")
    out: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(search_root)
        except ValueError:
            continue
        if any(ignore in rel.parts for ignore in _ALWAYS_IGNORE):
            continue
        if spec is not None and spec.match_file(rel.as_posix()):
            continue
        out.append(path)
    out.sort()
    return out


def _build_pattern(pattern: str, *, literal: bool, ignore_case: bool) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(re.escape(pattern) if literal else pattern, flags)


def create_grep_tool(
    cwd: str,
    *,
    operations: GrepOperations | None = None,
) -> AgentTool[dict[str, Any], GrepToolDetails | None]:
    """Build the ``grep`` :class:`AgentTool` rooted at ``cwd``."""
    ops = operations or _default_operations()
    is_directory = ops.is_directory or _default_is_directory
    read_file = ops.read_file or _default_read_file

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[GrepToolDetails | None]:
        pattern_str: str = params["pattern"]
        search_dir: str = params.get("path") or "."
        file_glob: str | None = params.get("glob")
        ignore_case: bool = bool(params.get("ignoreCase"))
        literal: bool = bool(params.get("literal"))
        context_lines: int = max(0, int(params.get("context") or 0))
        limit: int = max(1, int(params.get("limit") or _DEFAULT_LIMIT))

        search_path = resolve_to_cwd(search_dir, cwd)
        try:
            isdir = await is_directory(search_path)
        except FileNotFoundError as exc:
            raise ValueError(f"Path not found: {search_path}") from exc

        try:
            regex = _build_pattern(pattern_str, literal=literal, ignore_case=ignore_case)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc

        # Build the file list. For a single-file search the list is just one entry.
        if isdir:
            files = await asyncio.to_thread(_walk_files, Path(search_path), file_glob=file_glob)
        else:
            files = [Path(search_path)]
        search_root = Path(search_path) if isdir else Path(search_path).parent

        match_lines: list[str] = []
        match_limit_reached = False
        lines_truncated = False
        for file_path in files:
            if match_limit_reached:
                break
            try:
                content = await read_file(str(file_path))
            except (OSError, UnicodeDecodeError):
                continue
            lines = content.split("\n")
            try:
                rel = file_path.relative_to(search_root).as_posix() if isdir else file_path.name
            except ValueError:
                rel = file_path.name
            for idx, line in enumerate(lines):
                if regex.search(line) is None:
                    continue
                if context_lines:
                    start = max(0, idx - context_lines)
                    end = min(len(lines), idx + context_lines + 1)
                    for ctx_idx in range(start, end):
                        truncated, was_truncated = truncate_line(lines[ctx_idx])
                        lines_truncated = lines_truncated or was_truncated
                        sep = ":" if ctx_idx == idx else "-"
                        match_lines.append(f"{rel}{sep}{ctx_idx + 1}{sep}{truncated}")
                    match_lines.append("--")
                else:
                    truncated, was_truncated = truncate_line(line)
                    lines_truncated = lines_truncated or was_truncated
                    match_lines.append(f"{rel}:{idx + 1}:{truncated}")
                if len(match_lines) >= limit:
                    match_limit_reached = True
                    break

        # Drop a trailing context separator if present.
        if match_lines and match_lines[-1] == "--":
            match_lines.pop()

        if not match_lines:
            return AgentToolResult(
                content=[TextContent(text="No matches found")],
                details=None,
            )

        raw_output = "\n".join(match_lines)
        truncation = truncate_head(raw_output, max_lines=2**31 - 1)
        output = truncation.content
        details = GrepToolDetails(lines_truncated=lines_truncated)
        notices: list[str] = []
        if match_limit_reached:
            notices.append(f"{limit} matches limit reached")
            details.match_limit_reached = limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if lines_truncated:
            notices.append(f"some lines truncated to {GREP_MAX_LINE_LENGTH} chars")
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        has_details = (
            details.match_limit_reached is not None or details.truncation is not None or details.lines_truncated
        )
        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details if has_details else None,
        )

    return AgentTool[dict[str, Any], GrepToolDetails | None](
        name="grep",
        description=_build_description(),
        parameters=_GREP_PARAMETERS,
        label="grep",
        execute=execute,
    )


__all__ = ["GrepOperations", "GrepToolDetails", "create_grep_tool"]
