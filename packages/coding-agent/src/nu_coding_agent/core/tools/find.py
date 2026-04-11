"""``find`` tool — file search by glob pattern.

Direct port of ``packages/coding-agent/src/core/tools/find.ts``. The TS
version shells out to ``fd`` and downloads it on demand if not present;
the Python port uses ``pathlib.Path.rglob`` plus a small ``pathspec``
helper to honour ``.gitignore`` semantics, so no external binary is
required.

Behaviour parity:

* Glob patterns work via ``pathlib.Path.glob``/``rglob``.
* ``.gitignore`` files at the search root and one level deep are merged
  into a :class:`pathspec.PathSpec` and used to filter results.
* ``node_modules`` and ``.git`` are always excluded.
* Results are returned as POSIX-style relative paths.
* Output is capped at a result limit and the standard byte limit.
"""

from __future__ import annotations

import asyncio
import contextlib
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


_DEFAULT_LIMIT = 1000
_ALWAYS_IGNORE = ("node_modules", ".git")


@dataclass(slots=True)
class FindToolDetails:
    truncation: TruncationResult | None = None
    result_limit_reached: int | None = None


@dataclass(slots=True)
class FindOperations:
    """Pluggable I/O hook for ``find``.

    The default implementation walks the filesystem with ``pathlib`` and
    parses ``.gitignore`` files via ``pathspec``. Override ``glob`` to
    delegate the search to a remote backend.
    """

    exists: Callable[[str], Awaitable[bool]] | None = None
    glob: Callable[..., Awaitable[list[str]]] | None = None


# ---------------------------------------------------------------------------
# Default operations
# ---------------------------------------------------------------------------


def _exists_sync(path: str) -> bool:
    return Path(path).exists()


async def _default_exists(path: str) -> bool:
    return await asyncio.to_thread(_exists_sync, path)


def _load_gitignore_spec(search_path: Path) -> Any | None:
    """Build a :class:`pathspec.PathSpec` from gitignore files in ``search_path``."""
    try:
        import pathspec  # noqa: PLC0415 — optional, lazy import
    except ImportError:
        return None

    patterns: list[str] = []
    # Always-ignore directories.
    patterns.extend(f"{ignore}/" for ignore in _ALWAYS_IGNORE)

    root_gitignore = search_path / ".gitignore"
    if root_gitignore.exists():
        with contextlib.suppress(OSError):
            patterns.extend(root_gitignore.read_text(encoding="utf-8").splitlines())

    # ``gitignore`` is the modern pattern name; ``gitwildmatch`` is deprecated.
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def load_gitignore_spec(search_path: Path) -> Any | None:
    """Public alias for :func:`_load_gitignore_spec`.

    Re-exported under a non-underscore name so the grep tool (which lives
    in a sibling module) can reuse the parsing logic without tripping
    pyright's ``reportPrivateUsage`` rule.
    """
    return _load_gitignore_spec(search_path)


def _glob_sync(
    pattern: str,
    search_path_str: str,
    limit: int,
) -> list[str]:
    """Run a synchronous glob and return POSIX-style relative paths."""
    search_path = Path(search_path_str)
    if not search_path.exists():
        raise FileNotFoundError(f"Path not found: {search_path_str}")

    spec = _load_gitignore_spec(search_path)

    matches: list[str] = []
    # Use glob if the pattern is a single segment, rglob if it contains globstars
    # or path separators. Both return absolute paths.
    iterator = search_path.glob(pattern) if ("**" in pattern or "/" in pattern) else search_path.rglob(pattern)

    for path in iterator:
        try:
            rel = path.relative_to(search_path)
        except ValueError:
            continue
        rel_str = rel.as_posix()
        # Always-skip directories that show up at any depth.
        parts = set(rel.parts)
        if any(ignore in parts for ignore in _ALWAYS_IGNORE):
            continue
        if spec is not None and spec.match_file(rel_str):
            continue
        matches.append(rel_str)
        if len(matches) >= limit:
            break

    matches.sort()
    return matches


async def _default_glob(
    pattern: str,
    search_path: str,
    *,
    limit: int,
) -> list[str]:
    return await asyncio.to_thread(_glob_sync, pattern, search_path, limit)


def _default_operations() -> FindOperations:
    return FindOperations(exists=_default_exists, glob=_default_glob)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


_FIND_PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": ("Glob pattern to match files, e.g. '*.py', '**/*.json', or 'src/**/*.test.py'"),
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (default: current directory)",
        },
        "limit": {
            "type": "number",
            "description": f"Maximum number of results (default: {_DEFAULT_LIMIT})",
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


def _build_description() -> str:
    return (
        f"Search for files by glob pattern. Returns matching file paths relative "
        f"to the search directory. Respects .gitignore. Output is truncated to "
        f"{_DEFAULT_LIMIT} results or {DEFAULT_MAX_BYTES // 1024}KB (whichever is "
        f"hit first)."
    )


def create_find_tool(
    cwd: str,
    *,
    operations: FindOperations | None = None,
) -> AgentTool[dict[str, Any], FindToolDetails | None]:
    """Build the ``find`` :class:`AgentTool` rooted at ``cwd``."""
    ops = operations or _default_operations()
    exists = ops.exists or _default_exists
    glob = ops.glob or _default_glob

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[FindToolDetails | None]:
        pattern: str = params["pattern"]
        search_dir: str = params.get("path") or "."
        limit: int = params.get("limit") or _DEFAULT_LIMIT

        search_path = resolve_to_cwd(search_dir, cwd)
        if not await exists(search_path):
            raise ValueError(f"Path not found: {search_path}")

        results = await glob(pattern, search_path, limit=limit)

        if not results:
            return AgentToolResult(
                content=[TextContent(text="No files found matching pattern")],
                details=None,
            )

        result_limit_reached = len(results) >= limit
        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, max_lines=2**31 - 1)
        output = truncation.content
        details = FindToolDetails()
        notices: list[str] = []
        if result_limit_reached:
            notices.append(f"{limit} results limit reached. Use limit={limit * 2} for more, or refine pattern")
            details.result_limit_reached = limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        has_details = details.result_limit_reached is not None or details.truncation is not None
        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details if has_details else None,
        )

    return AgentTool[dict[str, Any], FindToolDetails | None](
        name="find",
        description=_build_description(),
        parameters=_FIND_PARAMETERS,
        label="find",
        execute=execute,
        prompt_snippet="Find files by glob pattern (respects .gitignore)",
    )


__all__ = ["FindOperations", "FindToolDetails", "create_find_tool"]
