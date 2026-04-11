"""Tool factories exported from the coding-agent's core/tools package.

Each tool is constructed via a ``create_<name>_tool(cwd, ...)`` factory
that returns a fully-typed :class:`nu_agent_core.types.AgentTool`.
``create_all_tools(cwd)`` is the convenience helper used by the CLI to
build the canonical seven-tool set in one call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nu_coding_agent.core.tools.bash import create_bash_tool
from nu_coding_agent.core.tools.edit import create_edit_tool
from nu_coding_agent.core.tools.find import create_find_tool
from nu_coding_agent.core.tools.grep import create_grep_tool
from nu_coding_agent.core.tools.ls import create_ls_tool
from nu_coding_agent.core.tools.read import create_read_tool
from nu_coding_agent.core.tools.write import create_write_tool

# Mirror the upstream ``allTools`` registry. Key order matches the
# default-builder above so ``--tools read,write,edit,bash`` reflects the
# documented "four core tools" emphasis.
ALL_TOOL_NAMES: tuple[str, ...] = (
    "read",
    "write",
    "edit",
    "bash",
    "ls",
    "find",
    "grep",
)

if TYPE_CHECKING:
    from typing import Any

    from nu_agent_core.types import AgentTool


def create_all_tools(cwd: str) -> list[AgentTool[Any, Any]]:
    """Build the canonical seven-tool set rooted at ``cwd``.

    Order matters for prompt-snippet rendering and dispatch fallbacks —
    keep ``read``/``write``/``edit``/``bash`` first to mirror the
    upstream "four core tools" emphasis.
    """
    return [
        create_read_tool(cwd),
        create_write_tool(cwd),
        create_edit_tool(cwd),
        create_bash_tool(cwd),
        create_ls_tool(cwd),
        create_find_tool(cwd),
        create_grep_tool(cwd),
    ]


__all__ = [
    "ALL_TOOL_NAMES",
    "create_all_tools",
    "create_bash_tool",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_tool",
    "create_write_tool",
]
