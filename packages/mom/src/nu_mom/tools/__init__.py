"""Custom tool wrappers for nu-mom.

Each tool delegates execution to a sandbox :class:`~nu_mom.sandbox.Executor`
so that commands run either on the host or inside a Docker container,
transparently.

Exported factory: :func:`create_mom_tools`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nu_mom.tools.attach import create_attach_tool
from nu_mom.tools.bash import create_bash_tool
from nu_mom.tools.edit import create_edit_tool
from nu_mom.tools.read import create_read_tool
from nu_mom.tools.write import create_write_tool

if TYPE_CHECKING:
    from nu_agent_core.types import AgentTool

    from nu_mom.sandbox import Executor

__all__ = [
    "create_attach_tool",
    "create_bash_tool",
    "create_edit_tool",
    "create_mom_tools",
    "create_read_tool",
    "create_write_tool",
]


def create_mom_tools(executor: Executor, runner: Any = None) -> list[AgentTool]:  # type: ignore[type-arg]
    """Return the full set of mom tools wired to *executor*."""
    return [
        create_read_tool(executor),
        create_bash_tool(executor),
        create_edit_tool(executor),
        create_write_tool(executor),
        create_attach_tool(runner),
    ]
