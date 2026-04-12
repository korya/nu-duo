"""Tool wrappers — Python port of ``packages/coding-agent/src/core/tools/tool-definition-wrapper.ts``.

The TS file splits authoring concerns across two types: ``ToolDefinition``
(extension-author facing — includes optional ``promptSnippet`` /
``promptGuidelines`` / ``renderCall`` / ``renderResult`` plus an
``execute`` callback that receives an :class:`ExtensionContext`) and
``AgentTool`` (agent-loop facing — strictly the runtime contract).
``wrapToolDefinition`` adapts the former to the latter by closing over
the runner's :func:`createContext` factory.

The Python port has already collapsed both types into a single
:class:`nu_agent_core.types.AgentTool` dataclass — the optional
``prompt_snippet`` / ``prompt_guidelines`` fields cover the
extension-author surface, and the agent-loop reads only the runtime
fields. That means the wrapper here is intentionally minimal:

* Extension authors register :class:`AgentTool` instances directly via
  ``api.register_tool(tool)``.
* :func:`wrap_registered_tool` returns the tool unchanged for now —
  there is no separate ``ToolDefinition`` shape to translate from.
  The function exists so future ctx-injection (when extension tools
  need to call action methods on the runtime — see follow-up
  sub-slice 4 for ``bind_core``) has a place to land without
  changing every call site.

In short: this slice unblocks "extensions can ship tools the LLM can
call". Extension tools that need to call back into the runtime (e.g.
``ctx.send_message``) still wait on sub-slice 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nu_agent_core.types import AgentTool

    from nu_coding_agent.core.extensions.runner import ExtensionRunner


def wrap_registered_tool(
    registered_tool: AgentTool[Any, Any],
    runner: ExtensionRunner | None = None,
) -> AgentTool[Any, Any]:
    """Adapt an extension-registered tool for the agent loop.

    Currently a passthrough: the Python port's :class:`AgentTool` is
    already the right shape for the agent loop. ``runner`` is accepted
    for forward compatibility — once sub-slice 4 lands the action
    method runtime, this function will close over ``runner.create_context()``
    and inject the resulting :class:`ExtensionContext` into the tool's
    ``execute`` callback so extension tools can call back into the
    runtime.
    """
    return registered_tool


def wrap_registered_tools(
    registered_tools: list[AgentTool[Any, Any]],
    runner: ExtensionRunner | None = None,
) -> list[AgentTool[Any, Any]]:
    """List form of :func:`wrap_registered_tool`."""
    return [wrap_registered_tool(t, runner) for t in registered_tools]
