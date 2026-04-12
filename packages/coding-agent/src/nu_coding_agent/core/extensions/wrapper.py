"""Tool / handler wrappers — slice-1 stub.

The TS ``wrapper.ts`` is 27 LoC and exposes ``wrapRegisteredTool`` /
``wrapRegisteredTools``: helpers that adapt an extension-registered
tool definition into an :class:`AgentTool` so the agent loop can
execute it like a built-in. The Python equivalent will land alongside
the tool-definition-wrapper port (a separate slice — extension tools
are useless without something that knows how to convert their Pydantic
parameter models into the JSON Schema the LLM provider expects).

For now this module exists so the import path matches upstream and a
TS-shaped factory that calls ``wrap_registered_tool`` doesn't blow up
the loader; the helpers raise :class:`NotImplementedError` so the
deferred surface is loud.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nu_coding_agent.core.extensions.runner import ExtensionRunner


def wrap_registered_tool(registered_tool: Any, runner: ExtensionRunner) -> Any:
    """Wrap an extension-registered tool into an :class:`AgentTool`.

    Not yet implemented; the tool-definition-wrapper port is a follow-up
    slice. Calling this raises :class:`NotImplementedError` rather than
    silently doing the wrong thing.
    """
    raise NotImplementedError(
        "wrap_registered_tool is not implemented yet — extension tools land "
        "alongside the tool-definition-wrapper port. Tracked under the "
        "extensions follow-up slice."
    )


def wrap_registered_tools(registered_tools: list[Any], runner: ExtensionRunner) -> list[Any]:
    """List form of :func:`wrap_registered_tool`. Same NotImplementedError contract."""
    return [wrap_registered_tool(t, runner) for t in registered_tools]
