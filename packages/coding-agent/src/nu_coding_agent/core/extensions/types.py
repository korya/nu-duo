"""Extension surface types — slice-1 subset of ``packages/coding-agent/src/core/extensions/types.ts``.

The upstream TS file is 1450 LoC and exports 127 symbols spanning every
hook the coding agent emits, plus tool / command / shortcut / flag /
message-renderer registration, plus action handlers (sendMessage,
appendEntry, setLabel, setModel, …). This Python port is the **first**
of several extension slices and intentionally covers only the
foundation:

* The dataclass shape of an :class:`Extension` instance.
* The lifecycle event payloads — agent / message / tool / session +
  the corresponding handler protocol.
* :class:`ExtensionContext` and :class:`ExtensionRuntime` skeletons.
* :class:`ExtensionAPI` as a runtime-checkable :class:`typing.Protocol`
  with the methods that are actually wired in this slice.

Out of scope (will land in follow-up slices):

* ``registerTool`` / tool definition wrapping (depends on the
  tool-definition-wrapper port).
* ``registerCommand`` / ``registerShortcut`` / ``registerFlag`` /
  ``registerMessageRenderer`` (none of these have a consumer in the
  Python port yet — interactive mode and slash-commands are deferred).
* ``registerProvider`` (custom provider injection — needs the model
  registry's per-extension state machine).
* The UI context surface (:class:`ExtensionUIContext`) — interactive
  mode only.
* The specialised ``emitToolCall`` / ``emitToolResult`` / ``emitInput``
  paths that compose handler results back into the agent loop. The
  generic :meth:`ExtensionRunner.emit` is enough for fire-and-forget
  lifecycle events used by tests today.
* Wiring into :class:`AgentSession`. The runner is exercised in
  isolation in this slice.

The naming follows the rest of the port: TS ``camelCase`` becomes
Python ``snake_case`` for fields and method names, class names stay
verbatim, and TS string literal types map to ``Literal[...]``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nu_coding_agent.core.source_info import SourceInfo

# ---------------------------------------------------------------------------
# Event names
# ---------------------------------------------------------------------------

#: Lifecycle events that fire from inside the agent loop / session.
type LifecycleEventName = Literal[
    "session_start",
    "session_shutdown",
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
]

#: All event names known to the runner. Extra string ids will simply
#: route through ``hasHandlers``/``emit`` without dispatching anywhere.
type ExtensionEventName = str


# ---------------------------------------------------------------------------
# Event payload dataclasses — one per lifecycle event
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionStartEvent:
    """Fired once when the SessionManager attaches to an extension runner."""

    type: Literal["session_start"] = "session_start"
    cwd: str = ""
    session_id: str = ""


@dataclass(slots=True)
class SessionShutdownEvent:
    """Fired right before the runner detaches and unloads its extensions."""

    type: Literal["session_shutdown"] = "session_shutdown"


@dataclass(slots=True)
class AgentStartEvent:
    """Fired when ``Agent.prompt`` begins streaming the first turn."""

    type: Literal["agent_start"] = "agent_start"


@dataclass(slots=True)
class AgentEndEvent:
    """Fired after the agent finishes the final turn (success or error)."""

    type: Literal["agent_end"] = "agent_end"
    error: str | None = None


@dataclass(slots=True)
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass(slots=True)
class TurnEndEvent:
    type: Literal["turn_end"] = "turn_end"


@dataclass(slots=True)
class MessageStartEvent:
    type: Literal["message_start"] = "message_start"
    role: str = ""


@dataclass(slots=True)
class MessageUpdateEvent:
    type: Literal["message_update"] = "message_update"
    payload: Any = None


@dataclass(slots=True)
class MessageEndEvent:
    type: Literal["message_end"] = "message_end"
    message: Any = None


@dataclass(slots=True)
class ToolExecutionStartEvent:
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_name: str = ""
    update: Any = None


@dataclass(slots=True)
class ToolExecutionEndEvent:
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_name: str = ""
    is_error: bool = False
    result: Any = None


#: Discriminated union of every lifecycle event payload covered by this
#: slice. Extension handlers and the runner accept the union; downstream
#: code dispatches on ``event.type``.
type LifecycleEvent = (
    SessionStartEvent
    | SessionShutdownEvent
    | AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)


# ---------------------------------------------------------------------------
# Extension data shape + handler signatures
# ---------------------------------------------------------------------------


#: Handler signature: takes the event + context, optionally returns a value.
#: Mirrors TS ``ExtensionHandler<E, R>``. Sync or async are both fine.
type ExtensionHandler = Callable[..., Any | Awaitable[Any]]


@dataclass(slots=True)
class ExtensionError:
    """Captured error from a handler invocation. Mirrors TS ``ExtensionError``."""

    extension_path: str
    event: str
    error: str
    stack: str | None = None


@dataclass(slots=True)
class Extension:
    """Loaded extension instance.

    Mirrors TS ``Extension``. The collections start empty; the extension
    factory mutates them via the :class:`ExtensionAPI` it receives.
    Tools / commands / shortcuts / flags / message-renderers are kept as
    plain dicts in this slice — none of them are consumed yet, but the
    fields exist so a TS-shaped extension factory can call the
    registration methods without exploding.
    """

    path: str
    resolved_path: str
    source_info: SourceInfo
    handlers: dict[str, list[ExtensionHandler]] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    message_renderers: dict[str, Any] = field(default_factory=dict)
    commands: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    shortcuts: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extension runtime / context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExtensionContext:
    """Per-call context handed to handlers and (later) tool executions.

    The TS version splits this across ``ExtensionContext`` and
    ``ExtensionContextActions``; this slice keeps just the data fields
    that are stable. Action methods (``send_message``, ``set_label``,
    etc.) move into a follow-up slice once :class:`AgentSession` knows
    how to expose them.
    """

    cwd: str
    session_id: str | None = None
    extension_path: str = "<unknown>"


@dataclass(slots=True)
class ExtensionRuntime:
    """Shared mutable state held by the runner.

    Mirrors a small subset of the TS ``ExtensionRuntime`` interface —
    just the bits that are useful before the action methods are wired.
    Flag values, registered tools and the error stream all live here so
    follow-up slices can layer the action surface on top without
    breaking the existing call sites.
    """

    flag_values: dict[str, bool | str] = field(default_factory=dict)
    pending_provider_registrations: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ExtensionAPI — what the factory function receives
# ---------------------------------------------------------------------------


@runtime_checkable
class ExtensionAPI(Protocol):
    """Subset of the TS ``ExtensionAPI`` interface ported in this slice.

    Methods that aren't wired yet are still declared on the protocol so
    a TS-shaped factory can call them without crashing — they're
    implemented as no-ops by :func:`create_extension_api` below. Real
    implementations land in follow-up slices.
    """

    def on(self, event: str, handler: ExtensionHandler) -> None: ...

    def register_tool(self, tool: Any) -> None: ...

    def register_command(self, name: str, options: dict[str, Any]) -> None: ...

    def register_shortcut(self, shortcut: str, options: dict[str, Any]) -> None: ...

    def register_flag(self, name: str, options: dict[str, Any]) -> None: ...

    def register_message_renderer(self, custom_type: str, renderer: Any) -> None: ...

    def get_flag(self, name: str) -> bool | str | None: ...


type ExtensionFactory = Callable[[ExtensionAPI], None | Awaitable[None]]


# ---------------------------------------------------------------------------
# Loader result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoadExtensionsResult:
    """Result of :func:`load_extensions_from_factories`."""

    extensions: list[Extension]
    errors: list[dict[str, str]]
    runtime: ExtensionRuntime
