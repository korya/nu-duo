"""Extension surface — slice-1 foundation port.

Re-exports the public symbols of the four submodules. See
``types.py`` for the deferred-surface roadmap; the rest of the
extension API (tool registration, commands, shortcuts, flags,
message renderers, provider injection, AgentSession wiring) lands in
follow-up slices.
"""

from nu_coding_agent.core.extensions.loader import (
    ENTRY_POINT_GROUP,
    discover_and_load_extensions,
    load_extensions_from_factories,
    load_extensions_from_paths,
)
from nu_coding_agent.core.extensions.runner import (
    ExtensionErrorListener,
    ExtensionRunner,
    emit_session_shutdown_event,
)
from nu_coding_agent.core.extensions.types import (
    AgentEndEvent,
    AgentStartEvent,
    Extension,
    ExtensionAPI,
    ExtensionContext,
    ExtensionError,
    ExtensionFactory,
    ExtensionHandler,
    ExtensionRuntime,
    LifecycleEvent,
    LifecycleEventName,
    LoadExtensionsResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
    SessionCompactEvent,
    SessionShutdownEvent,
    SessionStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "AgentEndEvent",
    "AgentStartEvent",
    "Extension",
    "ExtensionAPI",
    "ExtensionContext",
    "ExtensionError",
    "ExtensionErrorListener",
    "ExtensionFactory",
    "ExtensionHandler",
    "ExtensionRunner",
    "ExtensionRuntime",
    "LifecycleEvent",
    "LifecycleEventName",
    "LoadExtensionsResult",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "SessionBeforeCompactEvent",
    "SessionBeforeCompactResult",
    "SessionCompactEvent",
    "SessionShutdownEvent",
    "SessionStartEvent",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
    "discover_and_load_extensions",
    "emit_session_shutdown_event",
    "load_extensions_from_factories",
    "load_extensions_from_paths",
]
