"""Extension runner — slice-1 subset of ``packages/coding-agent/src/core/extensions/runner.ts``.

The TS file is 915 LoC and orchestrates every hook the agent loop
emits, plus tool / command / shortcut state, plus the UI context, plus
the model registry binding. This Python port covers the **runtime
foundation**: keeping a list of loaded extensions, dispatching
lifecycle events to their handlers, capturing errors, and broadcasting
``session_shutdown`` on ``shutdown()``.

Out of scope (covered by follow-up slices, see ``types.py`` for the
full deferred surface):

* The specialised ``emit_tool_call`` / ``emit_tool_result`` /
  ``emit_input`` / ``emit_context`` / ``emit_before_provider_request``
  paths that compose handler results back into the agent loop.
* ``bind_core``: connecting the runtime's action methods to a real
  :class:`AgentSession`.
* ``set_ui_context``: the interactive-mode UI surface.
* Command / shortcut / flag resolution and conflict detection.

Concurrency note: handlers run sequentially in registration order, with
errors swallowed and recorded so a misbehaving extension cannot take
the whole agent down (matches the TS contract).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from nu_coding_agent.core.extensions.types import (
    Extension,
    ExtensionContext,
    ExtensionError,
    ExtensionHandler,
    ExtensionRuntime,
    LifecycleEvent,
)

type ExtensionErrorListener = Callable[[ExtensionError], None]


@runtime_checkable
class _BindableSession(Protocol):
    """Structural shape ``bind_core`` reads from an :class:`AgentSession`.

    Declared as a Protocol so the runner doesn't have to import
    :class:`AgentSession` directly (which would create a cycle —
    AgentSession imports the extensions package, the extensions
    package would then import AgentSession). The real
    :class:`nu_coding_agent.core.agent_session.AgentSession` satisfies
    this Protocol structurally.
    """

    @property
    def session_manager(self) -> Any: ...

    @property
    def agent(self) -> Any: ...

    def set_model(self, model: Any) -> None: ...


class ExtensionRunner:
    """Owns the loaded :class:`Extension` set and dispatches events to handlers.

    Tests construct a runner directly (see :class:`ExtensionRunner.create`).
    Production code will go through :func:`loader.discover_and_load_extensions`
    once that's wired into ``AgentSession``.
    """

    def __init__(
        self,
        extensions: list[Extension],
        runtime: ExtensionRuntime,
        cwd: str,
    ) -> None:
        self._extensions = list(extensions)
        self._runtime = runtime
        self._cwd = cwd
        self._error_listeners: list[ExtensionErrorListener] = []
        self._errors: list[ExtensionError] = []
        self._shutdown_called = False

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        extensions: list[Extension] | None = None,
        runtime: ExtensionRuntime | None = None,
        cwd: str = "",
    ) -> ExtensionRunner:
        """Build a runner from already-loaded extensions (test convenience)."""
        return cls(
            extensions=extensions or [],
            runtime=runtime or ExtensionRuntime(),
            cwd=cwd,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def extensions(self) -> list[Extension]:
        """The list of loaded extensions, in registration order."""
        return list(self._extensions)

    @property
    def runtime(self) -> ExtensionRuntime:
        return self._runtime

    @property
    def cwd(self) -> str:
        return self._cwd

    def get_extension_paths(self) -> list[str]:
        return [ext.path for ext in self._extensions]

    def get_all_registered_tools(self) -> list[Any]:
        """Return every tool registered by every loaded extension.

        Tools are returned in load order, with each extension's tools
        in registration order. Names are not deduplicated — that's the
        caller's job (typically :func:`AgentSession.apply_extension_tools`,
        which lets later extensions override built-in tools by name).
        """
        out: list[Any] = []
        for extension in self._extensions:
            out.extend(extension.tools.values())
        return out

    def has_handlers(self, event_type: str) -> bool:
        """Return ``True`` iff at least one extension handles ``event_type``."""
        return any(event_type in ext.handlers for ext in self._extensions)

    # ------------------------------------------------------------------
    # bind_core — wires runtime action methods to a real AgentSession
    # ------------------------------------------------------------------

    def bind_core(self, session: _BindableSession) -> None:
        """Bind the runtime action slots to a real :class:`AgentSession`.

        Subset of TS ``ExtensionRunner.bindCore``: covers the actions
        that are useful given the slice-3-and-earlier surface
        (``set_label``, ``append_custom_entry``, ``set_session_name``,
        ``get_session_name``, ``get_active_tools`` / ``get_all_tools``
        / ``set_active_tools``, ``set_model``, ``get_thinking_level`` /
        ``set_thinking_level``).

        The TS ``bindCore`` also wires ``send_message`` /
        ``send_user_message`` / ``refresh_tools`` / ``get_commands`` /
        provider registration / context-actions; those depend on the
        steering queue, slash commands, and the model registry's
        per-extension state which arrive in follow-up sub-slices.

        After calling this method, extensions can invoke action methods
        from inside event handlers and (eventually) from inside their
        registered tools' execute callbacks.
        """
        runtime = self._runtime
        sm = session.session_manager
        agent = session.agent

        def _set_label(entry_id: str, label: str | None) -> None:
            sm.append_label_change(entry_id, label)

        def _append_custom_entry(custom_type: str, data: Any = None) -> str:
            return sm.append_custom_entry(custom_type, data)

        def _set_session_name(name: str) -> None:
            sm.append_session_info(name)

        def _get_session_name() -> str | None:
            return sm.get_session_name()

        def _get_active_tools() -> list[str]:
            return [getattr(t, "name", "") for t in agent.state.tools if getattr(t, "name", None)]

        def _get_all_tools() -> list[dict[str, Any]]:
            return [
                {
                    "name": getattr(t, "name", ""),
                    "description": getattr(t, "description", ""),
                    "parameters": getattr(t, "parameters", {}),
                }
                for t in agent.state.tools
                if getattr(t, "name", None)
            ]

        def _set_active_tools(tool_names: list[str]) -> None:
            keep = set(tool_names)
            agent.state.tools = [t for t in agent.state.tools if getattr(t, "name", None) in keep]

        async def _set_model(model: Any) -> bool:
            session.set_model(model)
            return True

        # Thinking level lives on the agent state when present, but
        # the simplified Python AgentSession doesn't track it directly.
        # We project it through the session manager's most recent
        # ``thinking_level_change`` entry — that's where
        # ``set_thinking_level`` lands it, so the round-trip is
        # consistent.
        def _get_thinking_level() -> str:
            for entry in reversed(sm.get_entries()):
                if entry.get("type") == "thinking_level_change":
                    return str(entry.get("thinkingLevel", "off"))
            return "off"

        def _set_thinking_level(level: str) -> None:
            sm.append_thinking_level_change(level)

        runtime.set_label = _set_label
        runtime.append_custom_entry = _append_custom_entry
        runtime.set_session_name = _set_session_name
        runtime.get_session_name = _get_session_name
        runtime.get_active_tools = _get_active_tools
        runtime.get_all_tools = _get_all_tools
        runtime.set_active_tools = _set_active_tools
        runtime.set_model = _set_model
        runtime.get_thinking_level = _get_thinking_level
        runtime.set_thinking_level = _set_thinking_level

    def create_context(self, extension_path: str = "<unknown>") -> ExtensionContext:
        """Build a per-call :class:`ExtensionContext` for handler / tool execution."""
        return ExtensionContext(
            cwd=self._cwd,
            session_id=None,
            extension_path=extension_path,
        )

    # ------------------------------------------------------------------
    # Error tracking
    # ------------------------------------------------------------------

    def on_error(self, listener: ExtensionErrorListener) -> Callable[[], None]:
        """Register an error listener. Returns an unsubscribe callable."""
        self._error_listeners.append(listener)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._error_listeners.remove(listener)

        return unsubscribe

    def emit_error(self, error: ExtensionError) -> None:
        """Record an error and notify any registered listeners."""
        self._errors.append(error)
        for listener in list(self._error_listeners):
            # Swallow listener exceptions — a broken UI subscriber shouldn't
            # compound a misbehaving extension.
            with contextlib.suppress(Exception):
                listener(error)

    def drain_errors(self) -> list[ExtensionError]:
        """Snapshot + clear the captured error list (used by tests / UI)."""
        out = list(self._errors)
        self._errors.clear()
        return out

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def emit(self, event: LifecycleEvent | dict[str, Any]) -> None:
        """Dispatch a lifecycle event to every matching handler.

        Handlers run sequentially per extension, in registration order;
        the implicit guarantee is that an earlier extension's handler
        for a given event has finished before the next extension's
        handler starts. This matches the TS runner's "sequential
        per-event" semantics and is the simplest model that lets
        extensions reason about ordering.

        Handler exceptions are caught, recorded as
        :class:`ExtensionError` instances, and forwarded to error
        listeners. The dispatch loop continues to the next handler so
        a single broken extension can't silently take the rest down.
        """
        event_type = _event_type(event)
        if not event_type:
            return

        for extension in self._extensions:
            handlers = extension.handlers.get(event_type)
            if not handlers:
                continue
            ctx = self.create_context(extension.path)
            for handler in list(handlers):
                await self._invoke_handler(extension, event_type, handler, event, ctx)

    async def _invoke_handler(
        self,
        extension: Extension,
        event_type: str,
        handler: ExtensionHandler,
        event: LifecycleEvent | dict[str, Any],
        ctx: ExtensionContext,
    ) -> None:
        try:
            result = handler(event, ctx)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            stack = traceback.format_exc()
            self.emit_error(
                ExtensionError(
                    extension_path=extension.path,
                    event=event_type,
                    error=str(exc) or exc.__class__.__name__,
                    stack=stack,
                )
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Emit ``session_shutdown`` to every interested extension and detach."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self.has_handlers("session_shutdown"):
            from nu_coding_agent.core.extensions.types import (  # noqa: PLC0415
                SessionShutdownEvent,
            )

            await self.emit(SessionShutdownEvent())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_type(event: LifecycleEvent | dict[str, Any]) -> str:
    """Pull the discriminator off either a dataclass event or a raw dict."""
    if isinstance(event, dict):
        candidate = event.get("type")
        return str(candidate) if candidate else ""
    return getattr(event, "type", "") or ""


# ---------------------------------------------------------------------------
# Public helper for the slim "session shutdown" pattern used by AgentSession
# ---------------------------------------------------------------------------


async def emit_session_shutdown_event(runner: ExtensionRunner | None) -> bool:
    """Mirror of TS ``emitSessionShutdownEvent``: nudge the runner to broadcast.

    Returns ``True`` if a shutdown event was emitted, ``False`` if there
    was no runner or no handler. Used by ``AgentSession.close`` once the
    AgentSession integration lands.
    """
    if runner is None:
        return False
    if not runner.has_handlers("session_shutdown"):
        return False
    await runner.shutdown()
    return True


# ---------------------------------------------------------------------------
# Compatibility re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "ExtensionErrorListener",
    "ExtensionRunner",
    "emit_session_shutdown_event",
]


# Keep ``asyncio`` referenced — it's used by tests that schedule the
# runner against a real loop and helps pyright keep the import alive.
_ = asyncio  # pragma: no cover
_ = Awaitable
