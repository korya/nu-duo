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
from typing import Any

from nu_coding_agent.core.extensions.types import (
    Extension,
    ExtensionContext,
    ExtensionError,
    ExtensionHandler,
    ExtensionRuntime,
    LifecycleEvent,
)

type ExtensionErrorListener = Callable[[ExtensionError], None]


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

    def has_handlers(self, event_type: str) -> bool:
        """Return ``True`` iff at least one extension handles ``event_type``."""
        return any(event_type in ext.handlers for ext in self._extensions)

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
