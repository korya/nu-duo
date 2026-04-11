"""Channel-based pub/sub event bus — direct port of ``packages/coding-agent/src/core/event-bus.ts``.

The upstream version wraps Node's :class:`EventEmitter`. Python's stdlib
has no equivalent, so we implement a minimal dict-based fan-out with the
same public surface (``emit``/``on`` returning an unsubscribe callable,
plus ``clear``). Handlers may be sync or async; async handlers are
scheduled on the running loop and any exception is logged to ``stderr``
without propagating to the publisher (mirrors the upstream's
``safeHandler``).
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    type Handler = Callable[[Any], Awaitable[None] | None]


class EventBus(Protocol):
    """Public emit/on surface returned to subscribers."""

    def emit(self, channel: str, data: Any) -> None: ...
    def on(self, channel: str, handler: Handler) -> Callable[[], None]: ...


class EventBusController(EventBus, Protocol):
    """Privileged surface that also lets the owner ``clear()`` all handlers."""

    def clear(self) -> None: ...


@dataclass(slots=True)
class _EventBusImpl:
    _handlers: dict[str, list[Handler]] = field(default_factory=dict)
    _tasks: set[asyncio.Task[None]] = field(default_factory=set)

    def emit(self, channel: str, data: Any) -> None:
        for handler in list(self._handlers.get(channel, ())):
            self._dispatch(channel, handler, data)

    def clear(self) -> None:
        self._handlers.clear()

    def on(self, channel: str, handler: Handler) -> Callable[[], None]:
        self._handlers.setdefault(channel, []).append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers.get(channel)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                return
            if not handlers:
                self._handlers.pop(channel, None)

        return unsubscribe

    def _dispatch(self, channel: str, handler: Handler, data: Any) -> None:
        try:
            result = handler(data)
        except Exception:
            print(f"Event handler error ({channel}):", file=sys.stderr)
            traceback.print_exc()
            return
        if result is None:
            return

        async def _run() -> None:
            try:
                await result  # type: ignore[misc]
            except Exception:
                print(f"Event handler error ({channel}):", file=sys.stderr)
                traceback.print_exc()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_run())
        else:
            task = loop.create_task(_run())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)


def create_event_bus() -> EventBusController:
    """Build a fresh :class:`EventBusController`."""
    return _EventBusImpl()


__all__ = ["EventBus", "EventBusController", "create_event_bus"]
