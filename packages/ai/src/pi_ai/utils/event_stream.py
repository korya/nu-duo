"""Async event stream primitives.

Direct port of ``packages/ai/src/utils/event-stream.ts`` — a generic
producer/consumer queue used by every provider streaming implementation.

Differences from the TS version:

* TypeScript ``AsyncIterable<T>`` → Python ``AsyncIterator[T]`` (the class
  implements ``__aiter__`` / ``__anext__`` directly so instances can be used
  in ``async for``).
* The TS ``finalResultPromise`` pattern is replaced by an ``asyncio.Future``
  exposed via :meth:`result`; semantically identical.
* ``push`` after completion is silently ignored — matches the TS behaviour
  where ``if (this.done) return;`` guards the method.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    DoneEvent,
    ErrorEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class EventStream[T, R]:
    """Generic async event queue with a single completing-event contract.

    Usage::

        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e < 0,
            extract_result=lambda e: e,
        )
        stream.push(1)
        stream.push(-1)
        async for event in stream:
            ...
        final = await stream.result()
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ) -> None:
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._queue: list[T] = []
        self._waiting: list[asyncio.Future[tuple[T, bool]]] = []
        self._done = False
        # Loop-agnostic completion signal. The Event is created lazily on
        # first async access so constructing an EventStream doesn't require
        # a running event loop.
        self._final_event: asyncio.Event | None = None
        self._final_value: R | None = None
        self._has_final = False

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def _set_final(self, value: R) -> None:
        if self._has_final:
            return
        self._final_value = value
        self._has_final = True
        if self._final_event is not None:
            self._final_event.set()

    def push(self, event: T) -> None:
        """Emit ``event``. Ignored if the stream has already completed."""
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            self._set_final(self._extract_result(event))

        # Deliver to a waiting consumer, otherwise buffer.
        while self._waiting:
            waiter = self._waiting.pop(0)
            if not waiter.done():
                waiter.set_result((event, False))
                return
        self._queue.append(event)

    def end(self, result: R | None = None) -> None:
        """Terminate the stream.

        If ``result`` is provided it resolves :meth:`result` (unless a
        completing event has already done so). Any waiting consumers receive
        an end-of-stream signal.
        """
        self._done = True
        if result is not None:
            self._set_final(result)
        while self._waiting:
            waiter = self._waiting.pop(0)
            if not waiter.done():
                # Sentinel pair: event is unused when done=True.
                waiter.set_result((None, True))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def __aiter__(self) -> EventStream[T, R]:
        return self

    async def __anext__(self) -> T:
        while True:
            if self._queue:
                return self._queue.pop(0)
            if self._done:
                raise StopAsyncIteration
            waiter: asyncio.Future[tuple[T, bool]] = asyncio.get_event_loop().create_future()
            self._waiting.append(waiter)
            event, is_done = await waiter
            if is_done:
                raise StopAsyncIteration
            return event

    async def result(self) -> R:
        """Await the final result extracted from the completing event."""
        if self._has_final:
            return self._final_value  # type: ignore[return-value]
        if self._final_event is None:
            self._final_event = asyncio.Event()
            if self._has_final:
                self._final_event.set()
        await self._final_event.wait()
        return self._final_value  # type: ignore[return-value]


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Specialised stream for LLM responses.

    Treats :class:`pi_ai.types.DoneEvent` and :class:`pi_ai.types.ErrorEvent`
    as completing events and resolves :meth:`result` with the final
    :class:`pi_ai.types.AssistantMessage`.
    """

    def __init__(self) -> None:
        super().__init__(
            is_complete=lambda e: e.type in {"done", "error"},
            extract_result=self._extract,
        )

    @staticmethod
    def _extract(event: AssistantMessageEvent) -> AssistantMessage:
        if isinstance(event, DoneEvent):
            return event.message
        if isinstance(event, ErrorEvent):
            return event.error
        raise RuntimeError("Unexpected event type for final result")


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory for :class:`AssistantMessageEventStream` (for use in extensions)."""
    return AssistantMessageEventStream()


__all__ = [
    "AssistantMessageEventStream",
    "EventStream",
    "create_assistant_message_event_stream",
]
