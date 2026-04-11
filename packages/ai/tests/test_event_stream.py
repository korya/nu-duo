"""Tests for pi_ai.utils.event_stream.

Ported from ``packages/ai/src/utils/event-stream.ts`` — upstream has no
dedicated test file (EventStream is exercised via provider tests), so these
assertions are new but mirror the documented contract:

* ``push`` delivers to a waiting consumer or queues the event.
* Iteration yields queued events then any new ones, then stops on ``end``.
* ``result()`` resolves with the value extracted from the completing event
  (or the explicit value passed to ``end``).
* ``AssistantMessageEventStream`` treats ``done`` and ``error`` as completing
  events, resolving ``result()`` with the final ``AssistantMessage``.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    Usage,
)
from pi_ai.utils.event_stream import (
    AssistantMessageEventStream,
    EventStream,
    create_assistant_message_event_stream,
)


def _mk_message(stop: str = "stop", error_message: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="hi")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude",
        usage=Usage(
            input=1,
            output=1,
            cache_read=0,
            cache_write=0,
            total_tokens=2,
            cost=Cost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0, total=0.0),
        ),
        stop_reason=stop,  # type: ignore[arg-type]
        error_message=error_message,
        timestamp=1,
    )


# ---------------------------------------------------------------------------
# Generic EventStream
# ---------------------------------------------------------------------------


class TestEventStreamGeneric:
    async def test_push_then_iterate_delivers_queued_events(self) -> None:
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e < 0,
            extract_result=lambda e: e,
        )
        stream.push(1)
        stream.push(2)
        stream.push(-1)  # completing event
        collected: list[int] = []
        async for event in stream:
            collected.append(event)
        assert collected == [1, 2, -1]

    async def test_waiter_is_woken_by_push(self) -> None:
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e < 0,
            extract_result=lambda e: e,
        )

        async def producer() -> None:
            await asyncio.sleep(0.01)
            stream.push(42)
            await asyncio.sleep(0.01)
            stream.push(-1)

        collected: list[int] = []

        async def consumer() -> None:
            async for event in stream:
                collected.append(event)

        await asyncio.gather(producer(), consumer())
        assert collected == [42, -1]

    async def test_end_terminates_iteration_without_completing_event(self) -> None:
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e < 0,
            extract_result=lambda e: e,
        )
        stream.push(1)
        stream.end(99)  # explicit final result
        collected: list[int] = [e async for e in stream]
        assert collected == [1]
        assert await stream.result() == 99

    async def test_result_resolves_when_completing_event_pushed(self) -> None:
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e >= 100,
            extract_result=lambda e: e * 2,
        )
        stream.push(50)
        stream.push(100)
        assert await stream.result() == 200

    async def test_push_after_done_is_ignored(self) -> None:
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e < 0,
            extract_result=lambda e: e,
        )
        stream.push(-1)
        stream.push(99)  # ignored
        collected: list[int] = [e async for e in stream]
        assert collected == [-1]

    async def test_multiple_waiters_are_woken_on_end(self) -> None:
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: False,
            extract_result=lambda e: e,
        )

        async def consume() -> list[int]:
            return [e async for e in stream]

        # Two concurrent iterators share the same queue; when end() fires
        # they should all terminate cleanly without hanging.
        consumer_task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let consumer register its waiter
        stream.end(0)
        result = await asyncio.wait_for(consumer_task, timeout=1.0)
        assert result == []


# ---------------------------------------------------------------------------
# AssistantMessageEventStream
# ---------------------------------------------------------------------------


class TestAssistantMessageEventStream:
    async def test_done_event_resolves_result(self) -> None:
        stream = AssistantMessageEventStream()
        msg = _mk_message()
        stream.push(StartEvent(partial=msg))
        stream.push(DoneEvent(reason="stop", message=msg))
        assert await stream.result() == msg

    async def test_error_event_resolves_result(self) -> None:
        stream = AssistantMessageEventStream()
        error_msg = _mk_message(stop="error", error_message="boom")
        stream.push(ErrorEvent(reason="error", error=error_msg))
        assert await stream.result() == error_msg

    async def test_iterates_all_events(self) -> None:
        stream = AssistantMessageEventStream()
        msg = _mk_message()
        stream.push(StartEvent(partial=msg))
        stream.push(TextDeltaEvent(content_index=0, delta="h", partial=msg))
        stream.push(TextDeltaEvent(content_index=0, delta="i", partial=msg))
        stream.push(DoneEvent(reason="stop", message=msg))
        events = [e async for e in stream]
        assert [e.type for e in events] == ["start", "text_delta", "text_delta", "done"]

    async def test_factory_function_returns_fresh_stream(self) -> None:
        s1 = create_assistant_message_event_stream()
        s2 = create_assistant_message_event_stream()
        assert s1 is not s2
        assert isinstance(s1, AssistantMessageEventStream)

    async def test_result_without_terminal_event_raises_on_timeout(self) -> None:
        stream = AssistantMessageEventStream()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stream.result(), timeout=0.05)
