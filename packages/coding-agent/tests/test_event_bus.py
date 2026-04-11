"""Tests for ``nu_coding_agent.core.event_bus``."""

from __future__ import annotations

import asyncio

import pytest
from nu_coding_agent.core.event_bus import create_event_bus


def test_emit_to_sync_handler() -> None:
    bus = create_event_bus()
    received: list[object] = []
    bus.on("ch", received.append)
    bus.emit("ch", 1)
    bus.emit("ch", "two")
    assert received == [1, "two"]


def test_unsubscribe_stops_delivery() -> None:
    bus = create_event_bus()
    received: list[object] = []
    unsub = bus.on("ch", received.append)
    bus.emit("ch", "before")
    unsub()
    bus.emit("ch", "after")
    assert received == ["before"]


def test_unsubscribe_is_idempotent() -> None:
    bus = create_event_bus()
    unsub = bus.on("ch", lambda _data: None)
    unsub()
    unsub()  # second call must not raise


def test_clear_drops_all_handlers() -> None:
    bus = create_event_bus()
    received: list[object] = []
    bus.on("a", lambda d: received.append(("a", d)))
    bus.on("b", lambda d: received.append(("b", d)))
    bus.clear()
    bus.emit("a", 1)
    bus.emit("b", 2)
    assert received == []


def test_handler_exception_is_logged_and_swallowed(capsys: pytest.CaptureFixture[str]) -> None:
    bus = create_event_bus()

    def boom(_data: object) -> None:
        raise RuntimeError("kaboom")

    bus.on("ch", boom)
    received: list[object] = []
    bus.on("ch", received.append)
    bus.emit("ch", 42)
    captured = capsys.readouterr()
    assert "Event handler error (ch)" in captured.err
    assert received == [42]  # subsequent handlers still run


@pytest.mark.asyncio
async def test_async_handler_runs() -> None:
    bus = create_event_bus()
    received: list[object] = []

    async def handler(data: object) -> None:
        await asyncio.sleep(0)
        received.append(data)

    bus.on("ch", handler)
    bus.emit("ch", "x")
    # Give the event loop a couple of ticks to run the scheduled task.
    for _ in range(5):
        await asyncio.sleep(0)
    assert received == ["x"]
