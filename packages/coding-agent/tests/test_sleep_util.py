"""Tests for ``nu_coding_agent.utils.sleep``."""

from __future__ import annotations

import asyncio
import time

import pytest
from nu_coding_agent.utils.sleep import Aborted, sleep


async def test_sleep_returns_after_duration() -> None:
    start = time.monotonic()
    await sleep(0.05)
    assert time.monotonic() - start >= 0.04


async def test_sleep_with_unset_signal_completes() -> None:
    start = time.monotonic()
    await sleep(0.05, asyncio.Event())
    assert time.monotonic() - start >= 0.04


async def test_sleep_already_set_signal_raises_immediately() -> None:
    signal = asyncio.Event()
    signal.set()
    with pytest.raises(Aborted):
        await sleep(10, signal)


async def test_sleep_signal_set_during_wait_raises() -> None:
    signal = asyncio.Event()

    async def fire() -> None:
        await asyncio.sleep(0.02)
        signal.set()

    fire_task = asyncio.create_task(fire())
    with pytest.raises(Aborted):
        await sleep(10, signal)
    await fire_task
