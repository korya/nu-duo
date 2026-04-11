"""Tests for ``nu_coding_agent.core.tools.file_mutation_queue``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from nu_coding_agent.core.tools.file_mutation_queue import with_file_mutation_queue


@pytest.mark.asyncio
async def test_serialises_same_file(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("")
    log: list[str] = []

    async def make_op(label: str) -> str:
        async def _op() -> str:
            log.append(f"start-{label}")
            await asyncio.sleep(0.05)
            log.append(f"end-{label}")
            return label

        return await with_file_mutation_queue(str(target), _op)

    results = await asyncio.gather(make_op("a"), make_op("b"), make_op("c"))
    assert results == ["a", "b", "c"]
    # Each op must complete fully before the next one starts.
    for i in range(0, len(log), 2):
        assert log[i].startswith("start-")
        assert log[i + 1].startswith("end-")
        assert log[i].split("-")[1] == log[i + 1].split("-")[1]


@pytest.mark.asyncio
async def test_different_files_run_in_parallel(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("")
    b.write_text("")
    started = asyncio.Event()
    other_started = asyncio.Event()

    async def op_a() -> None:
        async def _op() -> None:
            started.set()
            await asyncio.wait_for(other_started.wait(), timeout=1.0)

        await with_file_mutation_queue(str(a), _op)

    async def op_b() -> None:
        async def _op() -> None:
            await asyncio.wait_for(started.wait(), timeout=1.0)
            other_started.set()

        await with_file_mutation_queue(str(b), _op)

    await asyncio.gather(op_a(), op_b())


@pytest.mark.asyncio
async def test_lock_released_on_exception(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    target.write_text("")

    async def boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await with_file_mutation_queue(str(target), boom)

    # Subsequent operation must still be able to acquire the lock.
    async def ok() -> str:
        return "ok"

    assert await with_file_mutation_queue(str(target), ok) == "ok"


@pytest.mark.asyncio
async def test_resolves_to_same_key_for_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "shared.txt"
    target.write_text("")
    monkeypatch.chdir(tmp_path)
    log: list[str] = []

    async def op(label: str) -> None:
        async def _op() -> None:
            log.append(f"start-{label}")
            await asyncio.sleep(0.02)
            log.append(f"end-{label}")

        await with_file_mutation_queue(label, _op)

    await asyncio.gather(op("shared.txt"), op(str(target)))
    # If keys collapsed correctly, the operations were serialised, so
    # log alternates start/end without interleaving.
    assert log[0].startswith("start-")
    assert log[1].startswith("end-")
    assert log[2].startswith("start-")
    assert log[3].startswith("end-")
