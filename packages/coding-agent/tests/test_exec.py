"""Tests for ``nu_coding_agent.core.exec``."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import pytest
from nu_coding_agent.core.exec import ExecOptions, exec_command

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_exec_simple_echo(tmp_path: Path) -> None:
    result = await exec_command(sys.executable, ["-c", "print('hi')"], str(tmp_path))
    assert result.code == 0
    assert "hi" in result.stdout
    assert result.killed is False


@pytest.mark.asyncio
async def test_exec_nonzero_exit(tmp_path: Path) -> None:
    result = await exec_command(sys.executable, ["-c", "import sys; sys.exit(7)"], str(tmp_path))
    assert result.code == 7
    assert result.killed is False


@pytest.mark.asyncio
async def test_exec_captures_stderr(tmp_path: Path) -> None:
    result = await exec_command(
        sys.executable,
        ["-c", "import sys; sys.stderr.write('boom')"],
        str(tmp_path),
    )
    assert "boom" in result.stderr


@pytest.mark.asyncio
async def test_exec_missing_binary(tmp_path: Path) -> None:
    result = await exec_command("definitely-not-a-real-binary-xyz", [], str(tmp_path))
    assert result.code == 1
    assert result.stdout == ""


@pytest.mark.asyncio
async def test_exec_timeout_kills(tmp_path: Path) -> None:
    result = await exec_command(
        sys.executable,
        ["-c", "import time; time.sleep(10)"],
        str(tmp_path),
        ExecOptions(timeout=0.2),
    )
    assert result.killed is True


@pytest.mark.asyncio
async def test_exec_signal_aborts(tmp_path: Path) -> None:
    signal = asyncio.Event()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.1)
        signal.set()

    cancel_task = asyncio.create_task(cancel_soon())
    result = await exec_command(
        sys.executable,
        ["-c", "import time; time.sleep(10)"],
        str(tmp_path),
        ExecOptions(signal=signal),
    )
    await cancel_task
    assert result.killed is True


@pytest.mark.asyncio
async def test_exec_signal_already_set(tmp_path: Path) -> None:
    signal = asyncio.Event()
    signal.set()
    result = await exec_command(
        sys.executable,
        ["-c", "import time; time.sleep(10)"],
        str(tmp_path),
        ExecOptions(signal=signal),
    )
    assert result.killed is True
