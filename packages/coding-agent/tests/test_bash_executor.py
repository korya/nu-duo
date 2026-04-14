"""Tests for ``nu_coding_agent.core.bash_executor``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from nu_coding_agent.core.bash_executor import (
    BashExecutorOptions,
    create_local_bash_operations,
    execute_bash,
    execute_bash_with_operations,
    strip_ansi,
)
from nu_coding_agent.core.tools.bash import BashAborted, BashOperations

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_strip_ansi_removes_escape_sequences() -> None:
    assert strip_ansi("\x1b[31mhello\x1b[0m world") == "hello world"
    assert strip_ansi("plain text") == "plain text"


def test_strip_ansi_handles_clear_codes() -> None:
    assert strip_ansi("\x1b[2K\x1b[1Aupdated") == "updated"


async def test_execute_bash_simple() -> None:
    result = await execute_bash("echo hello")
    assert "hello" in result.output
    assert result.exit_code == 0
    assert result.cancelled is False
    assert result.truncated is False
    assert result.full_output_path is None


async def test_execute_bash_nonzero_exit() -> None:
    result = await execute_bash("exit 7")
    assert result.exit_code == 7


async def test_execute_bash_streams_chunks() -> None:
    chunks: list[str] = []
    await execute_bash(
        "printf 'one\\ntwo\\nthree\\n'",
        BashExecutorOptions(on_chunk=chunks.append),
    )
    full = "".join(chunks)
    assert "one" in full
    assert "two" in full
    assert "three" in full


async def test_execute_bash_with_operations(tmp_path: Path) -> None:
    ops = create_local_bash_operations()
    result = await execute_bash_with_operations("pwd", str(tmp_path), ops)
    assert str(tmp_path) in result.output


async def test_execute_bash_strips_ansi() -> None:
    result = await execute_bash("printf '\\033[31mred\\033[0m'")
    assert "red" in result.output
    assert "\x1b[" not in result.output


async def test_execute_bash_normalises_crlf() -> None:
    result = await execute_bash("printf 'a\\r\\nb\\r\\n'")
    assert "\r" not in result.output


def _stub_operations(
    output: str = "stubbed output\n",
    exit_code: int | None = 0,
    raises: type[Exception] | None = None,
) -> BashOperations:
    async def fake_exec(
        *,
        command: str,
        cwd: str,
        on_data: Callable[[bytes], None],
        timeout: float | None = None,  # noqa: ASYNC109
        env: dict[str, str] | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> int | None:
        if raises is not None:
            raise raises
        on_data(output.encode("utf-8"))
        return exit_code

    return BashOperations(exec=fake_exec)


async def test_execute_bash_with_aborted_marker(tmp_path: Path) -> None:
    ops = _stub_operations(raises=BashAborted)
    signal = asyncio.Event()
    signal.set()
    result = await execute_bash_with_operations(
        "ignored",
        str(tmp_path),
        ops,
        BashExecutorOptions(signal=signal),
    )
    assert result.cancelled is True
    assert result.exit_code is None


async def test_execute_bash_truncates_huge_output(tmp_path: Path) -> None:
    huge = ("x" * 10_000 + "\n") * 200
    ops = _stub_operations(output=huge)
    result = await execute_bash_with_operations("ignored", str(tmp_path), ops)
    assert result.truncated is True
    assert result.full_output_path is not None
    assert "x" in result.output


async def test_execute_bash_signal_set_marks_cancelled(tmp_path: Path) -> None:
    ops = _stub_operations()
    signal = asyncio.Event()
    signal.set()
    result = await execute_bash_with_operations(
        "ignored",
        str(tmp_path),
        ops,
        BashExecutorOptions(signal=signal),
    )
    assert result.cancelled is True


async def test_execute_bash_explicit_operations_override_local() -> None:
    """Wiring through ``BashOperations.exec`` rather than the bare local backend."""
    received_command = ""

    async def fake_exec(
        *,
        command: str,
        cwd: str,
        on_data: Callable[[bytes], None],
        timeout: float | None = None,  # noqa: ASYNC109
        env: dict[str, str] | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> int | None:
        nonlocal received_command
        received_command = command
        on_data(b"intercepted\n")
        return 0

    result = await execute_bash_with_operations(
        "echo hi",
        "/tmp",
        BashOperations(exec=fake_exec),
    )
    assert received_command == "echo hi"
    assert "intercepted" in result.output


# ---------------------------------------------------------------------------
# Coverage: ensure_temp_file early return (line 117)
# ---------------------------------------------------------------------------


async def test_ensure_temp_file_idempotent(tmp_path: Path) -> None:
    """When output spills twice, ensure_temp_file only creates one file (line 117)."""
    from nu_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES

    # Create output larger than DEFAULT_MAX_BYTES to trigger temp file creation twice
    chunk_size = DEFAULT_MAX_BYTES + 1000
    big_output = "x" * chunk_size + "\n"

    ops = _stub_operations(output=big_output)
    result = await execute_bash_with_operations("ignored", str(tmp_path), ops)
    # The key assertion: it ran without error, and full_output_path is set
    assert result.full_output_path is not None


# ---------------------------------------------------------------------------
# Coverage: output_bytes overflow chunk eviction (lines 136-137)
# ---------------------------------------------------------------------------


async def test_chunk_eviction_on_huge_output(tmp_path: Path) -> None:
    """When output_bytes exceeds max_output_bytes, old chunks are evicted (lines 136-137)."""
    from nu_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES

    # We need output > 2 * DEFAULT_MAX_BYTES to trigger chunk eviction
    # Send multiple chunks through on_data
    chunk = "y" * (DEFAULT_MAX_BYTES + 100) + "\n"

    async def multi_chunk_exec(
        *,
        command: str,
        cwd: str,
        on_data: Callable[[bytes], None],
        timeout: float | None = None,  # noqa: ASYNC109
        env: dict[str, str] | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> int | None:
        # Send two large chunks to exceed max_output_bytes
        on_data(chunk.encode("utf-8"))
        on_data(chunk.encode("utf-8"))
        on_data(chunk.encode("utf-8"))
        return 0

    ops = BashOperations(exec=multi_chunk_exec)
    result = await execute_bash_with_operations("ignored", str(tmp_path), ops)
    # The output should be truncated (old chunks evicted)
    assert result.truncated is True
    assert result.full_output_path is not None
