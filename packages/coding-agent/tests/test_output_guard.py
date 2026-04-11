"""Tests for ``nu_coding_agent.core.output_guard``."""

from __future__ import annotations

import sys

import pytest
from nu_coding_agent.core import output_guard


@pytest.fixture(autouse=True)
def _restore_after_test():  # pyright: ignore[reportUnusedFunction]
    yield
    output_guard.restore_stdout()


def test_takeover_redirects_stdout_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    output_guard.take_over_stdout()
    assert output_guard.is_stdout_taken_over()
    sys.stdout.write("hello\n")
    captured = capsys.readouterr()
    assert captured.err == "hello\n"
    assert captured.out == ""


def test_double_takeover_is_noop() -> None:
    output_guard.take_over_stdout()
    first_stdout = sys.stdout
    output_guard.take_over_stdout()
    assert sys.stdout is first_stdout


def test_restore_restores_real_stdout() -> None:
    real = sys.stdout
    output_guard.take_over_stdout()
    assert sys.stdout is not real
    output_guard.restore_stdout()
    assert sys.stdout is real
    assert not output_guard.is_stdout_taken_over()


def test_restore_without_takeover_is_noop() -> None:
    real = sys.stdout
    output_guard.restore_stdout()
    assert sys.stdout is real


def test_write_raw_stdout_bypasses_takeover(capsys: pytest.CaptureFixture[str]) -> None:
    output_guard.take_over_stdout()
    output_guard.write_raw_stdout("raw\n")
    captured = capsys.readouterr()
    assert captured.out == "raw\n"
    assert captured.err == ""


def test_write_raw_stdout_without_takeover(capsys: pytest.CaptureFixture[str]) -> None:
    output_guard.write_raw_stdout("plain\n")
    captured = capsys.readouterr()
    assert captured.out == "plain\n"


@pytest.mark.asyncio
async def test_flush_raw_stdout_works_in_both_states() -> None:
    await output_guard.flush_raw_stdout()
    output_guard.take_over_stdout()
    await output_guard.flush_raw_stdout()
