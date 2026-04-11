"""Tests for ``nu_coding_agent.core.timings``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import nu_coding_agent.core.timings as timings_module
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType


@pytest.fixture
def enabled_timings(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setenv("NU_TIMING", "1")
    reloaded = importlib.reload(timings_module)
    yield reloaded
    monkeypatch.delenv("NU_TIMING", raising=False)
    importlib.reload(timings_module)


def test_disabled_is_noop(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("NU_TIMING", raising=False)
    t = importlib.reload(timings_module)
    t.reset_timings()
    t.time_event("startup")
    t.print_timings()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_enabled_records_and_prints(
    enabled_timings: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t = enabled_timings
    t.reset_timings()
    t.time_event("first")
    t.time_event("second")
    t.print_timings()
    captured = capsys.readouterr()
    assert "Startup Timings" in captured.err
    assert "first" in captured.err
    assert "second" in captured.err
    assert "TOTAL" in captured.err


def test_print_timings_noop_when_empty(
    enabled_timings: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t = enabled_timings
    t.reset_timings()
    t.print_timings()
    assert capsys.readouterr().err == ""
