"""Tests for ``nu_tui.terminal``."""

from __future__ import annotations

import pytest
from nu_tui.terminal import Terminal, TerminalSize


def test_terminal_size_dataclass_round_trip() -> None:
    size = TerminalSize(columns=120, rows=50)
    assert size.columns == 120
    assert size.rows == 50


def test_terminal_get_size_uses_fixed_size_when_provided() -> None:
    """Tests can pin the terminal dimensions without involving stdin/stdout."""
    term = Terminal(size=TerminalSize(columns=132, rows=42))
    size = term.get_size()
    assert size.columns == 132
    assert size.rows == 42
    assert term.get_columns() == 132
    assert term.get_rows() == 42


def test_terminal_get_size_falls_back_when_not_a_tty() -> None:
    """``shutil.get_terminal_size`` always returns something usable."""
    term = Terminal()
    size = term.get_size()
    assert size.columns > 0
    assert size.rows > 0


def test_terminal_no_color_env_disables_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("CLICOLOR_FORCE", raising=False)
    assert Terminal().supports_color() is False


def test_terminal_clicolor_force_overrides_no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("CLICOLOR_FORCE", "1")
    assert Terminal().supports_color() is True


def test_terminal_no_color_takes_precedence_over_clicolor_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``NO_COLOR`` is the documented opt-out and wins over ``CLICOLOR_FORCE``."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("CLICOLOR_FORCE", "1")
    assert Terminal().supports_color() is False


def test_terminal_is_termux_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    assert Terminal().is_termux() is False
    monkeypatch.setenv("TERMUX_VERSION", "0.118")
    assert Terminal().is_termux() is True
