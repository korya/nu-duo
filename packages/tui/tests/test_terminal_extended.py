"""Extended tests for ``nu_tui.terminal`` capability detection."""

from __future__ import annotations

import pytest
from nu_tui.terminal import Terminal, TerminalSize


def test_supports_true_color_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert Terminal().supports_true_color() is True


def test_supports_true_color_24bit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLORTERM", "24bit")
    assert Terminal().supports_true_color() is True


def test_supports_true_color_kitty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    assert Terminal().supports_true_color() is True


def test_supports_true_color_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert Terminal().supports_true_color() is False


def test_supports_mouse_false_for_dumb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "dumb")
    assert Terminal().supports_mouse() is False


def test_supports_kitty_keyboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    assert Terminal().supports_kitty_keyboard() is True
    monkeypatch.setenv("TERM_PROGRAM", "xterm")
    monkeypatch.setenv("TERM", "xterm-256color")
    assert Terminal().supports_kitty_keyboard() is False


def test_supports_hyperlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "iTerm2.app")
    monkeypatch.delenv("VTE_VERSION", raising=False)
    assert Terminal().supports_hyperlinks() is True
    monkeypatch.setenv("TERM_PROGRAM", "unknown")
    assert Terminal().supports_hyperlinks() is False


def test_supports_hyperlinks_vte(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VTE_VERSION", "6800")
    monkeypatch.setenv("TERM_PROGRAM", "unknown")
    assert Terminal().supports_hyperlinks() is True


def test_rows_and_columns_properties() -> None:

    term = Terminal(size=TerminalSize(columns=132, rows=42))
    assert term.rows == 42
    assert term.columns == 132
