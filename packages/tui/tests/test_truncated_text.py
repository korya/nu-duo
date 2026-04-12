"""Tests for ``nu_tui.components.TruncatedText``."""

from __future__ import annotations

from nu_tui.components.truncated_text import TruncatedText
from nu_tui.utils import visible_width


def test_short_text_no_truncation() -> None:
    tt = TruncatedText("hi")
    lines = tt.render(20)
    assert len(lines) == 1
    assert "hi" in lines[0]
    assert visible_width(lines[0]) == 20  # padded to width


def test_long_text_truncated() -> None:
    tt = TruncatedText("this is a very long string that will be truncated")
    lines = tt.render(15)
    assert len(lines) == 1
    assert "…" in lines[0]
    assert visible_width(lines[0]) <= 15


def test_multiline_input_takes_first_line() -> None:
    tt = TruncatedText("first\nsecond\nthird")
    lines = tt.render(40)
    assert len(lines) == 1
    assert "first" in lines[0]
    assert "second" not in lines[0]


def test_padding_x_adds_horizontal_space() -> None:
    tt = TruncatedText("hi", padding_x=3)
    lines = tt.render(20)
    assert lines[0].startswith("   ")  # 3 spaces


def test_padding_y_adds_vertical_blank_rows() -> None:
    tt = TruncatedText("hi", padding_y=2)
    lines = tt.render(20)
    # 2 above + 1 content + 2 below = 5
    assert len(lines) == 5
    assert lines[0] == " " * 20
    assert lines[-1] == " " * 20


def test_set_text_replaces_value() -> None:
    tt = TruncatedText("first")
    tt.set_text("second")
    assert tt.text == "second"
    lines = tt.render(20)
    assert "second" in lines[0]


def test_empty_text_still_pads_to_width() -> None:
    tt = TruncatedText("")
    lines = tt.render(10)
    assert len(lines) == 1
    assert visible_width(lines[0]) == 10
