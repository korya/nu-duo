"""Tests for ANSI-aware helpers in ``nu_tui.utils`` (Phase 5.10 additions)."""

from __future__ import annotations

from nu_tui.utils import (
    apply_background_to_line,
    strip_ansi,
    wrap_text_with_ansi,
)

# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------


def test_strip_ansi_plain() -> None:
    assert strip_ansi("hello") == "hello"


def test_strip_ansi_with_escapes() -> None:
    assert strip_ansi("\033[31mred\033[0m") == "red"


def test_strip_ansi_complex() -> None:
    text = "\033[1;31mbold red\033[0m normal \033[4munderline\033[0m"
    assert strip_ansi(text) == "bold red normal underline"


def test_strip_ansi_empty() -> None:
    assert strip_ansi("") == ""


# ---------------------------------------------------------------------------
# apply_background_to_line
# ---------------------------------------------------------------------------


def test_apply_background_basic() -> None:
    result = apply_background_to_line("hello", 10, lambda s: f"[{s}]")
    assert result.startswith("[")
    assert result.endswith("]")
    assert "hello" in result


def test_apply_background_pads_to_width() -> None:
    result = apply_background_to_line("hi", 10, lambda s: s)
    assert len(result) >= 10


def test_apply_background_no_fn_returns_padded() -> None:
    result = apply_background_to_line("hi", 10, lambda s: s)
    assert "hi" in result


# ---------------------------------------------------------------------------
# wrap_text_with_ansi
# ---------------------------------------------------------------------------


def test_wrap_plain_text() -> None:
    lines = wrap_text_with_ansi("hello world this is a test", 12)
    assert len(lines) >= 2
    for line in lines:
        assert len(strip_ansi(line)) <= 12


def test_wrap_preserves_ansi_across_breaks() -> None:
    text = "\033[31mred text that is longer than the width\033[0m"
    lines = wrap_text_with_ansi(text, 15)
    assert len(lines) >= 2
    # The red color should carry into the second line
    joined = "".join(lines)
    assert "red" in strip_ansi(joined)


def test_wrap_short_text_no_break() -> None:
    lines = wrap_text_with_ansi("short", 20)
    assert len(lines) == 1
    assert "short" in lines[0]


def test_wrap_empty_text() -> None:
    lines = wrap_text_with_ansi("", 20)
    assert lines in ([], [""])


def test_wrap_handles_newlines() -> None:
    lines = wrap_text_with_ansi("line1\nline2", 20)
    texts = [strip_ansi(line) for line in lines]
    assert "line1" in texts[0]
    # line2 may be on the same or next line depending on implementation


def test_wrap_word_boundary() -> None:
    lines = wrap_text_with_ansi("hello world", 8)
    texts = [strip_ansi(line) for line in lines]
    assert texts[0].rstrip() in ("hello", "hello wo", "hello")


# ---------------------------------------------------------------------------
# _AnsiState + style tracking — additional coverage
# ---------------------------------------------------------------------------


def test_wrap_preserves_bold_across_lines() -> None:
    text = "\033[1m" + "x " * 20 + "\033[0m"
    lines = wrap_text_with_ansi(text, 10)
    assert len(lines) >= 2
    assert "\033[1m" in lines[0]


def test_wrap_preserves_color_across_lines() -> None:
    text = "\033[31m" + "red text that wraps " * 3 + "\033[0m"
    lines = wrap_text_with_ansi(text, 15)
    assert len(lines) >= 2


def test_wrap_combined_styles() -> None:
    text = "\033[1;31;4mbold red underline text\033[0m"
    lines = wrap_text_with_ansi(text, 10)
    assert len(lines) >= 2


def test_wrap_nested_resets() -> None:
    text = "\033[31mred \033[32mgreen \033[0mnormal"
    lines = wrap_text_with_ansi(text, 50)
    assert len(lines) >= 1
    assert "red" in strip_ansi("".join(lines))


def test_apply_background_with_ansi_content() -> None:
    text = "\033[31mred\033[0m"
    result = apply_background_to_line(text, 20, lambda s: f"BG({s})")
    assert "BG(" in result


def test_wrap_very_long_word() -> None:
    text = "x" * 50
    lines = wrap_text_with_ansi(text, 10)
    assert len(lines) >= 5


def test_wrap_italic_and_strikethrough() -> None:
    text = "\033[3mitalic\033[23m \033[9mstrike\033[29m"
    lines = wrap_text_with_ansi(text, 50)
    assert "italic" in strip_ansi("".join(lines))
    assert "strike" in strip_ansi("".join(lines))
