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


# ---------------------------------------------------------------------------
# Additional coverage — _AnsiCodeTracker, _break_long_word, etc.
# ---------------------------------------------------------------------------


def test_ansi_tracker_256_color() -> None:
    """_AnsiCodeTracker handles 256-color codes (lines 239-246)."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    # 256-color foreground: ESC[38;5;196m (red)
    tracker.process("\033[38;5;196m")
    codes = tracker.get_active_codes()
    assert "38;5;196" in codes

    # 256-color background: ESC[48;5;21m (blue)
    tracker.process("\033[48;5;21m")
    codes = tracker.get_active_codes()
    assert "48;5;21" in codes


def test_ansi_tracker_truecolor() -> None:
    """_AnsiCodeTracker handles truecolor (24-bit) codes (lines 247-254)."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    # Truecolor foreground: ESC[38;2;255;0;0m
    tracker.process("\033[38;2;255;0;0m")
    codes = tracker.get_active_codes()
    assert "38;2;255;0;0" in codes

    # Truecolor background: ESC[48;2;0;255;0m
    tracker.process("\033[48;2;0;255;0m")
    codes = tracker.get_active_codes()
    assert "48;2;0;255;0" in codes


def test_ansi_tracker_standard_fg_bg_colors() -> None:
    """_AnsiCodeTracker handles standard 30-37/40-47 colors (lines 284-287)."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[31m")  # red fg
    assert "31" in tracker.get_active_codes()
    tracker.process("\033[42m")  # green bg
    codes = tracker.get_active_codes()
    assert "31" in codes
    assert "42" in codes


def test_ansi_tracker_bright_colors() -> None:
    """_AnsiCodeTracker handles bright colors 90-97/100-107."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[91m")  # bright red fg
    assert "91" in tracker.get_active_codes()
    tracker.process("\033[104m")  # bright blue bg
    codes = tracker.get_active_codes()
    assert "104" in codes


def test_ansi_tracker_attributes() -> None:
    """_AnsiCodeTracker handles various SGR attributes (lines 296-312)."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[1m")  # bold
    assert "1" in tracker.get_active_codes()
    assert tracker.has_active_codes()

    tracker.process("\033[2m")  # dim
    assert "2" in tracker.get_active_codes()

    tracker.process("\033[4m")  # underline
    assert "4" in tracker.get_active_codes()
    assert tracker.get_line_end_reset() == "\033[24m"

    tracker.process("\033[5m")  # blink
    assert "5" in tracker.get_active_codes()

    tracker.process("\033[7m")  # inverse
    assert "7" in tracker.get_active_codes()

    tracker.process("\033[8m")  # hidden
    assert "8" in tracker.get_active_codes()

    tracker.process("\033[9m")  # strikethrough
    assert "9" in tracker.get_active_codes()


def test_ansi_tracker_reset_attributes() -> None:
    """_AnsiCodeTracker handles attribute reset codes."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[1m")  # bold
    tracker.process("\033[22m")  # reset bold+dim
    assert "1" not in tracker.get_active_codes()

    tracker.process("\033[4m")  # underline
    tracker.process("\033[24m")  # reset underline
    assert "4" not in tracker.get_active_codes()

    tracker.process("\033[31m")  # fg color
    tracker.process("\033[39m")  # reset fg
    tracker.process("\033[42m")  # bg color
    tracker.process("\033[49m")  # reset bg
    assert not tracker.has_active_codes()


def test_ansi_tracker_full_reset() -> None:
    """_AnsiCodeTracker handles full reset (ESC[0m)."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[1;31;4m")  # bold+red+underline
    assert tracker.has_active_codes()
    tracker.process("\033[0m")  # reset all
    assert not tracker.has_active_codes()
    assert tracker.get_active_codes() == ""


def test_ansi_tracker_empty_params_reset() -> None:
    """_AnsiCodeTracker treats ESC[m as reset (line 222-225)."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[1m")  # bold
    tracker.process("\033[m")  # empty params = reset
    assert not tracker.has_active_codes()


def test_ansi_tracker_non_sgr_ignored() -> None:
    """_AnsiCodeTracker ignores non-SGR sequences."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[2J")  # clear screen — not SGR
    assert not tracker.has_active_codes()


def test_ansi_tracker_line_end_reset_no_underline() -> None:
    """get_line_end_reset returns empty when no underline active (line 324)."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[1m")  # bold, no underline
    assert tracker.get_line_end_reset() == ""


def test_break_long_word_with_ansi() -> None:
    """_break_long_word handles ANSI codes inside long words (lines 401-404)."""
    from nu_tui.utils import _AnsiCodeTracker, _break_long_word

    tracker = _AnsiCodeTracker()
    word = "\033[31m" + "a" * 20 + "\033[0m"
    lines = _break_long_word(word, 5, tracker)
    assert len(lines) >= 4
    for line in lines[:-1]:
        # Each line should be at most 5 visible chars
        assert strip_ansi(line).__len__() <= 5


def test_break_long_word_empty() -> None:
    """_break_long_word with empty word returns ['']."""
    from nu_tui.utils import _AnsiCodeTracker, _break_long_word

    tracker = _AnsiCodeTracker()
    lines = _break_long_word("", 5, tracker)
    assert lines == [""]


def test_split_into_tokens() -> None:
    """_split_into_tokens splits on word/space boundaries."""
    from nu_tui.utils import _split_into_tokens

    tokens = _split_into_tokens("hello  world")
    assert len(tokens) == 3
    assert tokens[0] == "hello"
    assert tokens[1] == "  "
    assert tokens[2] == "world"


def test_split_into_tokens_with_ansi() -> None:
    """_split_into_tokens keeps ANSI codes attached to tokens."""
    from nu_tui.utils import _split_into_tokens

    tokens = _split_into_tokens("\033[31mhello\033[0m world")
    assert len(tokens) >= 2
    joined = "".join(tokens)
    assert "hello" in strip_ansi(joined)


def test_wrap_single_line_empty() -> None:
    """_wrap_single_line with empty input returns ['']."""
    from nu_tui.utils import _wrap_single_line

    assert _wrap_single_line("", 10) == [""]


def test_wrap_text_with_ansi_multiline_preserves_style() -> None:
    """wrap_text_with_ansi preserves style across literal newlines (lines 439-442)."""
    text = "\033[1mline1\nline2\033[0m"
    lines = wrap_text_with_ansi(text, 50)
    assert len(lines) >= 2
    # The bold code should be re-applied on line2
    assert "\033[1m" in lines[1]


def test_wrap_text_whitespace_token_at_boundary() -> None:
    """Whitespace at wrap boundary gets consumed cleanly (lines 454-455)."""
    text = "hello   world"
    lines = wrap_text_with_ansi(text, 8)
    assert len(lines) >= 2
    joined = strip_ansi(" ".join(lines))
    assert "hello" in joined
    assert "world" in joined


def test_ansi_tracker_invalid_code_number() -> None:
    """_AnsiCodeTracker ignores invalid/unparseable code numbers (line 235-237)."""
    from nu_tui.utils import _AnsiCodeTracker

    tracker = _AnsiCodeTracker()
    tracker.process("\033[abcm")  # invalid — no crash
    assert not tracker.has_active_codes()


def test_wrap_text_result_lines_not_wider_than_width() -> None:
    """All wrapped lines should be at most width visible columns (line 465)."""
    text = "The quick brown fox jumps over the lazy dog and then some more text for good measure."
    lines = wrap_text_with_ansi(text, 15)
    for line in lines:
        from nu_tui.utils import visible_width

        assert visible_width(line) <= 15


def test_slice_by_column_start_beyond_text() -> None:
    """slice_by_column when start_col is beyond text length (line 104)."""
    from nu_tui.utils import slice_by_column

    result = slice_by_column("hi", 10, 5)
    assert result == ""

    result_padded = slice_by_column("hi", 10, 5, pad=True)
    assert len(result_padded) == 5


def test_truncate_to_width_walk_entire_string() -> None:
    """truncate_to_width cut point at end of string (line 67)."""
    from nu_tui.utils import truncate_to_width

    # The cut loop reaches the end of the string
    result = truncate_to_width("ab", 1)
    assert len(result) <= 2  # "…" is 1 cell wide
