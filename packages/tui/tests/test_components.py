"""Tests for ``nu_tui.components.{spacer,text}``."""

from __future__ import annotations

from nu_tui.components import Spacer, Text
from nu_tui.components.text import cell_len

# ---------------------------------------------------------------------------
# Spacer
# ---------------------------------------------------------------------------


def test_spacer_default_height_is_one() -> None:
    assert Spacer().render(40) == [""]


def test_spacer_height_emits_n_blank_rows() -> None:
    assert Spacer(3).render(40) == ["", "", ""]


def test_spacer_clamps_height_to_one() -> None:
    """A misconfigured ``height < 1`` still renders something visible."""
    assert Spacer(0).render(40) == [""]
    assert Spacer(-5).render(40) == [""]


def test_spacer_width_is_irrelevant() -> None:
    """``Spacer`` produces the same output regardless of width."""
    assert Spacer(2).render(10) == Spacer(2).render(120)


# ---------------------------------------------------------------------------
# Text — basic rendering
# ---------------------------------------------------------------------------


def test_text_empty_returns_no_lines() -> None:
    assert Text("").render(40) == []


def test_text_short_string_fits_on_one_line() -> None:
    assert Text("hi").render(40) == ["hi"]


def test_text_explicit_newlines_become_row_boundaries() -> None:
    assert Text("a\nb\nc").render(40) == ["a", "b", "c"]


def test_text_set_text_replaces_value() -> None:
    text = Text("first")
    text.set_text("second")
    assert text.render(40) == ["second"]
    assert text.text == "second"


# ---------------------------------------------------------------------------
# Text — wrapping
# ---------------------------------------------------------------------------


def test_text_wraps_long_line_at_word_boundary() -> None:
    """Lines longer than the width wrap on spaces."""
    text = Text("the quick brown fox jumps over the lazy dog")
    assert text.render(15) == [
        "the quick brown",
        "fox jumps over",
        "the lazy dog",
    ]


def test_text_does_not_wrap_when_line_fits() -> None:
    text = Text("short string")
    assert text.render(80) == ["short string"]


def test_text_handles_word_longer_than_width_via_hard_break() -> None:
    """A single word longer than the width is hard-broken into chunks."""
    text = Text("xxxxxxxxxxxxxxxxxx")  # 18 chars, no spaces
    assert text.render(5) == ["xxxxx", "xxxxx", "xxxxx", "xxx"]


def test_text_mixes_short_and_long_words() -> None:
    """A mix of short words and one over-width word still wraps cleanly."""
    text = Text("a longwordthatdoesntfit b")
    rendered = text.render(8)
    # All output rows respect the width (or come from the hard-break path).
    assert all(cell_len(row) <= 8 for row in rendered)
    # No content was lost.
    flattened = "".join(rendered)
    assert "longwordthatdoesntfit" in flattened
    assert "a" in flattened
    assert "b" in flattened


def test_text_wraps_each_logical_line_independently() -> None:
    """Hard newlines aren't crossed by wrapping."""
    text = Text("first line\nthe quick brown fox jumps")
    assert text.render(15) == [
        "first line",
        "the quick brown",
        "fox jumps",
    ]


def test_text_preserves_unicode_and_double_width_chars() -> None:
    """Unicode and double-width characters wrap on visible cells, not codepoints."""
    text = Text("héllo wörld")
    # 11 visible cells, fits in 12.
    assert text.render(12) == ["héllo wörld"]
    # Wraps on the space when narrower than 11.
    assert text.render(7) == ["héllo", "wörld"]


def test_text_zero_width_returns_unwrapped_input() -> None:
    """Pathological width=0 falls back to the source string verbatim."""
    text = Text("hello")
    assert text.render(0) == ["hello"]
