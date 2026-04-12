"""Additional tests for ``nu_tui.utils`` to push coverage above threshold."""

from __future__ import annotations

from nu_tui.utils import (
    CURSOR_MARKER,
    decode_kitty_printable,
    is_punctuation_char,
    is_whitespace_char,
    slice_by_column,
)


def test_is_whitespace_char() -> None:
    assert is_whitespace_char(" ") is True
    assert is_whitespace_char("\t") is True
    assert is_whitespace_char("a") is False
    assert is_whitespace_char("ab") is False


def test_is_punctuation_char() -> None:
    assert is_punctuation_char(".") is True
    assert is_punctuation_char(",") is True
    assert is_punctuation_char("!") is True
    assert is_punctuation_char("a") is False
    assert is_punctuation_char("1") is False
    assert is_punctuation_char(" ") is False
    assert is_punctuation_char("é") is False  # non-ASCII → not punctuation
    assert is_punctuation_char("ab") is False  # multi-char → False


def test_slice_by_column_basic() -> None:
    result = slice_by_column("hello world", 0, 5)
    assert result == "hello"


def test_slice_by_column_offset() -> None:
    result = slice_by_column("hello world", 6, 5)
    assert result == "world"


def test_slice_by_column_with_pad() -> None:
    result = slice_by_column("hi", 0, 10, pad=True)
    assert len(result) == 10
    assert result.startswith("hi")


def test_slice_by_column_beyond_end() -> None:
    result = slice_by_column("hi", 0, 100)
    assert result == "hi"


def test_slice_by_column_beyond_end_pad() -> None:
    result = slice_by_column("hi", 0, 100, pad=True)
    assert result.startswith("hi")
    assert len(result) == 100


def test_decode_kitty_printable_valid() -> None:
    assert decode_kitty_printable("\x1b[97u") == "a"
    assert decode_kitty_printable("\x1b[65u") == "A"


def test_decode_kitty_printable_with_modifier() -> None:
    # ESC [ 97 ; 2 u = 'a' with shift
    assert decode_kitty_printable("\x1b[97;2u") == "a"


def test_decode_kitty_printable_control_rejected() -> None:
    assert decode_kitty_printable("\x1b[1u") is None  # C-a
    assert decode_kitty_printable("\x1b[127u") is None  # DEL


def test_decode_kitty_printable_invalid_format() -> None:
    assert decode_kitty_printable("not-kitty") is None
    assert decode_kitty_printable("\x1b[u") is None
    assert decode_kitty_printable("\x1b[abcu") is None


def test_cursor_marker_is_zero_width_escape() -> None:
    assert CURSOR_MARKER == "\x1b_pi:c\x07"
