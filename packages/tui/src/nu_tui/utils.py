"""Rendering utility functions — subset port of ``packages/tui/src/utils.ts``.

The upstream module is 1068 LoC and includes ANSI escape parsing,
cell-width computation, text truncation and slicing, background color
application, grapheme segmentation, and more. This slice ports only
the functions consumed by the components landing in Phase 5.2:

* :func:`visible_width` — visible cell width of a string,
  ignoring ANSI escapes.
* :func:`truncate_to_width` — truncate a string to ``max_width``
  visible cells, appending a suffix (default ``…``) when truncated.

The rest land alongside their consumers in later sub-slices.
"""

from __future__ import annotations

import re

from rich.text import Text as RichText

#: Regex that matches ANSI escape sequences (CSI + OSC + ESC + APC).
#: Used by :func:`visible_width` to strip escapes before measuring.
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-9;]*[A-Za-z]|\][^\x07]*\x07|_[^\x07]*\x07|.)")


def visible_width(text: str) -> int:
    """Return the visible cell width of ``text``, ignoring ANSI escapes.

    Delegates to :class:`rich.text.Text` for Unicode/CJK width
    computation after stripping ANSI control sequences that Rich
    would otherwise count as zero-width.
    """
    stripped = ANSI_ESCAPE_RE.sub("", text)
    return RichText(stripped).cell_len


def truncate_to_width(text: str, max_width: int, suffix: str = "…") -> str:
    """Truncate ``text`` to at most ``max_width`` visible cells.

    If the string already fits, it's returned unchanged. Otherwise
    it's clipped and ``suffix`` is appended. The combined result is
    guaranteed to not exceed ``max_width`` visible cells.

    Mirrors upstream ``truncateToWidth``.
    """
    if visible_width(text) <= max_width:
        return text

    suffix_width = visible_width(suffix)
    target = max(0, max_width - suffix_width)

    # Walk characters until we exceed the target width.
    current_width = 0
    cut = 0
    for i, ch in enumerate(text):
        ch_width = visible_width(ch)
        if current_width + ch_width > target:
            cut = i
            break
        current_width += ch_width
    else:
        cut = len(text)

    return text[:cut] + suffix


def is_whitespace_char(ch: str) -> bool:
    """Return ``True`` for whitespace characters (matching upstream ``isWhitespaceChar``)."""
    return len(ch) == 1 and ch.isspace()


def is_punctuation_char(ch: str) -> bool:
    """Return ``True`` for punctuation (non-alphanumeric, non-whitespace ASCII)."""
    if len(ch) != 1:
        return False
    code = ord(ch)
    if code > 127:
        return False  # treat non-ASCII as word characters
    return not ch.isalnum() and not ch.isspace()


def slice_by_column(text: str, start_col: int, col_count: int, pad: bool = False) -> str:
    """Slice ``text`` to a visible column range.

    Walk characters from column ``start_col``, take up to ``col_count``
    visible columns. If ``pad`` is ``True`` and the result is shorter
    than ``col_count``, right-pad with spaces. Mirrors upstream
    ``sliceByColumn``.
    """
    current_col = 0
    start_idx = 0
    # Advance to start_col
    for i, ch in enumerate(text):
        if current_col >= start_col:
            start_idx = i
            break
        current_col += visible_width(ch)
    else:
        start_idx = len(text)

    # Collect col_count columns from start_idx
    collected = 0
    end_idx = start_idx
    for i in range(start_idx, len(text)):
        ch_width = visible_width(text[i])
        if collected + ch_width > col_count:
            break
        collected += ch_width
        end_idx = i + 1

    result = text[start_idx:end_idx]
    if pad and collected < col_count:
        result += " " * (col_count - collected)
    return result


def decode_kitty_printable(data: str) -> str | None:
    """Decode a Kitty CSI-u printable key sequence.

    Kitty terminals with flag 1 (disambiguate) send all keys as
    ``ESC [ <codepoint> u``. If ``data`` matches that pattern and the
    codepoint is a printable character, return the character; otherwise
    return ``None``.
    """
    if not data.startswith("\x1b[") or not data.endswith("u"):
        return None
    body = data[2:-1]
    # May have modifiers: codepoint;modifier
    parts = body.split(";")
    try:
        codepoint = int(parts[0])
    except ValueError:
        return None
    ch = chr(codepoint)
    # Reject control characters
    if codepoint < 32 or codepoint == 0x7F or (0x80 <= codepoint <= 0x9F):
        return None
    return ch


#: Cursor position marker — APC sequence that terminals ignore.
#: Components emit this at the cursor position when focused;
#: the TUI bridge finds + strips it and positions the hardware cursor.
CURSOR_MARKER = "\x1b_pi:c\x07"


__all__ = [
    "CURSOR_MARKER",
    "decode_kitty_printable",
    "is_punctuation_char",
    "is_whitespace_char",
    "slice_by_column",
    "truncate_to_width",
    "visible_width",
]
