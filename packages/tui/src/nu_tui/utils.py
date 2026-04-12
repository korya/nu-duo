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
from typing import TYPE_CHECKING

from rich.text import Text as RichText

if TYPE_CHECKING:
    from collections.abc import Callable

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


# =============================================================================
# strip_ansi
# =============================================================================


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from *text* and return the plain string.

    Uses the already-compiled :data:`ANSI_ESCAPE_RE` pattern so there is no
    regex compilation overhead at call time.

    Mirrors the ``stripped`` construction inside upstream ``visibleWidth``.
    """
    return ANSI_ESCAPE_RE.sub("", text)


# =============================================================================
# _AnsiCodeTracker — preserves active SGR state across line breaks
# =============================================================================


class _AnsiCodeTracker:
    """Track active ANSI SGR codes so styles can be re-applied after wrapping.

    Port of the ``AnsiCodeTracker`` class in upstream ``utils.ts``.
    """

    __slots__ = (
        "_bg_color",
        "_blink",
        "_bold",
        "_dim",
        "_fg_color",
        "_hidden",
        "_inverse",
        "_italic",
        "_strikethrough",
        "_underline",
    )

    _SGR_RE = re.compile(r"\x1b\[([\d;]*)m")

    def __init__(self) -> None:
        self._bold = False
        self._dim = False
        self._italic = False
        self._underline = False
        self._blink = False
        self._inverse = False
        self._hidden = False
        self._strikethrough = False
        self._fg_color: str | None = None
        self._bg_color: str | None = None

    def clear(self) -> None:
        """Reset all tracked state."""
        self._bold = False
        self._dim = False
        self._italic = False
        self._underline = False
        self._blink = False
        self._inverse = False
        self._hidden = False
        self._strikethrough = False
        self._fg_color = None
        self._bg_color = None

    def process(self, ansi_code: str) -> None:
        """Update internal state from a single ANSI escape *ansi_code*."""
        if not ansi_code.endswith("m"):
            return
        m = self._SGR_RE.match(ansi_code)
        if not m:
            return
        params = m.group(1)
        if params in ("", "0"):
            self.clear()
            return
        parts = params.split(";")
        i = 0
        while i < len(parts):
            try:
                code = int(parts[i])
            except ValueError:
                i += 1
                continue
            if code in (38, 48):
                if i + 2 < len(parts) and parts[i + 1] == "5":
                    color_code = f"{parts[i]};{parts[i + 1]};{parts[i + 2]}"
                    if code == 38:
                        self._fg_color = color_code
                    else:
                        self._bg_color = color_code
                    i += 3
                    continue
                elif i + 4 < len(parts) and parts[i + 1] == "2":
                    color_code = ";".join(parts[i : i + 5])
                    if code == 38:
                        self._fg_color = color_code
                    else:
                        self._bg_color = color_code
                    i += 5
                    continue
            _map: dict[int, object] = {
                0: None,  # handled above
                1: ("_bold", True),
                2: ("_dim", True),
                3: ("_italic", True),
                4: ("_underline", True),
                5: ("_blink", True),
                7: ("_inverse", True),
                8: ("_hidden", True),
                9: ("_strikethrough", True),
                21: ("_bold", False),
                22: [("_bold", False), ("_dim", False)],
                23: ("_italic", False),
                24: ("_underline", False),
                25: ("_blink", False),
                27: ("_inverse", False),
                28: ("_hidden", False),
                29: ("_strikethrough", False),
                39: ("_fg_color", None),
                49: ("_bg_color", None),
            }
            if code in _map:
                action = _map[code]
                if isinstance(action, list):
                    for attr, val in action:
                        setattr(self, attr, val)
                elif action is not None:
                    attr, val = action  # type: ignore[misc]
                    setattr(self, attr, val)
            elif 30 <= code <= 37 or 90 <= code <= 97:
                self._fg_color = str(code)
            elif 40 <= code <= 47 or 100 <= code <= 107:
                self._bg_color = str(code)
            i += 1

    def get_active_codes(self) -> str:
        """Return the minimal CSI sequence to restore current style state."""
        codes: list[str] = []
        if self._bold:
            codes.append("1")
        if self._dim:
            codes.append("2")
        if self._italic:
            codes.append("3")
        if self._underline:
            codes.append("4")
        if self._blink:
            codes.append("5")
        if self._inverse:
            codes.append("7")
        if self._hidden:
            codes.append("8")
        if self._strikethrough:
            codes.append("9")
        if self._fg_color:
            codes.append(self._fg_color)
        if self._bg_color:
            codes.append(self._bg_color)
        if not codes:
            return ""
        return f"\x1b[{';'.join(codes)}m"

    def get_line_end_reset(self) -> str:
        """Return an underline-reset if underline is active (prevents bleed into padding)."""
        if self._underline:
            return "\x1b[24m"
        return ""

    def has_active_codes(self) -> bool:
        return bool(
            self._bold
            or self._dim
            or self._italic
            or self._underline
            or self._blink
            or self._inverse
            or self._hidden
            or self._strikethrough
            or self._fg_color
            or self._bg_color
        )


# =============================================================================
# Internal helpers for wrap_text_with_ansi
# =============================================================================

_ANSI_CODE_RE = re.compile(
    r"\x1b(?:"
    r"\[[0-9;]*[A-Za-z]"  # CSI
    r"|\][^\x07]*\x07"  # OSC (BEL terminated)
    r"|_[^\x07]*\x07"  # APC (BEL terminated)
    r"|."  # Other two-char sequences
    r")"
)


def _update_tracker(text: str, tracker: _AnsiCodeTracker) -> None:
    """Feed all ANSI SGR codes found in *text* into *tracker*."""
    for m in _ANSI_CODE_RE.finditer(text):
        tracker.process(m.group(0))


def _split_into_tokens(text: str) -> list[str]:
    """Split *text* into word/space tokens while keeping ANSI codes attached.

    Port of upstream ``splitIntoTokensWithAnsi``.
    """
    tokens: list[str] = []
    current: list[str] = []
    pending_ansi: list[str] = []
    in_whitespace = False
    i = 0
    while i < len(text):
        m = _ANSI_CODE_RE.match(text, i)
        if m:
            pending_ansi.append(m.group(0))
            i += len(m.group(0))
            continue
        char = text[i]
        char_is_space = char == " "
        if char_is_space != in_whitespace and current:
            tokens.append("".join(current))
            current = []
        if pending_ansi:
            current.extend(pending_ansi)
            pending_ansi = []
        in_whitespace = char_is_space
        current.append(char)
        i += 1
    if pending_ansi:
        current.extend(pending_ansi)
    if current:
        tokens.append("".join(current))
    return tokens


def _break_long_word(word: str, width: int, tracker: _AnsiCodeTracker) -> list[str]:
    """Break a token that is wider than *width* into lines."""
    lines: list[str] = []
    current = tracker.get_active_codes()
    current_w = 0
    i = 0
    while i < len(word):
        m = _ANSI_CODE_RE.match(word, i)
        if m:
            current += m.group(0)
            tracker.process(m.group(0))
            i += len(m.group(0))
            continue
        ch = word[i]
        ch_w = visible_width(ch)
        if current_w + ch_w > width:
            end_reset = tracker.get_line_end_reset()
            lines.append(current + end_reset)
            current = tracker.get_active_codes()
            current_w = 0
        current += ch
        current_w += ch_w
        i += 1
    if current:
        lines.append(current)
    return lines if lines else [""]


def _wrap_single_line(line: str, width: int) -> list[str]:
    """Wrap a single line of text (no embedded newlines) to *width* columns."""
    if not line:
        return [""]
    if visible_width(line) <= width:
        return [line]

    wrapped: list[str] = []
    tracker = _AnsiCodeTracker()
    tokens = _split_into_tokens(line)
    current = ""
    current_len = 0

    for token in tokens:
        token_len = visible_width(token)
        is_whitespace = not token.strip()

        if token_len > width and not is_whitespace:
            if current:
                end_reset = tracker.get_line_end_reset()
                wrapped.append(current + end_reset)
                current = ""
                current_len = 0
            broken = _break_long_word(token, width, tracker)
            wrapped.extend(broken[:-1])
            current = broken[-1]
            current_len = visible_width(current)
            continue

        if current_len + token_len > width and current_len > 0:
            line_to_wrap = current.rstrip()
            end_reset = tracker.get_line_end_reset()
            wrapped.append(line_to_wrap + end_reset)
            if is_whitespace:
                current = tracker.get_active_codes()
                current_len = 0
            else:
                current = tracker.get_active_codes() + token
                current_len = token_len
        else:
            current += token
            current_len += token_len

        _update_tracker(token, tracker)

    if current:
        wrapped.append(current)

    return [ln.rstrip() for ln in wrapped] if wrapped else [""]


# =============================================================================
# Public API additions
# =============================================================================


def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    """Word-wrap *text* to *width* columns, preserving ANSI escape codes.

    Only does word wrapping — no padding, no background colours.  Returns
    a list of lines where each line is at most *width* visible columns.
    Active ANSI formatting codes are re-applied at the start of each new
    line so styles carry over across breaks.

    Mirrors upstream ``wrapTextWithAnsi`` in ``utils.ts``.
    """
    if not text:
        return [""]

    input_lines = text.split("\n")
    result: list[str] = []
    tracker = _AnsiCodeTracker()

    for input_line in input_lines:
        prefix = tracker.get_active_codes() if result else ""
        result.extend(_wrap_single_line(prefix + input_line, width))
        _update_tracker(input_line, tracker)

    return result if result else [""]


def apply_background_to_line(
    line: str,
    width: int,
    bg_fn: Callable[[str], str],
) -> str:
    """Apply a background colour function to *line*, padding it to *width*.

    Calculates the visible width of *line*, pads with spaces to *width*,
    then passes the result through *bg_fn*. Mirrors upstream
    ``applyBackgroundToLine`` in ``utils.ts``.

    Example::

        apply_background_to_line("hello", 20, lambda t: f"\\x1b[44m{t}\\x1b[0m")
    """
    visible_len = visible_width(line)
    padding = " " * max(0, width - visible_len)
    return bg_fn(line + padding)


__all__ = [
    "ANSI_ESCAPE_RE",
    "CURSOR_MARKER",
    "apply_background_to_line",
    "decode_kitty_printable",
    "is_punctuation_char",
    "is_whitespace_char",
    "slice_by_column",
    "strip_ansi",
    "truncate_to_width",
    "visible_width",
    "wrap_text_with_ansi",
]
