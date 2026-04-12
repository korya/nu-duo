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
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-9;]*[A-Za-z]|\][^\x07]*\x07|_[^\x07]*\x07|.)")


def visible_width(text: str) -> int:
    """Return the visible cell width of ``text``, ignoring ANSI escapes.

    Delegates to :class:`rich.text.Text` for Unicode/CJK width
    computation after stripping ANSI control sequences that Rich
    would otherwise count as zero-width.
    """
    stripped = _ANSI_ESCAPE_RE.sub("", text)
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


__all__ = [
    "truncate_to_width",
    "visible_width",
]
