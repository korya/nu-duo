"""``Text`` — port of ``packages/tui/src/components/text.ts``.

Renders a string into output lines that fit the supplied viewport
width. The Python port follows upstream's behaviour:

* Hard newlines in the source string become row boundaries.
* Lines longer than ``width`` columns are word-wrapped (no mid-word
  breaks unless a single word is longer than the width, in which
  case the word is hard-broken).
* Empty source strings produce an empty list (zero output lines).

ANSI escape sequences in the source are preserved verbatim — the
upstream renderer treats them as zero-width and we do the same.
Wrapping is computed against the **visible width** of the line,
which Rich's :class:`rich.text.Text` already exposes via
``cell_len``; using it keeps width calculations consistent with how
Textual will eventually paint the result.
"""

from __future__ import annotations

from rich.text import Text as RichText

from nu_tui.component import Component


def cell_len(value: str) -> int:
    """Return the visible cell width of ``value``.

    Thin wrapper around :class:`rich.text.Text` so the rest of the
    module doesn't have to construct a Rich Text object inline.
    Tests use this for assertions about wrapping behaviour.
    """
    return RichText(value).cell_len


def _wrap_line(line: str, width: int) -> list[str]:
    """Word-wrap a single line to ``width`` columns.

    Mirrors upstream ``Text``'s wrap algorithm: split on spaces,
    rebuild words greedily until the next word would exceed the
    width, then start a new line. Words longer than ``width`` are
    hard-broken into ``width``-column chunks (slow path; matches
    upstream).
    """
    if cell_len(line) <= width:
        return [line]

    words = line.split(" ")
    out: list[str] = []
    current = ""
    current_width = 0
    for word in words:
        word_width = cell_len(word)
        if word_width > width:
            # Flush whatever we have, then hard-break the word.
            if current:
                out.append(current)
                current = ""
                current_width = 0
            for i in range(0, len(word), width):
                chunk = word[i : i + width]
                if cell_len(chunk) == width:
                    out.append(chunk)
                else:
                    current = chunk
                    current_width = cell_len(chunk)
            continue
        # Try to append the word to the current line.
        if not current:
            current = word
            current_width = word_width
        elif current_width + 1 + word_width <= width:
            current = f"{current} {word}"
            current_width += 1 + word_width
        else:
            out.append(current)
            current = word
            current_width = word_width
    if current:
        out.append(current)
    return out


class Text(Component):
    """Render a (possibly multi-line) string with word-wrapping.

    The constructor accepts an optional ``text`` value (default
    empty). :meth:`set_text` updates the value and invalidates any
    cached layout. ``render`` is pure with respect to ``width``
    (each call recomputes the wrap) so a window resize automatically
    reflows the content without an explicit ``invalidate``.
    """

    def __init__(self, text: str = "") -> None:
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        """Replace the displayed text. Equivalent to assigning :attr:`text`."""
        self._text = text

    def render(self, width: int) -> list[str]:
        if not self._text:
            return []
        if width <= 0:
            return [self._text]
        out: list[str] = []
        for raw_line in self._text.split("\n"):
            out.extend(_wrap_line(raw_line, width))
        return out


__all__ = ["Text"]
