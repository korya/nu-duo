"""``Markdown`` — port of ``packages/tui/src/components/markdown.ts``.

The upstream component (824 LoC) hand-rolls a markdown-to-ANSI
renderer using the ``marked`` library's token stream. The Python port
takes a pragmatic shortcut: it delegates to
`Rich <https://rich.readthedocs.io/>`_'s built-in ``Markdown``
renderable, which already handles headings, code blocks (with syntax
highlighting via Pygments), lists, bold, italic, links, blockquotes,
horizontal rules, and tables. The result is captured into a string
buffer via ``Console(record=True)`` and split into terminal lines.

This means the visual output won't be pixel-identical to upstream,
but it's functionally richer (Rich handles many edge cases the
hand-rolled renderer doesn't) and much less code. The porting plan
explicitly called nu_tui out as the one place we deviate at
*implementation* level; this is one of those deviations.

The component still exposes the same ``render(width) → list[str]``
contract and the same constructor shape (text, padding, theme) so
consumers don't need to know which renderer is behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

from nu_tui.component import Component

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class MarkdownTheme:
    """Styling callbacks for markdown elements.

    Each callable takes a plain string and returns a styled version
    (typically by wrapping in ANSI escapes). This theme is used for
    the few places where the Rich renderer's output needs
    post-processing or where the component adds its own chrome
    (padding, background). For the actual markdown element styling,
    Rich uses its own theme system.
    """

    heading: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    code: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    code_block: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    code_block_border: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    link: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    bold: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    italic: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    quote: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    hr: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    list_bullet: Callable[[str], str] = field(default_factory=lambda: lambda s: s)


def default_markdown_theme() -> MarkdownTheme:
    """Build a passthrough theme (no styling beyond what Rich provides)."""
    return MarkdownTheme()


class Markdown(Component):
    """Render markdown text to terminal lines using Rich.

    The constructor accepts the raw markdown source, padding, and an
    optional :class:`MarkdownTheme`. ``render(width)`` produces a
    cached list of ANSI-styled strings — the cache is invalidated
    when either the text or the width changes.
    """

    def __init__(
        self,
        text: str = "",
        padding_x: int = 0,
        padding_y: int = 0,
        theme: MarkdownTheme | None = None,
        **_kwargs: Any,
    ) -> None:
        self._text = text
        self.padding_x = padding_x
        self.padding_y = padding_y
        self._theme = theme or default_markdown_theme()
        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        """Replace the markdown source and invalidate the cache."""
        self._text = text
        self._cached_lines = None
        self._cached_text = None

    def invalidate(self) -> None:
        self._cached_lines = None
        self._cached_text = None
        self._cached_width = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_text == self._text and self._cached_width == width:
            return self._cached_lines

        inner_width = max(1, width - self.padding_x * 2)
        left_pad = " " * self.padding_x

        if not self._text:
            self._cached_lines = []
            self._cached_text = self._text
            self._cached_width = width
            return self._cached_lines

        # Use Rich to render markdown to a string buffer.
        console = Console(
            width=inner_width,
            record=True,
            force_terminal=True,
            force_jupyter=False,
            no_color=False,
        )
        md = RichMarkdown(self._text, code_theme="monokai")
        console.print(md, width=inner_width)
        raw = console.export_text(styles=True)

        # Split into lines and apply horizontal padding.
        content_lines = raw.rstrip("\n").split("\n") if raw.strip() else []

        result: list[str] = []

        # Vertical padding above
        for _ in range(self.padding_y):
            result.append(" " * width)

        for line in content_lines:
            result.append(f"{left_pad}{line}")

        # Vertical padding below
        for _ in range(self.padding_y):
            result.append(" " * width)

        self._cached_lines = result
        self._cached_text = self._text
        self._cached_width = width
        return result


__all__ = ["Markdown", "MarkdownTheme", "default_markdown_theme"]
