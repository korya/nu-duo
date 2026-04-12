"""``TruncatedText`` — port of ``packages/tui/src/components/truncated-text.ts``.

Renders a single line of text, truncated to fit the viewport width
with optional horizontal and vertical padding. Multi-line input is
clipped to the first line (matching upstream's ``indexOf("\\n")``
behaviour).
"""

from __future__ import annotations

from nu_tui.component import Component
from nu_tui.utils import truncate_to_width, visible_width


class TruncatedText(Component):
    """Single-line text with ellipsis truncation and optional padding.

    ``padding_x`` adds space on both left and right; ``padding_y``
    adds blank lines above and below the content row. The content
    row is padded with trailing spaces to exactly ``width`` columns
    so it fills its allocation in a Box or Container.
    """

    def __init__(self, text: str = "", padding_x: int = 0, padding_y: int = 0) -> None:
        self._text = text
        self.padding_x = padding_x
        self.padding_y = padding_y

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text

    def render(self, width: int) -> list[str]:
        result: list[str] = []
        empty_line = " " * width

        # Vertical padding above
        for _ in range(self.padding_y):
            result.append(empty_line)

        available_width = max(1, width - self.padding_x * 2)

        # Take only the first line
        single_line = self._text.split("\n", 1)[0]

        # Truncate
        display_text = truncate_to_width(single_line, available_width)

        # Horizontal padding
        left_pad = " " * self.padding_x
        right_pad = " " * self.padding_x
        line_with_padding = f"{left_pad}{display_text}{right_pad}"

        # Pad to exactly ``width``
        line_vis_width = visible_width(line_with_padding)
        pad_needed = max(0, width - line_vis_width)
        result.append(line_with_padding + " " * pad_needed)

        # Vertical padding below
        for _ in range(self.padding_y):
            result.append(empty_line)

        return result


__all__ = ["TruncatedText"]
