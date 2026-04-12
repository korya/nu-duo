"""``Image`` — port of ``packages/tui/src/components/image.ts``.

Renders an image inline in the terminal using the Kitty or iTerm2
graphics protocols. Falls back to a placeholder text if neither
protocol is supported.
"""

from __future__ import annotations

from nu_tui.component import Component
from nu_tui.terminal_image import render_image, supports_inline_images


class Image(Component):
    """Render an image inline in the terminal.

    ``image_data`` is raw image bytes (PNG preferred). The component
    auto-detects Kitty/iTerm2 support and falls back to a text
    placeholder.
    """

    def __init__(
        self,
        image_data: bytes = b"",
        *,
        alt_text: str = "[image]",
        columns: int | None = None,
        rows: int | None = None,
    ) -> None:
        self._image_data = image_data
        self._alt_text = alt_text
        self._columns = columns
        self._rows = rows

    def set_image(self, image_data: bytes) -> None:
        self._image_data = image_data

    def render(self, width: int) -> list[str]:
        if not self._image_data:
            return [self._alt_text] if self._alt_text else []

        if not supports_inline_images():
            return [self._alt_text]

        lines = render_image(
            self._image_data,
            columns=self._columns or min(width, 80),
            rows=self._rows,
        )
        return lines if lines else [self._alt_text]


__all__ = ["Image"]
