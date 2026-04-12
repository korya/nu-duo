"""``Spacer`` — port of ``packages/tui/src/components/spacer.ts``.

Renders ``height`` empty lines. Used for vertical spacing between
other components in a :class:`Container`. The empty string is the
right rendering at every width because :meth:`Component.render`
returns one entry per row, and an empty entry is a blank row.
"""

from __future__ import annotations

from nu_tui.component import Component


class Spacer(Component):
    """Vertical spacer — emits ``height`` empty lines.

    Default ``height=1`` matches upstream ``Spacer``. The constructor
    accepts a positive integer; values less than 1 are coerced to 1
    so a misconfigured Spacer still renders something visible.
    """

    def __init__(self, height: int = 1) -> None:
        self.height = max(1, int(height))

    def render(self, width: int) -> list[str]:
        return [""] * self.height


__all__ = ["Spacer"]
