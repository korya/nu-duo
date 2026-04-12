"""``Box`` — port of ``packages/tui/src/components/box.ts``.

A container that applies horizontal/vertical padding and an optional
background function to every rendered child line. The background
function takes a string and returns a styled variant (e.g.
``lambda s: f"\\033[44m{s}\\033[0m"`` for a blue background). When no
background function is set, the box just pads lines to ``width``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from nu_tui.component import Component
from nu_tui.utils import visible_width

if TYPE_CHECKING:
    from collections.abc import Callable


class Box(Component):
    """Padded container with optional background styling.

    ``padding_x`` adds space on both left and right of every child
    line. ``padding_y`` adds blank lines above and below the content.
    ``bg_fn`` is an optional callable that wraps an entire line
    string with background escapes.

    :meth:`render` caches the last output and reuses it when neither
    the width nor the child output has changed (matching upstream's
    caching behaviour).
    """

    def __init__(
        self,
        padding_x: int = 1,
        padding_y: int = 1,
        bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.children: list[Component] = []
        self.padding_x = padding_x
        self.padding_y = padding_y
        self._bg_fn = bg_fn
        self._cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Child management (mirrors Container but Box is its own root)
    # ------------------------------------------------------------------

    def add_child(self, component: Component) -> None:
        self.children.append(component)
        self._cache = None

    def remove_child(self, component: Component) -> None:
        with contextlib.suppress(ValueError):
            self.children.remove(component)
        self._cache = None

    def clear(self) -> None:
        self.children = []
        self._cache = None

    def set_bg_fn(self, bg_fn: Callable[[str], str] | None = None) -> None:
        self._bg_fn = bg_fn
        # Don't invalidate cache here — we detect bg_fn changes by
        # sampling output (matching upstream behaviour).

    def invalidate(self) -> None:
        self._cache = None
        for child in self.children:
            child.invalidate()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        if not self.children:
            return []

        content_width = max(1, width - self.padding_x * 2)
        left_pad = " " * self.padding_x

        child_lines: list[str] = []
        for child in self.children:
            for line in child.render(content_width):
                child_lines.append(f"{left_pad}{line}")

        if not child_lines:
            return []

        # Cache check (matching upstream: compare child_lines + width + bg sample)
        bg_sample = self._bg_fn("test") if self._bg_fn else None
        if self._cache is not None:
            c = self._cache
            if c["width"] == width and c["bg_sample"] == bg_sample and c["child_lines"] == child_lines:
                return c["lines"]

        result: list[str] = []

        # Top padding
        for _ in range(self.padding_y):
            result.append(self._apply_bg("", width))

        # Content
        for line in child_lines:
            result.append(self._apply_bg(line, width))

        # Bottom padding
        for _ in range(self.padding_y):
            result.append(self._apply_bg("", width))

        self._cache = {
            "child_lines": child_lines,
            "width": width,
            "bg_sample": bg_sample,
            "lines": result,
        }
        return result

    def _apply_bg(self, line: str, width: int) -> str:
        vis_len = visible_width(line)
        pad_needed = max(0, width - vis_len)
        padded = line + " " * pad_needed
        if self._bg_fn:
            return self._bg_fn(padded)
        return padded


__all__ = ["Box"]
