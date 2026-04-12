"""``SelectList`` — port of ``packages/tui/src/components/select-list.ts``.

A vertical pick-list with keyboard navigation, filtering, and
scroll-window support. The upstream version uses the full keybinding
system for arrow/enter/escape mapping; this Python port accepts
simple key strings and delegates to the existing
:class:`nu_tui.keybindings.Keybindings` port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nu_tui.component import Component
from nu_tui.utils import truncate_to_width, visible_width

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFAULT_PRIMARY_COLUMN_WIDTH = 32
_PRIMARY_COLUMN_GAP = 2
_MIN_DESCRIPTION_WIDTH = 10


@dataclass(slots=True)
class SelectItem:
    """A single entry in a :class:`SelectList`."""

    value: str
    label: str
    description: str | None = None


@dataclass(slots=True)
class SelectListTheme:
    """Callable suite for styling the pick-list.

    Each callable takes a plain string and returns a styled version
    (typically by wrapping it in ANSI escape sequences). A no-op
    default theme is provided by :func:`default_select_list_theme`.
    """

    selected_prefix: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    selected_text: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    description: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    scroll_info: Callable[[str], str] = field(default_factory=lambda: lambda s: s)
    no_match: Callable[[str], str] = field(default_factory=lambda: lambda s: s)


def default_select_list_theme() -> SelectListTheme:
    """Build a passthrough theme (no styling)."""
    return SelectListTheme()


def _normalize_to_single_line(text: str) -> str:
    return " ".join(text.splitlines()).strip()


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


class SelectList(Component):
    """Scrollable vertical pick-list with filter support.

    The constructor accepts the full item list, visible-window size,
    and a theme. Filtering is applied via :meth:`set_filter`; the
    filtered subset is what :meth:`render` + :meth:`handle_input`
    operate on. Selection wraps at both ends (up from first → last,
    down from last → first).
    """

    def __init__(
        self,
        items: list[SelectItem],
        max_visible: int = 5,
        theme: SelectListTheme | None = None,
        *,
        min_primary_column_width: int | None = None,
        max_primary_column_width: int | None = None,
    ) -> None:
        self._items = list(items)
        self._filtered: list[SelectItem] = list(items)
        self._selected_index = 0
        self._max_visible = max_visible
        self._theme = theme or default_select_list_theme()
        self._min_primary_col = min_primary_column_width
        self._max_primary_col = max_primary_column_width

        # Callbacks set by the consumer.
        self.on_select: Callable[[SelectItem], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self.on_selection_change: Callable[[SelectItem], None] | None = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_filter(self, filter_text: str) -> None:
        lower = filter_text.lower()
        self._filtered = [item for item in self._items if item.value.lower().startswith(lower)]
        self._selected_index = 0

    def set_selected_index(self, index: int) -> None:
        self._selected_index = _clamp(index, 0, max(0, len(self._filtered) - 1))

    def get_selected_item(self) -> SelectItem | None:
        if 0 <= self._selected_index < len(self._filtered):
            return self._filtered[self._selected_index]
        return None

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @property
    def filtered_items(self) -> list[SelectItem]:
        return list(self._filtered)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        lines: list[str] = []

        if not self._filtered:
            lines.append(self._theme.no_match("  No matching commands"))
            return lines

        primary_col_width = self._get_primary_column_width()

        # Visible window with scroll
        start = max(
            0,
            min(
                self._selected_index - self._max_visible // 2,
                len(self._filtered) - self._max_visible,
            ),
        )
        end = min(start + self._max_visible, len(self._filtered))

        for i in range(start, end):
            item = self._filtered[i]
            is_selected = i == self._selected_index
            desc_line = _normalize_to_single_line(item.description) if item.description else None
            lines.append(self._render_item(item, is_selected, width, desc_line, primary_col_width))

        # Scroll indicator
        if start > 0 or end < len(self._filtered):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._filtered)})"
            lines.append(self._theme.scroll_info(truncate_to_width(scroll_text, width - 2, "")))

        return lines

    # ------------------------------------------------------------------
    # Keyboard input
    # ------------------------------------------------------------------

    def handle_input(self, data: str) -> None:
        from nu_tui.keybindings import get_keybindings  # noqa: PLC0415

        kb = get_keybindings()

        if kb.matches(data, "tui.select.up"):
            self._selected_index = len(self._filtered) - 1 if self._selected_index == 0 else self._selected_index - 1
            self._notify_selection_change()
        elif kb.matches(data, "tui.select.down"):
            self._selected_index = 0 if self._selected_index == len(self._filtered) - 1 else self._selected_index + 1
            self._notify_selection_change()
        elif kb.matches(data, "tui.select.confirm"):
            item = self.get_selected_item()
            if item and self.on_select:
                self.on_select(item)
        elif kb.matches(data, "tui.select.cancel"):
            if self.on_cancel:
                self.on_cancel()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_primary_column_width(self) -> int:
        raw_min = self._min_primary_col or self._max_primary_col or _DEFAULT_PRIMARY_COLUMN_WIDTH
        raw_max = self._max_primary_col or self._min_primary_col or _DEFAULT_PRIMARY_COLUMN_WIDTH
        lo = max(1, min(raw_min, raw_max))
        hi = max(1, raw_min, raw_max)

        widest = 0
        for item in self._filtered:
            display = item.label or item.value
            widest = max(widest, visible_width(display) + _PRIMARY_COLUMN_GAP)
        return _clamp(widest, lo, hi)

    def _render_item(
        self,
        item: SelectItem,
        is_selected: bool,
        width: int,
        desc_single_line: str | None,
        primary_col_width: int,
    ) -> str:
        prefix = "→ " if is_selected else "  "
        prefix_width = visible_width(prefix)
        display_value = item.label or item.value

        if desc_single_line and width > 40:
            effective_primary = max(1, min(primary_col_width, width - prefix_width - 4))
            max_primary = max(1, effective_primary - _PRIMARY_COLUMN_GAP)
            truncated_value = truncate_to_width(display_value, max_primary, "")
            trunc_width = visible_width(truncated_value)
            spacing = " " * max(1, effective_primary - trunc_width)
            desc_start = prefix_width + trunc_width + len(spacing)
            remaining = width - desc_start - 2

            if remaining > _MIN_DESCRIPTION_WIDTH:
                truncated_desc = truncate_to_width(desc_single_line, remaining, "")
                if is_selected:
                    return self._theme.selected_text(f"{prefix}{truncated_value}{spacing}{truncated_desc}")
                return f"{prefix}{truncated_value}{self._theme.description(spacing + truncated_desc)}"

        max_width = width - prefix_width - 2
        truncated_value = truncate_to_width(display_value, max_width, "")
        if is_selected:
            return self._theme.selected_text(f"{prefix}{truncated_value}")
        return f"{prefix}{truncated_value}"

    def _notify_selection_change(self) -> None:
        item = self.get_selected_item()
        if item and self.on_selection_change:
            self.on_selection_change(item)


__all__ = ["SelectItem", "SelectList", "SelectListTheme", "default_select_list_theme"]
