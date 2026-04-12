"""``SettingsList`` — port of ``packages/tui/src/components/settings-list.ts``.

A two-column settings panel with label/value rows, optional per-item
descriptions, keyboard navigation, and a pluggable theme.  The upstream
version also supports a search-filter ``Input`` and sub-menu components;
this Python port covers the core list behaviour consumed by Phase 5 consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nu_tui.component import Component
from nu_tui.utils import truncate_to_width, visible_width, wrap_text_with_ansi

if TYPE_CHECKING:
    from collections.abc import Callable

# Maximum label column width before we stop growing it.
_MAX_LABEL_WIDTH = 30


@dataclass(slots=True)
class SettingsItem:
    """A single row in a :class:`SettingsList`.

    Attributes:
        label: Human-readable name shown in the left column.
        value: Current value shown in the right column.
        description: Optional multi-line description displayed below the
            list when this item is selected.
    """

    label: str
    value: str
    description: str | None = None


@dataclass(slots=True)
class SettingsListTheme:
    """Callable suite for styling a :class:`SettingsList`.

    Each callable accepts a plain string and returns a styled version
    (typically by wrapping it in ANSI escape sequences).  A passthrough
    default theme is provided by :func:`default_settings_list_theme`.
    """

    label: Callable[[str, bool], str] = field(default_factory=lambda: lambda text, _selected: text)
    """Style the label text.  Receives ``(text, is_selected)``."""

    value: Callable[[str, bool], str] = field(default_factory=lambda: lambda text, _selected: text)
    """Style the value text.  Receives ``(text, is_selected)``."""

    description: Callable[[str], str] = field(default_factory=lambda: lambda text: text)
    """Style a description line."""

    cursor: str = "→ "
    """Prefix string used on the selected row (plain text, no ANSI needed)."""

    hint: Callable[[str], str] = field(default_factory=lambda: lambda text: text)
    """Style the hint / scroll-indicator line."""


def default_settings_list_theme() -> SettingsListTheme:
    """Build a no-op passthrough theme."""
    return SettingsListTheme()


class SettingsList(Component):
    """Scrollable settings panel with label/value columns.

    The list centres its scroll window around the selected item and
    mirrors the upstream two-column layout.  Navigation wraps at both
    ends.  An :attr:`on_select` callback is invoked when the user
    confirms a selection.

    Example::

        items = [
            SettingsItem("Theme", "dark", "Choose colour scheme"),
            SettingsItem("Font size", "14"),
        ]
        sl = SettingsList(items, max_visible=5)
        sl.on_select = lambda item: print(item.label, "->", item.value)
        lines = sl.render(80)
    """

    def __init__(
        self,
        items: list[SettingsItem],
        max_visible: int = 8,
        theme: SettingsListTheme | None = None,
    ) -> None:
        self._items: list[SettingsItem] = list(items)
        self._selected_index: int = 0
        self._max_visible: int = max_visible
        self._theme: SettingsListTheme = theme or default_settings_list_theme()

        # Callbacks set by the consumer.
        self.on_select: Callable[[SettingsItem], None] | None = None
        self.on_cancel: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_items(self, items: list[SettingsItem]) -> None:
        """Replace the full item list.

        The selected index is clamped so it remains valid after the
        update, but does *not* automatically jump to 0.
        """
        self._items = list(items)
        self._selected_index = max(0, min(self._selected_index, len(self._items) - 1))

    @property
    def selected_index(self) -> int:
        """Index of the currently highlighted item."""
        return self._selected_index

    def get_selected_item(self) -> SettingsItem | None:
        """Return the highlighted :class:`SettingsItem`, or ``None`` if empty."""
        if 0 <= self._selected_index < len(self._items):
            return self._items[self._selected_index]
        return None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """Render the settings list into a list of fixed-width strings.

        Args:
            width: Available terminal column width.

        Returns:
            List of strings (one per display row), none containing a
            newline character.
        """
        lines: list[str] = []

        if not self._items:
            lines.append(self._theme.hint("  No settings available"))
            lines.append("")
            lines.append(
                truncate_to_width(
                    self._theme.hint("  Enter/Space to change · Esc to cancel"),
                    width,
                )
            )
            return lines

        # Scroll window centred on selected item
        start = max(
            0,
            min(
                self._selected_index - self._max_visible // 2,
                len(self._items) - self._max_visible,
            ),
        )
        end = min(start + self._max_visible, len(self._items))

        # Compute label column width (capped at _MAX_LABEL_WIDTH)
        max_label_w = min(
            _MAX_LABEL_WIDTH,
            max((visible_width(item.label) for item in self._items), default=0),
        )

        cursor_text = self._theme.cursor
        cursor_w = visible_width(cursor_text)
        indent = " " * cursor_w  # same width as cursor, used on non-selected rows

        separator = "  "
        sep_w = visible_width(separator)
        value_max_w = max(0, width - cursor_w - max_label_w - sep_w - 2)

        for i in range(start, end):
            item = self._items[i]
            is_selected = i == self._selected_index

            prefix = cursor_text if is_selected else indent

            # Pad label to align value column
            pad = " " * max(0, max_label_w - visible_width(item.label))
            label_text = self._theme.label(item.label + pad, is_selected)

            value_truncated = truncate_to_width(item.value, value_max_w, "")
            value_text = self._theme.value(value_truncated, is_selected)

            row = truncate_to_width(
                prefix + label_text + separator + value_text,
                width,
            )
            lines.append(row)

        # Scroll indicator when list is taller than the window
        if start > 0 or end < len(self._items):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._items)})"
            lines.append(self._theme.hint(truncate_to_width(scroll_text, width - 2, "")))

        # Description for the selected item
        selected = self.get_selected_item()
        if selected and selected.description:
            lines.append("")
            for desc_line in wrap_text_with_ansi(selected.description, width - 4):
                lines.append(self._theme.description(f"  {desc_line}"))

        # Hint line
        lines.append("")
        lines.append(
            truncate_to_width(
                self._theme.hint("  Enter/Space to change · Esc to cancel"),
                width,
            )
        )

        return lines

    # ------------------------------------------------------------------
    # Keyboard input
    # ------------------------------------------------------------------

    def handle_input(self, data: str) -> None:
        """Handle a raw terminal input event.

        Supports up/down navigation (wrapping), enter/space to select,
        and escape to cancel.  Delegates to :mod:`nu_tui.keybindings`
        for the concrete key bindings so they stay in sync with the rest
        of the TUI.

        Args:
            data: Raw bytes/string received from the terminal.
        """
        from nu_tui.keybindings import get_keybindings  # noqa: PLC0415

        kb = get_keybindings()
        n = len(self._items)

        if kb.matches(data, "tui.select.up"):
            if n:
                self._selected_index = (self._selected_index - 1) % n
        elif kb.matches(data, "tui.select.down"):
            if n:
                self._selected_index = (self._selected_index + 1) % n
        elif kb.matches(data, "tui.select.confirm") or data == " ":
            item = self.get_selected_item()
            if item and self.on_select:
                self.on_select(item)
        elif kb.matches(data, "tui.select.cancel") and self.on_cancel:
            self.on_cancel()


__all__ = [
    "SettingsItem",
    "SettingsList",
    "SettingsListTheme",
    "default_settings_list_theme",
]
