"""Overlay system — port of the overlay surface in ``packages/tui/src/tui.ts``.

The upstream TUI (1243 LoC) includes an overlay system for modal
dialogs and popups (117 overlay-related lines). In the Python port,
Textual's ``ModalScreen`` handles this role for the interactive mode.
This module provides the **data types** (``OverlayOptions``,
``OverlayHandle``) so code that references the overlay API can be
ported structurally, even though the actual painting is done by
Textual.

The ``TUI.show_overlay`` / ``TUI.hide_overlay`` methods on the
nu_tui TUI class can delegate to these types in a future slice that
wants to drive overlays through the nu_tui rendering pipeline instead
of Textual's modal stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

type OverlayAnchor = Literal[
    "center",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "top-center",
    "bottom-center",
    "left-center",
    "right-center",
]

type SizeValue = int | str  # absolute pixels or percentage like "50%"


@dataclass(slots=True)
class OverlayMargin:
    """Margin from terminal edges."""

    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0


@dataclass(slots=True)
class OverlayOptions:
    """Options for overlay positioning and sizing.

    Mirrors TS ``OverlayOptions``. In the Python port these are consumed
    by the TUI class but the actual rendering is done by Textual.
    """

    width: SizeValue | None = None
    min_width: int | None = None
    max_height: SizeValue | None = None
    anchor: OverlayAnchor = "center"
    offset_x: int = 0
    offset_y: int = 0
    row: SizeValue | None = None
    col: SizeValue | None = None
    margin: OverlayMargin | int | None = None
    visible: Any = None  # Callable[[int, int], bool] | None
    non_capturing: bool = False


class OverlayHandle:
    """Handle returned by ``TUI.show_overlay`` for controlling the overlay.

    Mirrors TS ``OverlayHandle``. In the Textual-backed Python port,
    the real show/hide mechanics are handled by Textual's modal screen
    system; this handle provides the structural compatibility so
    extension code referencing ``OverlayHandle`` compiles without
    changes.
    """

    def __init__(self) -> None:
        self._hidden = False
        self._focused = False
        self._removed = False

    def hide(self) -> None:
        """Permanently remove the overlay (cannot be shown again)."""
        self._removed = True

    def set_hidden(self, hidden: bool) -> None:
        """Temporarily hide or show the overlay."""
        if not self._removed:
            self._hidden = hidden

    def is_hidden(self) -> bool:
        return self._hidden or self._removed

    def focus(self) -> None:
        """Focus this overlay."""
        self._focused = True

    def unfocus(self) -> None:
        """Release focus."""
        self._focused = False

    def is_focused(self) -> bool:
        return self._focused


__all__ = [
    "OverlayAnchor",
    "OverlayHandle",
    "OverlayMargin",
    "OverlayOptions",
    "SizeValue",
]
