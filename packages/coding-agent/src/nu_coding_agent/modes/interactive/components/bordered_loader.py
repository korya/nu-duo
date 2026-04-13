"""Bordered loader widget — port of ``bordered-loader.ts``.

A ``LoadingIndicator`` wrapped in a visual border, used by extension UI
and the login dialog.  Optionally cancellable via an ``AbortController``
analogue (a threading ``Event``).
"""

from __future__ import annotations

import threading

from rich.text import Text as RichText
from textual.widgets import LoadingIndicator, Static

from nu_coding_agent.modes.interactive.components.keybinding_hints import key_hint


class BorderedLoader(Static):
    """Loading indicator with decorative borders and optional cancel support.

    Port of ``BorderedLoader`` (66 LoC).  In the Textual port we use
    ``LoadingIndicator`` for the spinner and a ``Static`` for the border
    lines, composed into a single ``Static`` using Rich ``Text``.

    ``cancel_event`` (if provided) is set when the user presses Escape or
    calls :meth:`cancel` programmatically — analogous to the upstream's
    ``AbortController.signal``.
    """

    DEFAULT_CSS = """
    BorderedLoader {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        message: str,
        *,
        cancellable: bool = True,
    ) -> None:
        super().__init__("")
        self._message = message
        self._cancellable = cancellable
        self._cancel_event = threading.Event() if cancellable else None
        self._cancelled = False
        self._update_display()

    # ------------------------------------------------------------------

    @property
    def cancel_event(self) -> threading.Event | None:
        """Event that is set when the user cancels the operation."""
        return self._cancel_event

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        """Programmatically cancel the operation."""
        self._cancelled = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._update_display()

    def on_key(self, event: object) -> None:
        """Cancel on Escape."""
        key = getattr(event, "key", None)
        if key == "escape" and self._cancellable and not self._cancelled:
            self.cancel()
            if hasattr(event, "stop"):
                event.stop()  # type: ignore[union-attr]

    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        border = "─" * 40
        t = RichText()
        t.append(border + "\n", style="dim")
        if self._cancelled:
            t.append("(cancelled)", style="yellow")
        else:
            t.append(f"⟳ {self._message}", style="bold")
            if self._cancellable:
                hint = key_hint("tui.select.cancel", "to cancel")
                t.append(f"\n{hint}", style="dim")
        t.append("\n" + border, style="dim")
        self.update(t)


class BorderedLoaderWithSpinner(Static):
    """Variant that mounts a Textual ``LoadingIndicator`` for animation.

    Use this when you need the live spinner; ``BorderedLoader`` is fine
    for static tests and SSR contexts.
    """

    DEFAULT_CSS = """
    BorderedLoaderWithSpinner {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, message: str, *, cancellable: bool = True) -> None:
        super().__init__("")
        self._message = message
        self._cancellable = cancellable
        self._cancel_event = threading.Event() if cancellable else None
        self._cancelled = False

    def compose(self):  # type: ignore[override]
        yield LoadingIndicator()

    def on_mount(self) -> None:
        border = "─" * 40
        header = RichText()
        header.append(border + "\n", style="dim")
        header.append(self._message, style="bold")
        if self._cancellable:
            hint = key_hint("tui.select.cancel", "to cancel")
            header.append(f"\n{hint}", style="dim")
        self.update(header)

    def cancel(self) -> None:
        self._cancelled = True
        if self._cancel_event is not None:
            self._cancel_event.set()


__all__ = ["BorderedLoader", "BorderedLoaderWithSpinner"]
