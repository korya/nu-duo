"""Terminal capability facade — port of ``packages/tui/src/terminal.ts``.

The upstream module probes the terminal for a long list of
capabilities (true color, mouse, Kitty keyboard protocol, image
protocols, hyperlinks, etc.) and exposes them as a structured
``Terminal`` object. The Python port wraps Textual's driver
detection — Textual already knows about most of these features and
exposes them via ``App.driver`` / ``App.console``. We expose a
small façade with the upstream method names so consumer code in
``nu_coding_agent.modes.interactive`` doesn't have to learn a new
API.

This is the foundation slice (5.1) — only the methods that the
``TUI`` wrapper actually consumes are implemented. The rest land in
follow-up slices alongside their consumers (image rendering needs
Kitty/iTerm2 protocol detection, etc.).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class TerminalSize:
    """Terminal dimensions in columns x rows."""

    columns: int
    rows: int


class Terminal:
    """Capability facade for the active terminal.

    Construct directly (``Terminal()``) for the running process's
    terminal, or pass an explicit :class:`TerminalSize` for tests
    that want a deterministic size without involving stdin/stdout.
    """

    def __init__(self, *, size: TerminalSize | None = None) -> None:
        self._fixed_size = size

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def get_size(self) -> TerminalSize:
        """Return the current terminal dimensions.

        Falls back to (80, 24) when stdout is not a TTY (e.g. CI),
        matching ``shutil.get_terminal_size``'s default contract.
        """
        if self._fixed_size is not None:
            return self._fixed_size
        size = shutil.get_terminal_size(fallback=(80, 24))
        return TerminalSize(columns=size.columns, rows=size.lines)

    def get_columns(self) -> int:
        return self.get_size().columns

    def get_rows(self) -> int:
        return self.get_size().rows

    # ------------------------------------------------------------------
    # Capability detection (subset)
    # ------------------------------------------------------------------

    def is_tty(self) -> bool:
        """``True`` iff stdout is connected to a terminal."""
        return sys.stdout.isatty()

    def supports_color(self) -> bool:
        """Best-effort detection of color capability.

        Honours the standard ``NO_COLOR`` opt-out plus the
        ``CLICOLOR`` / ``CLICOLOR_FORCE`` conventions. Defers to
        Textual's renderer for the actual ANSI emission — this
        method is consulted by code that wants to *decide* whether
        to emit color escapes at all (e.g. when piping to a file).
        """
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("CLICOLOR_FORCE"):
            return True
        return self.is_tty()

    def is_termux(self) -> bool:
        """``True`` iff running inside Termux on Android."""
        return bool(os.environ.get("TERMUX_VERSION"))

    # ------------------------------------------------------------------
    # Extended capability detection (Phase 5.10)
    # ------------------------------------------------------------------

    def supports_true_color(self) -> bool:
        """Best-effort detection of 24-bit true-color support.

        Checks ``COLORTERM=truecolor`` / ``COLORTERM=24bit`` env vars,
        plus terminal programs known to support true color.
        """
        colorterm = os.environ.get("COLORTERM", "").lower()
        if colorterm in ("truecolor", "24bit"):
            return True
        term_program = os.environ.get("TERM_PROGRAM", "").lower()
        return term_program in ("iterm2.app", "iterm2", "wezterm", "kitty", "alacritty", "hyper")

    def supports_mouse(self) -> bool:
        """``True`` if the terminal likely supports mouse events.

        Most modern terminals do; we check for known exceptions
        (e.g. dumb terminals, Emacs TERM=dumb).
        """
        term = os.environ.get("TERM", "")
        if term in ("dumb", ""):
            return False
        return self.is_tty()

    def supports_kitty_keyboard(self) -> bool:
        """``True`` if the terminal supports the Kitty keyboard protocol.

        The Kitty protocol provides unambiguous key reporting.
        Supported by Kitty, WezTerm, and foot.
        """
        term_program = os.environ.get("TERM_PROGRAM", "").lower()
        term = os.environ.get("TERM", "").lower()
        return "kitty" in term or "kitty" in term_program or term_program in ("wezterm", "foot")

    def supports_hyperlinks(self) -> bool:
        """``True`` if the terminal supports OSC 8 hyperlinks.

        Most modern terminals (iTerm2, WezTerm, Kitty, VTE-based
        terminals, Windows Terminal) support clickable hyperlinks.
        """
        term_program = os.environ.get("TERM_PROGRAM", "").lower()
        if term_program in ("iterm2.app", "iterm2", "wezterm", "kitty"):
            return True
        # VTE-based (GNOME Terminal, Tilix, etc.)
        return bool(os.environ.get("VTE_VERSION"))

    def supports_images(self) -> bool:
        """``True`` if the terminal supports inline images.

        Delegates to :func:`nu_tui.terminal_image.supports_inline_images`.
        """
        from nu_tui.terminal_image import supports_inline_images  # noqa: PLC0415

        return supports_inline_images()

    @property
    def rows(self) -> int:
        """Alias for :meth:`get_rows` used by the TUI component."""
        return self.get_rows()

    @property
    def columns(self) -> int:
        """Alias for :meth:`get_columns`."""
        return self.get_columns()


__all__ = ["Terminal", "TerminalSize"]
