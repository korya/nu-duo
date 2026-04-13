"""Keybinding hint utilities — port of ``keybinding-hints.ts``.

In upstream pi-mono, ``keyText`` calls ``getKeybindings().getKeys(keybinding)``
to resolve the bound key names at runtime. The Python port currently has a
static fallback table covering every keybinding used in the interactive mode.
Once ``nu_tui.keybindings`` is fully wired these can delegate to it.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default key labels (same mnemonics the upstream default bindings file uses)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, str] = {
    "tui.select.cancel": "Esc",
    "tui.select.confirm": "Enter",
    "tui.select.up": "↑",
    "tui.select.down": "↓",
    "app.tools.expand": "Tab",
    "app.tools.collapse": "Tab",
    "app.quit": "Ctrl+C",
    "app.submit": "Enter",
}


def key_text(keybinding: str) -> str:
    """Return the display label for *keybinding* (e.g. ``"Esc"``).

    Port of upstream ``keyText(keybinding: Keybinding): string``.
    """
    return _DEFAULTS.get(keybinding, keybinding)


def key_hint(keybinding: str, description: str) -> str:
    """Return ``"<key> <description>"`` formatted as a dim hint string.

    Port of upstream ``keyHint(keybinding, description)``.
    Returns plain text (no Rich markup); callers may style it themselves.
    """
    return f"{key_text(keybinding)} {description}"


def raw_key_hint(key: str, description: str) -> str:
    """Same as :func:`key_hint` but with a literal key string."""
    return f"{key} {description}"


__all__ = ["key_hint", "key_text", "raw_key_hint"]
