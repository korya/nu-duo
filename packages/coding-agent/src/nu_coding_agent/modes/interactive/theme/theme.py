"""Theme loading and color resolution helpers — port of theme.ts (HTML-export subset).

This module provides the two functions consumed by the HTML exporter:
  - ``get_resolved_theme_colors`` — all theme colors as CSS-compatible hex strings.
  - ``get_theme_export_colors`` — explicit export background overrides from the theme JSON.

The full TUI Theme class (ANSI rendering, watcher, highlight helpers) is not
ported here; those live in the interactive-mode modules once pi_tui is complete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# A colour value in theme JSON is either a hex string, an empty string
# (meaning "terminal default"), a variable reference (bare name), or
# a 256-colour index (integer).
_ColorValue = str | int


# ---------------------------------------------------------------------------
# 256-colour → hex
# ---------------------------------------------------------------------------

_BASIC_COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#c0c0c0",
    "#808080",
    "#ff0000",
    "#00ff00",
    "#ffff00",
    "#0000ff",
    "#ff00ff",
    "#00ffff",
    "#ffffff",
]


def _ansi256_to_hex(index: int) -> str:
    """Convert a 256-colour palette index to a CSS hex colour."""
    if index < 16:
        return _BASIC_COLORS[index]
    if index < 232:
        cube = index - 16
        r = cube // 36
        g = (cube % 36) // 6
        b = cube % 6

        def to_hex(n: int) -> str:
            v = 0 if n == 0 else 55 + n * 40
            return f"{v:02x}"

        return f"#{to_hex(r)}{to_hex(g)}{to_hex(b)}"
    gray = 8 + (index - 232) * 10
    h = f"{gray:02x}"
    return f"#{h}{h}{h}"


# ---------------------------------------------------------------------------
# Variable resolution
# ---------------------------------------------------------------------------


def _resolve_var_refs(
    value: _ColorValue,
    vars_: dict[str, _ColorValue],
    visited: set[str] | None = None,
) -> str | int:
    """Recursively resolve a colour value, following variable references."""
    if visited is None:
        visited = set()
    if isinstance(value, int) or value == "" or value.startswith("#"):
        return value
    if value in visited:
        raise ValueError(f"Circular variable reference: {value}")
    if value not in vars_:
        raise ValueError(f"Variable reference not found: {value}")
    visited.add(value)
    return _resolve_var_refs(vars_[value], vars_, visited)


def _resolve_theme_colors(
    colors: dict[str, _ColorValue],
    vars_: dict[str, _ColorValue] | None = None,
) -> dict[str, str | int]:
    v = vars_ or {}
    return {key: _resolve_var_refs(val, v) for key, val in colors.items()}


# ---------------------------------------------------------------------------
# Theme JSON loading
# ---------------------------------------------------------------------------


def _get_default_theme() -> str:
    """Detect dark/light preference from COLORFGBG env var; default to dark."""
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        parts = colorfgbg.split(";")
        try:
            bg = int(parts[-1])
            return "dark" if bg < 8 else "light"
        except (ValueError, IndexError):
            pass
    return "dark"


def _load_theme_json(name: str) -> dict[str, object]:
    """Load a theme JSON file by name (built-in first, then custom dir)."""
    from nu_coding_agent.config import get_custom_themes_dir, get_themes_dir  # noqa: PLC0415

    builtin_path = Path(get_themes_dir()) / f"{name}.json"
    if builtin_path.exists():
        return json.loads(builtin_path.read_text(encoding="utf-8"))  # type: ignore[return-value]

    custom_path = Path(get_custom_themes_dir()) / f"{name}.json"
    if custom_path.exists():
        return json.loads(custom_path.read_text(encoding="utf-8"))  # type: ignore[return-value]

    raise ValueError(f"Theme not found: {name}")


# ---------------------------------------------------------------------------
# Public helpers (used by HTML export)
# ---------------------------------------------------------------------------

# Module-level theme name set by interactive mode when it loads a theme.
_current_theme_name: str | None = None


def get_resolved_theme_colors(theme_name: str | None = None) -> dict[str, str]:
    """Return all theme colours as CSS-compatible hex strings.

    Resolves variable references and converts 256-colour indices to hex.
    Empty values (terminal default) become a sensible fallback colour.
    """
    name = theme_name or _current_theme_name or _get_default_theme()
    is_light = name == "light"
    try:
        theme_json = _load_theme_json(name)
    except ValueError:
        theme_json = _load_theme_json("dark")
        is_light = False

    colors = theme_json.get("colors", {})
    vars_ = theme_json.get("vars", {})
    resolved = _resolve_theme_colors(colors, vars_)  # type: ignore[arg-type]

    default_text = "#000000" if is_light else "#e5e5e7"
    css: dict[str, str] = {}
    for key, value in resolved.items():
        if isinstance(value, int):
            css[key] = _ansi256_to_hex(value)
        elif value == "":
            css[key] = default_text
        else:
            css[key] = value
    return css


def get_theme_export_colors(theme_name: str | None = None) -> dict[str, str | None]:
    """Return the explicit export background colours from the theme JSON.

    Returns a dict with keys ``pageBg``, ``cardBg``, ``infoBg`` (each may be
    ``None`` if not specified in the theme).
    """
    name = theme_name or _current_theme_name or _get_default_theme()
    try:
        theme_json = _load_theme_json(name)
    except ValueError:
        return {"pageBg": None, "cardBg": None, "infoBg": None}

    export_section = theme_json.get("export")
    if not export_section:
        return {"pageBg": None, "cardBg": None, "infoBg": None}

    vars_ = theme_json.get("vars", {})

    def resolve(value: object) -> str | None:
        if value is None:
            return None
        resolved = _resolve_var_refs(value, vars_)  # type: ignore[arg-type]
        if isinstance(resolved, int):
            return _ansi256_to_hex(resolved)
        return resolved if resolved else None

    assert isinstance(export_section, dict)
    return {
        "pageBg": resolve(export_section.get("pageBg")),
        "cardBg": resolve(export_section.get("cardBg")),
        "infoBg": resolve(export_section.get("infoBg")),
    }


__all__ = ["get_resolved_theme_colors", "get_theme_export_colors"]
