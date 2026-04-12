"""Theme system — pragmatic first pass.

The upstream theme is 1141 LoC with 50+ color slots, variable
resolution, JSON loading, and hex/256-color support. This first pass
provides:

* A :class:`Theme` dataclass with callable color-function slots for
  the core UI colors and markdown elements that existing components
  consume.
* ``dark_theme()`` and ``light_theme()`` factory functions that
  produce the two built-in presets.
* A module-level ``get_theme()`` / ``set_theme()`` accessor so
  components can read the active theme without threading it through
  every constructor.

The full JSON schema, on-disk theme loading, variable resolution, and
the 50+ color slots land in a follow-up slice alongside the settings
selector and interactive-mode chrome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nu_tui.components.markdown import MarkdownTheme

# ---------------------------------------------------------------------------
# ANSI styling helpers
# ---------------------------------------------------------------------------


def _ansi(code: int) -> str:
    return f"\033[{code}m"


def _fg(code: int):
    """Build a callable that wraps text in a foreground color."""

    def apply(text: str) -> str:
        return f"{_ansi(code)}{text}{_ansi(0)}"

    return apply


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[22m"


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[22m"


def _italic(text: str) -> str:
    return f"\033[3m{text}\033[23m"


def _underline(text: str) -> str:
    return f"\033[4m{text}\033[24m"



def _noop(text: str) -> str:
    return text


# ---------------------------------------------------------------------------
# SelectListTheme re-used from the components
# ---------------------------------------------------------------------------

from nu_tui.components.select_list import SelectListTheme  # noqa: E402

# ---------------------------------------------------------------------------
# Theme dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Theme:
    """Active theme with callable color-function slots.

    Each slot is a ``Callable[[str], str]`` that wraps text in ANSI
    escapes. A no-op (``lambda s: s``) is valid for any slot.
    """

    name: str = "dark"

    # Core UI
    accent: Any = field(default_factory=lambda: _fg(36))  # cyan
    border: Any = field(default_factory=lambda: _dim)
    border_accent: Any = field(default_factory=lambda: _fg(36))
    muted: Any = field(default_factory=lambda: _dim)
    dim: Any = field(default_factory=lambda: _dim)
    text: Any = field(default_factory=lambda: _noop)
    error: Any = field(default_factory=lambda: _fg(31))  # red
    success: Any = field(default_factory=lambda: _fg(32))  # green
    warning: Any = field(default_factory=lambda: _fg(33))  # yellow

    # Markdown
    md_heading: Any = field(default_factory=lambda: _bold)
    md_code: Any = field(default_factory=lambda: _fg(33))
    md_code_block: Any = field(default_factory=lambda: _noop)
    md_code_block_border: Any = field(default_factory=lambda: _dim)
    md_link: Any = field(default_factory=lambda: _underline)
    md_quote: Any = field(default_factory=lambda: _italic)
    md_hr: Any = field(default_factory=lambda: _dim)
    md_list_bullet: Any = field(default_factory=lambda: _fg(36))
    md_bold: Any = field(default_factory=lambda: _bold)
    md_italic: Any = field(default_factory=lambda: _italic)

    # Select list
    selected_text: Any = field(default_factory=lambda: _bold)
    description: Any = field(default_factory=lambda: _dim)
    scroll_info: Any = field(default_factory=lambda: _dim)
    no_match: Any = field(default_factory=lambda: _dim)

    def to_markdown_theme(self) -> MarkdownTheme:
        """Build a :class:`MarkdownTheme` from the active theme."""
        return MarkdownTheme(
            heading=self.md_heading,
            code=self.md_code,
            code_block=self.md_code_block,
            code_block_border=self.md_code_block_border,
            link=self.md_link,
            bold=self.md_bold,
            italic=self.md_italic,
            quote=self.md_quote,
            hr=self.md_hr,
            list_bullet=self.md_list_bullet,
        )

    def to_select_list_theme(self) -> SelectListTheme:
        """Build a :class:`SelectListTheme` from the active theme."""
        return SelectListTheme(
            selected_prefix=self.accent,
            selected_text=self.selected_text,
            description=self.description,
            scroll_info=self.scroll_info,
            no_match=self.no_match,
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def dark_theme() -> Theme:
    """Build the default dark theme preset."""
    return Theme(name="dark")


def light_theme() -> Theme:
    """Build a light-background theme preset."""
    return Theme(
        name="light",
        accent=_fg(34),  # blue
        border=_dim,
        border_accent=_fg(34),
        error=_fg(31),
        success=_fg(32),
        warning=_fg(33),
        md_heading=_bold,
        md_code=_fg(34),
        md_link=_underline,
    )


# ---------------------------------------------------------------------------
# Global accessor
# ---------------------------------------------------------------------------

_active_theme: Theme = dark_theme()


def get_theme() -> Theme:
    """Return the active theme."""
    return _active_theme


def set_theme(theme: Theme) -> Theme:
    """Set the active theme. Returns the previous theme."""
    global _active_theme  # noqa: PLW0603
    prev = _active_theme
    _active_theme = theme
    return prev


__all__ = [
    "Theme",
    "dark_theme",
    "get_theme",
    "light_theme",
    "set_theme",
]
