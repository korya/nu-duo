"""ANSI escape code to HTML converter — port of export-html/ansi-to-html.ts.

Converts terminal ANSI colour/style codes to HTML with inline styles.
Supports:
- Standard foreground colours (30-37) and bright variants (90-97)
- Standard background colours (40-47) and bright variants (100-107)
- 256-colour palette (38;5;N and 48;5;N)
- RGB true colour (38;2;R;G;B and 48;2;R;G;B)
- Text styles: bold (1), dim (2), italic (3), underline (4)
- Reset (0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Standard ANSI colour palette (0-15)
_ANSI_COLORS = [
    "#000000",  # 0: black
    "#800000",  # 1: red
    "#008000",  # 2: green
    "#808000",  # 3: yellow
    "#000080",  # 4: blue
    "#800080",  # 5: magenta
    "#008080",  # 6: cyan
    "#c0c0c0",  # 7: white
    "#808080",  # 8: bright black
    "#ff0000",  # 9: bright red
    "#00ff00",  # 10: bright green
    "#ffff00",  # 11: bright yellow
    "#0000ff",  # 12: bright blue
    "#ff00ff",  # 13: bright magenta
    "#00ffff",  # 14: bright cyan
    "#ffffff",  # 15: bright white
]


def _color256_to_hex(index: int) -> str:
    """Convert a 256-colour index to hex."""
    if index < 16:
        return _ANSI_COLORS[index]
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


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


@dataclass
class _TextStyle:
    fg: str | None = None
    bg: str | None = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False


def _style_to_inline_css(style: _TextStyle) -> str:
    parts: list[str] = []
    if style.fg:
        parts.append(f"color:{style.fg}")
    if style.bg:
        parts.append(f"background-color:{style.bg}")
    if style.bold:
        parts.append("font-weight:bold")
    if style.dim:
        parts.append("opacity:0.6")
    if style.italic:
        parts.append("font-style:italic")
    if style.underline:
        parts.append("text-decoration:underline")
    return ";".join(parts)


def _has_style(style: _TextStyle) -> bool:
    return bool(style.fg or style.bg or style.bold or style.dim or style.italic or style.underline)


def _apply_sgr_code(params: list[int], style: _TextStyle) -> None:
    """Apply ANSI SGR (Select Graphic Rendition) codes to style in-place."""
    i = 0
    while i < len(params):
        code = params[i]
        if code == 0:
            style.fg = None
            style.bg = None
            style.bold = False
            style.dim = False
            style.italic = False
            style.underline = False
        elif code == 1:
            style.bold = True
        elif code == 2:
            style.dim = True
        elif code == 3:
            style.italic = True
        elif code == 4:
            style.underline = True
        elif code == 22:
            style.bold = False
            style.dim = False
        elif code == 23:
            style.italic = False
        elif code == 24:
            style.underline = False
        elif 30 <= code <= 37:
            style.fg = _ANSI_COLORS[code - 30]
        elif code == 38:
            if i + 2 < len(params) and params[i + 1] == 5:
                style.fg = _color256_to_hex(params[i + 2])
                i += 2
            elif i + 4 < len(params) and params[i + 1] == 2:
                r, g, b = params[i + 2], params[i + 3], params[i + 4]
                style.fg = f"rgb({r},{g},{b})"
                i += 4
        elif code == 39:
            style.fg = None
        elif 40 <= code <= 47:
            style.bg = _ANSI_COLORS[code - 40]
        elif code == 48:
            if i + 2 < len(params) and params[i + 1] == 5:
                style.bg = _color256_to_hex(params[i + 2])
                i += 2
            elif i + 4 < len(params) and params[i + 1] == 2:
                r, g, b = params[i + 2], params[i + 3], params[i + 4]
                style.bg = f"rgb({r},{g},{b})"
                i += 4
        elif code == 49:
            style.bg = None
        elif 90 <= code <= 97:
            style.fg = _ANSI_COLORS[code - 90 + 8]
        elif 100 <= code <= 107:
            style.bg = _ANSI_COLORS[code - 100 + 8]
        i += 1


_ANSI_RE = re.compile(r"\x1b\[([\d;]*)m")


def ansi_to_html(text: str) -> str:
    """Convert ANSI-escaped text to HTML with inline styles."""
    style = _TextStyle()
    result: list[str] = []
    last_index = 0
    in_span = False

    for match in _ANSI_RE.finditer(text):
        before = text[last_index : match.start()]
        if before:
            result.append(_escape_html(before))

        param_str = match.group(1)
        params = [int(p) if p else 0 for p in param_str.split(";")] if param_str else [0]

        if in_span:
            result.append("</span>")
            in_span = False

        _apply_sgr_code(params, style)

        if _has_style(style):
            result.append(f'<span style="{_style_to_inline_css(style)}">')
            in_span = True

        last_index = match.end()

    remaining = text[last_index:]
    if remaining:
        result.append(_escape_html(remaining))

    if in_span:
        result.append("</span>")

    return "".join(result)


def ansi_lines_to_html(lines: list[str]) -> str:
    """Convert a list of ANSI-escaped lines to HTML divs."""
    return "\n".join(f'<div class="ansi-line">{ansi_to_html(line) or "&nbsp;"}</div>' for line in lines)


__all__ = ["ansi_lines_to_html", "ansi_to_html"]
