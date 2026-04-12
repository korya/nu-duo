"""Terminal image rendering — port of ``packages/tui/src/terminal-image.ts``.

Supports rendering images inline in terminals via:

* **Kitty graphics protocol** — base64-encoded PNG chunks sent via APC
* **iTerm2 inline images** — base64-encoded via OSC 1337

The upstream is 381 LoC. This port covers the protocol detection and
the two main rendering paths. Actual image loading/resizing uses
Pillow (already a nu_tui dependency).
"""

from __future__ import annotations

import base64
import os

# ---------------------------------------------------------------------------
# Protocol detection
# ---------------------------------------------------------------------------


def supports_kitty_graphics() -> bool:
    """Return ``True`` if the terminal likely supports the Kitty graphics protocol."""
    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")
    # Kitty identifies itself
    if "kitty" in term.lower() or "kitty" in term_program.lower():
        return True
    # WezTerm supports Kitty graphics
    return "wezterm" in term_program.lower()


def supports_iterm2_images() -> bool:
    """Return ``True`` if the terminal likely supports iTerm2 inline images."""
    term_program = os.environ.get("TERM_PROGRAM", "")
    return term_program.lower() in ("iterm2.app", "iterm2", "wezterm")


def supports_inline_images() -> bool:
    """Return ``True`` if the terminal supports any inline image protocol."""
    return supports_kitty_graphics() or supports_iterm2_images()


def is_image_line(line: str) -> bool:
    """Return ``True`` if a rendered line contains an image escape sequence.

    Used by the markdown renderer to detect lines that are images and
    should not be word-wrapped.
    """
    return "\x1b_G" in line or "\x1b]1337;" in line


# ---------------------------------------------------------------------------
# Kitty graphics protocol
# ---------------------------------------------------------------------------


def render_kitty_image(
    image_data: bytes,
    *,
    width: int | None = None,
    height: int | None = None,
    columns: int | None = None,
    rows: int | None = None,
) -> list[str]:
    """Render an image via the Kitty graphics protocol.

    ``image_data`` is raw PNG bytes. Returns a list of terminal lines
    containing APC escape sequences. The terminal renders the image
    across ``columns`` x ``rows`` cells.
    """
    encoded = base64.standard_b64encode(image_data).decode("ascii")

    # Build the Kitty command
    parts: list[str] = ["a=T", "f=100"]  # action=transmit, format=PNG
    if columns is not None:
        parts.append(f"c={columns}")
    if rows is not None:
        parts.append(f"r={rows}")

    cmd = ",".join(parts)

    # Split into 4096-byte chunks (Kitty protocol limit)
    chunk_size = 4096
    lines: list[str] = []
    for i in range(0, len(encoded), chunk_size):
        chunk = encoded[i : i + chunk_size]
        is_last = i + chunk_size >= len(encoded)
        more = "m=0" if is_last else "m=1"
        if i == 0:
            lines.append(f"\x1b_G{cmd},{more};{chunk}\x1b\\")
        else:
            lines.append(f"\x1b_G{more};{chunk}\x1b\\")

    return lines


# ---------------------------------------------------------------------------
# iTerm2 inline images
# ---------------------------------------------------------------------------


def render_iterm2_image(
    image_data: bytes,
    *,
    width: str | None = None,
    height: str | None = None,
    name: str = "",
) -> str:
    """Render an image via the iTerm2 inline image protocol.

    Returns a single terminal line containing the OSC 1337 escape.
    ``width``/``height`` can be pixel values, cell values, or percentages
    (e.g. "auto", "50%", "100px").
    """
    encoded = base64.standard_b64encode(image_data).decode("ascii")
    name_b64 = base64.standard_b64encode(name.encode()).decode("ascii") if name else ""

    parts = [f"name={name_b64}"] if name_b64 else []
    parts.append(f"size={len(image_data)}")
    if width:
        parts.append(f"width={width}")
    if height:
        parts.append(f"height={height}")
    parts.append("inline=1")
    params = ";".join(parts)

    return f"\x1b]1337;File={params}:{encoded}\x07"


# ---------------------------------------------------------------------------
# High-level render helper
# ---------------------------------------------------------------------------


def render_image(
    image_data: bytes,
    *,
    columns: int | None = None,
    rows: int | None = None,
) -> list[str]:
    """Render an image using the best available protocol.

    Returns empty list if no image protocol is supported.
    """
    if supports_kitty_graphics():
        return render_kitty_image(image_data, columns=columns, rows=rows)
    if supports_iterm2_images():
        return [render_iterm2_image(image_data)]
    return []


__all__ = [
    "is_image_line",
    "render_image",
    "render_iterm2_image",
    "render_kitty_image",
    "supports_inline_images",
    "supports_iterm2_images",
    "supports_kitty_graphics",
]
