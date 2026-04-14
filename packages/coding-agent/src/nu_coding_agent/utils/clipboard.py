"""Cross-platform clipboard copy — port of ``packages/coding-agent/src/utils/clipboard.ts``.

Copies text to the system clipboard using multiple fallback strategies:

1. **OSC 52** escape sequence (always emitted first) — works over SSH/mosh
   and is harmless locally.
2. **Platform-specific CLI tools** — ``pbcopy`` (macOS), ``clip`` (Windows),
   ``termux-clipboard-set`` / ``wl-copy`` / ``xclip`` / ``xsel`` (Linux).

All subprocess errors are swallowed silently; the OSC 52 emission acts as
the ultimate fallback.
"""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys


def _copy_to_x11_clipboard(text: str, *, timeout: float = 5.0) -> None:
    """Try ``xclip``, then fall back to ``xsel``."""
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode(),
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        subprocess.run(
            ["xsel", "--clipboard", "--input"],
            input=text.encode(),
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _copy_sync(text: str) -> None:
    """Blocking implementation — run via :func:`asyncio.to_thread`."""
    # Always emit OSC 52 — works over SSH/mosh, harmless locally
    encoded = base64.b64encode(text.encode()).decode("ascii")
    sys.stdout.write(f"\033]52;c;{encoded}\a")
    sys.stdout.flush()

    timeout = 5.0

    try:
        if sys.platform == "darwin":
            subprocess.run(
                ["pbcopy"],
                input=text.encode(),
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        elif sys.platform == "win32":
            subprocess.run(
                ["clip"],
                input=text.encode(),
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            # Linux — try Termux, Wayland, or X11 clipboard tools
            if os.environ.get("TERMUX_VERSION"):
                try:
                    subprocess.run(
                        ["termux-clipboard-set"],
                        input=text.encode(),
                        timeout=timeout,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    return
                except Exception:
                    pass  # Fall back to Wayland or X11 tools

            from nu_coding_agent.utils.clipboard_image import is_wayland_session  # noqa: PLC0415

            has_wayland_display = bool(os.environ.get("WAYLAND_DISPLAY"))
            has_x11_display = bool(os.environ.get("DISPLAY"))
            wayland = is_wayland_session()

            if wayland and has_wayland_display:
                try:
                    # Verify wl-copy exists
                    subprocess.run(["which", "wl-copy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=timeout)
                    # wl-copy with subprocess.run hangs due to fork behaviour; use Popen
                    proc = subprocess.Popen(
                        ["wl-copy"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    try:
                        proc.stdin.write(text.encode())  # type: ignore[union-attr]
                        proc.stdin.close()  # type: ignore[union-attr]
                    except BrokenPipeError:
                        pass  # wl-copy exited early
                except Exception:
                    if has_x11_display:
                        _copy_to_x11_clipboard(text, timeout=timeout)
            elif has_x11_display:
                _copy_to_x11_clipboard(text, timeout=timeout)
    except Exception:
        # Ignore — OSC 52 already emitted as fallback
        pass


async def copy_to_clipboard(text: str) -> None:
    """Copy *text* to the system clipboard (cross-platform, multiple fallbacks)."""
    await asyncio.to_thread(_copy_sync, text)


__all__ = [
    "copy_to_clipboard",
]
