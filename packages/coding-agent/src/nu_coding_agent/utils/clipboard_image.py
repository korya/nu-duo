"""Cross-platform clipboard image reading — port of ``packages/coding-agent/src/utils/clipboard-image.ts``.

Reads image data from the system clipboard on Linux (Wayland & X11),
macOS, and WSL.  Unsupported formats (e.g. BMP from WSLg) are converted
to PNG via Pillow.

Platform strategies
~~~~~~~~~~~~~~~~~~~
* **Wayland** — ``wl-paste --list-types`` → ``wl-paste -t <mime>``
* **X11** — ``xclip -selection clipboard -t TARGETS -o`` →
  ``xclip -selection clipboard -t <mime> -o``
* **macOS** — ``pngpaste`` (preferred, binary output) with ``osascript``
  fallback
* **WSL** — falls back to PowerShell ``[Clipboard]::GetImage()``
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_MIME_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
)

_DEFAULT_LIST_TIMEOUT_MS: float = 1.0  # seconds
_DEFAULT_READ_TIMEOUT_MS: float = 3.0
_DEFAULT_POWERSHELL_TIMEOUT_MS: float = 5.0
_DEFAULT_MAX_BUFFER_BYTES: int = 50 * 1024 * 1024  # 50 MiB


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ClipboardImage:
    """Raw image bytes together with their MIME type."""

    bytes: bytes
    mime_type: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_wayland_session(env: dict[str, str] | None = None) -> bool:
    """Return ``True`` when the session looks like Wayland."""
    if env is None:
        env = dict(os.environ)
    return bool(env.get("WAYLAND_DISPLAY")) or env.get("XDG_SESSION_TYPE") == "wayland"


def _base_mime_type(mime_type: str) -> str:
    """Strip parameters (e.g. ``image/png; charset=utf-8`` → ``image/png``)."""
    return mime_type.split(";", maxsplit=1)[0].strip().lower()


def extension_for_image_mime_type(mime_type: str) -> str | None:
    """Map a MIME type to its conventional file extension (or ``None``)."""
    base = _base_mime_type(mime_type)
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }.get(base)


def _select_preferred_image_mime_type(mime_types: list[str]) -> str | None:
    """Pick the best MIME type from a list, preferring our supported types."""
    normalised = [
        (raw, _base_mime_type(raw))
        for raw in (t.strip() for t in mime_types)
        if raw.strip()
    ]
    # Prefer our supported types in order
    for preferred in SUPPORTED_IMAGE_MIME_TYPES:
        for raw, base in normalised:
            if base == preferred:
                return raw
    # Fall back to any image type
    for raw, base in normalised:
        if base.startswith("image/"):
            return raw
    return None


def _is_supported_image_mime_type(mime_type: str) -> bool:
    return _base_mime_type(mime_type) in SUPPORTED_IMAGE_MIME_TYPES


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _CmdResult:
    ok: bool
    stdout: bytes


def _run_command(
    command: str,
    args: list[str],
    *,
    timeout: float = _DEFAULT_READ_TIMEOUT_MS,
    max_buffer: int = _DEFAULT_MAX_BUFFER_BYTES,
    env: dict[str, str] | None = None,
) -> _CmdResult:
    """Run *command* synchronously and return ``(ok, stdout)``."""
    try:
        result = subprocess.run(
            [command, *args],
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return _CmdResult(ok=False, stdout=b"")

    if result.returncode != 0:
        return _CmdResult(ok=False, stdout=b"")

    stdout = result.stdout
    if len(stdout) > max_buffer:
        return _CmdResult(ok=False, stdout=b"")

    return _CmdResult(ok=True, stdout=stdout)


# ---------------------------------------------------------------------------
# Platform readers
# ---------------------------------------------------------------------------

def _read_clipboard_image_via_wl_paste() -> ClipboardImage | None:
    list_result = _run_command("wl-paste", ["--list-types"], timeout=_DEFAULT_LIST_TIMEOUT_MS)
    if not list_result.ok:
        return None

    types = [t.strip() for t in list_result.stdout.decode("utf-8", errors="replace").splitlines() if t.strip()]
    selected = _select_preferred_image_mime_type(types)
    if not selected:
        return None

    data = _run_command("wl-paste", ["--type", selected, "--no-newline"])
    if not data.ok or len(data.stdout) == 0:
        return None

    return ClipboardImage(bytes=data.stdout, mime_type=_base_mime_type(selected))


def _is_wsl(env: dict[str, str] | None = None) -> bool:
    if env is None:
        env = dict(os.environ)
    if env.get("WSL_DISTRO_NAME") or env.get("WSLENV"):
        return True
    try:
        release = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
        return bool(
            "microsoft" in release.lower() or "wsl" in release.lower()
        )
    except OSError:
        return False


def _read_clipboard_image_via_powershell() -> ClipboardImage | None:
    """Use PowerShell on WSL to access the Windows clipboard."""
    tmp_file = os.path.join(tempfile.gettempdir(), f"nu-wsl-clip-{uuid.uuid4()}.png")

    try:
        win_path_result = _run_command("wslpath", ["-w", tmp_file], timeout=_DEFAULT_LIST_TIMEOUT_MS)
        if not win_path_result.ok:
            return None

        win_path = win_path_result.stdout.decode("utf-8", errors="replace").strip()
        if not win_path:
            return None

        ps_script = "; ".join([
            "Add-Type -AssemblyName System.Windows.Forms",
            "Add-Type -AssemblyName System.Drawing",
            "$path = $env:NU_WSL_CLIPBOARD_IMAGE_PATH",
            "$img = [System.Windows.Forms.Clipboard]::GetImage()",
            "if ($img) { $img.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); Write-Output 'ok' } else { Write-Output 'empty' }",
        ])

        run_env = {**os.environ, "NU_WSL_CLIPBOARD_IMAGE_PATH": win_path}
        result = _run_command(
            "powershell.exe",
            ["-NoProfile", "-Command", ps_script],
            timeout=_DEFAULT_POWERSHELL_TIMEOUT_MS,
            env=run_env,
        )
        if not result.ok:
            return None

        output = result.stdout.decode("utf-8", errors="replace").strip()
        if output != "ok":
            return None

        data = Path(tmp_file).read_bytes()
        if len(data) == 0:
            return None

        return ClipboardImage(bytes=data, mime_type="image/png")
    except Exception:
        return None
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_file)


def _read_clipboard_image_via_xclip() -> ClipboardImage | None:
    targets = _run_command(
        "xclip",
        ["-selection", "clipboard", "-t", "TARGETS", "-o"],
        timeout=_DEFAULT_LIST_TIMEOUT_MS,
    )

    candidate_types: list[str] = []
    if targets.ok:
        candidate_types = [
            t.strip()
            for t in targets.stdout.decode("utf-8", errors="replace").splitlines()
            if t.strip()
        ]

    preferred = _select_preferred_image_mime_type(candidate_types) if candidate_types else None
    try_types: list[str] = (
        [preferred, *SUPPORTED_IMAGE_MIME_TYPES] if preferred else list(SUPPORTED_IMAGE_MIME_TYPES)
    )

    for mime_type in try_types:
        data = _run_command("xclip", ["-selection", "clipboard", "-t", mime_type, "-o"])
        if data.ok and len(data.stdout) > 0:
            return ClipboardImage(bytes=data.stdout, mime_type=_base_mime_type(mime_type))

    return None


def _read_clipboard_image_via_pngpaste() -> ClipboardImage | None:
    """macOS: use ``pngpaste`` to read clipboard image to stdout."""
    result = _run_command("pngpaste", ["-"], timeout=_DEFAULT_READ_TIMEOUT_MS)
    if result.ok and len(result.stdout) > 0:
        return ClipboardImage(bytes=result.stdout, mime_type="image/png")
    return None


def _read_clipboard_image_via_osascript() -> ClipboardImage | None:
    """macOS fallback: use AppleScript to grab PNGf clipboard data."""
    result = _run_command(
        "osascript",
        ["-e", "the clipboard as «class PNGf»"],
        timeout=_DEFAULT_READ_TIMEOUT_MS,
    )
    if not result.ok:
        return None

    # osascript returns hex like «data PNGfXXXX…»
    raw = result.stdout.decode("utf-8", errors="replace").strip()
    # Strip «data PNGf ... »
    hex_start = raw.find("«data PNGf")
    if hex_start == -1:
        return None
    hex_data = raw[hex_start + len("«data PNGf"):]
    hex_end = hex_data.find("»")
    if hex_end == -1:
        return None
    hex_data = hex_data[:hex_end].strip()

    try:
        image_bytes = bytes.fromhex(hex_data)
    except ValueError:
        return None

    if len(image_bytes) == 0:
        return None

    return ClipboardImage(bytes=image_bytes, mime_type="image/png")


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def _convert_to_png(data: bytes) -> bytes | None:
    """Convert arbitrary image bytes to PNG via Pillow. Returns ``None`` on failure."""
    try:
        from PIL import Image  # noqa: PLC0415 — lazy import to keep Pillow optional

        with Image.open(io.BytesIO(data)) as img:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _read_clipboard_image_sync(
    env: dict[str, str] | None = None,
    platform: str | None = None,
) -> ClipboardImage | None:
    """Blocking implementation — call via :func:`asyncio.to_thread`."""
    if env is None:
        env = dict(os.environ)
    if platform is None:
        platform = sys.platform

    if env.get("TERMUX_VERSION"):
        return None

    image: ClipboardImage | None = None

    if platform == "linux":
        wsl = _is_wsl(env)
        wayland = is_wayland_session(env)

        if wayland or wsl:
            image = _read_clipboard_image_via_wl_paste() or _read_clipboard_image_via_xclip()

        if image is None and wsl:
            image = _read_clipboard_image_via_powershell()

        if image is None and not wayland:
            image = _read_clipboard_image_via_xclip()
    elif platform == "darwin":
        image = _read_clipboard_image_via_pngpaste() or _read_clipboard_image_via_osascript()
    # Windows clipboard image reading is not supported in the CLI context

    if image is None:
        return None

    # Convert unsupported formats (e.g. BMP from WSLg) to PNG
    if not _is_supported_image_mime_type(image.mime_type):
        png_bytes = _convert_to_png(image.bytes)
        if png_bytes is None:
            return None
        return ClipboardImage(bytes=png_bytes, mime_type="image/png")

    return image


async def read_clipboard_image(
    env: dict[str, str] | None = None,
    platform: str | None = None,
) -> ClipboardImage | None:
    """Read an image from the system clipboard (cross-platform).

    Returns a :class:`ClipboardImage` with the raw bytes and MIME type,
    or ``None`` when no image is available.
    """
    return await asyncio.to_thread(_read_clipboard_image_sync, env, platform)


__all__ = [
    "ClipboardImage",
    "extension_for_image_mime_type",
    "is_wayland_session",
    "read_clipboard_image",
]
