"""Image MIME-type sniffer — direct port of ``packages/coding-agent/src/utils/mime.ts``.

The TS upstream uses the ``file-type`` npm package which sniffs magic
bytes from the first ~4 KB of the file. The Python port replaces it
with a hand-rolled magic-byte check for the four formats the upstream
allows (JPEG, PNG, GIF, WEBP). No third-party deps.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

_FILE_TYPE_SNIFF_BYTES = 4100


_IMAGE_MIME_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def _sniff_mime_from_bytes(buffer: bytes) -> str | None:
    """Return one of the four supported image MIME types, or ``None``.

    Mirrors the upstream ``file-type`` whitelist: ``image/jpeg``,
    ``image/png``, ``image/gif``, ``image/webp``. Anything else (HEIC,
    AVIF, BMP, SVG, …) is rejected by returning ``None``.
    """
    if len(buffer) < 4:
        return None
    # JPEG: FF D8 FF
    if buffer[0] == 0xFF and buffer[1] == 0xD8 and buffer[2] == 0xFF:
        return "image/jpeg"
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if buffer[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # GIF: 47 49 46 38 (GIF8)
    if buffer[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    # WebP: RIFF....WEBP
    if buffer[:4] == b"RIFF" and len(buffer) >= 12 and buffer[8:12] == b"WEBP":
        return "image/webp"
    return None


def _read_sniff_sync(file_path: str) -> bytes:
    return Path(file_path).read_bytes()[:_FILE_TYPE_SNIFF_BYTES]


async def detect_supported_image_mime_type_from_file(file_path: str) -> str | None:
    """Sniff ``file_path`` and return its MIME type if it's a supported image."""
    try:
        buffer = await asyncio.to_thread(_read_sniff_sync, file_path)
    except OSError:
        return None
    if not buffer:
        return None
    mime = _sniff_mime_from_bytes(buffer)
    if mime is None or mime not in _IMAGE_MIME_TYPES:
        return None
    return mime


__all__ = ["detect_supported_image_mime_type_from_file"]
