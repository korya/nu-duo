"""EXIF orientation — port of ``packages/coding-agent/src/utils/exif-orientation.ts`` (184 LOC).

Parse the EXIF orientation tag from raw image bytes (JPEG / WebP) and
apply rotation/flip transforms using Pillow.

Two entry-points:

* :func:`get_exif_orientation` — byte-level parser, no PIL Image needed.
* :func:`apply_exif_orientation` — accepts a PIL Image, returns a
  correctly-oriented copy (identity when orientation is already normal).
"""

from __future__ import annotations

import struct

from PIL import Image, ImageOps

# EXIF orientation tag ID
_ORIENTATION_TAG = 0x0112


# ---------------------------------------------------------------------------
# Low-level byte parsing
# ---------------------------------------------------------------------------


def _has_exif_header(data: bytes, offset: int) -> bool:
    """Return *True* if ``data[offset:offset+6]`` is the ``Exif\\x00\\x00`` header."""
    return data[offset : offset + 6] == b"Exif\x00\x00"


def _read_orientation_from_tiff(data: bytes, tiff_start: int) -> int:
    """Read the orientation value from a TIFF header embedded in EXIF data.

    Returns a value in ``1..8``, defaulting to ``1`` (normal) when the
    tag is absent or the data is malformed.
    """
    if tiff_start + 8 > len(data):
        return 1

    byte_order = data[tiff_start : tiff_start + 2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return 1

    (ifd_offset,) = struct.unpack_from(f"{endian}I", data, tiff_start + 4)
    ifd_start = tiff_start + ifd_offset
    if ifd_start + 2 > len(data):
        return 1

    (entry_count,) = struct.unpack_from(f"{endian}H", data, ifd_start)
    for i in range(entry_count):
        entry_pos = ifd_start + 2 + i * 12
        if entry_pos + 12 > len(data):
            return 1
        (tag,) = struct.unpack_from(f"{endian}H", data, entry_pos)
        if tag == _ORIENTATION_TAG:
            (value,) = struct.unpack_from(f"{endian}H", data, entry_pos + 8)
            return value if 1 <= value <= 8 else 1

    return 1


def _find_jpeg_tiff_offset(data: bytes) -> int:
    """Locate the TIFF header inside a JPEG's APP1 (EXIF) segment.

    Returns ``-1`` when no EXIF segment is found.
    """
    offset = 2  # skip SOI marker (FF D8)
    while offset < len(data) - 1:
        if data[offset] != 0xFF:
            return -1
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue

        # APP1 — the EXIF segment
        if marker == 0xE1:
            if offset + 4 >= len(data):
                return -1
            segment_start = offset + 4
            if segment_start + 6 > len(data):
                return -1
            if not _has_exif_header(data, segment_start):
                return -1
            return segment_start + 6

        # Skip other segments
        if offset + 4 > len(data):
            return -1
        length = (data[offset + 2] << 8) | data[offset + 3]
        offset += 2 + length

    return -1


def _find_webp_tiff_offset(data: bytes) -> int:
    """Locate the TIFF header inside a WebP EXIF chunk.

    Returns ``-1`` when no EXIF chunk is found.
    """
    offset = 12  # skip RIFF header + "WEBP"
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        (chunk_size,) = struct.unpack_from("<I", data, offset + 4)
        data_start = offset + 8

        if chunk_id == b"EXIF":
            if data_start + chunk_size > len(data):
                return -1
            # Some WebP files have "Exif\x00\x00" prefix before the TIFF header
            if chunk_size >= 6 and _has_exif_header(data, data_start):
                return data_start + 6
            return data_start

        # RIFF chunks are padded to even size
        offset = data_start + chunk_size + (chunk_size % 2)

    return -1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_exif_orientation(image_bytes: bytes) -> int:
    """Extract the EXIF orientation value (``1``-``8``) from raw image bytes.

    Supports JPEG and WebP.  Returns ``1`` (normal / no rotation) when
    the orientation tag is absent, the format is unrecognised, or the
    data is malformed.
    """
    if len(image_bytes) < 2:
        return 1

    tiff_offset = -1

    # JPEG: starts with FF D8
    if image_bytes[0] == 0xFF and image_bytes[1] == 0xD8:
        tiff_offset = _find_jpeg_tiff_offset(image_bytes)
    # WebP: starts with RIFF....WEBP
    elif len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        tiff_offset = _find_webp_tiff_offset(image_bytes)

    if tiff_offset == -1:
        return 1
    return _read_orientation_from_tiff(image_bytes, tiff_offset)


def apply_exif_orientation(
    image: Image.Image,
    original_bytes: bytes | None = None,
) -> Image.Image:
    """Apply EXIF orientation correction to a PIL :class:`~PIL.Image.Image`.

    When *original_bytes* are supplied the orientation is read from the
    raw bytes (useful when the Image was constructed without EXIF info).
    Otherwise :meth:`PIL.Image.Image.getexif` is consulted.

    Returns a new, correctly-oriented image — or the original image
    unchanged when the orientation is already normal (``1``).
    """
    # Determine the orientation value
    if original_bytes is not None:
        orientation = get_exif_orientation(original_bytes)
    else:
        exif = image.getexif()
        orientation = exif.get(_ORIENTATION_TAG, 1)

    if orientation == 1:
        return image

    # ImageOps.exif_transpose handles all 8 cases.  It returns None when
    # there is nothing to do, so fall back to the original image.
    try:
        corrected = ImageOps.exif_transpose(image)
    except Exception:
        # If Pillow can't transpose (e.g. palette issues), do it manually.
        corrected = None

    if corrected is not None:
        return corrected

    # Manual fallback keyed on the orientation we already parsed.
    if orientation == 2:
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if orientation == 3:
        return image.transpose(Image.Transpose.ROTATE_180)
    if orientation == 4:
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if orientation == 5:
        return image.transpose(Image.Transpose.TRANSPOSE)
    if orientation == 6:
        return image.transpose(Image.Transpose.ROTATE_270)
    if orientation == 7:
        return image.transpose(Image.Transpose.TRANSVERSE)
    if orientation == 8:
        return image.transpose(Image.Transpose.ROTATE_90)

    return image  # pragma: no cover


__all__ = ["apply_exif_orientation", "get_exif_orientation"]
