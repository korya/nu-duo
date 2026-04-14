"""Tests for ``nu_coding_agent.utils.exif_orientation``."""

from __future__ import annotations

import struct

from PIL import Image

from nu_coding_agent.utils.exif_orientation import (
    _ORIENTATION_TAG,
    _find_jpeg_tiff_offset,
    _find_webp_tiff_offset,
    _has_exif_header,
    _read_orientation_from_tiff,
    apply_exif_orientation,
    get_exif_orientation,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic EXIF data
# ---------------------------------------------------------------------------

def _build_tiff_with_orientation(orientation: int, big_endian: bool = False) -> bytes:
    """Build a minimal TIFF header with a single IFD entry for orientation."""
    endian = ">" if big_endian else "<"
    byte_order = b"MM" if big_endian else b"II"

    # TIFF header: byte order (2) + magic 42 (2) + IFD offset (4)
    header = byte_order + struct.pack(f"{endian}H", 42) + struct.pack(f"{endian}I", 8)

    # IFD: entry count (2) + one entry (12)
    entry_count = struct.pack(f"{endian}H", 1)
    # Tag=0x0112, Type=SHORT(3), Count=1, Value=orientation (padded to 4 bytes)
    ifd_entry = (
        struct.pack(f"{endian}H", _ORIENTATION_TAG)
        + struct.pack(f"{endian}H", 3)  # SHORT
        + struct.pack(f"{endian}I", 1)  # count
        + struct.pack(f"{endian}H", orientation)
        + b"\x00\x00"  # padding
    )

    return header + entry_count + ifd_entry


def _build_jpeg_with_orientation(orientation: int) -> bytes:
    """Build a minimal JPEG with EXIF APP1 segment containing orientation."""
    tiff = _build_tiff_with_orientation(orientation)
    exif_header = b"Exif\x00\x00"
    app1_data = exif_header + tiff
    # APP1 segment: FF E1 + length (2 bytes, includes the length field itself)
    length = len(app1_data) + 2
    app1_segment = b"\xff\xe1" + struct.pack(">H", length) + app1_data
    # SOI + APP1 + some dummy data
    return b"\xff\xd8" + app1_segment + b"\xff\xd9"


def _build_webp_with_orientation(orientation: int) -> bytes:
    """Build a minimal WebP with EXIF chunk containing orientation."""
    tiff = _build_tiff_with_orientation(orientation)
    exif_payload = b"Exif\x00\x00" + tiff

    # EXIF chunk: "EXIF" + size (LE 32-bit) + payload
    exif_chunk = b"EXIF" + struct.pack("<I", len(exif_payload)) + exif_payload
    # Pad to even
    if len(exif_payload) % 2:
        exif_chunk += b"\x00"

    # RIFF header
    file_size = 4 + len(exif_chunk)  # "WEBP" + chunks
    return b"RIFF" + struct.pack("<I", file_size) + b"WEBP" + exif_chunk


# ---------------------------------------------------------------------------
# _has_exif_header
# ---------------------------------------------------------------------------

class TestHasExifHeader:
    def test_valid(self) -> None:
        data = b"Exif\x00\x00more"
        assert _has_exif_header(data, 0) is True

    def test_at_offset(self) -> None:
        data = b"XXXXExif\x00\x00"
        assert _has_exif_header(data, 4) is True

    def test_invalid(self) -> None:
        assert _has_exif_header(b"notexif!", 0) is False


# ---------------------------------------------------------------------------
# _read_orientation_from_tiff
# ---------------------------------------------------------------------------

class TestReadOrientationFromTiff:
    def test_little_endian(self) -> None:
        tiff = _build_tiff_with_orientation(6, big_endian=False)
        assert _read_orientation_from_tiff(tiff, 0) == 6

    def test_big_endian(self) -> None:
        tiff = _build_tiff_with_orientation(3, big_endian=True)
        assert _read_orientation_from_tiff(tiff, 0) == 3

    def test_no_orientation_tag(self) -> None:
        # Build a TIFF with a different tag
        endian = "<"
        header = b"II" + struct.pack(f"{endian}H", 42) + struct.pack(f"{endian}I", 8)
        entry_count = struct.pack(f"{endian}H", 1)
        ifd_entry = (
            struct.pack(f"{endian}H", 0x0100)  # ImageWidth, not orientation
            + struct.pack(f"{endian}H", 3)
            + struct.pack(f"{endian}I", 1)
            + struct.pack(f"{endian}H", 100)
            + b"\x00\x00"
        )
        tiff = header + entry_count + ifd_entry
        assert _read_orientation_from_tiff(tiff, 0) == 1

    def test_truncated_data(self) -> None:
        assert _read_orientation_from_tiff(b"\x00\x00", 0) == 1

    def test_bad_byte_order(self) -> None:
        data = b"XX" + b"\x00" * 20
        assert _read_orientation_from_tiff(data, 0) == 1

    def test_out_of_range_value(self) -> None:
        tiff = _build_tiff_with_orientation(99, big_endian=False)
        # Manually poke value 99 — our builder sets it, but the parser should clamp
        assert _read_orientation_from_tiff(tiff, 0) == 1


# ---------------------------------------------------------------------------
# get_exif_orientation (JPEG)
# ---------------------------------------------------------------------------

class TestGetExifOrientationJpeg:
    def test_orientation_6(self) -> None:
        jpeg = _build_jpeg_with_orientation(6)
        assert get_exif_orientation(jpeg) == 6

    def test_orientation_1(self) -> None:
        jpeg = _build_jpeg_with_orientation(1)
        assert get_exif_orientation(jpeg) == 1

    def test_all_orientations(self) -> None:
        for v in range(1, 9):
            jpeg = _build_jpeg_with_orientation(v)
            assert get_exif_orientation(jpeg) == v

    def test_too_short(self) -> None:
        assert get_exif_orientation(b"\xff") == 1

    def test_empty(self) -> None:
        assert get_exif_orientation(b"") == 1

    def test_not_jpeg_or_webp(self) -> None:
        assert get_exif_orientation(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == 1


# ---------------------------------------------------------------------------
# get_exif_orientation (WebP)
# ---------------------------------------------------------------------------

class TestGetExifOrientationWebp:
    def test_orientation_3(self) -> None:
        webp = _build_webp_with_orientation(3)
        assert get_exif_orientation(webp) == 3

    def test_orientation_8(self) -> None:
        webp = _build_webp_with_orientation(8)
        assert get_exif_orientation(webp) == 8


# ---------------------------------------------------------------------------
# _find_jpeg_tiff_offset
# ---------------------------------------------------------------------------

class TestFindJpegTiffOffset:
    def test_finds_offset(self) -> None:
        jpeg = _build_jpeg_with_orientation(6)
        offset = _find_jpeg_tiff_offset(jpeg)
        assert offset > 0

    def test_no_app1(self) -> None:
        # SOI + SOS (no APP1)
        data = b"\xff\xd8\xff\xda\x00\x02"
        assert _find_jpeg_tiff_offset(data) == -1

    def test_non_exif_app1(self) -> None:
        # APP1 with non-EXIF data
        payload = b"NOTEXIF!"
        length = len(payload) + 2
        data = b"\xff\xd8\xff\xe1" + struct.pack(">H", length) + payload
        assert _find_jpeg_tiff_offset(data) == -1

    def test_ff_padding(self) -> None:
        # Consecutive FF bytes should be skipped
        data = b"\xff\xd8\xff\xff\xff\xe1"
        # Should handle padding without crashing (but won't find valid EXIF)
        result = _find_jpeg_tiff_offset(data)
        assert result == -1  # too short for valid EXIF header

    def test_skips_other_segments(self) -> None:
        # SOI + APP0 (short) + APP1 with EXIF
        app0_payload = b"\x00" * 4
        app0_len = len(app0_payload) + 2
        app0 = b"\xff\xe0" + struct.pack(">H", app0_len) + app0_payload

        tiff = _build_tiff_with_orientation(5)
        exif_header = b"Exif\x00\x00"
        app1_data = exif_header + tiff
        app1_len = len(app1_data) + 2
        app1 = b"\xff\xe1" + struct.pack(">H", app1_len) + app1_data

        data = b"\xff\xd8" + app0 + app1
        offset = _find_jpeg_tiff_offset(data)
        assert offset > 0

    def test_truncated_at_marker(self) -> None:
        data = b"\xff\xd8\xff"
        assert _find_jpeg_tiff_offset(data) == -1

    def test_non_ff_byte(self) -> None:
        data = b"\xff\xd8\x00"
        assert _find_jpeg_tiff_offset(data) == -1


# ---------------------------------------------------------------------------
# _find_webp_tiff_offset
# ---------------------------------------------------------------------------

class TestFindWebpTiffOffset:
    def test_finds_offset(self) -> None:
        webp = _build_webp_with_orientation(5)
        offset = _find_webp_tiff_offset(webp)
        assert offset > 0

    def test_no_exif_chunk(self) -> None:
        # Minimal RIFF/WEBP with VP8 chunk (no EXIF)
        vp8 = b"VP8 " + struct.pack("<I", 4) + b"\x00\x00\x00\x00"
        file_size = 4 + len(vp8)
        data = b"RIFF" + struct.pack("<I", file_size) + b"WEBP" + vp8
        assert _find_webp_tiff_offset(data) == -1


# ---------------------------------------------------------------------------
# apply_exif_orientation
# ---------------------------------------------------------------------------

class TestApplyExifOrientation:
    def test_orientation_1_returns_same(self) -> None:
        img = Image.new("RGB", (10, 20), "red")
        result = apply_exif_orientation(img, original_bytes=_build_jpeg_with_orientation(1))
        assert result.size == (10, 20)

    def test_orientation_6_rotates(self) -> None:
        """ImageOps.exif_transpose reads the Image's own EXIF, not original_bytes.
        Since we pass a plain Image with no embedded EXIF, PIL returns the image
        unchanged. The function still detects orientation != 1 from original_bytes
        and calls exif_transpose, but PIL can't rotate without EXIF in the Image.
        We mock exif_transpose to simulate a real JPEG with embedded EXIF."""
        img = Image.new("RGB", (10, 20), "red")
        rotated = img.transpose(Image.Transpose.ROTATE_270)
        from unittest.mock import patch

        with patch("nu_coding_agent.utils.exif_orientation.ImageOps.exif_transpose", return_value=rotated):
            result = apply_exif_orientation(img, original_bytes=_build_jpeg_with_orientation(6))
            assert result.size == (20, 10)

    def test_orientation_3_exif_transpose_returns_none_manual_fallback(self) -> None:
        """When ImageOps.exif_transpose returns None, manual fallback kicks in."""
        img = Image.new("RGB", (10, 20), "red")
        from unittest.mock import patch

        with patch("nu_coding_agent.utils.exif_orientation.ImageOps.exif_transpose", return_value=None):
            result = apply_exif_orientation(img, original_bytes=_build_jpeg_with_orientation(3))
            assert result.size == (10, 20)  # 180 rotation preserves dimensions

    def test_orientation_8_manual_fallback(self) -> None:
        img = Image.new("RGB", (10, 20), "red")
        from unittest.mock import patch

        with patch("nu_coding_agent.utils.exif_orientation.ImageOps.exif_transpose", return_value=None):
            result = apply_exif_orientation(img, original_bytes=_build_jpeg_with_orientation(8))
            assert result.size == (20, 10)

    def test_uses_pil_exif_when_no_bytes(self) -> None:
        img = Image.new("RGB", (10, 20), "red")
        # No EXIF in a plain new image -> orientation 1 -> returns same
        result = apply_exif_orientation(img)
        assert result.size == (10, 20)

    def test_orientation_2_manual_fallback(self) -> None:
        img = Image.new("RGB", (10, 20), "red")
        from unittest.mock import patch

        with patch("nu_coding_agent.utils.exif_orientation.ImageOps.exif_transpose", return_value=None):
            result = apply_exif_orientation(img, original_bytes=_build_jpeg_with_orientation(2))
            assert result.size == (10, 20)  # flip doesn't change dimensions

    def test_exif_transpose_exception_falls_to_manual(self) -> None:
        img = Image.new("RGB", (10, 20), "red")
        from unittest.mock import patch

        with patch("nu_coding_agent.utils.exif_orientation.ImageOps.exif_transpose", side_effect=Exception("oops")):
            result = apply_exif_orientation(img, original_bytes=_build_jpeg_with_orientation(5))
            # orientation 5 = TRANSPOSE: swaps dimensions
            assert result.size == (20, 10)

    def test_all_manual_fallback_orientations(self) -> None:
        """Exercise every manual fallback path (orientations 2-8)."""
        from unittest.mock import patch

        for orient in range(2, 9):
            img = Image.new("RGB", (10, 20), "red")
            with patch("nu_coding_agent.utils.exif_orientation.ImageOps.exif_transpose", return_value=None):
                result = apply_exif_orientation(img, original_bytes=_build_jpeg_with_orientation(orient))
                assert result is not None, f"orientation {orient} returned None"
