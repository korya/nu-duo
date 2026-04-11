"""Tests for ``nu_coding_agent.utils.mime``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nu_coding_agent.utils.mime import detect_supported_image_mime_type_from_file

if TYPE_CHECKING:
    from pathlib import Path


# Tiny valid image headers — magic bytes only, body is junk that the
# sniffer never reads. The sniffer only inspects the first ~4 KB so a
# few bytes is enough.

_PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
_JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 10
_GIF87_HEADER = b"GIF87a" + b"\x00" * 10
_GIF89_HEADER = b"GIF89a" + b"\x00" * 10
_WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10
_BMP_HEADER = b"BM\x00\x00\x00\x00\x00\x00"  # not in the whitelist


async def test_detects_png(tmp_path: Path) -> None:
    f = tmp_path / "x.png"
    f.write_bytes(_PNG_HEADER)
    assert await detect_supported_image_mime_type_from_file(str(f)) == "image/png"


async def test_detects_jpeg(tmp_path: Path) -> None:
    f = tmp_path / "x.jpg"
    f.write_bytes(_JPEG_HEADER)
    assert await detect_supported_image_mime_type_from_file(str(f)) == "image/jpeg"


async def test_detects_gif87(tmp_path: Path) -> None:
    f = tmp_path / "x.gif"
    f.write_bytes(_GIF87_HEADER)
    assert await detect_supported_image_mime_type_from_file(str(f)) == "image/gif"


async def test_detects_gif89(tmp_path: Path) -> None:
    f = tmp_path / "x.gif"
    f.write_bytes(_GIF89_HEADER)
    assert await detect_supported_image_mime_type_from_file(str(f)) == "image/gif"


async def test_detects_webp(tmp_path: Path) -> None:
    f = tmp_path / "x.webp"
    f.write_bytes(_WEBP_HEADER)
    assert await detect_supported_image_mime_type_from_file(str(f)) == "image/webp"


async def test_rejects_bmp_not_in_whitelist(tmp_path: Path) -> None:
    f = tmp_path / "x.bmp"
    f.write_bytes(_BMP_HEADER)
    assert await detect_supported_image_mime_type_from_file(str(f)) is None


async def test_rejects_text_file(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello world")
    assert await detect_supported_image_mime_type_from_file(str(f)) is None


async def test_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert await detect_supported_image_mime_type_from_file(str(tmp_path / "missing")) is None


async def test_returns_none_for_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.png"
    f.write_bytes(b"")
    assert await detect_supported_image_mime_type_from_file(str(f)) is None
