"""Tests for ``nu_coding_agent.file_processor``."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nu_coding_agent.file_processor import ProcessedFiles, process_file_arguments


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    p = tmp_path / "hello.txt"
    p.write_text("Hello, world!", encoding="utf-8")
    return p


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    """Create a minimal valid PNG file."""
    from PIL import Image
    import io

    p = tmp_path / "test.png"
    img = Image.new("RGB", (10, 10), "red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


@pytest.mark.asyncio
async def test_text_file(text_file: Path) -> None:
    result = await process_file_arguments([str(text_file)], auto_resize=False)
    assert isinstance(result, ProcessedFiles)
    assert "Hello, world!" in result.text
    assert f'<file name="{text_file}">' in result.text
    assert result.images == []


@pytest.mark.asyncio
async def test_empty_file_skipped(empty_file: Path) -> None:
    result = await process_file_arguments([str(empty_file)], auto_resize=False)
    assert result.text == ""
    assert result.images == []


@pytest.mark.asyncio
async def test_nonexistent_file_exits() -> None:
    with pytest.raises(SystemExit):
        await process_file_arguments(["/tmp/nonexistent_test_file_12345.txt"], auto_resize=False)


@pytest.mark.asyncio
async def test_image_file_no_resize(png_file: Path) -> None:
    result = await process_file_arguments([str(png_file)], auto_resize=False)
    assert len(result.images) == 1
    assert result.images[0]["type"] == "image"
    assert result.images[0]["mime_type"] == "image/png"
    # Verify the data is valid base64
    decoded = base64.b64decode(result.images[0]["data"])
    assert len(decoded) > 0
    assert f'<file name="{png_file}">' in result.text


@pytest.mark.asyncio
async def test_image_file_with_resize(png_file: Path) -> None:
    result = await process_file_arguments([str(png_file)], auto_resize=True)
    assert len(result.images) == 1
    assert result.images[0]["type"] == "image"
    assert result.images[0]["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_multiple_files(text_file: Path, png_file: Path) -> None:
    result = await process_file_arguments(
        [str(text_file), str(png_file)], auto_resize=False
    )
    assert "Hello, world!" in result.text
    assert len(result.images) == 1


@pytest.mark.asyncio
async def test_image_resize_returns_none(png_file: Path) -> None:
    """When resize_image returns None, image is omitted with a note."""
    with patch("nu_coding_agent.file_processor.resize_image", return_value=None):
        result = await process_file_arguments([str(png_file)], auto_resize=True)
        assert result.images == []
        assert "could not be resized" in result.text


@pytest.mark.asyncio
async def test_binary_file_read_error(tmp_path: Path) -> None:
    """When a non-image file can't be read as UTF-8, exits."""
    # Write bytes that are not valid UTF-8 and don't match any image magic bytes.
    # Start with something that's not JPEG/PNG/GIF/WebP magic.
    p = tmp_path / "binary.dat"
    p.write_bytes(b"\x80\x81\x82\x83" + bytes(range(128, 256)) * 10)
    with pytest.raises(SystemExit):
        await process_file_arguments([str(p)], auto_resize=False)
