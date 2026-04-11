"""Tests for ``nu_coding_agent.utils.image_convert``."""

from __future__ import annotations

import base64
import io

from nu_coding_agent.utils.image_convert import convert_to_png
from PIL import Image


def _png_b64(width: int = 4, height: int = 4) -> str:
    img = Image.new("RGB", (width, height), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _jpeg_b64(width: int = 4, height: int = 4) -> str:
    img = Image.new("RGB", (width, height), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_passthrough_png() -> None:
    data = _png_b64()
    result = convert_to_png(data, "image/png")
    assert result is not None
    assert result.data == data
    assert result.mime_type == "image/png"


def test_converts_jpeg_to_png() -> None:
    data = _jpeg_b64()
    result = convert_to_png(data, "image/jpeg")
    assert result is not None
    assert result.mime_type == "image/png"
    decoded = base64.b64decode(result.data)
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_invalid_base64_returns_none() -> None:
    assert convert_to_png("not-base64!@#", "image/jpeg") is None


def test_non_image_bytes_returns_none() -> None:
    junk = base64.b64encode(b"not an image at all").decode("ascii")
    assert convert_to_png(junk, "image/jpeg") is None
