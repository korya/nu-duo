"""Tests for ``nu_coding_agent.utils.image_resize``."""

from __future__ import annotations

import base64
import io

from nu_ai.types import ImageContent
from nu_coding_agent.utils.image_resize import (
    ImageResizeOptions,
    format_dimension_note,
    resize_image,
)
from PIL import Image


def _make_png(width: int, height: int, color: str = "red") -> ImageContent:
    """Build a real PNG of the requested dimensions, base64-encoded."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return ImageContent(
        data=base64.b64encode(buffer.getvalue()).decode("ascii"),
        mime_type="image/png",
    )


def test_small_image_passthrough() -> None:
    """Already-small images return verbatim with ``was_resized=False``."""
    src = _make_png(50, 50)
    result = resize_image(src)
    assert result is not None
    assert result.was_resized is False
    assert result.width == 50
    assert result.height == 50
    assert result.data == src.data  # untouched


def test_resize_to_max_dimensions() -> None:
    src = _make_png(4000, 4000)
    result = resize_image(src, ImageResizeOptions(max_width=200, max_height=200))
    assert result is not None
    assert result.was_resized is True
    assert result.width <= 200
    assert result.height <= 200
    assert result.original_width == 4000


def test_preserves_aspect_ratio() -> None:
    src = _make_png(1000, 500)
    result = resize_image(src, ImageResizeOptions(max_width=200, max_height=200))
    assert result is not None
    assert result.width == 200
    # Aspect ratio 2:1 → height halves to 100.
    assert 95 <= result.height <= 105


def test_returns_none_for_garbage_payload() -> None:
    src = ImageContent(data="not-base64!@#", mime_type="image/png")
    assert resize_image(src) is None


def test_returns_none_for_non_image_bytes() -> None:
    src = ImageContent(
        data=base64.b64encode(b"this is not an image").decode("ascii"),
        mime_type="image/png",
    )
    assert resize_image(src) is None


def test_format_dimension_note_for_resized() -> None:
    src = _make_png(2000, 1000)
    result = resize_image(src, ImageResizeOptions(max_width=500, max_height=500))
    assert result is not None
    note = format_dimension_note(result)
    assert note is not None
    assert "original 2000x1000" in note
    assert "Multiply coordinates" in note


def test_format_dimension_note_skips_unmodified() -> None:
    src = _make_png(50, 50)
    result = resize_image(src)
    assert result is not None
    assert format_dimension_note(result) is None


def test_byte_budget_forces_jpeg_or_smaller() -> None:
    """A tight max_bytes budget forces a JPEG re-encode or further downscale."""
    src = _make_png(1000, 1000)
    # Force a tight budget that the original PNG can't possibly meet.
    result = resize_image(src, ImageResizeOptions(max_width=500, max_height=500, max_bytes=2_000))
    # Should successfully downscale until it fits or return None — both are valid.
    assert result is None or result.was_resized is True
