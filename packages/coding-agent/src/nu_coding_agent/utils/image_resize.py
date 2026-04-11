"""Image resize helper — port of ``packages/coding-agent/src/utils/image-resize.ts``.

The TS upstream uses Photon (a Rust/WASM image library). The Python
port replaces it with Pillow, which already handles EXIF orientation,
PNG/JPEG encoding, and Lanczos resampling natively. The resize +
encode strategy mirrors upstream byte-for-byte:

1. Resize to ``max_width`` / ``max_height`` preserving aspect ratio.
2. Encode as PNG and as JPEG at decreasing qualities; pick the
   smallest one that fits under ``max_bytes`` (size of the base64
   payload).
3. If still too large, halve dimensions (x0.75) and try again until
   1x1.
4. Return ``None`` if the encoded payload can't fit even at 1x1.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image, ImageOps

if TYPE_CHECKING:
    from nu_ai.types import ImageContent


# 4.5MB of base64 payload — just under Anthropic's 5MB attachment limit.
_DEFAULT_MAX_BYTES = int(4.5 * 1024 * 1024)


@dataclass(slots=True)
class ImageResizeOptions:
    max_width: int = 2000
    max_height: int = 2000
    max_bytes: int = _DEFAULT_MAX_BYTES
    jpeg_quality: int = 80


@dataclass(slots=True)
class ResizedImage:
    data: str
    """Base64-encoded payload."""
    mime_type: str
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool


@dataclass(slots=True)
class _EncodedCandidate:
    data: str
    encoded_size: int
    mime_type: str


def _encode(buffer: bytes, mime_type: str) -> _EncodedCandidate:
    encoded = base64.b64encode(buffer).decode("ascii")
    return _EncodedCandidate(data=encoded, encoded_size=len(encoded), mime_type=mime_type)


def _try_encodings(image: Image.Image, qualities: list[int]) -> list[_EncodedCandidate]:
    candidates: list[_EncodedCandidate] = []
    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG")
    candidates.append(_encode(png_buffer.getvalue(), "image/png"))
    rgb_image = image.convert("RGB") if image.mode not in ("RGB", "L") else image
    for quality in qualities:
        jpeg_buffer = io.BytesIO()
        rgb_image.save(jpeg_buffer, format="JPEG", quality=quality)
        candidates.append(_encode(jpeg_buffer.getvalue(), "image/jpeg"))
    return candidates


def resize_image(
    image_content: ImageContent,
    options: ImageResizeOptions | None = None,
) -> ResizedImage | None:
    """Resize ``image_content`` to fit within ``options``' limits.

    Returns ``None`` when the input is unreadable or the encoded payload
    can't be brought under ``max_bytes`` even at 1x1. The Pillow round
    trip preserves the bytes verbatim when the image is already within
    every limit (no re-encoding cost in the common case).
    """
    opts = options or ImageResizeOptions()
    try:
        input_buffer = base64.b64decode(image_content.data)
    except (ValueError, TypeError):
        return None

    input_base64_size = len(image_content.data)

    try:
        raw_image = Image.open(io.BytesIO(input_buffer))
        # Honour EXIF orientation so portrait phone photos don't end up sideways.
        image = ImageOps.exif_transpose(raw_image) or raw_image
    except (OSError, ValueError):
        return None

    original_width, original_height = image.size
    declared_format = (image_content.mime_type or "").split("/")[-1] or "png"

    # Already within every limit — return the original payload untouched.
    if original_width <= opts.max_width and original_height <= opts.max_height and input_base64_size < opts.max_bytes:
        return ResizedImage(
            data=image_content.data,
            mime_type=image_content.mime_type or f"image/{declared_format}",
            original_width=original_width,
            original_height=original_height,
            width=original_width,
            height=original_height,
            was_resized=False,
        )

    # Compute initial target dimensions.
    target_width = original_width
    target_height = original_height
    if target_width > opts.max_width:
        target_height = round(target_height * opts.max_width / target_width)
        target_width = opts.max_width
    if target_height > opts.max_height:
        target_width = round(target_width * opts.max_height / target_height)
        target_height = opts.max_height

    quality_steps = list(dict.fromkeys([opts.jpeg_quality, 85, 70, 55, 40]))
    current_width, current_height = target_width, target_height

    while True:
        resized = image.resize((current_width, current_height), Image.Resampling.LANCZOS)
        try:
            for candidate in _try_encodings(resized, quality_steps):
                if candidate.encoded_size < opts.max_bytes:
                    return ResizedImage(
                        data=candidate.data,
                        mime_type=candidate.mime_type,
                        original_width=original_width,
                        original_height=original_height,
                        width=current_width,
                        height=current_height,
                        was_resized=True,
                    )
        finally:
            resized.close()

        if current_width == 1 and current_height == 1:
            break
        next_width = 1 if current_width == 1 else max(1, int(current_width * 0.75))
        next_height = 1 if current_height == 1 else max(1, int(current_height * 0.75))
        if next_width == current_width and next_height == current_height:
            break
        current_width, current_height = next_width, next_height

    return None


def format_dimension_note(result: ResizedImage) -> str | None:
    """Render the "image was scaled by 1.42x" hint the upstream tacks onto the prompt."""
    if not result.was_resized:
        return None
    scale = result.original_width / result.width
    return (
        f"[Image: original {result.original_width}x{result.original_height}, "
        f"displayed at {result.width}x{result.height}. "
        f"Multiply coordinates by {scale:.2f} to map to original image.]"
    )


__all__ = [
    "ImageResizeOptions",
    "ResizedImage",
    "format_dimension_note",
    "resize_image",
]
