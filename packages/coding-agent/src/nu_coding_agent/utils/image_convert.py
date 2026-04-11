"""Image format conversion — port of ``packages/coding-agent/src/utils/image-convert.ts``.

The TS upstream uses Photon (Rust/WASM) to round-trip a base64 image
into PNG bytes for the Kitty terminal graphics protocol. The Python
port replaces it with Pillow, which already understands every format
we care about.

Returns ``None`` on any decode/encode failure so callers can fall
through to a textual representation instead of crashing the renderer.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass(slots=True)
class ConvertedImage:
    data: str
    """Base64-encoded PNG payload."""
    mime_type: str = "image/png"


def convert_to_png(base64_data: str, mime_type: str) -> ConvertedImage | None:
    """Convert ``base64_data`` to a PNG, returning ``None`` on failure.

    Returns the input untouched when ``mime_type`` is already
    ``image/png`` to avoid an unnecessary decode/encode round-trip.
    """
    if mime_type == "image/png":
        return ConvertedImage(data=base64_data, mime_type=mime_type)
    try:
        decoded = base64.b64decode(base64_data)
    except (ValueError, TypeError):
        return None
    try:
        raw_image = Image.open(io.BytesIO(decoded))
        image = ImageOps.exif_transpose(raw_image) or raw_image
    except (OSError, ValueError):
        return None
    try:
        out = io.BytesIO()
        image.save(out, format="PNG")
    except (OSError, ValueError):
        return None
    finally:
        image.close()
    return ConvertedImage(data=base64.b64encode(out.getvalue()).decode("ascii"), mime_type="image/png")


__all__ = ["ConvertedImage", "convert_to_png"]
