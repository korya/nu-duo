"""Process ``@file`` CLI arguments into text content and image attachments.

Direct port of ``packages/coding-agent/src/cli/file-processor.ts``.

Each ``@path`` argument on the CLI becomes either:
- **text** — wrapped in ``<file name="…">…</file>`` tags, or
- **image** — base64-encoded with optional resize, plus an XML reference
  in the text stream so the model knows the image's filename.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nu_coding_agent.core.tools.path_utils import resolve_read_path
from nu_coding_agent.utils.image_resize import (
    format_dimension_note,
    resize_image,
)
from nu_coding_agent.utils.mime import detect_supported_image_mime_type_from_file


@dataclass(slots=True)
class ProcessedFiles:
    """Combined result of processing one or more ``@file`` arguments."""

    text: str = ""
    images: list[dict[str, Any]] = field(default_factory=list)


async def process_file_arguments(
    file_paths: list[str],
    *,
    auto_resize: bool = True,
) -> ProcessedFiles:
    """Read each file, detecting images vs text.

    Parameters
    ----------
    file_paths:
        Raw paths from the CLI (already stripped of the leading ``@``).
    auto_resize:
        When ``True`` (default), images are resized to fit within 2000x2000
        and 4.5 MB of base64 payload — matching the upstream behaviour.

    Returns
    -------
    ProcessedFiles
        ``.text`` contains the concatenated ``<file>`` XML blocks.
        ``.images`` contains the ``ImageContent``-shaped dicts ready to be
        sent as message content blocks.
    """
    cwd = str(Path.cwd())
    text_parts: list[str] = []
    images: list[dict[str, Any]] = []

    for file_path in file_paths:
        # Resolve tilde, macOS Unicode spaces, etc.
        resolved = await asyncio.to_thread(lambda p=file_path: str(Path(resolve_read_path(p, cwd)).resolve()))
        absolute_path = resolved

        # Check existence
        path_obj = Path(absolute_path)
        if not await asyncio.to_thread(path_obj.exists):
            print(f"Error: File not found: {absolute_path}", file=sys.stderr)
            sys.exit(1)

        # Skip empty files
        try:
            stat = await asyncio.to_thread(path_obj.stat)
        except OSError:
            print(f"Error: Could not stat file {absolute_path}", file=sys.stderr)
            sys.exit(1)

        if stat.st_size == 0:
            continue

        # Sniff MIME type
        mime_type = await detect_supported_image_mime_type_from_file(absolute_path)

        if mime_type:
            # --- image file ---
            raw_bytes = await asyncio.to_thread(path_obj.read_bytes)
            b64_data = base64.b64encode(raw_bytes).decode("ascii")

            if auto_resize:
                from nu_ai.types import ImageContent as ImageContentCls  # noqa: PLC0415

                image_content = ImageContentCls(data=b64_data, mime_type=mime_type)
                resized = resize_image(image_content)
                if resized is None:
                    text_parts.append(
                        f'<file name="{absolute_path}">'
                        "[Image omitted: could not be resized below the inline image size limit.]"
                        "</file>\n"
                    )
                    continue
                dimension_note = format_dimension_note(resized)
                attachment: dict[str, Any] = {
                    "type": "image",
                    "mime_type": resized.mime_type,
                    "data": resized.data,
                }
            else:
                dimension_note = None
                attachment = {
                    "type": "image",
                    "mime_type": mime_type,
                    "data": b64_data,
                }

            images.append(attachment)

            if dimension_note:
                text_parts.append(f'<file name="{absolute_path}">{dimension_note}</file>\n')
            else:
                text_parts.append(f'<file name="{absolute_path}"></file>\n')
        else:
            # --- text file ---
            try:
                content = await asyncio.to_thread(path_obj.read_text, "utf-8")
                text_parts.append(f'<file name="{absolute_path}">\n{content}\n</file>\n')
            except (OSError, UnicodeDecodeError) as exc:
                print(f"Error: Could not read file {absolute_path}: {exc}", file=sys.stderr)
                sys.exit(1)

    return ProcessedFiles(text="".join(text_parts), images=images)


__all__ = [
    "ProcessedFiles",
    "process_file_arguments",
]
