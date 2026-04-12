"""Shared truncation utilities for mom tool outputs.

Port of ``packages/mom/src/tools/truncate.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_tail",
]

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50 KB


@dataclass(slots=True)
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


def truncate_head(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep first N lines/bytes (for file reads)."""
    total_bytes = _byte_len(content)
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
        )

    first_line_bytes = _byte_len(lines[0])
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
        )

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = _byte_len(line) + (1 if i > 0 else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        out_lines.append(line)
        out_bytes += line_bytes

    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = "lines"

    out_content = "\n".join(out_lines)
    return TruncationResult(
        content=out_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(out_lines),
        output_bytes=_byte_len(out_content),
        last_line_partial=False,
        first_line_exceeds_limit=False,
    )


def truncate_tail(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep last N lines/bytes (for bash output)."""
    total_bytes = _byte_len(content)
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
        )

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(out_lines) >= max_lines:
            break
        line = lines[i]
        line_bytes = _byte_len(line) + (1 if out_lines else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not out_lines:
                # Edge case: single line exceeds limit — take end of it
                buf = line.encode("utf-8")
                trimmed = buf[-max_bytes:].decode("utf-8", errors="ignore")
                out_lines.insert(0, trimmed)
                out_bytes = _byte_len(trimmed)
                last_line_partial = True
            break
        out_lines.insert(0, line)
        out_bytes += line_bytes

    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = "lines"

    out_content = "\n".join(out_lines)
    return TruncationResult(
        content=out_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(out_lines),
        output_bytes=_byte_len(out_content),
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
    )
