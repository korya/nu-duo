"""Shared truncation utilities for tool outputs.

Direct port of ``packages/coding-agent/src/core/tools/truncate.ts``.
Truncation enforces two independent limits — the line count and the byte
count — and the first limit hit wins. Never returns partial lines except
in the bash tail-truncation edge case where the *last* line of the
original content already exceeds the byte limit.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
GREP_MAX_LINE_LENGTH = 500


@dataclass(slots=True)
class TruncationResult:
    """Result of a head- or tail-truncation operation."""

    content: str
    truncated: bool
    truncated_by: str | None  # "lines", "bytes", or None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


def format_size(byte_count: int) -> str:
    """Render ``byte_count`` as ``"123B"``, ``"1.5KB"``, or ``"2.3MB"``."""
    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / (1024 * 1024):.1f}MB"


def _bytelen(s: str) -> int:
    return len(s.encode("utf-8"))


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the first ``max_lines``/``max_bytes`` of ``content``.

    Suitable for file reads where you want the beginning. Never returns a
    partial line — if the very first line exceeds ``max_bytes`` the result
    has empty content and ``first_line_exceeds_limit=True``.
    """
    total_bytes = _bytelen(content)
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
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line_bytes = _bytelen(lines[0])
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
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: list[str] = []
    output_bytes_count = 0
    truncated_by: str = "lines"

    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = _bytelen(line) + (1 if i > 0 else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes_count += line_bytes

    if len(output_lines) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    final_output_bytes = _bytelen(output_content)

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=final_output_bytes,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the last ``max_lines``/``max_bytes`` of ``content``.

    Suitable for bash output where you want the end (errors, final
    results). May return a partial first line if the original content's
    last line alone exceeds the byte limit.
    """
    total_bytes = _bytelen(content)
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
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: list[str] = []
    output_bytes_count = 0
    truncated_by: str = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(output_lines) >= max_lines:
            break
        line = lines[i]
        line_bytes = _bytelen(line) + (1 if output_lines else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                truncated_line = _truncate_bytes_from_end(line, max_bytes)
                output_lines.insert(0, truncated_line)
                output_bytes_count = _bytelen(truncated_line)
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes_count += line_bytes

    if len(output_lines) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    final_output_bytes = _bytelen(output_content)

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=final_output_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_bytes_from_end(s: str, max_bytes: int) -> str:
    """Return the suffix of ``s`` whose UTF-8 encoding fits in ``max_bytes``.

    Walks the byte sequence forward from the candidate start until it
    lands on a UTF-8 character boundary so multi-byte chars stay intact.
    """
    buf = s.encode("utf-8")
    if len(buf) <= max_bytes:
        return s
    start = len(buf) - max_bytes
    while start < len(buf) and (buf[start] & 0xC0) == 0x80:
        start += 1
    return buf[start:].decode("utf-8")


def truncate_line(
    line: str,
    max_chars: int = GREP_MAX_LINE_LENGTH,
) -> tuple[str, bool]:
    """Truncate a single line to ``max_chars``, appending ``... [truncated]``."""
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_LENGTH",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
]
