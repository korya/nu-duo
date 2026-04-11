"""Tests for nu_coding_agent.core.tools.truncate."""

from __future__ import annotations

import pytest
from nu_coding_agent.core.tools.truncate import (
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
)


class TestFormatSize:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [(100, "100B"), (2048, "2.0KB"), (5 * 1024 * 1024, "5.0MB")],
    )
    def test_basic(self, size: int, expected: str) -> None:
        assert format_size(size) == expected


class TestTruncateHead:
    def test_no_truncation_when_within_limits(self) -> None:
        result = truncate_head("a\nb\nc")
        assert result.truncated is False
        assert result.content == "a\nb\nc"
        assert result.total_lines == 3

    def test_line_limit_truncates(self) -> None:
        content = "\n".join(str(i) for i in range(10))
        result = truncate_head(content, max_lines=5)
        assert result.truncated is True
        assert result.truncated_by == "lines"
        assert result.output_lines == 5

    def test_byte_limit_truncates(self) -> None:
        content = "\n".join("xxxxx" for _ in range(100))
        result = truncate_head(content, max_bytes=20)
        assert result.truncated is True
        assert result.truncated_by == "bytes"
        assert len(result.content.encode("utf-8")) <= 20

    def test_first_line_exceeds_limit(self) -> None:
        content = "x" * 1000 + "\nshort"
        result = truncate_head(content, max_bytes=100)
        assert result.first_line_exceeds_limit is True
        assert result.content == ""


class TestTruncateTail:
    def test_no_truncation_when_within_limits(self) -> None:
        result = truncate_tail("a\nb\nc")
        assert result.truncated is False
        assert result.content == "a\nb\nc"

    def test_keeps_last_n_lines(self) -> None:
        content = "\n".join(str(i) for i in range(10))
        result = truncate_tail(content, max_lines=3)
        assert result.truncated is True
        assert result.content == "7\n8\n9"

    def test_partial_last_line_when_too_long(self) -> None:
        content = "x" * 200
        result = truncate_tail(content, max_bytes=50)
        assert result.truncated is True
        assert result.last_line_partial is True
        assert len(result.content.encode("utf-8")) <= 50


class TestTruncateLine:
    def test_short_line_unchanged(self) -> None:
        text, was = truncate_line("short", max_chars=100)
        assert text == "short"
        assert was is False

    def test_long_line_truncated(self) -> None:
        text, was = truncate_line("x" * 200, max_chars=100)
        assert was is True
        assert text.endswith("[truncated]")
        assert text.startswith("x" * 100)
