"""Tests for the ``edit`` AgentTool and edit-diff helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pi_ai.types import TextContent
from pi_coding_agent.core.tools.edit import create_edit_tool

if TYPE_CHECKING:
    from pathlib import Path
from pi_coding_agent.core.tools.edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    fuzzy_find_text,
    generate_diff_string,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)


class TestLineEndings:
    def test_lf_only(self) -> None:
        assert detect_line_ending("a\nb\nc") == "\n"

    def test_crlf(self) -> None:
        assert detect_line_ending("a\r\nb\r\nc") == "\r\n"

    def test_normalize_to_lf(self) -> None:
        assert normalize_to_lf("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_restore_crlf(self) -> None:
        assert restore_line_endings("a\nb", "\r\n") == "a\r\nb"

    def test_restore_lf_noop(self) -> None:
        assert restore_line_endings("a\nb", "\n") == "a\nb"


class TestStripBom:
    def test_no_bom(self) -> None:
        bom, text = strip_bom("hello")
        assert bom == ""
        assert text == "hello"

    def test_with_bom(self) -> None:
        bom, text = strip_bom("\ufeffhello")
        assert bom == "\ufeff"
        assert text == "hello"


class TestNormalizeForFuzzyMatch:
    def test_smart_quotes(self) -> None:
        assert normalize_for_fuzzy_match("\u201chello\u201d") == '"hello"'
        assert normalize_for_fuzzy_match("don\u2019t") == "don't"

    def test_dashes(self) -> None:
        assert normalize_for_fuzzy_match("a\u2014b") == "a-b"

    def test_special_spaces(self) -> None:
        assert normalize_for_fuzzy_match("a\u00a0b") == "a b"

    def test_strip_trailing_whitespace_per_line(self) -> None:
        assert normalize_for_fuzzy_match("a   \nb\n") == "a\nb\n"


class TestFuzzyFindText:
    def test_exact_match(self) -> None:
        result = fuzzy_find_text("hello world", "world")
        assert result.found is True
        assert result.used_fuzzy_match is False
        assert result.index == 6

    def test_fuzzy_match_smart_quotes(self) -> None:
        # Source has a curly quote, query has a straight quote.
        result = fuzzy_find_text("don\u2019t go", "don't")
        assert result.found is True
        assert result.used_fuzzy_match is True

    def test_not_found(self) -> None:
        result = fuzzy_find_text("hello", "missing")
        assert result.found is False


class TestApplyEdits:
    def test_single_edit(self) -> None:
        result = apply_edits_to_normalized_content(
            "line1\nline2\nline3",
            [Edit(old_text="line2", new_text="LINE2")],
            "f.txt",
        )
        assert result.new_content == "line1\nLINE2\nline3"

    def test_multi_edit_disjoint(self) -> None:
        result = apply_edits_to_normalized_content(
            "alpha\nbeta\ngamma",
            [
                Edit(old_text="alpha", new_text="A"),
                Edit(old_text="gamma", new_text="G"),
            ],
            "f.txt",
        )
        assert result.new_content == "A\nbeta\nG"

    def test_overlapping_edits_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            apply_edits_to_normalized_content(
                "abcdef",
                [
                    Edit(old_text="abcd", new_text="X"),
                    Edit(old_text="cdef", new_text="Y"),
                ],
                "f.txt",
            )

    def test_duplicate_old_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="occurrences"):
            apply_edits_to_normalized_content(
                "foo\nfoo",
                [Edit(old_text="foo", new_text="bar")],
                "f.txt",
            )

    def test_old_text_not_found_rejected(self) -> None:
        with pytest.raises(ValueError, match="Could not find"):
            apply_edits_to_normalized_content(
                "hello",
                [Edit(old_text="missing", new_text="x")],
                "f.txt",
            )

    def test_empty_old_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            apply_edits_to_normalized_content(
                "hello",
                [Edit(old_text="", new_text="x")],
                "f.txt",
            )

    def test_no_change_rejected(self) -> None:
        with pytest.raises(ValueError, match="No changes"):
            apply_edits_to_normalized_content(
                "hello",
                [Edit(old_text="hello", new_text="hello")],
                "f.txt",
            )


class TestGenerateDiff:
    def test_simple_change(self) -> None:
        result = generate_diff_string("a\nb\nc", "a\nB\nc")
        # Should contain a +/- line for the changed line.
        assert any(line.startswith("+") for line in result.diff.split("\n"))
        assert any(line.startswith("-") for line in result.diff.split("\n"))
        assert result.first_changed_line == 2

    def test_pure_addition(self) -> None:
        result = generate_diff_string("a\nb", "a\nb\nc")
        assert "+" in result.diff
        assert result.first_changed_line == 3


class TestEditTool:
    async def test_basic_edit(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hello world")
        tool = create_edit_tool(str(tmp_path))
        result = await tool.execute(
            "c1",
            {
                "path": "f.txt",
                "edits": [{"oldText": "hello", "newText": "goodbye"}],
            },
        )
        assert f.read_text() == "goodbye world"
        assert isinstance(result.content[0], TextContent)
        assert "Successfully replaced 1 block" in result.content[0].text

    async def test_multi_edit(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("alpha\nbeta\ngamma")
        tool = create_edit_tool(str(tmp_path))
        await tool.execute(
            "c1",
            {
                "path": "f.txt",
                "edits": [
                    {"oldText": "alpha", "newText": "A"},
                    {"oldText": "gamma", "newText": "G"},
                ],
            },
        )
        assert f.read_text() == "A\nbeta\nG"

    async def test_legacy_flat_form_promoted(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hello world")
        tool = create_edit_tool(str(tmp_path))
        # Legacy single-edit form (no edits[] array).
        await tool.execute(
            "c1",
            {"path": "f.txt", "oldText": "hello", "newText": "hi"},
        )
        assert f.read_text() == "hi world"

    async def test_missing_file(self, tmp_path: Path) -> None:
        tool = create_edit_tool(str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            await tool.execute(
                "c1",
                {
                    "path": "missing.txt",
                    "edits": [{"oldText": "x", "newText": "y"}],
                },
            )

    async def test_preserves_crlf(self, tmp_path: Path) -> None:
        f = tmp_path / "win.txt"
        f.write_bytes(b"line1\r\nline2\r\nline3")
        tool = create_edit_tool(str(tmp_path))
        await tool.execute(
            "c1",
            {
                "path": "win.txt",
                "edits": [{"oldText": "line2", "newText": "LINE2"}],
            },
        )
        # Result should still use CRLF endings.
        assert f.read_bytes() == b"line1\r\nLINE2\r\nline3"

    async def test_diff_in_details(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hello world")
        tool = create_edit_tool(str(tmp_path))
        result = await tool.execute(
            "c1",
            {
                "path": "f.txt",
                "edits": [{"oldText": "hello", "newText": "hi"}],
            },
        )
        assert result.details is not None
        assert "+" in result.details.diff
        assert result.details.first_changed_line == 1
