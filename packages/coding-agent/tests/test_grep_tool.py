"""Tests for the ``grep`` AgentTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from nu_ai.types import TextContent
from nu_coding_agent.core.tools.grep import create_grep_tool

if TYPE_CHECKING:
    from pathlib import Path


class TestGrep:
    async def test_basic_match(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("alpha\nbeta\ngamma")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "beta"})
        assert isinstance(result.content[0], TextContent)
        assert "f.txt:2:beta" in result.content[0].text

    async def test_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("nothing here")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "missing"})
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "No matches found"

    async def test_regex_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("def foo():\n    pass\ndef bar():\n    pass")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": r"^def \w+"})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "def foo" in text
        assert "def bar" in text

    async def test_literal_match(self, tmp_path: Path) -> None:
        # Without literal=True the . in "a.b" would match any char.
        (tmp_path / "f.txt").write_text("acb\na.b\nadb")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "a.b", "literal": True})
        assert isinstance(result.content[0], TextContent)
        lines = [line for line in result.content[0].text.split("\n") if line]
        assert len(lines) == 1
        assert ":a.b" in lines[0]

    async def test_ignore_case(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("Hello\nWORLD\nhello")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "hello", "ignoreCase": True})
        assert isinstance(result.content[0], TextContent)
        lines = [line for line in result.content[0].text.split("\n") if line]
        # Hello (line 1) + hello (line 3).
        assert len(lines) == 2

    async def test_glob_filter(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("target")
        (tmp_path / "doc.txt").write_text("target")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "target", "glob": "*.py"})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "code.py" in text
        assert "doc.txt" not in text

    async def test_context_lines(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("one\ntwo\nTARGET\nfour\nfive")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "TARGET", "context": 1})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        # Match line uses ":", context lines use "-".
        assert "two" in text
        assert "TARGET" in text
        assert "four" in text

    async def test_recursive_walk(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "x.txt").write_text("hit")
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "y.txt").write_text("nothing")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "hit"})
        assert isinstance(result.content[0], TextContent)
        assert "a/x.txt" in result.content[0].text

    async def test_gitignore_respected(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("ignored.txt\n")
        (tmp_path / "keep.txt").write_text("target")
        (tmp_path / "ignored.txt").write_text("target")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "target"})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "keep.txt" in text
        assert "ignored.txt" not in text

    async def test_match_limit(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("\n".join(["target"] * 10))
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "target", "limit": 3})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "3 matches limit reached" in text
        assert result.details is not None
        assert result.details.match_limit_reached == 3

    async def test_invalid_regex_raises(self, tmp_path: Path) -> None:
        tool = create_grep_tool(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid regex"):
            await tool.execute("c1", {"pattern": "[unclosed"})

    async def test_missing_path_raises(self, tmp_path: Path) -> None:
        tool = create_grep_tool(str(tmp_path))
        with pytest.raises(ValueError, match="Path not found"):
            await tool.execute("c1", {"pattern": "x", "path": "nope"})

    async def test_single_file_path(self, tmp_path: Path) -> None:
        f = tmp_path / "lone.txt"
        f.write_text("hit me")
        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "hit", "path": "lone.txt"})
        assert isinstance(result.content[0], TextContent)
        assert "lone.txt" in result.content[0].text
