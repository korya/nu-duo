"""Tests for the ``find`` AgentTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from nu_ai.types import TextContent
from nu_coding_agent.core.tools.find import create_find_tool

if TYPE_CHECKING:
    from pathlib import Path


class TestFind:
    async def test_basic_glob(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        tool = create_find_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "*.py"})
        assert isinstance(result.content[0], TextContent)
        lines = result.content[0].text.split("\n")
        assert "a.py" in lines
        assert "b.py" in lines
        assert "c.txt" not in lines

    async def test_recursive_globstar(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("")
        (tmp_path / "src" / "deep").mkdir()
        (tmp_path / "src" / "deep" / "b.py").write_text("")
        tool = create_find_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "**/*.py"})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "src/a.py" in text
        assert "src/deep/b.py" in text

    async def test_no_matches(self, tmp_path: Path) -> None:
        tool = create_find_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "*.nonexistent"})
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "No files found matching pattern"

    async def test_search_path_relative(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "found.md").write_text("")
        tool = create_find_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "*.md", "path": "sub"})
        assert isinstance(result.content[0], TextContent)
        assert "found.md" in result.content[0].text

    async def test_node_modules_excluded(self, tmp_path: Path) -> None:
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "x.py").write_text("")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "y.py").write_text("")
        tool = create_find_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "**/*.py"})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "src/y.py" in text
        assert "node_modules" not in text

    async def test_gitignore_respected(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
        (tmp_path / "keep.py").write_text("")
        (tmp_path / "skip.log").write_text("")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "out.py").write_text("")
        tool = create_find_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "**/*"})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "keep.py" in text
        assert "skip.log" not in text
        assert "build/out.py" not in text

    async def test_missing_path_raises(self, tmp_path: Path) -> None:
        tool = create_find_tool(str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            await tool.execute("c1", {"pattern": "*", "path": "nope"})

    async def test_result_limit(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"f{i}.py").write_text("")
        tool = create_find_tool(str(tmp_path))
        result = await tool.execute("c1", {"pattern": "*.py", "limit": 3})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "3 results limit reached" in text
        assert result.details is not None
        assert result.details.result_limit_reached == 3
