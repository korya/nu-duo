"""Tests for the ``ls`` AgentTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from nu_ai.types import TextContent
from nu_coding_agent.core.tools.ls import create_ls_tool

if TYPE_CHECKING:
    from pathlib import Path


class TestLs:
    async def test_basic_listing(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.txt").write_text("a")
        (tmp_path / "beta.txt").write_text("b")
        (tmp_path / "subdir").mkdir()
        tool = create_ls_tool(str(tmp_path))
        result = await tool.execute("c1", {})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "alpha.txt" in text
        assert "beta.txt" in text
        assert "subdir/" in text

    async def test_alphabetical_case_insensitive(self, tmp_path: Path) -> None:
        for name in ("Banana", "apple", "Cherry"):
            (tmp_path / name).write_text("")
        tool = create_ls_tool(str(tmp_path))
        result = await tool.execute("c1", {})
        assert isinstance(result.content[0], TextContent)
        lines = result.content[0].text.split("\n")
        # apple, Banana, Cherry — case-insensitive sort.
        assert lines == ["apple", "Banana", "Cherry"]

    async def test_includes_dotfiles(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.txt").write_text("")
        tool = create_ls_tool(str(tmp_path))
        result = await tool.execute("c1", {})
        assert isinstance(result.content[0], TextContent)
        assert ".hidden" in result.content[0].text

    async def test_directories_get_slash(self, tmp_path: Path) -> None:
        (tmp_path / "dir1").mkdir()
        (tmp_path / "file1").write_text("")
        tool = create_ls_tool(str(tmp_path))
        result = await tool.execute("c1", {})
        assert isinstance(result.content[0], TextContent)
        assert "dir1/" in result.content[0].text
        assert "file1\n" in result.content[0].text or result.content[0].text.endswith("file1")

    async def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        tool = create_ls_tool(str(tmp_path))
        result = await tool.execute("c1", {"path": "empty"})
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "(empty directory)"

    async def test_relative_path(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "x.txt").write_text("")
        tool = create_ls_tool(str(tmp_path))
        result = await tool.execute("c1", {"path": "sub"})
        assert isinstance(result.content[0], TextContent)
        assert "x.txt" in result.content[0].text

    async def test_missing_path_raises(self, tmp_path: Path) -> None:
        tool = create_ls_tool(str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            await tool.execute("c1", {"path": "nope"})

    async def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("")
        tool = create_ls_tool(str(tmp_path))
        with pytest.raises(ValueError, match="Not a directory"):
            await tool.execute("c1", {"path": "f.txt"})

    async def test_entry_limit(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("")
        tool = create_ls_tool(str(tmp_path))
        result = await tool.execute("c1", {"limit": 3})
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "3 entries limit reached" in text
        assert result.details is not None
        assert result.details.entry_limit_reached == 3
