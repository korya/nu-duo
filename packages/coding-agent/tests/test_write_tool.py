"""Tests for the ``write`` AgentTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_ai.types import TextContent
from pi_coding_agent.core.tools.write import create_write_tool

if TYPE_CHECKING:
    from pathlib import Path


class TestWrite:
    async def test_creates_new_file(self, tmp_path: Path) -> None:
        tool = create_write_tool(str(tmp_path))
        result = await tool.execute("c1", {"path": "new.txt", "content": "hello"})
        assert (tmp_path / "new.txt").read_text() == "hello"
        assert isinstance(result.content[0], TextContent)
        assert "Successfully wrote" in result.content[0].text

    async def test_overwrites_existing(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("old")
        tool = create_write_tool(str(tmp_path))
        await tool.execute("c1", {"path": "x.txt", "content": "new content"})
        assert f.read_text() == "new content"

    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        tool = create_write_tool(str(tmp_path))
        await tool.execute(
            "c1",
            {"path": "nested/sub/file.txt", "content": "ok"},
        )
        assert (tmp_path / "nested" / "sub" / "file.txt").read_text() == "ok"

    async def test_absolute_path(self, tmp_path: Path) -> None:
        target = tmp_path / "abs.txt"
        tool = create_write_tool("/some/other/cwd")
        await tool.execute("c1", {"path": str(target), "content": "abs"})
        assert target.read_text() == "abs"

    async def test_byte_count_in_message(self, tmp_path: Path) -> None:
        tool = create_write_tool(str(tmp_path))
        result = await tool.execute("c1", {"path": "f.txt", "content": "héllo"})
        assert isinstance(result.content[0], TextContent)
        # 'héllo' encodes to 6 bytes in UTF-8.
        assert "6 bytes" in result.content[0].text
