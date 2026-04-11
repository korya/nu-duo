"""Tests for the ``read`` AgentTool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from nu_ai.types import TextContent
from nu_coding_agent.core.tools.read import create_read_tool

if TYPE_CHECKING:
    from pathlib import Path


def _params(path: str, **rest: Any) -> dict[str, Any]:
    return {"path": path, **rest}


class TestReadBasic:
    async def test_full_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3")
        tool = create_read_tool(str(tmp_path))
        result = await tool.execute("c1", _params("hello.txt"))
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "line1\nline2\nline3"

    async def test_relative_path(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x.txt").write_text("hi")
        tool = create_read_tool(str(tmp_path))
        result = await tool.execute("c1", _params("sub/x.txt"))
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "hi"

    async def test_absolute_path(self, tmp_path: Path) -> None:
        f = tmp_path / "abs.txt"
        f.write_text("absolute")
        tool = create_read_tool("/some/other/cwd")
        result = await tool.execute("c1", _params(str(f)))
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "absolute"


class TestReadOffsetLimit:
    async def test_offset(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("\n".join(["one", "two", "three", "four"]))
        tool = create_read_tool(str(tmp_path))
        result = await tool.execute("c1", _params("f.txt", offset=2))
        assert isinstance(result.content[0], TextContent)
        # Reads from line 2 onwards.
        assert "two" in result.content[0].text
        assert "one" not in result.content[0].text

    async def test_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("\n".join(str(i) for i in range(10)))
        tool = create_read_tool(str(tmp_path))
        result = await tool.execute("c1", _params("f.txt", limit=3))
        assert isinstance(result.content[0], TextContent)
        # Showed lines 1-3 plus a continuation hint.
        assert "0\n1\n2" in result.content[0].text
        assert "more lines" in result.content[0].text

    async def test_offset_and_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("\n".join(str(i) for i in range(10)))
        tool = create_read_tool(str(tmp_path))
        result = await tool.execute("c1", _params("f.txt", offset=3, limit=2))
        assert isinstance(result.content[0], TextContent)
        # Lines 3 and 4 (0-indexed: indices 2 and 3 → "2", "3").
        assert "2\n3" in result.content[0].text

    async def test_offset_beyond_eof_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("only one line")
        tool = create_read_tool(str(tmp_path))
        with pytest.raises(ValueError, match="beyond end of file"):
            await tool.execute("c1", _params("f.txt", offset=99))


class TestReadErrors:
    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        tool = create_read_tool(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await tool.execute("c1", _params("missing.txt"))

    async def test_directory_raises(self, tmp_path: Path) -> None:
        tool = create_read_tool(str(tmp_path))
        with pytest.raises(IsADirectoryError):
            await tool.execute("c1", _params("."))


class TestReadTruncation:
    async def test_long_file_truncated_with_continuation(self, tmp_path: Path) -> None:
        # Generate enough content to exceed the default max bytes.
        content = "\n".join(["x" * 200 for _ in range(500)])
        f = tmp_path / "big.txt"
        f.write_text(content)
        tool = create_read_tool(str(tmp_path))
        result = await tool.execute("c1", _params("big.txt"))
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "Use offset=" in text
