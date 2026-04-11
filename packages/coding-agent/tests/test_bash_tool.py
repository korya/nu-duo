"""Tests for the ``bash`` AgentTool."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import pytest
from nu_ai.types import TextContent
from nu_coding_agent.core.tools.bash import create_bash_tool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashBasic:
    async def test_simple_command(self, tmp_path: Path) -> None:
        tool = create_bash_tool(str(tmp_path))
        result = await tool.execute("c1", {"command": "echo hello"})
        assert isinstance(result.content[0], TextContent)
        assert "hello" in result.content[0].text

    async def test_cwd_is_used(self, tmp_path: Path) -> None:
        (tmp_path / "marker.txt").write_text("here")
        tool = create_bash_tool(str(tmp_path))
        result = await tool.execute("c1", {"command": "ls"})
        assert isinstance(result.content[0], TextContent)
        assert "marker.txt" in result.content[0].text

    async def test_nonzero_exit_raises(self, tmp_path: Path) -> None:
        tool = create_bash_tool(str(tmp_path))
        with pytest.raises(RuntimeError, match="exited with code"):
            await tool.execute("c1", {"command": "exit 7"})

    async def test_stderr_captured(self, tmp_path: Path) -> None:
        tool = create_bash_tool(str(tmp_path))
        result = await tool.execute("c1", {"command": "echo err 1>&2"})
        assert isinstance(result.content[0], TextContent)
        assert "err" in result.content[0].text

    async def test_command_substitution(self, tmp_path: Path) -> None:
        tool = create_bash_tool(str(tmp_path))
        result = await tool.execute("c1", {"command": "echo $((2+3))"})
        assert isinstance(result.content[0], TextContent)
        assert "5" in result.content[0].text


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashTimeout:
    async def test_timeout_raises(self, tmp_path: Path) -> None:
        tool = create_bash_tool(str(tmp_path))
        with pytest.raises(RuntimeError, match="timed out"):
            await tool.execute("c1", {"command": "sleep 5", "timeout": 0.5})


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashAbort:
    async def test_abort_kills_process(self, tmp_path: Path) -> None:
        tool = create_bash_tool(str(tmp_path))
        signal = asyncio.Event()

        async def trigger_abort() -> None:
            await asyncio.sleep(0.05)
            signal.set()

        abort_task = asyncio.create_task(trigger_abort())
        try:
            with pytest.raises(RuntimeError, match="aborted"):
                await tool.execute("c1", {"command": "sleep 5"}, signal=signal)
        finally:
            await abort_task


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashCwdMissing:
    async def test_missing_cwd_raises(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nope"
        tool = create_bash_tool(str(nonexistent))
        with pytest.raises(ValueError, match="does not exist"):
            await tool.execute("c1", {"command": "echo hi"})


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashOnUpdate:
    async def test_on_update_called_with_streaming_output(self, tmp_path: Path) -> None:
        tool = create_bash_tool(str(tmp_path))
        updates: list[Any] = []
        await tool.execute(
            "c1",
            {"command": "echo line1; echo line2"},
            on_update=updates.append,
        )
        # First call is the empty seed; subsequent calls carry partial output.
        assert len(updates) > 1
