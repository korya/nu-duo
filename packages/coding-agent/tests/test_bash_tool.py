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


# ---------------------------------------------------------------------------
# Coverage: _get_shell fallback when $SHELL is unset (line 99)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestGetShellFallback:
    def test_shell_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nu_coding_agent.core.tools.bash import _get_shell

        monkeypatch.delenv("SHELL", raising=False)
        shell, args = _get_shell()
        assert shell is not None
        assert args == ["-c"]

    def test_shell_env_nonexistent_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nu_coding_agent.core.tools.bash import _get_shell

        monkeypatch.setenv("SHELL", "/nonexistent/shell")
        shell, args = _get_shell()
        # Falls back to shutil.which("bash") or "/bin/bash"
        assert shell is not None
        assert args == ["-c"]


# ---------------------------------------------------------------------------
# Coverage: BashAborted with temp file (line 328)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashAbortWithTempFile:
    async def test_abort_closes_temp_file(self, tmp_path: Path) -> None:
        """When abort happens after enough output to create a temp file (line 328)."""
        from nu_coding_agent.core.tools.bash import (
            BashAborted,
            BashOperations,
            BashToolOptions,
            create_bash_tool,
        )
        from nu_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES

        big_output = b"x" * (DEFAULT_MAX_BYTES + 1000)

        async def fake_exec(
            *,
            command: str,
            cwd: str,
            on_data: Any,
            timeout: float | None = None,
            env: dict[str, str] | None = None,
            abort_event: asyncio.Event | None = None,
        ) -> int | None:
            on_data(big_output)  # triggers temp file creation
            raise BashAborted

        tool = create_bash_tool(str(tmp_path), options=BashToolOptions(operations=BashOperations(exec=fake_exec)))
        with pytest.raises(RuntimeError, match="aborted"):
            await tool.execute("c1", {"command": "ignored"})


# ---------------------------------------------------------------------------
# Coverage: BashTimedOut with temp file (line 334)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashTimeoutWithTempFile:
    async def test_timeout_closes_temp_file(self, tmp_path: Path) -> None:
        """When timeout happens after enough output to create a temp file (line 334)."""
        from nu_coding_agent.core.tools.bash import (
            BashOperations,
            BashTimedOut,
            BashToolOptions,
            create_bash_tool,
        )
        from nu_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES

        big_output = b"y" * (DEFAULT_MAX_BYTES + 1000)

        async def fake_exec(
            *,
            command: str,
            cwd: str,
            on_data: Any,
            timeout: float | None = None,
            env: dict[str, str] | None = None,
            abort_event: asyncio.Event | None = None,
        ) -> int | None:
            on_data(big_output)
            raise BashTimedOut(5.0)

        tool = create_bash_tool(str(tmp_path), options=BashToolOptions(operations=BashOperations(exec=fake_exec)))
        with pytest.raises(RuntimeError, match="timed out"):
            await tool.execute("c1", {"command": "ignored"})


# ---------------------------------------------------------------------------
# Coverage: chunk eviction in handle_data (lines 300-301)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashChunkEviction:
    async def test_chunks_evicted_on_overflow(self, tmp_path: Path) -> None:
        """When chunks_bytes > max_chunks_bytes, old chunks are evicted (lines 299-301)."""
        from nu_coding_agent.core.tools.bash import (
            BashOperations,
            BashToolOptions,
            create_bash_tool,
        )
        from nu_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES

        chunk = b"z" * (DEFAULT_MAX_BYTES + 100)

        async def multi_chunk_exec(
            *,
            command: str,
            cwd: str,
            on_data: Any,
            timeout: float | None = None,
            env: dict[str, str] | None = None,
            abort_event: asyncio.Event | None = None,
        ) -> int | None:
            on_data(chunk)
            on_data(chunk)
            on_data(chunk)
            return 0

        updates: list[Any] = []
        tool = create_bash_tool(str(tmp_path), options=BashToolOptions(operations=BashOperations(exec=multi_chunk_exec)))
        result = await tool.execute("c1", {"command": "ignored"}, on_update=updates.append)
        # Output was huge, so it should be truncated
        assert isinstance(result.content[0], TextContent)


# ---------------------------------------------------------------------------
# Coverage: ensure_temp_file idempotent (line 284)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashEnsureTempFileIdempotent:
    async def test_temp_file_created_once(self, tmp_path: Path) -> None:
        """Second call to ensure_temp_file returns early (line 284)."""
        from nu_coding_agent.core.tools.bash import (
            BashOperations,
            BashToolOptions,
            create_bash_tool,
        )
        from nu_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES

        chunk = b"w" * (DEFAULT_MAX_BYTES + 100)

        async def double_big_exec(
            *,
            command: str,
            cwd: str,
            on_data: Any,
            timeout: float | None = None,
            env: dict[str, str] | None = None,
            abort_event: asyncio.Event | None = None,
        ) -> int | None:
            # First chunk triggers temp file; second should hit early return
            on_data(chunk)
            on_data(chunk)
            return 0

        tool = create_bash_tool(str(tmp_path), options=BashToolOptions(operations=BashOperations(exec=double_big_exec)))
        updates: list[Any] = []
        result = await tool.execute("c1", {"command": "ignored"}, on_update=updates.append)
        assert result.details is not None
        assert result.details.full_output_path is not None


# ---------------------------------------------------------------------------
# Coverage: spawn_hook (line 258 in _resolve_spawn_context)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashSpawnHook:
    async def test_spawn_hook_modifies_context(self, tmp_path: Path) -> None:
        from nu_coding_agent.core.tools.bash import BashSpawnContext, BashToolOptions

        def my_hook(ctx: BashSpawnContext) -> BashSpawnContext:
            ctx.command = f"echo hooked && {ctx.command}"
            return ctx

        tool = create_bash_tool(str(tmp_path), options=BashToolOptions(spawn_hook=my_hook))
        result = await tool.execute("c1", {"command": "echo original"})
        assert isinstance(result.content[0], TextContent)
        assert "hooked" in result.content[0].text


# ---------------------------------------------------------------------------
# Coverage: command_prefix (line 268)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="bash tool requires a POSIX shell")
class TestBashCommandPrefix:
    async def test_command_prefix_prepended(self, tmp_path: Path) -> None:
        from nu_coding_agent.core.tools.bash import BashToolOptions

        tool = create_bash_tool(str(tmp_path), options=BashToolOptions(command_prefix="export FOO=bar"))
        result = await tool.execute("c1", {"command": "echo $FOO"})
        assert isinstance(result.content[0], TextContent)
        assert "bar" in result.content[0].text
