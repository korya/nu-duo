"""``bash`` tool — async subprocess execution with timeout, abort, tail truncation.

Direct port of ``packages/coding-agent/src/core/tools/bash.ts``. Streams
combined stdout+stderr through ``on_update`` callbacks while accumulating
a rolling tail buffer for the final result. When output exceeds the
in-memory threshold the rest is spilled to a temp file referenced by
:class:`BashToolDetails.full_output_path` so the model can ``read`` it
later.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import shutil
import signal as signal_module
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai.types import TextContent

from nu_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_tail,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BashToolDetails:
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


@dataclass(slots=True)
class BashSpawnContext:
    command: str
    cwd: str
    env: dict[str, str]


type BashSpawnHook = Callable[[BashSpawnContext], BashSpawnContext]


@dataclass(slots=True)
class BashOperations:
    """Pluggable subprocess backend.

    The ``exec`` callable streams output via ``on_data`` and resolves with
    the process's exit code (``None`` if killed). Override to delegate to
    SSH or another remote runner.
    """

    exec: Callable[..., Awaitable[int | None]] | None = None


@dataclass(slots=True)
class BashToolOptions:
    operations: BashOperations | None = None
    command_prefix: str | None = None
    spawn_hook: BashSpawnHook | None = None


# ---------------------------------------------------------------------------
# Default subprocess backend
# ---------------------------------------------------------------------------


def _temp_file_path() -> str:
    suffix = secrets.token_hex(8)
    return str(Path(tempfile.gettempdir()) / f"pi-bash-{suffix}.log")


def _get_shell() -> tuple[str, list[str]]:
    """Return the shell binary + flag args used to run a single command.

    Mirrors upstream's ``getShellConfig``: prefer ``$SHELL`` if set,
    otherwise ``/bin/bash``. Always invoke with ``-c`` so the command
    string is parsed by the shell rather than executed directly.
    """
    user_shell = os.environ.get("SHELL")
    if user_shell and Path(user_shell).exists():
        return user_shell, ["-c"]
    return shutil.which("bash") or "/bin/bash", ["-c"]


def _get_shell_env() -> dict[str, str]:
    """Return a fresh environment dict suitable for spawning shells."""
    return dict(os.environ)


async def _default_exec(
    *,
    command: str,
    cwd: str,
    on_data: Callable[[bytes], None],
    timeout: float | None = None,  # noqa: ASYNC109 — enforced via watchdog task below
    env: dict[str, str] | None = None,
    abort_event: asyncio.Event | None = None,
) -> int | None:
    """Spawn ``command`` and stream stdout+stderr through ``on_data``.

    The ``timeout`` parameter is intentionally a plain ``float | None`` rather
    than a context-manager-driven approach: the function spawns a long-lived
    subprocess whose lifetime is owned by an internal watchdog task that
    SIGTERMs the process tree when the timeout fires. ``asyncio.timeout`` is
    not appropriate here because we still need to drain the captured output
    after the kill so partial results survive into the error message.
    """
    if not await asyncio.to_thread(Path(cwd).exists):
        raise ValueError(f"Working directory does not exist: {cwd}\nCannot execute bash commands.")
    shell, args = _get_shell()
    process = await asyncio.create_subprocess_exec(
        shell,
        *args,
        command,
        cwd=cwd,
        env=env or _get_shell_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Detach into a new process group so we can signal the whole tree.
        start_new_session=True,
    )

    timed_out = False
    abort_seen = False

    async def stream_pipe(reader: asyncio.StreamReader | None) -> None:
        if reader is None:
            return
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                return
            on_data(chunk)

    async def watch_abort() -> None:
        nonlocal abort_seen
        if abort_event is None:
            return
        await abort_event.wait()
        abort_seen = True
        _kill_tree(process.pid)

    async def watch_timeout() -> None:
        nonlocal timed_out
        if timeout is None or timeout <= 0:
            return
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        timed_out = True
        _kill_tree(process.pid)

    abort_task = asyncio.create_task(watch_abort())
    timeout_task = asyncio.create_task(watch_timeout())
    try:
        await asyncio.gather(stream_pipe(process.stdout), stream_pipe(process.stderr))
        exit_code = await process.wait()
    finally:
        abort_task.cancel()
        timeout_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await abort_task
        with contextlib.suppress(asyncio.CancelledError):
            await timeout_task

    if abort_seen:
        raise BashAborted
    if timed_out:
        raise BashTimedOut(timeout or 0)
    return exit_code


def _kill_tree(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(pid), signal_module.SIGTERM)


class BashAborted(Exception):
    pass


class BashTimedOut(Exception):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"timeout:{timeout_seconds}")
        self.timeout_seconds = timeout_seconds


def _default_operations() -> BashOperations:
    return BashOperations(exec=_default_exec)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


_BASH_PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {
            "type": "number",
            "description": "Timeout in seconds (optional, no default timeout)",
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}


def _build_description() -> str:
    return (
        f"Execute a bash command in the current working directory. Returns stdout and stderr. "
        f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
        f"(whichever is hit first). If truncated, full output is saved to a temp file. "
        f"Optionally provide a timeout in seconds."
    )


def create_bash_tool(
    cwd: str,
    *,
    options: BashToolOptions | None = None,
) -> AgentTool[dict[str, Any], BashToolDetails | None]:
    """Build the ``bash`` :class:`AgentTool` rooted at ``cwd``."""
    opts = options or BashToolOptions()
    ops = opts.operations or _default_operations()
    exec_fn = ops.exec or _default_exec
    command_prefix = opts.command_prefix
    spawn_hook = opts.spawn_hook

    def _resolve_spawn_context(command: str) -> BashSpawnContext:
        base = BashSpawnContext(command=command, cwd=cwd, env=_get_shell_env())
        return spawn_hook(base) if spawn_hook is not None else base

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[BashToolDetails | None]], None] | None = None,
    ) -> AgentToolResult[BashToolDetails | None]:
        command: str = params["command"]
        timeout: float | None = params.get("timeout")
        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command
        spawn_context = _resolve_spawn_context(resolved_command)

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        chunks: list[bytes] = []
        chunks_bytes = 0
        max_chunks_bytes = DEFAULT_MAX_BYTES * 2
        total_bytes = 0
        temp_file_path: str | None = None
        temp_file: Any = None

        def ensure_temp_file() -> None:
            nonlocal temp_file_path, temp_file
            if temp_file_path is not None:
                return
            temp_file_path = _temp_file_path()
            temp_file = Path(temp_file_path).open("wb")  # noqa: SIM115 — closed manually
            for chunk in chunks:
                temp_file.write(chunk)

        def handle_data(data: bytes) -> None:
            nonlocal chunks_bytes, total_bytes
            total_bytes += len(data)
            if total_bytes > DEFAULT_MAX_BYTES:
                ensure_temp_file()
            if temp_file is not None:
                temp_file.write(data)
            chunks.append(data)
            chunks_bytes += len(data)
            while chunks_bytes > max_chunks_bytes and len(chunks) > 1:
                removed = chunks.pop(0)
                chunks_bytes -= len(removed)
            if on_update is not None:
                full_text = b"".join(chunks).decode("utf-8", errors="replace")
                truncation = truncate_tail(full_text)
                if truncation.truncated:
                    ensure_temp_file()
                on_update(
                    AgentToolResult(
                        content=[TextContent(text=truncation.content or "")],
                        details=BashToolDetails(
                            truncation=truncation if truncation.truncated else None,
                            full_output_path=temp_file_path,
                        ),
                    )
                )

        try:
            exit_code = await exec_fn(
                command=spawn_context.command,
                cwd=spawn_context.cwd,
                on_data=handle_data,
                timeout=timeout,
                env=spawn_context.env,
                abort_event=signal,
            )
        except BashAborted as exc:
            if temp_file is not None:
                temp_file.close()
            output = b"".join(chunks).decode("utf-8", errors="replace")
            output += ("\n\n" if output else "") + "Command aborted"
            raise RuntimeError(output) from exc
        except BashTimedOut as exc:
            if temp_file is not None:
                temp_file.close()
            output = b"".join(chunks).decode("utf-8", errors="replace")
            output += ("\n\n" if output else "") + f"Command timed out after {exc.timeout_seconds} seconds"
            raise RuntimeError(output) from exc
        except Exception:
            if temp_file is not None:
                temp_file.close()
            raise

        if temp_file is not None:
            temp_file.close()

        full_output = b"".join(chunks).decode("utf-8", errors="replace")
        truncation = truncate_tail(full_output)
        if truncation.truncated and temp_file_path is None:
            ensure_temp_file()
            if temp_file is not None:
                temp_file.close()

        output_text = truncation.content or "(no output)"
        details: BashToolDetails | None = None
        if truncation.truncated:
            details = BashToolDetails(truncation=truncation, full_output_path=temp_file_path)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                last_line = full_output.split("\n")[-1] if full_output else ""
                last_line_size = format_size(len(last_line.encode("utf-8")))
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line "
                    f"{end_line} (line is {last_line_size}). Full output: {temp_file_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {temp_file_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp_file_path}]"
                )

        if exit_code is not None and exit_code != 0:
            output_text += f"\n\nCommand exited with code {exit_code}"
            raise RuntimeError(output_text)

        return AgentToolResult(
            content=[TextContent(text=output_text)],
            details=details,
        )

    return AgentTool[dict[str, Any], BashToolDetails | None](
        name="bash",
        description=_build_description(),
        parameters=_BASH_PARAMETERS,
        label="bash",
        execute=execute,
    )


__all__ = [
    "BashAborted",
    "BashOperations",
    "BashSpawnContext",
    "BashSpawnHook",
    "BashTimedOut",
    "BashToolDetails",
    "BashToolOptions",
    "create_bash_tool",
]
