"""Subprocess execution helper — direct port of ``packages/coding-agent/src/core/exec.ts``.

Used by extensions and custom tools that need to shell out without
re-implementing timeout/cancel handling. The Python port uses
:func:`asyncio.create_subprocess_exec` directly; the upstream
``waitForChildProcess`` shim that worked around inherited stdio handles
on Node has no Python equivalent because :mod:`asyncio` already pipes
stdio explicitly.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal as _signal
from dataclasses import dataclass


@dataclass(slots=True)
class ExecOptions:
    """Knobs for :func:`exec_command`."""

    signal: asyncio.Event | None = None
    """Cancellation flag — when set, the process is killed."""
    timeout: float | None = None
    """Wall-clock timeout in seconds (TS port: milliseconds; converted at the call site)."""
    cwd: str | None = None
    """Override working directory for the spawned child."""


@dataclass(slots=True)
class ExecResult:
    """Captured output and exit status from :func:`exec_command`."""

    stdout: str
    stderr: str
    code: int
    killed: bool


_FORCE_KILL_DELAY_SECONDS = 5.0


async def exec_command(
    command: str,
    args: list[str],
    cwd: str,
    options: ExecOptions | None = None,
) -> ExecResult:
    """Run ``command`` with ``args`` in ``cwd`` and capture stdout/stderr.

    Errors are encoded in the result rather than raised: a ``code`` of
    1 with empty output means the process failed to spawn. ``killed`` is
    set when the timeout fired or the abort signal was raised.
    """
    opts = options or ExecOptions()
    effective_cwd = opts.cwd or cwd

    try:
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=effective_cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError):
        return ExecResult(stdout="", stderr="", code=1, killed=False)

    killed = False
    force_kill_task: asyncio.Task[None] | None = None

    async def _force_kill_after_delay() -> None:
        await asyncio.sleep(_FORCE_KILL_DELAY_SECONDS)
        if proc.returncode is None:  # pragma: no cover — escalation only fires when SIGTERM was ignored
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    def _kill() -> None:
        nonlocal killed, force_kill_task
        if killed:
            return
        killed = True
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(_signal.SIGTERM)
        # Mirror the upstream 5s SIGTERM → SIGKILL escalation.
        force_kill_task = asyncio.create_task(_force_kill_after_delay())

    async def _watch_signal(signal: asyncio.Event) -> None:
        await signal.wait()
        _kill()

    async def _watch_timeout(timeout: float) -> None:  # noqa: ASYNC109 — explicit timeout watcher
        await asyncio.sleep(timeout)
        _kill()

    watchers: list[asyncio.Task[None]] = []
    if opts.signal is not None:
        if opts.signal.is_set():
            _kill()
        else:
            watchers.append(asyncio.create_task(_watch_signal(opts.signal)))
    if opts.timeout is not None and opts.timeout > 0:
        watchers.append(asyncio.create_task(_watch_timeout(opts.timeout)))

    try:
        stdout_bytes, stderr_bytes = await proc.communicate()
    except Exception:  # pragma: no cover — communicate exceptions are platform-specific edge cases
        _kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return ExecResult(stdout="", stderr="", code=1, killed=killed)
    finally:
        for watcher in watchers:
            watcher.cancel()
        for watcher in watchers:
            with contextlib.suppress(BaseException):
                await watcher
        if force_kill_task is not None:
            force_kill_task.cancel()
            with contextlib.suppress(BaseException):
                await force_kill_task

    code = proc.returncode if proc.returncode is not None else 0
    return ExecResult(
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        code=code,
        killed=killed,
    )


__all__ = ["ExecOptions", "ExecResult", "exec_command"]
