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

    def _kill() -> None:
        nonlocal killed
        if killed or proc.returncode is not None:
            return
        killed = True
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(_signal.SIGTERM)

    async def _watch_signal() -> None:
        if opts.signal is None:
            return
        await opts.signal.wait()
        _kill()

    async def _watch_timeout() -> None:
        if opts.timeout is None or opts.timeout <= 0:
            return
        await asyncio.sleep(opts.timeout)
        _kill()

    watchers: list[asyncio.Task[None]] = []
    if opts.signal is not None:
        if opts.signal.is_set():
            _kill()
        else:
            watchers.append(asyncio.create_task(_watch_signal()))
    if opts.timeout is not None and opts.timeout > 0:
        watchers.append(asyncio.create_task(_watch_timeout()))

    try:
        stdout_bytes, stderr_bytes = await proc.communicate()
    except Exception:
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

    if killed and proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=_FORCE_KILL_DELAY_SECONDS)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()

    code = proc.returncode if proc.returncode is not None else 0
    return ExecResult(
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        code=code,
        killed=killed,
    )


__all__ = ["ExecOptions", "ExecResult", "exec_command"]
