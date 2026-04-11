"""SSH command wrappers.

Port of ``packages/pods/src/ssh.ts``. Like the TS original we shell out
to the user's local ``ssh``/``scp`` binaries (so existing host configs
and keys "just work") rather than reimplementing SSH in-process — this
keeps auth semantics identical and avoids dragging in ``asyncssh``.

The module exposes a private ``_runner`` indirection so tests can stub
out subprocess execution without spinning up a real SSH server.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class SshResult:
    """Result of a non-streaming SSH command."""

    stdout: str
    stderr: str
    exit_code: int


# Type aliases for the injectable subprocess runners. Tests swap these
# out wholesale via :func:`set_runners`.
SshRunner = Callable[[Sequence[str]], Awaitable[SshResult]]
StreamRunner = Callable[[Sequence[str], bool], Awaitable[int]]
ScpRunner = Callable[[Sequence[str]], Awaitable[bool]]


_KEEPALIVE_OPTS: tuple[str, ...] = (
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=120",
)


async def _default_run(argv: Sequence[str]) -> SshResult:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return SshResult(
        stdout=stdout_b.decode("utf-8", errors="replace"),
        stderr=stderr_b.decode("utf-8", errors="replace"),
        exit_code=proc.returncode if proc.returncode is not None else -1,
    )


async def _default_stream(argv: Sequence[str], silent: bool) -> int:
    if silent:
        stdout = asyncio.subprocess.DEVNULL
        stderr = asyncio.subprocess.DEVNULL
    else:
        stdout = sys.stdout
        stderr = sys.stderr
    proc = await asyncio.create_subprocess_exec(*argv, stdout=stdout, stderr=stderr)
    await proc.wait()
    return proc.returncode if proc.returncode is not None else -1


async def _default_scp(argv: Sequence[str]) -> bool:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    return proc.returncode == 0


@dataclass(slots=True)
class _Runners:
    run: SshRunner
    stream: StreamRunner
    scp: ScpRunner


_state = _Runners(run=_default_run, stream=_default_stream, scp=_default_scp)


def set_runners(
    *,
    run: SshRunner | None = None,
    stream: StreamRunner | None = None,
    scp: ScpRunner | None = None,
) -> tuple[SshRunner, StreamRunner, ScpRunner]:
    """Override the subprocess runners (for tests). Returns previous tuple."""
    prev = (_state.run, _state.stream, _state.scp)
    if run is not None:
        _state.run = run
    if stream is not None:
        _state.stream = stream
    if scp is not None:
        _state.scp = scp
    return prev


def reset_runners() -> None:
    """Restore the default subprocess runners."""
    _state.run = _default_run
    _state.stream = _default_stream
    _state.scp = _default_scp


def split_ssh(ssh_cmd: str) -> tuple[str, list[str]]:
    """Split a TS-style ``"ssh root@host"`` string into ``(cmd, args)``."""
    parts = shlex.split(ssh_cmd)
    if not parts:
        raise ValueError("ssh command is empty")
    return parts[0], parts[1:]


def scp_host_args(ssh_cmd: str) -> tuple[str, list[str]]:
    """Extract ``(host_token, scp_extra_args)`` from a TS-style SSH cmd.

    The TS version parses ``-p PORT`` as the SSH port and rewrites it to
    ``-P PORT`` for scp; everything else after the host token is dropped
    (the TS implementation only honours the port flag). We follow the
    same convention so the on-disk pod config remains compatible.
    """
    _, args = split_ssh(ssh_cmd)
    extras: list[str] = []
    host: str | None = None
    i = 0
    while i < len(args):
        token = args[i]
        if token == "-p" and i + 1 < len(args):
            extras.extend(["-P", args[i + 1]])
            i += 2
            continue
        if "@" in token or host is None:
            host = token
            i += 1
            continue
        i += 1
    if host is None:
        raise ValueError(f"could not extract host from ssh command: {ssh_cmd!r}")
    return host, extras


async def ssh_exec(ssh_cmd: str, command: str, *, keep_alive: bool = False) -> SshResult:
    """Run a single command over SSH and capture stdout/stderr."""
    cmd, args = split_ssh(ssh_cmd)
    argv: list[str] = [cmd]
    if keep_alive:
        argv.extend(_KEEPALIVE_OPTS)
    argv.extend(args)
    argv.append(command)
    return await _state.run(argv)


async def ssh_exec_stream(
    ssh_cmd: str,
    command: str,
    *,
    silent: bool = False,
    force_tty: bool = False,
    keep_alive: bool = False,
) -> int:
    """Run a command over SSH and stream stdout/stderr to the terminal."""
    cmd, args = split_ssh(ssh_cmd)
    argv: list[str] = [cmd]
    if keep_alive:
        argv.extend(_KEEPALIVE_OPTS)
    if force_tty:
        argv.append("-tt")
    argv.extend(args)
    argv.append(command)
    return await _state.stream(argv, silent)


async def scp_file(ssh_cmd: str, local_path: str, remote_path: str) -> bool:
    """Copy a local file to ``host:remote_path`` via ``scp``."""
    try:
        host, extras = scp_host_args(ssh_cmd)
    except ValueError:
        return False
    argv: list[str] = ["scp", *extras, local_path, f"{host}:{remote_path}"]
    return await _state.scp(argv)
