"""Sandbox configuration and executor types for nu-mom.

Port of ``packages/mom/src/sandbox.ts``. Supports two execution modes:

- ``host``   — commands run directly via ``asyncio.create_subprocess_exec``
- ``docker`` — commands are wrapped in ``docker exec <container> sh -c ...``

The :class:`Executor` protocol matches the upstream TS interface so tools
can be written once and run in either environment.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "DockerExecutor",
    "DockerSandboxConfig",
    "ExecOptions",
    "ExecResult",
    "Executor",
    "HostExecutor",
    "HostSandboxConfig",
    "SandboxConfig",
    "create_executor",
    "parse_sandbox_arg",
    "validate_sandbox",
]

# ---------------------------------------------------------------------------
# Config types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HostSandboxConfig:
    """Sandbox running directly on the host machine."""

    type: str = "host"


@dataclass(slots=True)
class DockerSandboxConfig:
    """Sandbox running inside a Docker container."""

    container: str
    type: str = "docker"


type SandboxConfig = HostSandboxConfig | DockerSandboxConfig


# ---------------------------------------------------------------------------
# Arg parsing / validation
# ---------------------------------------------------------------------------


def parse_sandbox_arg(value: str) -> SandboxConfig:
    """Parse a ``--sandbox`` argument string into a :class:`SandboxConfig`."""
    if value == "host":
        return HostSandboxConfig()
    if value.startswith("docker:"):
        container = value[len("docker:") :]
        if not container:
            print("Error: docker sandbox requires container name (e.g., docker:mom-sandbox)", file=sys.stderr)
            sys.exit(1)
        return DockerSandboxConfig(container=container)
    print(
        f"Error: Invalid sandbox type '{value}'. Use 'host' or 'docker:<container-name>'",
        file=sys.stderr,
    )
    sys.exit(1)


async def validate_sandbox(config: SandboxConfig) -> None:
    """Check that the sandbox is available, exiting with an error if not."""
    if config.type == "host":
        return

    assert isinstance(config, DockerSandboxConfig)

    # Check Docker availability
    try:
        await _exec_simple("docker", ["--version"])
    except Exception:
        print("Error: Docker is not installed or not in PATH", file=sys.stderr)
        sys.exit(1)

    # Check container is running
    try:
        result = await _exec_simple(
            "docker",
            ["inspect", "-f", "{{.State.Running}}", config.container],
        )
        if result.strip() != "true":
            print(f"Error: Container '{config.container}' is not running.", file=sys.stderr)
            print(f"Start it with: docker start {config.container}", file=sys.stderr)
            sys.exit(1)
    except Exception:
        print(f"Error: Container '{config.container}' does not exist.", file=sys.stderr)
        print("Create it with: ./docker.sh create <data-dir>", file=sys.stderr)
        sys.exit(1)

    print(f"  Docker container '{config.container}' is running.")


# ---------------------------------------------------------------------------
# Executor types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecOptions:
    """Options forwarded to the executor."""

    timeout: float | None = None
    signal: asyncio.Event | None = None


@dataclass(slots=True)
class ExecResult:
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    code: int


MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10 MB cap per stream


class Executor(Protocol):
    """Protocol shared by :class:`HostExecutor` and :class:`DockerExecutor`."""

    async def exec(self, command: str, options: ExecOptions | None = None) -> ExecResult:
        """Execute *command* and return the combined result."""
        ...

    def get_workspace_path(self, host_path: str) -> str:
        """Translate a host path to the path as seen by this executor."""
        ...


# ---------------------------------------------------------------------------
# HostExecutor
# ---------------------------------------------------------------------------


class HostExecutor:
    """Runs commands directly on the host via ``asyncio.create_subprocess_exec``."""

    async def exec(self, command: str, options: ExecOptions | None = None) -> ExecResult:
        opts = options or ExecOptions()
        timeout = opts.timeout
        signal = opts.signal

        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stdout_bytes = 0
        stderr_bytes = 0

        async def read_stdout() -> None:
            nonlocal stdout_bytes
            assert proc.stdout
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                stdout_bytes += len(chunk)
                if stdout_bytes <= MAX_OUTPUT_BYTES:
                    stdout_chunks.append(chunk)

        async def read_stderr() -> None:
            nonlocal stderr_bytes
            assert proc.stderr
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_bytes += len(chunk)
                if stderr_bytes <= MAX_OUTPUT_BYTES:
                    stderr_chunks.append(chunk)

        async def run() -> int | None:
            await asyncio.gather(read_stdout(), read_stderr())
            return await proc.wait()

        try:
            if timeout and timeout > 0:
                code = await asyncio.wait_for(run(), timeout=timeout)
            else:
                code = await run()
        except TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
            stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            raise TimeoutError(f"{stdout}\n{stderr}\nCommand timed out after {timeout} seconds".strip())

        if signal and signal.is_set():
            raise RuntimeError(
                f"{b''.join(stdout_chunks).decode('utf-8', errors='replace')}\n"
                f"{b''.join(stderr_chunks).decode('utf-8', errors='replace')}\n"
                "Command aborted"
            )

        return ExecResult(
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            code=code or 0,
        )

    def get_workspace_path(self, host_path: str) -> str:
        """Return *host_path* unchanged — host executor sees the real FS."""
        return host_path


# ---------------------------------------------------------------------------
# DockerExecutor
# ---------------------------------------------------------------------------


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


class DockerExecutor:
    """Runs commands inside a Docker container via ``docker exec``."""

    def __init__(self, container: str) -> None:
        self._container = container
        self._host = HostExecutor()

    async def exec(self, command: str, options: ExecOptions | None = None) -> ExecResult:
        docker_cmd = f"docker exec {self._container} sh -c {_shell_escape(command)}"
        return await self._host.exec(docker_cmd, options)

    def get_workspace_path(self, _host_path: str) -> str:
        """Docker containers use ``/workspace`` as their working root."""
        return "/workspace"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_executor(config: SandboxConfig) -> HostExecutor | DockerExecutor:
    """Create the appropriate executor for *config*."""
    if config.type == "host":
        return HostExecutor()
    assert isinstance(config, DockerSandboxConfig)
    return DockerExecutor(config.container)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _exec_simple(cmd: str, args: list[str]) -> str:
    proc = await asyncio.create_subprocess_exec(
        cmd,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace") or f"Exit code {proc.returncode}")
    return stdout.decode("utf-8", errors="replace")
