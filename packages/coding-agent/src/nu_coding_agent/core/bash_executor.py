"""Bash command execution wrapper — direct port of ``packages/coding-agent/src/core/bash-executor.ts``.

A higher-level layer over :mod:`nu_coding_agent.core.tools.bash`'s
:class:`BashOperations`. Used by :class:`AgentSession` (interactive +
RPC modes) and by direct callers that need a ``BashResult`` rather than
the streaming tool interface.

Behaviour parity with the upstream:

* Streams stdout+stderr through ``options.on_chunk`` (already sanitised
  and ANSI-stripped).
* Spills to a temp file once total output crosses
  :data:`nu_coding_agent.core.tools.truncate.DEFAULT_MAX_BYTES`.
* Truncates the in-memory buffer to ``2x DEFAULT_MAX_BYTES`` so we
  don't blow up on multi-GB log dumps.
* Returns a :class:`BashResult` with ``output``, ``exit_code``,
  ``cancelled``, ``truncated``, and an optional ``full_output_path``.
"""

from __future__ import annotations

import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nu_coding_agent.core.tools.bash import BashAborted, BashOperations, local_bash_exec
from nu_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES, truncate_tail
from nu_coding_agent.utils.shell import sanitize_binary_output

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable


# Strip ECMA-48 control sequences (CSI, OSC, single-char ESC dispatches).
# Mirrors the ``strip-ansi`` npm package the upstream depends on.
_ANSI_RE = re.compile(
    r"[\u001b\u009b](?:\[|\]|\(|\)|#|;|\?)*"
    r"(?:(?:(?:[a-zA-Z\d]*(?:;[a-zA-Z\d]*)*)?\u0007)"
    r"|(?:(?:\d{1,4}(?:;\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]))"
)


def strip_ansi(text: str) -> str:
    """Drop ANSI escape sequences from ``text``."""
    return _ANSI_RE.sub("", text)


@dataclass(slots=True)
class BashExecutorOptions:
    """Knobs for :func:`execute_bash` / :func:`execute_bash_with_operations`."""

    on_chunk: Callable[[str], None] | None = None
    """Streaming callback. Receives each sanitised text chunk."""
    signal: asyncio.Event | None = None
    """Cancellation flag — set to abort the running command."""


@dataclass(slots=True)
class BashResult:
    """Captured outcome of a bash run."""

    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    full_output_path: str | None = None


def create_local_bash_operations() -> BashOperations:
    """Build a :class:`BashOperations` that targets the local subprocess backend."""
    return BashOperations(exec=local_bash_exec)


async def execute_bash(
    command: str,
    options: BashExecutorOptions | None = None,
) -> BashResult:
    """Run ``command`` locally in :func:`os.getcwd` and capture the output."""
    import os  # noqa: PLC0415 — only used to read cwd

    return await execute_bash_with_operations(
        command,
        os.getcwd(),
        create_local_bash_operations(),
        options,
    )


async def execute_bash_with_operations(
    command: str,
    cwd: str,
    operations: BashOperations,
    options: BashExecutorOptions | None = None,
) -> BashResult:
    """Like :func:`execute_bash` but with a pluggable :class:`BashOperations`.

    Used by remote backends (SSH, container) that ship their own
    ``BashOperations.exec`` callable.
    """
    opts = options or BashExecutorOptions()
    output_chunks: list[str] = []
    output_bytes = 0
    max_output_bytes = DEFAULT_MAX_BYTES * 2

    temp_file_path: str | None = None
    temp_file: Path | None = None  # marker only — actual handle below
    temp_handle = None  # type: ignore[assignment]
    total_bytes = 0

    def ensure_temp_file() -> None:
        nonlocal temp_file_path, temp_file, temp_handle
        if temp_file_path is not None:
            return
        suffix = secrets.token_hex(8)
        temp_file_path = str(Path(tempfile.gettempdir()) / f"nu-bash-{suffix}.log")
        temp_file = Path(temp_file_path)
        temp_handle = temp_file.open("w", encoding="utf-8")
        for chunk in output_chunks:
            temp_handle.write(chunk)

    def on_data(data: bytes) -> None:
        nonlocal output_bytes, total_bytes
        total_bytes += len(data)
        text = sanitize_binary_output(strip_ansi(data.decode("utf-8", errors="replace"))).replace("\r", "")
        if total_bytes > DEFAULT_MAX_BYTES:
            ensure_temp_file()
        if temp_handle is not None:
            temp_handle.write(text)
        output_chunks.append(text)
        output_bytes += len(text)
        while output_bytes > max_output_bytes and len(output_chunks) > 1:
            removed = output_chunks.pop(0)
            output_bytes -= len(removed)
        if opts.on_chunk is not None:
            opts.on_chunk(text)

    exec_fn = operations.exec or local_bash_exec
    cancelled = False
    exit_code: int | None = None
    try:
        exit_code = await exec_fn(
            command=command,
            cwd=cwd,
            on_data=on_data,
            abort_event=opts.signal,
        )
    except BashAborted:
        cancelled = True
    finally:
        if temp_handle is not None:
            temp_handle.close()

    if opts.signal is not None and opts.signal.is_set():
        cancelled = True

    full_output = "".join(output_chunks)
    truncation = truncate_tail(full_output)
    if truncation.truncated and temp_file_path is None:
        ensure_temp_file()
        if temp_handle is not None:
            temp_handle.close()

    return BashResult(
        output=truncation.content if truncation.truncated else full_output,
        exit_code=None if cancelled else exit_code,
        cancelled=cancelled,
        truncated=truncation.truncated,
        full_output_path=temp_file_path,
    )


__all__ = [
    "BashExecutorOptions",
    "BashResult",
    "create_local_bash_operations",
    "execute_bash",
    "execute_bash_with_operations",
    "strip_ansi",
]
