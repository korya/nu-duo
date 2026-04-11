"""Stdout takeover — direct port of ``packages/coding-agent/src/core/output-guard.ts``.

The interactive TUI owns the terminal and any stray ``print()`` from a
tool, library, or extension would corrupt its rendering. ``take_over_stdout``
swaps :data:`sys.stdout` for a shim that forwards every write to
:data:`sys.stderr`; tools that legitimately need to reach the original
stdout call :func:`write_raw_stdout`. ``restore_stdout`` puts everything
back so print/RPC modes can flush their output normally on shutdown.

The upstream version monkey-patches ``process.stdout.write``; the Python
analogue replaces ``sys.stdout`` with a thin wrapper class because writes
in CPython go through ``sys.stdout``, not a single bound method.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import IO, Any


@dataclass(slots=True)
class _Takeover:
    raw_stdout: IO[str]
    raw_stderr: IO[str]
    shim: _StdoutShim


@dataclass(slots=True)
class _GuardState:
    current: _Takeover | None = None


_state = _GuardState()


class _StdoutShim:
    """File-like object that forwards every write to ``sys.stderr``."""

    def __init__(self, raw_stderr: IO[str]) -> None:
        self._raw_stderr = raw_stderr

    def write(self, data: str) -> int:
        return self._raw_stderr.write(data)

    def flush(self) -> None:
        self._raw_stderr.flush()

    def isatty(self) -> bool:
        return self._raw_stderr.isatty()

    def __getattr__(self, name: str) -> Any:
        # Forward everything else (encoding, fileno, etc.) to stderr so
        # libraries probing the stream don't blow up.
        return getattr(self._raw_stderr, name)


def take_over_stdout() -> None:
    """Redirect ``sys.stdout`` to ``sys.stderr`` until :func:`restore_stdout`."""
    if _state.current is not None:
        return
    raw_stdout = sys.stdout
    raw_stderr = sys.stderr
    shim = _StdoutShim(raw_stderr)
    sys.stdout = shim  # type: ignore[assignment]
    _state.current = _Takeover(raw_stdout=raw_stdout, raw_stderr=raw_stderr, shim=shim)


def restore_stdout() -> None:
    """Put ``sys.stdout`` back the way it was."""
    current = _state.current
    if current is None:
        return
    sys.stdout = current.raw_stdout
    _state.current = None


def is_stdout_taken_over() -> bool:
    return _state.current is not None


def write_raw_stdout(text: str) -> None:
    """Write directly to the original stdout, bypassing any active takeover."""
    current = _state.current
    if current is not None:
        current.raw_stdout.write(text)
        return
    sys.stdout.write(text)


async def flush_raw_stdout() -> None:
    """Flush the original stdout. Async to match the upstream signature."""
    current = _state.current
    if current is not None:
        current.raw_stdout.flush()
        return
    sys.stdout.flush()


__all__ = [
    "flush_raw_stdout",
    "is_stdout_taken_over",
    "restore_stdout",
    "take_over_stdout",
    "write_raw_stdout",
]
