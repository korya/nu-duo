"""Startup timing instrumentation — direct port of ``packages/coding-agent/src/core/timings.ts``.

Enabled by setting ``NU_TIMING=1`` in the environment (the upstream variable
is ``PI_TIMING``; we keep parity with the project rename). When disabled,
all functions are no-ops so the call sites cost nothing.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

_ENABLED = os.environ.get("NU_TIMING") == "1"


@dataclass(slots=True)
class _Timing:
    label: str
    ms: float


@dataclass(slots=True)
class _State:
    timings: list[_Timing]
    last_time: float


_state = _State(timings=[], last_time=time.monotonic())


def reset_timings() -> None:
    """Drop any recorded timings and reset the clock."""
    if not _ENABLED:
        return
    _state.timings.clear()
    _state.last_time = time.monotonic()


def time_event(label: str) -> None:
    """Record a labelled split since the previous call (or reset)."""
    if not _ENABLED:
        return
    now = time.monotonic()
    _state.timings.append(_Timing(label=label, ms=(now - _state.last_time) * 1000))
    _state.last_time = now


def print_timings() -> None:
    """Dump the recorded splits to ``stderr``."""
    if not _ENABLED or not _state.timings:
        return
    print("\n--- Startup Timings ---", file=sys.stderr)
    for t in _state.timings:
        print(f"  {t.label}: {t.ms:.0f}ms", file=sys.stderr)
    total = sum(t.ms for t in _state.timings)
    print(f"  TOTAL: {total:.0f}ms", file=sys.stderr)
    print("------------------------\n", file=sys.stderr)


__all__ = ["print_timings", "reset_timings", "time_event"]
