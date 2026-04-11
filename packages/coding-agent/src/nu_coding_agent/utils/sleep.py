"""Cancellable async sleep — direct port of ``packages/coding-agent/src/utils/sleep.ts``.

The TS upstream takes an :class:`AbortSignal`. The Python equivalent
takes an :class:`asyncio.Event` and races it against the timer; mirrors
the upstream's "raise Aborted on cancel" semantics.
"""

from __future__ import annotations

import asyncio


class Aborted(Exception):
    """Raised by :func:`sleep` when the abort event is set."""


async def sleep(seconds: float, signal: asyncio.Event | None = None) -> None:
    """Sleep for ``seconds``, raising :class:`Aborted` if ``signal`` fires.

    If ``signal`` is ``None`` this is just a thin wrapper around
    :func:`asyncio.sleep`. If it's an :class:`asyncio.Event` we race the
    sleep against ``signal.wait()`` so a cancel comes through immediately.
    """
    if signal is None:
        await asyncio.sleep(seconds)
        return
    if signal.is_set():
        raise Aborted("Aborted")
    sleep_task = asyncio.create_task(asyncio.sleep(seconds))
    abort_task = asyncio.create_task(signal.wait())
    done, pending = await asyncio.wait(
        {sleep_task, abort_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    if abort_task in done:
        raise Aborted("Aborted")


__all__ = ["Aborted", "sleep"]
