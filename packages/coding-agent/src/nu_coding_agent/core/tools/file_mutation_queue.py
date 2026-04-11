"""Per-file mutation serialization — direct port of ``packages/coding-agent/src/core/tools/file-mutation-queue.ts``.

The agent loop runs tool calls in parallel, so two ``write`` (or
``edit``) calls hitting the same file would race. This helper guarantees
that mutations targeting the same resolved path run one after another,
while mutations on different files keep their parallelism.

Implementation: a module-level dict of :class:`asyncio.Lock` keyed by the
resolved (``Path.resolve``, with ``realpath`` fallthrough already
folded in) path. Locks are removed once their queue drains so the dict
doesn't grow unbounded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


_locks: dict[str, asyncio.Lock] = {}
_pending: dict[str, int] = {}


def _mutation_queue_key(file_path: str) -> str:
    """Return a canonical absolute key for ``file_path``.

    Mirrors the upstream ``getMutationQueueKey`` which calls
    ``realpathSync.native`` and falls back to the resolved path. We use
    :meth:`pathlib.Path.resolve` (which itself follows symlinks where
    possible) and fall back to the absolute path on error.
    """
    try:
        return str(Path(file_path).resolve())
    except OSError:
        return str(Path(file_path).absolute())


async def with_file_mutation_queue[T](
    file_path: str,
    fn: Callable[[], Awaitable[T]],
) -> T:
    """Run ``fn`` while holding the mutation lock for ``file_path``.

    Operations targeting different files still run concurrently;
    operations on the same file (by resolved path) are serialized in
    request order.
    """
    key = _mutation_queue_key(file_path)
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    _pending[key] = _pending.get(key, 0) + 1
    try:
        async with lock:
            return await fn()
    finally:
        _pending[key] -= 1
        if _pending[key] <= 0:
            _pending.pop(key, None)
            _locks.pop(key, None)


__all__ = ["with_file_mutation_queue"]
