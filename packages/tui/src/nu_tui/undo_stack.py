"""Generic undo stack with clone-on-push semantics.

Direct port of ``packages/tui/src/undo-stack.ts``. Stores deep clones of
state snapshots; popped snapshots are returned directly (no re-cloning)
since they are already detached.
"""

from __future__ import annotations

import copy


class UndoStack[S]:
    """LIFO stack of deep-cloned state snapshots."""

    def __init__(self) -> None:
        self._stack: list[S] = []

    def push(self, state: S) -> None:
        """Push a deep clone of ``state`` onto the stack."""
        self._stack.append(copy.deepcopy(state))

    def pop(self) -> S | None:
        """Pop and return the most recent snapshot, or ``None`` if empty."""
        if not self._stack:
            return None
        return self._stack.pop()

    def clear(self) -> None:
        """Remove all snapshots."""
        self._stack.clear()

    @property
    def length(self) -> int:
        return len(self._stack)


__all__ = ["UndoStack"]
