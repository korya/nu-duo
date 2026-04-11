"""Tests for nu_tui.undo_stack.

Ports the documented contract from
``packages/tui/src/undo-stack.ts``: clone-on-push, pop returns the most
recent snapshot, ``length`` reports stack depth.
"""

from __future__ import annotations

from nu_tui.undo_stack import UndoStack


class TestUndoStack:
    def test_push_clones_state(self) -> None:
        stack: UndoStack[dict[str, int]] = UndoStack()
        original = {"x": 1}
        stack.push(original)
        # Mutating the original after push should not affect the snapshot.
        original["x"] = 99
        popped = stack.pop()
        assert popped == {"x": 1}

    def test_pop_returns_most_recent(self) -> None:
        stack: UndoStack[int] = UndoStack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        assert stack.pop() == 3
        assert stack.pop() == 2
        assert stack.pop() == 1
        assert stack.pop() is None

    def test_pop_empty_returns_none(self) -> None:
        stack: UndoStack[int] = UndoStack()
        assert stack.pop() is None

    def test_clear_removes_all(self) -> None:
        stack: UndoStack[int] = UndoStack()
        stack.push(1)
        stack.push(2)
        stack.clear()
        assert stack.length == 0
        assert stack.pop() is None

    def test_length_property(self) -> None:
        stack: UndoStack[int] = UndoStack()
        assert stack.length == 0
        stack.push(1)
        assert stack.length == 1
        stack.push(2)
        assert stack.length == 2
        stack.pop()
        assert stack.length == 1
