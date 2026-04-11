"""Tests for nu_tui.kill_ring.

Port of the documented contract from ``packages/tui/src/kill-ring.ts``:
push (with prepend / accumulate), peek, rotate, length.
"""

from __future__ import annotations

from nu_tui.kill_ring import KillRing


class TestPush:
    def test_empty_text_is_ignored(self) -> None:
        ring = KillRing()
        ring.push("", prepend=False)
        assert ring.length == 0

    def test_first_push_creates_entry(self) -> None:
        ring = KillRing()
        ring.push("hello", prepend=False)
        assert ring.peek() == "hello"
        assert ring.length == 1

    def test_subsequent_push_creates_new_entry(self) -> None:
        ring = KillRing()
        ring.push("a", prepend=False)
        ring.push("b", prepend=False)
        assert ring.length == 2
        assert ring.peek() == "b"


class TestAccumulate:
    def test_accumulate_appends_forward(self) -> None:
        ring = KillRing()
        ring.push("hello ", prepend=False)
        ring.push("world", prepend=False, accumulate=True)
        assert ring.peek() == "hello world"
        assert ring.length == 1

    def test_accumulate_prepends_backward(self) -> None:
        ring = KillRing()
        ring.push("world", prepend=False)
        ring.push("hello ", prepend=True, accumulate=True)
        assert ring.peek() == "hello world"
        assert ring.length == 1

    def test_accumulate_on_empty_ring_creates_entry(self) -> None:
        ring = KillRing()
        ring.push("first", prepend=False, accumulate=True)
        assert ring.peek() == "first"
        assert ring.length == 1


class TestRotate:
    def test_rotate_moves_last_to_front(self) -> None:
        ring = KillRing()
        ring.push("a", prepend=False)
        ring.push("b", prepend=False)
        ring.push("c", prepend=False)
        # Now order: [a, b, c], peek == c
        ring.rotate()
        assert ring.peek() == "b"  # c moved to front, last is now b
        ring.rotate()
        assert ring.peek() == "a"
        ring.rotate()
        assert ring.peek() == "c"  # full cycle

    def test_rotate_with_one_entry_is_noop(self) -> None:
        ring = KillRing()
        ring.push("only", prepend=False)
        ring.rotate()
        assert ring.peek() == "only"

    def test_rotate_empty_is_noop(self) -> None:
        ring = KillRing()
        ring.rotate()
        assert ring.length == 0


class TestPeek:
    def test_peek_on_empty_returns_none(self) -> None:
        assert KillRing().peek() is None
