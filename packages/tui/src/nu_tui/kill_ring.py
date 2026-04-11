"""Ring buffer for Emacs-style kill/yank operations.

Direct port of ``packages/tui/src/kill-ring.ts``.

Tracks killed (deleted) text entries. Consecutive kills can accumulate
into a single entry. Supports yank (paste most recent) and yank-pop
(cycle through older entries via :meth:`rotate`).
"""

from __future__ import annotations


class KillRing:
    """Emacs-style kill ring."""

    def __init__(self) -> None:
        self._ring: list[str] = []

    def push(
        self,
        text: str,
        *,
        prepend: bool,
        accumulate: bool = False,
    ) -> None:
        """Add ``text`` to the kill ring.

        ``prepend`` controls direction when accumulating:

        * ``False`` (forward deletion) — append to the existing entry.
        * ``True`` (backward deletion) — prepend to the existing entry.

        ``accumulate=True`` merges with the most recent entry instead of
        creating a new one. Empty strings are ignored.
        """
        if not text:
            return
        if accumulate and self._ring:
            last = self._ring.pop()
            self._ring.append(text + last if prepend else last + text)
        else:
            self._ring.append(text)

    def peek(self) -> str | None:
        """Return the most recent entry without modifying the ring."""
        return self._ring[-1] if self._ring else None

    def rotate(self) -> None:
        """Move the last entry to the front (yank-pop cycling)."""
        if len(self._ring) > 1:
            last = self._ring.pop()
            self._ring.insert(0, last)

    @property
    def length(self) -> int:
        return len(self._ring)


__all__ = ["KillRing"]
