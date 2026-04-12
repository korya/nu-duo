"""``Input`` — port of ``packages/tui/src/components/input.ts``.

Single-line text input with cursor movement, horizontal scrolling,
Emacs-style kill ring (C-k/C-y/M-y), undo (C-z), word-by-word
navigation, and bracketed paste support. Renders a ``"> "`` prompt
followed by the text with the cursor shown in reverse-video.

Integrates with the existing :mod:`nu_tui.keybindings`,
:class:`nu_tui.kill_ring.KillRing`, and
:class:`nu_tui.undo_stack.UndoStack` ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nu_tui.component import Component
from nu_tui.kill_ring import KillRing
from nu_tui.undo_stack import UndoStack
from nu_tui.utils import (
    CURSOR_MARKER,
    decode_kitty_printable,
    is_punctuation_char,
    is_whitespace_char,
    slice_by_column,
    visible_width,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class _InputState:
    value: str
    cursor: int


class Input(Component):
    """Single-line text input with horizontal scrolling.

    Callbacks:
    - ``on_submit(value)`` — Enter was pressed.
    - ``on_escape()`` — Escape or Ctrl-C was pressed.

    The ``focused`` flag mirrors the TS ``Focusable`` interface: when
    ``True``, the render output includes a ``CURSOR_MARKER`` escape at
    the cursor position so the TUI bridge can position the hardware
    cursor for IME support.
    """

    def __init__(self) -> None:
        self._value = ""
        self._cursor = 0
        self.focused = False

        # Callbacks
        self.on_submit: Callable[[str], None] | None = None
        self.on_escape: Callable[[], None] | None = None

        # Bracketed paste
        self._paste_buffer = ""
        self._in_paste = False

        # Kill ring + undo
        self._kill_ring = KillRing()
        self._last_action: str | None = None  # "kill" | "yank" | "type-word" | None
        self._undo_stack: UndoStack[_InputState] = UndoStack()

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    def get_value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value
        self._cursor = min(self._cursor, len(value))

    def set_cursor(self, position: int) -> None:
        """Move the cursor to ``position``, clamped to ``[0, len(value)]``."""
        self._cursor = max(0, min(position, len(self._value)))

    @property
    def value(self) -> str:
        return self._value

    @property
    def cursor(self) -> int:
        return self._cursor

    # ------------------------------------------------------------------
    # handle_input — the main key dispatch
    # ------------------------------------------------------------------

    def handle_input(self, data: str) -> None:
        # Bracketed paste
        if "\x1b[200~" in data:
            self._in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "")

        if self._in_paste:
            self._paste_buffer += data
            end_idx = self._paste_buffer.find("\x1b[201~")
            if end_idx != -1:
                paste_content = self._paste_buffer[:end_idx]
                self._handle_paste(paste_content)
                self._in_paste = False
                remaining = self._paste_buffer[end_idx + 6 :]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
            return

        from nu_tui.keybindings import get_keybindings  # noqa: PLC0415

        kb = get_keybindings()

        if kb.matches(data, "tui.select.cancel"):
            if self.on_escape:
                self.on_escape()
            return
        if kb.matches(data, "tui.editor.undo"):
            self._undo()
            return
        if kb.matches(data, "tui.input.submit") or data == "\n":
            if self.on_submit:
                self.on_submit(self._value)
            return

        # Deletion
        if kb.matches(data, "tui.editor.deleteCharBackward"):
            self._backspace()
            return
        if kb.matches(data, "tui.editor.deleteCharForward"):
            self._forward_delete()
            return
        if kb.matches(data, "tui.editor.deleteWordBackward"):
            self._delete_word_backwards()
            return
        if kb.matches(data, "tui.editor.deleteWordForward"):
            self._delete_word_forward()
            return
        if kb.matches(data, "tui.editor.deleteToLineStart"):
            self._delete_to_line_start()
            return
        if kb.matches(data, "tui.editor.deleteToLineEnd"):
            self._delete_to_line_end()
            return

        # Kill ring
        if kb.matches(data, "tui.editor.yank"):
            self._yank()
            return
        if kb.matches(data, "tui.editor.yankPop"):
            self._yank_pop()
            return

        # Cursor movement
        if kb.matches(data, "tui.editor.cursorLeft"):
            self._last_action = None
            if self._cursor > 0:
                self._cursor -= 1
            return
        if kb.matches(data, "tui.editor.cursorRight"):
            self._last_action = None
            if self._cursor < len(self._value):
                self._cursor += 1
            return
        if kb.matches(data, "tui.editor.cursorLineStart"):
            self._last_action = None
            self._cursor = 0
            return
        if kb.matches(data, "tui.editor.cursorLineEnd"):
            self._last_action = None
            self._cursor = len(self._value)
            return
        if kb.matches(data, "tui.editor.cursorWordLeft"):
            self._move_word_backwards()
            return
        if kb.matches(data, "tui.editor.cursorWordRight"):
            self._move_word_forwards()
            return

        # Kitty CSI-u printable
        kitty = decode_kitty_printable(data)
        if kitty is not None:
            self._insert_character(kitty)
            return

        # Regular printable character
        if not any(ord(ch) < 32 or ord(ch) == 0x7F or 0x80 <= ord(ch) <= 0x9F for ch in data):
            self._insert_character(data)

    # ------------------------------------------------------------------
    # Editing operations
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        self._undo_stack.push(_InputState(value=self._value, cursor=self._cursor))

    def _undo(self) -> None:
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._value = snapshot.value
        self._cursor = snapshot.cursor
        self._last_action = None

    def _insert_character(self, char: str) -> None:
        if is_whitespace_char(char) or self._last_action != "type-word":
            self._push_undo()
        self._last_action = "type-word"
        self._value = self._value[: self._cursor] + char + self._value[self._cursor :]
        self._cursor += len(char)

    def _backspace(self) -> None:
        self._last_action = None
        if self._cursor > 0:
            self._push_undo()
            self._value = self._value[: self._cursor - 1] + self._value[self._cursor :]
            self._cursor -= 1

    def _forward_delete(self) -> None:
        self._last_action = None
        if self._cursor < len(self._value):
            self._push_undo()
            self._value = self._value[: self._cursor] + self._value[self._cursor + 1 :]

    def _delete_to_line_start(self) -> None:
        if self._cursor == 0:
            return
        self._push_undo()
        deleted = self._value[: self._cursor]
        self._kill_ring.push(deleted, prepend=True, accumulate=self._last_action == "kill")
        self._last_action = "kill"
        self._value = self._value[self._cursor :]
        self._cursor = 0

    def _delete_to_line_end(self) -> None:
        if self._cursor >= len(self._value):
            return
        self._push_undo()
        deleted = self._value[self._cursor :]
        self._kill_ring.push(deleted, prepend=False, accumulate=self._last_action == "kill")
        self._last_action = "kill"
        self._value = self._value[: self._cursor]

    def _delete_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        was_kill = self._last_action == "kill"
        self._push_undo()
        old_cursor = self._cursor
        self._move_word_backwards()
        delete_from = self._cursor
        self._cursor = old_cursor
        deleted = self._value[delete_from : self._cursor]
        self._kill_ring.push(deleted, prepend=True, accumulate=was_kill)
        self._last_action = "kill"
        self._value = self._value[:delete_from] + self._value[self._cursor :]
        self._cursor = delete_from

    def _delete_word_forward(self) -> None:
        if self._cursor >= len(self._value):
            return
        was_kill = self._last_action == "kill"
        self._push_undo()
        old_cursor = self._cursor
        self._move_word_forwards()
        delete_to = self._cursor
        self._cursor = old_cursor
        deleted = self._value[self._cursor : delete_to]
        self._kill_ring.push(deleted, prepend=False, accumulate=was_kill)
        self._last_action = "kill"
        self._value = self._value[: self._cursor] + self._value[delete_to:]

    def _yank(self) -> None:
        text = self._kill_ring.peek()
        if not text:
            return
        self._push_undo()
        self._value = self._value[: self._cursor] + text + self._value[self._cursor :]
        self._cursor += len(text)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return
        self._push_undo()
        prev_text = self._kill_ring.peek() or ""
        self._value = self._value[: self._cursor - len(prev_text)] + self._value[self._cursor :]
        self._cursor -= len(prev_text)
        self._kill_ring.rotate()
        text = self._kill_ring.peek() or ""
        self._value = self._value[: self._cursor] + text + self._value[self._cursor :]
        self._cursor += len(text)
        self._last_action = "yank"

    # ------------------------------------------------------------------
    # Word movement
    # ------------------------------------------------------------------

    def _move_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        self._last_action = None
        # Skip trailing whitespace
        while self._cursor > 0 and is_whitespace_char(self._value[self._cursor - 1]):
            self._cursor -= 1
        if self._cursor > 0:
            if is_punctuation_char(self._value[self._cursor - 1]):
                while self._cursor > 0 and is_punctuation_char(self._value[self._cursor - 1]):
                    self._cursor -= 1
            else:
                while (
                    self._cursor > 0
                    and not is_whitespace_char(self._value[self._cursor - 1])
                    and not is_punctuation_char(self._value[self._cursor - 1])
                ):
                    self._cursor -= 1

    def _move_word_forwards(self) -> None:
        if self._cursor >= len(self._value):
            return
        self._last_action = None
        # Skip leading whitespace
        while self._cursor < len(self._value) and is_whitespace_char(self._value[self._cursor]):
            self._cursor += 1
        if self._cursor < len(self._value):
            if is_punctuation_char(self._value[self._cursor]):
                while self._cursor < len(self._value) and is_punctuation_char(self._value[self._cursor]):
                    self._cursor += 1
            else:
                while (
                    self._cursor < len(self._value)
                    and not is_whitespace_char(self._value[self._cursor])
                    and not is_punctuation_char(self._value[self._cursor])
                ):
                    self._cursor += 1

    # ------------------------------------------------------------------
    # Paste
    # ------------------------------------------------------------------

    def _handle_paste(self, pasted_text: str) -> None:
        self._last_action = None
        self._push_undo()
        clean = pasted_text.replace("\r\n", "").replace("\r", "").replace("\n", "").replace("\t", "    ")
        self._value = self._value[: self._cursor] + clean + self._value[self._cursor :]
        self._cursor += len(clean)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        prompt = "> "
        available = width - len(prompt)
        if available <= 0:
            return [prompt]

        total_width = visible_width(self._value)
        if total_width < available:
            visible_text = self._value
            cursor_display = self._cursor
        else:
            scroll_width = available - 1 if self._cursor == len(self._value) else available
            cursor_col = visible_width(self._value[: self._cursor])
            if scroll_width > 0:
                half = scroll_width // 2
                if cursor_col < half:
                    start_col = 0
                elif cursor_col > total_width - half:
                    start_col = max(0, total_width - scroll_width)
                else:
                    start_col = max(0, cursor_col - half)
                visible_text = slice_by_column(self._value, start_col, scroll_width, pad=True)
                before_cursor = slice_by_column(self._value, start_col, max(0, cursor_col - start_col), pad=True)
                cursor_display = len(before_cursor)
            else:
                visible_text = ""
                cursor_display = 0

        before = visible_text[:cursor_display]
        at_cursor = visible_text[cursor_display] if cursor_display < len(visible_text) else " "
        after = visible_text[cursor_display + len(at_cursor) :]

        marker = CURSOR_MARKER if self.focused else ""
        cursor_char = f"\x1b[7m{at_cursor}\x1b[27m"
        text_with_cursor = f"{before}{marker}{cursor_char}{after}"

        vis_len = visible_width(text_with_cursor)
        padding = " " * max(0, available - vis_len)
        return [f"{prompt}{text_with_cursor}{padding}"]


__all__ = ["Input"]
