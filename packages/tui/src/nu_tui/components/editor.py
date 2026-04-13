"""``Editor`` — multi-line text editor component.

Port of the relevant subset of ``packages/tui/src/components/editor.ts``.

Supports multi-line editing with cursor movement, word-wrap rendering,
Emacs-style kill ring (C-k/C-y/M-y), undo (C-/), word-by-word navigation,
page scrolling, and bracketed paste.  No autocomplete, no paste markers,
no history navigation, no jump mode, no slash commands.

Submit contract (matching upstream TS):
- Enter  → submit (if ``on_submit`` set and content exists).
- Shift+Enter / Alt+Enter → insert newline.
- Ctrl+Enter → also insert newline (fallback for terminals that can't
  distinguish Shift+Enter).

The component is self-contained and does not require a TUI instance;
pass it ``width`` in :meth:`render` and wire callbacks yourself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nu_tui.component import Component
from nu_tui.kill_ring import KillRing
from nu_tui.undo_stack import UndoStack
from nu_tui.utils import (
    CURSOR_MARKER,
    decode_kitty_printable,
    is_punctuation_char,
    is_whitespace_char,
    truncate_to_width,
    visible_width,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# State snapshot (used by undo stack)
# ---------------------------------------------------------------------------


@dataclass
class EditorState:
    """Snapshot of the editor's mutable text + cursor state."""

    lines: list[str]
    cursor_line: int
    cursor_col: int

    def clone(self) -> EditorState:
        return EditorState(
            lines=list(self.lines),
            cursor_line=self.cursor_line,
            cursor_col=self.cursor_col,
        )


# ---------------------------------------------------------------------------
# Word-wrap helpers (no ANSI, plain text only — lines never contain escapes)
# ---------------------------------------------------------------------------


def _wrap_line(line: str, max_width: int) -> list[str]:
    """Word-wrap a single logical line into visual segments of ``max_width`` columns.

    Tries soft-wrapping at word boundaries first; falls back to hard
    character breaks when a single word is wider than ``max_width``.
    Always returns at least one segment (empty string for an empty line).
    """
    if max_width <= 0:
        return [line] if line else [""]

    if visible_width(line) <= max_width:
        return [line]

    segments: list[str] = []
    remaining = line

    while visible_width(remaining) > max_width:
        # Try to find the last soft-break opportunity within max_width columns.
        best_break = -1
        current_width = 0
        for i, ch in enumerate(remaining):
            ch_w = visible_width(ch)
            if current_width + ch_w > max_width:
                break
            current_width += ch_w
            if ch == " ":
                best_break = i

        if best_break > 0:
            # Soft break: include everything up to and including the space,
            # then strip leading spaces on the continuation.
            segments.append(remaining[: best_break + 1])
            remaining = remaining[best_break + 1 :]
        else:
            # Hard break: slice exactly max_width visible columns.
            hard_cut = 0
            accumulated = 0
            for i, ch in enumerate(remaining):
                ch_w = visible_width(ch)
                if accumulated + ch_w > max_width:
                    hard_cut = i
                    break
                accumulated += ch_w
            else:
                hard_cut = len(remaining)

            if hard_cut == 0:
                hard_cut = 1  # prevent infinite loop on very wide single chars
            segments.append(remaining[:hard_cut])
            remaining = remaining[hard_cut:]

    segments.append(remaining)
    return segments


# ---------------------------------------------------------------------------
# Main Editor class
# ---------------------------------------------------------------------------


class Editor(Component):
    """Multi-line text editor.

    Args:
        padding_x:        Horizontal padding on each side (columns).
        max_visible_lines: Maximum number of visual lines shown at once
                           (scroll window height).
        border_fn:        Optional callable that wraps a rendered line
                          with border/decoration characters.  Receives the
                          raw line string and must return a decorated string
                          of the same visible width + border overhead.

    Callbacks:
        on_submit(text)   Called on Enter when content is non-empty.
        on_change(text)   Called after every edit.
        on_escape()       Called on Escape / Ctrl-C.
    """

    def __init__(
        self,
        padding_x: int = 0,
        max_visible_lines: int = 15,
        border_fn: Callable[[str], str] | None = None,
    ) -> None:
        # Text state
        self._lines: list[str] = [""]
        self._cursor_line: int = 0
        self._cursor_col: int = 0

        # Visual scroll offset (in *visual* lines, not logical lines)
        self._scroll_offset: int = 0

        # Layout config
        self.padding_x = padding_x
        self.max_visible_lines = max_visible_lines
        self.border_fn = border_fn

        # Focus / behaviour flags
        self.focused: bool = False
        self.disable_submit: bool = False

        # Callbacks
        self.on_submit: Callable[[str], None] | None = None
        self.on_change: Callable[[str], None] | None = None
        self.on_escape: Callable[[], None] | None = None

        # Bracketed paste state
        self._paste_buffer: str = ""
        self._in_paste: bool = False

        # Kill ring + undo
        self._kill_ring = KillRing()
        self._last_action: str | None = None  # "kill" | "yank" | "type-word" | None
        self._undo_stack: UndoStack[EditorState] = UndoStack()

        # Prompt history (up/down navigation)
        self._history: list[str] = []
        self._history_index: int = -1  # -1 = not browsing
        self._saved_text: str = ""  # text before entering history mode

        # Autocomplete (set via set_autocomplete_provider)
        self._autocomplete_provider: Any = None

    # ------------------------------------------------------------------
    # Public state API
    # ------------------------------------------------------------------

    def get_text(self) -> str:
        """Return the full editor content as a newline-joined string."""
        return "\n".join(self._lines)

    def set_text(self, text: str) -> None:
        """Replace all content and reset the cursor to end of text."""
        self._lines = text.split("\n") if text else [""]
        if not self._lines:
            self._lines = [""]
        self._cursor_line = len(self._lines) - 1
        self._cursor_col = len(self._lines[-1])
        self._scroll_offset = 0
        self._last_action = None

    def get_lines(self) -> list[str]:
        """Return a copy of the logical lines list."""
        return list(self._lines)

    def add_to_history(self, text: str) -> None:
        """Add a prompt to the history for up/down navigation."""
        text = text.strip()
        if not text:
            return
        # Deduplicate: remove if already at top
        if self._history and self._history[0] == text:
            return
        self._history.insert(0, text)
        # Cap at 100 entries
        if len(self._history) > 100:
            self._history = self._history[:100]
        self._history_index = -1

    def set_autocomplete_provider(self, provider: Any) -> None:
        """Set the autocomplete provider for tab completion."""
        self._autocomplete_provider = provider

    def get_cursor(self) -> tuple[int, int]:
        """Return the cursor position as ``(line_index, col_index)``."""
        return (self._cursor_line, self._cursor_col)

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

        # Escape / cancel
        if kb.matches(data, "tui.select.cancel"):
            if self.on_escape:
                self.on_escape()
            return

        # Undo
        if kb.matches(data, "tui.editor.undo"):
            self._undo()
            return

        # Newline insertion (Shift+Enter / Alt+Enter / Ctrl+Enter)
        if kb.matches(data, "tui.input.newLine") or data in ("\x1b\r", "\x0d"):
            self._add_newline()
            self._fire_change()
            return

        # Submit (Enter) — submit when content exists, otherwise insert newline
        if kb.matches(data, "tui.input.submit") or data == "\n":
            text = self.get_text()
            has_content = any(ln.strip() for ln in self._lines)
            if not self.disable_submit and has_content and self.on_submit:
                self.on_submit(text)
            else:
                self._add_newline()
                self._fire_change()
            return

        # Deletion
        if kb.matches(data, "tui.editor.deleteCharBackward"):
            self._backspace()
            self._fire_change()
            return
        if kb.matches(data, "tui.editor.deleteCharForward"):
            self._forward_delete()
            self._fire_change()
            return
        if kb.matches(data, "tui.editor.deleteWordBackward"):
            self._delete_word_backwards()
            self._fire_change()
            return
        if kb.matches(data, "tui.editor.deleteWordForward"):
            self._delete_word_forward()
            self._fire_change()
            return
        if kb.matches(data, "tui.editor.deleteToLineStart"):
            self._delete_to_line_start()
            self._fire_change()
            return
        if kb.matches(data, "tui.editor.deleteToLineEnd"):
            self._delete_to_line_end()
            self._fire_change()
            return

        # Kill ring yank
        if kb.matches(data, "tui.editor.yank"):
            self._yank()
            self._fire_change()
            return
        if kb.matches(data, "tui.editor.yankPop"):
            self._yank_pop()
            self._fire_change()
            return

        # Cursor movement
        if kb.matches(data, "tui.editor.cursorLeft"):
            self._last_action = None
            self._move_cursor(0, -1)
            return
        if kb.matches(data, "tui.editor.cursorRight"):
            self._last_action = None
            self._move_cursor(0, 1)
            return
        if kb.matches(data, "tui.editor.cursorUp"):
            self._last_action = None
            # History navigation: when on the first logical line, browse history
            if self._cursor_line == 0 and self._history:
                self._navigate_history(-1)
                return
            self._move_cursor(-1, 0)
            return
        if kb.matches(data, "tui.editor.cursorDown"):
            self._last_action = None
            # History navigation: when browsing history, go forward
            if self._history_index >= 0:
                self._navigate_history(1)
                return
            self._move_cursor(1, 0)
            return
        if kb.matches(data, "tui.editor.cursorLineStart"):
            self._last_action = None
            self._cursor_col = 0
            return
        if kb.matches(data, "tui.editor.cursorLineEnd"):
            self._last_action = None
            self._cursor_col = len(self._lines[self._cursor_line])
            return
        if kb.matches(data, "tui.editor.cursorWordLeft"):
            self._move_word_backwards()
            return
        if kb.matches(data, "tui.editor.cursorWordRight"):
            self._move_word_forwards()
            return
        if kb.matches(data, "tui.editor.pageUp"):
            self._page_scroll(-1)
            return
        if kb.matches(data, "tui.editor.pageDown"):
            self._page_scroll(1)
            return

        # Kitty CSI-u printable
        kitty = decode_kitty_printable(data)
        if kitty is not None:
            self._insert_character(kitty)
            self._fire_change()
            return

        # Regular printable character (no control chars)
        if not any(ord(ch) < 32 or ord(ch) == 0x7F or 0x80 <= ord(ch) <= 0x9F for ch in data):
            self._insert_character(data)
            self._fire_change()

    # ------------------------------------------------------------------
    # Undo / push helpers
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        self._undo_stack.push(
            EditorState(
                lines=list(self._lines),
                cursor_line=self._cursor_line,
                cursor_col=self._cursor_col,
            )
        )

    def _undo(self) -> None:
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._lines = snapshot.lines
        self._cursor_line = snapshot.cursor_line
        self._cursor_col = snapshot.cursor_col
        self._last_action = None
        self._fire_change()

    # ------------------------------------------------------------------
    # Core editing operations
    # ------------------------------------------------------------------

    def _insert_character(self, char: str) -> None:
        """Insert ``char`` at the cursor, coalescing undo entries for word runs."""
        if is_whitespace_char(char) or self._last_action != "type-word":
            self._push_undo()
        self._last_action = "type-word"

        line = self._lines[self._cursor_line]
        self._lines[self._cursor_line] = line[: self._cursor_col] + char + line[self._cursor_col :]
        self._cursor_col += len(char)

    def _backspace(self) -> None:
        """Delete the character immediately before the cursor.

        When the cursor is at the start of a line, the current line is
        merged into the previous one.
        """
        self._last_action = None

        if self._cursor_col > 0:
            self._push_undo()
            line = self._lines[self._cursor_line]
            self._lines[self._cursor_line] = line[: self._cursor_col - 1] + line[self._cursor_col :]
            self._cursor_col -= 1
        elif self._cursor_line > 0:
            # Merge into previous line
            self._push_undo()
            prev_line = self._lines[self._cursor_line - 1]
            cur_line = self._lines[self._cursor_line]
            new_col = len(prev_line)
            self._lines[self._cursor_line - 1] = prev_line + cur_line
            del self._lines[self._cursor_line]
            self._cursor_line -= 1
            self._cursor_col = new_col

    def _forward_delete(self) -> None:
        """Delete the character at the cursor.

        When the cursor is at the end of a line, the next line is merged in.
        """
        self._last_action = None
        line = self._lines[self._cursor_line]

        if self._cursor_col < len(line):
            self._push_undo()
            self._lines[self._cursor_line] = line[: self._cursor_col] + line[self._cursor_col + 1 :]
        elif self._cursor_line < len(self._lines) - 1:
            # Merge next line in
            self._push_undo()
            next_line = self._lines[self._cursor_line + 1]
            self._lines[self._cursor_line] = line + next_line
            del self._lines[self._cursor_line + 1]

    def _add_newline(self) -> None:
        """Split the current line at the cursor, inserting a new logical line."""
        self._push_undo()
        self._last_action = None

        line = self._lines[self._cursor_line]
        before = line[: self._cursor_col]
        after = line[self._cursor_col :]
        self._lines[self._cursor_line] = before
        self._lines.insert(self._cursor_line + 1, after)
        self._cursor_line += 1
        self._cursor_col = 0

    def _delete_to_line_start(self) -> None:
        """Delete from the cursor to the start of the logical line (kill ring)."""
        if self._cursor_col == 0:
            return
        self._push_undo()
        line = self._lines[self._cursor_line]
        deleted = line[: self._cursor_col]
        self._kill_ring.push(deleted, prepend=True, accumulate=self._last_action == "kill")
        self._last_action = "kill"
        self._lines[self._cursor_line] = line[self._cursor_col :]
        self._cursor_col = 0

    def _delete_to_line_end(self) -> None:
        """Delete from the cursor to the end of the logical line (kill ring)."""
        line = self._lines[self._cursor_line]
        if self._cursor_col >= len(line):
            return
        self._push_undo()
        deleted = line[self._cursor_col :]
        self._kill_ring.push(deleted, prepend=False, accumulate=self._last_action == "kill")
        self._last_action = "kill"
        self._lines[self._cursor_line] = line[: self._cursor_col]

    def _delete_word_backwards(self) -> None:
        """Delete the word immediately before the cursor (kill ring)."""
        if self._cursor_col == 0 and self._cursor_line == 0:
            return
        was_kill = self._last_action == "kill"
        self._push_undo()

        if self._cursor_col == 0:
            # At start of line: delete the newline (merge with previous line)
            prev_line = self._lines[self._cursor_line - 1]
            cur_line = self._lines[self._cursor_line]
            self._kill_ring.push("\n", prepend=True, accumulate=was_kill)
            self._last_action = "kill"
            self._lines[self._cursor_line - 1] = prev_line + cur_line
            del self._lines[self._cursor_line]
            self._cursor_line -= 1
            self._cursor_col = len(prev_line)
            return

        saved_line = self._cursor_line
        saved_col = self._cursor_col
        self._move_word_backwards()
        delete_from_col = self._cursor_col

        # Restore to measure deleted text
        self._cursor_line = saved_line
        self._cursor_col = saved_col

        line = self._lines[self._cursor_line]
        deleted = line[delete_from_col:saved_col]
        self._kill_ring.push(deleted, prepend=True, accumulate=was_kill)
        self._last_action = "kill"
        self._lines[self._cursor_line] = line[:delete_from_col] + line[saved_col:]
        self._cursor_col = delete_from_col

    def _delete_word_forward(self) -> None:
        """Delete the word immediately after the cursor (kill ring)."""
        line = self._lines[self._cursor_line]
        if self._cursor_col >= len(line) and self._cursor_line >= len(self._lines) - 1:
            return
        was_kill = self._last_action == "kill"
        self._push_undo()

        if self._cursor_col >= len(line):
            # At end of line: delete the newline (merge with next line)
            next_line = self._lines[self._cursor_line + 1]
            self._kill_ring.push("\n", prepend=False, accumulate=was_kill)
            self._last_action = "kill"
            self._lines[self._cursor_line] = line + next_line
            del self._lines[self._cursor_line + 1]
            return

        saved_col = self._cursor_col
        self._move_word_forwards()
        delete_to_col = self._cursor_col
        self._cursor_col = saved_col

        deleted = line[saved_col:delete_to_col]
        self._kill_ring.push(deleted, prepend=False, accumulate=was_kill)
        self._last_action = "kill"
        self._lines[self._cursor_line] = line[:saved_col] + line[delete_to_col:]

    def _yank(self) -> None:
        """Insert the most recent kill-ring entry at the cursor."""
        text = self._kill_ring.peek()
        if not text:
            return
        self._push_undo()
        self._handle_paste(text, record_undo=False)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        """Replace the last yanked text with the next kill-ring entry."""
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return
        self._push_undo()
        prev_text = self._kill_ring.peek() or ""
        # Remove the previously yanked text by rewinding character by character.
        # Since yanked text may span lines we use set_text / restore approach.
        self._remove_text_before_cursor(len(prev_text))
        self._kill_ring.rotate()
        new_text = self._kill_ring.peek() or ""
        self._handle_paste(new_text, record_undo=False)
        self._last_action = "yank"

    def _remove_text_before_cursor(self, char_count: int) -> None:
        """Remove exactly ``char_count`` characters (including newlines) before cursor."""
        for _ in range(char_count):
            if self._cursor_col > 0:
                line = self._lines[self._cursor_line]
                self._lines[self._cursor_line] = line[: self._cursor_col - 1] + line[self._cursor_col :]
                self._cursor_col -= 1
            elif self._cursor_line > 0:
                prev_line = self._lines[self._cursor_line - 1]
                cur_line = self._lines[self._cursor_line]
                new_col = len(prev_line)
                self._lines[self._cursor_line - 1] = prev_line + cur_line
                del self._lines[self._cursor_line]
                self._cursor_line -= 1
                self._cursor_col = new_col
            else:
                break

    # ------------------------------------------------------------------
    # Word movement
    # ------------------------------------------------------------------

    def _move_word_backwards(self) -> None:
        """Move cursor to the start of the current or previous word."""
        self._last_action = None
        # Skip trailing whitespace (or cross newlines)
        while True:
            if self._cursor_col > 0:
                ch = self._lines[self._cursor_line][self._cursor_col - 1]
                if is_whitespace_char(ch):
                    self._cursor_col -= 1
                    continue
            elif self._cursor_line > 0:
                self._cursor_line -= 1
                self._cursor_col = len(self._lines[self._cursor_line])
                continue
            break

        if self._cursor_col == 0 and self._cursor_line == 0:
            return

        # Move back through a word or punctuation run
        ch = self._lines[self._cursor_line][self._cursor_col - 1]
        if is_punctuation_char(ch):
            while self._cursor_col > 0 and is_punctuation_char(self._lines[self._cursor_line][self._cursor_col - 1]):
                self._cursor_col -= 1
        else:
            while self._cursor_col > 0:
                ch = self._lines[self._cursor_line][self._cursor_col - 1]
                if is_whitespace_char(ch) or is_punctuation_char(ch):
                    break
                self._cursor_col -= 1

    def _move_word_forwards(self) -> None:
        """Move cursor to the end of the current or next word."""
        self._last_action = None
        # Skip leading whitespace (or cross newlines)
        while True:
            line = self._lines[self._cursor_line]
            if self._cursor_col < len(line):
                ch = line[self._cursor_col]
                if is_whitespace_char(ch):
                    self._cursor_col += 1
                    continue
            elif self._cursor_line < len(self._lines) - 1:
                self._cursor_line += 1
                self._cursor_col = 0
                continue
            break

        line = self._lines[self._cursor_line]
        if self._cursor_col >= len(line):
            return

        ch = line[self._cursor_col]
        if is_punctuation_char(ch):
            while self._cursor_col < len(line) and is_punctuation_char(line[self._cursor_col]):
                self._cursor_col += 1
        else:
            while self._cursor_col < len(line):
                ch = line[self._cursor_col]
                if is_whitespace_char(ch) or is_punctuation_char(ch):
                    break
                self._cursor_col += 1

    # ------------------------------------------------------------------
    # Cursor movement
    # ------------------------------------------------------------------

    def _move_cursor(self, delta_line: int, delta_col: int) -> None:
        """Move the cursor by ``(delta_line, delta_col)``, wrapping across lines.

        When ``delta_line`` is non-zero the column is preserved (clamped
        to the new line's length) — this gives natural up/down arrow
        behaviour.  When ``delta_col`` is non-zero and the cursor hits
        the start/end of the line it wraps to the previous/next line.
        """
        if delta_line != 0:
            new_line = max(0, min(self._cursor_line + delta_line, len(self._lines) - 1))
            self._cursor_line = new_line
            self._cursor_col = min(self._cursor_col, len(self._lines[new_line]))
            return

        if delta_col < 0:
            if self._cursor_col > 0:
                self._cursor_col += delta_col
            elif self._cursor_line > 0:
                self._cursor_line -= 1
                self._cursor_col = len(self._lines[self._cursor_line])
        elif delta_col > 0:
            line = self._lines[self._cursor_line]
            if self._cursor_col < len(line):
                self._cursor_col += delta_col
            elif self._cursor_line < len(self._lines) - 1:
                self._cursor_line += 1
                self._cursor_col = 0

    def _page_scroll(self, direction: int) -> None:
        """Move the cursor by one page (``direction=-1`` up, ``+1`` down)."""
        page = max(1, self.max_visible_lines - 1)
        new_line = max(0, min(self._cursor_line + direction * page, len(self._lines) - 1))
        self._cursor_line = new_line
        self._cursor_col = min(self._cursor_col, len(self._lines[new_line]))

    # ------------------------------------------------------------------
    # Paste
    # ------------------------------------------------------------------

    def _navigate_history(self, direction: int) -> None:
        """Navigate prompt history. ``direction=-1`` = older, ``1`` = newer."""
        if not self._history:
            return
        if self._history_index == -1:
            # Entering history mode — save current text
            self._saved_text = self.get_text()
            self._history_index = 0
        else:
            self._history_index -= direction  # -1 = go older (higher index)

        if self._history_index < 0:
            # Back to the saved text (exiting history)
            self._history_index = -1
            self.set_text(self._saved_text)
            return
        if self._history_index >= len(self._history):
            self._history_index = len(self._history) - 1

        self.set_text(self._history[self._history_index])

    def _handle_paste(self, text: str, *, record_undo: bool = True) -> None:
        """Insert ``text`` (possibly multi-line) at the cursor."""
        if record_undo:
            self._push_undo()
        self._last_action = None

        # Normalise line endings; tabs → spaces
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        parts = text.split("\n")

        if not parts:
            return

        line = self._lines[self._cursor_line]
        before = line[: self._cursor_col]
        after = line[self._cursor_col :]

        if len(parts) == 1:
            # Simple single-line paste
            self._lines[self._cursor_line] = before + parts[0] + after
            self._cursor_col += len(parts[0])
        else:
            # Multi-line: first chunk goes onto current line, last chunk + tail
            # becomes a new line, middle chunks are new whole lines.
            self._lines[self._cursor_line] = before + parts[0]
            insert_at = self._cursor_line + 1
            for mid_part in parts[1:-1]:
                self._lines.insert(insert_at, mid_part)
                insert_at += 1
            self._lines.insert(insert_at, parts[-1] + after)
            self._cursor_line = insert_at
            self._cursor_col = len(parts[-1])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _fire_change(self) -> None:
        if self.on_change:
            self.on_change(self.get_text())

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """Render the editor to a list of terminal lines.

        Layout:
        1. Apply horizontal padding to compute the usable inner width.
        2. Word-wrap every logical line into visual segments.
        3. Find the visual line that contains the cursor.
        4. Adjust the scroll offset so the cursor stays visible.
        5. Slice ``max_visible_lines`` of visual output.
        6. Render each visible line, injecting the cursor reverse-video
           highlight and ``CURSOR_MARKER`` when focused.
        7. Prepend/append scroll-indicator lines when content is hidden.

        Returns a flat list of strings (one per terminal row).
        """
        pad = " " * self.padding_x
        inner_width = max(1, width - 2 * self.padding_x)

        # ----------------------------------------------------------------
        # 1. Build visual-line index
        #    visual_lines[i] = (logical_line_idx, segment_idx, segment_str)
        # ----------------------------------------------------------------
        visual_lines: list[tuple[int, int, str]] = []
        for log_idx, log_line in enumerate(self._lines):
            segs = _wrap_line(log_line, inner_width)
            for seg_idx, seg in enumerate(segs):
                visual_lines.append((log_idx, seg_idx, seg))

        # ----------------------------------------------------------------
        # 2. Find visual cursor line
        # ----------------------------------------------------------------
        cursor_vis_line = 0
        cursor_vis_col = self._cursor_col  # character index in segment

        # Identify which visual segment the cursor falls in.
        char_offset = 0
        for vis_idx, (log_idx, seg_idx, seg) in enumerate(visual_lines):
            if log_idx != self._cursor_line:
                char_offset = 0
                continue
            seg_len = len(seg)
            if (
                char_offset + seg_len >= self._cursor_col
                or seg_idx == len(_wrap_line(self._lines[log_idx], inner_width)) - 1
            ):
                cursor_vis_line = vis_idx
                cursor_vis_col = self._cursor_col - char_offset
                break
            char_offset += seg_len

        # ----------------------------------------------------------------
        # 3. Adjust scroll so cursor is visible
        # ----------------------------------------------------------------
        if cursor_vis_line < self._scroll_offset:
            self._scroll_offset = cursor_vis_line
        elif cursor_vis_line >= self._scroll_offset + self.max_visible_lines:
            self._scroll_offset = cursor_vis_line - self.max_visible_lines + 1

        # Clamp scroll offset
        max_scroll = max(0, len(visual_lines) - self.max_visible_lines)
        self._scroll_offset = max(0, min(self._scroll_offset, max_scroll))

        # ----------------------------------------------------------------
        # 4. Slice the visible window
        # ----------------------------------------------------------------
        vis_start = self._scroll_offset
        vis_end = vis_start + self.max_visible_lines
        visible_slice = visual_lines[vis_start:vis_end]

        hidden_above = vis_start
        hidden_below = max(0, len(visual_lines) - vis_end)

        # ----------------------------------------------------------------
        # 5. Render each visible line
        # ----------------------------------------------------------------
        output_lines: list[str] = []

        for slice_idx, (_log_idx, _seg_idx, seg) in enumerate(visible_slice):
            abs_vis_idx = vis_start + slice_idx
            is_cursor_line = abs_vis_idx == cursor_vis_line

            if is_cursor_line and self.focused:
                # Clamp cursor_vis_col to segment length
                col = max(0, min(cursor_vis_col, len(seg)))
                before = seg[:col]
                at_cursor = seg[col] if col < len(seg) else " "
                after = seg[col + len(at_cursor) :]
                cursor_char = f"\x1b[7m{at_cursor}\x1b[27m"
                rendered = f"{before}{CURSOR_MARKER}{cursor_char}{after}"
            elif is_cursor_line:
                # Focused=False: still show cursor position but no marker
                col = max(0, min(cursor_vis_col, len(seg)))
                before = seg[:col]
                at_cursor = seg[col] if col < len(seg) else " "
                after = seg[col + len(at_cursor) :]
                cursor_char = f"\x1b[7m{at_cursor}\x1b[27m"
                rendered = f"{before}{cursor_char}{after}"
            else:
                rendered = seg

            # Pad to inner_width (ignore ANSI widths already injected by cursor marker)
            raw_width = visible_width(rendered)
            if raw_width < inner_width:
                rendered += " " * (inner_width - raw_width)

            line_out = f"{pad}{rendered}{pad}"

            if self.border_fn:
                line_out = self.border_fn(line_out)

            output_lines.append(line_out)

        # ----------------------------------------------------------------
        # 6. Scroll indicators
        # ----------------------------------------------------------------
        if hidden_above > 0:
            indicator = truncate_to_width(f"↑ {hidden_above} more", inner_width)
            indicator = indicator.ljust(inner_width)
            top_line = f"{pad}{indicator}{pad}"
            if self.border_fn:
                top_line = self.border_fn(top_line)
            output_lines.insert(0, top_line)

        if hidden_below > 0:
            indicator = truncate_to_width(f"↓ {hidden_below} more", inner_width)
            indicator = indicator.ljust(inner_width)
            bot_line = f"{pad}{indicator}{pad}"
            if self.border_fn:
                bot_line = self.border_fn(bot_line)
            output_lines.append(bot_line)

        return output_lines


__all__ = ["Editor", "EditorState"]
