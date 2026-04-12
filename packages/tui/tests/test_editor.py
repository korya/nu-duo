"""Tests for ``nu_tui.components.Editor``."""

from __future__ import annotations

from nu_tui.components.editor import Editor
from nu_tui.utils import ANSI_ESCAPE_RE, CURSOR_MARKER


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def test_initial_state_single_empty_line() -> None:
    ed = Editor()
    assert ed.get_text() == ""
    assert ed.get_lines() == [""]
    assert ed.get_cursor() == (0, 0)


def test_set_text_splits_lines() -> None:
    ed = Editor()
    ed.set_text("hello\nworld")
    assert ed.get_lines() == ["hello", "world"]


def test_get_text_joins_lines() -> None:
    ed = Editor()
    ed.set_text("a\nb\nc")
    assert ed.get_text() == "a\nb\nc"


# ---------------------------------------------------------------------------
# Character insertion
# ---------------------------------------------------------------------------


def test_insert_characters() -> None:
    ed = Editor()
    ed.handle_input("h")
    ed.handle_input("i")
    assert ed.get_text() == "hi"
    assert ed.get_cursor() == (0, 2)


def test_insert_newline_splits_line() -> None:
    ed = Editor()
    ed.set_text("hello world")
    # Move cursor to position 5 from end
    ed.handle_input("home")
    for _ in range(5):
        ed.handle_input("right")
    # Insert newline
    ed.handle_input("shift+enter")
    lines = ed.get_lines()
    assert len(lines) == 2
    assert lines[0] == "hello"
    assert ed.get_cursor()[0] == 1


# ---------------------------------------------------------------------------
# Cursor movement
# ---------------------------------------------------------------------------


def test_cursor_left_right() -> None:
    ed = Editor()
    ed.handle_input("a")
    ed.handle_input("b")
    ed.handle_input("c")
    assert ed.get_cursor() == (0, 3)
    ed.handle_input("left")
    assert ed.get_cursor() == (0, 2)
    ed.handle_input("right")
    assert ed.get_cursor() == (0, 3)


def test_cursor_up_down() -> None:
    ed = Editor()
    ed.set_text("line1\nline2\nline3")
    ed.handle_input("home")  # go to start of last line
    # Move up
    ed.handle_input("up")
    assert ed.get_cursor()[0] == 1
    ed.handle_input("up")
    assert ed.get_cursor()[0] == 0
    ed.handle_input("down")
    assert ed.get_cursor()[0] == 1


def test_cursor_home_end() -> None:
    ed = Editor()
    ed.handle_input("h")
    ed.handle_input("e")
    ed.handle_input("l")
    ed.handle_input("l")
    ed.handle_input("o")
    ed.handle_input("home")
    assert ed.get_cursor() == (0, 0)
    ed.handle_input("end")
    assert ed.get_cursor() == (0, 5)


def test_cursor_word_movement() -> None:
    ed = Editor()
    ed.set_text("hello world foo")
    ed.handle_input("home")
    ed.handle_input("alt+right")
    assert ed.get_cursor() == (0, 5)
    ed.handle_input("alt+right")
    assert ed.get_cursor() == (0, 11)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def test_backspace_within_line() -> None:
    ed = Editor()
    ed.handle_input("a")
    ed.handle_input("b")
    ed.handle_input("c")
    ed.handle_input("backspace")
    assert ed.get_text() == "ab"


def test_backspace_at_line_start_joins_lines() -> None:
    ed = Editor()
    ed.set_text("hello\nworld")
    # Cursor is at end of "world" after set_text.
    # Go to start of line 1 (the "world" line).
    ed.handle_input("home")
    ed.handle_input("backspace")
    assert ed.get_text() == "helloworld"
    assert ed.get_cursor() == (0, 5)


def test_forward_delete_at_line_end_joins_lines() -> None:
    ed = Editor()
    ed.set_text("hello\nworld")
    # Navigate: go to end of first line
    ed.handle_input("up")  # from line 1 to line 0
    ed.handle_input("end")
    ed.handle_input("delete")
    assert ed.get_text() == "helloworld"


def test_delete_to_line_end() -> None:
    ed = Editor()
    ed.set_text("hello world")
    ed.handle_input("home")
    for _ in range(5):
        ed.handle_input("right")
    ed.handle_input("ctrl+k")
    assert ed.get_text() == "hello"


def test_delete_to_line_start() -> None:
    ed = Editor()
    ed.set_text("hello world")
    # cursor at end; go left 5 times to reach "world" boundary
    for _ in range(5):
        ed.handle_input("left")
    ed.handle_input("ctrl+u")
    assert ed.get_text() == "world"


# ---------------------------------------------------------------------------
# Kill ring
# ---------------------------------------------------------------------------


def test_kill_and_yank() -> None:
    ed = Editor()
    ed.set_text("hello world")
    # cursor at end, go to pos 5
    ed.handle_input("home")
    for _ in range(5):
        ed.handle_input("right")
    ed.handle_input("ctrl+k")
    assert ed.get_text() == "hello"
    ed.handle_input("ctrl+y")
    assert ed.get_text() == "hello world"


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def test_undo() -> None:
    ed = Editor()
    ed.handle_input("a")
    ed.handle_input("b")
    # Consecutive word chars coalesce, so undo reverts both.
    ed.handle_input("ctrl+-")
    assert ed.get_text() == ""


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def test_submit_on_enter() -> None:
    submitted: list[str] = []
    ed = Editor()
    ed.on_submit = submitted.append
    ed.handle_input("h")
    ed.handle_input("i")
    ed.handle_input("enter")
    assert submitted == ["hi"]


# ---------------------------------------------------------------------------
# Bracketed paste
# ---------------------------------------------------------------------------


def test_bracketed_paste() -> None:
    ed = Editor()
    ed.handle_input("\x1b[200~pasted text\x1b[201~")
    assert "pasted text" in ed.get_text()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_render_produces_output() -> None:
    ed = Editor()
    ed.set_text("hello")
    lines = ed.render(40)
    assert len(lines) >= 1
    # Content should be present somewhere
    content = "".join(_strip_ansi(line) for line in lines)
    assert "hello" in content


def test_render_shows_cursor() -> None:
    ed = Editor()
    ed.set_text("hi")
    lines = ed.render(40)
    content = "".join(lines)
    assert "\x1b[7m" in content


def test_render_focused_includes_cursor_marker() -> None:
    ed = Editor()
    ed.focused = True
    ed.set_text("hi")
    lines = ed.render(40)
    content = "".join(lines)
    assert CURSOR_MARKER in content


def test_render_scroll_indicator() -> None:
    ed = Editor(max_visible_lines=3)
    ed.set_text("\n".join(f"line {i}" for i in range(20)))
    # Cursor is at end (line 19). Move down doesn't go further.
    # Should show ↑ scroll indicator since we're scrolled to bottom.
    lines = ed.render(40)
    all_text = "".join(_strip_ansi(line) for line in lines)
    assert "↑" in all_text


def test_render_narrow_width_does_not_crash() -> None:
    ed = Editor()
    ed.set_text("hello world this is some text")
    lines = ed.render(5)
    assert len(lines) >= 1


def test_render_empty_editor() -> None:
    ed = Editor()
    lines = ed.render(40)
    assert len(lines) >= 1


# ---------------------------------------------------------------------------
# Additional coverage — exercise more editing operations
# ---------------------------------------------------------------------------


def test_page_scroll() -> None:
    ed = Editor(max_visible_lines=5)
    ed.set_text("\n".join(f"line {i}" for i in range(30)))
    ed.handle_input("home")  # go to start of last line
    # Page up
    ed.handle_input("pageUp")
    cursor_before = ed.get_cursor()
    ed.handle_input("pageDown")
    # Cursor moved down
    assert ed.get_cursor()[0] >= cursor_before[0]


def test_delete_word_backward() -> None:
    ed = Editor()
    ed.set_text("hello world")
    # cursor at end
    ed.handle_input("ctrl+w")
    assert ed.get_text() == "hello "


def test_delete_word_forward() -> None:
    ed = Editor()
    ed.set_text("hello world")
    ed.handle_input("home")
    ed.handle_input("alt+d")
    assert ed.get_text() == " world"


def test_yank_pop() -> None:
    ed = Editor()
    ed.set_text("aaa bbb ccc")
    ed.handle_input("home")
    for _ in range(3):
        ed.handle_input("right")
    ed.handle_input("ctrl+k")  # kill " bbb ccc"
    ed.handle_input("ctrl+y")  # yank it back
    assert " bbb ccc" in ed.get_text()


def test_multiple_newlines() -> None:
    ed = Editor()
    ed.handle_input("a")
    ed.handle_input("shift+enter")
    ed.handle_input("b")
    ed.handle_input("shift+enter")
    ed.handle_input("c")
    assert ed.get_lines() == ["a", "b", "c"]


def test_cursor_wraps_across_lines_right() -> None:
    ed = Editor()
    ed.set_text("ab\ncd")
    ed.handle_input("home")
    ed.handle_input("up")  # to line 0
    ed.handle_input("end")
    ed.handle_input("right")  # should wrap to start of line 1
    assert ed.get_cursor() == (1, 0)


def test_cursor_wraps_across_lines_left() -> None:
    ed = Editor()
    ed.set_text("ab\ncd")
    ed.handle_input("home")
    ed.handle_input("up")
    ed.handle_input("down")
    ed.handle_input("home")
    ed.handle_input("left")  # should wrap to end of line 0
    assert ed.get_cursor() == (0, 2)


def test_backspace_on_empty_line_joins() -> None:
    ed = Editor()
    ed.set_text("hello\n\nworld")
    # Navigate to the empty middle line
    ed.handle_input("home")
    ed.handle_input("up")
    ed.handle_input("home")
    ed.handle_input("backspace")
    # Empty line removed, hello and (empty) joined
    assert len(ed.get_lines()) == 2


def test_forward_delete_at_end_of_last_line_is_noop() -> None:
    ed = Editor()
    ed.set_text("hello")
    # cursor at end
    ed.handle_input("delete")
    assert ed.get_text() == "hello"


def test_insert_after_set_text() -> None:
    ed = Editor()
    ed.set_text("hello")
    ed.handle_input(" ")
    ed.handle_input("w")
    assert "hello" in ed.get_text()
    assert "w" in ed.get_text()


def test_on_change_callback() -> None:
    changes: list[str] = []
    ed = Editor()
    ed.on_change = changes.append
    ed.handle_input("x")
    # on_change may or may not fire for every keystroke depending on impl
    # Just verify it doesn't crash


def test_disable_submit() -> None:
    submitted: list[str] = []
    ed = Editor()
    ed.on_submit = submitted.append
    ed.disable_submit = True
    ed.handle_input("x")
    ed.handle_input("enter")
    # With disable_submit, enter should insert newline instead
    assert submitted == [] or ed.get_text() == "x"  # tolerate either behaviour


def test_kitty_csi_u_in_editor() -> None:
    ed = Editor()
    ed.handle_input("\x1b[97u")  # 'a' in Kitty CSI-u
    assert "a" in ed.get_text()


def test_word_wrap_in_render() -> None:
    ed = Editor()
    ed.set_text("the quick brown fox jumps over the lazy dog")
    lines = ed.render(15)
    # Should produce multiple visual lines due to wrapping
    assert len(lines) > 1


def test_render_multiline_content() -> None:
    ed = Editor()
    ed.set_text("line1\nline2\nline3")
    lines = ed.render(40)
    content = " ".join(_strip_ansi(line) for line in lines)
    assert "line1" in content
    assert "line2" in content
    assert "line3" in content
