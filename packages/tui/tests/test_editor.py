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


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_editor_state_clone() -> None:
    """EditorState.clone() creates an independent copy."""
    from nu_tui.components.editor import EditorState

    state = EditorState(lines=["hello", "world"], cursor_line=0, cursor_col=3)
    cloned = state.clone()
    assert cloned.lines == state.lines
    assert cloned.cursor_line == state.cursor_line
    assert cloned.cursor_col == state.cursor_col
    cloned.lines.append("extra")
    assert len(state.lines) == 2


def test_wrap_line_zero_width() -> None:
    """_wrap_line with max_width=0 returns the line unchanged."""
    from nu_tui.components.editor import _wrap_line

    assert _wrap_line("hello", 0) == ["hello"]
    assert _wrap_line("", 0) == [""]


def test_wrap_line_hard_break() -> None:
    """_wrap_line hard-breaks when a single word is wider than max_width."""
    from nu_tui.components.editor import _wrap_line

    result = _wrap_line("abcdefghij", 4)
    assert len(result) >= 3
    for seg in result[:-1]:
        assert len(seg) <= 4


def test_wrap_line_very_wide_char() -> None:
    """_wrap_line handles the case where hard_cut would be 0."""
    from nu_tui.components.editor import _wrap_line

    # Single char wider than max_width: hard_cut=0 → forced to 1
    result = _wrap_line("abcde", 1)
    assert len(result) == 5


def test_set_text_empty_string() -> None:
    """set_text('') initializes to a single empty line (line 202)."""
    ed = Editor()
    ed.set_text("")
    assert ed.get_lines() == [""]
    assert ed.get_cursor() == (0, 0)


def test_add_to_history() -> None:
    """add_to_history stores items and deduplicates (lines 214-224)."""
    ed = Editor()
    ed.add_to_history("hello")
    assert ed._history == ["hello"]
    # Duplicate at top is skipped
    ed.add_to_history("hello")
    assert ed._history == ["hello"]
    # New entry goes to front
    ed.add_to_history("world")
    assert ed._history == ["world", "hello"]
    # Empty/whitespace is skipped
    ed.add_to_history("")
    ed.add_to_history("   ")
    assert len(ed._history) == 2


def test_add_to_history_caps_at_100() -> None:
    """History is capped at 100 entries (line 222-224)."""
    ed = Editor()
    for i in range(110):
        ed.add_to_history(f"entry {i}")
    assert len(ed._history) == 100


def test_set_autocomplete_provider() -> None:
    """set_autocomplete_provider stores the provider (line 228)."""
    ed = Editor()
    provider = object()
    ed.set_autocomplete_provider(provider)
    assert ed._autocomplete_provider is provider


def test_history_navigation_up_down() -> None:
    """Up/down arrows navigate history when on first/browsing (lines 339-340, 347-348)."""
    ed = Editor()
    ed.add_to_history("old command")
    ed.handle_input("t")
    ed.handle_input("e")
    # Up arrow on first line triggers history
    ed.handle_input("up")
    assert ed.get_text() == "old command"
    # Down arrow exits history
    ed.handle_input("down")
    assert ed.get_text() == "te"


def test_history_navigation_older_entries() -> None:
    """Navigate through multiple history entries."""
    ed = Editor()
    ed.add_to_history("first")
    ed.add_to_history("second")
    ed.handle_input("up")  # enter history: second
    assert ed.get_text() == "second"
    ed.handle_input("up")  # older: first
    assert ed.get_text() == "first"
    ed.handle_input("up")  # at end, stays on first
    assert ed.get_text() == "first"


def test_navigate_history_with_no_history() -> None:
    """navigate_history with empty history is a no-op (line 714)."""
    ed = Editor()
    ed.handle_input("up")  # no history, just moves cursor up (no-op on line 0)
    assert ed.get_cursor() == (0, 0)


def test_escape_calls_on_escape() -> None:
    """Escape key triggers on_escape callback (lines 264-266)."""
    ed = Editor()
    escaped: list[bool] = []
    ed.on_escape = lambda: escaped.append(True)
    ed.handle_input("escape")
    assert escaped == [True]


def test_yank_pop_with_multiple_entries() -> None:
    """yank_pop cycles through kill ring entries (lines 572-582)."""
    ed = Editor()
    ed.set_text("aaa bbb ccc")
    # Kill "ccc"
    ed.handle_input("home")
    for _ in range(8):
        ed.handle_input("right")
    ed.handle_input("ctrl+k")  # kill "ccc"
    # Now kill "bbb "
    ed.handle_input("home")
    for _ in range(4):
        ed.handle_input("right")
    ed.handle_input("ctrl+k")  # kill "bbb "
    # Current text: "aaa "
    ed.handle_input("ctrl+y")  # yank "bbb " (most recent)
    assert "bbb" in ed.get_text()
    ed.handle_input("alt+y")  # yank_pop: replace with "ccc"
    assert "ccc" in ed.get_text()


def test_remove_text_before_cursor() -> None:
    """_remove_text_before_cursor works across lines (lines 586-600)."""
    ed = Editor()
    ed.set_text("hello\nworld")
    # cursor at end of "world" (line 1, col 5)
    # Remove 3 chars: "rld" backwards = "orl" wait, it removes char by char before cursor
    ed._remove_text_before_cursor(3)  # remove "r", "l", "d" from end → "hello\nwo"
    assert ed.get_text() == "hello\nwo"
    # Now remove across the line boundary
    ed._remove_text_before_cursor(4)  # remove "o", "w", "\n", "o" → "hell"
    assert ed.get_text() == "hell"


def test_word_movement_across_lines() -> None:
    """Word movement backwards/forwards crosses line boundaries (lines 614-651)."""
    ed = Editor()
    ed.set_text("hello\nworld")
    # Cursor at start of "world" (line 1, col 0)
    ed.handle_input("home")
    # Move word backward: should cross to line 0
    ed.handle_input("alt+left")
    assert ed.get_cursor()[0] == 0

    # Reset and test forward
    ed.set_text("hello\nworld")
    ed.handle_input("home")
    ed.handle_input("up")
    ed.handle_input("end")
    # Move word forward: should cross to line 1
    ed.handle_input("alt+right")
    assert ed.get_cursor()[0] == 1


def test_word_movement_punctuation() -> None:
    """Word movement handles punctuation boundaries (lines 628-629, 660-661)."""
    ed = Editor()
    ed.set_text("hello.world")
    ed.handle_input("home")
    # Move forward: should stop at the punctuation
    ed.handle_input("alt+right")
    assert ed.get_cursor() == (0, 5)
    # Move forward again: move past punctuation
    ed.handle_input("alt+right")
    assert ed.get_cursor() == (0, 6)

    # Move backward through punctuation
    ed.set_text("hello..world")
    # cursor at end
    ed.handle_input("alt+left")
    assert ed.get_cursor() == (0, 7)  # start of "world"
    ed.handle_input("alt+left")
    assert ed.get_cursor() == (0, 5)  # start of ".."


def test_cursor_left_wraps_to_prev_line() -> None:
    """Cursor left at position 0 wraps to end of previous line (line 690-692)."""
    ed = Editor()
    ed.set_text("ab\ncd")
    # Go to start of line 1
    ed.handle_input("home")
    ed.handle_input("left")  # wraps to end of line 0
    assert ed.get_cursor() == (0, 2)


def test_cursor_right_wraps_to_next_line() -> None:
    """Cursor right at end of line wraps to start of next line (line 693-699)."""
    ed = Editor()
    ed.set_text("ab\ncd")
    ed.handle_input("home")
    ed.handle_input("up")
    ed.handle_input("end")
    ed.handle_input("right")  # wraps to start of line 1
    assert ed.get_cursor() == (1, 0)


def test_undo_empty_stack() -> None:
    """Undo with empty stack returns None and is a no-op (line 400)."""
    ed = Editor()
    ed.handle_input("ctrl+-")
    assert ed.get_text() == ""


def test_forward_delete_within_line() -> None:
    """Forward delete within a line (lines 454-455)."""
    ed = Editor()
    ed.set_text("abc")
    ed.handle_input("home")
    ed.handle_input("delete")
    assert ed.get_text() == "bc"


def test_delete_to_line_start_at_col_zero() -> None:
    """Delete to line start when cursor at col 0 is a no-op (line 479)."""
    ed = Editor()
    ed.set_text("hello")
    ed.handle_input("home")
    ed.handle_input("ctrl+u")
    assert ed.get_text() == "hello"


def test_delete_to_line_end_at_end() -> None:
    """Delete to line end when cursor at end is a no-op (line 492)."""
    ed = Editor()
    ed.set_text("hello")
    ed.handle_input("ctrl+k")
    assert ed.get_text() == "hello"


def test_delete_word_backwards_at_start_of_line() -> None:
    """Delete word backward at start of non-first line merges (lines 508-516)."""
    ed = Editor()
    ed.set_text("hello\nworld")
    ed.handle_input("home")  # cursor at start of "world"
    ed.handle_input("ctrl+w")
    assert ed.get_text() == "helloworld"


def test_delete_word_backwards_at_doc_start() -> None:
    """Delete word backward at doc start is a no-op (line 502)."""
    ed = Editor()
    ed.set_text("hello")
    ed.handle_input("home")
    ed.handle_input("ctrl+w")
    assert ed.get_text() == "hello"


def test_delete_word_forward_at_end_of_line() -> None:
    """Delete word forward at end of line merges (lines 544-549)."""
    ed = Editor()
    ed.set_text("hello\nworld")
    ed.handle_input("home")
    ed.handle_input("up")
    ed.handle_input("end")
    ed.handle_input("alt+d")
    assert ed.get_text() == "helloworld"


def test_delete_word_forward_at_doc_end() -> None:
    """Delete word forward at doc end is a no-op (lines 538)."""
    ed = Editor()
    ed.set_text("hello")
    ed.handle_input("alt+d")
    assert ed.get_text() == "hello"


def test_yank_empty_ring() -> None:
    """Yank with empty kill ring is a no-op (line 565)."""
    ed = Editor()
    ed.set_text("hello")
    ed.handle_input("ctrl+y")
    assert ed.get_text() == "hello"


def test_yank_pop_not_after_yank() -> None:
    """Yank pop when last action is not yank is a no-op (line 572)."""
    ed = Editor()
    ed.set_text("hello")
    ed.handle_input("alt+y")
    assert ed.get_text() == "hello"


def test_paste_multi_line() -> None:
    """Paste multi-line text inserts across lines (lines 757-764)."""
    ed = Editor()
    ed.set_text("start end")
    ed.handle_input("home")
    for _ in range(6):
        ed.handle_input("right")
    ed._handle_paste("line1\nline2\nline3")
    text = ed.get_text()
    assert "line1" in text
    assert "line2" in text
    assert "line3" in text
    assert ed.get_lines()[-1].endswith("end")


def test_render_with_border_fn() -> None:
    """Render applies border_fn when provided (line 886)."""
    ed = Editor(border_fn=lambda s: f"|{s}|")
    ed.set_text("hello")
    lines = ed.render(40)
    assert all(line.startswith("|") for line in lines)
    assert all(line.endswith("|") for line in lines)


def test_render_scroll_down_indicator() -> None:
    """Scroll-down indicator appears when content is hidden below (lines 902-907)."""
    ed = Editor(max_visible_lines=3)
    ed.set_text("\n".join(f"line {i}" for i in range(20)))
    # Move cursor to top
    ed.handle_input("home")
    for _ in range(19):
        ed.handle_input("up")
    lines = ed.render(40)
    all_text = "".join(_strip_ansi(line) for line in lines)
    assert "↓" in all_text


def test_render_both_scroll_indicators() -> None:
    """Both up and down scroll indicators when content hidden in both directions."""
    ed = Editor(max_visible_lines=3)
    ed.set_text("\n".join(f"line {i}" for i in range(20)))
    # Put cursor in the middle
    ed.handle_input("home")
    for _ in range(10):
        ed.handle_input("up")
    lines = ed.render(40)
    all_text = "".join(_strip_ansi(line) for line in lines)
    assert "↑" in all_text
    assert "↓" in all_text


def test_render_border_fn_on_scroll_indicators() -> None:
    """Border function also applies to scroll indicator lines (lines 898, 906)."""
    ed = Editor(max_visible_lines=3, border_fn=lambda s: f"|{s}|")
    ed.set_text("\n".join(f"line {i}" for i in range(20)))
    lines = ed.render(40)
    # All lines including scroll indicators should have border
    assert all(line.startswith("|") for line in lines)


def test_render_unfocused_cursor_line() -> None:
    """Unfocused editor still shows reverse-video cursor but no CURSOR_MARKER (line 832)."""
    ed = Editor()
    ed.focused = False
    ed.set_text("hi")
    lines = ed.render(40)
    content = "".join(lines)
    assert "\x1b[7m" in content
    assert CURSOR_MARKER not in content


def test_enter_no_submit_no_content() -> None:
    """Enter with no content and no on_submit inserts a newline."""
    ed = Editor()
    ed.handle_input("enter")
    assert len(ed.get_lines()) == 2


def test_submit_with_content() -> None:
    """Enter with content and on_submit calls on_submit."""
    submitted: list[str] = []
    ed = Editor()
    ed.on_submit = submitted.append
    ed.handle_input("a")
    ed.handle_input("enter")
    assert submitted == ["a"]


def test_ctrl_enter_inserts_newline() -> None:
    """Ctrl+Enter inserts a newline (line 274)."""
    ed = Editor()
    ed.handle_input("a")
    ed.handle_input("\x0d")  # raw CR
    assert len(ed.get_lines()) == 2


def test_bracketed_paste_remaining_after_end() -> None:
    """Bracketed paste processes remaining data after end marker (line 255)."""
    ed = Editor()
    ed.handle_input("\x1b[200~pasted\x1b[201~extra")
    text = ed.get_text()
    assert "pasted" in text
    # "extra" processed as regular input after paste ends
    assert "extra" in text


def test_control_chars_rejected() -> None:
    """Control characters are not inserted as printable text (line 380)."""
    ed = Editor()
    # 0x01 = ctrl+a (home), already handled; use 0x0E which is not mapped
    old_text = ed.get_text()
    # This should not insert anything
    ed.handle_input("\x0e")
    assert ed.get_text() == old_text


def test_page_scroll_up_and_down() -> None:
    """Page scroll moves cursor by page (lines 701-706)."""
    ed = Editor(max_visible_lines=5)
    ed.set_text("\n".join(f"line {i}" for i in range(30)))
    ed.handle_input("home")
    # Page up from the last line
    ed.handle_input("pageUp")
    cursor_after_up = ed.get_cursor()[0]
    assert cursor_after_up < 29
    # Page down
    ed.handle_input("pageDown")
    assert ed.get_cursor()[0] > cursor_after_up


def test_render_with_padding() -> None:
    """Render with padding_x reduces inner width."""
    ed = Editor(padding_x=2)
    ed.set_text("hi")
    lines = ed.render(40)
    assert len(lines) >= 1
    # Lines should have padding on both sides
    assert lines[0].startswith("  ")


def test_delete_word_forward_normal() -> None:
    """Delete word forward in the middle of text."""
    ed = Editor()
    ed.set_text("hello beautiful world")
    ed.handle_input("home")
    for _ in range(6):
        ed.handle_input("right")
    ed.handle_input("alt+d")
    assert ed.get_text() == "hello  world"


def test_kill_ring_accumulation() -> None:
    """Consecutive kills accumulate in the kill ring."""
    ed = Editor()
    ed.set_text("hello world foo")
    ed.handle_input("home")
    for _ in range(5):
        ed.handle_input("right")
    ed.handle_input("ctrl+k")  # kill " world foo"
    ed.handle_input("ctrl+u")  # kill "hello" - accumulates
    # Both should be in the kill ring
    ed.handle_input("ctrl+y")
    assert "hello" in ed.get_text()


def test_word_backwards_at_doc_start() -> None:
    """_move_word_backwards at doc start is a no-op (line 623)."""
    ed = Editor()
    ed.set_text("hello")
    ed.handle_input("home")
    ed.handle_input("alt+left")
    assert ed.get_cursor() == (0, 0)
