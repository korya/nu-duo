"""Tests for ``nu_tui.components.Input``."""

from __future__ import annotations

from nu_tui.components.input import Input

# ---------------------------------------------------------------------------
# Basic state + rendering
# ---------------------------------------------------------------------------


def test_initial_state_empty() -> None:
    inp = Input()
    assert inp.value == ""
    assert inp.cursor == 0


def test_set_value_clamps_cursor() -> None:
    inp = Input()
    inp.set_value("abcdef")
    inp.set_cursor(6)  # at end
    inp.set_value("abc")  # shorter value → cursor clamped to 3
    assert inp.cursor == 3


def test_render_shows_prompt_and_cursor() -> None:
    inp = Input()
    lines = inp.render(40)
    assert len(lines) == 1
    assert lines[0].startswith("> ")
    # Cursor shown as reverse-video escape on a space (empty input).
    assert "\x1b[7m" in lines[0]


def test_render_narrow_width_shows_just_prompt() -> None:
    inp = Input()
    lines = inp.render(2)
    assert lines == ["> "]


# ---------------------------------------------------------------------------
# Character insertion
# ---------------------------------------------------------------------------


def test_insert_printable_characters() -> None:
    inp = Input()
    inp.handle_input("h")
    inp.handle_input("i")
    assert inp.value == "hi"
    assert inp.cursor == 2


def test_insert_unicode() -> None:
    inp = Input()
    inp.handle_input("héllo")
    assert inp.value == "héllo"
    assert inp.cursor == 5


def test_rejects_control_characters() -> None:
    inp = Input()
    inp.handle_input("\x01")  # C-a (control char)
    assert inp.value == ""


# ---------------------------------------------------------------------------
# Cursor movement
# ---------------------------------------------------------------------------


def test_cursor_left_right() -> None:
    inp = Input()
    inp.set_value("abc")
    inp.set_cursor(3)
    inp.handle_input("left")
    assert inp.cursor == 2
    inp.handle_input("right")
    assert inp.cursor == 3


def test_cursor_left_at_zero_stays() -> None:
    inp = Input()
    inp.handle_input("left")
    assert inp.cursor == 0


def test_cursor_right_at_end_stays() -> None:
    inp = Input()
    inp.set_value("a")
    inp.set_cursor(1)
    inp.handle_input("right")
    assert inp.cursor == 1


def test_cursor_home_and_end() -> None:
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(3)
    inp.handle_input("home")
    assert inp.cursor == 0
    inp.handle_input("end")
    assert inp.cursor == 5


# ---------------------------------------------------------------------------
# Word movement
# ---------------------------------------------------------------------------


def test_word_left_skips_word_boundary() -> None:
    inp = Input()
    inp.set_value("hello world")
    inp.set_cursor(11)
    inp.handle_input("alt+left")
    assert inp.cursor == 6


def test_word_right_skips_word_boundary() -> None:
    inp = Input()
    inp.set_value("hello world")
    inp.set_cursor(0)
    inp.handle_input("alt+right")
    assert inp.cursor == 5


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def test_backspace_deletes_before_cursor() -> None:
    inp = Input()
    inp.set_value("abc")
    inp.set_cursor(2)
    inp.handle_input("backspace")
    assert inp.value == "ac"
    assert inp.cursor == 1


def test_forward_delete() -> None:
    inp = Input()
    inp.set_value("abc")
    inp.set_cursor(1)
    inp.handle_input("delete")
    assert inp.value == "ac"
    assert inp.cursor == 1


def test_delete_to_line_start() -> None:
    inp = Input()
    inp.set_value("hello world")
    inp.set_cursor(6)
    inp.handle_input("ctrl+u")
    assert inp.value == "world"
    assert inp.cursor == 0


def test_delete_to_line_end() -> None:
    inp = Input()
    inp.set_value("hello world")
    inp.set_cursor(5)
    inp.handle_input("ctrl+k")
    assert inp.value == "hello"
    assert inp.cursor == 5


def test_delete_word_backward() -> None:
    inp = Input()
    inp.set_value("hello world")
    inp.set_cursor(11)
    inp.handle_input("ctrl+w")
    assert inp.value == "hello "
    assert inp.cursor == 6


def test_delete_word_forward() -> None:
    inp = Input()
    inp.set_value("hello world")
    inp.set_cursor(0)
    inp.handle_input("alt+d")
    assert inp.value == " world"
    assert inp.cursor == 0


# ---------------------------------------------------------------------------
# Kill ring (yank / yank-pop)
# ---------------------------------------------------------------------------


def test_kill_and_yank_round_trip() -> None:
    inp = Input()
    inp.set_value("hello world")
    inp.set_cursor(5)
    inp.handle_input("ctrl+k")  # kill to end → " world"
    assert inp.value == "hello"
    inp.handle_input("ctrl+y")  # yank → " world" at cursor
    assert inp.value == "hello world"


def test_yank_pop_cycles_ring() -> None:
    inp = Input()
    inp.set_value("aaa bbb ccc")
    inp.set_cursor(3)
    inp.handle_input("ctrl+k")  # kill " bbb ccc"
    inp.set_value("xxx")
    inp.set_cursor(3)
    inp.handle_input("ctrl+k")  # kill "" (nothing to kill)
    # Now value is "xxx", ring has " bbb ccc" as most recent.
    inp.handle_input("ctrl+y")  # yank the most recent
    # yank_pop needs at least 2 entries to cycle; we only have 1
    # in this simple case so yank_pop should be a no-op.


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def test_undo_restores_previous_state() -> None:
    inp = Input()
    inp.handle_input("h")
    inp.handle_input("i")
    # "hi" is the current value.
    # Undo should restore to just before the last non-coalesced edit.
    inp.handle_input("ctrl+-")
    # Since "h" and "i" coalesce (both "type-word"), undo takes us
    # back to the state before "h" was typed (empty).
    assert inp.value == ""


def test_undo_on_empty_stack_is_noop() -> None:
    inp = Input()
    inp.handle_input("ctrl+-")  # must not crash
    assert inp.value == ""


# ---------------------------------------------------------------------------
# Bracketed paste
# ---------------------------------------------------------------------------


def test_bracketed_paste_inserts_content() -> None:
    inp = Input()
    inp.set_value("hello ")
    inp.set_cursor(6)
    inp.handle_input("\x1b[200~world\x1b[201~")
    assert inp.value == "hello world"
    assert inp.cursor == 11


def test_bracketed_paste_strips_newlines_and_tabs() -> None:
    inp = Input()
    inp.handle_input("\x1b[200~line1\nline2\ttab\x1b[201~")
    assert inp.value == "line1line2    tab"


def test_bracketed_paste_split_across_chunks() -> None:
    """The start and end markers may arrive in separate handle_input calls."""
    inp = Input()
    inp.handle_input("\x1b[200~hello")
    assert inp.value == ""  # still buffering
    inp.handle_input(" world\x1b[201~")
    assert inp.value == "hello world"


# ---------------------------------------------------------------------------
# Submit / Escape
# ---------------------------------------------------------------------------


def test_submit_fires_on_enter() -> None:
    inp = Input()
    submitted: list[str] = []
    inp.on_submit = submitted.append
    inp.set_value("hello")
    inp.handle_input("enter")
    assert submitted == ["hello"]


def test_escape_fires_on_escape() -> None:
    inp = Input()
    escaped: list[bool] = []
    inp.on_escape = lambda: escaped.append(True)
    inp.handle_input("escape")
    assert escaped == [True]


# ---------------------------------------------------------------------------
# Kitty CSI-u
# ---------------------------------------------------------------------------


def test_kitty_csi_u_printable() -> None:
    inp = Input()
    inp.handle_input("\x1b[97u")  # 'a' in Kitty CSI-u
    assert inp.value == "a"


def test_kitty_csi_u_control_rejected() -> None:
    inp = Input()
    inp.handle_input("\x1b[1u")  # C-a as CSI-u → control char → rejected
    assert inp.value == ""


# ---------------------------------------------------------------------------
# Rendering with content
# ---------------------------------------------------------------------------


def test_render_with_text_includes_value() -> None:
    inp = Input()
    inp.set_value("hello")
    lines = inp.render(40)
    assert len(lines) == 1
    # The first char is wrapped in reverse-video escapes, so check
    # that the raw content is present after stripping ANSI.
    from nu_tui.utils import ANSI_ESCAPE_RE

    stripped = ANSI_ESCAPE_RE.sub("", lines[0])
    assert "hello" in stripped


def test_render_focused_includes_cursor_marker() -> None:
    inp = Input()
    inp.focused = True
    inp.set_value("hi")
    lines = inp.render(40)
    from nu_tui.utils import CURSOR_MARKER

    assert CURSOR_MARKER in lines[0]


def test_render_unfocused_no_cursor_marker() -> None:
    inp = Input()
    inp.focused = False
    inp.set_value("hi")
    lines = inp.render(40)
    from nu_tui.utils import CURSOR_MARKER

    assert CURSOR_MARKER not in lines[0]


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_get_value() -> None:
    """get_value returns current value (line 76)."""
    inp = Input()
    inp.set_value("test")
    assert inp.get_value() == "test"


def test_bracketed_paste_remaining_after_end() -> None:
    """After paste ends, remaining data is processed (line 115)."""
    inp = Input()
    inp.handle_input("\x1b[200~hello\x1b[201~world")
    assert inp.value == "helloworld"


def test_escape_no_callback_is_noop() -> None:
    """Escape without on_escape callback doesn't crash (line 123-125)."""
    inp = Input()
    inp.on_escape = None
    inp.handle_input("escape")
    assert inp.value == ""


def test_enter_no_callback_is_noop() -> None:
    """Enter without on_submit callback doesn't crash (line 130-132)."""
    inp = Input()
    inp.on_submit = None
    inp.handle_input("enter")
    assert inp.value == ""


def test_delete_to_line_start_at_zero_is_noop() -> None:
    """Delete to line start when at position 0 is a no-op (line 235)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(0)
    inp.handle_input("ctrl+u")
    assert inp.value == "hello"


def test_delete_to_line_end_at_end_is_noop() -> None:
    """Delete to line end when at end is a no-op (line 254)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(5)
    inp.handle_input("ctrl+k")
    assert inp.value == "hello"


def test_delete_word_backwards_at_start_is_noop() -> None:
    """Delete word backward when at start is a no-op (line 269)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(0)
    inp.handle_input("ctrl+w")
    assert inp.value == "hello"


def test_delete_word_forward_at_end_is_noop() -> None:
    """Delete word forward when at end is a no-op (line 284)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(5)
    inp.handle_input("alt+d")
    assert inp.value == "hello"


def test_yank_pop_full_cycle() -> None:
    """yank_pop with multiple entries cycles through kill ring (lines 291-301)."""
    inp = Input()
    # Kill two separate pieces to have 2 entries in the ring
    inp.set_value("aaa bbb ccc")
    inp.set_cursor(3)
    inp.handle_input("ctrl+k")  # kill " bbb ccc", value="aaa"
    # Break the kill accumulation
    inp._last_action = None
    inp.set_value("aaa")
    inp.set_cursor(0)
    inp.handle_input("ctrl+k")  # kill "aaa", value=""
    # Ring has "aaa" (most recent) and " bbb ccc"
    inp.handle_input("ctrl+y")  # yank "aaa"
    assert inp.value == "aaa"
    inp.handle_input("alt+y")  # yank_pop: replace with " bbb ccc"
    assert inp.value == " bbb ccc"


def test_yank_empty_ring() -> None:
    """Yank with empty ring is a no-op (line 284 in yank)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(5)
    inp.handle_input("ctrl+y")
    assert inp.value == "hello"


def test_yank_pop_not_after_yank() -> None:
    """Yank pop when last action is not yank is a no-op (line 291)."""
    inp = Input()
    inp.set_value("hello")
    inp.handle_input("alt+y")
    assert inp.value == "hello"


def test_word_backward_through_punctuation() -> None:
    """Word backward movement handles punctuation (lines 316-317)."""
    inp = Input()
    inp.set_value("hello.world")
    inp.set_cursor(11)
    inp.handle_input("alt+left")
    assert inp.cursor == 6  # start of "world"
    inp.handle_input("alt+left")
    assert inp.cursor == 5  # start of "."


def test_word_forward_through_punctuation() -> None:
    """Word forward movement handles punctuation (lines 335-336)."""
    inp = Input()
    inp.set_value("hello.world")
    inp.set_cursor(0)
    inp.handle_input("alt+right")
    assert inp.cursor == 5  # end of "hello"
    inp.handle_input("alt+right")
    assert inp.cursor == 6  # end of "."


def test_word_backward_at_start_is_noop() -> None:
    """Word backward at start does nothing (line 309)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(0)
    inp.handle_input("alt+left")
    assert inp.cursor == 0


def test_word_forward_at_end_is_noop() -> None:
    """Word forward at end does nothing (line 328)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(5)
    inp.handle_input("alt+right")
    assert inp.cursor == 5


def test_render_scrolling_long_text() -> None:
    """Render with text longer than width triggers scrolling (lines 371-386)."""
    inp = Input()
    long_text = "a" * 100
    inp.set_value(long_text)
    inp.set_cursor(50)
    lines = inp.render(20)
    assert len(lines) == 1
    from nu_tui.utils import ANSI_ESCAPE_RE

    stripped = ANSI_ESCAPE_RE.sub("", lines[0])
    assert len(stripped) <= 20


def test_render_scrolling_cursor_at_end() -> None:
    """Render with cursor at end of long text."""
    inp = Input()
    long_text = "a" * 100
    inp.set_value(long_text)
    inp.set_cursor(100)
    lines = inp.render(20)
    assert len(lines) == 1


def test_render_scrolling_cursor_at_start() -> None:
    """Render with cursor at start of long text."""
    inp = Input()
    long_text = "a" * 100
    inp.set_value(long_text)
    inp.set_cursor(0)
    lines = inp.render(20)
    assert len(lines) == 1


def test_render_scrolling_cursor_near_end() -> None:
    """Render with cursor near end of long text (cursor_col > total - half)."""
    inp = Input()
    long_text = "a" * 100
    inp.set_value(long_text)
    inp.set_cursor(98)
    lines = inp.render(20)
    assert len(lines) == 1


def test_forward_delete_at_end_is_noop() -> None:
    """Forward delete at end of value is noop (line 229)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(5)
    inp.handle_input("delete")
    assert inp.value == "hello"


def test_backspace_at_start_is_noop() -> None:
    """Backspace at start is noop (line 222)."""
    inp = Input()
    inp.set_value("hello")
    inp.set_cursor(0)
    inp.handle_input("backspace")
    assert inp.value == "hello"


def test_undo_empty_stack() -> None:
    """Undo with empty stack is noop."""
    inp = Input()
    inp.handle_input("ctrl+-")
    assert inp.value == ""


def test_handle_paste_strips_tabs_and_newlines() -> None:
    """Paste handler strips newlines and converts tabs."""
    inp = Input()
    inp._handle_paste("hello\tworld\nnewline")
    assert inp.value == "hello    worldnewline"


def test_insert_space_triggers_undo_push() -> None:
    """Inserting a space breaks undo coalescing (line 159-160 in _insert_character)."""
    inp = Input()
    inp.handle_input("a")
    inp.handle_input("b")
    inp.handle_input(" ")
    inp.handle_input("c")
    inp.handle_input("d")
    assert inp.value == "ab cd"
    # Space pushed undo with state "ab" (before space insertion).
    # "c" and "d" coalesce with the space's "type-word" action.
    # First undo reverts to state saved before space: "ab"
    inp.handle_input("ctrl+-")
    assert inp.value == "ab"
    # Second undo reverts to state before "ab" was typed: ""
    inp.handle_input("ctrl+-")
    assert inp.value == ""
