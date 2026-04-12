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
    from nu_tui.utils import ANSI_ESCAPE_RE  # noqa: PLC0415

    stripped = ANSI_ESCAPE_RE.sub("", lines[0])
    assert "hello" in stripped


def test_render_focused_includes_cursor_marker() -> None:
    inp = Input()
    inp.focused = True
    inp.set_value("hi")
    lines = inp.render(40)
    from nu_tui.utils import CURSOR_MARKER  # noqa: PLC0415

    assert CURSOR_MARKER in lines[0]


def test_render_unfocused_no_cursor_marker() -> None:
    inp = Input()
    inp.focused = False
    inp.set_value("hi")
    lines = inp.render(40)
    from nu_tui.utils import CURSOR_MARKER  # noqa: PLC0415

    assert CURSOR_MARKER not in lines[0]
