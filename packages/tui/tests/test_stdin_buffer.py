"""Tests for ``nu_tui.stdin_buffer``."""

from __future__ import annotations

from nu_tui.stdin_buffer import StdinBuffer


def test_single_printable_character() -> None:
    buf = StdinBuffer()
    events = buf.feed("a")
    assert len(events) == 1
    assert events[0].key == "a"


def test_enter_key() -> None:
    buf = StdinBuffer()
    events = buf.feed("\r")
    assert len(events) == 1
    assert events[0].key == "enter"


def test_tab_key() -> None:
    buf = StdinBuffer()
    events = buf.feed("\t")
    assert len(events) == 1
    assert events[0].key == "tab"


def test_backspace_key() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x7f")
    assert len(events) == 1
    assert events[0].key == "backspace"


def test_ctrl_c() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x03")
    assert len(events) == 1
    assert events[0].key == "ctrl+c"


def test_arrow_keys() -> None:
    buf = StdinBuffer()
    for seq, expected in [("\x1b[A", "up"), ("\x1b[B", "down"), ("\x1b[C", "right"), ("\x1b[D", "left")]:
        events = buf.feed(seq)
        assert len(events) == 1
        assert events[0].key == expected


def test_home_end() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1b[H")
    assert events[0].key == "home"
    events = buf.feed("\x1b[F")
    assert events[0].key == "end"


def test_delete_insert() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1b[3~")
    assert events[0].key == "delete"
    events = buf.feed("\x1b[2~")
    assert events[0].key == "insert"


def test_page_up_down() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1b[5~")
    assert events[0].key == "pageUp"
    events = buf.feed("\x1b[6~")
    assert events[0].key == "pageDown"


def test_alt_character() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1bb")
    assert len(events) == 1
    assert events[0].key == "alt+b"


def test_escape_flush() -> None:
    buf = StdinBuffer()
    buf.feed("\x1b")
    # Buffer holds ESC, waiting for more data
    events = buf.flush()
    assert len(events) == 1
    assert events[0].key == "escape"


def test_bracketed_paste() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1b[200~hello world\x1b[201~")
    assert len(events) == 1
    assert events[0].is_paste is True
    assert events[0].key == "hello world"


def test_bracketed_paste_strips_markers() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1b[200~pasted text\x1b[201~")
    assert events[0].raw == "pasted text"


def test_kitty_csi_u() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1b[97u")
    assert len(events) == 1
    assert events[0].key == "a"


def test_kitty_csi_u_with_shift() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1b[97;2u")  # shift modifier = 2
    assert len(events) == 1
    assert events[0].key == "shift+a"


def test_modified_arrow_key() -> None:
    buf = StdinBuffer()
    events = buf.feed("\x1b[1;5A")  # ctrl+up
    assert len(events) == 1
    assert events[0].key == "ctrl+up"
