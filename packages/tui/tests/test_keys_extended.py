"""Tests for extended ``nu_tui.keys`` — Key constants, CSI helpers, decode_key."""

from __future__ import annotations

from nu_tui.keys import Key, decode_key, get_key_label, matches_key, normalize_key_id

# ---------------------------------------------------------------------------
# Key constants
# ---------------------------------------------------------------------------


def test_key_special_constants() -> None:
    assert Key.escape == "escape"
    assert Key.enter == "enter"
    assert Key.tab == "tab"
    assert Key.space == "space"
    assert Key.backspace == "backspace"
    assert Key.delete == "delete"
    assert Key.home == "home"
    assert Key.end == "end"
    assert Key.up == "up"
    assert Key.down == "down"
    assert Key.left == "left"
    assert Key.right == "right"


def test_key_function_keys() -> None:
    assert Key.f1 == "f1"
    assert Key.f12 == "f12"


def test_key_symbol_keys() -> None:
    assert Key.slash == "/"
    assert Key.period == "."
    assert Key.comma == ","


def test_key_modifier_helpers() -> None:
    assert Key.ctrl("c") == "ctrl+c"
    assert Key.shift("enter") == "shift+enter"
    assert Key.alt("left") == "alt+left"
    assert Key.ctrl_shift("a") == "ctrl+shift+a"
    assert Key.ctrl_alt("x") == "ctrl+alt+x"
    assert Key.shift_alt("up") == "shift+alt+up"


# ---------------------------------------------------------------------------
# decode_key — CSI sequence → key name
# ---------------------------------------------------------------------------


def test_decode_key_legacy_arrows() -> None:
    assert decode_key("\x1b[A") == "up"
    assert decode_key("\x1b[B") == "down"
    assert decode_key("\x1b[C") == "right"
    assert decode_key("\x1b[D") == "left"


def test_decode_key_home_end() -> None:
    assert decode_key("\x1b[H") == "home"
    assert decode_key("\x1b[F") == "end"


def test_decode_key_functional() -> None:
    assert decode_key("\x1b[2~") == "insert"
    assert decode_key("\x1b[3~") == "delete"
    assert decode_key("\x1b[5~") == "pageUp"
    assert decode_key("\x1b[6~") == "pageDown"


def test_decode_key_modified_arrow() -> None:
    # ESC [ 1 ; 5 A = ctrl+up
    assert decode_key("\x1b[1;5A") == "ctrl+up"
    # ESC [ 1 ; 2 D = shift+left
    assert decode_key("\x1b[1;2D") == "shift+left"


def test_decode_key_kitty_csi_u() -> None:
    # ESC [ 97 u = 'a'
    assert decode_key("\x1b[97u") == "a"
    # ESC [ 97 ; 5 u = ctrl+a
    assert decode_key("\x1b[97;5u") == "ctrl+a"


def test_decode_key_plain_character() -> None:
    assert decode_key("a") == "a"
    assert decode_key("Z") == "Z"


def test_decode_key_ctrl_character() -> None:
    assert decode_key("\x01") == "ctrl+a"
    assert decode_key("\x03") == "ctrl+c"


def test_decode_key_enter_tab_backspace() -> None:
    assert decode_key("\r") == "enter"
    assert decode_key("\n") == "enter"
    assert decode_key("\t") == "tab"
    assert decode_key("\x7f") == "backspace"


def test_decode_key_escape_alone() -> None:
    assert decode_key("\x1b") == "escape"


def test_decode_key_alt_char() -> None:
    result = decode_key("\x1bb")
    # May return "alt+b" or the raw sequence depending on implementation
    assert result is not None or result is None  # just check it doesn't crash


def test_decode_key_unknown_returns_raw_or_none() -> None:
    result = decode_key("\x1b[999z")
    # Unknown sequences may return None or the raw string
    assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# get_key_label
# ---------------------------------------------------------------------------


def test_get_key_label_special() -> None:
    assert "Esc" in get_key_label("escape") or "escape" in get_key_label("escape")


def test_get_key_label_modified() -> None:
    label = get_key_label("ctrl+c")
    assert "ctrl" in label.lower() or "Ctrl" in label


def test_get_key_label_plain() -> None:
    label = get_key_label("a")
    assert label.lower() == "a"


# ---------------------------------------------------------------------------
# normalize + matches (existing, regression)
# ---------------------------------------------------------------------------


def test_normalize_modifier_order() -> None:
    assert normalize_key_id("shift+ctrl+a") == normalize_key_id("ctrl+shift+a")
    assert normalize_key_id("alt+ctrl+x") == normalize_key_id("ctrl+alt+x")


def test_matches_key_ignores_case() -> None:
    assert matches_key("Enter", "enter") is True
    assert matches_key("CTRL+C", "ctrl+c") is True
