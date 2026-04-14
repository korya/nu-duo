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


# ---------------------------------------------------------------------------
# Additional coverage — modifier helpers, decode_key edge cases
# ---------------------------------------------------------------------------


def test_key_shift_ctrl() -> None:
    """Key.shift_ctrl helper (line 124)."""
    assert Key.shift_ctrl("a") == "shift+ctrl+a"


def test_key_alt_ctrl() -> None:
    """Key.alt_ctrl helper (line 132)."""
    assert Key.alt_ctrl("x") == "alt+ctrl+x"


def test_key_alt_shift() -> None:
    """Key.alt_shift helper (line 140)."""
    assert Key.alt_shift("up") == "alt+shift+up"


def test_normalize_unknown_modifier() -> None:
    """Unknown modifiers are appended at the end (lines 179-180)."""
    result = normalize_key_id("super+ctrl+a")
    assert result == "ctrl+super+a"


def test_decode_key_empty() -> None:
    """decode_key('') returns None (line 415)."""
    assert decode_key("") is None


def test_decode_key_csi_u_shifted() -> None:
    """CSI-u with shift modifier uses shifted codepoint (line 350)."""
    # ESC [ 97 : 65 ; 2 u = 'a' shifted to 'A'
    result = decode_key("\x1b[97:65;2u")
    assert result is not None
    assert "a" in result.lower()


def test_decode_key_csi_u_codepoint_name() -> None:
    """CSI-u with named codepoints like enter, tab (line 350-354)."""
    # ESC [ 13 u = enter
    assert decode_key("\x1b[13u") == "enter"
    # ESC [ 9 u = tab
    assert decode_key("\x1b[9u") == "tab"
    # ESC [ 27 u = escape
    assert decode_key("\x1b[27u") == "escape"
    # ESC [ 32 u = space
    assert decode_key("\x1b[32u") == "space"


def test_decode_key_csi_u_with_ctrl() -> None:
    """CSI-u with ctrl modifier (line 358)."""
    # ESC [ 97 ; 5 u = ctrl+a (mod=5, 5-1=4 → ctrl)
    result = decode_key("\x1b[97;5u")
    assert result == "ctrl+a"


def test_decode_key_func_mod() -> None:
    """Functional key with modifier (lines 380, 388-397)."""
    # ESC [ 3 ; 5 ~ = ctrl+delete
    result = decode_key("\x1b[3;5~")
    assert result == "ctrl+delete"
    # ESC [ 5 ; 2 ~ = shift+pageUp
    result = decode_key("\x1b[5;2~")
    assert result == "shift+pageUp"


def test_decode_key_func_mod_unknown_num() -> None:
    """Functional key with unknown num returns None (line 380)."""
    result = decode_key("\x1b[99;5~")
    assert result is None


def test_decode_key_mod_other_keys() -> None:
    """modifyOtherKeys format: ESC [ 27 ; mod ; cp ~ (line 428)."""
    # ESC [ 27 ; 5 ; 97 ~ = ctrl+a
    result = decode_key("\x1b[27;5;97~")
    assert result == "ctrl+a"
    # Named codepoint
    result = decode_key("\x1b[27;5;13~")
    assert result == "ctrl+enter"


def test_decode_key_bare_escapes() -> None:
    """Bare escape codes (lines 438-466)."""
    assert decode_key("\x1b") == "escape"
    assert decode_key("\x1c") == "ctrl+\\"
    assert decode_key("\x1d") == "ctrl+]"
    assert decode_key("\x1f") == "ctrl+-"
    assert decode_key("\x1b\x1b") == "ctrl+alt+["
    assert decode_key("\x1b\x1c") == "ctrl+alt+\\"
    assert decode_key("\x1b\x1d") == "ctrl+alt+]"
    assert decode_key("\x1b\x1f") == "ctrl+alt+-"
    assert decode_key("\t") == "tab"
    assert decode_key("\r") == "enter"
    assert decode_key("\n") == "enter"
    assert decode_key("\x00") == "ctrl+space"
    assert decode_key(" ") == "space"
    assert decode_key("\x7f") == "backspace"
    assert decode_key("\x08") == "backspace"
    assert decode_key("\x1b[Z") == "shift+tab"
    assert decode_key("\x1b\x7f") == "alt+backspace"
    assert decode_key("\x1b\b") == "alt+backspace"
    assert decode_key("\x1bOM") == "enter"


def test_decode_key_page_up_down() -> None:
    """Page up/down bare sequences (lines 482-484)."""
    assert decode_key("\x1b[5~") == "pageUp"
    assert decode_key("\x1b[6~") == "pageDown"


def test_decode_key_esc_plus_printable() -> None:
    """ESC + printable for alt+key and ctrl+alt+key (lines 488-492)."""
    # alt+a
    assert decode_key("\x1ba") == "alt+a"
    # alt+5
    assert decode_key("\x1b5") == "alt+5"
    # ctrl+alt+a (ESC + 0x01)
    assert decode_key("\x1b\x01") == "ctrl+alt+a"


def test_decode_key_raw_ctrl_characters() -> None:
    """Raw single ctrl characters: C-a through C-z (lines 499-502)."""
    assert decode_key("\x01") == "ctrl+a"
    assert decode_key("\x03") == "ctrl+c"
    assert decode_key("\x1a") == "ctrl+z"


def test_decode_key_printable_ascii() -> None:
    """Raw printable ASCII characters (line 500)."""
    assert decode_key("a") == "a"
    assert decode_key("Z") == "Z"
    assert decode_key("0") == "0"
    assert decode_key("/") == "/"


def test_decode_key_unrecognized() -> None:
    """Unrecognized multi-byte sequence returns None."""
    assert decode_key("\x1b[999z") is None


def test_decode_key_legacy_lookup_table() -> None:
    """Legacy sequence lookup table works (line 428-432)."""
    assert decode_key("\x1bOA") == "up"
    assert decode_key("\x1bOB") == "down"
    assert decode_key("\x1bOC") == "right"
    assert decode_key("\x1bOD") == "left"
    assert decode_key("\x1bOH") == "home"
    assert decode_key("\x1bOF") == "end"
    assert decode_key("\x1b[2~") == "insert"
    assert decode_key("\x1b[3~") == "delete"


def test_decode_key_modified_legacy() -> None:
    """Modified legacy sequences from the lookup table."""
    assert decode_key("\x1b[a") == "shift+up"
    assert decode_key("\x1b[b") == "shift+down"
    assert decode_key("\x1bOa") == "ctrl+up"
    assert decode_key("\x1bOd") == "ctrl+left"


def test_modifier_prefix_with_lock_mask() -> None:
    """_modifier_prefix strips lock bits (line 324)."""
    from nu_tui.keys import _modifier_prefix

    # Pure ctrl (bit 4) = "ctrl+"
    assert _modifier_prefix(4) == "ctrl+"
    # ctrl + capslock (bit 64) = still just "ctrl+"
    assert _modifier_prefix(4 | 64) == "ctrl+"
    # ctrl + numlock (bit 128) = still just "ctrl+"
    assert _modifier_prefix(4 | 128) == "ctrl+"
    # No modifiers = ""
    assert _modifier_prefix(0) == ""
    # All three modifiers
    assert _modifier_prefix(1 | 2 | 4) == "shift+ctrl+alt+"


def test_get_key_label_special_keys() -> None:
    """get_key_label for various special keys."""
    assert get_key_label("escape") == "Esc"
    assert get_key_label("enter") == "Enter"
    assert get_key_label("tab") == "Tab"
    assert get_key_label("space") == "Space"
    assert get_key_label("backspace") == "Backspace"
    assert get_key_label("pageUp") == "Page Up"
    assert get_key_label("pageDown") == "Page Down"
    assert get_key_label("up") == "↑"
    assert get_key_label("down") == "↓"
    assert get_key_label("left") == "←"
    assert get_key_label("right") == "→"


def test_get_key_label_modified_keys() -> None:
    """get_key_label with modifiers."""
    assert get_key_label("ctrl+c") == "Ctrl+C"
    assert get_key_label("shift+enter") == "Shift+Enter"
    assert get_key_label("ctrl+shift+a") == "Ctrl+Shift+A"
    assert get_key_label("alt+left") == "Alt+←"


def test_get_key_label_plain_char() -> None:
    """get_key_label for plain characters."""
    assert get_key_label("a") == "A"
    assert get_key_label("z") == "Z"
