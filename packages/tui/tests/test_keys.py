"""Tests for pi_tui.keys.

The Python port exposes a much smaller surface than the TS original
because Textual already parses raw stdin into normalized key strings.
We only need:

* :data:`KeyId` — a string type alias for keybinding identifiers.
* :func:`matches_key` — case-insensitive comparison with modifier order
  normalization (so ``"shift+ctrl+a"`` matches ``"ctrl+shift+a"``).
* :class:`Key` — a builder helper exposing constants and modifier helpers.
"""

from __future__ import annotations

from pi_tui.keys import Key, matches_key, normalize_key_id


class TestKeyConstants:
    def test_basic_special_keys(self) -> None:
        assert Key.escape == "escape"
        assert Key.enter == "enter"
        assert Key.tab == "tab"
        assert Key.space == "space"
        assert Key.backspace == "backspace"

    def test_function_keys(self) -> None:
        assert Key.f1 == "f1"
        assert Key.f12 == "f12"

    def test_arrow_keys(self) -> None:
        assert Key.up == "up"
        assert Key.down == "down"
        assert Key.left == "left"
        assert Key.right == "right"

    def test_symbol_keys(self) -> None:
        assert Key.backtick == "`"
        assert Key.hyphen == "-"
        assert Key.leftbracket == "["
        assert Key.semicolon == ";"


class TestModifierHelpers:
    def test_ctrl(self) -> None:
        assert Key.ctrl("c") == "ctrl+c"

    def test_shift(self) -> None:
        assert Key.shift("a") == "shift+a"

    def test_alt(self) -> None:
        assert Key.alt("x") == "alt+x"

    def test_ctrl_shift(self) -> None:
        assert Key.ctrl_shift("p") == "ctrl+shift+p"

    def test_ctrl_alt(self) -> None:
        assert Key.ctrl_alt("x") == "ctrl+alt+x"

    def test_ctrl_shift_alt(self) -> None:
        assert Key.ctrl_shift_alt("z") == "ctrl+shift+alt+z"


class TestNormalizeKeyId:
    def test_lowercased(self) -> None:
        assert normalize_key_id("CTRL+C") == "ctrl+c"

    def test_modifier_order_canonical(self) -> None:
        # Modifiers should be sorted into canonical order: ctrl, shift, alt
        assert normalize_key_id("alt+shift+ctrl+x") == "ctrl+shift+alt+x"

    def test_no_modifiers_passthrough(self) -> None:
        assert normalize_key_id("enter") == "enter"

    def test_single_modifier_passthrough(self) -> None:
        assert normalize_key_id("ctrl+a") == "ctrl+a"

    def test_strips_whitespace(self) -> None:
        assert normalize_key_id(" ctrl + a ") == "ctrl+a"


class TestMatchesKey:
    def test_exact_match(self) -> None:
        assert matches_key("ctrl+c", "ctrl+c") is True

    def test_case_insensitive(self) -> None:
        assert matches_key("CTRL+C", "ctrl+c") is True

    def test_modifier_order_irrelevant(self) -> None:
        assert matches_key("shift+ctrl+a", "ctrl+shift+a") is True

    def test_different_keys_dont_match(self) -> None:
        assert matches_key("ctrl+a", "ctrl+b") is False

    def test_modified_does_not_match_unmodified(self) -> None:
        assert matches_key("ctrl+a", "a") is False

    def test_special_keys(self) -> None:
        assert matches_key("escape", Key.escape) is True
        assert matches_key("enter", Key.enter) is True
