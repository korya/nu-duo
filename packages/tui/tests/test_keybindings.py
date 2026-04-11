"""Tests for nu_tui.keybindings.

Ports the documented contract from
``packages/tui/src/keybindings.ts``: definitions, user overrides,
conflict detection, key resolution, and the global ``get_keybindings``
singleton.
"""

from __future__ import annotations

from nu_tui.keybindings import (
    TUI_KEYBINDINGS,
    KeybindingDefinition,
    KeybindingsManager,
    get_keybindings,
    set_keybindings,
)


def _defs() -> dict[str, KeybindingDefinition]:
    return {
        "demo.up": KeybindingDefinition(default_keys="up", description="Move up"),
        "demo.down": KeybindingDefinition(default_keys="down", description="Move down"),
        "demo.copy": KeybindingDefinition(default_keys=["ctrl+c", "ctrl+insert"], description="Copy"),
    }


class TestDefaults:
    def test_default_keys_used_when_no_overrides(self) -> None:
        mgr = KeybindingsManager(_defs())
        assert mgr.get_keys("demo.up") == ["up"]
        assert mgr.get_keys("demo.copy") == ["ctrl+c", "ctrl+insert"]

    def test_definition_lookup(self) -> None:
        mgr = KeybindingsManager(_defs())
        defn = mgr.get_definition("demo.up")
        assert defn.description == "Move up"


class TestUserOverrides:
    def test_user_override_replaces_default(self) -> None:
        mgr = KeybindingsManager(_defs(), user_bindings={"demo.up": "k"})
        assert mgr.get_keys("demo.up") == ["k"]

    def test_user_override_with_list(self) -> None:
        mgr = KeybindingsManager(_defs(), user_bindings={"demo.up": ["k", "ctrl+p"]})
        assert mgr.get_keys("demo.up") == ["k", "ctrl+p"]

    def test_unknown_user_keybinding_ignored(self) -> None:
        mgr = KeybindingsManager(_defs(), user_bindings={"demo.unknown": "x"})
        # Existing bindings unaffected.
        assert mgr.get_keys("demo.up") == ["up"]
        assert mgr.get_conflicts() == []

    def test_set_user_bindings_rebuilds(self) -> None:
        mgr = KeybindingsManager(_defs())
        mgr.set_user_bindings({"demo.up": "ctrl+u"})
        assert mgr.get_keys("demo.up") == ["ctrl+u"]


class TestConflicts:
    def test_conflict_detected_when_same_key_used_for_two_bindings(self) -> None:
        mgr = KeybindingsManager(
            _defs(),
            user_bindings={"demo.up": "ctrl+x", "demo.down": "ctrl+x"},
        )
        conflicts = mgr.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].key == "ctrl+x"
        assert set(conflicts[0].keybindings) == {"demo.up", "demo.down"}

    def test_no_conflict_with_distinct_keys(self) -> None:
        mgr = KeybindingsManager(
            _defs(),
            user_bindings={"demo.up": "k", "demo.down": "j"},
        )
        assert mgr.get_conflicts() == []


class TestMatches:
    def test_matches_returns_true_for_assigned_key(self) -> None:
        mgr = KeybindingsManager(_defs())
        assert mgr.matches("up", "demo.up") is True

    def test_matches_returns_false_for_unassigned_key(self) -> None:
        mgr = KeybindingsManager(_defs())
        assert mgr.matches("down", "demo.up") is False

    def test_matches_any_of_multiple_keys(self) -> None:
        mgr = KeybindingsManager(_defs())
        assert mgr.matches("ctrl+c", "demo.copy") is True
        assert mgr.matches("ctrl+insert", "demo.copy") is True


class TestResolvedBindings:
    def test_resolved_bindings_returns_single_or_list(self) -> None:
        mgr = KeybindingsManager(_defs())
        resolved = mgr.get_resolved_bindings()
        assert resolved["demo.up"] == "up"
        assert resolved["demo.copy"] == ["ctrl+c", "ctrl+insert"]


class TestGlobalSingleton:
    def test_get_keybindings_returns_default_when_unset(self) -> None:
        # Reset and verify the default returns the TUI bindings.
        set_keybindings(KeybindingsManager(TUI_KEYBINDINGS))
        mgr = get_keybindings()
        # tui.editor.cursorUp must exist with "up" as a default.
        assert "up" in mgr.get_keys("tui.editor.cursorUp")

    def test_set_keybindings_replaces_global(self) -> None:
        custom = KeybindingsManager(_defs())
        set_keybindings(custom)
        assert get_keybindings() is custom
        # Restore default for other tests.
        set_keybindings(KeybindingsManager(TUI_KEYBINDINGS))
