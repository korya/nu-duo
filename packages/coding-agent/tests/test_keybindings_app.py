"""Tests for ``nu_coding_agent.core.keybindings``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nu_coding_agent.core.keybindings import (
    KEYBINDINGS,
    KeybindingsManager,
    migrate_keybindings_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_catalog_includes_app_and_tui_bindings() -> None:
    assert "app.interrupt" in KEYBINDINGS
    assert "app.model.cycleForward" in KEYBINDINGS
    assert "tui.editor.cursorUp" in KEYBINDINGS
    # ~28 app + ~30 tui
    assert len(KEYBINDINGS) > 50


def test_app_interrupt_default_key() -> None:
    binding = KEYBINDINGS["app.interrupt"]
    assert binding.default_keys == "escape"


def test_migrate_legacy_id() -> None:
    migrated, was_migrated = migrate_keybindings_config({"cursorUp": "up"})
    assert was_migrated is True
    assert "tui.editor.cursorUp" in migrated
    assert migrated["tui.editor.cursorUp"] == "up"


def test_migrate_no_change_for_modern_ids() -> None:
    migrated, was_migrated = migrate_keybindings_config({"app.interrupt": "escape"})
    assert was_migrated is False
    assert migrated == {"app.interrupt": "escape"}


def test_migrate_drops_legacy_when_modern_present() -> None:
    migrated, was_migrated = migrate_keybindings_config({"cursorUp": "up", "tui.editor.cursorUp": "ctrl+u"})
    assert was_migrated is True
    assert migrated == {"tui.editor.cursorUp": "ctrl+u"}


def test_migrate_orders_known_bindings_first() -> None:
    migrated, _ = migrate_keybindings_config(
        {"zzzzz.unknown": "x", "app.interrupt": "escape", "tui.editor.cursorUp": "up"}
    )
    keys = list(migrated.keys())
    assert keys.index("tui.editor.cursorUp") < keys.index("zzzzz.unknown")
    assert keys.index("app.interrupt") < keys.index("zzzzz.unknown")


def test_manager_loads_from_file(tmp_path: Path) -> None:
    config_file = tmp_path / "keybindings.json"
    config_file.write_text(json.dumps({"app.interrupt": "ctrl+c"}))
    manager = KeybindingsManager.create(agent_dir=str(tmp_path))
    keys = manager.get_keys("app.interrupt")
    assert "ctrl+c" in keys


def test_manager_handles_missing_file(tmp_path: Path) -> None:
    manager = KeybindingsManager.create(agent_dir=str(tmp_path))
    # Default key still resolves.
    assert manager.get_keys("app.interrupt") == ["escape"]


def test_manager_handles_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "keybindings.json").write_text("not json {{{")
    manager = KeybindingsManager.create(agent_dir=str(tmp_path))
    assert manager.get_keys("app.interrupt") == ["escape"]


def test_manager_migrates_legacy_on_load(tmp_path: Path) -> None:
    (tmp_path / "keybindings.json").write_text(json.dumps({"cursorUp": "ctrl+u"}))
    manager = KeybindingsManager.create(agent_dir=str(tmp_path))
    assert "ctrl+u" in manager.get_keys("tui.editor.cursorUp")


def test_manager_reload_picks_up_changes(tmp_path: Path) -> None:
    config_file = tmp_path / "keybindings.json"
    config_file.write_text(json.dumps({"app.interrupt": "ctrl+c"}))
    manager = KeybindingsManager.create(agent_dir=str(tmp_path))
    config_file.write_text(json.dumps({"app.interrupt": "ctrl+x"}))
    manager.reload()
    assert "ctrl+x" in manager.get_keys("app.interrupt")


def test_manager_reload_noop_when_no_path() -> None:
    manager = KeybindingsManager()
    # Should not raise.
    manager.reload()
