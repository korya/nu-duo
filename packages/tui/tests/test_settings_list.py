"""Tests for ``nu_tui.components.SettingsList``."""

from __future__ import annotations

from nu_tui.components.settings_list import SettingsItem, SettingsList


def _items() -> list[SettingsItem]:
    return [
        SettingsItem(label="Name", value="Dark", description="Theme name"),
        SettingsItem(label="Size", value="Large"),
        SettingsItem(label="Mode", value="Auto"),
    ]


def test_render_shows_items() -> None:
    sl = SettingsList(_items())
    lines = sl.render(60)
    assert len(lines) >= 3
    content = " ".join(lines)
    assert "Name" in content
    assert "Dark" in content


def test_render_selected_item_has_marker() -> None:
    sl = SettingsList(_items())
    lines = sl.render(60)
    assert lines[0].startswith("→") or "→" in lines[0]


def test_handle_input_navigation() -> None:
    sl = SettingsList(_items())
    assert sl.selected_index == 0
    sl.handle_input("down")
    assert sl.selected_index == 1
    sl.handle_input("up")
    assert sl.selected_index == 0


def test_handle_input_select_fires_callback() -> None:
    sl = SettingsList(_items())
    selected: list[SettingsItem] = []
    sl.on_select = selected.append
    sl.handle_input("enter")
    assert len(selected) == 1
    assert selected[0].label == "Name"


def test_set_items_replaces() -> None:
    sl = SettingsList(_items())
    sl.set_items([SettingsItem(label="New", value="Item")])
    lines = sl.render(60)
    content = " ".join(lines)
    assert "New" in content
