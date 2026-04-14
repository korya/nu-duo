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


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_get_selected_item_empty_list() -> None:
    """get_selected_item returns None for empty list (line 126)."""
    sl = SettingsList([])
    assert sl.get_selected_item() is None


def test_render_empty_items() -> None:
    """Render with empty items shows 'No settings available' (lines 145-153)."""
    sl = SettingsList([])
    lines = sl.render(60)
    content = " ".join(lines)
    assert "No settings available" in content
    assert "Enter/Space" in content


def test_scroll_indicator_on_large_list() -> None:
    """Scroll indicator shows when list is taller than max_visible (lines 200-201)."""
    items = [SettingsItem(label=f"Item {i}", value=f"Val {i}") for i in range(20)]
    sl = SettingsList(items, max_visible=5)
    lines = sl.render(60)
    content = " ".join(lines)
    assert "(1/20)" in content


def test_navigation_wraps_around() -> None:
    """Navigation wraps at both ends (lines 242-245)."""
    sl = SettingsList(_items())
    assert sl.selected_index == 0
    sl.handle_input("up")  # wraps to end
    assert sl.selected_index == 2
    sl.handle_input("down")  # wraps to start
    assert sl.selected_index == 0


def test_space_selects_item() -> None:
    """Space key selects the current item (line 249)."""
    sl = SettingsList(_items())
    selected: list[SettingsItem] = []
    sl.on_select = selected.append
    sl.handle_input(" ")
    assert len(selected) == 1
    assert selected[0].label == "Name"


def test_cancel_fires_callback() -> None:
    """Escape fires on_cancel callback (lines 251-252)."""
    sl = SettingsList(_items())
    cancelled: list[bool] = []
    sl.on_cancel = lambda: cancelled.append(True)
    sl.handle_input("escape")
    assert cancelled == [True]


def test_cancel_without_callback_no_crash() -> None:
    """Escape without on_cancel callback doesn't crash."""
    sl = SettingsList(_items())
    sl.on_cancel = None
    sl.handle_input("escape")  # should not raise


def test_select_without_callback_no_crash() -> None:
    """Enter without on_select callback doesn't crash (line 249)."""
    sl = SettingsList(_items())
    sl.on_select = None
    sl.handle_input("enter")  # should not raise


def test_navigation_empty_list_no_crash() -> None:
    """Navigation with empty list doesn't crash."""
    sl = SettingsList([])
    sl.handle_input("up")
    sl.handle_input("down")
    assert sl.selected_index == 0


def test_description_shown_for_selected_item() -> None:
    """Description is shown when selected item has one."""
    items = [
        SettingsItem(label="Theme", value="dark", description="Choose your colour scheme"),
        SettingsItem(label="Size", value="14"),
    ]
    sl = SettingsList(items)
    lines = sl.render(60)
    content = " ".join(lines)
    assert "colour scheme" in content


def test_set_items_clamps_selected_index() -> None:
    """set_items clamps selected_index when list shrinks."""
    sl = SettingsList(_items())
    sl.handle_input("down")
    sl.handle_input("down")
    assert sl.selected_index == 2
    sl.set_items([SettingsItem(label="Only", value="One")])
    assert sl.selected_index == 0
