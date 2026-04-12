"""Tests for ``nu_tui.components.SelectList``."""

from __future__ import annotations

from nu_tui.components.select_list import SelectItem, SelectList, default_select_list_theme


def _items() -> list[SelectItem]:
    return [
        SelectItem(value="alpha", label="Alpha", description="first letter"),
        SelectItem(value="beta", label="Beta", description="second letter"),
        SelectItem(value="gamma", label="Gamma"),
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_shows_selected_prefix() -> None:
    sl = SelectList(_items())
    lines = sl.render(40)
    assert lines[0].startswith("→")
    assert lines[1].startswith("  ")


def test_render_shows_all_items_when_fits() -> None:
    sl = SelectList(_items(), max_visible=10)
    lines = sl.render(40)
    assert any("Alpha" in line for line in lines)
    assert any("Beta" in line for line in lines)
    assert any("Gamma" in line for line in lines)


def test_render_truncates_to_max_visible_and_shows_scroll() -> None:
    items = [SelectItem(value=str(i), label=f"Item {i}") for i in range(20)]
    sl = SelectList(items, max_visible=5)
    lines = sl.render(40)
    # 5 item lines + 1 scroll indicator
    assert len(lines) == 6
    assert "(1/20)" in lines[-1]


def test_render_empty_filter_shows_no_match() -> None:
    sl = SelectList(_items())
    sl.set_filter("zzzzz")
    lines = sl.render(40)
    assert any("No matching" in line for line in lines)


def test_render_with_descriptions_when_wide_enough() -> None:
    sl = SelectList(_items(), max_visible=5)
    lines = sl.render(80)
    # At 80 cols, descriptions should fit.
    assert any("first letter" in line for line in lines)


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------


def test_handle_input_down_wraps() -> None:
    sl = SelectList(_items())
    assert sl.selected_index == 0
    sl.handle_input("down")
    assert sl.selected_index == 1
    sl.handle_input("down")
    sl.handle_input("down")  # past the end → wraps to 0
    assert sl.selected_index == 0


def test_handle_input_up_wraps() -> None:
    sl = SelectList(_items())
    sl.handle_input("up")  # from 0 → wraps to last
    assert sl.selected_index == 2


def test_handle_input_enter_triggers_on_select() -> None:
    sl = SelectList(_items())
    selected: list[SelectItem] = []
    sl.on_select = selected.append
    sl.handle_input("enter")
    assert len(selected) == 1
    assert selected[0].value == "alpha"


def test_handle_input_escape_triggers_on_cancel() -> None:
    sl = SelectList(_items())
    cancelled = []
    sl.on_cancel = lambda: cancelled.append(True)
    sl.handle_input("escape")
    assert cancelled == [True]


def test_on_selection_change_callback_fires() -> None:
    sl = SelectList(_items())
    changes: list[SelectItem] = []
    sl.on_selection_change = changes.append
    sl.handle_input("down")
    assert len(changes) == 1
    assert changes[0].value == "beta"


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_set_filter_narrows_items() -> None:
    sl = SelectList(_items())
    sl.set_filter("al")
    assert len(sl.filtered_items) == 1
    assert sl.filtered_items[0].value == "alpha"


def test_set_filter_resets_selection_to_zero() -> None:
    sl = SelectList(_items())
    sl.set_selected_index(2)
    sl.set_filter("b")
    assert sl.selected_index == 0


def test_set_filter_empty_restores_all_items() -> None:
    sl = SelectList(_items())
    sl.set_filter("xyz")
    assert len(sl.filtered_items) == 0
    sl.set_filter("")
    assert len(sl.filtered_items) == 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_get_selected_item_returns_current() -> None:
    sl = SelectList(_items())
    assert sl.get_selected_item() is not None
    assert sl.get_selected_item().value == "alpha"  # type: ignore[union-attr]


def test_set_selected_index_clamps() -> None:
    sl = SelectList(_items())
    sl.set_selected_index(999)
    assert sl.selected_index == 2
    sl.set_selected_index(-5)
    assert sl.selected_index == 0


def test_default_theme_is_passthrough() -> None:
    theme = default_select_list_theme()
    assert theme.selected_text("x") == "x"
    assert theme.no_match("y") == "y"
