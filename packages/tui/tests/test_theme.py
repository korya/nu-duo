"""Tests for ``nu_tui.theme``."""

from __future__ import annotations

from nu_tui.theme import dark_theme, get_theme, light_theme, set_theme


def test_dark_theme_default() -> None:
    t = dark_theme()
    assert t.name == "dark"
    styled = t.accent("hello")
    assert "hello" in styled
    assert "\033[" in styled  # has ANSI escape


def test_light_theme() -> None:
    t = light_theme()
    assert t.name == "light"


def test_get_set_theme() -> None:
    original = get_theme()
    try:
        new_theme = light_theme()
        prev = set_theme(new_theme)
        assert get_theme() is new_theme
        assert prev is original
    finally:
        set_theme(original)


def test_theme_to_markdown_theme() -> None:
    t = dark_theme()
    md_theme = t.to_markdown_theme()
    assert md_theme.heading("x") == t.md_heading("x")
    assert md_theme.bold("y") == t.md_bold("y")


def test_theme_to_select_list_theme() -> None:
    t = dark_theme()
    sl_theme = t.to_select_list_theme()
    assert sl_theme.selected_text("x") == t.selected_text("x")


def test_noop_slots_return_unchanged() -> None:
    t = dark_theme()
    assert t.text("hello") == "hello"


def test_error_and_success_styled() -> None:
    t = dark_theme()
    assert "\033[" in t.error("err")
    assert "\033[" in t.success("ok")


def test_dim_styled() -> None:
    t = dark_theme()
    assert "\033[2m" in t.dim("faded")
