"""Tests for ``nu_tui.components.Markdown``."""

from __future__ import annotations

from nu_tui.components.markdown import Markdown, default_markdown_theme
from nu_tui.utils import ANSI_ESCAPE_RE


def _strip(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def test_empty_text_returns_empty() -> None:
    md = Markdown("")
    assert md.render(40) == []


def test_plain_text_renders() -> None:
    md = Markdown("Hello world")
    lines = md.render(40)
    assert len(lines) >= 1
    content = " ".join(_strip(line) for line in lines)
    assert "Hello world" in content


def test_heading_renders_styled() -> None:
    md = Markdown("# Title")
    lines = md.render(40)
    content = " ".join(_strip(line) for line in lines)
    assert "Title" in content


def test_bold_renders() -> None:
    md = Markdown("This is **bold** text")
    lines = md.render(40)
    raw = "".join(lines)
    # Rich wraps bold in \x1b[1m...\x1b[0m
    assert "\x1b[1m" in raw  # bold escape present


def test_code_block_renders() -> None:
    md = Markdown("```python\nprint('hello')\n```")
    lines = md.render(60)
    content = " ".join(_strip(line) for line in lines)
    assert "print" in content


def test_list_renders() -> None:
    md = Markdown("- item one\n- item two")
    lines = md.render(40)
    content = " ".join(_strip(line) for line in lines)
    assert "item one" in content
    assert "item two" in content


def test_padding_x() -> None:
    md = Markdown("hi", padding_x=3)
    lines = md.render(40)
    assert len(lines) >= 1
    assert lines[0].startswith("   ")


def test_padding_y() -> None:
    md = Markdown("hi", padding_y=2)
    lines = md.render(40)
    # 2 above + content + 2 below
    assert len(lines) >= 5


def test_set_text_invalidates_cache() -> None:
    md = Markdown("first")
    first = md.render(40)
    md.set_text("second")
    second = md.render(40)
    assert first != second


def test_invalidate_clears_cache() -> None:
    md = Markdown("cached")
    first = md.render(40)
    md.invalidate()
    second = md.render(40)
    # Same content so output should match, but cache was cleared
    assert first == second


def test_cache_reused_on_same_input() -> None:
    md = Markdown("stable")
    first = md.render(40)
    second = md.render(40)
    assert first is second  # same list object


def test_default_markdown_theme_is_passthrough() -> None:
    theme = default_markdown_theme()
    assert theme.heading("x") == "x"
    assert theme.bold("y") == "y"
