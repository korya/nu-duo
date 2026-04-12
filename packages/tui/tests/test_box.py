"""Tests for ``nu_tui.components.Box``."""

from __future__ import annotations

from nu_tui.components.box import Box
from nu_tui.components.text import Text
from nu_tui.utils import visible_width


def test_empty_box_returns_no_lines() -> None:
    box = Box()
    assert box.render(40) == []


def test_box_adds_padding_y_above_and_below() -> None:
    box = Box(padding_x=0, padding_y=2)
    box.add_child(Text("content"))
    lines = box.render(20)
    # 2 above + 1 content + 2 below = 5
    assert len(lines) == 5
    assert "content" in lines[2]


def test_box_adds_padding_x_to_each_line() -> None:
    box = Box(padding_x=3, padding_y=0)
    box.add_child(Text("hi"))
    lines = box.render(20)
    assert len(lines) == 1
    assert lines[0].startswith("   ")  # 3 spaces left padding


def test_box_pads_lines_to_full_width() -> None:
    box = Box(padding_x=1, padding_y=0)
    box.add_child(Text("x"))
    lines = box.render(20)
    assert all(visible_width(line) == 20 for line in lines)


def test_box_bg_fn_applied_to_every_line() -> None:
    box = Box(padding_x=0, padding_y=1, bg_fn=lambda s: f"[{s}]")
    box.add_child(Text("hi"))
    lines = box.render(10)
    # 1 top padding + 1 content + 1 bottom padding = 3
    assert len(lines) == 3
    assert all(line.startswith("[") and line.endswith("]") for line in lines)


def test_box_add_remove_clear() -> None:
    box = Box(padding_x=0, padding_y=0)
    a = Text("a")
    b = Text("b")
    box.add_child(a)
    box.add_child(b)
    assert len(box.children) == 2
    box.remove_child(a)
    assert len(box.children) == 1
    box.clear()
    assert box.children == []


def test_box_remove_unknown_child_is_noop() -> None:
    box = Box()
    box.remove_child(Text("ghost"))  # must not raise


def test_box_invalidate_cascades_to_children() -> None:
    from nu_tui.component import Component  # noqa: PLC0415

    invalidated: list[str] = []

    class _Tracker(Component):
        def __init__(self, name: str) -> None:
            self.name = name

        def render(self, width: int) -> list[str]:
            return [self.name]

        def invalidate(self) -> None:
            invalidated.append(self.name)

    box = Box(padding_x=0, padding_y=0)
    box.add_child(_Tracker("a"))
    box.add_child(_Tracker("b"))
    box.invalidate()
    assert invalidated == ["a", "b"]


def test_box_caches_output() -> None:
    box = Box(padding_x=0, padding_y=0)
    box.add_child(Text("stable"))
    first = box.render(20)
    second = box.render(20)
    # Same list object returned from cache.
    assert first is second


def test_box_cache_invalidated_on_width_change() -> None:
    box = Box(padding_x=0, padding_y=0)
    box.add_child(Text("stable"))
    first = box.render(20)
    second = box.render(30)
    assert first is not second


def test_box_set_bg_fn_changes_output_on_next_render() -> None:
    box = Box(padding_x=0, padding_y=0)
    box.add_child(Text("hi"))
    lines_plain = box.render(10)
    box.set_bg_fn(lambda s: f"[{s}]")
    lines_styled = box.render(10)
    assert lines_plain != lines_styled
    assert all("[" in line for line in lines_styled)
