"""Tests for ``nu_tui.component`` (Component / Container base classes)."""

from __future__ import annotations

import pytest
from nu_tui.component import Component, Container


class _Recorder(Component):
    """Test component that records every method call it receives."""

    def __init__(self, name: str, lines: list[str] | None = None) -> None:
        self.name = name
        self._lines = lines or []
        self.invalidations = 0
        self.inputs: list[str] = []
        self.last_width: int | None = None

    def render(self, width: int) -> list[str]:
        self.last_width = width
        return list(self._lines)

    def invalidate(self) -> None:
        self.invalidations += 1

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)


def test_component_is_abstract() -> None:
    """Subclasses without ``render`` cannot be instantiated."""
    with pytest.raises(TypeError):
        Component()  # type: ignore[abstract]


def test_component_default_invalidate_is_noop() -> None:
    """The default ``invalidate`` does not raise."""

    class _Static(Component):
        def render(self, width: int) -> list[str]:
            return ["x" * width]

    _Static().invalidate()  # must not raise


def test_component_default_handle_input_drops_silently() -> None:
    """The default ``handle_input`` accepts and discards the data."""

    class _Static(Component):
        def render(self, width: int) -> list[str]:
            return []

    _Static().handle_input("x")  # must not raise


def test_component_wants_key_release_default_false() -> None:
    """``wants_key_release`` defaults to False."""

    class _Static(Component):
        def render(self, width: int) -> list[str]:
            return []

    assert _Static().wants_key_release is False


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


def test_container_starts_empty() -> None:
    assert Container().children == []


def test_container_add_remove_clear() -> None:
    container = Container()
    a = _Recorder("a")
    b = _Recorder("b")
    container.add_child(a)
    container.add_child(b)
    assert container.children == [a, b]
    container.remove_child(a)
    assert container.children == [b]
    container.clear()
    assert container.children == []


def test_container_remove_unknown_child_is_noop() -> None:
    container = Container()
    container.remove_child(_Recorder("ghost"))  # must not raise
    assert container.children == []


def test_container_render_concatenates_in_order() -> None:
    container = Container()
    container.add_child(_Recorder("a", lines=["one", "two"]))
    container.add_child(_Recorder("b", lines=["three"]))
    assert container.render(40) == ["one", "two", "three"]


def test_container_render_passes_width_through() -> None:
    container = Container()
    a = _Recorder("a")
    b = _Recorder("b")
    container.add_child(a)
    container.add_child(b)
    container.render(72)
    assert a.last_width == 72
    assert b.last_width == 72


def test_container_invalidate_cascades_to_children() -> None:
    container = Container()
    a = _Recorder("a")
    b = _Recorder("b")
    container.add_child(a)
    container.add_child(b)
    container.invalidate()
    assert a.invalidations == 1
    assert b.invalidations == 1


def test_container_render_with_no_children_returns_empty_list() -> None:
    assert Container().render(40) == []


def test_container_empty_child_lines_dont_break_concatenation() -> None:
    """A child returning ``[]`` is skipped without injecting a blank row."""
    container = Container()
    container.add_child(_Recorder("a", lines=[]))
    container.add_child(_Recorder("b", lines=["only"]))
    assert container.render(40) == ["only"]
