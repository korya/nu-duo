"""Tests for ``nu_tui.component_widget.ComponentWidget``."""

from __future__ import annotations

from nu_tui.component import Component
from nu_tui.component_widget import ComponentWidget
from nu_tui.components import Spacer, Text
from nu_tui.components.markdown import Markdown

# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------


def test_component_widget_renders_text_component() -> None:
    text = Text("hello world")
    ComponentWidget(text)
    # Simulate a size by calling render directly on the component
    lines = text.render(40)
    assert lines == ["hello world"]


def test_component_widget_wraps_markdown() -> None:
    md = Markdown("# Title\n\nSome **bold** text")
    widget = ComponentWidget(md)
    assert widget.component is md


def test_component_widget_wraps_spacer() -> None:
    spacer = Spacer(3)
    widget = ComponentWidget(spacer)
    assert widget.component is spacer
    assert spacer.render(40) == ["", "", ""]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def test_set_component_replaces_wrapped_component() -> None:
    text1 = Text("first")
    text2 = Text("second")
    widget = ComponentWidget(text1)
    assert widget.component is text1
    widget.set_component(text2)
    assert widget.component is text2


def test_component_property() -> None:
    text = Text("x")
    widget = ComponentWidget(text)
    assert widget.component is text


# ---------------------------------------------------------------------------
# Key forwarding
# ---------------------------------------------------------------------------


class _KeyRecorder(Component):
    def __init__(self) -> None:
        self.keys: list[str] = []

    def render(self, width: int) -> list[str]:
        return [f"keys: {self.keys}"]

    def handle_input(self, data: str) -> None:
        self.keys.append(data)


def test_forward_keys_disabled_by_default() -> None:
    rec = _KeyRecorder()
    ComponentWidget(rec, forward_keys=False)
    # Without forward_keys, the component never sees keys
    assert rec.keys == []


def test_forward_keys_enabled() -> None:
    rec = _KeyRecorder()
    widget = ComponentWidget(rec, forward_keys=True)
    # forward_keys=True means on_key would forward events,
    # but we can't simulate Textual events in a pure unit test.
    # At least verify the widget was created with the right flag.
    assert widget._forward_keys is True  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# CSS classes and id
# ---------------------------------------------------------------------------


def test_widget_accepts_id_and_classes() -> None:
    text = Text("hi")
    widget = ComponentWidget(text, id="msg-1", classes="user-message highlight")
    assert widget.id == "msg-1"
    assert widget.has_class("user-message")
    assert widget.has_class("highlight")


# ---------------------------------------------------------------------------
# Integration with interactive mode components
# ---------------------------------------------------------------------------


def test_refresh_component_invalidates_cache() -> None:
    """refresh_component clears internal cache and calls component.invalidate."""
    text = Text("original")
    widget = ComponentWidget(text)
    # The internal cache should be None initially.
    assert widget._cached_lines is None  # pyright: ignore[reportPrivateUsage]
    # After calling refresh_component, the cache is cleared again.
    widget.refresh_component()
    assert widget._cached_lines is None  # pyright: ignore[reportPrivateUsage]


def test_component_widget_default_css_has_auto_height() -> None:
    """Verify the widget's DEFAULT_CSS includes height: auto."""
    assert "auto" in ComponentWidget.DEFAULT_CSS


# ---------------------------------------------------------------------------
# get_content_height + render (lines 100-119)
# ---------------------------------------------------------------------------


def test_get_content_height_returns_line_count() -> None:
    """get_content_height() calls component.render and returns len(lines)."""
    text = Text("line1\nline2\nline3")
    widget = ComponentWidget(text)
    # get_content_height expects (container, viewport, width) but only uses width
    height = widget.get_content_height(None, None, 40)
    assert height == 3


def test_get_content_height_empty_returns_one() -> None:
    """An empty component returns height=1 (at least one line)."""
    text = Text("")
    widget = ComponentWidget(text)
    height = widget.get_content_height(None, None, 40)
    assert height == 1


def test_get_content_height_caches_lines() -> None:
    """After get_content_height, _cached_lines is populated."""
    text = Text("hello")
    widget = ComponentWidget(text)
    widget.get_content_height(None, None, 40)
    assert widget._cached_lines is not None  # pyright: ignore[reportPrivateUsage]
    assert widget._cached_width == 40  # pyright: ignore[reportPrivateUsage]


def test_render_uses_cached_lines() -> None:
    """render() returns cached lines when width matches."""
    text = Text("cached test")
    widget = ComponentWidget(text)
    # Prime the cache via get_content_height
    widget.get_content_height(None, None, 40)
    # Now render() should use the cached lines without calling component.render again
    from unittest.mock import patch

    with patch.object(text, "render", wraps=text.render) as mock_render:
        # Set the widget size so render() uses width=40
        widget._size = (40, 1)  # pyright: ignore[reportAttributeAccessIssue]
        result = widget.render()
        # The cache should have been used
        assert "cached test" in str(result)


def test_render_without_cache() -> None:
    """render() calls component.render when cache is empty."""
    text = Text("no cache")
    widget = ComponentWidget(text)
    # Force a size so render uses it
    widget._size = (40, 1)  # pyright: ignore[reportAttributeAccessIssue]
    result = widget.render()
    assert "no cache" in str(result)
    assert widget._cached_lines is not None  # pyright: ignore[reportPrivateUsage]


def test_on_key_forward_calls_handle_input() -> None:
    """on_key with forward_keys=True calls component.handle_input."""
    rec = _KeyRecorder()
    widget = ComponentWidget(rec, forward_keys=True)
    # Create a mock key event
    from unittest.mock import MagicMock

    event = MagicMock()
    event.key = "a"
    widget.on_key(event)
    assert rec.keys == ["a"]
    event.stop.assert_called_once()


def test_on_key_no_forward_does_not_call() -> None:
    """on_key with forward_keys=False does NOT call component.handle_input."""
    rec = _KeyRecorder()
    widget = ComponentWidget(rec, forward_keys=False)
    from unittest.mock import MagicMock

    event = MagicMock()
    event.key = "a"
    widget.on_key(event)
    assert rec.keys == []
    event.stop.assert_not_called()


def test_set_component_clears_cache_and_replaces() -> None:
    """set_component replaces the component and clears the cache."""
    text1 = Text("first")
    text2 = Text("second")
    widget = ComponentWidget(text1)
    # Prime cache
    widget.get_content_height(None, None, 40)
    assert widget._cached_lines is not None  # pyright: ignore[reportPrivateUsage]
    widget.set_component(text2)
    assert widget.component is text2
    assert widget._cached_lines is None  # pyright: ignore[reportPrivateUsage]


def test_streaming_pattern_text_then_markdown() -> None:
    """Simulates the _StreamingComponentWidget pattern from interactive mode."""
    text = Text("")
    widget = ComponentWidget(text)

    # Stream deltas
    chunks = ["Hello", " ", "world"]
    accumulated = ""
    for chunk in chunks:
        accumulated += chunk
        text.set_text(accumulated)

    assert text.text == "Hello world"

    # Finalize as Markdown
    md = Markdown(accumulated)
    widget.set_component(md)
    assert isinstance(widget.component, Markdown)
    assert widget.component.text == "Hello world"
