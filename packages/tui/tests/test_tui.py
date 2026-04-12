"""Tests for ``nu_tui.tui`` — the Textual-backed TUI wrapper.

The pure-API tests don't actually launch Textual; they call
:meth:`TUI.render` directly to assert on the line list. The
``Pilot``-backed tests at the bottom exercise the Textual bridge
itself (key dispatch, widget compose, etc.) without painting to a
real terminal.
"""

from __future__ import annotations

from typing import Any

from nu_tui import Component, Spacer, Terminal, TerminalSize, Text
from nu_tui.tui import TUI

# ---------------------------------------------------------------------------
# Pure-API tests — no Textual event loop
# ---------------------------------------------------------------------------


def test_tui_starts_empty_and_renders_nothing() -> None:
    assert TUI().render(40) == []


def test_tui_inherits_container_render_concatenation() -> None:
    tui = TUI()
    tui.add_child(Text("hello"))
    tui.add_child(Spacer(1))
    tui.add_child(Text("world"))
    assert tui.render(40) == ["hello", "", "world"]


def test_tui_default_terminal_is_constructed() -> None:
    tui = TUI()
    assert isinstance(tui.terminal, Terminal)


def test_tui_explicit_terminal_passed_through() -> None:
    term = Terminal(size=TerminalSize(columns=132, rows=42))
    tui = TUI(terminal=term)
    assert tui.terminal is term
    assert tui.terminal.get_columns() == 132


def test_tui_show_hardware_cursor_get_set() -> None:
    tui = TUI()
    assert tui.get_show_hardware_cursor() is False
    tui.set_show_hardware_cursor(True)
    assert tui.get_show_hardware_cursor() is True


def test_tui_set_focus_records_focused_component() -> None:
    tui = TUI()
    text = Text("focus me")
    tui.add_child(text)
    tui.set_focus(text)
    assert tui.get_focused() is text
    tui.set_focus(None)
    assert tui.get_focused() is None


# ---------------------------------------------------------------------------
# Input listeners + key dispatch (driven directly, no Textual)
# ---------------------------------------------------------------------------


class _InputRecorder(Component):
    """Component that records every ``handle_input`` call."""

    def __init__(self) -> None:
        self.received: list[str] = []

    def render(self, width: int) -> list[str]:
        return []

    def handle_input(self, data: str) -> None:
        self.received.append(data)


def test_tui_focused_component_receives_keys() -> None:
    tui = TUI()
    rec = _InputRecorder()
    tui.add_child(rec)
    tui.set_focus(rec)
    consumed = tui.dispatch_key("a")
    assert consumed is True
    assert rec.received == ["a"]


def test_tui_no_focus_drops_keys_quietly() -> None:
    tui = TUI()
    consumed = tui.dispatch_key("x")
    assert consumed is False


def test_tui_input_listener_receives_keys() -> None:
    tui = TUI()
    received: list[str] = []
    tui.add_input_listener(lambda key: received.append(key) or None)
    tui.dispatch_key("a")
    tui.dispatch_key("b")
    assert received == ["a", "b"]


def test_tui_input_listener_can_consume_key() -> None:
    """A listener returning ``{'consume': True}`` blocks the focused component."""
    tui = TUI()
    rec = _InputRecorder()
    tui.add_child(rec)
    tui.set_focus(rec)
    tui.add_input_listener(lambda key: {"consume": True})
    consumed = tui.dispatch_key("x")
    assert consumed is True
    assert rec.received == []  # listener swallowed it


def test_tui_input_listener_can_substitute_key() -> None:
    """A listener returning ``{'data': 'X'}`` rewrites the key for downstream listeners + focus."""
    tui = TUI()
    rec = _InputRecorder()
    tui.add_child(rec)
    tui.set_focus(rec)
    tui.add_input_listener(lambda key: {"data": "X"} if key == "a" else None)
    tui.dispatch_key("a")
    assert rec.received == ["X"]


def test_tui_input_listener_unsubscribe_stops_delivery() -> None:
    tui = TUI()
    received: list[str] = []
    unsubscribe = tui.add_input_listener(lambda key: received.append(key) or None)
    tui.dispatch_key("a")
    unsubscribe()
    tui.dispatch_key("b")
    assert received == ["a"]


def test_tui_remove_input_listener_unknown_is_noop() -> None:
    tui = TUI()
    tui.remove_input_listener(lambda key: None)  # must not raise


def test_tui_listener_exception_does_not_break_dispatch() -> None:
    """A broken listener is logged but does not block other listeners or focus."""
    tui = TUI()
    rec = _InputRecorder()
    tui.add_child(rec)
    tui.set_focus(rec)

    def boom(_key: str) -> dict[str, Any]:
        raise RuntimeError("listener exploded")

    tui.add_input_listener(boom)
    tui.dispatch_key("a")
    assert rec.received == ["a"]


def test_tui_focused_component_exception_does_not_break_tui() -> None:
    """A broken focused component logs but the dispatcher still reports consumed."""
    tui = TUI()

    class _Bad(Component):
        def render(self, width: int) -> list[str]:
            return []

        def handle_input(self, data: str) -> None:
            raise RuntimeError("component exploded")

    bad = _Bad()
    tui.add_child(bad)
    tui.set_focus(bad)
    consumed = tui.dispatch_key("a")
    assert consumed is True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_tui_request_render_before_start_is_noop() -> None:
    """Calling ``request_render`` before the app exists must not raise."""
    TUI().request_render()


def test_tui_stop_before_start_is_safe() -> None:
    tui = TUI()
    tui.stop()
    # And idempotent.
    tui.stop()


# ---------------------------------------------------------------------------
# Textual bridge (Pilot-based) — proves the Textual app actually mounts
# ---------------------------------------------------------------------------


async def test_tui_pilot_mounts_widget_and_renders_children() -> None:
    """End-to-end: launch the Textual app via Pilot, assert the widget is up."""
    tui = TUI()
    tui.add_child(Text("pilot says hi"))
    async with tui.run_test():
        # Just confirming the app started; the widget is created lazily in
        # compose() so it should be present after Pilot enters context.
        assert tui.app is not None
        assert tui.app.widget is not None


async def test_tui_request_render_after_start_calls_widget_refresh() -> None:
    """``request_render`` after ``start`` is safe and reaches the widget."""
    tui = TUI()
    tui.add_child(Text("hi"))
    async with tui.run_test():
        # Sanity: no crash, widget exists.
        tui.request_render()
        tui.request_render(force=True)
        assert tui.app is not None and tui.app.widget is not None
