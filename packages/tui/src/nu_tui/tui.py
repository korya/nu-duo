"""``TUI`` — Textual-backed wrapper that drives the nu_tui contract.

Phase 5.1 of the nu_tui port. The upstream ``packages/tui/src/tui.ts``
is a 1243-line hand-rolled differential renderer. The Python port
**replaces the renderer with Textual** but exposes the same public
API surface so consumer code (eventually
``nu_coding_agent.modes.interactive``) can be ported file-for-file
without rewriting.

How the bridge works
--------------------

1. :class:`TUI` is a :class:`Container` (so callers add components
   via the inherited ``add_child`` / ``remove_child`` / ``clear``).
2. ``start()`` builds a :class:`textual.app.App` whose body is a
   single ``_TUIWidget`` instance. The widget's ``render`` method
   walks the TUI's container tree, calls ``Component.render(width)``
   on each child, joins the resulting lines into a Rich
   :class:`rich.text.Text`, and returns it.
3. ``request_render()`` calls ``widget.refresh()`` so Textual
   schedules a repaint on its next idle tick. Most callers can
   leave it to Textual — the widget refreshes automatically when
   Textual detects size changes — but explicit ``request_render()``
   matches the upstream contract for sites that mutate component
   state outside an event handler.
4. ``set_focus(component)`` records the focused child; the bridge
   forwards Textual ``key`` events to its ``handle_input``.
5. ``add_input_listener(listener)`` registers a callback that
   receives the raw key string. Listeners can return
   ``{"consume": True}`` to suppress the key reaching the focused
   component.

What this slice does **not** ship
---------------------------------

* Overlays (``show_overlay`` / ``hide_overlay``). Defer to a
  follow-up slice — needs Textual's screen/modal stack mapped onto
  upstream's ``OverlayHandle`` contract.
* Hardware cursor placement / ``CURSOR_MARKER`` parsing. Textual
  manages its own cursor; surfacing the upstream cursor protocol
  is a separate slice.
* Differential rendering optimisations. Textual already does the
  right thing — we don't need ``getClearOnShrink`` /
  ``setClearOnShrink`` plumbing yet.
* Multi-screen / popup management. Phase 5.7+ territory.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rich.text import Text as RichText
from textual.app import App, ComposeResult
from textual.widget import Widget

from nu_tui.component import Component, Container
from nu_tui.terminal import Terminal

if TYPE_CHECKING:
    from textual import events


#: Result type for input listeners — same shape as upstream's
#: ``InputListenerResult``. Listeners may return ``None`` to fall
#: through, ``{"consume": True}`` to swallow the key, or
#: ``{"data": "..."}`` to substitute the key string.
type InputListenerResult = dict[str, Any] | None


#: Listener signature: receives the raw key string, returns a result.
type InputListener = Callable[[str], InputListenerResult]


# ---------------------------------------------------------------------------
# Internal Textual widget that paints the TUI's component tree
# ---------------------------------------------------------------------------


class _TUIWidget(Widget):
    """Textual widget that renders the parent :class:`TUI` once per frame."""

    DEFAULT_CSS = """
    _TUIWidget {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(self, tui: TUI) -> None:
        super().__init__()
        self._tui = tui

    def render(self) -> RichText:
        width = self.size.width if self.size.width > 0 else self._tui.terminal.get_columns()
        lines = self._tui.render(width)
        return RichText("\n".join(lines))


# ---------------------------------------------------------------------------
# Internal Textual app that hosts the _TUIWidget
# ---------------------------------------------------------------------------


class _TUIApp(App[None]):
    """Textual ``App`` whose only body widget is the bridged TUI tree."""

    def __init__(self, tui: TUI) -> None:
        super().__init__()
        self._tui = tui
        self._widget: _TUIWidget | None = None

    def compose(self) -> ComposeResult:
        self._widget = _TUIWidget(self._tui)
        yield self._widget

    @property
    def widget(self) -> _TUIWidget | None:
        return self._widget

    async def on_key(self, event: events.Key) -> None:
        # Forward to the TUI's key dispatch — the TUI decides whether
        # to consume the key (input listeners), forward it to the
        # focused component, or drop it.
        consumed = self._tui.dispatch_key(event.key)
        if consumed:
            event.stop()


# ---------------------------------------------------------------------------
# Public TUI class
# ---------------------------------------------------------------------------


class TUI(Container):
    """Top-level :class:`Container` driving a Textual app.

    Construct directly (``TUI()``) and add children via the
    inherited :meth:`Container.add_child`. Call :meth:`start` to
    enter the Textual event loop; :meth:`stop` exits cleanly.
    Tests typically don't call ``start`` — they call
    :meth:`render` directly to get the line list and assert on it.
    """

    def __init__(
        self,
        *,
        terminal: Terminal | None = None,
        show_hardware_cursor: bool = False,
    ) -> None:
        super().__init__()
        self._terminal = terminal or Terminal()
        self._show_hardware_cursor = show_hardware_cursor
        self._focused: Component | None = None
        self._input_listeners: list[InputListener] = []
        self._app: _TUIApp | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def terminal(self) -> Terminal:
        return self._terminal

    @property
    def app(self) -> _TUIApp | None:
        """Return the underlying Textual app, or ``None`` before :meth:`start`."""
        return self._app

    def get_show_hardware_cursor(self) -> bool:
        return self._show_hardware_cursor

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        self._show_hardware_cursor = enabled

    # ------------------------------------------------------------------
    # Focus
    # ------------------------------------------------------------------

    def set_focus(self, component: Component | None) -> None:
        """Record which child currently receives keyboard input.

        ``None`` clears the focus. The Textual bridge looks at
        :attr:`_focused` when a key event arrives and routes the
        event to that component's :meth:`Component.handle_input`.
        """
        self._focused = component

    def get_focused(self) -> Component | None:
        return self._focused

    # ------------------------------------------------------------------
    # Input listeners
    # ------------------------------------------------------------------

    def add_input_listener(self, listener: InputListener) -> Callable[[], None]:
        """Register a key listener. Returns an unsubscribe callable.

        Listeners run in registration order. A listener may return
        ``{"consume": True}`` to swallow the key (the next listener
        and the focused component never see it). Errors raised by a
        listener are caught and logged so a misbehaving subscriber
        can't break the whole TUI.
        """
        self._input_listeners.append(listener)

        def unsubscribe() -> None:
            self.remove_input_listener(listener)

        return unsubscribe

    def remove_input_listener(self, listener: InputListener) -> None:
        with contextlib.suppress(ValueError):
            self._input_listeners.remove(listener)

    def dispatch_key(self, key: str) -> bool:
        """Internal: route a key event through listeners + the focused component.

        Returns ``True`` iff the key was consumed (so the Textual
        bridge can stop event propagation).
        """
        consumed = False
        forwarded_key = key
        for listener in list(self._input_listeners):
            try:
                result = listener(forwarded_key)
            except Exception as exc:
                # Mirrors the upstream's "swallow + log" pattern; a
                # broken listener mustn't take the TUI down.
                import sys  # noqa: PLC0415

                print(f"TUI input listener error: {exc}", file=sys.stderr)
                continue
            if result is None:
                continue
            if result.get("consume"):
                consumed = True
                break
            substitute = result.get("data")
            if isinstance(substitute, str):
                forwarded_key = substitute

        if not consumed and self._focused is not None:
            try:
                self._focused.handle_input(forwarded_key)
            except Exception as exc:
                import sys  # noqa: PLC0415

                print(f"TUI focused component handle_input error: {exc}", file=sys.stderr)
            consumed = True
        return consumed

    # ------------------------------------------------------------------
    # Rendering / lifecycle
    # ------------------------------------------------------------------

    def request_render(self, force: bool = False) -> None:
        """Schedule a repaint of the underlying Textual widget.

        ``force`` is accepted for upstream-API parity but currently
        unused — Textual decides when to repaint based on widget
        invalidation. Calling :meth:`Component.invalidate` is the
        idiomatic way to force a repaint of cached layout state.
        """
        if self._app is None:
            return
        widget = self._app.widget
        if widget is not None:
            widget.refresh()

    async def start(self) -> None:
        """Run the Textual app until :meth:`stop` is called.

        ``start`` is async; call ``await tui.start()`` from inside
        an event loop. For tests, prefer
        :meth:`run_test` which uses Textual's ``Pilot`` and never
        actually paints to a terminal.
        """
        if self._closed:
            raise RuntimeError("TUI is closed")
        self._app = _TUIApp(self)
        await self._app.run_async()

    def stop(self) -> None:
        """Exit the Textual event loop. Safe to call before :meth:`start`."""
        self._closed = True
        if self._app is not None:
            with contextlib.suppress(Exception):
                self._app.exit()

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def run_test(self) -> Any:
        """Return a Textual ``Pilot`` context manager for headless tests.

        Builds the underlying app on demand. Usage::

            tui = TUI()
            tui.add_child(Text("hello"))
            async with tui.run_test() as pilot:
                await pilot.pause()
                # ... assert on tui.app.widget render ...
        """
        if self._app is None:
            self._app = _TUIApp(self)
        return self._app.run_test()


__all__ = [
    "TUI",
    "InputListener",
    "InputListenerResult",
]


# Keep ``asyncio`` referenced for tests that schedule the runner
# against a real loop (and to keep static analyzers from culling
# the import across edits).
_ = asyncio
