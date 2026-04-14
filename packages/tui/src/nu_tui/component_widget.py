"""``ComponentWidget`` — bridge a nu_tui :class:`Component` into a Textual widget.

This is the key integration point between the nu_tui component
library and the Textual-backed interactive mode. Each nu_tui
component (Markdown, Text, Loader, Editor, etc.) can be hosted
inside a Textual layout by wrapping it in a :class:`ComponentWidget`.

The widget calls ``component.render(width)`` on each Textual paint
cycle, joins the lines, and returns a Rich ``Text`` renderable.
This means:

* The nu_tui component's ``render`` contract is preserved exactly —
  it still receives ``width`` and returns ``list[str]``.
* Textual handles layout, scrolling, and focus management.
* The component's visual output matches what the nu_tui ``TUI``
  wrapper would produce (since both call the same ``render`` method).

For interactive components (Input, Editor), the widget also forwards
Textual ``Key`` events to the component's ``handle_input`` method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text as RichText
from textual.widget import Widget

if TYPE_CHECKING:
    from textual import events

    from nu_tui.component import Component


class ComponentWidget(Widget):
    """Textual widget that paints a nu_tui :class:`Component`.

    Usage::

        from nu_tui import Markdown, Text
        from nu_tui.component_widget import ComponentWidget

        # In a Textual App's compose():
        yield ComponentWidget(Markdown("# Hello"), id="msg-1")
        yield ComponentWidget(Text("user prompt"), id="msg-2")

    The widget auto-sizes its height based on the component's output
    line count. Call :meth:`refresh_component` after mutating the
    component's state to trigger a repaint.
    """

    DEFAULT_CSS = """
    ComponentWidget {
        width: 1fr;
        height: auto;
    }
    """

    def __init__(
        self,
        component: Component,
        *,
        forward_keys: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._component = component
        self._forward_keys = forward_keys
        self._cached_lines: list[str] | None = None
        self._cached_width: int | None = None

    @property
    def component(self) -> Component:
        """The wrapped nu_tui component."""
        return self._component

    def set_component(self, component: Component) -> None:
        """Replace the wrapped component and trigger a repaint + relayout."""
        self._component = component
        self._cached_lines = None
        self.refresh(layout=True)

    def refresh_component(self) -> None:
        """Invalidate the cache and trigger a repaint + relayout.

        Call this after mutating the component's internal state
        (e.g. ``Text.set_text(...)`` or ``Loader.set_message(...)``).
        The ``layout=True`` flag is critical — without it Textual
        repaints but does **not** recalculate the widget height, so
        growing content (e.g. streaming deltas) stays clipped to
        the original 1-line height.
        """
        self._cached_lines = None
        self._component.invalidate()
        self.refresh(layout=True)

    def get_content_height(self, container: object, viewport: object, width: int) -> int:
        """Tell Textual how many lines the component needs at *width*."""
        lines = self._component.render(width)
        self._cached_lines = lines
        self._cached_width = width
        return max(1, len(lines)) if lines else 1

    def render(self) -> RichText:
        width = self.size.width if self.size.width > 0 else 80
        if self._cached_lines is not None and self._cached_width == width:
            return RichText("\n".join(self._cached_lines))
        lines = self._component.render(width)
        self._cached_lines = lines
        self._cached_width = width
        return RichText("\n".join(lines))

    def on_key(self, event: events.Key) -> None:
        """Forward key events to the component if ``forward_keys`` is enabled."""
        if self._forward_keys:
            self._component.handle_input(event.key)
            self.refresh_component()
            event.stop()


__all__ = ["ComponentWidget"]
