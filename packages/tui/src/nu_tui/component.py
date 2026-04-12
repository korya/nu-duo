"""Component / Container — port of ``packages/tui/src/tui.ts`` (interfaces).

This module declares the *contract* every nu_tui component implements:
:meth:`Component.render` produces a list of plain strings (one per
output line) given the current viewport width, and :meth:`Component.invalidate`
clears any cached state. The Textual integration in :mod:`nu_tui.tui`
walks an entire :class:`Container` tree once per frame, joins the
results, and hands them to a single :class:`textual.widget.Widget`
that paints to the terminal.

Design rationale (deviation from upstream)
------------------------------------------

Upstream ``pi-tui`` is a hand-rolled differential renderer: every
component returns lines as strings, and the TUI computes the minimal
set of terminal updates needed each frame. This Python port keeps
the **same Component contract** (so consumer code in ``nu_coding_agent``
doesn't have to be rewritten when interactive mode lands) but
delegates the actual painting to `Textual <https://textual.textualize.io/>`_.
The porting plan called this out as the one place we deviate at
implementation level, not API level.

Concretely:

* :class:`Component` and :class:`Container` are pure-Python — no
  Textual import. Tests can exercise them in isolation.
* :mod:`nu_tui.tui` provides the bridge: a Textual ``App`` whose
  body widget renders the result of walking the container tree.
* Future component implementations (Spacer, Text, Editor, etc.)
  subclass :class:`Component` directly and only need to implement
  ``render(width)`` — they have no Textual dependency either.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod


class Component(ABC):
    """Abstract base class for every nu_tui component.

    Subclasses must implement :meth:`render`. The default
    :meth:`invalidate` is a no-op so simple components don't need to
    override it. :attr:`wants_key_release` defaults to ``False``
    (matching upstream ``Component.wantsKeyRelease``).
    """

    #: If ``True``, this component receives key release events
    #: (Kitty keyboard protocol). Defaults to ``False`` so release
    #: events are filtered out before reaching :meth:`handle_input`.
    wants_key_release: bool = False

    @abstractmethod
    def render(self, width: int) -> list[str]:
        """Render this component to a list of output lines.

        ``width`` is the current viewport width in columns. Each
        returned string is one terminal row; embedded newlines are
        not allowed (the renderer assumes one entry per row).
        """

    def invalidate(self) -> None:  # noqa: B027 — intentionally not abstract; default no-op
        """Clear any cached rendering state.

        Default implementation is a no-op. Components that cache
        layout / wrapping / colour data override this to drop their
        caches when (e.g.) the theme changes.
        """

    def handle_input(self, data: str) -> None:  # noqa: B027 — intentionally not abstract; default drop
        """Optional handler for keyboard input when this component has focus.

        Default implementation drops the input. Components that need
        to react to keypresses (input fields, editors, list pickers)
        override this. The Textual bridge in :mod:`nu_tui.tui` calls
        this method on the focused component for every key event.
        """


class Container(Component):
    """A :class:`Component` that owns an ordered list of child components.

    The default :meth:`render` walks every child in order and
    concatenates their line lists. :meth:`invalidate` cascades to
    every child. Subclasses can override :meth:`render` to add layout
    decorations (borders, padding, etc.).
    """

    def __init__(self) -> None:
        self.children: list[Component] = []

    def add_child(self, component: Component) -> None:
        """Append ``component`` to the end of the child list."""
        self.children.append(component)

    def remove_child(self, component: Component) -> None:
        """Remove ``component`` if present. No-op if it isn't a child."""
        with contextlib.suppress(ValueError):
            self.children.remove(component)

    def clear(self) -> None:
        """Drop every child."""
        self.children = []

    def invalidate(self) -> None:
        """Cascade :meth:`Component.invalidate` to every child."""
        for child in self.children:
            child.invalidate()

    def render(self, width: int) -> list[str]:
        """Render every child and concatenate their line lists."""
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines


__all__ = ["Component", "Container"]
