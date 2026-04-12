"""nu_tui — Python port of ``@mariozechner/pi-tui``.

Foundation slice (Phase 5.1) re-exports the public surface that's
been ported so far:

* :class:`Component` / :class:`Container` — abstract base classes
  every nu_tui component implements.
* :class:`TUI` — Textual-backed top-level container that drives the
  Textual app and bridges key events to focused components.
* :class:`Terminal` / :class:`TerminalSize` — capability façade
  over the running terminal.
* :class:`Spacer` / :class:`Text` — the two simplest built-in
  components, used to validate the wrapping pattern. Editor /
  Input / Markdown / Loader / Image / SelectList / SettingsList /
  Box / TruncatedText land in follow-up slices.

Pure-logic utilities (already ported in earlier slices):

* :mod:`nu_tui.fuzzy` — fuzzy substring scoring.
* :mod:`nu_tui.keys` — key code constants and parser.
* :mod:`nu_tui.keybindings` — keybinding config + lookup.
* :mod:`nu_tui.kill_ring` — Emacs-style kill ring for editors.
* :mod:`nu_tui.undo_stack` — bounded undo/redo stack.
"""

from nu_tui.component import Component, Container
from nu_tui.component_widget import ComponentWidget
from nu_tui.components import Box, SelectItem, SelectList, SelectListTheme, Spacer, Text, TruncatedText
from nu_tui.terminal import Terminal, TerminalSize
from nu_tui.tui import TUI, InputListener, InputListenerResult
from nu_tui.utils import truncate_to_width, visible_width

__all__ = [
    "TUI",
    "Box",
    "Component",
    "ComponentWidget",
    "Container",
    "InputListener",
    "InputListenerResult",
    "SelectItem",
    "SelectList",
    "SelectListTheme",
    "Spacer",
    "Terminal",
    "TerminalSize",
    "Text",
    "TruncatedText",
    "truncate_to_width",
    "visible_width",
]
