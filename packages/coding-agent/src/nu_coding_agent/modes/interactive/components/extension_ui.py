"""Extension UI modal screens — port of ``extension-selector.ts``,
``extension-input.ts``, ``extension-editor.ts``.

Three Textual ``ModalScreen`` overlays that extensions can push to gather
input from the user:

- :class:`ExtensionSelectorScreen` — pick one option from a list
- :class:`ExtensionInputScreen` — single-line text input
- :class:`ExtensionEditorScreen` — multi-line text editor (with ``$EDITOR`` support)

Each screen accepts an optional *timeout* (seconds).  When the timeout
expires the screen auto-dismisses with ``None``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from typing import TYPE_CHECKING, Any

from textual import on
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static, TextArea

if TYPE_CHECKING:
    from textual.app import ComposeResult

from nu_coding_agent.modes.interactive.components.keybinding_hints import key_hint

# ---------------------------------------------------------------------------
# Extension Selector — list of string options
# ---------------------------------------------------------------------------


class ExtensionSelectorScreen(ModalScreen[str | None]):
    """Modal list-picker for extension-supplied options.

    Dismisses with the selected string or ``None`` on cancel/timeout.
    """

    CSS = """
    ExtensionSelectorScreen {
        align: center middle;
    }
    #ext-sel-box {
        width: 60;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #ext-sel-title {
        color: $accent;
        margin-bottom: 1;
    }
    #ext-sel-hints {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        title: str,
        options: list[str],
        *,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._options = options
        self._timeout = timeout
        self._timeout_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ext-sel-box"):
            yield Label(self._title, id="ext-sel-title")
            items = [ListItem(Label(opt), name=opt) for opt in self._options]
            if items:
                yield ListView(*items, id="ext-sel-list")
            else:
                yield Label("  (no options)")
            hints = (
                "↑↓ navigate  "
                + key_hint("tui.select.confirm", "select")
                + "  "
                + key_hint("tui.select.cancel", "cancel")
            )
            yield Static(hints, id="ext-sel-hints")

    def on_mount(self) -> None:
        if self._timeout and self._timeout > 0:
            self._timeout_task = asyncio.get_event_loop().create_task(self._auto_cancel())

    async def _auto_cancel(self) -> None:
        for remaining in range(int(self._timeout or 0), 0, -1):  # type: ignore[arg-type]
            lbl = self.query_one("#ext-sel-title", Label)
            lbl.update(f"{self._title} ({remaining}s)")
            await asyncio.sleep(1)
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self._cancel_timeout()
        self.dismiss(event.item.name)

    def action_cancel(self) -> None:
        self._cancel_timeout()
        self.dismiss(None)

    def _cancel_timeout(self) -> None:
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None


# ---------------------------------------------------------------------------
# Extension Input — single-line text prompt
# ---------------------------------------------------------------------------


class ExtensionInputScreen(ModalScreen[str | None]):
    """Modal single-line text input for extensions.

    Dismisses with the entered string or ``None`` on cancel/timeout.
    """

    CSS = """
    ExtensionInputScreen {
        align: center middle;
    }
    #ext-inp-box {
        width: 60;
        max-height: 20;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #ext-inp-title {
        color: $accent;
        margin-bottom: 1;
    }
    #ext-inp-hints {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        title: str,
        placeholder: str = "",
        *,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._placeholder = placeholder
        self._timeout = timeout
        self._timeout_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ext-inp-box"):
            yield Label(self._title, id="ext-inp-title")
            yield Input(placeholder=self._placeholder, id="ext-inp-input")
            hints = key_hint("tui.select.confirm", "submit") + "  " + key_hint("tui.select.cancel", "cancel")
            yield Static(hints, id="ext-inp-hints")

    def on_mount(self) -> None:
        self.query_one("#ext-inp-input", Input).focus()
        if self._timeout and self._timeout > 0:
            self._timeout_task = asyncio.get_event_loop().create_task(self._auto_cancel())

    async def _auto_cancel(self) -> None:
        for remaining in range(int(self._timeout or 0), 0, -1):  # type: ignore[arg-type]
            lbl = self.query_one("#ext-inp-title", Label)
            lbl.update(f"{self._title} ({remaining}s)")
            await asyncio.sleep(1)
        self.dismiss(None)

    @on(Input.Submitted, "#ext-inp-input")
    def _on_submit(self, event: Input.Submitted) -> None:
        self._cancel_timeout()
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self._cancel_timeout()
        self.dismiss(None)

    def _cancel_timeout(self) -> None:
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None


# ---------------------------------------------------------------------------
# Extension Editor — multi-line text area (+ optional $EDITOR support)
# ---------------------------------------------------------------------------


class ExtensionEditorScreen(ModalScreen[str | None]):
    """Modal multi-line editor for extensions.

    Ctrl+G opens ``$VISUAL`` / ``$EDITOR`` if set.  Dismisses with the
    edited text or ``None`` on cancel.
    """

    CSS = """
    ExtensionEditorScreen {
        align: center middle;
    }
    #ext-ed-box {
        width: 80;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #ext-ed-title {
        color: $accent;
        margin-bottom: 1;
    }
    #ext-ed-area {
        height: 10;
        border: solid $panel;
        margin-bottom: 1;
    }
    #ext-ed-hints {
        color: $text-muted;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
        ("ctrl+g", "open_external", "External editor"),
    ]

    def __init__(
        self,
        title: str,
        prefill: str = "",
    ) -> None:
        super().__init__()
        self._title = title
        self._prefill = prefill

    def compose(self) -> ComposeResult:
        has_ext = bool(os.environ.get("VISUAL") or os.environ.get("EDITOR"))
        hints = key_hint("tui.select.confirm", "submit") + "  " + key_hint("tui.select.cancel", "cancel")
        if has_ext:
            hints += "  Ctrl+G external editor"
        with VerticalScroll(id="ext-ed-box"):
            yield Label(self._title, id="ext-ed-title")
            yield TextArea(self._prefill, id="ext-ed-area")
            yield Static(hints, id="ext-ed-hints")

    def on_mount(self) -> None:
        self.query_one("#ext-ed-area", TextArea).focus()

    def on_key(self, event: Any) -> None:
        # Ctrl+Enter submits (Enter inserts newlines in TextArea)
        if getattr(event, "key", None) == "ctrl+enter":
            self._submit()
            event.stop()

    def _submit(self) -> None:
        text = self.query_one("#ext-ed-area", TextArea).text
        self.dismiss(text)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_open_external(self) -> None:
        """Open ``$VISUAL`` / ``$EDITOR`` on a temp file then load the result."""
        import subprocess  # noqa: PLC0415

        editor_cmd = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor_cmd:
            return
        area = self.query_one("#ext-ed-area", TextArea)
        current = area.text
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="pi-ext-editor-", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(current)
            tmp_path = fh.name
        try:
            parts = editor_cmd.split()
            with self.app.suspend():
                result = subprocess.run([*parts, tmp_path], check=False)
            if result.returncode == 0:
                with open(tmp_path, encoding="utf-8") as fh:
                    new_text = fh.read().rstrip("\n")
                area.load_text(new_text)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


__all__ = [
    "ExtensionEditorScreen",
    "ExtensionInputScreen",
    "ExtensionSelectorScreen",
]
