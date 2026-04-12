"""Textual modal screens for interactive-mode selectors.

Phase 5.8: model picker, session list, theme switcher, settings panel.
These are presented as Textual ``ModalScreen`` overlays that dismiss
with Escape and are launched by slash commands in the REPL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from nu_coding_agent.core.agent_session import AgentSession


# ---------------------------------------------------------------------------
# Model Picker — /models
# ---------------------------------------------------------------------------


class ModelPickerScreen(ModalScreen[str | None]):
    """Modal list of available models. Returns the selected model id or None."""

    CSS = """
    ModelPickerScreen {
        align: center middle;
    }
    #model-picker-box {
        width: 60;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, session: AgentSession) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        models = self._session.model_registry.get_available_models()
        current = self._session.model
        current_id = current.id if current else ""

        with VerticalScroll(id="model-picker-box"):
            yield Label("Select a model (Enter to confirm, Escape to cancel):")
            items = []
            for m in models:
                marker = "* " if m.id == current_id else "  "
                items.append(ListItem(Label(f"{marker}{m.provider}/{m.id}"), name=m.id))
            yield ListView(*items, id="model-list")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        model_id = event.item.name
        self.dismiss(model_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Session List — /sessions
# ---------------------------------------------------------------------------


class SessionListScreen(ModalScreen[str | None]):
    """Modal list of sessions for the current cwd. Returns a session path or None."""

    CSS = """
    SessionListScreen {
        align: center middle;
    }
    #session-list-box {
        width: 70;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, sessions: list[dict[str, Any]]) -> None:
        super().__init__()
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="session-list-box"):
            yield Label("Sessions (Enter to open, Escape to cancel):")
            items = []
            for s in self._sessions:
                path = s.get("path", "?")
                first_msg = s.get("first_message", "")[:50]
                label = f"{path}\n  {first_msg}" if first_msg else path
                items.append(ListItem(Label(label), name=path))
            if not items:
                yield Label("  No sessions found.")
            else:
                yield ListView(*items, id="session-list")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Theme Switcher — /theme
# ---------------------------------------------------------------------------


class ThemeSwitcherScreen(ModalScreen[str | None]):
    """Modal that lets the user pick dark or light theme."""

    CSS = """
    ThemeSwitcherScreen {
        align: center middle;
    }
    #theme-box {
        width: 40;
        max-height: 60%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="theme-box"):
            yield Label("Select theme:")
            yield ListView(
                ListItem(Label("dark"), name="dark"),
                ListItem(Label("light"), name="light"),
                id="theme-list",
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Settings Panel — /settings
# ---------------------------------------------------------------------------


class SettingsScreen(ModalScreen[None]):
    """Read-only settings panel showing the current session configuration."""

    CSS = """
    SettingsScreen {
        align: center middle;
    }
    #settings-box {
        width: 60;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "dismiss_screen", "Close"),
    ]

    def __init__(self, session: AgentSession) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        model = self._session.model
        sm = self._session.session_manager

        lines = [
            f"Model:    {model.id if model else 'none'}",
            f"Provider: {model.provider if model else 'none'}",
            f"CWD:      {self._session.cwd}",
            f"Session:  {sm.get_session_file() or '(in-memory)'}",
            f"Entries:  {len(sm.get_entries())}",
        ]
        runner = self._session.extension_runner
        if runner is not None:
            lines.append(f"Extensions: {len(runner.extensions)}")

        with VerticalScroll(id="settings-box"):
            yield Label("Settings")
            yield Static("\n".join(lines))

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)


__all__ = [
    "ModelPickerScreen",
    "SessionListScreen",
    "SettingsScreen",
    "ThemeSwitcherScreen",
]
