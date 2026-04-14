"""Textual modal screens for interactive-mode selectors.

Phase 5.8+: model picker, session list, theme switcher, settings panel,
fork selector, session-resume selector, OAuth provider selector.
These are presented as Textual ``ModalScreen`` overlays that dismiss
with Escape and are launched by slash commands in the REPL.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from textual import work
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, LoadingIndicator, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from nu_coding_agent.core.agent_session import AgentSession
    from nu_coding_agent.core.auth_storage import AuthStorage


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

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#model-list", ListView).focus()

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

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#session-list", ListView).focus()

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

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#theme-list", ListView).focus()

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


# ---------------------------------------------------------------------------
# Fork Selector — /fork
# ---------------------------------------------------------------------------


class ForkSelectorScreen(ModalScreen[tuple[str, str] | None]):
    """Modal that lists user messages and lets the user pick a fork point.

    Dismisses with ``(entry_id, text)`` on selection or ``None`` on cancel.
    """

    CSS = """
    ForkSelectorScreen {
        align: center middle;
    }
    #fork-box {
        width: 80;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, messages: list[dict[str, str]]) -> None:
        super().__init__()
        self._messages = messages

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="fork-box"):
            yield Label("Fork from message (Enter to select, Escape to cancel):")
            items = []
            for m in self._messages:
                entry_id = m.get("entryId", "")
                text = m.get("text", "")
                preview = text[:80].replace("\n", " ")
                items.append(ListItem(Label(preview), name=entry_id))
            if not items:
                yield Label("  No user messages to fork from.")
            else:
                yield ListView(*items, id="fork-list")

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#fork-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        entry_id = event.item.name or ""
        # Look up the text for this entry
        text = next((m.get("text", "") for m in self._messages if m.get("entryId") == entry_id), "")
        self.dismiss((entry_id, text))

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Resume Selector — /resume
# ---------------------------------------------------------------------------


class ResumeSelectorScreen(ModalScreen[str | None]):
    """Modal that lists recent sessions and lets the user switch to one.

    Performs async session discovery on mount and shows a spinner while
    loading. Dismisses with a session file path or ``None`` on cancel.
    """

    CSS = """
    ResumeSelectorScreen {
        align: center middle;
    }
    #resume-box {
        width: 80;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #resume-spinner {
        margin: 1 0;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, cwd: str, session_dir: str | None = None) -> None:
        super().__init__()
        self._cwd = cwd
        self._session_dir = session_dir

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="resume-box"):
            yield Label("Loading sessions...")
            yield LoadingIndicator(id="resume-spinner")

    def on_mount(self) -> None:
        self._load_sessions()

    @work
    async def _load_sessions(self) -> None:
        from nu_coding_agent.core.session_manager import SessionManager  # noqa: PLC0415

        sessions = await SessionManager.list(self._cwd, self._session_dir)
        self._render_sessions(sessions)

    def _render_sessions(self, sessions: list[Any]) -> None:
        box = self.query_one("#resume-box", VerticalScroll)
        box.remove_children()
        box.mount(Label("Select session (Enter to resume, Escape to cancel):"))
        if not sessions:
            box.mount(Label("  No sessions found."))
            return
        items = []
        for s in sessions:
            path = str(getattr(s, "path", s))
            name = getattr(s, "name", None) or ""
            modified = getattr(s, "modified", None)
            mod_str = ""
            if modified:
                import contextlib  # noqa: PLC0415
                import datetime  # noqa: PLC0415

                with contextlib.suppress(Exception):
                    mod_str = f" ({datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M')})"
            label = f"{name or path}{mod_str}" if name else f"{path}{mod_str}"
            items.append(ListItem(Label(label), name=path))
        lv = ListView(*items, id="resume-list")
        box.mount(lv)
        lv.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# OAuth Selector — /login, /logout
# ---------------------------------------------------------------------------


class OAuthSelectorScreen(ModalScreen[str | None]):
    """Modal that lists OAuth providers for login or logout.

    Dismisses with the selected ``provider_id`` or ``None`` on cancel.
    The ``mode`` parameter is ``"login"`` or ``"logout"``.
    """

    CSS = """
    OAuthSelectorScreen {
        align: center middle;
    }
    #oauth-box {
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

    def __init__(self, mode: str, auth_storage: AuthStorage) -> None:
        super().__init__()
        self._mode = mode
        self._auth_storage = auth_storage

    def compose(self) -> ComposeResult:
        verb = "Login to" if self._mode == "login" else "Logout from"
        all_providers = self._auth_storage.get_oauth_providers()

        if self._mode == "logout":
            # Only show providers that are currently logged in
            providers = [p for p in all_providers if self._auth_storage.get(p) is not None]
        else:
            providers = all_providers

        with VerticalScroll(id="oauth-box"):
            yield Label(f"{verb} provider (Enter to confirm, Escape to cancel):")
            if not providers:
                msg = "No OAuth providers logged in." if self._mode == "logout" else "No OAuth providers available."
                yield Label(f"  {msg}")
            else:
                items = [ListItem(Label(p), name=p) for p in providers]
                yield ListView(*items, id="oauth-list")

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#oauth-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Thinking Level Selector — /thinking
# ---------------------------------------------------------------------------

_THINKING_DESCRIPTIONS: dict[str, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning (~1k tokens)",
    "low": "Light reasoning (~2k tokens)",
    "medium": "Moderate reasoning (~8k tokens)",
    "high": "Deep reasoning (~16k tokens)",
    "xhigh": "Maximum reasoning (~32k tokens)",
}


class ThinkingSelectorScreen(ModalScreen[str | None]):
    """Modal that lists available thinking levels and returns the chosen one.

    Port of ``ThinkingSelectorComponent`` (thinking-selector.ts).
    """

    CSS = """
    ThinkingSelectorScreen {
        align: center middle;
    }
    #thinking-box {
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

    def __init__(self, current_level: str, available_levels: list[str]) -> None:
        super().__init__()
        self._current = current_level
        self._available = available_levels

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="thinking-box"):
            yield Label("Select thinking level (Enter to confirm, Escape to cancel):")
            items = []
            for level in self._available:
                marker = "* " if level == self._current else "  "
                desc = _THINKING_DESCRIPTIONS.get(level, "")
                label_text = f"{marker}{level}" + (f"  —  {desc}" if desc else "")
                items.append(ListItem(Label(label_text), name=level))
            yield ListView(*items, id="thinking-list")

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#thinking-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Show Images Selector — /images
# ---------------------------------------------------------------------------


class ShowImagesSelectorScreen(ModalScreen[bool | None]):
    """Modal that asks whether to show images inline.

    Port of ``ShowImagesSelectorComponent`` (show-images-selector.ts).
    """

    CSS = """
    ShowImagesSelectorScreen {
        align: center middle;
    }
    #images-box {
        width: 50;
        max-height: 40%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_value: bool) -> None:
        super().__init__()
        self._current = current_value

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="images-box"):
            yield Label("Show images inline? (Enter to confirm, Escape to cancel):")
            items = [
                ListItem(
                    Label(("* " if self._current else "  ") + "Yes  —  Show images inline in terminal"), name="yes"
                ),
                ListItem(Label(("  " if self._current else "* ") + "No   —  Show text placeholder instead"), name="no"),
            ]
            yield ListView(*items, id="images-list")

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#images-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name == "yes")

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = [
    "ForkSelectorScreen",
    "ModelPickerScreen",
    "OAuthSelectorScreen",
    "ResumeSelectorScreen",
    "SessionListScreen",
    "SettingsScreen",
    "ShowImagesSelectorScreen",
    "ThemeSwitcherScreen",
    "ThinkingSelectorScreen",
]
