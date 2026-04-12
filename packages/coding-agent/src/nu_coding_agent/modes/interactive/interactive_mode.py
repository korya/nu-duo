"""Interactive REPL for ``nu`` — Phase 5.9 (nu_tui component tree wiring).

The interactive mode now renders through the nu_tui component library:

* **Assistant messages** render via ``nu_tui.components.Markdown``,
  hosted in a ``ComponentWidget``. During streaming, the widget
  re-renders on each text delta; on completion it finalizes as
  Rich Markdown with syntax-highlighted code blocks.
* **User messages** render via ``nu_tui.components.Text`` in a
  ``ComponentWidget``.
* **Tool calls** and **info messages** render via ``nu_tui.components.Text``.
* **Thinking indicator** renders via ``nu_tui.components.Loader``
  in a ``ComponentWidget`` with animated spinner.

The Textual layout system (VerticalScroll, Header, Footer, Input,
ModalScreen) is preserved — it handles scrolling, focus, and modals.
The nu_tui components are bridged into Textual via ``ComponentWidget``,
which calls ``component.render(width)`` on each paint cycle. This
hybrid approach gets both Textual's robust layout AND upstream-shaped
component rendering.

The remaining Textual-native widget is the prompt ``Input`` at the
bottom. Replacing it with a ``ComponentWidget(Editor(...))`` is
possible but requires non-trivial focus/key routing work — deferred
to a follow-up when the full Editor integration is needed (e.g. for
multi-line prompt entry with Shift+Enter).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from nu_tui.component_widget import ComponentWidget
from nu_tui.components import Markdown as NuMarkdown
from nu_tui.components import Text as NuText
from nu_tui.components.loader import Loader as NuLoader
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Input

from nu_coding_agent.modes.interactive.components.message_renderers import (
    AssistantMessageWidget,
    CompactionSummaryWidget,
    ErrorWidget,
    InfoWidget,
    ToolExecutionWidget,
    UserMessageWidget,
)
from nu_coding_agent.modes.interactive.footer import InteractiveFooter

if TYPE_CHECKING:
    from nu_coding_agent.core.agent_session import AgentSession


class _StreamingComponentWidget(ComponentWidget):
    """ComponentWidget that accumulates streaming text deltas.

    During streaming, the wrapped ``Text`` component is updated with
    each delta. On finalization, the component is swapped to a
    ``Markdown`` for rich rendering.
    """

    DEFAULT_CSS = """
    _StreamingComponentWidget {
        width: 1fr;
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self) -> None:
        self._text_component = NuText("")
        super().__init__(self._text_component)
        self._chunks: list[str] = []

    def append_delta(self, delta: str) -> None:
        """Append a text delta and refresh."""
        self._chunks.append(delta)
        self._text_component.set_text("".join(self._chunks))
        self.refresh_component()

    def get_text(self) -> str:
        return "".join(self._chunks)

    def finalize_as_markdown(self) -> None:
        """Swap the Text component for a Markdown component."""
        text = self.get_text()
        if text.strip():
            md = NuMarkdown(text)
            self.set_component(md)


class InteractiveApp(App[None]):
    """Textual app for the interactive REPL with nu_tui component rendering."""

    TITLE = "nu"
    CSS = """
    Screen {
        layout: vertical;
    }
    #message-area {
        height: 1fr;
    }
    #prompt-input {
        dock: bottom;
        height: 3;
        border-top: tall $accent;
    }
    ComponentWidget {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    .user-message {
        background: $primary-background-darken-2;
        color: $text;
    }
    .assistant-message {
        color: $text-muted;
    }
    .tool-message {
        color: $text-muted;
    }
    .error-message {
        color: red;
    }
    .info-message {
        color: $text-muted;
    }
    .thinking-indicator {
        color: $text-muted;
    }
    """

    BINDINGS = [  # noqa: RUF012 — Textual convention
        ("ctrl+c", "quit", "Exit"),
    ]

    def __init__(self, session: AgentSession, quiet: bool = False) -> None:
        super().__init__()
        self._session = session
        self._quiet = quiet
        self._working = False
        self._streaming_widget: _StreamingComponentWidget | None = None
        self._thinking_widget: ComponentWidget | None = None

    def compose(self) -> ComposeResult:
        model = self._session.model
        model_name = model.id if model else "no model"
        yield Header(show_clock=False)
        yield VerticalScroll(id="message-area")
        yield Input(
            placeholder=f"Message ({model_name}) — /help for commands",
            id="prompt-input",
        )
        yield InteractiveFooter(self._session)

    def on_mount(self) -> None:
        self.sub_title = f"cwd: {self._session.cwd}"
        inp = self.query_one("#prompt-input", Input)
        inp.focus()

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    _HELP_TEXT = """\
Available commands:
  /help      — show this help
  /exit      — exit the REPL
  /model     — show the current model
  /models    — pick a model from available models
  /compact   — trigger context compaction
  /clear     — clear the message area
  /session   — show session file path
  /sessions  — list and switch sessions
  /settings  — show current settings
  /theme     — switch dark/light theme
"""

    def _handle_slash_command(self, text: str) -> bool:
        """Process a slash command. Returns True if handled."""
        cmd = text.split(maxsplit=1)[0].lower() if text else ""
        if cmd == "/exit":
            self.exit()
            return True
        if cmd == "/help":
            self._add_info(self._HELP_TEXT)
            return True
        if cmd == "/model":
            model = self._session.model
            self._add_info(f"Model: {model.id if model else 'none'}")
            return True
        if cmd == "/models":
            self._show_model_picker()
            return True
        if cmd == "/compact":
            self._run_compact()
            return True
        if cmd == "/clear":
            area = self.query_one("#message-area", VerticalScroll)
            area.remove_children()
            return True
        if cmd == "/session":
            sf = self._session.session_manager.get_session_file()
            self._add_info(f"Session: {sf or '(in-memory)'}")
            return True
        if cmd == "/sessions":
            self._show_session_list()
            return True
        if cmd == "/settings":
            self._show_settings()
            return True
        if cmd == "/theme":
            self._show_theme_picker()
            return True
        return False

    # ------------------------------------------------------------------
    # Selector screens
    # ------------------------------------------------------------------

    def _show_model_picker(self) -> None:
        from nu_coding_agent.modes.interactive.selectors import ModelPickerScreen  # noqa: PLC0415

        def on_dismiss(model_id: str | None) -> None:
            if model_id is None:
                return
            model = self._session.model_registry.find_by_id(model_id)
            if model is not None:
                self._session.set_model(model)
                self._add_info(f"Switched to {model.provider}/{model.id}")
            else:
                self._add_info(f"Model {model_id} not found in registry")

        self.push_screen(ModelPickerScreen(self._session), on_dismiss)

    def _show_session_list(self) -> None:
        from nu_coding_agent.modes.interactive.selectors import SessionListScreen  # noqa: PLC0415

        sm = self._session.session_manager
        entries = sm.get_entries()
        sessions = [
            {
                "path": sm.get_session_file() or "(current)",
                "first_message": entries[0].get("message", {}).get("content", "")[:80] if entries else "",
            }
        ]
        self.push_screen(SessionListScreen(sessions), lambda _path: None)

    def _show_settings(self) -> None:
        from nu_coding_agent.modes.interactive.selectors import SettingsScreen  # noqa: PLC0415

        self.push_screen(SettingsScreen(self._session))

    def _show_theme_picker(self) -> None:
        from nu_coding_agent.modes.interactive.selectors import ThemeSwitcherScreen  # noqa: PLC0415

        def on_dismiss(theme_name: str | None) -> None:
            if theme_name is None:
                return
            self.theme = theme_name
            self._add_info(f"Theme: {theme_name}")

        self.push_screen(ThemeSwitcherScreen(), on_dismiss)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        event.input.value = ""

        if text.startswith("/") and self._handle_slash_command(text):
            return

        if self._working:
            return

        self._add_user_message(f"> {text}")
        self._run_prompt(text)

    # ------------------------------------------------------------------
    # Prompt execution with streaming via nu_tui components
    # ------------------------------------------------------------------

    @work(thread=False)
    async def _run_prompt(self, text: str) -> None:
        self._working = True
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = True

        # Show thinking indicator via nu_tui Loader
        self._show_thinking()

        # Create streaming assistant message widget
        assistant_widget = AssistantMessageWidget()
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(assistant_widget)
        self._streaming_widget = assistant_widget  # type: ignore[assignment]

        def listener(event: Any) -> None:
            event_type = event["type"]
            if event_type == "message_update":
                inner = event.get("assistant_message_event")
                if inner is not None:
                    inner_type = getattr(inner, "type", None)
                    if inner_type == "text_delta":
                        delta = getattr(inner, "delta", "")
                        if delta:
                            self.call_from_thread(assistant_widget.append_delta, delta)
            if event_type == "tool_execution_end" and not self._quiet:
                name = event.get("tool_name", "?")
                is_error = event.get("is_error", False)
                self.call_from_thread(self._mount_tool_widget, name, is_error)

        unsub = self._session.subscribe(listener)

        try:
            await self._session.prompt(text)

            assistant_widget.finalize()
            assistant_widget.scroll_visible()

            messages = self._session.agent.state.messages
            if messages:
                last = messages[-1]
                if getattr(last, "stop_reason", None) == "error":
                    err = getattr(last, "error_message", None) or "unknown error"
                    self._mount_error(err)

            # Update footer with new entry count
            try:
                footer = self.query_one(InteractiveFooter)
                footer.refresh_content()
            except Exception:
                pass

            if self._session.should_compact():
                self._run_compact()

        except KeyboardInterrupt:
            self._mount_error("interrupted")
        except Exception as exc:
            self._mount_error(str(exc))
        finally:
            unsub()
            self._streaming_widget = None
            self._hide_thinking()
            self._working = False
            inp.disabled = False
            inp.focus()

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    @work(thread=False)
    async def _run_compact(self) -> None:
        try:
            result = await self._session.compact()
            if result is not None:
                widget = CompactionSummaryWidget(result.summary, result.tokens_before)
                area = self.query_one("#message-area", VerticalScroll)
                area.mount(widget)
                widget.scroll_visible()
        except Exception as exc:
            self._mount_error(f"compaction: {exc}")

    # ------------------------------------------------------------------
    # Thinking indicator — nu_tui Loader component
    # ------------------------------------------------------------------

    def _show_thinking(self) -> None:
        if self._thinking_widget is not None:
            return
        loader = NuLoader(message="Thinking...")
        self._thinking_widget = ComponentWidget(loader, classes="thinking-indicator")
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(self._thinking_widget)
        self._thinking_widget.scroll_visible()

    def _hide_thinking(self) -> None:
        if self._thinking_widget is not None:
            component = self._thinking_widget.component
            if isinstance(component, NuLoader):
                component.stop()
            self._thinking_widget.remove()
            self._thinking_widget = None

    # ------------------------------------------------------------------
    # Message rendering via specialized widgets
    # ------------------------------------------------------------------

    def _add_user_message(self, text: str) -> None:
        """Add a user message via UserMessageWidget."""
        widget = UserMessageWidget(text)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()

    def _mount_tool_widget(self, tool_name: str, is_error: bool) -> None:
        """Add a tool execution widget."""
        widget = ToolExecutionWidget(tool_name, is_error=is_error)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()

    def _add_info(self, text: str) -> None:
        """Add an informational message."""
        widget = InfoWidget(text)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()

    def _mount_error(self, message: str) -> None:
        """Add an error message."""
        widget = ErrorWidget(message)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()


async def run_interactive_mode(session: AgentSession, *, quiet: bool = False) -> int:
    """Launch the interactive REPL.

    Returns an exit code (0 for clean exit, 1 for error).
    """
    app = InteractiveApp(session, quiet=quiet)
    try:
        await app.run_async()
    except Exception as exc:
        print(f"Interactive mode error: {exc}", file=sys.stderr)
        return 1
    return 0
