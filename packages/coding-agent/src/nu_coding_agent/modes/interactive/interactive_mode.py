"""Interactive REPL for ``nu`` — Phase 5.7 (streaming + markdown + compaction + extensions).

Textual-based terminal UI. Improvements over the Phase 5.6 skeleton:

* **Streaming text deltas** — assistant text renders live as the LLM
  produces it via ``message_update`` events carrying ``text_delta``.
* **Rich Markdown rendering** — each completed assistant message is
  re-rendered through Rich Markdown for headings, code blocks (with
  syntax highlighting), bold, italic, lists, etc.
* **Auto-compaction** — after each turn, ``should_compact()`` is
  checked; when true, ``session.compact()`` runs in the background
  and a ``[compacted]`` indicator is shown.
* **Extension hooks** — the CLI wiring creates the AgentSession with
  an ``extension_runner`` when extensions are loaded, so lifecycle
  hooks (session_start, agent_start/end, message_start/update/end,
  tool_execution_start/end, session_shutdown) fire during interactive
  turns.
* **Slash commands** — ``/exit``, ``/model``, ``/compact``, ``/clear``,
  ``/session``, ``/help`` (documented in the input placeholder).
* **Thinking indicator** — a ``⠋ Thinking...`` Loader widget appears
  while the agent is streaming and disappears when the turn ends.

Still deferred:

* nu_tui component tree wiring (renders via Textual widgets directly).
* Session selectors, model pickers, theme pickers, settings panels.
* Overlay dialogs (login, extensions).
* Full slash command expansion from upstream.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text as RichText
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

if TYPE_CHECKING:
    from nu_coding_agent.core.agent_session import AgentSession


class _MessageWidget(Static):
    """A single message in the conversation history."""

    DEFAULT_CSS = """
    _MessageWidget {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """


class _StreamingWidget(Static):
    """Widget that accumulates streaming text deltas and re-renders live."""

    DEFAULT_CSS = """
    _StreamingWidget {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._chunks: list[str] = []

    def append_delta(self, delta: str) -> None:
        """Append a text delta and refresh the display."""
        self._chunks.append(delta)
        self.update(RichText("".join(self._chunks)))

    def get_text(self) -> str:
        return "".join(self._chunks)

    def finalize_as_markdown(self) -> None:
        """Re-render the accumulated text as Rich Markdown."""
        text = self.get_text()
        if text.strip():
            self.update(RichMarkdown(text, code_theme="monokai"))


class InteractiveApp(App[None]):
    """Textual app for the interactive REPL."""

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
        self._streaming_widget: _StreamingWidget | None = None
        self._thinking_widget: Static | None = None

    def compose(self) -> ComposeResult:
        model = self._session.model
        model_name = model.id if model else "no model"
        yield Header(show_clock=False)
        yield VerticalScroll(id="message-area")
        yield Input(
            placeholder=f"Message ({model_name}) — /help for commands",
            id="prompt-input",
        )
        yield Footer()

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
            # Find and set the model
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
            self.theme = theme_name  # Textual's built-in theme switching
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

        self._add_message(f"> {text}", "user-message")
        self._run_prompt(text)

    # ------------------------------------------------------------------
    # Prompt execution with streaming
    # ------------------------------------------------------------------

    @work(thread=False)
    async def _run_prompt(self, text: str) -> None:
        self._working = True
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = True

        # Show thinking indicator
        self._show_thinking()

        # Create streaming widget for the assistant response
        self._streaming_widget = _StreamingWidget()
        self._streaming_widget.add_class("assistant-message")
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(self._streaming_widget)

        def listener(event: Any) -> None:
            event_type = event["type"]

            # Stream text deltas live
            if event_type == "message_update":
                inner = event.get("assistant_message_event")
                if inner is not None:
                    inner_type = getattr(inner, "type", None)
                    if inner_type == "text_delta" and self._streaming_widget is not None:
                        delta = getattr(inner, "delta", "")
                        if delta:
                            self.call_from_thread(self._streaming_widget.append_delta, delta)

            # Show tool calls
            if event_type == "tool_execution_end" and not self._quiet:
                name = event.get("tool_name", "?")
                is_error = event.get("is_error", False)
                status = "error" if is_error else "ok"
                self.call_from_thread(self._add_message, f"[tool {status}] {name}", "tool-message")

        unsub = self._session.subscribe(listener)

        try:
            await self._session.prompt(text)

            # Finalize: re-render as markdown
            if self._streaming_widget is not None:  # pyright: ignore[reportUnnecessaryComparison]
                self._streaming_widget.finalize_as_markdown()
                self._streaming_widget.scroll_visible()

            # Check for error stop reason
            messages = self._session.agent.state.messages
            if messages:
                last = messages[-1]
                if getattr(last, "stop_reason", None) == "error":
                    err = getattr(last, "error_message", None) or "unknown error"
                    self._add_message(f"[error] {err}", "error-message")

            # Auto-compaction check
            if self._session.should_compact():
                self._run_compact()

        except KeyboardInterrupt:
            self._add_message("[interrupted]", "error-message")
        except Exception as exc:
            self._add_message(f"[error] {exc}", "error-message")
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
                self._add_info(f"[compacted] {result.tokens_before} tokens → summarized")
        except Exception as exc:
            self._add_message(f"[compaction error] {exc}", "error-message")

    # ------------------------------------------------------------------
    # Thinking indicator
    # ------------------------------------------------------------------

    def _show_thinking(self) -> None:
        if self._thinking_widget is not None:
            return
        self._thinking_widget = Static("⠋ Thinking...")
        self._thinking_widget.add_class("thinking-indicator")
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(self._thinking_widget)
        self._thinking_widget.scroll_visible()

    def _hide_thinking(self) -> None:
        if self._thinking_widget is not None:
            self._thinking_widget.remove()
            self._thinking_widget = None

    # ------------------------------------------------------------------
    # Message rendering helpers
    # ------------------------------------------------------------------

    def _add_message(self, text: str, css_class: str) -> None:
        area = self.query_one("#message-area", VerticalScroll)
        content: Any = RichText.from_ansi(text) if "\033[" in text else text
        widget = _MessageWidget(content)
        widget.add_class(css_class)
        area.mount(widget)
        widget.scroll_visible()

    def _add_info(self, text: str) -> None:
        self._add_message(text, "info-message")


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
