"""Minimum viable interactive REPL — Phase 5.6 skeleton.

This is **not** a port of the upstream 4749-LoC ``interactive-mode.ts``.
It's a pragmatic skeleton that gives ``nu`` a working interactive mode
backed by Textual, so the full rendering pipeline and component library
from nu_tui can be layered on iteratively.

What works in this skeleton:

* A Textual app with a header (model info), a scrollable message area,
  and a text input at the bottom.
* User submits a prompt → ``AgentSession.prompt()`` streams events →
  assistant text is rendered as a new message widget → input is
  re-enabled.
* Tool calls are shown as ``[tool] name(args)`` lines.
* Errors surface as red text in the message area.
* Ctrl-C / ``/exit`` exits the app.
* ``/model`` shows the current model.

What's deferred to follow-up slices:

* The nu_tui component tree (upstream renders via Container/Component;
  this skeleton renders via Textual widgets directly).
* Session selectors, theme selectors, model pickers.
* Overlay dialogs (login, extensions, settings).
* Slash command expansion (only ``/exit`` and ``/model`` work).
* Markdown rendering of assistant messages (currently plain text with
  a dim style; Rich Markdown can be layered in).
* Streaming text deltas (currently waits for the full response).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

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
    """

    BINDINGS = [  # noqa: RUF012 — Textual convention
        ("ctrl+c", "quit", "Exit"),
    ]

    def __init__(self, session: AgentSession, quiet: bool = False) -> None:
        super().__init__()
        self._session = session
        self._quiet = quiet
        self._working = False

    def compose(self) -> ComposeResult:
        model = self._session.model
        model_name = model.id if model else "no model"
        yield Header(show_clock=False)
        yield VerticalScroll(id="message-area")
        yield Input(placeholder=f"Message ({model_name})...", id="prompt-input")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"cwd: {self._session.cwd}"
        inp = self.query_one("#prompt-input", Input)
        inp.focus()

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        # Slash commands
        if text == "/exit":
            self.exit()
            return
        if text == "/model":
            model = self._session.model
            self._add_message(f"[model] {model.id if model else 'none'}", "tool-message")
            event.input.value = ""
            return

        if self._working:
            return

        event.input.value = ""
        self._add_message(f"> {text}", "user-message")
        self._run_prompt(text)

    @work(thread=False)
    async def _run_prompt(self, text: str) -> None:
        self._working = True
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = True

        # Collect events for display
        events_text: list[str] = []

        def listener(event: Any) -> None:
            event_type = event["type"]
            if event_type == "tool_execution_end" and not self._quiet:
                name = event.get("tool_name", "?")
                is_error = event.get("is_error", False)
                status = "error" if is_error else "ok"
                events_text.append(f"[tool {status}] {name}")

        unsub = self._session.subscribe(listener)

        try:
            await self._session.prompt(text)

            # Display tool calls
            for tool_line in events_text:
                self._add_message(tool_line, "tool-message")

            # Display assistant response
            messages = self._session.agent.state.messages
            if messages:
                last = messages[-1]
                role = getattr(last, "role", None)
                if role == "assistant":
                    content = getattr(last, "content", [])
                    text_parts = []
                    for block in content:
                        block_type = getattr(block, "type", None)
                        if block_type == "text":
                            text_parts.append(getattr(block, "text", ""))
                    if text_parts:
                        self._add_message("\n".join(text_parts), "assistant-message")
                    stop_reason = getattr(last, "stop_reason", None)
                    if stop_reason == "error":
                        err = getattr(last, "error_message", None) or "unknown error"
                        self._add_message(f"[error] {err}", "error-message")

        except KeyboardInterrupt:
            self._add_message("[interrupted]", "error-message")
        except Exception as exc:
            self._add_message(f"[error] {exc}", "error-message")
        finally:
            unsub()
            self._working = False
            inp.disabled = False
            inp.focus()

    def _add_message(self, text: str, css_class: str) -> None:
        area = self.query_one("#message-area", VerticalScroll)
        widget = _MessageWidget(RichText.from_ansi(text) if "\033[" in text else text)
        widget.add_class(css_class)
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
