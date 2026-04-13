"""Interactive REPL for ``nu`` — Phase 5.10 (tool-aware widget rendering).

The interactive mode renders through the nu_tui component library:

* **Assistant messages** stream via ``_StreamingComponentWidget``.
* **User messages** use ``UserMessageWidget``.
* **Bash executions** use ``BashExecutionWidget`` — created at
  ``tool_execution_start``, updated with streaming output on
  ``tool_execution_update``, finalized on ``tool_execution_end``.
* **Other tool calls** use ``ToolExecutionWidget`` — created at
  ``tool_execution_start``, finalized with result on
  ``tool_execution_end``.
* **Compaction events** use ``CompactionSummaryWidget``.

Widgets are tracked in ``_tool_widgets`` keyed by ``tool_call_id``
so concurrent tool calls each update their own widget.

The Textual layout (VerticalScroll, Header, Footer, Input,
ModalScreen) handles scrolling, focus, and modals.
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

from nu_coding_agent.modes.interactive.components.custom_message import CustomMessageWidget
from nu_coding_agent.modes.interactive.components.message_renderers import (
    BashExecutionWidget,
    BranchSummaryWidget,
    CompactionSummaryWidget,
    ErrorWidget,
    InfoWidget,
    ToolExecutionWidget,
    UserMessageWidget,
)
from nu_coding_agent.modes.interactive.components.skill_invocation_message import (
    ParsedSkillBlock,
    SkillInvocationWidget,
)
from nu_coding_agent.modes.interactive.footer import InteractiveFooter

if TYPE_CHECKING:
    from nu_coding_agent.core.agent_session import AgentSession

_BASH_TOOL_NAMES = {"bash", "Bash"}


class _StreamingComponentWidget(ComponentWidget):
    """ComponentWidget that accumulates streaming text deltas."""

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
        self._chunks.append(delta)
        self._text_component.set_text("".join(self._chunks))
        self.refresh_component()

    def get_text(self) -> str:
        return "".join(self._chunks)

    def finalize_as_markdown(self) -> None:
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
        # Widgets keyed by tool_call_id for concurrent tool tracking
        self._tool_widgets: dict[str, ToolExecutionWidget | BashExecutionWidget] = {}

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
        self._restore_history()
        inp = self.query_one("#prompt-input", Input)
        inp.focus()

    def _restore_history(self) -> None:
        """Render previously-persisted session entries on app start.

        Mirrors the upstream's ``addMessageToChat`` loop that runs on
        ``TUI.run()`` to populate the chat container from the session
        branch before the first prompt.
        """
        area = self.query_one("#message-area", VerticalScroll)
        sm = self._session.session_manager
        for entry in sm.get_branch():
            widget = _widget_for_entry(entry, self._session)
            if widget is not None:
                area.mount(widget)

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
    # Prompt execution
    # ------------------------------------------------------------------

    @work(thread=False)
    async def _run_prompt(self, text: str) -> None:
        self._working = True
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = True

        # Show thinking indicator
        self._show_thinking()

        # Create streaming assistant message widget
        assistant_widget = _StreamingComponentWidget()
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(assistant_widget)

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

            elif event_type == "message_start":
                msg = event.get("message")
                role = getattr(msg, "role", None)
                if role == "custom":
                    self.call_from_thread(self._mount_custom_message, msg)

            elif event_type == "tool_execution_start" and not self._quiet:
                tool_name = str(event.get("tool_name", ""))
                args: dict[str, Any] = event.get("args") or {}
                tool_call_id = str(event.get("tool_call_id", ""))
                self.call_from_thread(self._on_tool_start, tool_call_id, tool_name, args)

            elif event_type == "tool_execution_update" and not self._quiet:
                tool_call_id = str(event.get("tool_call_id", ""))
                partial = event.get("partial_result")
                self.call_from_thread(self._on_tool_update, tool_call_id, partial)

            elif event_type == "tool_execution_end" and not self._quiet:
                tool_call_id = str(event.get("tool_call_id", ""))
                is_error = bool(event.get("is_error", False))
                result = event.get("result")
                self.call_from_thread(self._on_tool_end, tool_call_id, is_error, result)

        unsub = self._session.subscribe(listener)

        try:
            await self._session.prompt(text)

            assistant_widget.finalize_as_markdown()
            assistant_widget.scroll_visible()

            messages = self._session.agent.state.messages
            if messages:
                last = messages[-1]
                if getattr(last, "stop_reason", None) == "error":
                    err = getattr(last, "error_message", None) or "unknown error"
                    self._mount_error(err)

            # Update footer
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
            self._tool_widgets.clear()
            self._hide_thinking()
            self._working = False
            inp.disabled = False
            inp.focus()

    # ------------------------------------------------------------------
    # Tool widget management (called from main thread via call_from_thread)
    # ------------------------------------------------------------------

    def _on_tool_start(self, tool_call_id: str, tool_name: str, args: dict[str, Any]) -> None:
        area = self.query_one("#message-area", VerticalScroll)
        if tool_name in _BASH_TOOL_NAMES:
            command = str(args.get("command", ""))
            widget: ToolExecutionWidget | BashExecutionWidget = BashExecutionWidget(command)
        else:
            widget = ToolExecutionWidget(tool_name, args)
        key = tool_call_id or tool_name
        self._tool_widgets[key] = widget
        area.mount(widget)
        widget.scroll_visible()

    def _on_tool_update(self, tool_call_id: str, partial_result: Any) -> None:
        key = tool_call_id
        widget = self._tool_widgets.get(key)
        if widget is None:
            return
        if not isinstance(widget, BashExecutionWidget):
            return
        # partial_result is an AgentToolResult; extract cumulative text
        output = _extract_text_content(partial_result)
        if output:
            widget.set_output(output)

    def _on_tool_end(self, tool_call_id: str, is_error: bool, result: Any) -> None:
        key = tool_call_id
        widget = self._tool_widgets.get(key)
        if widget is None:
            return
        if isinstance(widget, BashExecutionWidget):
            # Extract exit code from result details if available
            exit_code: int | None = None
            if result is not None:
                details = getattr(result, "details", None) or {}
                if isinstance(details, dict):
                    exit_code = details.get("exit_code")
            widget.set_complete(exit_code, cancelled=False)
        else:
            content = _extract_content_list(result)
            widget.set_result(content, is_error=is_error)

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
    # Thinking indicator
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
    # Helpers for mounting widgets
    # ------------------------------------------------------------------

    def _mount_custom_message(self, message: Any) -> None:
        """Mount a CustomMessageWidget for an extension-injected message."""
        from nu_coding_agent.core.messages import CustomMessage  # noqa: PLC0415

        if not isinstance(message, CustomMessage) or not message.display:
            return
        renderer = None
        if self._session.extension_runner is not None:
            renderer = self._session.extension_runner.get_message_renderer(message.custom_type)
        widget = CustomMessageWidget(message, renderer=renderer)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()

    def _add_user_message(self, text: str) -> None:
        widget = UserMessageWidget(text)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()

    def _add_info(self, text: str) -> None:
        widget = InfoWidget(text)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()

    def _mount_error(self, message: str) -> None:
        widget = ErrorWidget(message)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_skill_re: object = None  # lazy-compiled regex


def _parse_skill_block_from_text(text: str) -> ParsedSkillBlock | None:
    """Parse a ``<skill …>`` block from a user message string."""
    import re  # noqa: PLC0415

    global _skill_re  # noqa: PLW0603
    if _skill_re is None:
        _skill_re = re.compile(r'^<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n</skill>(?:\n\n([\s\S]+))?$')
    m = _skill_re.match(text)  # type: ignore[union-attr]
    if not m:
        return None
    return ParsedSkillBlock(
        name=m.group(1),
        location=m.group(2),
        content=m.group(3),
        user_message=m.group(4).strip() if m.group(4) else None,
    )


def _widget_for_entry(entry: dict[str, Any], session: Any) -> Any:
    """Return the appropriate widget for a session entry, or ``None``.

    Mirrors the upstream ``addMessageToChat`` / ``renderMessage`` dispatch.
    """
    from nu_ai import AssistantMessage, TextContent, UserMessage  # noqa: PLC0415

    from nu_coding_agent.core.messages import (  # noqa: PLC0415
        BashExecutionMessage,
        BranchSummaryMessage,
        CompactionSummaryMessage,
        CustomMessage,
    )
    from nu_coding_agent.modes.interactive.components.message_renderers import (  # noqa: PLC0415
        AssistantMessageWidget as _AMW,
    )

    entry_type = entry.get("type")
    message = entry.get("message")

    if entry_type == "compaction":
        summary = entry.get("summary", "")
        tokens_before = int(entry.get("tokensBefore") or entry.get("tokens_before") or 0)
        return CompactionSummaryWidget(summary, tokens_before)

    if entry_type == "branch_summary":
        summary = entry.get("summary", "")
        from_id = entry.get("fromId") or entry.get("from_id") or ""
        return BranchSummaryWidget(summary, from_id)

    if entry_type == "custom":
        msg = message
        if isinstance(msg, dict):
            import time as _time  # noqa: PLC0415

            from nu_coding_agent.core.messages import CustomMessage as _CM  # noqa: PLC0415

            msg = _CM(
                role="custom",
                custom_type=msg.get("customType") or msg.get("custom_type", ""),
                content=msg.get("content", ""),
                display=bool(msg.get("display", True)),
                details=msg.get("details"),
                timestamp=int(_time.time() * 1000),
            )
        if not isinstance(msg, CustomMessage) or not msg.display:
            return None
        renderer = None
        runner = getattr(session, "extension_runner", None)
        if runner is not None:
            renderer = runner.get_message_renderer(msg.custom_type)
        return CustomMessageWidget(msg, renderer=renderer)

    if entry_type != "message" or message is None:
        return None

    # Deserialize message dict if needed
    if isinstance(message, dict):
        role = message.get("role")
        if role == "bashExecution":
            cmd = message.get("command", "")
            output = message.get("output", "")
            exit_code = message.get("exitCode")
            cancelled = bool(message.get("cancelled", False))
            widget = BashExecutionWidget(cmd)
            if output:
                widget.set_output(output)
            widget.set_complete(exit_code, cancelled=cancelled)
            return widget
        if role == "user":
            content = message.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content = block.get("text", "")
                        break
                else:
                    content = ""
            text = str(content)
            skill = _parse_skill_block_from_text(text)
            if skill:
                return SkillInvocationWidget(skill)
            return UserMessageWidget(text)
        if role == "assistant":
            content_blocks = message.get("content", [])
            text = ""
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
            if text.strip():
                w = _AMW()
                # Append then finalize to render as Markdown
                w.append_delta(text)
                w.finalize()
                return w
        return None

    # Pydantic/dataclass messages
    if isinstance(message, BashExecutionMessage):
        widget = BashExecutionWidget(message.command)
        if message.output:
            widget.set_output(message.output)
        widget.set_complete(message.exit_code, cancelled=message.cancelled)
        return widget

    if isinstance(message, (CompactionSummaryMessage,)):
        return CompactionSummaryWidget(message.summary, message.tokens_before)

    if isinstance(message, BranchSummaryMessage):
        return BranchSummaryWidget(message.summary, message.from_id)

    if isinstance(message, CustomMessage):
        if not message.display:
            return None
        runner = getattr(session, "extension_runner", None)
        renderer = runner.get_message_renderer(message.custom_type) if runner else None
        return CustomMessageWidget(message, renderer=renderer)

    if isinstance(message, UserMessage):
        text = ""
        for block in message.content:
            if isinstance(block, TextContent):
                text += block.text
        skill = _parse_skill_block_from_text(text)
        if skill:
            return SkillInvocationWidget(skill)
        return UserMessageWidget(text)

    if isinstance(message, AssistantMessage):
        text = ""
        for block in message.content:
            if isinstance(block, TextContent):
                text += block.text
        if text.strip():
            w = _AMW()
            w.append_delta(text)
            w.finalize()
            return w

    return None


def _extract_text_content(result: Any) -> str:
    """Extract concatenated text from an AgentToolResult or similar."""
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _extract_content_list(result: Any) -> list[Any]:
    """Return the content list from an AgentToolResult."""
    if result is None:
        return []
    content = getattr(result, "content", None)
    if isinstance(content, list):
        return content
    return []


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
