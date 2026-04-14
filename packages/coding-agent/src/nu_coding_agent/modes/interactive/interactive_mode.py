"""Interactive REPL for ``nu`` — Phase 5.11 (full slash-command surface).

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
* **Slash commands** — full parity with the upstream TS surface
  (``/export``, ``/copy``, ``/name``, ``/changelog``, ``/hotkeys``,
  ``/fork``, ``/resume``, ``/login``, ``/logout``, ``/new``,
  ``/reload``, ``/debug``, ``/quit``).
* **Bash prefix** — ``!command`` and ``!!command`` run ad-hoc shell
  commands outside the agent loop.

Widgets are tracked in ``_tool_widgets`` keyed by ``tool_call_id``
so concurrent tool calls each update their own widget.

The Textual layout (VerticalScroll, Header, Footer, Input,
ModalScreen) handles scrolling, focus, and modals.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_tui.component_widget import ComponentWidget
from nu_tui.components import Markdown as NuMarkdown
from nu_tui.components.loader import Loader as NuLoader
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Input, Static

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


class _ThinkingBlockWidget(Static):
    """Collapsible widget showing the model's thinking/reasoning text."""

    DEFAULT_CSS = """
    _ThinkingBlockWidget {
        width: 1fr;
        height: auto;
        padding: 0 1;
        margin: 0 0 0 0;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._chunks: list[str] = []
        self._collapsed = True
        self._finalized = False
        self._update_display()

    def append_delta(self, delta: str) -> None:
        self._chunks.append(delta)
        # Show expanded while streaming so the user sees thinking live
        self._collapsed = False
        self._update_display()

    def finalize(self) -> None:
        self._finalized = True
        # Collapse after streaming finishes
        self._collapsed = True
        self._update_display()

    def on_click(self) -> None:
        if self._finalized:
            self._collapsed = not self._collapsed
            self._update_display()

    def _update_display(self) -> None:
        from rich.text import Text as RichText  # noqa: PLC0415

        text = "".join(self._chunks)
        t = RichText()
        if self._collapsed:
            line_count = text.count("\n") + 1 if text else 0
            t.append(f"💭 Thinking ({line_count} lines) ", style="dim italic")
            if self._finalized:
                t.append("(click to expand)", style="dim")
        else:
            t.append("💭 Thinking\n", style="dim italic")
            t.append(text, style="dim")
        self.update(t)


class _StreamingComponentWidget(ComponentWidget):
    """ComponentWidget that streams markdown progressively.

    Uses :class:`NuMarkdown` from the start so deltas are rendered
    as formatted markdown while they arrive, not just on finalize.
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
        self._md_component = NuMarkdown("")
        super().__init__(self._md_component)
        self._chunks: list[str] = []

    def append_delta(self, delta: str) -> None:
        self._chunks.append(delta)
        self._md_component.set_text("".join(self._chunks))
        self.refresh_component()

    def get_text(self) -> str:
        return "".join(self._chunks)

    def finalize_as_markdown(self) -> None:
        """Final re-render — ensures the complete text is laid out."""
        text = self.get_text()
        if text.strip():
            self._md_component.set_text(text)
            self.refresh_component()


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
        ("ctrl+c", "interrupt_or_quit", "Cancel / Exit"),
    ]

    def __init__(
        self,
        session: AgentSession,
        quiet: bool = False,
        scoped_models: list[Any] | None = None,
    ) -> None:
        super().__init__()
        self._session = session
        self._quiet = quiet
        self._scoped_models: list[Any] = scoped_models or []
        self._working = False
        self._streaming_widget: _StreamingComponentWidget | None = None
        self._thinking_widget: ComponentWidget | None = None
        # Widgets keyed by tool_call_id for concurrent tool tracking
        self._tool_widgets: dict[str, ToolExecutionWidget | BashExecutionWidget] = {}
        # Running !command subprocess (None when idle)
        self._bash_proc: Any | None = None
        # Emacs-style kill ring for the prompt input (ctrl+w / ctrl+y)
        self._kill_ring: list[str] = []
        # Input history (arrow up/down)
        self._input_history: list[str] = []
        self._history_index: int = -1  # -1 = not browsing
        self._history_stash: str = ""  # saves current input when entering history

    def action_interrupt_or_quit(self) -> None:
        """Ctrl+C: kill a running !command subprocess, or exit when idle."""
        import contextlib  # noqa: PLC0415
        import os as _os  # noqa: PLC0415
        import signal as _signal  # noqa: PLC0415

        if self._bash_proc is not None:
            proc = self._bash_proc
            self._bash_proc = None  # cleared first so worker detects cancellation
            with contextlib.suppress(Exception):
                # Kill the entire process group — proc.pid is the PGID because
                # start_new_session=True makes the shell its own group leader.
                # This also kills grandchildren (e.g. `yes` spawned by `sh -c`).
                _os.killpg(proc.pid, _signal.SIGTERM)
            with contextlib.suppress(Exception):
                # Close the read end of the pipe so that any readline() currently
                # blocking in the thread pool gets an immediate IOError/EOF rather
                # than waiting for all buffered data to be drained (O(buf) → O(1)).
                if proc.stdout:
                    proc.stdout.close()
            return
        self.exit()

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

    def on_key(self, event: Any) -> None:
        """Readline-style kill ring and missing navigation for the prompt input.

        Textual's :class:`Input` handles the deletion half of ``ctrl+w``,
        ``ctrl+u``, ``ctrl+k``, and ``ctrl+f`` (delete word right — note:
        NOT cursor-forward) but has no kill ring, so ``ctrl+y`` (yank) is
        a no-op.  We intercept the kill keys *before* Textual processes
        them to capture the doomed text, then let Textual do the actual
        deletion.  ``ctrl+y`` is handled entirely by us.

        We also add a few readline keys that Textual's ``Input`` doesn't
        bind at all: ``ctrl+b`` (cursor left), ``alt+b`` / ``alt+f``
        (word left/right), ``alt+d`` (kill word forward), and ``ctrl+t``
        (transpose characters).
        """
        key = getattr(event, "key", "")
        inp = self.query_one("#prompt-input", Input)
        if inp is not self.focused:
            return

        value = inp.value
        cursor = inp.cursor_position

        # -- Kill operations (capture text, then let Textual delete) ------

        if key == "ctrl+w":
            # Kill word backward
            i = cursor
            while i > 0 and value[i - 1] == " ":
                i -= 1
            while i > 0 and value[i - 1] != " ":
                i -= 1
            killed = value[i:cursor]
            if killed:
                self._kill_ring.append(killed)
            return  # Textual handles the deletion

        if key == "ctrl+u":
            # Kill to start of line
            killed = value[:cursor]
            if killed:
                self._kill_ring.append(killed)
            return

        if key == "ctrl+k":
            # Kill to end of line
            killed = value[cursor:]
            if killed:
                self._kill_ring.append(killed)
            return

        if key == "ctrl+f":
            # Textual binds ctrl+f to "delete word right" (not cursor-forward).
            # Capture the killed word for the ring.
            i = cursor
            while i < len(value) and value[i] == " ":
                i += 1
            while i < len(value) and value[i] != " ":
                i += 1
            killed = value[cursor:i]
            if killed:
                self._kill_ring.append(killed)
            return

        # -- Yank ---------------------------------------------------------

        if key == "ctrl+y":
            event.prevent_default()
            event.stop()
            if self._kill_ring:
                text = self._kill_ring[-1]
                inp.value = value[:cursor] + text + value[cursor:]
                inp.cursor_position = cursor + len(text)
            return

        # -- Missing readline navigation ----------------------------------

        if key == "ctrl+b":
            # Cursor left (Textual only binds the arrow key)
            event.prevent_default()
            event.stop()
            if cursor > 0:
                inp.cursor_position = cursor - 1
            return

        if key == "alt+b":
            # Word left
            event.prevent_default()
            event.stop()
            i = cursor
            while i > 0 and value[i - 1] == " ":
                i -= 1
            while i > 0 and value[i - 1] != " ":
                i -= 1
            inp.cursor_position = i
            return

        if key == "alt+f":
            # Word right
            event.prevent_default()
            event.stop()
            i = cursor
            while i < len(value) and value[i] == " ":
                i += 1
            while i < len(value) and value[i] != " ":
                i += 1
            inp.cursor_position = i
            return

        if key == "alt+d":
            # Kill word forward (not natively in Textual)
            event.prevent_default()
            event.stop()
            i = cursor
            while i < len(value) and value[i] == " ":
                i += 1
            while i < len(value) and value[i] != " ":
                i += 1
            killed = value[cursor:i]
            if killed:
                self._kill_ring.append(killed)
            inp.value = value[:cursor] + value[i:]
            inp.cursor_position = cursor
            return

        if key == "ctrl+t":
            # Transpose characters
            event.prevent_default()
            event.stop()
            if cursor >= 2:
                chars = list(value)
                chars[cursor - 2], chars[cursor - 1] = chars[cursor - 1], chars[cursor - 2]
                inp.value = "".join(chars)
                inp.cursor_position = cursor
            elif cursor == 1 and len(value) >= 2:
                chars = list(value)
                chars[0], chars[1] = chars[1], chars[0]
                inp.value = "".join(chars)
                inp.cursor_position = min(cursor + 1, len(value))
            return

        # -- Input history (arrow up / down) ------------------------------

        if key == "up" and self._input_history:
            event.prevent_default()
            event.stop()
            if self._history_index == -1:
                # Entering history — stash whatever is currently typed
                self._history_stash = value
                self._history_index = len(self._input_history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            else:
                return  # already at oldest
            inp.value = self._input_history[self._history_index]
            inp.cursor_position = len(inp.value)
            return

        if key == "down":
            event.prevent_default()
            event.stop()
            if self._history_index == -1:
                return  # not browsing history
            if self._history_index < len(self._input_history) - 1:
                self._history_index += 1
                inp.value = self._input_history[self._history_index]
            else:
                # Past newest entry — restore stashed input
                self._history_index = -1
                inp.value = self._history_stash
            inp.cursor_position = len(inp.value)
            return

        # -- Model cycling (Ctrl+P / Ctrl+Shift+P) -----------------------

        if key == "ctrl+p":
            event.prevent_default()
            event.stop()
            self._cycle_model(direction=1)
            return

        if key == "ctrl+shift+p":
            event.prevent_default()
            event.stop()
            self._cycle_model(direction=-1)
            return

        # -- Thinking level cycling (Ctrl+T / Ctrl+Shift+T) ---------------

        if key == "ctrl+t":
            event.prevent_default()
            event.stop()
            self._cycle_thinking_level(direction=1)
            return

        if key == "ctrl+shift+t":
            event.prevent_default()
            event.stop()
            self._cycle_thinking_level(direction=-1)
            return

    def _cycle_model(self, direction: int = 1) -> None:
        """Cycle through available (or scoped) models."""
        available = self._scoped_models or self._session.model_registry.get_available_models()
        if not available:
            self._add_info("No models available.")
            return

        current = self._session.model
        current_id = current.id if current else ""

        # Find current index
        idx = -1
        for i, m in enumerate(available):
            if m.id == current_id:
                idx = i
                break

        next_idx = (idx + direction) % len(available)
        new_model = available[next_idx]
        self._session.set_model(new_model)

        # Update prompt placeholder and footer
        inp = self.query_one("#prompt-input", Input)
        inp.placeholder = f"Message ({new_model.id}) — /help for commands"
        try:
            footer = self.query_one(InteractiveFooter)
            footer.refresh_content()
        except Exception:
            pass
        self._add_info(f"Model: {new_model.provider}/{new_model.id}")

    def _cycle_thinking_level(self, direction: int = 1) -> None:
        """Cycle through thinking levels."""
        levels = ["off", "low", "medium", "high"]
        current = self._session.thinking_level
        try:
            idx = levels.index(current)
        except ValueError:
            idx = 0
        next_idx = (idx + direction) % len(levels)
        new_level = levels[next_idx]
        self._session.set_thinking_level(new_level)

        try:
            footer = self.query_one(InteractiveFooter)
            footer.refresh_content()
        except Exception:
            pass
        self._add_info(f"Thinking: {new_level}")

    def on_mount(self) -> None:
        self.sub_title = f"cwd: {self._session.cwd}"
        self._restore_history()
        inp = self.query_one("#prompt-input", Input)
        inp.focus()

    def _restore_history(self) -> None:
        """Render previously-persisted session entries on app start.

        Mirrors the upstream's ``addMessageToChat`` loop that runs on
        ``TUI.run()`` to populate the chat container from the session
        branch before the first prompt. Also populates the input history
        from user messages so arrow-up works across restarts (matching
        pi's ``populateHistory: true`` behaviour).
        """
        area = self.query_one("#message-area", VerticalScroll)
        sm = self._session.session_manager
        for entry in sm.get_branch():
            widget = _widget_for_entry(entry, self._session)
            if widget is not None:
                area.mount(widget)
            # Populate input history from user messages
            if entry.get("type") == "message":
                msg = entry.get("message")
                if msg is not None:
                    role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                    if role == "user":
                        text = _extract_user_text(msg)
                        if text and (not self._input_history or self._input_history[-1] != text):
                            self._input_history.append(text)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    _HELP_TEXT = """\
Available commands:
  /help            — show this help
  /exit, /quit     — exit the REPL
  /model           — show the current model
  /models          — pick a model from available models
  /compact [instr] — trigger context compaction (optional instructions)
  /clear           — clear the message area
  /session         — show session stats
  /sessions        — list and switch sessions
  /settings        — show current settings
  /theme           — switch dark/light theme
  /export [path]   — export session to HTML (or .jsonl if path ends with .jsonl)
  /copy            — copy last assistant message to clipboard
  /name [name]     — get or set the session name
  /changelog       — show the changelog
  /hotkeys         — show keyboard shortcuts
  /fork            — fork from a prior user message
  /resume          — resume a previous session
  /login           — log in to an OAuth provider
  /logout          — log out from an OAuth provider
  /new             — start a new session
  /reload          — reload extensions, keybindings, and settings
  /share           — share session as a GitHub Gist
  /import <path>   — import a session from a .jsonl file
  /thinking        — cycle thinking level (off/low/medium/high)
  /debug           — show debug information
  !command         — run a shell command outside the agent loop
  !!command        — run a shell command (excluded from context)

Keyboard shortcuts:
  Ctrl+P / Ctrl+Shift+P  — cycle model forward / backward
  Ctrl+T / Ctrl+Shift+T  — cycle thinking level
"""

    def _handle_slash_command(self, text: str) -> bool:
        """Process a slash command. Returns True if handled."""
        cmd = text.split(maxsplit=1)[0].lower() if text else ""
        if cmd in ("/exit", "/quit"):
            self.exit()
            return True
        if cmd == "/help":
            self._add_info(self._HELP_TEXT)
            return True
        if cmd == "/model":
            model = self._session.model
            self._add_info(f"Model: {model.id if model else 'none'}")
            return True
        if cmd in ("/models", "/scoped-models", "/model-select"):
            self._show_model_picker()
            return True
        if cmd == "/compact":
            instructions = text[len("/compact") :].strip() or None
            self._run_compact(instructions)
            return True
        if cmd == "/clear":
            area = self.query_one("#message-area", VerticalScroll)
            area.remove_children()
            return True
        if cmd == "/session":
            self._handle_session_command()
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
        if cmd == "/export":
            self._run_export(text)
            return True
        if cmd == "/copy":
            self._run_copy()
            return True
        if cmd == "/name":
            self._handle_name_command(text)
            return True
        if cmd == "/changelog":
            self._handle_changelog()
            return True
        if cmd == "/hotkeys":
            self._handle_hotkeys()
            return True
        if cmd == "/fork":
            self._show_fork_selector()
            return True
        if cmd == "/resume":
            self._show_resume_selector()
            return True
        if cmd == "/login":
            self._show_oauth_selector("login")
            return True
        if cmd == "/logout":
            self._show_oauth_selector("logout")
            return True
        if cmd == "/new":
            self._run_new_session()
            return True
        if cmd == "/reload":
            self._run_reload()
            return True
        if cmd == "/debug":
            self._handle_debug()
            return True
        if cmd == "/share":
            self._run_share()
            return True
        if cmd == "/import":
            path = text[len("/import") :].strip()
            self._run_import(path)
            return True
        if cmd == "/thinking":
            self._cycle_thinking_level()
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

    def _show_fork_selector(self) -> None:
        from nu_coding_agent.modes.interactive.selectors import ForkSelectorScreen  # noqa: PLC0415

        messages = self._session.get_user_messages_for_forking()
        if not messages:
            self._add_info("No user messages to fork from.")
            return

        def on_dismiss(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            entry_id, selected_text = result
            self._run_fork(entry_id, selected_text)

        self.push_screen(ForkSelectorScreen(messages), on_dismiss)

    def _show_resume_selector(self) -> None:
        from nu_coding_agent.modes.interactive.selectors import ResumeSelectorScreen  # noqa: PLC0415

        sm = self._session.session_manager
        cwd = self._session.cwd

        def on_dismiss(session_path: str | None) -> None:
            if session_path is None:
                return
            self._run_resume(session_path)

        self.push_screen(ResumeSelectorScreen(cwd, sm.get_session_dir()), on_dismiss)

    def _show_oauth_selector(self, mode: str) -> None:
        from nu_coding_agent.modes.interactive.selectors import OAuthSelectorScreen  # noqa: PLC0415

        auth = self._session.auth_storage
        if mode == "logout":
            providers = auth.get_oauth_providers()
            logged_in = [p for p in providers if auth.get(p) is not None]
            if not logged_in:
                self._add_info("No OAuth providers logged in. Use /login first.")
                return

        def on_dismiss(provider_id: str | None) -> None:
            if provider_id is None:
                return
            if mode == "login":
                self._run_oauth_login(provider_id)
            else:
                self._run_oauth_logout(provider_id)

        self.push_screen(OAuthSelectorScreen(mode, auth), on_dismiss)

    # ------------------------------------------------------------------
    # Slash command handlers
    # ------------------------------------------------------------------

    @work(thread=False)
    async def _run_export(self, text: str) -> None:
        """``/export [path]`` — export session to HTML or JSONL."""
        parts = text.split()
        output_path = parts[1] if len(parts) > 1 else None
        try:
            if output_path and output_path.endswith(".jsonl"):
                sm = self._session.session_manager
                sf = sm.get_session_file()
                if sf is None:
                    self._add_info("Session is in-memory; nothing to export as JSONL.")
                    return
                import shutil  # noqa: PLC0415

                shutil.copy2(sf, output_path)
                self._add_info(f"Session exported to: {output_path}")
            else:
                file_path = await self._session.export_to_html(output_path)
                self._add_info(f"Session exported to: {file_path}")
        except Exception as exc:
            self._mount_error(f"Export failed: {exc}")

    @work(thread=False)
    async def _run_copy(self) -> None:
        """``/copy`` — copy the last assistant message to the clipboard."""
        text = self._session.get_last_assistant_text()
        if not text:
            self._mount_error("No agent messages to copy yet.")
            return
        try:
            import asyncio as _asyncio  # noqa: PLC0415

            import pyperclip  # noqa: PLC0415

            await _asyncio.to_thread(pyperclip.copy, text)
            self._add_info("Copied last agent message to clipboard.")
        except Exception as exc:
            self._mount_error(f"Copy failed: {exc}")

    def _handle_name_command(self, text: str) -> None:
        """``/name [name]`` — get or set the session name."""
        name = text[len("/name") :].strip()
        sm = self._session.session_manager
        if not name:
            current = sm.get_session_name()
            if current:
                self._add_info(f"Session name: {current}")
            else:
                self._add_info("Usage: /name <name>")
            return
        sm.append_session_info(name)
        self.sub_title = f"cwd: {self._session.cwd} | {name}"
        self._add_info(f"Session name set: {name}")

    def _handle_session_command(self) -> None:
        """``/session`` — show session statistics."""
        stats = self._session.get_stats()
        lines = [
            "**Session Info**",
            "",
            f"File: {stats.session_file or '(in-memory)'}",
            f"ID: {stats.session_id}",
            "",
            "**Messages**",
            f"User: {stats.user_messages}",
            f"Assistant: {stats.assistant_messages}",
            f"Tool Calls: {stats.tool_calls}",
            f"Tool Results: {stats.tool_results}",
            f"Total: {stats.total_messages}",
            "",
            "**Tokens**",
            f"Input: {stats.tokens_input:,}",
            f"Output: {stats.tokens_output:,}",
        ]
        if stats.tokens_cache_read > 0:
            lines.append(f"Cache Read: {stats.tokens_cache_read:,}")
        if stats.tokens_cache_write > 0:
            lines.append(f"Cache Write: {stats.tokens_cache_write:,}")
        total_tokens = stats.tokens_input + stats.tokens_output + stats.tokens_cache_read + stats.tokens_cache_write
        lines.append(f"Total: {total_tokens:,}")
        if stats.cost > 0:
            lines += ["", "**Cost**", f"Total: ${stats.cost:.4f}"]
        self._add_info("\n".join(lines))

    def _handle_changelog(self) -> None:
        """``/changelog`` — show changelog entries."""
        from nu_coding_agent.utils.changelog import parse_changelog  # noqa: PLC0415

        # Look for CHANGELOG.md in the package resources or current directory
        candidates = [
            Path(__file__).parents[4] / "CHANGELOG.md",
            Path(self._session.cwd) / "CHANGELOG.md",
        ]
        changelog_path = next((str(p) for p in candidates if p.exists()), "")
        entries = parse_changelog(changelog_path) if changelog_path else []
        if entries:
            md = "\n\n".join(e.content for e in reversed(entries))
            self._add_info(f"**What's New**\n\n{md}")
        else:
            self._add_info("No changelog entries found.")

    def _handle_hotkeys(self) -> None:
        """``/hotkeys`` — show keyboard shortcuts."""
        hotkeys = """\
**Navigation**
| Key | Action |
|-----|--------|
| `Up` / `Down` | Browse history (when input is empty) |
| `Ctrl+A` | Start of line |
| `Ctrl+E` | End of line |

**Editing**
| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `Ctrl+W` | Delete word backwards |
| `Ctrl+K` | Delete to end of line |
| `Ctrl+U` | Delete to start of line |
| `Ctrl+Y` | Yank (paste deleted text) |

**Other**
| Key | Action |
|-----|--------|
| `Tab` | Autocomplete / expand |
| `Ctrl+C` | Cancel / abort |
| `Ctrl+D` | Exit (when input is empty) |
| `/` | Slash commands |
| `!` | Run shell command |
| `!!` | Run shell command (excluded from context) |

**Commands**
Type `/help` to see all slash commands.
"""
        self._add_info(hotkeys)

    def _handle_debug(self) -> None:
        """``/debug`` — show debug information."""
        stats = self._session.get_stats()
        model = self._session.model
        lines = [
            "**Debug Info**",
            "",
            f"Session file: {stats.session_file or '(in-memory)'}",
            f"Session ID: {stats.session_id}",
            f"CWD: {self._session.cwd}",
            f"Model: {model.provider}/{model.id}" if model else "Model: none",
            f"Messages: {stats.total_messages}",
            f"Tokens in: {stats.tokens_input:,}  out: {stats.tokens_output:,}",
            f"Working: {self._working}",
        ]
        runner = self._session.extension_runner
        if runner is not None:
            ext_names = [getattr(e, "name", str(e)) for e in runner.extensions]
            lines.append(f"Extensions: {', '.join(ext_names) or 'none'}")
        self._add_info("\n".join(lines))

    @work(thread=False)
    async def _run_share(self) -> None:
        """``/share`` — export session to a GitHub Gist and show the share URL."""
        sf = self._session.session_manager.get_session_file()
        if sf is None:
            self._mount_error("Cannot share: session is not persisted to disk.")
            return
        try:
            html = await self._session.export_to_html()
            # Create a gist via gh CLI
            import subprocess as _sp  # noqa: PLC0415
            import tempfile  # noqa: PLC0415

            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as fh:
                fh.write(html)
                tmp = fh.name
            try:
                import asyncio as _aio  # noqa: PLC0415

                result = await _aio.to_thread(
                    _sp.run,
                    ["gh", "gist", "create", "--public", "-f", "session.html", tmp],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode == 0:
                    gist_url = result.stdout.strip()
                    from nu_coding_agent.config import get_share_viewer_url  # noqa: PLC0415

                    gist_id = gist_url.rsplit("/", 1)[-1] if "/" in gist_url else gist_url
                    viewer_url = get_share_viewer_url(gist_id)
                    self._add_info(f"Shared! View at: {viewer_url}\nGist: {gist_url}")
                else:
                    self._mount_error(f"Failed to create gist: {result.stderr.strip()}")
            finally:
                import os  # noqa: PLC0415

                os.unlink(tmp)
        except Exception as exc:
            self._mount_error(f"Share failed: {exc}")

    def _run_import(self, path: str) -> None:
        """``/import <path>`` — import a session from a .jsonl file."""
        if not path:
            self._add_info("Usage: /import <session.jsonl>")
            return
        import os  # noqa: PLC0415

        resolved = os.path.expanduser(path)
        if not os.path.isfile(resolved):
            self._mount_error(f"File not found: {resolved}")
            return
        try:
            from nu_coding_agent.core.session_manager import SessionManager  # noqa: PLC0415

            sm = SessionManager(
                cwd=self._session.cwd,
                session_dir=os.path.dirname(resolved),
                session_file=resolved,
                persist=True,
            )
            self._session.set_session_manager(sm)
            # Reload the chat view
            area = self.query_one("#message-area", VerticalScroll)
            area.remove_children()
            self._restore_history()
            self._add_info(f"Imported session from {resolved}")
            try:
                footer = self.query_one(InteractiveFooter)
                footer.refresh_content()
            except Exception:
                pass
        except Exception as exc:
            self._mount_error(f"Import failed: {exc}")

    @work(thread=False)
    async def _run_fork(self, entry_id: str, selected_text: str) -> None:
        """Fork the session at the given entry."""
        sm = self._session.session_manager
        sf = sm.get_session_file()
        if sf is None:
            self._mount_error("Cannot fork: session is not persisted to disk.")
            return
        try:
            from nu_coding_agent.core.session_manager import SessionManager  # noqa: PLC0415

            # Find entries up to (and including) the selected entry
            entries = sm.get_entries()
            kept: list[dict[str, Any]] = []
            for e in entries:
                kept.append(e)
                if e.get("id") == entry_id:
                    break

            # Create new session manager and replay kept entries
            new_session = SessionManager(
                cwd=sm.get_cwd(), session_dir=sm.get_session_dir(), session_file=None, persist=True
            )
            for e in kept:
                if e.get("type") == "session":
                    continue
                new_session._append_entry(e)  # type: ignore[attr-defined]

            # Switch the current session's session manager to the new one
            self._session._session_manager = new_session  # type: ignore[attr-defined]

            # Reload the UI from the new session
            area = self.query_one("#message-area", VerticalScroll)
            area.remove_children()
            self._restore_history()
            self._add_info(f"Forked to new session. You can continue from: {selected_text[:60]}")
        except Exception as exc:
            self._mount_error(f"Fork failed: {exc}")

    @work(thread=False)
    async def _run_resume(self, session_path: str) -> None:
        """Switch to a previous session."""
        try:
            sm = self._session.session_manager
            sm.set_session_file(session_path)
            area = self.query_one("#message-area", VerticalScroll)
            area.remove_children()
            self._restore_history()
            self._add_info(f"Resumed session: {session_path}")
        except Exception as exc:
            self._mount_error(f"Resume failed: {exc}")

    @work(thread=False)
    async def _run_new_session(self) -> None:
        """Start a fresh session (``/new``)."""
        try:
            sm = self._session.session_manager
            sm.new_session()
            area = self.query_one("#message-area", VerticalScroll)
            area.remove_children()
            self._add_info("New session started.")
        except Exception as exc:
            self._mount_error(f"New session failed: {exc}")

    @work(thread=False)
    async def _run_reload(self) -> None:
        """Reload extensions, keybindings, and settings (``/reload``)."""
        try:
            await self._session.reload()
            self._add_info("Reloaded extensions, keybindings, and settings.")
        except Exception as exc:
            self._mount_error(f"Reload failed: {exc}")

    @work(thread=False)
    async def _run_oauth_login(self, provider_id: str) -> None:
        """Push a LoginDialog for the given OAuth provider.

        The dialog drives the OAuth flow; on success the model registry
        is refreshed so newly-unlocked models become available.
        """
        from nu_coding_agent.modes.interactive.components.login_dialog import LoginDialog  # noqa: PLC0415

        auth = self._session.auth_storage

        def on_complete(ok: bool, _msg: str) -> None:
            if ok:
                auth.reload()
                self._session.model_registry.refresh()
                self._add_info(f"Logged in to {provider_id}.")
            else:
                self._add_info(f"Login to {provider_id} cancelled.")

        dialog = LoginDialog(provider_id, on_complete=on_complete)
        await self.push_screen_wait(dialog)

    def _run_oauth_logout(self, provider_id: str) -> None:
        """Logout from the given OAuth provider."""
        try:
            self._session.auth_storage.logout(provider_id)
            self._session.model_registry.refresh()
            self._add_info(f"Logged out from {provider_id}.")
        except Exception as exc:
            self._mount_error(f"Logout failed: {exc}")

    # Max bytes accumulated in memory for a !command output stream.
    _BASH_CMD_MAX_BYTES = 512 * 1024  # 512 KB

    @work(thread=False)
    async def _run_bash_command(self, command: str, *, excluded: bool = False) -> None:
        """Run a shell command outside the agent loop (``!command``).

        **Why the subprocess is isolated from terminal SIGINT:**
        ``start_new_session=True`` puts the subprocess in its own session so
        Ctrl+C in the terminal sends SIGINT only to ``nu``, not to the child.

        **Why SIGINT is intercepted while the command runs:**
        A temporary SIGINT handler replaces Python's default (which would
        raise ``KeyboardInterrupt`` and kill ``nu``).  The handler terminates
        the subprocess and restores the original handler so the app lives on.

        **Why stdout is read via ``run_in_executor``:**
        ``readline()`` blocks in a thread-pool thread, keeping the asyncio
        event loop free to process key events even for programs like ``yes``
        that produce output faster than the loop could drain it.

        Output is capped at :attr:`_BASH_CMD_MAX_BYTES` to prevent OOM.
        """
        import asyncio as _asyncio  # noqa: PLC0415
        import contextlib  # noqa: PLC0415
        import signal  # noqa: PLC0415
        import subprocess as _subprocess  # noqa: PLC0415

        widget = BashExecutionWidget(command)
        area = self.query_one("#message-area", VerticalScroll)
        area.mount(widget)
        widget.scroll_visible()

        self._working = True
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = True

        cancelled = False
        exit_code: int | None = None
        old_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_for_bash(sig: int, frame: object) -> None:
            """Kill only the child process group; keep nu alive."""
            import os as _os  # noqa: PLC0415

            if self._bash_proc is not None:
                proc_ref = self._bash_proc
                self._bash_proc = None
                with contextlib.suppress(Exception):
                    _os.killpg(proc_ref.pid, signal.SIGTERM)
                with contextlib.suppress(Exception):
                    if proc_ref.stdout:
                        proc_ref.stdout.close()
            signal.signal(signal.SIGINT, old_sigint)

        try:
            proc = _subprocess.Popen(  # noqa: ASYNC220
                command,
                shell=True,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.STDOUT,
                cwd=self._session.cwd,
                start_new_session=True,  # isolate from terminal Ctrl+C SIGINT
            )
            self._bash_proc = proc
            signal.signal(signal.SIGINT, _sigint_for_bash)

            loop = _asyncio.get_running_loop()
            output_chunks: list[str] = []
            total_bytes = 0

            assert proc.stdout is not None
            while True:
                line: bytes = await loop.run_in_executor(None, proc.stdout.readline)
                # Break on EOF or if cancelled (action_interrupt_or_quit / SIGINT
                # handler clears _bash_proc and closes stdout).  Checking here
                # makes cancellation O(1) — we discard buffered data rather than
                # draining the entire pipe (up to 1 MB at 50 µs/line = seconds).
                if not line or self._bash_proc is None:
                    break
                output_chunks.append(line.decode("utf-8", errors="replace"))
                total_bytes += len(line)
                widget.set_output("".join(output_chunks))
                if total_bytes >= self._BASH_CMD_MAX_BYTES:
                    output_chunks.append("\n[output truncated]")
                    widget.set_output("".join(output_chunks))
                    with contextlib.suppress(Exception):
                        import os as _os2  # noqa: PLC0415

                        _os2.killpg(proc.pid, signal.SIGTERM)
                    with contextlib.suppress(Exception):
                        proc.stdout.close()
                    break

            await loop.run_in_executor(None, proc.wait)
            exit_code = proc.returncode
            cancelled = self._bash_proc is None and exit_code not in (0,)

            if not excluded:
                output_text = "".join(output_chunks)
                self._add_info(f"$ {command}\n(exit {exit_code})\n{output_text[:500]}")
        except Exception as exc:
            self._mount_error(f"Bash command failed: {exc}")
            exit_code = 1
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            self._bash_proc = None
            widget.set_complete(exit_code, cancelled=cancelled)
            self._working = False
            inp.disabled = False
            inp.focus()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        # Save to history (dedup consecutive identical entries)
        if not self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)
        self._history_index = -1
        self._history_stash = ""

        event.input.value = ""

        # Handle bash exec prefix (! or !!) before slash-command check
        if text.startswith("!") and not text.startswith("/"):
            is_excluded = text.startswith("!!")
            command = text[2:].strip() if is_excluded else text[1:].strip()
            if command:
                if self._working:
                    self._add_info("A command is already running. Wait for it to finish.")
                    event.input.value = text
                    return
                self._run_bash_command(command, excluded=is_excluded)
            return

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

        # Create streaming widgets
        area = self.query_one("#message-area", VerticalScroll)
        thinking_widget: _ThinkingBlockWidget | None = None
        assistant_widget = _StreamingComponentWidget()
        area.mount(assistant_widget)

        def listener(event: Any) -> None:
            nonlocal thinking_widget
            # NOTE: This listener runs inside the async event loop (not a
            # thread) because _run_prompt is ``@work(thread=False)``.  We
            # must NOT use ``call_from_thread`` — that queues callbacks that
            # cannot execute until the awaited prompt() yields, so all
            # updates would be batched until the prompt finishes.  Instead
            # we mutate the widgets directly and call ``refresh()`` to
            # request a repaint.
            event_type = event["type"]

            if event_type == "message_update":
                inner = event.get("assistant_message_event")
                if inner is not None:
                    inner_type = getattr(inner, "type", None)
                    if inner_type == "text_delta":
                        delta = getattr(inner, "delta", "")
                        if delta:
                            assistant_widget.append_delta(delta)
                    elif inner_type == "thinking_start":
                        # Mount thinking block BEFORE the assistant text widget
                        thinking_widget = _ThinkingBlockWidget()
                        try:
                            area.mount(thinking_widget, before=assistant_widget)
                        except Exception:
                            area.mount(thinking_widget)
                    elif inner_type == "thinking_delta":
                        delta = getattr(inner, "delta", "")
                        if delta and thinking_widget is not None:
                            thinking_widget.append_delta(delta)
                    elif inner_type == "thinking_end":
                        if thinking_widget is not None:
                            thinking_widget.finalize()

            elif event_type == "message_start":
                msg = event.get("message")
                role = getattr(msg, "role", None)
                if role == "custom":
                    self._mount_custom_message(msg)

            elif event_type == "tool_execution_start" and not self._quiet:
                tool_name = str(event.get("tool_name", ""))
                args: dict[str, Any] = event.get("args") or {}
                tool_call_id = str(event.get("tool_call_id", ""))
                self._on_tool_start(tool_call_id, tool_name, args)

            elif event_type == "tool_execution_update" and not self._quiet:
                tool_call_id = str(event.get("tool_call_id", ""))
                partial = event.get("partial_result")
                self._on_tool_update(tool_call_id, partial)

            elif event_type == "tool_execution_end" and not self._quiet:
                tool_call_id = str(event.get("tool_call_id", ""))
                is_error = bool(event.get("is_error", False))
                result = event.get("result")
                self._on_tool_end(tool_call_id, is_error, result)

        unsub = self._session.subscribe(listener)

        try:
            await self._session.prompt(text)

            assistant_widget.finalize_as_markdown()
            assistant_widget.scroll_visible()

            messages = self._session.agent.state.messages
            error_detected = False
            if messages:
                last = messages[-1]
                stop = getattr(last, "stop_reason", None)
                if stop == "error":
                    err = getattr(last, "error_message", None) or "unknown error"
                    error_detected = True

                    # Auto-retry: if the error looks like a context overflow
                    # and auto-retry is enabled, compact and re-prompt.
                    is_overflow = "context" in err.lower() or "token" in err.lower() or "overflow" in err.lower()
                    if is_overflow and self._session.auto_retry_enabled and self._session.should_compact():
                        self._add_info(f"Context overflow detected: {err}")
                        self._add_info("Auto-compacting and retrying...")
                        try:
                            await self._session.compact()
                            # Remove the failed assistant message from agent state
                            # so the retry doesn't see it
                            if self._session.agent.state.messages:
                                self._session.agent.state.messages.pop()
                            await self._session.agent.continue_run()
                            # Re-finalize the widget with any new content
                            assistant_widget.finalize_as_markdown()
                            assistant_widget.scroll_visible()
                            error_detected = False
                        except Exception as retry_exc:
                            self._mount_error(f"Auto-retry failed: {retry_exc}")
                    else:
                        self._mount_error(err)

            # Update footer
            try:
                footer = self.query_one(InteractiveFooter)
                footer.refresh_content()
            except Exception:
                pass

            if not error_detected and self._session.should_compact():
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
    async def _run_compact(self, instructions: str | None = None) -> None:
        try:
            result = await self._session.compact(custom_instructions=instructions)
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


def _extract_user_text(msg: Any) -> str:
    """Extract the plain-text body from a user message (dict or Pydantic)."""
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "").strip()
            if hasattr(block, "type") and getattr(block, "type", None) == "text":
                return getattr(block, "text", "").strip()
        return ""
    return str(content).strip()


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


async def run_interactive_mode(
    session: AgentSession,
    *,
    quiet: bool = False,
    scoped_models: list[Any] | None = None,
) -> int:
    """Launch the interactive REPL.

    Returns an exit code (0 for clean exit, 1 for error).
    """
    app = InteractiveApp(session, quiet=quiet, scoped_models=scoped_models)
    try:
        await app.run_async()
    except Exception as exc:
        print(f"Interactive mode error: {exc}", file=sys.stderr)
        return 1
    return 0
