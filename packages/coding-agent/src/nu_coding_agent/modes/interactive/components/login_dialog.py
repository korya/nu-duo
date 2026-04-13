"""Login dialog — port of ``login-dialog.ts``.

Implements an OAuth login flow as a Textual ``ModalScreen``.  The dialog
progresses through several states driven by the provider's OAuth callbacks:

1. **auth** — shows the authorization URL and tries to open a browser.
2. **manual_input** — shows a prompt and waits for the user to paste a code/URL.
3. **waiting** — shows a polling message (e.g. GitHub Device Flow).
4. **progress** — appends incremental status lines.

The caller controls the dialog by calling the async methods
:meth:`show_auth`, :meth:`show_manual_input`, :meth:`show_waiting`,
:meth:`show_progress` from a background task.  The dialog calls
:attr:`on_complete` when the flow succeeds or is cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any

from rich.text import Text as RichText
from textual import on
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

from nu_coding_agent.modes.interactive.components.keybinding_hints import key_hint


class LoginDialog(ModalScreen[bool]):
    """Modal OAuth login dialog.

    Port of ``LoginDialogComponent`` (178 LoC) adapted to Textual's
    screen-push model.  The dialog is pushed onto the screen stack and
    pops itself when the flow completes (returning ``True`` on success,
    ``False`` on cancellation).

    Usage::

        async def run_oauth(dialog: LoginDialog) -> None:
            await dialog.show_auth("https://provider.example/authorize")
            # ... drive the OAuth flow ...
            dialog.dismiss(True)

        dialog = LoginDialog("anthropic", on_complete=run_oauth)
        await app.push_screen_wait(dialog)
    """

    DEFAULT_CSS = """
    LoginDialog {
        align: center middle;
    }
    LoginDialog > VerticalScroll {
        width: 80;
        max-height: 30;
        border: tall $accent;
        padding: 1 2;
        background: $surface;
    }
    LoginDialog Input {
        margin: 1 0;
    }
    """

    def __init__(
        self,
        provider_id: str,
        *,
        provider_name: str | None = None,
        on_complete: Any | None = None,
    ) -> None:
        super().__init__()
        self._provider_id = provider_id
        self._provider_name = provider_name or provider_id
        self._on_complete = on_complete
        self._input_future: asyncio.Future[str] | None = None
        self._cancelled = False

    # ------------------------------------------------------------------
    # Textual lifecycle
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(
                RichText(f"Login to {self._provider_name}", style="bold yellow"),
                id="login-title",
            )
            yield Static("", id="login-content")
            yield Input(placeholder="", id="login-input")
            yield Static("", id="login-hints")

    def on_mount(self) -> None:
        # Hide input until needed
        self.query_one("#login-input", Input).display = False

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def on_key(self, event: Any) -> None:
        if getattr(event, "key", None) == "escape":
            self._cancel()
            event.stop()

    @on(Input.Submitted, "#login-input")
    def _on_input_submit(self, _event: Input.Submitted) -> None:
        inp = self.query_one("#login-input", Input)
        value = inp.value
        if self._input_future and not self._input_future.done():
            self._input_future.set_result(value)
        inp.value = ""
        inp.display = False

    # ------------------------------------------------------------------
    # Public API (called from background coroutine driving the flow)
    # ------------------------------------------------------------------

    def show_auth(self, url: str, instructions: str | None = None) -> None:
        """Display the authorization URL and try to open a browser."""
        content = self.query_one("#login-content", Static)
        existing = self._get_content_text()

        t = RichText(existing)
        t.append(f"\n{url}\n", style="bold cyan")
        if instructions:
            t.append(f"\n{instructions}\n", style="yellow")
        content.update(t)

        # Try to open browser
        _try_open_url(url)

    def show_waiting(self, message: str) -> None:
        """Show a waiting message (e.g. polling)."""
        self._append_content(message, style="dim")
        hints = self.query_one("#login-hints", Static)
        hints.update(RichText(key_hint("tui.select.cancel", "to cancel"), style="dim"))

    def show_progress(self, message: str) -> None:
        """Append a progress line."""
        self._append_content(message, style="dim")

    async def show_manual_input(self, prompt: str) -> str:
        """Show an input prompt and wait for user submission.

        Returns the entered text, or raises ``asyncio.CancelledError`` if
        the dialog is cancelled while waiting.
        """
        self._append_content(prompt, style="dim")
        inp = self.query_one("#login-input", Input)
        inp.placeholder = prompt
        inp.display = True
        inp.focus()
        hints = self.query_one("#login-hints", Static)
        hints.update(RichText(key_hint("tui.select.cancel", "to cancel"), style="dim"))

        loop = asyncio.get_event_loop()
        self._input_future = loop.create_future()
        try:
            return await self._input_future
        except asyncio.CancelledError:
            raise

    async def show_prompt(self, message: str, placeholder: str | None = None) -> str:
        """Show a prompt (preserves existing content) and wait for input.

        Port of ``showPrompt`` — analogous to :meth:`show_manual_input`
        but appends to existing content rather than replacing.
        """
        self._append_content(message, style="dim")
        inp = self.query_one("#login-input", Input)
        inp.placeholder = placeholder or ""
        inp.value = ""
        inp.display = True
        inp.focus()
        hints = self.query_one("#login-hints", Static)
        cancel_hint = key_hint("tui.select.cancel", "to cancel")
        confirm_hint = key_hint("tui.select.confirm", "to submit")
        hints.update(RichText(f"{cancel_hint}  {confirm_hint}", style="dim"))

        loop = asyncio.get_event_loop()
        self._input_future = loop.create_future()
        try:
            return await self._input_future
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        if self._input_future and not self._input_future.done():
            self._input_future.cancel()
        if self._on_complete:
            self._on_complete(False, "Login cancelled")
        self.dismiss(False)

    def _get_content_text(self) -> str:
        content = self.query_one("#login-content", Static)
        renderable = getattr(content, "renderable", None)
        if isinstance(renderable, RichText):
            return renderable.plain
        if isinstance(renderable, str):
            return renderable
        return ""

    def _append_content(self, message: str, style: str = "") -> None:
        content = self.query_one("#login-content", Static)
        existing = self._get_content_text()
        t = RichText(existing)
        t.append(f"\n{message}", style=style)
        content.update(t)


def _try_open_url(url: str) -> None:
    """Best-effort browser open — silently ignored on failure."""
    if sys.platform == "darwin":
        cmd = ["open", url]
    elif sys.platform == "win32":
        cmd = ["start", url]
    else:
        cmd = ["xdg-open", url]
    with contextlib.suppress(Exception):
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


__all__ = ["LoginDialog"]
