"""Interactive mode footer — model info, session status, keybinding hints.

Port of ``packages/coding-agent/src/modes/interactive/components/footer.ts``
(220 LoC). The upstream footer renders via the nu_tui component tree;
this Python version is a Textual ``Static`` widget that updates its
content from the AgentSession state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text as RichText
from textual.widgets import Static

if TYPE_CHECKING:
    from nu_coding_agent.core.agent_session import AgentSession


class InteractiveFooter(Static):
    """Footer bar showing model, session, and keybinding info."""

    DEFAULT_CSS = """
    InteractiveFooter {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-3;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, session: AgentSession) -> None:
        super().__init__("")
        self._session = session
        self.refresh_content()

    def refresh_content(self) -> None:
        """Update the footer content from the current session state."""
        parts: list[str] = []

        # Model
        model = self._session.model
        if model is not None:
            parts.append(f"model: {model.provider}/{model.id}")
        else:
            parts.append("model: none")

        # Session
        sf = self._session.session_manager.get_session_file()
        if sf:
            # Show just the filename, not the full path
            import os  # noqa: PLC0415

            parts.append(f"session: {os.path.basename(sf)}")
        else:
            parts.append("session: (in-memory)")

        # Entry count
        entry_count = len(self._session.session_manager.get_entries())
        parts.append(f"entries: {entry_count}")

        # Extensions
        runner = self._session.extension_runner
        if runner is not None:
            ext_count = len(runner.extensions)
            if ext_count > 0:
                parts.append(f"extensions: {ext_count}")

        # Keybinding hints
        parts.append("Ctrl+C: exit")
        parts.append("/help: commands")

        text = " │ ".join(parts)
        self.update(RichText(text, style="dim"))


__all__ = ["InteractiveFooter"]
