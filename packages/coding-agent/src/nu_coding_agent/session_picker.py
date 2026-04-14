"""TUI session picker for the ``--resume`` flag.

Direct port of ``packages/coding-agent/src/cli/session-picker.ts``.

The TS upstream creates a bare ``TUI`` + ``SessionSelectorComponent``.
The Python port delegates to the already-ported Textual-based
:class:`~nu_coding_agent.modes.interactive.components.session_selector.SessionSelectorScreen`,
running it as a standalone Textual app. The result is the selected
session file path, or ``None`` if the user cancelled.
"""

from __future__ import annotations

from textual.app import App

from nu_coding_agent.modes.interactive.components.session_selector import (
    SessionSelectorScreen,
)


class _SessionPickerApp(App[str | None]):
    """Minimal Textual app that shows the session selector and exits."""

    CSS = """
    Screen {
        background: $surface;
    }
    """

    def __init__(
        self,
        cwd: str,
        session_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._cwd = cwd
        self._session_dir = session_dir

    async def on_mount(self) -> None:
        result = await self.push_screen_wait(
            SessionSelectorScreen(self._cwd, session_dir=self._session_dir),
        )
        self.exit(result)


async def select_session(
    cwd: str,
    *,
    session_dir: str | None = None,
) -> str | None:
    """Show the TUI session selector and return the selected session path.

    Parameters
    ----------
    cwd:
        The working directory whose sessions are shown by default.
    session_dir:
        Optional override for the session storage directory.

    Returns
    -------
    str | None
        The absolute path to the selected session JSONL file, or ``None``
        if the user pressed Escape / quit without selecting.
    """
    app = _SessionPickerApp(cwd, session_dir=session_dir)
    result = await app.run_async()
    return result


__all__ = [
    "select_session",
]
