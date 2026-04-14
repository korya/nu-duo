"""TUI config selector for the ``nu config`` sub-command.

Direct port of ``packages/coding-agent/src/cli/config-selector.ts``.

The TS upstream opens a full-screen TUI listing skills, extensions,
themes, and prompt templates with inline enable/disable toggles. The
Python port provides the same entry-point signature; the actual
component will be wired in once the interactive config-selector
component is ported.

Currently launches a minimal Textual app that displays available
configuration resources and lets the user toggle them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

if TYPE_CHECKING:
    from nu_coding_agent.core.settings_manager import SettingsManager


@dataclass(slots=True)
class ConfigSelectorOptions:
    """Options for :func:`select_config`."""

    cwd: str
    agent_dir: str
    settings_manager: SettingsManager


class _ConfigSelectorApp(App[None]):
    """Standalone Textual app for browsing/toggling configuration resources."""

    BINDINGS = [  # noqa: RUF012
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        background: $surface;
    }
    #config-list {
        height: 1fr;
    }
    .config-header {
        text-style: bold;
        margin: 1 0;
    }
    """

    def __init__(self, options: ConfigSelectorOptions) -> None:
        super().__init__()
        self._options = options

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(
            Static("Configuration", classes="config-header"),
            Static(f"Working directory: {self._options.cwd}"),
            Static(f"Agent directory:   {self._options.agent_dir}"),
            Static(""),
            Static("Skills, extensions, themes, and prompt templates", classes="config-header"),
            self._build_resource_list(),
        )
        yield Footer()

    def _build_resource_list(self) -> ListView:
        items: list[ListItem] = []
        sm = self._options.settings_manager

        # Skills
        items.append(ListItem(Label("[bold]Skills[/bold]")))
        for skill_name in getattr(sm, "get_enabled_skills", list)():
            items.append(ListItem(Label(f"  {skill_name}")))

        # Extensions
        items.append(ListItem(Label("[bold]Extensions[/bold]")))
        for ext_name in getattr(sm, "get_enabled_extensions", list)():
            items.append(ListItem(Label(f"  {ext_name}")))

        if not items:
            items.append(ListItem(Label("No configuration resources found.")))

        return ListView(*items, id="config-list")

    async def action_quit(self) -> None:
        self.exit()


async def select_config(options: ConfigSelectorOptions) -> None:
    """Show the TUI config selector.

    Parameters
    ----------
    options:
        Configuration context including the settings manager, working
        directory, and agent directory.

    The function returns when the user closes the selector.
    """
    app = _ConfigSelectorApp(options)
    await app.run_async()


__all__ = [
    "ConfigSelectorOptions",
    "select_config",
]
