"""Built-in slash command catalog — direct port of ``packages/coding-agent/src/core/slash-commands.ts``.

Slash commands are surfaced by the interactive REPL (not yet ported); the
catalog is exposed here so the print-mode and RPC layers can list them
without pulling in the TUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nu_coding_agent.core.source_info import SourceInfo

type SlashCommandSource = Literal["extension", "prompt", "skill"]


@dataclass(slots=True)
class SlashCommandInfo:
    """A user-defined slash command contributed by an extension/prompt/skill."""

    name: str
    source: SlashCommandSource
    source_info: SourceInfo
    description: str | None = None


@dataclass(slots=True)
class BuiltinSlashCommand:
    name: str
    description: str


BUILTIN_SLASH_COMMANDS: tuple[BuiltinSlashCommand, ...] = (
    BuiltinSlashCommand(name="settings", description="Open settings menu"),
    BuiltinSlashCommand(name="model", description="Select model (opens selector UI)"),
    BuiltinSlashCommand(name="scoped-models", description="Enable/disable models for Ctrl+P cycling"),
    BuiltinSlashCommand(
        name="export",
        description="Export session (HTML default, or specify path: .html/.jsonl)",
    ),
    BuiltinSlashCommand(name="import", description="Import and resume a session from a JSONL file"),
    BuiltinSlashCommand(name="share", description="Share session as a secret GitHub gist"),
    BuiltinSlashCommand(name="copy", description="Copy last agent message to clipboard"),
    BuiltinSlashCommand(name="name", description="Set session display name"),
    BuiltinSlashCommand(name="session", description="Show session info and stats"),
    BuiltinSlashCommand(name="changelog", description="Show changelog entries"),
    BuiltinSlashCommand(name="hotkeys", description="Show all keyboard shortcuts"),
    BuiltinSlashCommand(name="fork", description="Create a new fork from a previous message"),
    BuiltinSlashCommand(name="tree", description="Navigate session tree (switch branches)"),
    BuiltinSlashCommand(name="login", description="Login with OAuth provider"),
    BuiltinSlashCommand(name="logout", description="Logout from OAuth provider"),
    BuiltinSlashCommand(name="new", description="Start a new session"),
    BuiltinSlashCommand(name="compact", description="Manually compact the session context"),
    BuiltinSlashCommand(name="resume", description="Resume a different session"),
    BuiltinSlashCommand(
        name="reload",
        description="Reload keybindings, extensions, skills, prompts, and themes",
    ),
    BuiltinSlashCommand(name="quit", description="Quit nu"),
)


__all__ = [
    "BUILTIN_SLASH_COMMANDS",
    "BuiltinSlashCommand",
    "SlashCommandInfo",
    "SlashCommandSource",
]
