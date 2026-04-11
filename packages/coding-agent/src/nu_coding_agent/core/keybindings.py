"""Coding-agent keybinding catalog & manager — direct port of ``packages/coding-agent/src/core/keybindings.ts``.

Extends nu_tui's :data:`TUI_KEYBINDINGS` with the ``app.*`` ids the
interactive coding agent uses (interrupt, clear, model selection, …)
and provides a :class:`KeybindingsManager` subclass that loads / saves
``<agent_dir>/keybindings.json``. The migration map renames the legacy
flat ids (``cursorUp`` → ``tui.editor.cursorUp``, etc.) so config files
written by older versions still load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_tui.keybindings import (
    TUI_KEYBINDINGS,
    KeybindingDefinition,
    KeybindingDefinitions,
    KeybindingsConfig,
)
from nu_tui.keybindings import (
    KeybindingsManager as TuiKeybindingsManager,
)

from nu_coding_agent.config import get_agent_dir

if TYPE_CHECKING:
    from nu_tui.keys import KeyId


# ---------------------------------------------------------------------------
# Keybinding catalog
# ---------------------------------------------------------------------------


_PASTE_IMAGE_KEY: KeyId = "alt+v" if sys.platform == "win32" else "ctrl+v"


_APP_KEYBINDINGS: KeybindingDefinitions = {
    "app.interrupt": KeybindingDefinition(default_keys="escape", description="Cancel or abort"),
    "app.clear": KeybindingDefinition(default_keys="ctrl+c", description="Clear editor"),
    "app.exit": KeybindingDefinition(default_keys="ctrl+d", description="Exit when editor is empty"),
    "app.suspend": KeybindingDefinition(default_keys="ctrl+z", description="Suspend to background"),
    "app.thinking.cycle": KeybindingDefinition(default_keys="shift+tab", description="Cycle thinking level"),
    "app.model.cycleForward": KeybindingDefinition(default_keys="ctrl+p", description="Cycle to next model"),
    "app.model.cycleBackward": KeybindingDefinition(default_keys="shift+ctrl+p", description="Cycle to previous model"),
    "app.model.select": KeybindingDefinition(default_keys="ctrl+l", description="Open model selector"),
    "app.tools.expand": KeybindingDefinition(default_keys="ctrl+o", description="Toggle tool output"),
    "app.thinking.toggle": KeybindingDefinition(default_keys="ctrl+t", description="Toggle thinking blocks"),
    "app.session.toggleNamedFilter": KeybindingDefinition(
        default_keys="ctrl+n", description="Toggle named session filter"
    ),
    "app.editor.external": KeybindingDefinition(default_keys="ctrl+g", description="Open external editor"),
    "app.message.followUp": KeybindingDefinition(default_keys="alt+enter", description="Queue follow-up message"),
    "app.message.dequeue": KeybindingDefinition(default_keys="alt+up", description="Restore queued messages"),
    "app.clipboard.pasteImage": KeybindingDefinition(
        default_keys=_PASTE_IMAGE_KEY, description="Paste image from clipboard"
    ),
    "app.session.new": KeybindingDefinition(default_keys=[], description="Start a new session"),
    "app.session.tree": KeybindingDefinition(default_keys=[], description="Open session tree"),
    "app.session.fork": KeybindingDefinition(default_keys=[], description="Fork current session"),
    "app.session.resume": KeybindingDefinition(default_keys=[], description="Resume a session"),
    "app.tree.foldOrUp": KeybindingDefinition(
        default_keys=["ctrl+left", "alt+left"], description="Fold tree branch or move up"
    ),
    "app.tree.unfoldOrDown": KeybindingDefinition(
        default_keys=["ctrl+right", "alt+right"], description="Unfold tree branch or move down"
    ),
    "app.tree.editLabel": KeybindingDefinition(default_keys="shift+l", description="Edit tree label"),
    "app.tree.toggleLabelTimestamp": KeybindingDefinition(
        default_keys="shift+t", description="Toggle tree label timestamps"
    ),
    "app.session.togglePath": KeybindingDefinition(default_keys="ctrl+p", description="Toggle session path display"),
    "app.session.toggleSort": KeybindingDefinition(default_keys="ctrl+s", description="Toggle session sort mode"),
    "app.session.rename": KeybindingDefinition(default_keys="ctrl+r", description="Rename session"),
    "app.session.delete": KeybindingDefinition(default_keys="ctrl+d", description="Delete session"),
    "app.session.deleteNoninvasive": KeybindingDefinition(
        default_keys="ctrl+backspace", description="Delete session when query is empty"
    ),
}


KEYBINDINGS: KeybindingDefinitions = {**TUI_KEYBINDINGS, **_APP_KEYBINDINGS}
"""Combined ``tui.*`` + ``app.*`` keybinding catalog."""


# ---------------------------------------------------------------------------
# Legacy id migrations
# ---------------------------------------------------------------------------


_KEYBINDING_NAME_MIGRATIONS: dict[str, str] = {
    "cursorUp": "tui.editor.cursorUp",
    "cursorDown": "tui.editor.cursorDown",
    "cursorLeft": "tui.editor.cursorLeft",
    "cursorRight": "tui.editor.cursorRight",
    "cursorWordLeft": "tui.editor.cursorWordLeft",
    "cursorWordRight": "tui.editor.cursorWordRight",
    "cursorLineStart": "tui.editor.cursorLineStart",
    "cursorLineEnd": "tui.editor.cursorLineEnd",
    "jumpForward": "tui.editor.jumpForward",
    "jumpBackward": "tui.editor.jumpBackward",
    "pageUp": "tui.editor.pageUp",
    "pageDown": "tui.editor.pageDown",
    "deleteCharBackward": "tui.editor.deleteCharBackward",
    "deleteCharForward": "tui.editor.deleteCharForward",
    "deleteWordBackward": "tui.editor.deleteWordBackward",
    "deleteWordForward": "tui.editor.deleteWordForward",
    "deleteToLineStart": "tui.editor.deleteToLineStart",
    "deleteToLineEnd": "tui.editor.deleteToLineEnd",
    "yank": "tui.editor.yank",
    "yankPop": "tui.editor.yankPop",
    "undo": "tui.editor.undo",
    "newLine": "tui.input.newLine",
    "submit": "tui.input.submit",
    "tab": "tui.input.tab",
    "copy": "tui.input.copy",
    "selectUp": "tui.select.up",
    "selectDown": "tui.select.down",
    "selectPageUp": "tui.select.pageUp",
    "selectPageDown": "tui.select.pageDown",
    "selectConfirm": "tui.select.confirm",
    "selectCancel": "tui.select.cancel",
    "interrupt": "app.interrupt",
    "clear": "app.clear",
    "exit": "app.exit",
    "suspend": "app.suspend",
    "cycleThinkingLevel": "app.thinking.cycle",
    "cycleModelForward": "app.model.cycleForward",
    "cycleModelBackward": "app.model.cycleBackward",
    "selectModel": "app.model.select",
    "expandTools": "app.tools.expand",
    "toggleThinking": "app.thinking.toggle",
    "toggleSessionNamedFilter": "app.session.toggleNamedFilter",
    "externalEditor": "app.editor.external",
    "followUp": "app.message.followUp",
    "dequeue": "app.message.dequeue",
    "pasteImage": "app.clipboard.pasteImage",
    "newSession": "app.session.new",
    "tree": "app.session.tree",
    "fork": "app.session.fork",
    "resume": "app.session.resume",
    "treeFoldOrUp": "app.tree.foldOrUp",
    "treeUnfoldOrDown": "app.tree.unfoldOrDown",
    "treeEditLabel": "app.tree.editLabel",
    "treeToggleLabelTimestamp": "app.tree.toggleLabelTimestamp",
    "toggleSessionPath": "app.session.togglePath",
    "toggleSessionSort": "app.session.toggleSort",
    "renameSession": "app.session.rename",
    "deleteSession": "app.session.delete",
    "deleteSessionNoninvasive": "app.session.deleteNoninvasive",
}


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _to_keybindings_config(value: Any) -> KeybindingsConfig:
    if not _is_record(value):
        return {}
    config: KeybindingsConfig = {}
    for key, binding in value.items():
        if isinstance(binding, str):
            config[key] = binding
            continue
        if isinstance(binding, list) and all(isinstance(entry, str) for entry in binding):
            config[key] = list(binding)
    return config


def _order_keybindings_config(config: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for keybinding in KEYBINDINGS:
        if keybinding in config:
            ordered[keybinding] = config[keybinding]
    extras = sorted(key for key in config if key not in ordered)
    for key in extras:
        ordered[key] = config[key]
    return ordered


def migrate_keybindings_config(raw_config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Rename legacy flat ids to their ``tui.*`` / ``app.*`` form.

    Returns ``(new_config, migrated)``. The ``migrated`` flag indicates
    that the file should be written back so future loads are cheap.
    """
    config: dict[str, Any] = {}
    migrated = False
    for key, value in raw_config.items():
        next_key = _KEYBINDING_NAME_MIGRATIONS.get(key, key)
        if next_key != key:
            migrated = True
        if key != next_key and next_key in raw_config:
            # Both legacy and new id present — drop the legacy entry.
            migrated = True
            continue
        config[next_key] = value
    return _order_keybindings_config(config), migrated


def _load_raw_config(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        parsed = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# KeybindingsManager subclass
# ---------------------------------------------------------------------------


class KeybindingsManager(TuiKeybindingsManager):
    """nu_coding_agent's keybindings manager — extends nu_tui's with persistence."""

    def __init__(
        self,
        user_bindings: KeybindingsConfig | None = None,
        config_path: str | None = None,
    ) -> None:
        super().__init__(KEYBINDINGS, user_bindings or {})
        self._config_path = config_path

    @classmethod
    def create(cls, agent_dir: str | None = None) -> KeybindingsManager:
        """Build a manager rooted at ``<agent_dir>/keybindings.json``."""
        resolved_agent_dir = agent_dir if agent_dir is not None else get_agent_dir()
        config_path = str(Path(resolved_agent_dir) / "keybindings.json")
        user_bindings = cls._load_from_file(config_path)
        return cls(user_bindings, config_path)

    def reload(self) -> None:
        """Re-read the on-disk config file. No-op if no config path was set."""
        if self._config_path is None:
            return
        self.set_user_bindings(self._load_from_file(self._config_path))

    def get_effective_config(self) -> KeybindingsConfig:
        return self.get_resolved_bindings()

    @staticmethod
    def _load_from_file(path: str) -> KeybindingsConfig:
        raw_config = _load_raw_config(path)
        if not raw_config:
            return {}
        migrated_config, _ = migrate_keybindings_config(raw_config)
        return _to_keybindings_config(migrated_config)


__all__ = [
    "KEYBINDINGS",
    "KeybindingsManager",
    "migrate_keybindings_config",
]
