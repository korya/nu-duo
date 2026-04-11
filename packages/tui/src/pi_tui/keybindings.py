"""Keybinding registry and resolver.

Direct port of ``packages/tui/src/keybindings.ts``. Definitions provide
default keys; user bindings override them. The manager exposes
``matches`` for runtime dispatch and tracks conflicts when two
bindings claim the same key.

The TypeScript version uses interface declaration merging to let
downstream packages add new ``Keybinding`` literals; Python has no
direct equivalent, so :data:`KeybindingDefinitions` is just a plain
``dict[str, KeybindingDefinition]`` and downstream packages register
their own ids by extending the dict at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_tui.keys import KeyId, matches_key


@dataclass(slots=True)
class KeybindingDefinition:
    """Default keys + description for a single binding."""

    default_keys: KeyId | list[KeyId]
    description: str | None = None


KeybindingDefinitions = dict[str, KeybindingDefinition]
"""Mapping of keybinding id → definition."""

KeybindingsConfig = dict[str, KeyId | list[KeyId] | None]
"""User-supplied override mapping (id → key or list of keys)."""


@dataclass(slots=True)
class KeybindingConflict:
    """A key claimed by more than one binding."""

    key: KeyId
    keybindings: list[str]


def _normalize_keys(keys: KeyId | list[KeyId] | None) -> list[KeyId]:
    if keys is None:
        return []
    key_list = [keys] if isinstance(keys, str) else list(keys)
    seen: set[KeyId] = set()
    result: list[KeyId] = []
    for key in key_list:
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


class KeybindingsManager:
    """Resolves user/default key assignments and matches input against them."""

    def __init__(
        self,
        definitions: KeybindingDefinitions,
        user_bindings: KeybindingsConfig | None = None,
    ) -> None:
        self._definitions = definitions
        self._user_bindings: KeybindingsConfig = dict(user_bindings or {})
        self._keys_by_id: dict[str, list[KeyId]] = {}
        self._conflicts: list[KeybindingConflict] = []
        self._rebuild()

    def _rebuild(self) -> None:
        self._keys_by_id.clear()
        self._conflicts = []

        # Detect conflicts: keys claimed by more than one user binding.
        user_claims: dict[KeyId, set[str]] = {}
        for keybinding, keys in self._user_bindings.items():
            if keybinding not in self._definitions:
                continue
            for key in _normalize_keys(keys):
                user_claims.setdefault(key, set()).add(keybinding)

        for key, keybindings in user_claims.items():
            if len(keybindings) > 1:
                self._conflicts.append(KeybindingConflict(key=key, keybindings=list(keybindings)))

        # Resolve final keys: user override wins, otherwise default.
        for id_, definition in self._definitions.items():
            user_keys = self._user_bindings.get(id_)
            if id_ in self._user_bindings:
                # Explicit override (even if it normalises to []).
                keys = _normalize_keys(user_keys)
            else:
                keys = _normalize_keys(definition.default_keys)
            self._keys_by_id[id_] = keys

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def matches(self, data: str, keybinding: str) -> bool:
        """Return ``True`` iff ``data`` matches one of ``keybinding``'s keys."""
        keys = self._keys_by_id.get(keybinding, [])
        return any(matches_key(data, key) for key in keys)

    def get_keys(self, keybinding: str) -> list[KeyId]:
        return list(self._keys_by_id.get(keybinding, []))

    def get_definition(self, keybinding: str) -> KeybindingDefinition:
        return self._definitions[keybinding]

    def get_conflicts(self) -> list[KeybindingConflict]:
        return [KeybindingConflict(key=c.key, keybindings=list(c.keybindings)) for c in self._conflicts]

    def set_user_bindings(self, user_bindings: KeybindingsConfig) -> None:
        self._user_bindings = dict(user_bindings)
        self._rebuild()

    def get_user_bindings(self) -> KeybindingsConfig:
        return dict(self._user_bindings)

    def get_resolved_bindings(self) -> KeybindingsConfig:
        resolved: KeybindingsConfig = {}
        for id_ in self._definitions:
            keys = self._keys_by_id.get(id_, [])
            resolved[id_] = keys[0] if len(keys) == 1 else list(keys)
        return resolved


# ---------------------------------------------------------------------------
# Built-in TUI keybindings (mirror of upstream ``TUI_KEYBINDINGS``)
# ---------------------------------------------------------------------------


TUI_KEYBINDINGS: KeybindingDefinitions = {
    "tui.editor.cursorUp": KeybindingDefinition(default_keys="up", description="Move cursor up"),
    "tui.editor.cursorDown": KeybindingDefinition(default_keys="down", description="Move cursor down"),
    "tui.editor.cursorLeft": KeybindingDefinition(default_keys=["left", "ctrl+b"], description="Move cursor left"),
    "tui.editor.cursorRight": KeybindingDefinition(default_keys=["right", "ctrl+f"], description="Move cursor right"),
    "tui.editor.cursorWordLeft": KeybindingDefinition(
        default_keys=["alt+left", "ctrl+left", "alt+b"],
        description="Move cursor word left",
    ),
    "tui.editor.cursorWordRight": KeybindingDefinition(
        default_keys=["alt+right", "ctrl+right", "alt+f"],
        description="Move cursor word right",
    ),
    "tui.editor.cursorLineStart": KeybindingDefinition(
        default_keys=["home", "ctrl+a"], description="Move to line start"
    ),
    "tui.editor.cursorLineEnd": KeybindingDefinition(default_keys=["end", "ctrl+e"], description="Move to line end"),
    "tui.editor.jumpForward": KeybindingDefinition(default_keys="ctrl+]", description="Jump forward to character"),
    "tui.editor.jumpBackward": KeybindingDefinition(
        default_keys="ctrl+alt+]", description="Jump backward to character"
    ),
    "tui.editor.pageUp": KeybindingDefinition(default_keys="pageUp", description="Page up"),
    "tui.editor.pageDown": KeybindingDefinition(default_keys="pageDown", description="Page down"),
    "tui.editor.deleteCharBackward": KeybindingDefinition(
        default_keys="backspace", description="Delete character backward"
    ),
    "tui.editor.deleteCharForward": KeybindingDefinition(
        default_keys=["delete", "ctrl+d"], description="Delete character forward"
    ),
    "tui.editor.deleteWordBackward": KeybindingDefinition(
        default_keys=["ctrl+w", "alt+backspace"], description="Delete word backward"
    ),
    "tui.editor.deleteWordForward": KeybindingDefinition(
        default_keys=["alt+d", "alt+delete"], description="Delete word forward"
    ),
    "tui.editor.deleteToLineStart": KeybindingDefinition(default_keys="ctrl+u", description="Delete to line start"),
    "tui.editor.deleteToLineEnd": KeybindingDefinition(default_keys="ctrl+k", description="Delete to line end"),
    "tui.editor.yank": KeybindingDefinition(default_keys="ctrl+y", description="Yank"),
    "tui.editor.yankPop": KeybindingDefinition(default_keys="alt+y", description="Yank pop"),
    "tui.editor.undo": KeybindingDefinition(default_keys="ctrl+-", description="Undo"),
    "tui.input.newLine": KeybindingDefinition(default_keys="shift+enter", description="Insert newline"),
    "tui.input.submit": KeybindingDefinition(default_keys="enter", description="Submit input"),
    "tui.input.tab": KeybindingDefinition(default_keys="tab", description="Tab / autocomplete"),
    "tui.input.copy": KeybindingDefinition(default_keys="ctrl+c", description="Copy selection"),
    "tui.select.up": KeybindingDefinition(default_keys="up", description="Move selection up"),
    "tui.select.down": KeybindingDefinition(default_keys="down", description="Move selection down"),
    "tui.select.pageUp": KeybindingDefinition(default_keys="pageUp", description="Selection page up"),
    "tui.select.pageDown": KeybindingDefinition(default_keys="pageDown", description="Selection page down"),
    "tui.select.confirm": KeybindingDefinition(default_keys="enter", description="Confirm selection"),
    "tui.select.cancel": KeybindingDefinition(default_keys=["escape", "ctrl+c"], description="Cancel selection"),
}


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


_global_keybindings: KeybindingsManager | None = None


def set_keybindings(keybindings: KeybindingsManager) -> None:
    global _global_keybindings  # noqa: PLW0603 — module-level singleton
    _global_keybindings = keybindings


def get_keybindings() -> KeybindingsManager:
    global _global_keybindings  # noqa: PLW0603
    if _global_keybindings is None:
        _global_keybindings = KeybindingsManager(TUI_KEYBINDINGS)
    return _global_keybindings


__all__ = [
    "TUI_KEYBINDINGS",
    "KeybindingConflict",
    "KeybindingDefinition",
    "KeybindingDefinitions",
    "KeybindingsConfig",
    "KeybindingsManager",
    "get_keybindings",
    "set_keybindings",
]
