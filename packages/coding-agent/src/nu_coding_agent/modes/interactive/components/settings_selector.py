"""Advanced selector screens — settings, config, scoped models, user messages.

Port of the remaining TypeScript interactive selector components:
- ``settings-selector.ts``  -> :class:`SettingsSelectorScreen`
- ``config-selector.ts``    -> :class:`ConfigSelectorScreen`
- ``scoped-models-selector.ts`` -> :class:`ScopedModelsSelectorScreen`
- ``user-message-selector.ts``  -> :class:`UserMessageSelectorScreen`

All are Textual :class:`~textual.screen.ModalScreen` overlays that dismiss
with Escape and are launched by slash commands in the REPL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from textual import on
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fuzzy_match(query: str, text: str) -> bool:
    """Simple fuzzy containment check (case-insensitive)."""
    q = query.lower()
    t = text.lower()
    idx = 0
    for ch in q:
        pos = t.find(ch, idx)
        if pos == -1:
            return False
        idx = pos + 1
    return True


# ---------------------------------------------------------------------------
# Data-classes for settings items
# ---------------------------------------------------------------------------


@dataclass
class SettingItem:
    """A single toggleable setting row."""

    id: str
    label: str
    description: str
    current_value: str
    values: list[str]
    # If non-empty, this setting has a submenu of choices rather than
    # simple cycling through ``values``.
    submenu_choices: list[dict[str, str]] = field(default_factory=list)


_THINKING_DESCRIPTIONS: dict[str, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning (~1k tokens)",
    "low": "Light reasoning (~2k tokens)",
    "medium": "Moderate reasoning (~8k tokens)",
    "high": "Deep reasoning (~16k tokens)",
    "xhigh": "Maximum reasoning (~32k tokens)",
}


def _build_settings_items(config: dict[str, Any]) -> list[SettingItem]:
    """Build the canonical list of setting items from a config dict.

    Expected keys mirror :pyclass:`SettingsConfig` from the TS side:

    * ``auto_compact`` (bool)
    * ``show_images`` (bool)
    * ``auto_resize_images`` (bool)
    * ``block_images`` (bool)
    * ``enable_skill_commands`` (bool)
    * ``steering_mode`` (``"all"`` | ``"one-at-a-time"``)
    * ``follow_up_mode`` (``"all"`` | ``"one-at-a-time"``)
    * ``transport`` (``"sse"`` | ``"websocket"`` | ``"auto"``)
    * ``thinking_level`` (str)
    * ``available_thinking_levels`` (list[str])
    * ``current_theme`` (str)
    * ``available_themes`` (list[str])
    * ``hide_thinking_block`` (bool)
    * ``collapse_changelog`` (bool)
    * ``double_escape_action`` (``"fork"`` | ``"tree"`` | ``"none"``)
    * ``tree_filter_mode`` (str)
    * ``show_hardware_cursor`` (bool)
    * ``editor_padding_x`` (int)
    * ``autocomplete_max_visible`` (int)
    * ``quiet_startup`` (bool)
    * ``clear_on_shrink`` (bool)
    """

    def _bval(key: str) -> str:
        return "true" if config.get(key) else "false"

    items: list[SettingItem] = [
        SettingItem(
            id="autocompact",
            label="Auto-compact",
            description="Automatically compact context when it gets too large",
            current_value=_bval("auto_compact"),
            values=["true", "false"],
        ),
        SettingItem(
            id="show-images",
            label="Show images",
            description="Render images inline in terminal",
            current_value=_bval("show_images"),
            values=["true", "false"],
        ),
        SettingItem(
            id="auto-resize-images",
            label="Auto-resize images",
            description="Resize large images to 2000x2000 max for better model compatibility",
            current_value=_bval("auto_resize_images"),
            values=["true", "false"],
        ),
        SettingItem(
            id="block-images",
            label="Block images",
            description="Prevent images from being sent to LLM providers",
            current_value=_bval("block_images"),
            values=["true", "false"],
        ),
        SettingItem(
            id="skill-commands",
            label="Skill commands",
            description="Register skills as /skill:name commands",
            current_value=_bval("enable_skill_commands"),
            values=["true", "false"],
        ),
        SettingItem(
            id="steering-mode",
            label="Steering mode",
            description="Enter while streaming queues steering messages",
            current_value=config.get("steering_mode", "all"),
            values=["one-at-a-time", "all"],
        ),
        SettingItem(
            id="follow-up-mode",
            label="Follow-up mode",
            description="Alt+Enter queues follow-up messages until agent stops",
            current_value=config.get("follow_up_mode", "all"),
            values=["one-at-a-time", "all"],
        ),
        SettingItem(
            id="transport",
            label="Transport",
            description="Preferred transport for providers that support multiple transports",
            current_value=config.get("transport", "auto"),
            values=["sse", "websocket", "auto"],
        ),
        SettingItem(
            id="hide-thinking",
            label="Hide thinking",
            description="Hide thinking blocks in assistant responses",
            current_value=_bval("hide_thinking_block"),
            values=["true", "false"],
        ),
        SettingItem(
            id="collapse-changelog",
            label="Collapse changelog",
            description="Show condensed changelog after updates",
            current_value=_bval("collapse_changelog"),
            values=["true", "false"],
        ),
        SettingItem(
            id="quiet-startup",
            label="Quiet startup",
            description="Disable verbose printing at startup",
            current_value=_bval("quiet_startup"),
            values=["true", "false"],
        ),
        SettingItem(
            id="double-escape-action",
            label="Double-escape action",
            description="Action when pressing Escape twice with empty editor",
            current_value=config.get("double_escape_action", "tree"),
            values=["tree", "fork", "none"],
        ),
        SettingItem(
            id="tree-filter-mode",
            label="Tree filter mode",
            description="Default filter when opening /tree",
            current_value=config.get("tree_filter_mode", "default"),
            values=["default", "no-tools", "user-only", "labeled-only", "all"],
        ),
        SettingItem(
            id="show-hardware-cursor",
            label="Show hardware cursor",
            description="Show the terminal cursor while still positioning it for IME support",
            current_value=_bval("show_hardware_cursor"),
            values=["true", "false"],
        ),
        SettingItem(
            id="editor-padding",
            label="Editor padding",
            description="Horizontal padding for input editor (0-3)",
            current_value=str(config.get("editor_padding_x", 1)),
            values=["0", "1", "2", "3"],
        ),
        SettingItem(
            id="autocomplete-max-visible",
            label="Autocomplete max items",
            description="Max visible items in autocomplete dropdown (3-20)",
            current_value=str(config.get("autocomplete_max_visible", 10)),
            values=["3", "5", "7", "10", "15", "20"],
        ),
        SettingItem(
            id="clear-on-shrink",
            label="Clear on shrink",
            description="Clear empty rows when content shrinks (may cause flicker)",
            current_value=_bval("clear_on_shrink"),
            values=["true", "false"],
        ),
    ]

    # Thinking level — uses a submenu of labelled choices
    available_levels: list[str] = config.get("available_thinking_levels", list(_THINKING_DESCRIPTIONS.keys()))
    items.append(
        SettingItem(
            id="thinking",
            label="Thinking level",
            description="Reasoning depth for thinking-capable models",
            current_value=config.get("thinking_level", "medium"),
            values=available_levels,
            submenu_choices=[
                {"value": lvl, "label": lvl, "description": _THINKING_DESCRIPTIONS.get(lvl, "")}
                for lvl in available_levels
            ],
        ),
    )

    # Theme — submenu
    available_themes: list[str] = config.get("available_themes", ["dark", "light"])
    items.append(
        SettingItem(
            id="theme",
            label="Theme",
            description="Color theme for the interface",
            current_value=config.get("current_theme", "dark"),
            values=available_themes,
            submenu_choices=[{"value": t, "label": t} for t in available_themes],
        ),
    )

    return items


# ---------------------------------------------------------------------------
# SettingsSelectorScreen — /settings (full)
# ---------------------------------------------------------------------------


class SettingsSelectorScreen(ModalScreen["dict[str, str] | None"]):
    """Full settings panel with toggleable items.

    Port of ``SettingsSelectorComponent`` (settings-selector.ts).

    Each setting can be cycled through its allowed values with Enter.
    The screen dismisses with a ``dict`` mapping setting-id -> new-value
    for any settings that were changed, or ``None`` on cancel.
    """

    CSS = """
    SettingsSelectorScreen {
        align: center middle;
    }
    #ss-box {
        width: 80;
        max-height: 85%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #ss-search {
        margin: 1 0;
    }
    #ss-list {
        height: auto;
        max-height: 22;
    }
    #ss-status {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "move_up", "Up"),
        Binding("down", "move_down", "Down"),
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self._config = config
        self._items = _build_settings_items(config)
        self._filtered: list[SettingItem] = list(self._items)
        self._selected_index = 0
        self._changes: dict[str, str] = {}

    # -- compose / rebuild --------------------------------------------------

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ss-box"):
            yield Static("[bold]Settings[/bold]  [dim](Enter to toggle, Escape to close)[/dim]")
            yield Input(placeholder="Filter settings...", id="ss-search")
            yield ListView(id="ss-list")
            yield Static("", id="ss-status")

    def on_mount(self) -> None:
        self.query_one("#ss-search", Input).focus()
        self._apply_filter("")

    def _apply_filter(self, query: str) -> None:
        if query.strip():
            self._filtered = [
                item
                for item in self._items
                if _fuzzy_match(query.strip(), f"{item.label} {item.description} {item.id}")
            ]
        else:
            self._filtered = list(self._items)
        self._selected_index = min(self._selected_index, max(0, len(self._filtered) - 1))
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#ss-list", ListView)
        lv.clear()

        max_visible = 12
        start = max(0, min(self._selected_index - max_visible // 2, len(self._filtered) - max_visible))
        end = min(start + max_visible, len(self._filtered))

        for i in range(start, end):
            item = self._filtered[i]
            is_sel = i == self._selected_index
            marker = "-> " if is_sel else "   "
            val = self._changes.get(item.id, item.current_value)
            line = f"{marker}{item.label}: [{val}]"
            lv.append(ListItem(Label(line), name=str(i)))

        status = self.query_one("#ss-status", Static)
        if not self._filtered:
            status.update("[dim]No matching settings[/dim]")
        else:
            item = self._filtered[self._selected_index]
            changed = " (changed)" if item.id in self._changes else ""
            status.update(f"[dim]{item.description}{changed}[/dim]")

    # -- input events -------------------------------------------------------

    @on(Input.Changed, "#ss-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._selected_index = 0
        self._apply_filter(event.value)

    @on(Input.Submitted, "#ss-search")
    def _on_search_submit(self, _event: Input.Submitted) -> None:
        self._toggle_current()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx_str = event.item.name
        if idx_str is None:
            return
        try:
            idx = int(idx_str)
        except ValueError:
            return
        if 0 <= idx < len(self._filtered):
            self._selected_index = idx
            self._toggle_current()

    # -- actions ------------------------------------------------------------

    def action_move_up(self) -> None:
        if not self._filtered:
            return
        self._selected_index = (self._selected_index - 1) % len(self._filtered)
        self._rebuild_list()

    def action_move_down(self) -> None:
        if not self._filtered:
            return
        self._selected_index = (self._selected_index + 1) % len(self._filtered)
        self._rebuild_list()

    def action_cancel(self) -> None:
        self.dismiss(self._changes if self._changes else None)

    # -- cycling ------------------------------------------------------------

    def _toggle_current(self) -> None:
        if not self._filtered:
            return
        item = self._filtered[self._selected_index]
        current = self._changes.get(item.id, item.current_value)
        values = item.values
        try:
            idx = values.index(current)
        except ValueError:
            idx = -1
        new_idx = (idx + 1) % len(values)
        new_val = values[new_idx]
        if new_val == item.current_value:
            self._changes.pop(item.id, None)
        else:
            self._changes[item.id] = new_val
        self._rebuild_list()


# ---------------------------------------------------------------------------
# Resource / Config data-classes
# ---------------------------------------------------------------------------


@dataclass
class ResourceItem:
    """A single toggleable resource (extension, skill, prompt, or theme)."""

    path: str
    enabled: bool
    resource_type: str  # "extensions" | "skills" | "prompts" | "themes"
    display_name: str
    group_key: str
    scope: str  # "user" | "project"
    origin: str  # "package" | "top-level"
    source: str


@dataclass
class ResourceSubgroup:
    """Group of resources sharing a type within a parent group."""

    resource_type: str
    label: str
    items: list[ResourceItem] = field(default_factory=list)


@dataclass
class ResourceGroup:
    """Top-level grouping by scope + origin + source."""

    key: str
    label: str
    scope: str
    origin: str
    source: str
    subgroups: list[ResourceSubgroup] = field(default_factory=list)


_RESOURCE_TYPE_LABELS: dict[str, str] = {
    "extensions": "Extensions",
    "skills": "Skills",
    "prompts": "Prompts",
    "themes": "Themes",
}

_TYPE_ORDER: dict[str, int] = {"extensions": 0, "skills": 1, "prompts": 2, "themes": 3}


def _build_config_groups(resolved: dict[str, Any]) -> list[ResourceGroup]:
    """Build grouped resources from a resolved-paths dict.

    ``resolved`` should map resource type ("extensions", "skills", etc.) to a
    list of dicts with keys ``path``, ``enabled``, and ``metadata`` (a dict
    with ``scope``, ``origin``, ``source``).
    """
    import os  # noqa: PLC0415

    group_map: dict[str, ResourceGroup] = {}

    for rtype in ("extensions", "skills", "prompts", "themes"):
        for res in resolved.get(rtype, []):
            path = res.get("path", "")
            enabled = res.get("enabled", True)
            meta = res.get("metadata", {})
            scope = meta.get("scope", "user")
            origin = meta.get("origin", "top-level")
            source = meta.get("source", "auto")

            group_key = f"{origin}:{scope}:{source}"
            if group_key not in group_map:
                if origin == "package":
                    label = f"{source} ({scope})"
                elif source == "auto":
                    label = "User (~/.pi/agent/)" if scope == "user" else "Project (.pi/)"
                else:
                    label = "User settings" if scope == "user" else "Project settings"
                group_map[group_key] = ResourceGroup(
                    key=group_key,
                    label=label,
                    scope=scope,
                    origin=origin,
                    source=source,
                )

            group = group_map[group_key]
            subgroup = next((sg for sg in group.subgroups if sg.resource_type == rtype), None)
            if subgroup is None:
                subgroup = ResourceSubgroup(
                    resource_type=rtype,
                    label=_RESOURCE_TYPE_LABELS.get(rtype, rtype),
                )
                group.subgroups.append(subgroup)

            filename = os.path.basename(path)
            parent = os.path.basename(os.path.dirname(path))
            if rtype == "extensions" and parent != "extensions":
                display_name = f"{parent}/{filename}"
            elif rtype == "skills" and filename == "SKILL.md":
                display_name = parent
            else:
                display_name = filename

            subgroup.items.append(
                ResourceItem(
                    path=path,
                    enabled=enabled,
                    resource_type=rtype,
                    display_name=display_name,
                    group_key=group_key,
                    scope=scope,
                    origin=origin,
                    source=source,
                ),
            )

    groups = sorted(
        group_map.values(),
        key=lambda g: (0 if g.origin == "package" else 1, 0 if g.scope == "user" else 1, g.source),
    )
    for group in groups:
        group.subgroups.sort(key=lambda sg: _TYPE_ORDER.get(sg.resource_type, 99))
        for sg in group.subgroups:
            sg.items.sort(key=lambda it: it.display_name)
    return groups


# Flat-entry type tag for iteration
@dataclass
class _FlatGroup:
    group: ResourceGroup


@dataclass
class _FlatSubgroup:
    subgroup: ResourceSubgroup
    group: ResourceGroup


@dataclass
class _FlatItem:
    item: ResourceItem


_FlatEntry = _FlatGroup | _FlatSubgroup | _FlatItem


def _flatten_groups(groups: list[ResourceGroup]) -> list[_FlatEntry]:
    flat: list[_FlatEntry] = []
    for group in groups:
        flat.append(_FlatGroup(group))
        for subgroup in group.subgroups:
            flat.append(_FlatSubgroup(subgroup, group))
            for item in subgroup.items:
                flat.append(_FlatItem(item))
    return flat


# ---------------------------------------------------------------------------
# ConfigSelectorScreen — /config
# ---------------------------------------------------------------------------


class ConfigSelectorScreen(ModalScreen[None]):
    """Resource configuration panel — extensions, skills, prompts, themes.

    Port of ``ConfigSelectorComponent`` (config-selector.ts).

    Displays installed resources grouped by origin/scope, lets the user
    toggle enable/disable with Enter or Space.  Search by typing.
    Dismisses with ``None`` on Escape.
    """

    CSS = """
    ConfigSelectorScreen {
        align: center middle;
    }
    #cs-box {
        width: 80;
        max-height: 85%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #cs-search {
        margin: 1 0;
    }
    #cs-list {
        height: auto;
        max-height: 22;
    }
    #cs-status {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "move_up", "Up"),
        Binding("down", "move_down", "Down"),
        Binding("space", "toggle_item", "Toggle"),
    ]

    def __init__(
        self,
        resolved_paths: dict[str, Any],
        *,
        on_toggle: Any | None = None,
    ) -> None:
        super().__init__()
        self._groups = _build_config_groups(resolved_paths)
        self._flat: list[_FlatEntry] = _flatten_groups(self._groups)
        self._filtered: list[_FlatEntry] = list(self._flat)
        self._selected_index = self._first_item_index(self._filtered)
        self._on_toggle = on_toggle

    @staticmethod
    def _first_item_index(entries: list[_FlatEntry]) -> int:
        for i, e in enumerate(entries):
            if isinstance(e, _FlatItem):
                return i
        return 0

    # -- compose / rebuild --------------------------------------------------

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="cs-box"):
            yield Static("[bold]Resource Configuration[/bold]  [dim](Space/Enter toggle, Esc close)[/dim]")
            yield Input(placeholder="Type to filter resources...", id="cs-search")
            yield ListView(id="cs-list")
            yield Static("", id="cs-status")

    def on_mount(self) -> None:
        self.query_one("#cs-search", Input).focus()
        self._rebuild_list()

    def _apply_filter(self, query: str) -> None:
        if not query.strip():
            self._filtered = list(self._flat)
            self._selected_index = self._first_item_index(self._filtered)
            self._rebuild_list()
            return

        lq = query.lower()
        matching_items: set[str] = set()
        for entry in self._flat:
            if isinstance(entry, _FlatItem):
                it = entry.item
                if lq in it.display_name.lower() or lq in it.resource_type.lower() or lq in it.path.lower():
                    matching_items.add(it.path)

        # Include group/subgroup headers that contain matching items
        matching_subgroups: set[id] = set()  # type: ignore[type-arg]
        matching_group_keys: set[str] = set()
        for group in self._groups:
            for sg in group.subgroups:
                for it in sg.items:
                    if it.path in matching_items:
                        matching_subgroups.add(id(sg))
                        matching_group_keys.add(group.key)

        self._filtered = []
        for entry in self._flat:
            if (
                (isinstance(entry, _FlatGroup) and entry.group.key in matching_group_keys)
                or (isinstance(entry, _FlatSubgroup) and id(entry.subgroup) in matching_subgroups)
                or (isinstance(entry, _FlatItem) and entry.item.path in matching_items)
            ):
                self._filtered.append(entry)

        self._selected_index = self._first_item_index(self._filtered)
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#cs-list", ListView)
        lv.clear()

        if not self._filtered:
            lv.append(ListItem(Label("[dim]No resources found[/dim]")))
            self.query_one("#cs-status", Static).update("")
            return

        max_visible = 15
        start = max(0, min(self._selected_index - max_visible // 2, len(self._filtered) - max_visible))
        end = min(start + max_visible, len(self._filtered))

        for i in range(start, end):
            entry = self._filtered[i]
            is_sel = i == self._selected_index

            if isinstance(entry, _FlatGroup):
                line = f"  [bold]{entry.group.label}[/bold]"
                lv.append(ListItem(Label(line), name=f"g:{i}"))
            elif isinstance(entry, _FlatSubgroup):
                line = f"    [dim]{entry.subgroup.label}[/dim]"
                lv.append(ListItem(Label(line), name=f"s:{i}"))
            else:
                it = entry.item
                cursor = "> " if is_sel else "  "
                checkbox = "[x]" if it.enabled else "[ ]"
                name = f"[bold]{it.display_name}[/bold]" if is_sel else it.display_name
                line = f"{cursor}      {checkbox} {name}"
                lv.append(ListItem(Label(line), name=f"i:{i}"))

        # Scroll indicator
        status = self.query_one("#cs-status", Static)
        if start > 0 or end < len(self._filtered):
            status.update(f"[dim]({self._selected_index + 1}/{len(self._filtered)})[/dim]")
        else:
            status.update("")

    # -- navigation ---------------------------------------------------------

    def _find_next_item(self, from_idx: int, direction: int) -> int:
        idx = from_idx + direction
        while 0 <= idx < len(self._filtered):
            if isinstance(self._filtered[idx], _FlatItem):
                return idx
            idx += direction
        return from_idx

    def action_move_up(self) -> None:
        self._selected_index = self._find_next_item(self._selected_index, -1)
        self._rebuild_list()

    def action_move_down(self) -> None:
        self._selected_index = self._find_next_item(self._selected_index, 1)
        self._rebuild_list()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_toggle_item(self) -> None:
        self._toggle_selected()

    # -- input events -------------------------------------------------------

    @on(Input.Changed, "#cs-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._apply_filter(event.value)

    @on(Input.Submitted, "#cs-search")
    def _on_search_submit(self, _event: Input.Submitted) -> None:
        self._toggle_selected()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        name = event.item.name or ""
        if name.startswith("i:"):
            try:
                idx = int(name[2:])
            except ValueError:
                return
            self._selected_index = idx
            self._toggle_selected()

    # -- toggle logic -------------------------------------------------------

    def _toggle_selected(self) -> None:
        if not self._filtered:
            return
        entry = self._filtered[self._selected_index]
        if not isinstance(entry, _FlatItem):
            return
        entry.item.enabled = not entry.item.enabled
        if self._on_toggle is not None:
            self._on_toggle(entry.item, entry.item.enabled)
        self._rebuild_list()


# ---------------------------------------------------------------------------
# ScopedModelsSelectorScreen — /scoped-models
# ---------------------------------------------------------------------------


class ScopedModelsSelectorScreen(ModalScreen["list[str] | None"]):
    """Enable/disable models for Ctrl+P cycling.

    Port of ``ScopedModelsSelectorComponent`` (scoped-models-selector.ts).

    Changes are session-only until explicitly persisted with Ctrl+S.
    Dismisses with the list of enabled model IDs (or ``None`` on cancel).
    """

    CSS = """
    ScopedModelsSelectorScreen {
        align: center middle;
    }
    #sm-box {
        width: 75;
        max-height: 85%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #sm-search {
        margin: 1 0;
    }
    #sm-list {
        height: auto;
        max-height: 20;
    }
    #sm-status {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "move_up", "Up"),
        Binding("down", "move_down", "Down"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+a", "enable_all", "Enable all"),
        Binding("ctrl+x", "clear_all", "Clear all"),
    ]

    def __init__(
        self,
        all_models: list[dict[str, str]],
        enabled_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        # all_models: list of dicts with "id", "provider", "name"
        self._all_models = all_models
        self._all_ids = [f"{m['provider']}/{m['id']}" for m in all_models]
        self._models_by_id: dict[str, dict[str, str]] = {f"{m['provider']}/{m['id']}": m for m in all_models}
        # None means "all enabled" (no filter)
        self._enabled_ids: list[str] | None = list(enabled_ids) if enabled_ids is not None else None
        self._filtered: list[dict[str, Any]] = []
        self._selected_index = 0
        self._is_dirty = False

    def _is_enabled(self, full_id: str) -> bool:
        return self._enabled_ids is None or full_id in self._enabled_ids

    def _get_sorted_ids(self) -> list[str]:
        if self._enabled_ids is None:
            return list(self._all_ids)
        enabled_set = set(self._enabled_ids)
        return list(self._enabled_ids) + [aid for aid in self._all_ids if aid not in enabled_set]

    def _build_items(self) -> list[dict[str, Any]]:
        return [
            {"full_id": fid, "model": self._models_by_id.get(fid, {}), "enabled": self._is_enabled(fid)}
            for fid in self._get_sorted_ids()
            if fid in self._models_by_id
        ]

    # -- compose / rebuild --------------------------------------------------

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="sm-box"):
            yield Static("[bold]Model Configuration[/bold]")
            yield Static("[dim]Session-only. Ctrl+S to save to settings.[/dim]")
            yield Input(placeholder="Search models...", id="sm-search")
            yield ListView(id="sm-list")
            yield Static("", id="sm-status")

    def on_mount(self) -> None:
        self.query_one("#sm-search", Input).focus()
        self._refresh()

    def _refresh(self) -> None:
        query = self.query_one("#sm-search", Input).value
        items = self._build_items()
        if query.strip():
            items = [
                it
                for it in items
                if _fuzzy_match(query.strip(), f"{it['model'].get('id', '')} {it['model'].get('provider', '')}")
            ]
        self._filtered = items
        self._selected_index = min(self._selected_index, max(0, len(self._filtered) - 1))
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#sm-list", ListView)
        lv.clear()

        if not self._filtered:
            lv.append(ListItem(Label("[dim]No matching models[/dim]")))
            self._update_status()
            return

        max_visible = 15
        start = max(0, min(self._selected_index - max_visible // 2, len(self._filtered) - max_visible))
        end = min(start + max_visible, len(self._filtered))
        all_enabled = self._enabled_ids is None

        for i in range(start, end):
            it = self._filtered[i]
            is_sel = i == self._selected_index
            marker = "-> " if is_sel else "   "
            model_id = it["model"].get("id", "?")
            provider = it["model"].get("provider", "?")
            if all_enabled:
                status = ""
            elif it["enabled"]:
                status = " [green]Y[/green]"
            else:
                status = " [dim]X[/dim]"
            label = f"{marker}{model_id} [{provider}]{status}"
            lv.append(ListItem(Label(label), name=str(i)))

        self._update_status()

    def _update_status(self) -> None:
        status = self.query_one("#sm-status", Static)
        enabled_count = len(self._enabled_ids) if self._enabled_ids is not None else len(self._all_ids)
        count_text = "all enabled" if self._enabled_ids is None else f"{enabled_count}/{len(self._all_ids)} enabled"
        parts = ["Enter toggle", "^A all", "^X clear", "^S save", count_text]
        dirty = " (unsaved)" if self._is_dirty else ""
        status.update(f"[dim]{' | '.join(parts)}{dirty}[/dim]")

    # -- input events -------------------------------------------------------

    @on(Input.Changed, "#sm-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._selected_index = 0
        self._refresh()

    @on(Input.Submitted, "#sm-search")
    def _on_search_submit(self, _event: Input.Submitted) -> None:
        self._toggle_current()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx_str = event.item.name
        if idx_str is None:
            return
        try:
            idx = int(idx_str)
        except ValueError:
            return
        if 0 <= idx < len(self._filtered):
            self._selected_index = idx
            self._toggle_current()

    # -- actions ------------------------------------------------------------

    def action_move_up(self) -> None:
        if not self._filtered:
            return
        self._selected_index = (self._selected_index - 1) % len(self._filtered)
        self._rebuild_list()

    def action_move_down(self) -> None:
        if not self._filtered:
            return
        self._selected_index = (self._selected_index + 1) % len(self._filtered)
        self._rebuild_list()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        result = self._enabled_ids if self._enabled_ids is not None else list(self._all_ids)
        self.dismiss(result)

    def action_enable_all(self) -> None:
        query = self.query_one("#sm-search", Input).value
        if query.strip():
            # Enable only filtered items
            target_ids = {it["full_id"] for it in self._filtered}
            if self._enabled_ids is None:
                pass  # Already all enabled
            else:
                for tid in target_ids:
                    if tid not in self._enabled_ids:
                        self._enabled_ids.append(tid)
                if len(self._enabled_ids) >= len(self._all_ids):
                    self._enabled_ids = None
        else:
            self._enabled_ids = None
        self._is_dirty = True
        self._refresh()

    def action_clear_all(self) -> None:
        query = self.query_one("#sm-search", Input).value
        if query.strip():
            target_ids = {it["full_id"] for it in self._filtered}
            if self._enabled_ids is None:
                self._enabled_ids = [aid for aid in self._all_ids if aid not in target_ids]
            else:
                self._enabled_ids = [eid for eid in self._enabled_ids if eid not in target_ids]
        else:
            self._enabled_ids = []
        self._is_dirty = True
        self._refresh()

    # -- toggle -------------------------------------------------------------

    def _toggle_current(self) -> None:
        if not self._filtered:
            return
        item = self._filtered[self._selected_index]
        full_id = item["full_id"]
        if self._enabled_ids is None:
            # First toggle: switch from "all" to explicit list minus this one
            self._enabled_ids = [aid for aid in self._all_ids if aid != full_id]
        elif full_id in self._enabled_ids:
            self._enabled_ids.remove(full_id)
        else:
            self._enabled_ids.append(full_id)
        self._is_dirty = True
        self._refresh()


# ---------------------------------------------------------------------------
# UserMessageSelectorScreen — /branch (user message picker)
# ---------------------------------------------------------------------------


class UserMessageSelectorScreen(ModalScreen["str | None"]):
    """Shows user messages from the current session for branching.

    Port of ``UserMessageSelectorComponent`` (user-message-selector.ts).

    Each entry shows the message text; the most recent message is
    pre-selected.  Returns the ``entry_id`` of the chosen message, or
    ``None`` on cancel.
    """

    CSS = """
    UserMessageSelectorScreen {
        align: center middle;
    }
    #um-box {
        width: 80;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #um-list {
        height: auto;
        max-height: 22;
    }
    #um-status {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "move_up", "Up"),
        Binding("down", "move_down", "Down"),
    ]

    def __init__(self, messages: list[dict[str, str]]) -> None:
        """*messages*: list of dicts with ``id`` and ``text`` keys."""
        super().__init__()
        self._messages = messages
        self._selected_index = max(0, len(messages) - 1)

    # -- compose / rebuild --------------------------------------------------

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="um-box"):
            yield Static("[bold]Branch from Message[/bold]")
            yield Static("[dim]Select a message to create a new branch from that point[/dim]")
            yield ListView(id="um-list")
            yield Static("", id="um-status")

    def on_mount(self) -> None:
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#um-list", ListView)
        lv.clear()

        if not self._messages:
            lv.append(ListItem(Label("[dim]No user messages found[/dim]")))
            return

        max_visible = 10
        start = max(0, min(self._selected_index - max_visible // 2, len(self._messages) - max_visible))
        end = min(start + max_visible, len(self._messages))

        for i in range(start, end):
            msg = self._messages[i]
            is_sel = i == self._selected_index
            cursor = "> " if is_sel else "  "
            text = msg.get("text", "").replace("\n", " ").strip()[:120]
            line = f"{cursor}[bold]{text}[/bold]" if is_sel else f"{cursor}{text}"
            position = f"  [dim]Message {i + 1} of {len(self._messages)}[/dim]"
            # Two lines per message: text + metadata
            lv.append(ListItem(Label(f"{line}\n{position}"), name=msg.get("id", "")))

        status = self.query_one("#um-status", Static)
        if start > 0 or end < len(self._messages):
            status.update(f"[dim]({self._selected_index + 1}/{len(self._messages)})[/dim]")
        else:
            status.update("")

    # -- actions ------------------------------------------------------------

    def action_move_up(self) -> None:
        if not self._messages:
            return
        self._selected_index = (self._selected_index - 1) % len(self._messages)
        self._rebuild_list()

    def action_move_down(self) -> None:
        if not self._messages:
            return
        self._selected_index = (self._selected_index + 1) % len(self._messages)
        self._rebuild_list()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        entry_id = event.item.name
        if entry_id:
            self.dismiss(entry_id)


__all__ = [
    "ConfigSelectorScreen",
    "ResourceGroup",
    "ResourceItem",
    "ResourceSubgroup",
    "ScopedModelsSelectorScreen",
    "SettingItem",
    "SettingsSelectorScreen",
    "UserMessageSelectorScreen",
]
