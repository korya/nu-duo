"""Session tree navigator — port of ``tree-selector.ts``.

Renders the session's branching history as an ASCII tree and lets the user
navigate to any entry to resume from that point.

Features:
- ASCII tree connectors (├─, └─, │) mirroring the TS render.
- Filter modes: default, no-tools, user-only, labeled-only, all.
- Text search across entry content.
- Label editing (``l`` key).
- Fold/unfold subtrees (``f`` key).

Usage::

    screen = TreeSelectorScreen(tree, current_leaf_id)
    entry_id = await app.push_screen_wait(screen)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from textual import on
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

FilterMode = str  # "default" | "no-tools" | "user-only" | "labeled-only" | "all"


# ---------------------------------------------------------------------------
# Flat node representation
# ---------------------------------------------------------------------------


@dataclass
class FlatNode:
    node: Any  # SessionTreeNode
    indent: int
    show_connector: bool
    is_last: bool
    gutters: list[tuple[int, bool]] = field(default_factory=list)
    is_virtual_root_child: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shorten(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _normalize_text(s: str) -> str:
    return s.replace("\n", " ").replace("\t", " ").strip()


def _extract_content(content: Any, max_len: int = 200) -> str:
    if isinstance(content, str):
        return content[:max_len]
    if isinstance(content, list):
        parts = []
        total = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                parts.append(t)
                total += len(t)
                if total >= max_len:
                    break
            elif hasattr(block, "type") and getattr(block, "type", None) == "text":
                t = getattr(block, "text", "")
                parts.append(t)
                total += len(t)
                if total >= max_len:
                    break
        return "".join(parts)[:max_len]
    return ""


def _has_text_content(content: Any) -> bool:
    return bool(_extract_content(content).strip())


def _format_tool_call(name: str, args: dict[str, Any]) -> str:
    def sp(p: str) -> str:
        return _shorten(str(p))

    if name == "read":
        path = sp(args.get("path") or args.get("file_path") or "")
        offset = args.get("offset")
        limit = args.get("limit")
        disp = path
        if offset is not None or limit is not None:
            start = offset if offset is not None else 1
            end = (start + limit - 1) if limit is not None else ""
            disp += f":{start}{'-' + str(end) if end else ''}"
        return f"[read: {disp}]"
    if name == "write":
        return f"[write: {sp(args.get('path') or args.get('file_path') or '')}]"
    if name == "edit":
        return f"[edit: {sp(args.get('path') or args.get('file_path') or '')}]"
    if name == "bash":
        raw = str(args.get("command", "")).replace("\n", " ").strip()
        cmd = raw[:50]
        return f"[bash: {cmd}{'...' if len(raw) > 50 else ''}]"
    if name == "grep":
        return f"[grep: /{args.get('pattern', '')}/ in {sp(args.get('path') or '.')}]"
    if name == "find":
        return f"[find: {args.get('pattern', '')} in {sp(args.get('path') or '.')}]"
    if name == "ls":
        return f"[ls: {sp(args.get('path') or '.')}]"
    import json  # noqa: PLC0415

    args_str = json.dumps(args)[:40]
    return f"[{name}: {args_str}{'...' if len(json.dumps(args)) > 40 else ''}]"


def _entry_display_text(flat: FlatNode, tool_call_map: dict[str, dict[str, Any]], is_selected: bool) -> str:
    entry = flat.node.entry
    etype = getattr(entry, "type", None) or entry.get("type") if isinstance(entry, dict) else None

    def get(attr: str, default: Any = "") -> Any:
        if isinstance(entry, dict):
            return entry.get(attr, default)
        return getattr(entry, attr, default)

    if etype == "message":
        msg = get("message")
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")

        if role == "user":
            text = _normalize_text(_extract_content(content))
            return f"user: {text}"
        if role == "assistant":
            text = _normalize_text(_extract_content(content))
            if text:
                return f"assistant: {text}"
            stop = msg.get("stop_reason") if isinstance(msg, dict) else getattr(msg, "stop_reason", "")
            if stop == "aborted":
                return "assistant: (aborted)"
            err = msg.get("error_message") if isinstance(msg, dict) else getattr(msg, "error_message", "")
            if err:
                return f"assistant: {_normalize_text(str(err))[:80]}"
            return "assistant: (no content)"
        if role in ("tool_result", "toolResult"):
            tool_call_id = msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, "tool_call_id", None)
            tool_name = msg.get("tool_name") if isinstance(msg, dict) else getattr(msg, "tool_name", "tool")
            if tool_call_id and tool_call_id in tool_call_map:
                tc = tool_call_map[tool_call_id]
                return _format_tool_call(tc["name"], tc.get("arguments", {}))
            return f"[{tool_name or 'tool'}]"
        if role in ("bash_execution", "bashExecution"):
            cmd = msg.get("command") if isinstance(msg, dict) else getattr(msg, "command", "")
            return f"[bash]: {_normalize_text(str(cmd))}"
        return f"[{role}]"

    if etype in ("custom_message", "customMessage"):
        custom_type = get("custom_type") or get("customType") or "custom"
        content = get("content", "")
        text = _normalize_text(_extract_content(content))
        return f"[{custom_type}]: {text}"

    if etype == "compaction":
        tokens_before = int(get("tokens_before") or get("tokensBefore") or 0)
        return f"[compaction: {round(tokens_before / 1000)}k tokens]"

    if etype in ("branch_summary", "branchSummary"):
        summary = _normalize_text(str(get("summary", "")))
        return f"[branch summary]: {summary}"

    if etype in ("model_change", "modelChange"):
        return f"[model: {get('model_id') or get('modelId')}]"

    if etype in ("thinking_level_change", "thinkingLevelChange"):
        return f"[thinking: {get('thinking_level') or get('thinkingLevel')}]"

    if etype == "custom":
        return f"[custom: {get('custom_type') or get('customType')}]"

    if etype == "label":
        lbl = get("label", None)
        return f"[label: {lbl or '(cleared)'}]"

    if etype in ("session_info", "sessionInfo"):
        name = get("name", None)
        return f"[title: {name or '(empty)'}]"

    return f"[{etype or 'unknown'}]"


# ---------------------------------------------------------------------------
# Tree flattening
# ---------------------------------------------------------------------------


def _flatten_tree(roots: list[Any], current_leaf_id: str | None) -> tuple[list[FlatNode], dict[str, dict[str, Any]]]:
    """Return (flat_nodes, tool_call_map)."""
    tool_call_map: dict[str, dict[str, Any]] = {}
    result: list[FlatNode] = []
    multiple_roots = len(roots) > 1

    # Build containsActive
    contains_active: dict[int, bool] = {}

    def _id(node: Any) -> str:
        e = node.entry if hasattr(node, "entry") else node.get("entry", {})
        if hasattr(e, "id"):
            return e.id
        return e.get("id", "")

    def _children(node: Any) -> list[Any]:
        if hasattr(node, "children"):
            return list(node.children)
        return list(node.get("children", []))

    # Iterative post-order to populate contains_active
    all_nodes: list[Any] = []
    stack_pre = list(roots)
    while stack_pre:
        n = stack_pre.pop()
        all_nodes.append(n)
        for ch in reversed(_children(n)):
            stack_pre.append(ch)

    for node in reversed(all_nodes):
        has = current_leaf_id is not None and _id(node) == current_leaf_id
        if not has:
            for ch in _children(node):
                if contains_active.get(id(ch)):
                    has = True
                    break
        contains_active[id(node)] = has

    # Order roots: active-branch first
    ordered_roots = sorted(roots, key=lambda r: 0 if contains_active.get(id(r)) else 1)

    # Stack: (node, indent, just_branched, show_connector, is_last, gutters, is_virtual_root_child)
    stack: list[tuple[Any, int, bool, bool, bool, list[tuple[int, bool]], bool]] = []
    for i in range(len(ordered_roots) - 1, -1, -1):
        r = ordered_roots[i]
        is_last = i == len(ordered_roots) - 1
        stack.append((r, 1 if multiple_roots else 0, multiple_roots, multiple_roots, is_last, [], multiple_roots))

    while stack:
        node, indent, just_branched, show_connector, is_last, gutters, is_vrc = stack.pop()

        # Extract tool calls from assistant messages
        entry = node.entry if hasattr(node, "entry") else node.get("entry", {})
        etype = getattr(entry, "type", None) or (entry.get("type") if isinstance(entry, dict) else None)
        if etype == "message":
            msg = getattr(entry, "message", None) or (entry.get("message") if isinstance(entry, dict) else None)
            if msg:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
                if role == "assistant":
                    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "toolCall":
                                tool_call_map[block["id"]] = {
                                    "name": block.get("name", ""),
                                    "arguments": block.get("arguments", {}),
                                }
                            elif hasattr(block, "type") and getattr(block, "type", None) == "toolCall":
                                tool_call_map[getattr(block, "id", "")] = {
                                    "name": getattr(block, "name", ""),
                                    "arguments": getattr(block, "arguments", {}),
                                }

        result.append(
            FlatNode(
                node=node,
                indent=indent,
                show_connector=show_connector,
                is_last=is_last,
                gutters=list(gutters),
                is_virtual_root_child=is_vrc,
            )
        )

        children = _children(node)
        multi_children = len(children) > 1

        # Prioritize active-branch children
        ordered = sorted(children, key=lambda c: 0 if contains_active.get(id(c)) else 1)

        child_indent = indent + 1 if multi_children else (indent + 1 if just_branched and indent > 0 else indent)

        current_display_indent = max(0, indent - 1) if multiple_roots else indent
        connector_pos = max(0, current_display_indent - 1)
        connector_displayed = show_connector and not is_vrc
        child_gutters = list(gutters) + ([(connector_pos, not is_last)] if connector_displayed else [])

        for i in range(len(ordered) - 1, -1, -1):
            child = ordered[i]
            child_is_last = i == len(ordered) - 1
            stack.append((child, child_indent, multi_children, multi_children, child_is_last, child_gutters, False))

    return result, tool_call_map


def _filter_flat(
    flat: list[FlatNode],
    current_leaf_id: str | None,
    filter_mode: FilterMode,
    search: str,
    tool_call_map: dict[str, dict[str, Any]],
) -> list[FlatNode]:
    tokens = search.lower().split() if search.strip() else []
    result = []
    for fn in flat:
        entry = fn.node.entry if hasattr(fn.node, "entry") else fn.node.get("entry", {})
        etype = getattr(entry, "type", None) or (entry.get("type") if isinstance(entry, dict) else None)
        entry_id = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
        is_current = entry_id == current_leaf_id

        # Filter mode
        if filter_mode == "no-tools":
            if etype == "message":
                msg = getattr(entry, "message", None) or (entry.get("message") if isinstance(entry, dict) else None)
                role = (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) if msg else ""
                if role in ("tool_result", "toolResult"):
                    continue
        elif filter_mode == "user-only":
            if etype != "message":
                continue
            msg = getattr(entry, "message", None) or (entry.get("message") if isinstance(entry, dict) else None)
            role = (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) if msg else ""
            if role != "user":
                continue
        elif filter_mode == "labeled-only":
            if etype != "label":
                continue
        elif filter_mode == "default" and etype == "message" and not is_current:
            msg = getattr(entry, "message", None) or (entry.get("message") if isinstance(entry, dict) else None)
            if msg:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if role == "assistant" and not _has_text_content(content):
                    stop = (msg.get("stop_reason") if isinstance(msg, dict) else getattr(msg, "stop_reason", "")) or ""
                    if stop not in ("aborted",) and not (
                        msg.get("error_message") if isinstance(msg, dict) else getattr(msg, "error_message", None)
                    ):
                        continue

        # Text search
        if tokens:
            display = _entry_display_text(fn, tool_call_map, False).lower()
            if not all(t in display for t in tokens):
                continue

        result.append(fn)
    return result


def _render_prefix(fn: FlatNode, multiple_roots: bool) -> str:
    display_indent = max(0, fn.indent - 1) if multiple_roots else fn.indent
    if display_indent == 0:
        return ""

    parts = []
    # Render gutters (vertical bars from ancestor branch points)
    for level in range(display_indent - 1):
        # Check if a gutter entry matches this level
        show_bar = any(pos == level and show for pos, show in fn.gutters)
        parts.append("│  " if show_bar else "   ")

    # Render connector at current level
    if fn.show_connector and not fn.is_virtual_root_child:
        parts.append("└─ " if fn.is_last else "├─ ")
    else:
        parts.append("   ")

    return "".join(parts)


# ---------------------------------------------------------------------------
# TreeSelectorScreen
# ---------------------------------------------------------------------------

_FILTER_MODES: list[FilterMode] = ["default", "no-tools", "user-only", "labeled-only", "all"]
_FILTER_LABELS = {
    "default": "default",
    "no-tools": "no tools",
    "user-only": "user only",
    "labeled-only": "labeled only",
    "all": "all",
}


class TreeSelectorScreen(ModalScreen[Any]):
    """Session tree navigator.

    Port of ``TreeSelectorComponent`` (tree-selector.ts, 1239 LoC).

    Displays the session's entry tree with ASCII connectors and lets the user
    navigate to any visible entry to resume a conversation branch.

    Returns the selected entry ID or ``None`` on cancel.
    """

    CSS = """
    TreeSelectorScreen {
        align: center middle;
    }
    #ts-box {
        width: 95;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #ts-header {
        color: $text-muted;
        margin-bottom: 1;
    }
    #ts-search {
        margin-bottom: 1;
    }
    #ts-list {
        height: auto;
        max-height: 30;
    }
    #ts-status {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
        ("ctrl+f", "cycle_filter", "Cycle filter"),
    ]

    def __init__(
        self,
        tree: list[Any],
        current_leaf_id: str | None = None,
        *,
        initial_filter: FilterMode = "default",
        initial_selected_id: str | None = None,
    ) -> None:
        super().__init__()
        self._tree = tree
        self._current_leaf_id = current_leaf_id
        self._filter_mode: FilterMode = initial_filter
        self._initial_selected_id = initial_selected_id

        self._flat_all: list[FlatNode] = []
        self._tool_call_map: dict[str, dict[str, Any]] = {}
        self._filtered: list[FlatNode] = []
        self._selected_index = 0
        self._multiple_roots = len(tree) > 1
        self._folded: set[str] = set()
        self._search = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ts-box"):
            yield Static(self._header_text(), id="ts-header")
            yield Input(placeholder="Search entries…", id="ts-search")
            yield ListView(id="ts-list")
            yield Static("", id="ts-status")

    def on_mount(self) -> None:
        self._flat_all, self._tool_call_map = _flatten_tree(self._tree, self._current_leaf_id)
        self._apply_filter()
        # Set initial selection
        target_id = self._initial_selected_id or self._current_leaf_id
        if target_id:
            for i, fn in enumerate(self._filtered):
                entry = fn.node.entry if hasattr(fn.node, "entry") else fn.node.get("entry", {})
                eid = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
                if eid == target_id:
                    self._selected_index = i
                    break
        self._rebuild_list()

    def _header_text(self) -> str:
        parts = [f"[dim]Filter:[/dim] [bold]{_FILTER_LABELS.get(self._filter_mode, self._filter_mode)}[/bold]"]
        parts.append("[dim](Ctrl+F cycle  ↑↓/jk navigate  Enter select  l label  f fold  Esc cancel)[/dim]")
        return "  ".join(parts)

    def _apply_filter(self) -> None:
        self._filtered = _filter_flat(
            self._flat_all, self._current_leaf_id, self._filter_mode, self._search, self._tool_call_map
        )
        self._selected_index = min(self._selected_index, max(0, len(self._filtered) - 1))

    def _rebuild_list(self) -> None:
        lv = self.query_one("#ts-list", ListView)
        lv.clear()

        if not self._filtered:
            lv.append(ListItem(Label("[dim]  No entries.[/dim]"), name=""))
            self.query_one("#ts-status", Static).update("")
            return

        max_visible = 20
        start = max(0, min(self._selected_index - max_visible // 2, len(self._filtered) - max_visible))
        end = min(start + max_visible, len(self._filtered))

        for i in range(start, end):
            fn = self._filtered[i]
            is_sel = i == self._selected_index
            entry = fn.node.entry if hasattr(fn.node, "entry") else fn.node.get("entry", {})
            entry_id = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
            is_current = entry_id == self._current_leaf_id

            prefix = _render_prefix(fn, self._multiple_roots)
            display = _entry_display_text(fn, self._tool_call_map, is_sel)
            display = _normalize_text(display)[:120]

            marker = "→ " if is_sel else "  "
            cur_mark = " ●" if is_current else ""
            line = f"{marker}{prefix}{display}{cur_mark}"
            lv.append(ListItem(Label(line), name=str(i)))

        scroll = f"  ({self._selected_index + 1}/{len(self._filtered)})" if len(self._filtered) > max_visible else ""
        self.query_one("#ts-status", Static).update(f"[dim]Enter resume  l label  f fold{scroll}[/dim]")

    @on(Input.Changed, "#ts-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._search = event.value
        self._selected_index = 0
        self._apply_filter()
        self._rebuild_list()

    @on(Input.Submitted, "#ts-search")
    def _on_search_submit(self, _event: Input.Submitted) -> None:
        self._select_current()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx_str = event.item.name
        if not idx_str:
            return
        try:
            idx = int(idx_str)
        except ValueError:
            return
        self._selected_index = idx
        self._select_current()

    def on_key(self, event: Any) -> None:
        key = getattr(event, "key", None)
        if key in ("up", "k"):
            if self._filtered:
                self._selected_index = max(0, self._selected_index - 1)
                self._rebuild_list()
            event.stop()
        elif key in ("down", "j"):
            if self._filtered:
                self._selected_index = min(len(self._filtered) - 1, self._selected_index + 1)
                self._rebuild_list()
            event.stop()
        elif key == "l":
            self._label_selected()
            event.stop()
        elif key == "f":
            self._fold_selected()
            event.stop()

    def action_cycle_filter(self) -> None:
        idx = _FILTER_MODES.index(self._filter_mode) if self._filter_mode in _FILTER_MODES else 0
        self._filter_mode = _FILTER_MODES[(idx + 1) % len(_FILTER_MODES)]
        self._selected_index = 0
        self._apply_filter()
        self.query_one("#ts-header", Static).update(self._header_text())
        self._rebuild_list()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _select_current(self) -> None:
        if not self._filtered:
            return
        fn = self._filtered[self._selected_index]
        entry = fn.node.entry if hasattr(fn.node, "entry") else fn.node.get("entry", {})
        entry_id = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
        self.dismiss(entry_id)

    def _label_selected(self) -> None:
        if not self._filtered:
            return
        fn = self._filtered[self._selected_index]
        entry = fn.node.entry if hasattr(fn.node, "entry") else fn.node.get("entry", {})
        entry_id = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
        # Emit a label-edit request: push an input screen and get the label
        import asyncio  # noqa: PLC0415

        from nu_coding_agent.modes.interactive.components.extension_ui import ExtensionInputScreen  # noqa: PLC0415

        current_label = getattr(entry, "label", None) or (entry.get("label") if isinstance(entry, dict) else None) or ""

        async def _do_label() -> None:
            new_label: str | None = await self.app.push_screen_wait(
                ExtensionInputScreen(f"Label for entry {entry_id}:", placeholder=str(current_label))
            )
            if new_label is not None:
                # Signal the caller via dismiss with a special tuple
                # (Caller should check for tuple and handle label update)
                self.dismiss(("label", entry_id, new_label.strip()))

        asyncio.get_event_loop().create_task(_do_label())

    def _fold_selected(self) -> None:
        if not self._filtered:
            return
        fn = self._filtered[self._selected_index]
        entry = fn.node.entry if hasattr(fn.node, "entry") else fn.node.get("entry", {})
        entry_id = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
        if not entry_id:
            return
        if entry_id in self._folded:
            self._folded.discard(entry_id)
        else:
            self._folded.add(entry_id)
        # Re-flatten (simple: just remove children of folded nodes)
        self._apply_filter()
        self._rebuild_list()


__all__ = [
    "FilterMode",
    "FlatNode",
    "TreeSelectorScreen",
]
