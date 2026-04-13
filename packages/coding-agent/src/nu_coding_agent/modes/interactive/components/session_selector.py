"""Session selector — port of ``session-selector.ts`` + ``session-selector-search.ts``.

A full-featured session browser with:

- Current-cwd vs all-sessions scope toggle (Tab).
- Sort modes: threaded (default), recent, relevance.
- Fuzzy / phrase / regex search.
- Named-only filter.
- Session rename (r key).
- Session delete (d key, with confirmation).
- Progress indicator during async loading.

Usage::

    screen = SessionSelectorScreen(cwd, session_dir=session_dir)
    path = await app.push_screen_wait(screen)
"""

from __future__ import annotations

import asyncio
import datetime
import os
import re
from typing import TYPE_CHECKING, Any

from textual import on, work
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, LoadingIndicator, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


# ---------------------------------------------------------------------------
# Search helpers — port of ``session-selector-search.ts``
# ---------------------------------------------------------------------------

SortMode = str  # "threaded" | "recent" | "relevance"
NameFilter = str  # "all" | "named"


def _normalize(text: str) -> str:
    return text.lower().replace(" ", " ").strip()


def _session_text(session: Any) -> str:
    parts = [
        str(getattr(session, "id", "")),
        str(getattr(session, "name", "") or ""),
        str(getattr(session, "all_messages_text", "") or ""),
        str(getattr(session, "cwd", "") or ""),
    ]
    return " ".join(parts)


def has_session_name(session: Any) -> bool:
    name = getattr(session, "name", None)
    return bool(name and str(name).strip())


def _fuzzy_match(query: str, text: str) -> tuple[bool, float]:
    """Returns (matches, score) — lower score is better."""
    q = query.lower()
    t = text.lower()
    idx = 0
    score = 0.0
    for ch in q:
        pos = t.find(ch, idx)
        if pos == -1:
            return False, 0.0
        score += pos * 0.1
        idx = pos + 1
    return True, score


def _parse_query(query: str) -> dict[str, Any]:
    """Returns a dict with keys: mode, tokens, regex, error."""
    trimmed = query.strip()
    if not trimmed:
        return {"mode": "tokens", "tokens": [], "regex": None}

    if trimmed.startswith("re:"):
        pattern = trimmed[3:].strip()
        if not pattern:
            return {"mode": "regex", "tokens": [], "regex": None, "error": "Empty regex"}
        try:
            return {"mode": "regex", "tokens": [], "regex": re.compile(pattern, re.IGNORECASE)}
        except re.error as exc:
            return {"mode": "regex", "tokens": [], "regex": None, "error": str(exc)}

    # Token mode with quote support
    tokens: list[dict[str, str]] = []
    buf = ""
    in_quote = False
    had_unclosed = False

    for ch in trimmed:
        if ch == '"':
            if in_quote:
                v = buf.strip()
                if v:
                    tokens.append({"kind": "phrase", "value": v})
                buf = ""
                in_quote = False
            else:
                v = buf.strip()
                if v:
                    tokens.append({"kind": "fuzzy", "value": v})
                buf = ""
                in_quote = True
        elif not in_quote and ch == " ":
            v = buf.strip()
            if v:
                tokens.append({"kind": "fuzzy", "value": v})
            buf = ""
        else:
            buf += ch

    if in_quote:
        had_unclosed = True

    if had_unclosed:
        tokens = [{"kind": "fuzzy", "value": t} for t in trimmed.split() if t]
    else:
        v = buf.strip()
        if v:
            tokens.append({"kind": "fuzzy", "value": v})

    return {"mode": "tokens", "tokens": tokens, "regex": None}


def _match_session(session: Any, parsed: dict[str, Any]) -> tuple[bool, float]:
    text = _session_text(session)
    if parsed.get("error"):
        return False, 0.0

    if parsed["mode"] == "regex":
        rx = parsed.get("regex")
        if not rx:
            return False, 0.0
        m = rx.search(text)
        if not m:
            return False, 0.0
        return True, m.start() * 0.1

    tokens = parsed.get("tokens", [])
    if not tokens:
        return True, 0.0

    total_score = 0.0
    nt = _normalize(text)
    for token in tokens:
        if token["kind"] == "phrase":
            phrase = _normalize(token["value"])
            idx = nt.find(phrase)
            if idx < 0:
                return False, 0.0
            total_score += idx * 0.1
        else:
            matches, score = _fuzzy_match(token["value"], text)
            if not matches:
                return False, 0.0
            total_score += score

    return True, total_score


def filter_and_sort_sessions(
    sessions: list[Any],
    query: str,
    sort_mode: SortMode,
    name_filter: NameFilter = "all",
) -> list[Any]:
    if name_filter == "named":
        sessions = [s for s in sessions if has_session_name(s)]
    trimmed = query.strip()
    if not trimmed:
        return sessions

    parsed = _parse_query(query)
    if parsed.get("error"):
        return []

    if sort_mode == "recent":
        return [s for s in sessions if _match_session(s, parsed)[0]]

    scored = []
    for s in sessions:
        matches, score = _match_session(s, parsed)
        if matches:
            scored.append((s, score))
    scored.sort(key=lambda x: (x[1], -_modified_ts(x[0])))
    return [s for s, _ in scored]


def _modified_ts(session: Any) -> float:
    modified = getattr(session, "modified", None)
    if isinstance(modified, datetime.datetime):
        return modified.timestamp()
    return 0.0


def _format_age(session: Any) -> str:
    modified = getattr(session, "modified", None)
    if not modified:
        return ""
    if not isinstance(modified, datetime.datetime):
        try:
            modified = datetime.datetime.fromtimestamp(float(modified))
        except Exception:
            return ""
    diff = datetime.datetime.now() - modified
    mins = int(diff.total_seconds() / 60)
    if mins < 1:
        return "now"
    if mins < 60:
        return f"{mins}m"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 7:
        return f"{days}d"
    if days < 30:
        return f"{days // 7}w"
    if days < 365:
        return f"{days // 30}mo"
    return f"{days // 365}y"


def _shorten_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


# ---------------------------------------------------------------------------
# SessionSelectorScreen
# ---------------------------------------------------------------------------


class SessionSelectorScreen(ModalScreen[str | None]):
    """Full session browser modal.

    Port of ``SessionSelectorComponent`` (session-selector.ts).
    Returns a session file path or ``None`` on cancel.
    """

    CSS = """
    SessionSelectorScreen {
        align: center middle;
    }
    #ss-box {
        width: 90;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #ss-header {
        color: $text-muted;
        margin-bottom: 1;
    }
    #ss-search {
        margin-bottom: 1;
    }
    #ss-spinner {
        margin: 1 0;
    }
    #ss-status {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
        ("tab", "toggle_scope", "Toggle scope"),
        ("ctrl+s", "toggle_sort", "Toggle sort"),
        ("ctrl+n", "toggle_name_filter", "Named only"),
    ]

    def __init__(self, cwd: str, session_dir: str | None = None) -> None:
        super().__init__()
        self._cwd = cwd
        self._session_dir = session_dir
        self._all_sessions: list[Any] = []
        self._filtered: list[Any] = []
        self._selected_index = 0
        self._scope: str = "current"  # "current" | "all"
        self._sort_mode: SortMode = "threaded"
        self._name_filter: NameFilter = "all"
        self._loaded = False
        self._confirming_delete: str | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ss-box"):
            yield Static(self._header_text(), id="ss-header")
            yield Input(placeholder="Search sessions… (re:<regex> or fuzzy)", id="ss-search")
            yield LoadingIndicator(id="ss-spinner")
            yield ListView(id="ss-list")
            yield Static("", id="ss-status")

    def on_mount(self) -> None:
        self.query_one("#ss-search", Input).focus()
        self._load_sessions()

    @work(thread=False)
    async def _load_sessions(self) -> None:
        from nu_coding_agent.core.session_manager import SessionManager  # noqa: PLC0415

        sessions = await SessionManager.list(self._cwd, self._session_dir)
        self._all_sessions = sessions
        self._loaded = True
        # Hide spinner
        spinner = self.query_one("#ss-spinner", LoadingIndicator)
        spinner.display = False
        self._apply_filter(self.query_one("#ss-search", Input).value)

    def _header_text(self) -> str:
        scope_all = "[bold]all[/bold]" if self._scope == "all" else "[dim]all[/dim]"
        scope_cur = "[bold]current dir[/bold]" if self._scope == "current" else "[dim]current dir[/dim]"
        sort_label = {"threaded": "threaded", "recent": "recent", "relevance": "relevance"}.get(
            self._sort_mode, self._sort_mode
        )
        name_part = " [dim]|[/dim] [bold]named only[/bold]" if self._name_filter == "named" else ""
        return (
            f"[dim]Scope:[/dim] {scope_cur} [dim]|[/dim] {scope_all}  "
            f"[dim]Sort:[/dim] [bold]{sort_label}[/bold]{name_part}  "
            "[dim](Tab scope, Ctrl+S sort, Ctrl+N named)[/dim]"
        )

    def _sessions_for_scope(self) -> list[Any]:
        if self._scope == "current":
            return [s for s in self._all_sessions if str(getattr(s, "cwd", "")) == self._cwd]
        return self._all_sessions

    def _apply_filter(self, query: str) -> None:
        sessions = self._sessions_for_scope()
        self._filtered = filter_and_sort_sessions(sessions, query, self._sort_mode, self._name_filter)
        self._selected_index = min(self._selected_index, max(0, len(self._filtered) - 1))
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#ss-list", ListView)
        lv.clear()

        if not self._loaded:
            return

        if not self._filtered:
            lv.append(ListItem(Label("[dim]  No sessions found.[/dim]"), name=""))
            self.query_one("#ss-status", Static).update("")
            return

        max_visible = 12
        start = max(0, min(self._selected_index - max_visible // 2, len(self._filtered) - max_visible))
        end = min(start + max_visible, len(self._filtered))

        for i in range(start, end):
            s = self._filtered[i]
            is_sel = i == self._selected_index
            path = str(getattr(s, "path", s))
            name = getattr(s, "name", None)
            age = _format_age(s)
            short_path = _shorten_path(path)
            marker = "→ " if is_sel else "  "
            if name:
                label_text = f"{marker}[bold]{name}[/bold]  [dim]{short_path}  {age}[/dim]"
            else:
                label_text = f"{marker}{short_path}  [dim]{age}[/dim]"
            lv.append(ListItem(Label(label_text), name=str(i)))

        scroll = f"  ({self._selected_index + 1}/{len(self._filtered)})" if len(self._filtered) > max_visible else ""
        self.query_one("#ss-status", Static).update(f"[dim]Enter resume  d delete  r rename{scroll}[/dim]")

    @on(Input.Changed, "#ss-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._selected_index = 0
        self._apply_filter(event.value)

    @on(Input.Submitted, "#ss-search")
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
                self._selected_index = (self._selected_index - 1) % len(self._filtered)
                self._rebuild_list()
            event.stop()
        elif key in ("down", "j"):
            if self._filtered:
                self._selected_index = (self._selected_index + 1) % len(self._filtered)
                self._rebuild_list()
            event.stop()
        elif key == "d":
            self._delete_selected()
            event.stop()
        elif key == "r":
            self._rename_selected()
            event.stop()

    def action_toggle_scope(self) -> None:
        self._scope = "all" if self._scope == "current" else "current"
        self._selected_index = 0
        self.query_one("#ss-header", Static).update(self._header_text())
        self._apply_filter(self.query_one("#ss-search", Input).value)

    def action_toggle_sort(self) -> None:
        modes: list[SortMode] = ["threaded", "recent", "relevance"]
        idx = modes.index(self._sort_mode) if self._sort_mode in modes else 0
        self._sort_mode = modes[(idx + 1) % len(modes)]
        self.query_one("#ss-header", Static).update(self._header_text())
        self._apply_filter(self.query_one("#ss-search", Input).value)

    def action_toggle_name_filter(self) -> None:
        self._name_filter = "named" if self._name_filter == "all" else "all"
        self.query_one("#ss-header", Static).update(self._header_text())
        self._apply_filter(self.query_one("#ss-search", Input).value)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _select_current(self) -> None:
        if not self._filtered:
            return
        s = self._filtered[self._selected_index]
        path = str(getattr(s, "path", s))
        self.dismiss(path)

    def _delete_selected(self) -> None:
        if not self._filtered:
            return
        s = self._filtered[self._selected_index]
        path = str(getattr(s, "path", s))
        if self._confirming_delete == path:
            import contextlib  # noqa: PLC0415

            with contextlib.suppress(OSError):
                os.unlink(path)
            self._confirming_delete = None
            self._all_sessions = [x for x in self._all_sessions if str(getattr(x, "path", x)) != path]
            self._selected_index = max(0, self._selected_index - 1)
            self._apply_filter(self.query_one("#ss-search", Input).value)
        else:
            self._confirming_delete = path
            self.query_one("#ss-status", Static).update(
                f"[red]Press d again to confirm delete: {os.path.basename(path)}[/red]"
            )

    def _rename_selected(self) -> None:
        if not self._filtered:
            return
        s = self._filtered[self._selected_index]
        path = str(getattr(s, "path", s))
        current_name = str(getattr(s, "name", "") or "")
        # Push a simple input screen to get the new name
        from nu_coding_agent.modes.interactive.components.extension_ui import ExtensionInputScreen  # noqa: PLC0415

        async def _do_rename() -> None:
            new_name: str | None = await self.app.push_screen_wait(
                ExtensionInputScreen("Rename session:", placeholder=current_name)
            )
            if new_name is not None and new_name.strip():
                # Update session name via session manager append
                import contextlib  # noqa: PLC0415

                with contextlib.suppress(Exception):
                    from nu_coding_agent.core.session_manager import SessionManager  # noqa: PLC0415

                    sm = SessionManager(
                        cwd=self._cwd,
                        session_dir=self._session_dir or "",
                        session_file=path,
                        persist=True,
                    )
                    sm.append_session_info(new_name.strip())
                # Refresh list
                await self._load_sessions_and_refresh()  # type: ignore[attr-defined]

        asyncio.get_event_loop().create_task(_do_rename())

    @work(thread=False)
    async def _load_sessions_and_refresh(self) -> None:
        from nu_coding_agent.core.session_manager import SessionManager  # noqa: PLC0415

        sessions = await SessionManager.list(self._cwd, self._session_dir)
        self._all_sessions = sessions
        self._apply_filter(self.query_one("#ss-search", Input).value)


__all__ = [
    "SessionSelectorScreen",
    "filter_and_sort_sessions",
    "has_session_name",
]
