"""Full-featured model selector — port of ``model-selector.ts``.

Replaces the basic :class:`~nu_coding_agent.modes.interactive.selectors.ModelPickerScreen`
with a fuzzy-search version that supports scope toggling (all / scoped).

Usage::

    screen = ModelSelectorScreen(session, scoped_models=[...])
    model = await app.push_screen_wait(screen)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual import on, work
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from nu_coding_agent.core.agent_session import AgentSession


def _fuzzy_match(query: str, text: str) -> bool:
    """Simple fuzzy containment check (case-insensitive)."""
    q = query.lower()
    t = text.lower()
    # All characters of query must appear in order in text
    idx = 0
    for ch in q:
        pos = t.find(ch, idx)
        if pos == -1:
            return False
        idx = pos + 1
    return True


class ModelSelectorScreen(ModalScreen["Any | None"]):
    """Fuzzy-search model picker.

    Port of ``ModelSelectorComponent`` (model-selector.ts).  Supports:

    - Real-time fuzzy filtering as you type.
    - Tab to toggle between *all* and *scoped* model lists.
    - Arrow keys to navigate; Enter to select; Escape to cancel.
    """

    CSS = """
    ModelSelectorScreen {
        align: center middle;
    }
    #ms-box {
        width: 70;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #ms-scope {
        color: $text-muted;
    }
    #ms-search {
        margin: 1 0;
    }
    #ms-list-area {
        height: auto;
        max-height: 20;
    }
    #ms-status {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [  # noqa: RUF012
        ("escape", "cancel", "Cancel"),
        ("tab", "toggle_scope", "Toggle scope"),
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
    ]

    def __init__(
        self,
        session: AgentSession,
        *,
        scoped_models: list[Any] | None = None,
        initial_query: str = "",
    ) -> None:
        super().__init__()
        self._session = session
        self._scoped_models: list[Any] = scoped_models or []
        self._initial_query = initial_query

        self._all_models: list[Any] = []
        self._scoped_items: list[Any] = []
        self._filtered: list[Any] = []
        self._selected_index = 0
        self._scope: str = "scoped" if self._scoped_models else "all"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ms-box"):
            if self._scoped_models:
                yield Static(self._scope_text(), id="ms-scope")
            else:
                yield Static("Only models with configured API keys are shown.", id="ms-scope")
            yield Input(placeholder="Search models…", value=self._initial_query, id="ms-search")
            yield ListView(id="ms-list-area")
            yield Static("", id="ms-status")

    def on_mount(self) -> None:
        self.query_one("#ms-search", Input).focus()
        self._load_models()

    @work(thread=False)
    async def _load_models(self) -> None:
        registry = self._session.model_registry
        registry.refresh()
        try:
            models = registry.get_available_models()
        except Exception as exc:
            self.query_one("#ms-status", Static).update(f"[red]Error loading models: {exc}[/]")
            return

        current = self._session.model
        self._all_models = sorted(
            models,
            key=lambda m: (0 if current and m.id == current.id else 1, m.provider, m.id),
        )

        # Build scoped items (look up in registry for freshness)
        scoped_set = {
            (m.provider if hasattr(m, "provider") else "", m.id if hasattr(m, "id") else str(m))
            for m in self._scoped_models
        }
        self._scoped_items = [m for m in self._all_models if (m.provider, m.id) in scoped_set]

        self._apply_filter(self.query_one("#ms-search", Input).value)

    def _scope_text(self) -> str:
        all_part = "[bold]all[/bold]" if self._scope == "all" else "[dim]all[/dim]"
        scoped_part = "[bold]scoped[/bold]" if self._scope == "scoped" else "[dim]scoped[/dim]"
        return f"[dim]Scope:[/dim] {all_part} [dim]|[/dim] {scoped_part}  [dim](Tab to toggle)[/dim]"

    def _active_models(self) -> list[Any]:
        return self._scoped_items if self._scope == "scoped" else self._all_models

    def _apply_filter(self, query: str) -> None:
        models = self._active_models()
        if query.strip():
            q = query.strip()
            models = [m for m in models if _fuzzy_match(q, f"{m.id} {m.provider}")]
        self._filtered = models
        self._selected_index = min(self._selected_index, max(0, len(self._filtered) - 1))
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#ms-list-area", ListView)
        current = self._session.model
        current_id = current.id if current else ""

        max_visible = 10
        start = max(0, min(self._selected_index - max_visible // 2, len(self._filtered) - max_visible))
        end = min(start + max_visible, len(self._filtered))

        items = []
        for i in range(start, end):
            m = self._filtered[i]
            is_sel = i == self._selected_index
            is_cur = m.id == current_id
            marker = "→ " if is_sel else "  "
            check = " ✓" if is_cur else ""
            label = f"{marker}{m.id} [{m.provider}]{check}"
            items.append(ListItem(Label(label), name=str(i)))

        # Swap children
        lv.clear()
        for item in items:
            lv.append(item)

        # Status line
        status = self.query_one("#ms-status", Static)
        if not self._filtered:
            status.update("[dim]No matching models[/dim]")
        else:
            m = self._filtered[self._selected_index]
            name = getattr(m, "name", m.id)
            scroll = (
                f"  ({self._selected_index + 1}/{len(self._filtered)})" if len(self._filtered) > max_visible else ""
            )
            status.update(f"[dim]{name}{scroll}[/dim]")

    @on(Input.Changed, "#ms-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._selected_index = 0
        self._apply_filter(event.value)

    @on(Input.Submitted, "#ms-search")
    def _on_search_submit(self, _event: Input.Submitted) -> None:
        self._select_current()

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

    def action_toggle_scope(self) -> None:
        if not self._scoped_models:
            return
        self._scope = "scoped" if self._scope == "all" else "all"
        self._selected_index = 0
        scope_widget = self.query_one("#ms-scope", Static)
        scope_widget.update(self._scope_text())
        self._apply_filter(self.query_one("#ms-search", Input).value)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx_str = event.item.name
        if idx_str is None:
            return
        try:
            idx = int(idx_str)
        except ValueError:
            return
        if 0 <= idx < len(self._filtered):
            m = self._filtered[idx]
            self._save_and_dismiss(m)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _select_current(self) -> None:
        if not self._filtered:
            return
        m = self._filtered[self._selected_index]
        self._save_and_dismiss(m)

    def _save_and_dismiss(self, model: Any) -> None:
        # Persist as default in settings if available
        sm = getattr(self._session, "settings_manager", None)
        if sm is not None and hasattr(sm, "set_default_model_and_provider"):
            sm.set_default_model_and_provider(model.provider, model.id)
        self.dismiss(model)


__all__ = ["ModelSelectorScreen"]
