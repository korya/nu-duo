"""Interactive mode footer — model info, token stats, git branch, session status.

Port of ``packages/coding-agent/src/modes/interactive/footer.ts``
(220 LoC). The upstream footer renders via the nu_tui component tree;
this Python version is a Textual ``Static`` widget that updates its
content from the AgentSession state.
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Any

from rich.text import Text as RichText
from textual.widgets import Static

if TYPE_CHECKING:
    from nu_coding_agent.core.agent_session import AgentSession

# Thinking level display abbreviations
_THINKING_ABBREV: dict[str, str] = {
    "off": "",
    "minimal": "t:min",
    "low": "t:low",
    "medium": "t:med",
    "high": "t:hi",
    "xhigh": "t:xhi",
}


def _format_tokens(n: int) -> str:
    """Format token count as K/M shorthand."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_cost(cost: float) -> str:
    """Format cost in dollars."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _get_git_branch(cwd: str) -> str | None:
    """Return current git branch name or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch != "HEAD" else "detached"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


class InteractiveFooter(Static):
    """Footer bar showing model, tokens, git branch, and session info.

    Call :meth:`refresh_content` after any state change (model swap,
    turn end, session switch) to update the display.
    """

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
        self._cached_branch: str | None = None
        self._branch_checked = False
        self.refresh_content()

    def refresh_content(self) -> None:
        """Update the footer content from the current session state."""
        t = RichText()
        sep = RichText(" │ ", style="dim")

        # --- Model ---
        model = self._session.model
        if model is not None:
            t.append(f"{model.provider}/", style="dim")
            t.append(model.id, style="bold dim")
        else:
            t.append("no model", style="dim italic")

        # Thinking level
        thinking = getattr(self._session, "_thinking_level", "off")
        abbrev = _THINKING_ABBREV.get(thinking, "")
        if abbrev:
            t.append(f" ({abbrev})", style="dim")

        # --- Token stats ---
        stats = self._collect_usage_stats()
        if stats["total_tokens"] > 0:
            t.append_text(sep)
            t.append(f"in:{_format_tokens(stats['input'])}", style="dim")
            t.append(f" out:{_format_tokens(stats['output'])}", style="dim")
            if stats["cache_read"] > 0:
                t.append(f" cache:{_format_tokens(stats['cache_read'])}", style="dim")
            t.append(f" ({_format_cost(stats['cost'])})", style="dim")

            # Context usage percentage
            if stats["context_window"] > 0:
                pct = min(100, int(stats["total_tokens"] / stats["context_window"] * 100))
                style = "dim"
                if pct > 80:
                    style = "bold yellow"
                elif pct > 60:
                    style = "yellow"
                t.append(f" ctx:{pct}%", style=style)

        # --- Git branch ---
        if not self._branch_checked:
            self._cached_branch = _get_git_branch(self._session.cwd)
            self._branch_checked = True
        if self._cached_branch:
            t.append_text(sep)
            t.append(self._cached_branch, style="dim italic")

        # --- Session ---
        sf = self._session.session_manager.get_session_file()
        if sf:
            t.append_text(sep)
            t.append(os.path.basename(sf), style="dim")

        # --- CWD ---
        cwd = self._session.cwd
        home = os.path.expanduser("~")
        display_cwd = "~" + cwd[len(home) :] if cwd.startswith(home) else cwd
        t.append_text(sep)
        t.append(display_cwd, style="dim")

        self.update(t)

    def invalidate_git_branch(self) -> None:
        """Force re-detection of git branch on next refresh."""
        self._branch_checked = False

    def _collect_usage_stats(self) -> dict[str, Any]:
        """Aggregate token usage from all session entries."""
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        total_cost = 0.0
        context_window = 0

        entries = self._session.session_manager.get_entries()
        for entry in entries:
            entry_type = entry.get("type")
            if entry_type != "message":
                continue

            msg = entry.get("message")
            if msg is None:
                continue

            usage = msg.get("usage") if isinstance(msg, dict) else getattr(msg, "usage", None)
            if usage is None:
                continue

            if isinstance(usage, dict):
                total_input += usage.get("input", 0)
                total_output += usage.get("output", 0)
                total_cache_read += usage.get("cache_read", usage.get("cacheRead", 0))
                total_cache_write += usage.get("cache_write", usage.get("cacheWrite", 0))
                cost = usage.get("cost", {})
                if isinstance(cost, dict):
                    total_cost += cost.get("total", 0.0)
                elif isinstance(cost, (int, float)):
                    total_cost += cost
            else:
                total_input += getattr(usage, "input", 0)
                total_output += getattr(usage, "output", 0)
                total_cache_read += getattr(usage, "cache_read", 0)
                total_cache_write += getattr(usage, "cache_write", 0)
                cost_obj = getattr(usage, "cost", None)
                if cost_obj is not None:
                    total_cost += (
                        getattr(cost_obj, "total", 0.0) if not isinstance(cost_obj, (int, float)) else cost_obj
                    )

        # Try to get context window from current model
        model = self._session.model
        if model is not None:
            context_window = getattr(model, "context_window", 0) or getattr(model, "contextWindow", 0) or 0

        total_tokens = total_input + total_output + total_cache_read + total_cache_write

        return {
            "input": total_input,
            "output": total_output,
            "cache_read": total_cache_read,
            "cache_write": total_cache_write,
            "total_tokens": total_tokens,
            "cost": total_cost,
            "context_window": context_window,
        }


__all__ = ["InteractiveFooter"]
