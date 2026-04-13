"""Message renderer widgets for the interactive REPL.

Each widget renders a specific type of conversation entry (user
message, assistant reply, tool call, bash execution, compaction
summary) as a Textual ``Static`` backed by Rich renderables.

Port of:
- ``user-message.ts`` (33 LoC)
- ``assistant-message.ts`` (130 LoC)
- ``tool-execution.ts`` (328 LoC)
- ``bash-execution.ts`` (218 LoC)
- ``compaction-summary-message.ts`` (59 LoC)
- ``branch-summary-message.ts`` (58 LoC)
"""

from __future__ import annotations

import re
from typing import Any

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text as RichText
from textual.widgets import Static

from nu_coding_agent.modes.interactive.components.tool_renderers import (
    render_tool_call,
    render_tool_result,
)

# Background colours matching upstream theme values:
#   toolPendingBg  #282832  → Textual: $primary-background-darken-1 (approximate)
#   toolSuccessBg  #283228  → Textual: custom
#   toolErrorBg    #3c2828  → Textual: custom
_CSS_PENDING = "on #282832"
_CSS_SUCCESS = "on #283228"
_CSS_ERROR = "on #3c2828"

# Preview line limit for both bash and tool output (matches upstream PREVIEW_LINES)
PREVIEW_LINES = 20

# Strip ANSI codes pattern
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mKHFABCDJG]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class UserMessageWidget(Static):
    """Renders a user prompt with a ``>`` prefix."""

    DEFAULT_CSS = """
    UserMessageWidget {
        padding: 0 1;
        margin: 0 0 1 0;
        background: $primary-background-darken-2;
    }
    """

    def __init__(self, text: str) -> None:
        content = RichText(f"> {text}", style="bold")
        super().__init__(content)


class AssistantMessageWidget(Static):
    """Renders an assistant reply as Rich Markdown.

    Supports streaming: call :meth:`append_delta` during streaming,
    then :meth:`finalize` when the turn completes.
    """

    DEFAULT_CSS = """
    AssistantMessageWidget {
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._chunks: list[str] = []
        self._finalized = False

    def append_delta(self, delta: str) -> None:
        """Append a streaming text delta."""
        self._chunks.append(delta)
        self.update(RichText("".join(self._chunks)))

    def get_text(self) -> str:
        return "".join(self._chunks)

    def finalize(self) -> None:
        """Re-render the accumulated text as Markdown."""
        if self._finalized:
            return
        self._finalized = True
        text = self.get_text()
        if text.strip():
            self.update(RichMarkdown(text, code_theme="monokai"))


class ToolExecutionWidget(Static):
    """Renders a tool call with name, args, and (optionally) result.

    Created at ``tool_execution_start``; updated on ``tool_execution_end``
    via :meth:`set_result`. Expand/collapse toggled via :meth:`toggle_expand`.

    Background colours follow the upstream theme:
    * Pending  → dark blue-grey  (#282832)
    * Success  → dark green-grey (#283228)
    * Error    → dark red-grey   (#3c2828)
    """

    DEFAULT_CSS = """
    ToolExecutionWidget {
        padding: 0 1;
        margin: 0 0 0 0;
    }
    """

    def __init__(self, tool_name: str, args: dict[str, Any]) -> None:
        super().__init__("")
        self._tool_name = tool_name
        self._args = args
        self._is_error = False
        self._result_content: list[Any] = []
        self._expanded = False
        self._finalized = False
        self._update_display()

    # ------------------------------------------------------------------

    def set_result(
        self,
        content: list[Any],
        *,
        is_error: bool = False,
    ) -> None:
        """Called on ``tool_execution_end``."""
        self._result_content = content
        self._is_error = is_error
        self._finalized = True
        self._update_display()

    def toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self._update_display()

    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        t = RichText()

        # Status background hint
        if not self._finalized:
            bg = _CSS_PENDING
        elif self._is_error:
            bg = _CSS_ERROR
        else:
            bg = _CSS_SUCCESS

        # Call line
        call_text = render_tool_call(self._tool_name, self._args)
        if call_text is not None:
            call_text.stylize(bg)
            t.append_text(call_text)
        else:
            t.append(self._tool_name, style=f"bold {bg}")

        # Result
        if self._finalized:
            result_text = render_tool_result(
                self._tool_name,
                self._args,
                self._result_content,
                is_error=self._is_error,
                expanded=self._expanded,
            )
            if result_text is not None:
                t.append("\n")
                result_text.stylize(bg)
                t.append_text(result_text)

        self.update(t)


class BashExecutionWidget(Static):
    """Renders a bash command execution with streaming output.

    Port of ``BashExecutionComponent`` (bash-execution.ts, 218 LoC).

    Lifecycle:
    * Created at ``tool_execution_start`` with the command string.
    * :meth:`set_output` is called on each ``tool_execution_update``
      with the **cumulative** output (not a delta).
    * :meth:`set_complete` is called on ``tool_execution_end``.
    * Click the widget to toggle expand/collapse.
    """

    DEFAULT_CSS = """
    BashExecutionWidget {
        padding: 0 1;
        margin: 0 0 0 0;
    }
    """

    def __init__(self, command: str) -> None:
        super().__init__("")
        self._command = command
        self._output = ""
        self._exit_code: int | None = None
        self._status: str = "running"  # running | complete | error | cancelled
        self._expanded = False
        self._update_display()

    # ------------------------------------------------------------------

    def set_output(self, cumulative_output: str) -> None:
        """Replace the current output with a new cumulative snapshot."""
        self._output = _strip_ansi(cumulative_output).replace("\r\n", "\n").replace("\r", "\n")
        self._update_display()

    def set_complete(
        self,
        exit_code: int | None,
        *,
        cancelled: bool = False,
    ) -> None:
        """Called on ``tool_execution_end``."""
        self._exit_code = exit_code
        if cancelled:
            self._status = "cancelled"
        elif exit_code is not None and exit_code != 0:
            self._status = "error"
        else:
            self._status = "complete"
        self._update_display()

    def toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self._update_display()

    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        is_running = self._status == "running"
        is_error = self._status in ("error", "cancelled")
        bg = _CSS_PENDING if is_running else (_CSS_ERROR if is_error else _CSS_SUCCESS)

        t = RichText()

        # Header line: "$ <command>"
        t.append("$ ", style=f"bold green {bg}")
        t.append(self._command, style=f"green {bg}")

        # Output block
        if self._output:
            lines = self._output.splitlines()
            display_lines = lines if self._expanded else lines[-PREVIEW_LINES:]
            hidden = len(lines) - len(display_lines)

            t.append("\n")
            t.append("\n".join(display_lines), style=f"dim {bg}")

            if hidden > 0 and not self._expanded:
                t.append(f"\n… {hidden} more lines (click to expand)", style=f"dim italic {bg}")

        # Status footer (when not running)
        if not is_running:
            if self._status == "cancelled":
                t.append("\n(cancelled)", style=f"yellow {bg}")
            elif self._status == "error":
                t.append(f"\n(exit {self._exit_code})", style=f"red {bg}")

        self.update(t)

    def on_click(self) -> None:
        self.toggle_expand()


class CompactionSummaryWidget(Static):
    """Renders a compaction summary indicator with optional expand."""

    DEFAULT_CSS = """
    CompactionSummaryWidget {
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
    }
    """

    def __init__(self, summary: str, tokens_before: int = 0) -> None:
        self._summary = summary
        self._tokens_before = tokens_before
        self._expanded = False
        content = self._build()
        super().__init__(content)

    def _build(self) -> RichText:
        t = RichText()
        t.append(f"[compacted] {self._tokens_before} tokens → summarized", style="dim italic")
        if self._summary:
            if self._expanded:
                t.append(f"\n{self._summary}", style="dim")
            else:
                first_line = self._summary.split("\n", 1)[0][:120]
                t.append(f"\n  {first_line}", style="dim")
        return t

    def on_click(self) -> None:
        self._expanded = not self._expanded
        self.update(self._build())


class BranchSummaryWidget(Static):
    """Renders a branch summary indicator."""

    DEFAULT_CSS = """
    BranchSummaryWidget {
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
    }
    """

    def __init__(self, summary: str, from_id: str = "") -> None:
        self._summary = summary
        self._from_id = from_id
        self._expanded = False
        super().__init__(self._build())

    def _build(self) -> RichText:
        t = RichText()
        t.append("[branch] ", style="dim italic")
        if self._expanded:
            t.append(self._summary, style="dim")
        else:
            t.append(self._summary[:120], style="dim")
        return t

    def on_click(self) -> None:
        self._expanded = not self._expanded
        self.update(self._build())


class ErrorWidget(Static):
    """Renders an error message."""

    DEFAULT_CSS = """
    ErrorWidget {
        padding: 0 1;
        margin: 0 0 1 0;
        color: red;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__(RichText(f"[error] {message}", style="bold red"))


class InfoWidget(Static):
    """Renders an informational message."""

    DEFAULT_CSS = """
    InfoWidget {
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(RichText(text, style="dim"))


__all__ = [
    "AssistantMessageWidget",
    "BashExecutionWidget",
    "BranchSummaryWidget",
    "CompactionSummaryWidget",
    "ErrorWidget",
    "InfoWidget",
    "ToolExecutionWidget",
    "UserMessageWidget",
]
