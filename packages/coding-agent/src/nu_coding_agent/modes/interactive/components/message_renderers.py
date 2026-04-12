"""Message renderer widgets for the interactive REPL.

Each widget renders a specific type of conversation entry (user
message, assistant reply, tool call, compaction summary) as a
Textual ``Static`` backed by Rich renderables for styling.

Port of the component files in
``packages/coding-agent/src/modes/interactive/components/``:
- ``user-message.ts`` (33 LoC)
- ``assistant-message.ts`` (130 LoC)
- ``tool-execution.ts`` (328 LoC)
- ``compaction-summary-message.ts`` (59 LoC)
- ``branch-summary-message.ts`` (58 LoC)
- ``bash-execution.ts`` (218 LoC)
- ``custom-message.ts`` (99 LoC)
- ``skill-invocation-message.ts`` (55 LoC)
"""

from __future__ import annotations

import json
from typing import Any

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text as RichText
from textual.widgets import Static


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
    """Renders a tool call with name, args preview, and status."""

    DEFAULT_CSS = """
    ToolExecutionWidget {
        padding: 0 1;
        margin: 0 0 0 0;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        tool_name: str,
        *,
        args: dict[str, Any] | None = None,
        is_error: bool = False,
        result_preview: str = "",
    ) -> None:
        status = "error" if is_error else "ok"
        args_str = ""
        if args:
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 120:
                args_str = args_str[:120] + "…"
            args_str = f"({args_str})"

        line = f"[tool {status}] {tool_name}{args_str}"
        if result_preview:
            preview = result_preview[:200]
            line += f"\n  → {preview}"

        style = "red" if is_error else "dim"
        super().__init__(RichText(line, style=style))


class CompactionSummaryWidget(Static):
    """Renders a compaction summary indicator."""

    DEFAULT_CSS = """
    CompactionSummaryWidget {
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
    }
    """

    def __init__(self, summary: str, tokens_before: int = 0) -> None:
        text = f"[compacted] {tokens_before} tokens → summarized"
        if summary:
            # Show first line of summary
            first_line = summary.split("\n", 1)[0][:100]
            text += f"\n  {first_line}"
        super().__init__(RichText(text, style="dim italic"))


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
        text = f"[branch] {summary[:100]}"
        super().__init__(RichText(text, style="dim"))


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
    "BranchSummaryWidget",
    "CompactionSummaryWidget",
    "ErrorWidget",
    "InfoWidget",
    "ToolExecutionWidget",
    "UserMessageWidget",
]
