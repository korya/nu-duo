"""Per-tool Rich Text renderers — Python replacement for the TS renderCall/renderResult system.

In upstream pi-mono each tool definition carries ``renderCall(args, theme, ctx) -> Component``
and ``renderResult(...) -> Component`` factory functions that are dispatched by
``ToolExecutionComponent``. In Python we replace that pluggable component system with
plain functions dispatched by tool name, returning Rich ``Text`` objects.

Public API
----------
- ``render_tool_call(tool_name, args) -> Text | None``
- ``render_tool_result(tool_name, args, content, is_error, expanded) -> Text | None``
"""

from __future__ import annotations

import os
from typing import Any

from rich.text import Text

from nu_coding_agent.modes.interactive.components.diff import render_diff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shorten_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _text_content(content: list[Any]) -> str:
    """Extract and join all text blocks from a tool result content list."""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif hasattr(block, "type") and getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(parts)


def _preview(text: str, max_lines: int = 5) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n… ({len(lines) - max_lines} more lines)"


# ---------------------------------------------------------------------------
# Bash — call only (result is rendered by BashExecutionWidget directly)
# ---------------------------------------------------------------------------


def _render_bash_call(args: dict[str, Any]) -> Text:
    cmd = _str(args.get("command", ""))
    t = Text()
    t.append("$ ", style="bold green")
    t.append(cmd, style="green")
    return t


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _render_read_call(args: dict[str, Any]) -> Text:
    path = _shorten_path(_str(args.get("file_path") or args.get("path", "")))
    start = args.get("start_line")
    end = args.get("end_line")
    t = Text()
    t.append("Read ", style="dim")
    t.append(path, style="bold")
    if start is not None or end is not None:
        t.append(f"  :{start or 1}-{end or 'end'}", style="dim")
    return t


def _render_read_result(args: dict[str, Any], content: list[Any], is_error: bool) -> Text | None:
    output = _text_content(content).strip()
    if not output:
        return None
    t = Text()
    if is_error:
        t.append(output, style="red")
        return t
    lines = output.splitlines()
    count = len(lines)
    path = _shorten_path(_str(args.get("file_path") or args.get("path", "")))
    t.append(f"{count} lines", style="dim")
    t.append(f"  {path}", style="dim italic")
    return t


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _render_write_call(args: dict[str, Any]) -> Text:
    path = _shorten_path(_str(args.get("file_path") or args.get("path", "")))
    content = _str(args.get("content", ""))
    size = len(content.encode())
    t = Text()
    t.append("Write ", style="dim")
    t.append(path, style="bold")
    if size:
        t.append(f"  {size} bytes", style="dim")
    return t


def _render_write_result(args: dict[str, Any], content: list[Any], is_error: bool) -> Text | None:
    output = _text_content(content).strip()
    if is_error:
        t = Text()
        t.append(output or "error", style="red")
        return t
    # Success is silent like the upstream — the call line already shows the path.
    return None


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


def _render_edit_call(args: dict[str, Any]) -> Text:
    path = _shorten_path(_str(args.get("file_path") or args.get("path", "")))
    t = Text()
    t.append("Edit ", style="dim")
    t.append(path, style="bold")
    return t


def _render_edit_result(args: dict[str, Any], content: list[Any], is_error: bool, expanded: bool) -> Text | None:
    output = _text_content(content).strip()
    if is_error:
        t = Text()
        t.append(output or "error", style="red")
        return t
    if not output:
        return None
    # The result text may be a unified diff — render it coloured.
    if output.startswith(("-", "+")) or " " in output[:2]:
        return render_diff(output)
    t = Text()
    t.append(output if expanded else _preview(output), style="dim")
    return t


# ---------------------------------------------------------------------------
# Edit-diff
# ---------------------------------------------------------------------------


def _render_edit_diff_call(args: dict[str, Any]) -> Text:
    path = _shorten_path(_str(args.get("file_path") or args.get("path", "")))
    t = Text()
    t.append("EditDiff ", style="dim")
    t.append(path, style="bold")
    return t


def _render_edit_diff_result(args: dict[str, Any], content: list[Any], is_error: bool) -> Text | None:
    output = _text_content(content).strip()
    if is_error:
        t = Text()
        t.append(output or "error", style="red")
        return t
    return None


# ---------------------------------------------------------------------------
# Grep
# ---------------------------------------------------------------------------


def _render_grep_call(args: dict[str, Any]) -> Text:
    pattern = _str(args.get("pattern", ""))
    path = _shorten_path(_str(args.get("path", args.get("directory", ""))))
    t = Text()
    t.append("Grep ", style="dim")
    t.append(pattern, style="bold")
    if path:
        t.append(f"  in {path}", style="dim")
    return t


def _render_grep_result(args: dict[str, Any], content: list[Any], is_error: bool, expanded: bool) -> Text | None:
    output = _text_content(content).strip()
    if is_error:
        t = Text()
        t.append(output or "error", style="red")
        return t
    if not output:
        t = Text()
        t.append("no matches", style="dim")
        return t
    lines = output.splitlines()
    t = Text()
    display = output if expanded else _preview(output)
    t.append(display, style="dim")
    if not expanded and len(lines) > 5:
        t.append(f"\n{len(lines)} matches", style="dim italic")
    return t


# ---------------------------------------------------------------------------
# Find
# ---------------------------------------------------------------------------


def _render_find_call(args: dict[str, Any]) -> Text:
    pattern = _str(args.get("pattern", ""))
    path = _shorten_path(_str(args.get("path", args.get("directory", ""))))
    t = Text()
    t.append("Find ", style="dim")
    t.append(pattern, style="bold")
    if path:
        t.append(f"  in {path}", style="dim")
    return t


def _render_find_result(args: dict[str, Any], content: list[Any], is_error: bool) -> Text | None:
    output = _text_content(content).strip()
    if is_error:
        t = Text()
        t.append(output or "error", style="red")
        return t
    if not output:
        t = Text()
        t.append("no results", style="dim")
        return t
    lines = output.splitlines()
    t = Text()
    t.append(f"{len(lines)} results", style="dim")
    return t


# ---------------------------------------------------------------------------
# Ls
# ---------------------------------------------------------------------------


def _render_ls_call(args: dict[str, Any]) -> Text:
    path = _shorten_path(_str(args.get("path", args.get("directory", ""))))
    t = Text()
    t.append("Ls ", style="dim")
    t.append(path or ".", style="bold")
    return t


def _render_ls_result(args: dict[str, Any], content: list[Any], is_error: bool) -> Text | None:
    output = _text_content(content).strip()
    if is_error:
        t = Text()
        t.append(output or "error", style="red")
        return t
    if not output:
        return None
    lines = output.splitlines()
    t = Text()
    t.append(f"{len(lines)} entries", style="dim")
    return t


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def _render_generic_call(tool_name: str, args: dict[str, Any]) -> Text:
    t = Text()
    t.append(tool_name, style="bold")
    return t


def _render_generic_result(content: list[Any], is_error: bool, expanded: bool) -> Text | None:
    output = _text_content(content).strip()
    if not output:
        return None
    t = Text()
    style = "red" if is_error else "dim"
    t.append(output if expanded else _preview(output), style=style)
    return t


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

_BASH_NAMES = {"bash", "Bash"}
_READ_NAMES = {"read", "Read"}
_WRITE_NAMES = {"write", "Write"}
_EDIT_NAMES = {"edit", "Edit"}
_EDIT_DIFF_NAMES = {"edit_diff", "EditDiff", "edit-diff"}
_GREP_NAMES = {"grep", "Grep"}
_FIND_NAMES = {"find", "Find"}
_LS_NAMES = {"ls", "Ls"}


def render_tool_call(tool_name: str, args: dict[str, Any]) -> Text | None:
    """Return a Rich ``Text`` that represents the *call* side of a tool execution.

    Returns ``None`` for bash because bash's call line is rendered by
    ``BashExecutionWidget`` directly.
    """
    if tool_name in _BASH_NAMES:
        return _render_bash_call(args)
    if tool_name in _READ_NAMES:
        return _render_read_call(args)
    if tool_name in _WRITE_NAMES:
        return _render_write_call(args)
    if tool_name in _EDIT_NAMES:
        return _render_edit_call(args)
    if tool_name in _EDIT_DIFF_NAMES:
        return _render_edit_diff_call(args)
    if tool_name in _GREP_NAMES:
        return _render_grep_call(args)
    if tool_name in _FIND_NAMES:
        return _render_find_call(args)
    if tool_name in _LS_NAMES:
        return _render_ls_call(args)
    return _render_generic_call(tool_name, args)


def render_tool_result(
    tool_name: str,
    args: dict[str, Any],
    content: list[Any],
    *,
    is_error: bool = False,
    expanded: bool = False,
) -> Text | None:
    """Return a Rich ``Text`` for the *result* side of a tool execution.

    Returns ``None`` when the result has no useful display (e.g. a silent
    write success), indicating the caller should show nothing.
    """
    if tool_name in _READ_NAMES:
        return _render_read_result(args, content, is_error)
    if tool_name in _WRITE_NAMES:
        return _render_write_result(args, content, is_error)
    if tool_name in _EDIT_NAMES:
        return _render_edit_result(args, content, is_error, expanded)
    if tool_name in _EDIT_DIFF_NAMES:
        return _render_edit_diff_result(args, content, is_error)
    if tool_name in _GREP_NAMES:
        return _render_grep_result(args, content, is_error, expanded)
    if tool_name in _FIND_NAMES:
        return _render_find_result(args, content, is_error)
    if tool_name in _LS_NAMES:
        return _render_ls_result(args, content, is_error)
    # Bash result is handled by BashExecutionWidget; return None here.
    if tool_name in _BASH_NAMES:
        return None
    return _render_generic_result(content, is_error, expanded)


__all__ = ["render_tool_call", "render_tool_result"]
