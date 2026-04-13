"""HTML export for session data — port of export-html/index.ts.

Provides two public async functions:
  ``export_session_to_html`` — exports using a live SessionManager + optional AgentState.
  ``export_from_file``       — exports an on-disk JSONL session file without live state.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nu_coding_agent.config import APP_NAME, get_export_template_dir
from nu_coding_agent.core.session_manager import SessionManager
from nu_coding_agent.modes.interactive.theme.theme import (
    get_resolved_theme_colors,
    get_theme_export_colors,
)

# ---------------------------------------------------------------------------
# ToolHtmlRenderer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolHtmlRenderer(Protocol):
    """Protocol for rendering custom tool calls/results to HTML.

    Used by interactive-mode export when extension tools have visual renderers.
    Callers that do not need custom rendering simply omit this from ExportOptions.
    """

    def render_call(self, tool_call_id: str, tool_name: str, args: object) -> str | None:
        """Render a tool call to HTML; return None if the tool has no renderer."""
        ...

    def render_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: list[dict[str, Any]],
        details: object,
        is_error: bool,
    ) -> dict[str, str | None] | None:
        """Render a tool result to collapsed/expanded HTML; return None if no renderer."""
        ...


# ---------------------------------------------------------------------------
# ExportOptions
# ---------------------------------------------------------------------------


@dataclass
class ExportOptions:
    output_path: str | None = None
    theme_name: str | None = None
    tool_renderer: ToolHtmlRenderer | None = None


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------


def _parse_color(color: str) -> tuple[int, int, int] | None:
    hex_m = re.match(r"^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$", color)
    if hex_m:
        return int(hex_m.group(1), 16), int(hex_m.group(2), 16), int(hex_m.group(3), 16)
    rgb_m = re.match(r"^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", color)
    if rgb_m:
        return int(rgb_m.group(1)), int(rgb_m.group(2)), int(rgb_m.group(3))
    return None


def _get_luminance(r: int, g: int, b: int) -> float:
    def to_linear(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)


def _adjust_brightness(color: str, factor: float) -> str:
    parsed = _parse_color(color)
    if not parsed:
        return color
    r, g, b = parsed

    def _clamp(c: int) -> int:
        return min(255, max(0, round(c * factor)))

    return f"rgb({_clamp(r)}, {_clamp(g)}, {_clamp(b)})"


def _derive_export_colors(base_color: str) -> dict[str, str]:
    """Derive page/card/info background colours from the user-message background."""
    parsed = _parse_color(base_color)
    if not parsed:
        return {
            "pageBg": "rgb(24, 24, 30)",
            "cardBg": "rgb(30, 30, 36)",
            "infoBg": "rgb(60, 55, 40)",
        }
    r, g, b = parsed
    is_light = _get_luminance(r, g, b) > 0.5
    if is_light:
        return {
            "pageBg": _adjust_brightness(base_color, 0.96),
            "cardBg": base_color,
            "infoBg": f"rgb({min(255, r + 10)}, {min(255, g + 5)}, {max(0, b - 20)})",
        }
    return {
        "pageBg": _adjust_brightness(base_color, 0.7),
        "cardBg": _adjust_brightness(base_color, 0.85),
        "infoBg": f"rgb({min(255, r + 20)}, {min(255, g + 15)}, {b})",
    }


# ---------------------------------------------------------------------------
# Theme CSS generation
# ---------------------------------------------------------------------------


def _generate_theme_vars(theme_name: str | None) -> str:
    colors = get_resolved_theme_colors(theme_name)
    lines = [f"--{key}: {value};" for key, value in colors.items()]

    theme_export = get_theme_export_colors(theme_name)
    user_message_bg = colors.get("userMessageBg", "#343541")
    derived = _derive_export_colors(user_message_bg)

    lines.append(f"--exportPageBg: {theme_export.get('pageBg') or derived['pageBg']};")
    lines.append(f"--exportCardBg: {theme_export.get('cardBg') or derived['cardBg']};")
    lines.append(f"--exportInfoBg: {theme_export.get('infoBg') or derived['infoBg']};")

    return "\n      ".join(lines)


# ---------------------------------------------------------------------------
# Custom tool pre-rendering
# ---------------------------------------------------------------------------

_BUILTIN_TOOLS: frozenset[str] = frozenset({"bash", "read", "write", "edit", "ls", "find", "grep"})


def _pre_render_custom_tools(
    entries: list[dict[str, Any]],
    tool_renderer: ToolHtmlRenderer,
) -> dict[str, dict[str, str | None]]:
    rendered: dict[str, dict[str, str | None]] = {}

    for entry in entries:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})

        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if block.get("type") == "toolCall" and block.get("name") not in _BUILTIN_TOOLS:
                    call_html = tool_renderer.render_call(block["id"], block["name"], block.get("arguments"))
                    if call_html is not None:
                        rendered[block["id"]] = {"callHtml": call_html}

        if msg.get("role") == "toolResult" and msg.get("toolCallId"):
            tool_name: str = msg.get("toolName", "")
            existing = rendered.get(msg["toolCallId"])
            if existing is not None or tool_name not in _BUILTIN_TOOLS:
                result = tool_renderer.render_result(
                    msg["toolCallId"],
                    tool_name,
                    msg.get("content", []),
                    msg.get("details"),
                    msg.get("isError", False),
                )
                if result is not None:
                    rendered[msg["toolCallId"]] = {
                        **(existing or {}),
                        "resultHtmlCollapsed": result.get("collapsed"),
                        "resultHtmlExpanded": result.get("expanded"),
                    }

    return rendered


# ---------------------------------------------------------------------------
# Core HTML generation
# ---------------------------------------------------------------------------


def _generate_html(session_data: dict[str, Any], theme_name: str | None) -> str:
    template_dir = Path(get_export_template_dir())
    template = (template_dir / "template.html").read_text(encoding="utf-8")
    template_css = (template_dir / "template.css").read_text(encoding="utf-8")
    template_js = (template_dir / "template.js").read_text(encoding="utf-8")
    marked_js = (template_dir / "vendor" / "marked.min.js").read_text(encoding="utf-8")
    hljs_js = (template_dir / "vendor" / "highlight.min.js").read_text(encoding="utf-8")

    theme_vars = _generate_theme_vars(theme_name)
    colors = get_resolved_theme_colors(theme_name)
    theme_export = get_theme_export_colors(theme_name)
    derived = _derive_export_colors(colors.get("userMessageBg", "#343541"))

    body_bg = theme_export.get("pageBg") or derived["pageBg"]
    container_bg = theme_export.get("cardBg") or derived["cardBg"]
    info_bg = theme_export.get("infoBg") or derived["infoBg"]

    # Base64-encode session data to avoid escaping issues in the <script> tag.
    session_data_b64 = base64.b64encode(json.dumps(session_data, ensure_ascii=False).encode()).decode()

    css = (
        template_css.replace("{{THEME_VARS}}", theme_vars)
        .replace("{{BODY_BG}}", body_bg)
        .replace("{{CONTAINER_BG}}", container_bg)
        .replace("{{INFO_BG}}", info_bg)
    )

    return (
        template.replace("{{CSS}}", css)
        .replace("{{JS}}", template_js)
        .replace("{{SESSION_DATA}}", session_data_b64)
        .replace("{{MARKED_JS}}", marked_js)
        .replace("{{HIGHLIGHT_JS}}", hljs_js)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def export_session_to_html(
    sm: SessionManager,
    state: object | None = None,
    options: ExportOptions | str | None = None,
) -> str:
    """Export a live session to an HTML file.

    Args:
        sm: SessionManager that owns the session JSONL file.
        state: Optional AgentState (duck-typed) to extract system_prompt and tools.
        options: ExportOptions or a plain output path string.

    Returns:
        Path of the written HTML file.
    """
    if isinstance(options, str):
        opts = ExportOptions(output_path=options)
    elif options is None:
        opts = ExportOptions()
    else:
        opts = options

    session_file = sm.get_session_file()
    if session_file is None:
        raise ValueError("Cannot export in-memory session to HTML")
    if not Path(session_file).exists():
        raise ValueError("Nothing to export yet - start a conversation first")

    entries = sm.get_entries()

    rendered_tools: dict[str, Any] | None = None
    if opts.tool_renderer is not None:
        rendered = _pre_render_custom_tools(entries, opts.tool_renderer)
        if rendered:
            rendered_tools = rendered

    system_prompt: str | None = getattr(state, "system_prompt", None) if state is not None else None
    tools_info: list[dict[str, Any]] | None = None
    if state is not None:
        tools = getattr(state, "tools", None)
        if tools:
            tools_info = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]

    header = sm.get_header()
    session_data: dict[str, Any] = {
        "header": header.to_dict() if header is not None else None,
        "entries": entries,
        "leafId": sm.get_leaf_id(),
        "systemPrompt": system_prompt,
        "tools": tools_info,
        "renderedTools": rendered_tools,
    }

    html = _generate_html(session_data, opts.theme_name)

    output_path = opts.output_path or f"{APP_NAME}-session-{Path(session_file).stem}.html"
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


async def export_from_file(
    input_path: str,
    options: ExportOptions | str | None = None,
) -> str:
    """Export a session JSONL file to HTML without needing a live session.

    Args:
        input_path: Path to the ``.jsonl`` session file.
        options: ExportOptions or a plain output path string.

    Returns:
        Path of the written HTML file.
    """
    if isinstance(options, str):
        opts = ExportOptions(output_path=options)
    elif options is None:
        opts = ExportOptions()
    else:
        opts = options

    if not Path(input_path).exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    sm = SessionManager.open(input_path)

    header = sm.get_header()
    session_data: dict[str, Any] = {
        "header": header.to_dict() if header is not None else None,
        "entries": sm.get_entries(),
        "leafId": sm.get_leaf_id(),
        "systemPrompt": None,
        "tools": None,
    }

    html = _generate_html(session_data, opts.theme_name)

    output_path = opts.output_path or f"{APP_NAME}-session-{Path(input_path).stem}.html"
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


__all__ = [
    "ExportOptions",
    "ToolHtmlRenderer",
    "export_from_file",
    "export_session_to_html",
]
