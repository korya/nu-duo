"""HTML export module — port of packages/coding-agent/src/core/export-html/.

Converts session data (JSONL tree) into a self-contained HTML file with:
- ANSI colour/style rendering in terminal output
- Theme-aware CSS custom properties
- Sidebar session-tree navigation
- Syntax-highlighted code via highlight.js
- Markdown rendering via marked.js

Public API::

    from nu_coding_agent.core.export_html import export_session_to_html, export_from_file

    # From a running session:
    output = await export_session_to_html(session_manager, agent_state)

    # From an on-disk JSONL file:
    output = await export_from_file("/path/to/session.jsonl")
"""

from __future__ import annotations

from nu_coding_agent.core.export_html.ansi_to_html import ansi_lines_to_html, ansi_to_html
from nu_coding_agent.core.export_html.index import (
    ExportOptions,
    ToolHtmlRenderer,
    export_from_file,
    export_session_to_html,
)

__all__ = [
    "ExportOptions",
    "ToolHtmlRenderer",
    "ansi_lines_to_html",
    "ansi_to_html",
    "export_from_file",
    "export_session_to_html",
]
