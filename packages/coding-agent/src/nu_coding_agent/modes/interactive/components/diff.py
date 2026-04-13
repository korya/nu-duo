"""Diff rendering — port of components/diff.ts.

Renders unified diffs with:
- Context lines: dim
- Removed lines: red
- Added lines: green
- Intra-line word-level highlighting (inverted) when exactly one line changed
"""

from __future__ import annotations

import difflib
import re

from rich.text import Text


def _parse_diff_line(line: str) -> tuple[str, str, str] | None:
    """Return (prefix, line_num, content) or None if not a recognised diff line."""
    m = re.match(r"^([+\- ])(\s*\d*)\s(.*)$", line)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _replace_tabs(text: str) -> str:
    return text.replace("\t", "   ")


def _word_diff(old: str, new: str) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
    """Word-level diff. Returns (old_parts, new_parts) where bool = changed."""
    old_tokens = re.findall(r"\S+|\s+", old)
    new_tokens = re.findall(r"\S+|\s+", new)
    sm = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    old_parts: list[tuple[str, bool]] = []
    new_parts: list[tuple[str, bool]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old_chunk = "".join(old_tokens[i1:i2])
        new_chunk = "".join(new_tokens[j1:j2])
        if tag == "equal":
            if old_chunk:
                old_parts.append((old_chunk, False))
            if new_chunk:
                new_parts.append((new_chunk, False))
        elif tag == "replace":
            if old_chunk:
                old_parts.append((old_chunk, True))
            if new_chunk:
                new_parts.append((new_chunk, True))
        elif tag == "delete":
            if old_chunk:
                old_parts.append((old_chunk, True))
        elif tag == "insert":
            if new_chunk:
                new_parts.append((new_chunk, True))
    return old_parts, new_parts


def _append_intra_line(
    result: Text,
    prefix_text: str,
    parts: list[tuple[str, bool]],
    base_style: str,
) -> None:
    result.append(prefix_text, style=base_style)
    for text, changed in parts:
        style = f"{base_style} reverse" if changed else base_style
        result.append(text, style=style)


def render_diff(diff_text: str) -> Text:
    """Render a unified diff as a Rich Text with colour and intra-line highlights."""
    lines = diff_text.split("\n")
    result = Text()
    first = True

    i = 0
    while i < len(lines):
        line = lines[i]
        if not first:
            result.append("\n")
        first = False

        parsed = _parse_diff_line(line)
        if parsed is None:
            result.append(line, style="dim")
            i += 1
            continue

        prefix, line_num, content = parsed

        if prefix == "-":
            # Collect consecutive removed then consecutive added lines
            removed: list[tuple[str, str]] = []
            added: list[tuple[str, str]] = []
            while i < len(lines):
                p = _parse_diff_line(lines[i])
                if not p or p[0] != "-":
                    break
                removed.append((p[1], p[2]))
                i += 1
            while i < len(lines):
                p = _parse_diff_line(lines[i])
                if not p or p[0] != "+":
                    break
                added.append((p[1], p[2]))
                i += 1

            if len(removed) == 1 and len(added) == 1:
                r_num, r_content = removed[0]
                a_num, a_content = added[0]
                old_parts, new_parts = _word_diff(_replace_tabs(r_content), _replace_tabs(a_content))
                _append_intra_line(result, f"-{r_num} ", old_parts, "red")
                result.append("\n")
                _append_intra_line(result, f"+{a_num} ", new_parts, "green")
            else:
                for j, (num, c) in enumerate(removed):
                    if j > 0:
                        result.append("\n")
                    result.append(f"-{num} {_replace_tabs(c)}", style="red")
                for num, c in added:
                    result.append("\n")
                    result.append(f"+{num} {_replace_tabs(c)}", style="green")

        elif prefix == "+":
            result.append(f"+{line_num} {_replace_tabs(content)}", style="green")
            i += 1
        else:
            result.append(f" {line_num} {_replace_tabs(content)}", style="dim")
            i += 1

    return result


__all__ = ["render_diff"]
