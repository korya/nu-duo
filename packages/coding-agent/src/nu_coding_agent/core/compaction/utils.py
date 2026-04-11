"""Compaction helper utilities — direct port of ``packages/coding-agent/src/core/compaction/utils.ts``.

Pure functions: file-operation tracking from tool calls + the
serialiser used to render a conversation as plain text for the
summarisation prompt. No I/O, no LLM calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nu_ai.types import Message


# ---------------------------------------------------------------------------
# File operation tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FileOperations:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def create_file_ops() -> FileOperations:
    return FileOperations()


def extract_file_ops_from_message(message: Any, file_ops: FileOperations) -> None:
    """Extract read/write/edit file paths from an assistant message's tool calls."""
    role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
    if role != "assistant":
        return
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if block_type != "toolCall":
            continue
        name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
        args = getattr(block, "arguments", None)
        if args is None and isinstance(block, dict):
            args = block.get("arguments")
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not isinstance(path, str):
            continue
        if name == "read":
            file_ops.read.add(path)
        elif name == "write":
            file_ops.written.add(path)
        elif name == "edit":
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> tuple[list[str], list[str]]:
    """Return ``(read_only_files, modified_files)`` from a :class:`FileOperations`."""
    modified = file_ops.edited | file_ops.written
    read_only = sorted(f for f in file_ops.read if f not in modified)
    modified_files = sorted(modified)
    return read_only, modified_files


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Render the ``<read-files>`` / ``<modified-files>`` XML block for the summary."""
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Conversation serialisation
# ---------------------------------------------------------------------------


_TOOL_RESULT_MAX_CHARS = 2000


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated} more characters truncated]"


def _msg_role(msg: Any) -> str | None:
    role = getattr(msg, "role", None)
    if role is None and isinstance(msg, dict):
        role = msg.get("role")
    return role if isinstance(role, str) else None


def _msg_content(msg: Any) -> Any:
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    return content


def _block_field(block: Any, name: str) -> Any:
    value = getattr(block, name, None)
    if value is None and isinstance(block, dict):
        value = block.get(name)
    return value


def serialize_conversation(messages: list[Message]) -> str:
    """Render an LLM message list as plain text for the summarisation prompt.

    Tool results are truncated to ``_TOOL_RESULT_MAX_CHARS`` so a noisy
    log doesn't blow the summarisation token budget.
    """
    parts: list[str] = []
    for msg in messages:
        role = _msg_role(msg)
        content = _msg_content(msg)
        if role == "user":
            text: str
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(_block_field(b, "text") or "" for b in content if _block_field(b, "type") == "text")
            else:
                text = ""  # pragma: no cover — defensive
            if text:
                parts.append(f"[User]: {text}")
        elif role == "assistant":
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            if isinstance(content, list):
                for block in content:
                    block_type = _block_field(block, "type")
                    if block_type == "text":
                        text_parts.append(_block_field(block, "text") or "")
                    elif block_type == "thinking":
                        thinking_parts.append(_block_field(block, "thinking") or "")
                    elif block_type == "toolCall":
                        args = _block_field(block, "arguments") or {}
                        args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
                        tool_calls.append(f"{_block_field(block, 'name')}({args_str})")
            if thinking_parts:
                parts.append("[Assistant thinking]: " + "\n".join(thinking_parts))
            if text_parts:
                parts.append("[Assistant]: " + "\n".join(text_parts))
            if tool_calls:
                parts.append("[Assistant tool calls]: " + "; ".join(tool_calls))
        elif role == "toolResult":
            if isinstance(content, list):
                text = "".join(_block_field(b, "text") or "" for b in content if _block_field(b, "type") == "text")
                if text:
                    parts.append("[Tool result]: " + _truncate_for_summary(text, _TOOL_RESULT_MAX_CHARS))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Summarisation system prompt
# ---------------------------------------------------------------------------


SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary "
    "following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)


__all__ = [
    "SUMMARIZATION_SYSTEM_PROMPT",
    "FileOperations",
    "compute_file_lists",
    "create_file_ops",
    "extract_file_ops_from_message",
    "format_file_operations",
    "serialize_conversation",
]
