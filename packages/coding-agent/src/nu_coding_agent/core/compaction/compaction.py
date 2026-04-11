"""Context compaction — direct port of ``packages/coding-agent/src/core/compaction/compaction.ts``.

Pure functions for picking a cut point in a long session and asking the
LLM to summarise the discarded prefix. The :class:`SessionManager` does
the actual on-disk write after :func:`compact` returns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import ceil, floor
from typing import TYPE_CHECKING, Any

from nu_ai.stream import complete_simple
from nu_ai.types import Context, Message, SimpleStreamOptions, TextContent, UserMessage

from nu_coding_agent.core.compaction.utils import (
    SUMMARIZATION_SYSTEM_PROMPT,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)
from nu_coding_agent.core.messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from nu_coding_agent.core.session_manager import build_session_context

if TYPE_CHECKING:
    import asyncio

    from nu_ai.types import Model, Usage


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompactionSettings:
    enabled: bool
    reserve_tokens: int
    keep_recent_tokens: int


DEFAULT_COMPACTION_SETTINGS = CompactionSettings(
    enabled=True,
    reserve_tokens=16384,
    keep_recent_tokens=20000,
)


@dataclass(slots=True)
class CompactionDetails:
    read_files: list[str]
    modified_files: list[str]


@dataclass(slots=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: CompactionDetails | None = None


# ---------------------------------------------------------------------------
# Helpers used by both prepare_compaction and compact
# ---------------------------------------------------------------------------


def _entry_role(entry: dict[str, Any]) -> str | None:
    if entry.get("type") != "message":
        return None
    message = entry.get("message")
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _get_message_from_entry(entry: dict[str, Any]) -> Any:
    entry_type = entry.get("type")
    if entry_type == "message":
        return entry.get("message")
    if entry_type == "custom_message":
        return create_custom_message(
            entry.get("customType", ""),
            entry.get("content", ""),
            bool(entry.get("display", False)),
            entry.get("details"),
            entry.get("timestamp", ""),
        )
    if entry_type == "branch_summary":
        return create_branch_summary_message(
            entry.get("summary", ""),
            entry.get("fromId", ""),
            entry.get("timestamp", ""),
        )
    if entry_type == "compaction":
        return create_compaction_summary_message(
            entry.get("summary", ""),
            int(entry.get("tokensBefore", 0)),
            entry.get("timestamp", ""),
        )
    return None


def _get_message_from_entry_for_compaction(entry: dict[str, Any]) -> Any:
    if entry.get("type") == "compaction":
        return None
    return _get_message_from_entry(entry)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def calculate_context_tokens(usage: Usage) -> int:
    """Total context tokens, falling back to component sum when totalTokens=0."""
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def _get_assistant_usage(msg: Any) -> Usage | None:
    role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
    if role != "assistant":
        return None
    stop_reason = getattr(msg, "stop_reason", None) or (msg.get("stopReason") if isinstance(msg, dict) else None)
    if stop_reason in ("aborted", "error"):
        return None
    usage = getattr(msg, "usage", None) or (msg.get("usage") if isinstance(msg, dict) else None)
    return usage


def get_last_assistant_usage(entries: list[dict[str, Any]]) -> Usage | None:
    """Walk entries in reverse and return the last good assistant usage record."""
    for entry in reversed(entries):
        if entry.get("type") != "message":
            continue
        usage = _get_assistant_usage(entry.get("message"))
        if usage is not None:
            return usage
    return None


@dataclass(slots=True)
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def _get_last_assistant_usage_info(messages: list[Any]) -> tuple[Usage, int] | None:
    for i in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[i])
        if usage is not None:
            return usage, i
    return None


def estimate_context_tokens(messages: list[Any]) -> ContextUsageEstimate:
    """Use the last assistant usage when available; otherwise fall back to char/4."""
    info = _get_last_assistant_usage_info(messages)
    if info is None:
        estimated = sum(estimate_tokens(m) for m in messages)
        return ContextUsageEstimate(
            tokens=estimated,
            usage_tokens=0,
            trailing_tokens=estimated,
            last_usage_index=None,
        )
    usage, index = info
    usage_tokens = calculate_context_tokens(usage)
    trailing = sum(estimate_tokens(m) for m in messages[index + 1 :])
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=index,
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


def _block_field(block: Any, name: str) -> Any:
    value = getattr(block, name, None)
    if value is None and isinstance(block, dict):
        value = block.get(name)
    return value


def estimate_tokens(message: Any) -> int:
    """Conservative chars/4 token estimate for a single message."""
    role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
    chars = 0
    if role == "user":
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if _block_field(block, "type") == "text":
                    chars += len(_block_field(block, "text") or "")
        return ceil(chars / 4)
    if role == "assistant":
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, list):
            for block in content:
                block_type = _block_field(block, "type")
                if block_type == "text":
                    chars += len(_block_field(block, "text") or "")
                elif block_type == "thinking":
                    chars += len(_block_field(block, "thinking") or "")
                elif block_type == "toolCall":
                    name = _block_field(block, "name") or ""
                    args = _block_field(block, "arguments") or {}
                    chars += len(str(name)) + len(repr(args))
        return ceil(chars / 4)
    if role in ("custom", "toolResult"):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if _block_field(block, "type") == "text":
                    chars += len(_block_field(block, "text") or "")
                elif _block_field(block, "type") == "image":
                    chars += 4800  # estimate images at 1200 tokens
        return ceil(chars / 4)
    if role == "bashExecution":
        command = getattr(message, "command", None) or (message.get("command") if isinstance(message, dict) else "")
        output = getattr(message, "output", None) or (message.get("output") if isinstance(message, dict) else "")
        return ceil((len(command or "") + len(output or "")) / 4)
    if role in ("branchSummary", "compactionSummary"):
        summary = getattr(message, "summary", None) or (message.get("summary") if isinstance(message, dict) else "")
        return ceil(len(summary or "") / 4)
    return 0


# ---------------------------------------------------------------------------
# Cut point detection
# ---------------------------------------------------------------------------


def _find_valid_cut_points(entries: list[dict[str, Any]], start_index: int, end_index: int) -> list[int]:
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if entry.get("type") == "message":
            role = _entry_role(entry)
            if role in ("user", "assistant", "bashExecution", "custom", "branchSummary", "compactionSummary"):
                cut_points.append(i)
        if entry.get("type") in ("branch_summary", "custom_message"):
            cut_points.append(i)
    return cut_points


def find_turn_start_index(entries: list[dict[str, Any]], entry_index: int, start_index: int) -> int:
    """Walk backwards from ``entry_index`` and return the index where the turn began."""
    for i in range(entry_index, start_index - 1, -1):
        entry = entries[i]
        if entry.get("type") in ("branch_summary", "custom_message"):
            return i
        if entry.get("type") == "message":
            role = _entry_role(entry)
            if role in ("user", "bashExecution"):
                return i
    return -1


@dataclass(slots=True)
class CutPointResult:
    first_kept_entry_index: int
    turn_start_index: int
    is_split_turn: bool


def find_cut_point(
    entries: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """Walk backwards accumulating message tokens until we cross the budget."""
    cut_points = _find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)

    accumulated = 0
    cut_index = cut_points[0]
    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        accumulated += estimate_tokens(message)
        if accumulated >= keep_recent_tokens:
            for c in cut_points:
                if c >= i:
                    cut_index = c
                    break
            break

    while cut_index > start_index:
        prev = entries[cut_index - 1]
        if prev.get("type") in ("compaction", "message"):
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_message = cut_entry.get("type") == "message" and _entry_role(cut_entry) == "user"
    turn_start_index = -1 if is_user_message else find_turn_start_index(entries, cut_index, start_index)
    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=not is_user_message and turn_start_index != -1,
    )


# ---------------------------------------------------------------------------
# Summarisation prompts
# ---------------------------------------------------------------------------


_SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


_UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


_TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


async def generate_summary(
    current_messages: list[Any],
    model: Model,
    reserve_tokens: int,
    api_key: str,
    headers: dict[str, str] | None = None,
    signal: asyncio.Event | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
) -> str:
    """Drive the LLM to produce a structured checkpoint summary."""
    max_tokens = floor(0.8 * reserve_tokens)
    base_prompt = _UPDATE_SUMMARIZATION_PROMPT if previous_summary else _SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    llm_messages = convert_to_llm(current_messages)
    conversation_text = serialize_conversation(llm_messages)

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    summarization_messages: list[Message] = [
        UserMessage(
            content=[TextContent(text=prompt_text)],
            timestamp=int(time.time() * 1000),
        )
    ]

    options = SimpleStreamOptions(max_tokens=max_tokens, api_key=api_key, headers=headers)
    response = await complete_simple(
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        options,
    )
    _ = signal

    if response.stop_reason == "error":
        raise ValueError(f"Summarization failed: {response.error_message or 'Unknown error'}")

    return "\n".join(c.text for c in response.content if isinstance(c, TextContent))


async def _generate_turn_prefix_summary(
    messages: list[Any],
    model: Model,
    reserve_tokens: int,
    api_key: str,
    headers: dict[str, str] | None,
    signal: asyncio.Event | None,
) -> str:
    max_tokens = floor(0.5 * reserve_tokens)
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{_TURN_PREFIX_SUMMARIZATION_PROMPT}"
    summarization_messages: list[Message] = [
        UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000))
    ]
    options = SimpleStreamOptions(max_tokens=max_tokens, api_key=api_key, headers=headers)
    response = await complete_simple(
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        options,
    )
    _ = signal
    if response.stop_reason == "error":
        raise ValueError(f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}")
    return "\n".join(c.text for c in response.content if isinstance(c, TextContent))


# ---------------------------------------------------------------------------
# Compaction preparation + main entry point
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompactionPreparation:
    first_kept_entry_id: str
    messages_to_summarize: list[Any]
    turn_prefix_messages: list[Any]
    is_split_turn: bool
    tokens_before: int
    file_ops: FileOperations
    settings: CompactionSettings
    previous_summary: str | None = None


def _extract_file_operations(
    messages: list[Any],
    entries: list[dict[str, Any]],
    prev_compaction_index: int,
) -> FileOperations:
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev = entries[prev_compaction_index]
        if not prev.get("fromHook") and isinstance(prev.get("details"), dict):
            details = prev["details"]
            for f in details.get("readFiles") or []:
                if isinstance(f, str):
                    file_ops.read.add(f)
            for f in details.get("modifiedFiles") or []:
                if isinstance(f, str):
                    file_ops.edited.add(f)
    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)
    return file_ops


def prepare_compaction(
    path_entries: list[dict[str, Any]],
    settings: CompactionSettings,
) -> CompactionPreparation | None:
    """Pick the cut point and bundle the messages-to-summarise."""
    if path_entries and path_entries[-1].get("type") == "compaction":
        return None

    prev_compaction_index = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if path_entries[i].get("type") == "compaction":
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev = path_entries[prev_compaction_index]
        previous_summary = str(prev.get("summary", "") or "")
        first_kept_id = prev.get("firstKeptEntryId")
        first_kept_index = next(
            (i for i, entry in enumerate(path_entries) if entry.get("id") == first_kept_id),
            -1,
        )
        boundary_start = first_kept_index if first_kept_index >= 0 else prev_compaction_index + 1
    boundary_end = len(path_entries)

    context_estimate = estimate_context_tokens(build_session_context(path_entries).messages)
    tokens_before = context_estimate.tokens

    cut_point = find_cut_point(path_entries, boundary_start, boundary_end, settings.keep_recent_tokens)

    first_kept_entry = path_entries[cut_point.first_kept_entry_index]
    if not first_kept_entry.get("id"):
        return None
    first_kept_entry_id = str(first_kept_entry["id"])

    history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index

    messages_to_summarize: list[Any] = []
    for i in range(boundary_start, history_end):
        msg = _get_message_from_entry_for_compaction(path_entries[i])
        if msg is not None:
            messages_to_summarize.append(msg)

    turn_prefix_messages: list[Any] = []
    if cut_point.is_split_turn:
        for i in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            msg = _get_message_from_entry_for_compaction(path_entries[i])
            if msg is not None:
                turn_prefix_messages.append(msg)

    file_ops = _extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)
    if cut_point.is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut_point.is_split_turn,
        tokens_before=tokens_before,
        file_ops=file_ops,
        settings=settings,
        previous_summary=previous_summary,
    )


async def compact(
    preparation: CompactionPreparation,
    model: Model,
    api_key: str,
    headers: dict[str, str] | None = None,
    custom_instructions: str | None = None,
    signal: asyncio.Event | None = None,
) -> CompactionResult:
    """Drive the LLM to produce the compaction summary and bundle the result."""
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_summary = (
            await generate_summary(
                preparation.messages_to_summarize,
                model,
                preparation.settings.reserve_tokens,
                api_key,
                headers,
                signal,
                custom_instructions,
                preparation.previous_summary,
            )
            if preparation.messages_to_summarize
            else "No prior history."
        )
        turn_prefix_summary = await _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            model,
            preparation.settings.reserve_tokens,
            api_key,
            headers,
            signal,
        )
        summary = f"{history_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_summary}"
    else:
        summary = await generate_summary(
            preparation.messages_to_summarize,
            model,
            preparation.settings.reserve_tokens,
            api_key,
            headers,
            signal,
            custom_instructions,
            preparation.previous_summary,
        )

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        details=CompactionDetails(read_files=read_files, modified_files=modified_files),
    )


__all__ = [
    "DEFAULT_COMPACTION_SETTINGS",
    "CompactionDetails",
    "CompactionPreparation",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
    "CutPointResult",
    "calculate_context_tokens",
    "compact",
    "estimate_context_tokens",
    "estimate_tokens",
    "find_cut_point",
    "find_turn_start_index",
    "generate_summary",
    "get_last_assistant_usage",
    "prepare_compaction",
    "should_compact",
]


# ``field`` is referenced indirectly through dataclass slots metadata in
# the future; keep the import alive for static analyzers that strip
# unused imports.
_ = field
