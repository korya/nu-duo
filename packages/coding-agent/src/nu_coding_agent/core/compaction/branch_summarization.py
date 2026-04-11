"""Branch summarization — direct port of ``packages/coding-agent/src/core/compaction/branch-summarization.ts``.

When the user navigates to a different point in the session tree, this
module summarises the abandoned branch so context isn't lost. The
summary lands as a ``branch_summary`` entry on the new path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from nu_ai.stream import complete_simple
from nu_ai.types import Context, Message, SimpleStreamOptions, TextContent, UserMessage

from nu_coding_agent.core.compaction.compaction import estimate_tokens
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

if TYPE_CHECKING:
    import asyncio

    from nu_ai.types import Model


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BranchSummaryResult:
    summary: str | None = None
    read_files: list[str] | None = None
    modified_files: list[str] | None = None
    aborted: bool = False
    error: str | None = None


@dataclass(slots=True)
class BranchSummaryDetails:
    read_files: list[str]
    modified_files: list[str]


@dataclass(slots=True)
class BranchPreparation:
    messages: list[Any]
    file_ops: FileOperations
    total_tokens: int


@dataclass(slots=True)
class CollectEntriesResult:
    entries: list[dict[str, Any]]
    common_ancestor_id: str | None


@dataclass(slots=True)
class GenerateBranchSummaryOptions:
    model: Model
    api_key: str
    headers: dict[str, str] | None = None
    signal: asyncio.Event | None = None
    custom_instructions: str | None = None
    replace_instructions: bool = False
    reserve_tokens: int = 16384


# ---------------------------------------------------------------------------
# Read-only session source — structural protocol so callers don't have to
# import the full ``SessionManager`` shape.
# ---------------------------------------------------------------------------


class ReadOnlySessionSource(Protocol):
    def get_branch(self, from_id: str | None = None) -> list[dict[str, Any]]: ...
    def get_entry(self, entry_id: str) -> dict[str, Any] | None: ...


# ---------------------------------------------------------------------------
# Entry collection
# ---------------------------------------------------------------------------


def collect_entries_for_branch_summary(
    session: ReadOnlySessionSource,
    old_leaf_id: str | None,
    target_id: str,
) -> CollectEntriesResult:
    """Walk from ``old_leaf_id`` back to the lowest common ancestor with ``target_id``."""
    if old_leaf_id is None:
        return CollectEntriesResult(entries=[], common_ancestor_id=None)
    old_path = {entry.get("id") for entry in session.get_branch(old_leaf_id)}
    target_path = session.get_branch(target_id)
    common_ancestor_id: str | None = None
    for entry in reversed(target_path):
        if entry.get("id") in old_path:
            common_ancestor_id = entry.get("id")
            break
    entries: list[dict[str, Any]] = []
    current: str | None = old_leaf_id
    while current and current != common_ancestor_id:
        entry = session.get_entry(current)
        if entry is None:
            break
        entries.append(entry)
        current = entry.get("parentId")
    entries.reverse()
    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


# ---------------------------------------------------------------------------
# Entry → message conversion
# ---------------------------------------------------------------------------


def _get_message_from_entry(entry: dict[str, Any]) -> Any:
    entry_type = entry.get("type")
    if entry_type == "message":
        message = entry.get("message")
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "toolResult":
            return None
        return message
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


# ---------------------------------------------------------------------------
# prepare_branch_entries
# ---------------------------------------------------------------------------


def prepare_branch_entries(
    entries: list[dict[str, Any]],
    token_budget: int = 0,
) -> BranchPreparation:
    """Walk entries newest → oldest, accumulating until ``token_budget`` is hit.

    A ``token_budget`` of ``0`` means "no limit". File operations are
    extracted from every entry, including from previous pi-generated
    branch summaries (``fromHook != True``) so cumulative tracking
    survives nested branches.
    """
    messages: list[Any] = []
    file_ops = create_file_ops()
    total_tokens = 0

    # First pass — collect file ops from all entries (so cumulative tracking
    # survives even when the budget pushes them out of the message list).
    for entry in entries:
        if entry.get("type") != "branch_summary":
            continue
        if entry.get("fromHook"):
            continue
        details = entry.get("details")
        if not isinstance(details, dict):
            continue
        for f in details.get("readFiles") or []:
            if isinstance(f, str):
                file_ops.read.add(f)
        for f in details.get("modifiedFiles") or []:
            if isinstance(f, str):
                file_ops.edited.add(f)

    # Second pass — newest → oldest until we hit the budget.
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        message = _get_message_from_entry(entry)
        if message is None:
            continue
        extract_file_ops_from_message(message, file_ops)
        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            # Summary entries are too important to drop — squeeze them in
            # if we're still under 90% of the budget.
            if entry.get("type") in ("compaction", "branch_summary") and total_tokens < token_budget * 0.9:
                messages.insert(0, message)
                total_tokens += tokens
            break
        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(messages=messages, file_ops=file_ops, total_tokens=total_tokens)


# ---------------------------------------------------------------------------
# Summarisation prompt + entry point
# ---------------------------------------------------------------------------


_BRANCH_SUMMARY_PREAMBLE = (
    "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n"
)


_BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(
    entries: list[dict[str, Any]],
    options: GenerateBranchSummaryOptions,
) -> BranchSummaryResult:
    """Drive the LLM to summarise an abandoned branch."""
    context_window = options.model.context_window or 128000
    token_budget = context_window - options.reserve_tokens

    preparation = prepare_branch_entries(entries, token_budget)
    if not preparation.messages:
        return BranchSummaryResult(summary="No content to summarize")

    llm_messages = convert_to_llm(preparation.messages)
    conversation_text = serialize_conversation(llm_messages)

    if options.replace_instructions and options.custom_instructions:
        instructions = options.custom_instructions
    elif options.custom_instructions:
        instructions = f"{_BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {options.custom_instructions}"
    else:
        instructions = _BRANCH_SUMMARY_PROMPT

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"
    summarization_messages: list[Message] = [
        UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000))
    ]

    stream_options = SimpleStreamOptions(
        max_tokens=2048,
        api_key=options.api_key,
        headers=options.headers,
    )
    response = await complete_simple(
        options.model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        stream_options,
    )
    _ = options.signal

    if response.stop_reason == "aborted":
        return BranchSummaryResult(aborted=True)
    if response.stop_reason == "error":
        return BranchSummaryResult(error=response.error_message or "Summarization failed")

    summary = "\n".join(c.text for c in response.content if isinstance(c, TextContent))
    summary = _BRANCH_SUMMARY_PREAMBLE + summary

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return BranchSummaryResult(
        summary=summary or "No summary generated",
        read_files=read_files,
        modified_files=modified_files,
    )


__all__ = [
    "BranchPreparation",
    "BranchSummaryDetails",
    "BranchSummaryResult",
    "CollectEntriesResult",
    "GenerateBranchSummaryOptions",
    "ReadOnlySessionSource",
    "collect_entries_for_branch_summary",
    "generate_branch_summary",
    "prepare_branch_entries",
]


_ = field  # keep import alive across edits
