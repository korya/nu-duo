"""Cross-model message transformation.

Direct port of ``packages/ai/src/providers/transform-messages.ts``. Applied
before dispatching messages to a provider so that thinking blocks,
thought-signatures, and tool-call ids that are opaque to the target model are
rewritten or dropped, and orphaned tool calls receive synthetic error results
so providers that require strict tool-call/result pairing (Anthropic,
OpenAI Responses) don't reject the payload.

The two-pass algorithm mirrors upstream:

1. **Pass 1** (``_transform``): per-message rewriting. Thinking blocks are
   kept intact for same-model replays, converted to plain text for
   cross-model, and dropped entirely when empty or redacted-cross-model.
   Tool call ids can be normalized via ``normalize_tool_call_id``; the
   mapping is remembered so tool results can update their ``tool_call_id``
   on the fly.
2. **Pass 2** (``_insert_synthetic_results``): insert ``"No result
   provided"`` tool results for any tool call that lacks a matching result
   by the time a subsequent assistant or user message arrives, **unless**
   the assistant turn that produced them ended in error or was aborted — in
   which case the whole turn is skipped.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from nu_ai.types import (
    AssistantContent,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nu_ai.types import Message, Model


type NormalizeToolCallId = Callable[[str, "Model", AssistantMessage], str]


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: NormalizeToolCallId | None = None,
) -> list[Message]:
    """Rewrite ``messages`` so they are safe to replay against ``model``.

    See module docstring for the full algorithm. ``normalize_tool_call_id``
    is an optional callback — providers use it to force tool call ids into
    a shape the target API accepts (Anthropic requires
    ``^[a-zA-Z0-9_-]{1,64}$``, for example).
    """
    tool_call_id_map: dict[str, str] = {}
    transformed = [_transform(msg, model, normalize_tool_call_id, tool_call_id_map) for msg in messages]
    return _insert_synthetic_results(transformed)


# ---------------------------------------------------------------------------
# Pass 1 — per-message transformation
# ---------------------------------------------------------------------------


def _transform(
    msg: Message,
    model: Model,
    normalize_tool_call_id: NormalizeToolCallId | None,
    tool_call_id_map: dict[str, str],
) -> Message:
    if isinstance(msg, UserMessage):
        return msg

    if isinstance(msg, ToolResultMessage):
        normalized_id = tool_call_id_map.get(msg.tool_call_id)
        if normalized_id is not None and normalized_id != msg.tool_call_id:
            return msg.model_copy(update={"tool_call_id": normalized_id})
        return msg

    assert isinstance(msg, AssistantMessage)
    is_same_model = msg.provider == model.provider and msg.api == model.api and msg.model == model.id

    new_content: list[AssistantContent] = []
    for block in msg.content:
        transformed = _transform_assistant_block(
            block,
            model=model,
            source=msg,
            is_same_model=is_same_model,
            normalize_tool_call_id=normalize_tool_call_id,
            tool_call_id_map=tool_call_id_map,
        )
        if transformed is None:
            continue
        new_content.append(transformed)

    return msg.model_copy(update={"content": new_content})


def _transform_assistant_block(
    block: AssistantContent,
    *,
    model: Model,
    source: AssistantMessage,
    is_same_model: bool,
    normalize_tool_call_id: NormalizeToolCallId | None,
    tool_call_id_map: dict[str, str],
) -> AssistantContent | None:
    if isinstance(block, ThinkingContent):
        # Redacted thinking is opaque encrypted content. Only valid for the
        # same model; dropped cross-model to avoid API errors.
        if block.redacted:
            return block if is_same_model else None
        # Same-model thinking with a signature is preserved for replay even
        # when the thinking text is empty (OpenAI encrypted reasoning).
        if is_same_model and block.thinking_signature:
            return block
        # Drop empty thinking entirely.
        if not block.thinking or not block.thinking.strip():
            return None
        if is_same_model:
            return block
        return TextContent(text=block.thinking)

    if isinstance(block, TextContent):
        if is_same_model:
            return block
        # Strip text_signature on cross-model replay (provider-specific).
        return TextContent(text=block.text)

    # block is ToolCall — exhaustive over AssistantContent.
    normalized = block
    if not is_same_model and block.thought_signature is not None:
        normalized = normalized.model_copy(update={"thought_signature": None})
    if not is_same_model and normalize_tool_call_id is not None:
        new_id = normalize_tool_call_id(block.id, model, source)
        if new_id != block.id:
            tool_call_id_map[block.id] = new_id
            normalized = normalized.model_copy(update={"id": new_id})
    return normalized


# ---------------------------------------------------------------------------
# Pass 2 — synthetic tool-result insertion
# ---------------------------------------------------------------------------


def _synthetic_tool_result(tool_call: ToolCall) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=[TextContent(text="No result provided")],
        is_error=True,
        timestamp=int(time.time() * 1000),
    )


def _insert_synthetic_results(transformed: list[Message]) -> list[Message]:
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    def flush_orphans() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        for tc in pending_tool_calls:
            if tc.id not in existing_tool_result_ids:
                result.append(_synthetic_tool_result(tc))
        pending_tool_calls = []
        existing_tool_result_ids = set()

    for msg in transformed:
        if isinstance(msg, AssistantMessage):
            if pending_tool_calls:
                flush_orphans()

            # Skip errored/aborted assistant turns entirely — they are
            # incomplete and may have partial tool calls that would confuse
            # a replay.
            if msg.stop_reason in {"error", "aborted"}:
                continue

            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()

            result.append(msg)
            continue

        if isinstance(msg, ToolResultMessage):
            existing_tool_result_ids.add(msg.tool_call_id)
            result.append(msg)
            continue

        # User message interrupts the tool flow.
        if pending_tool_calls:
            flush_orphans()
        result.append(msg)

    return result


__all__ = ["NormalizeToolCallId", "transform_messages"]
