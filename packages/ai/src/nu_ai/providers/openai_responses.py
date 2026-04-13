"""OpenAI Responses API provider.

Direct port of ``packages/ai/src/providers/openai-responses.ts`` and
``packages/ai/src/providers/openai-responses-shared.ts``. Covers the OpenAI
Responses wire format — ``POST /v1/responses`` — which differs from the Chat
Completions API in several important ways:

* Items-based input/output rather than messages-based.
* Streaming uses different event types (``response.output_item.added``,
  ``response.output_text.delta``, etc.).
* Reasoning models receive encrypted reasoning items that are replayed as
  opaque ``thinkingSignature`` blobs.
* Tool calls carry both a ``call_id`` (short, stable) and an ``id``
  (``fc_xxx`` item id) that both need to be tracked.

This module has two logical sections:

1. **Pure transforms** — ``convert_responses_messages``,
   ``convert_responses_tools``, ``build_params``, ``map_stop_reason``.
   All SDK-free so they can be unit-tested in isolation.
2. **SDK glue** — ``create_client``, ``stream_openai_responses``,
   ``stream_simple_openai_responses`` built on the Python ``openai`` SDK.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, Any

from nu_ai.env_api_keys import get_env_api_key
from nu_ai.models import calculate_cost, supports_xhigh
from nu_ai.providers.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from nu_ai.providers.simple_options import build_base_options, clamp_reasoning
from nu_ai.providers.transform_messages import transform_messages
from nu_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from nu_ai.utils.event_stream import AssistantMessageEventStream
from nu_ai.utils.hash import short_hash
from nu_ai.utils.json_parse import parse_streaming_json
from nu_ai.utils.sanitize_unicode import sanitize_surrogates

if TYPE_CHECKING:
    from nu_ai.types import (
        Context,
        Model,
        SimpleStreamOptions,
    )


# ---------------------------------------------------------------------------
# Provider-specific options
# ---------------------------------------------------------------------------


class OpenAIResponsesOptions(StreamOptions):
    """OpenAI Responses API-specific stream options.

    Mirrors ``OpenAIResponsesOptions`` from
    ``packages/ai/src/providers/openai-responses.ts``.
    """

    reasoning_effort: ThinkingLevel | None = None
    """Reasoning effort level for reasoning models."""

    reasoning_summary: str | None = None
    """Reasoning summary format: ``"auto"``, ``"detailed"``, ``"concise"``, or ``None``."""

    service_tier: str | None = None
    """Service tier: ``"flex"``, ``"priority"``, or ``None`` for default."""


# Providers that use the native Responses API tool-call id format (call_id|item_id).
_OPENAI_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})

# Characters not allowed in Responses API call ids / item ids.
_ILLEGAL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _encode_text_signature_v1(item_id: str, phase: str | None = None) -> str:
    """Encode a Responses API item id + optional phase as a JSON signature."""
    payload: dict[str, Any] = {"v": 1, "id": item_id}
    if phase:
        payload["phase"] = phase
    return json.dumps(payload)


def _parse_text_signature(
    signature: str | None,
) -> dict[str, Any] | None:
    """Parse a text signature back into ``{id, phase?}``."""
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
            if parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
                result: dict[str, Any] = {"id": parsed["id"]}
                if parsed.get("phase") in ("commentary", "final_answer"):
                    result["phase"] = parsed["phase"]
                return result
        except (ValueError, TypeError):
            pass
    return {"id": signature}


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _normalize_id_part(part: str) -> str:
    sanitized = _ILLEGAL_ID_CHARS.sub("_", part)
    normalized = sanitized[:64]
    return normalized.rstrip("_")


def _build_foreign_responses_item_id(item_id: str) -> str:
    normalized = f"fc_{short_hash(item_id)}"
    return normalized[:64]


def _make_normalize_tool_call_id(
    model: Model,
) -> Any:
    """Build a ``normalize_tool_call_id`` callback for :func:`transform_messages`."""

    def normalize(tool_call_id: str, _target_model: Model, source: AssistantMessage) -> str:
        if model.provider not in _OPENAI_TOOL_CALL_PROVIDERS:
            return _normalize_id_part(tool_call_id)
        if "|" not in tool_call_id:
            return _normalize_id_part(tool_call_id)
        call_id, item_id_raw = tool_call_id.split("|", 1)
        normalized_call_id = _normalize_id_part(call_id)
        is_foreign = source.provider != model.provider or source.api != model.api
        if is_foreign:
            normalized_item_id = _build_foreign_responses_item_id(item_id_raw)
        else:
            normalized_item_id = _normalize_id_part(item_id_raw)
        # Responses API requires item id to start with "fc_"
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = _normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    return normalize


def convert_responses_messages(
    model: Model,
    context: Context,
    *,
    include_system_prompt: bool = True,
) -> list[dict[str, Any]]:
    """Convert nu_ai context messages to the OpenAI Responses API ``input`` array.

    Returns a list of Responses API input items (dicts). Mirrors
    ``convertResponsesMessages`` from ``openai-responses-shared.ts``.
    """
    messages: list[dict[str, Any]] = []
    transformed = transform_messages(context.messages, model, _make_normalize_tool_call_id(model))

    if include_system_prompt and context.system_prompt:
        role = "developer" if model.reasoning else "system"
        messages.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    msg_index = 0
    for msg in transformed:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}],
                    }
                )
            else:
                content: list[dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        content.append({"type": "input_text", "text": sanitize_surrogates(item.text)})
                    elif isinstance(item, ImageContent):
                        content.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.mime_type};base64,{item.data}",
                            }
                        )
                if "image" not in model.input:
                    content = [c for c in content if c["type"] != "input_image"]
                if not content:
                    msg_index += 1
                    continue
                messages.append({"role": "user", "content": content})

        elif isinstance(msg, AssistantMessage):
            output: list[dict[str, Any]] = []
            is_different_model = msg.model != model.id and msg.provider == model.provider and msg.api == model.api

            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    if block.thinking_signature:
                        try:
                            reasoning_item = json.loads(block.thinking_signature)
                            output.append(reasoning_item)
                        except (ValueError, TypeError):
                            pass
                elif isinstance(block, TextContent):
                    parsed_sig = _parse_text_signature(block.text_signature)
                    msg_id: str
                    if parsed_sig and parsed_sig.get("id"):
                        raw_id = parsed_sig["id"]
                        msg_id = f"msg_{short_hash(raw_id)}" if len(raw_id) > 64 else raw_id
                    else:
                        msg_id = f"msg_{msg_index}"

                    item: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": sanitize_surrogates(block.text),
                                "annotations": [],
                            }
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    if parsed_sig and parsed_sig.get("phase"):
                        item["phase"] = parsed_sig["phase"]
                    output.append(item)

                elif isinstance(block, ToolCall):
                    call_id, *rest = block.id.split("|", 1)
                    item_id: str | None = rest[0] if rest else None

                    # For different-model messages, omit id to avoid pairing validation.
                    if is_different_model and item_id and item_id.startswith("fc_"):
                        item_id = None

                    fc: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    }
                    if item_id is not None:
                        fc["id"] = item_id
                    output.append(fc)

            if output:
                messages.extend(output)

        elif isinstance(msg, ToolResultMessage):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            text_result = "\n".join(text_parts)
            has_images = any(isinstance(c, ImageContent) for c in msg.content)
            has_text = bool(text_result)
            call_id = msg.tool_call_id.split("|", 1)[0]

            if has_images and "image" in model.input:
                content_parts: list[dict[str, Any]] = []
                if has_text:
                    content_parts.append({"type": "input_text", "text": sanitize_surrogates(text_result)})
                for block in msg.content:
                    if isinstance(block, ImageContent):
                        content_parts.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{block.mime_type};base64,{block.data}",
                            }
                        )
                output_val: Any = content_parts
            else:
                output_val = sanitize_surrogates(text_result if has_text else "(see attached image)")

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_val,
                }
            )

        msg_index += 1

    return messages


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def convert_responses_tools(tools: list[Tool], *, strict: bool = False) -> list[dict[str, Any]]:
    """Convert nu_ai :class:`Tool` list to Responses API ``tools`` payload."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": strict,
        }
        for tool in tools
    ]


# ---------------------------------------------------------------------------
# Stop-reason mapping
# ---------------------------------------------------------------------------


def map_stop_reason(status: str | None) -> StopReason:
    """Map a Responses API ``response.status`` to nu_ai :data:`StopReason`."""
    if status is None or status in ("completed", "in_progress", "queued"):
        return "stop"
    if status == "incomplete":
        return "length"
    if status in ("failed", "cancelled"):
        return "error"
    return "stop"


# ---------------------------------------------------------------------------
# Params builder
# ---------------------------------------------------------------------------


def build_params(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
) -> dict[str, Any]:
    """Build the JSON payload for an OpenAI Responses API streaming request."""
    input_messages = convert_responses_messages(model, context)

    # Cache retention
    cache_retention = "short"
    if options is not None and options.cache_retention is not None:
        cache_retention = options.cache_retention
    else:
        import os  # noqa: PLC0415

        if os.environ.get("PI_CACHE_RETENTION") == "long":
            cache_retention = "long"

    # Prompt cache retention (24h only for direct api.openai.com calls)
    prompt_cache_retention: str | None = None
    if cache_retention == "long" and "api.openai.com" in model.base_url:
        prompt_cache_retention = "24h"

    params: dict[str, Any] = {
        "model": model.id,
        "input": input_messages,
        "stream": True,
        "store": False,
    }

    session_id = options.session_id if options is not None else None
    if cache_retention != "none" and session_id:
        params["prompt_cache_key"] = session_id
    if prompt_cache_retention:
        params["prompt_cache_retention"] = prompt_cache_retention

    if options is not None and options.max_tokens is not None:
        params["max_output_tokens"] = options.max_tokens

    if options is not None and options.temperature is not None:
        params["temperature"] = options.temperature

    if options is not None:
        service_tier = getattr(options, "service_tier", None)
        if service_tier is not None:
            params["service_tier"] = service_tier

    if context.tools:
        params["tools"] = convert_responses_tools(context.tools)

    # Reasoning params
    if model.reasoning:
        reasoning_effort = getattr(options, "reasoning_effort", None) if options is not None else None
        reasoning_summary = getattr(options, "reasoning_summary", None) if options is not None else None
        if reasoning_effort or reasoning_summary:
            params["reasoning"] = {
                "effort": reasoning_effort or "medium",
                "summary": reasoning_summary or "auto",
            }
            params["include"] = ["reasoning.encrypted_content"]
        elif model.provider != "github-copilot":
            params["reasoning"] = {"effort": "none"}

    return params


# ---------------------------------------------------------------------------
# Service tier pricing
# ---------------------------------------------------------------------------


def _get_service_tier_multiplier(service_tier: str | None) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.0
    return 1.0


def _apply_service_tier_pricing(usage: Usage, service_tier: str | None) -> None:
    multiplier = _get_service_tier_multiplier(service_tier)
    if multiplier == 1.0:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write


# ---------------------------------------------------------------------------
# Stream processing
# ---------------------------------------------------------------------------


def _empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


def _new_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_empty_usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def _copy_output(output: AssistantMessage) -> AssistantMessage:
    return output.model_copy(deep=True)


def _get(obj: Any, key: str) -> Any:
    """Duck-typed attribute accessor — works for both SDK objects and dicts."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


async def process_responses_stream(
    openai_stream: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    *,
    service_tier: str | None = None,
) -> None:
    """Consume an OpenAI Responses API async event stream and emit stream events.

    Mirrors ``processResponsesStream`` from ``openai-responses-shared.ts``.
    ``openai_stream`` must be an async iterable yielding event dicts or SDK
    objects with a ``.type`` attribute.
    """
    current_item: dict[str, Any] | None = None
    current_block: TextContent | ThinkingContent | ToolCall | None = None
    partial_json: str = ""

    def block_index() -> int:
        return len(output.content) - 1

    async for raw_event in openai_stream:
        event = raw_event if isinstance(raw_event, dict) else _sdk_event_to_dict(raw_event)
        event_type = event.get("type", "")

        if event_type == "response.created":
            response = event.get("response") or {}
            rid = response.get("id") if isinstance(response, dict) else _get(response, "id")
            if rid:
                output.response_id = rid

        elif event_type == "response.output_item.added":
            item = event.get("item") or {}
            item_type = item.get("type") if isinstance(item, dict) else _get(item, "type")
            if item_type == "reasoning":
                current_item = dict(item) if not isinstance(item, dict) else item
                current_block = ThinkingContent(thinking="")
                output.content.append(current_block)
                stream.push(
                    ThinkingStartEvent(
                        content_index=block_index(),
                        partial=_copy_output(output),
                    )
                )
            elif item_type == "message":
                current_item = dict(item)
                current_item.setdefault("content", [])
                current_block = TextContent(text="")
                output.content.append(current_block)
                stream.push(
                    TextStartEvent(
                        content_index=block_index(),
                        partial=_copy_output(output),
                    )
                )
            elif item_type == "function_call":
                call_id = (item.get("call_id") or "") if isinstance(item, dict) else (_get(item, "call_id") or "")
                item_id = (item.get("id") or "") if isinstance(item, dict) else (_get(item, "id") or "")
                name = (item.get("name") or "") if isinstance(item, dict) else (_get(item, "name") or "")
                init_args = (item.get("arguments") or "") if isinstance(item, dict) else (_get(item, "arguments") or "")
                current_item = {"type": "function_call", "call_id": call_id, "id": item_id, "name": name}
                current_block = ToolCall(id=f"{call_id}|{item_id}", name=name, arguments={})
                partial_json = init_args
                output.content.append(current_block)
                stream.push(
                    ToolCallStartEvent(
                        content_index=block_index(),
                        partial=_copy_output(output),
                    )
                )

        elif event_type == "response.reasoning_summary_part.added":
            if current_item and current_item.get("type") == "reasoning":
                current_item.setdefault("summary", [])
                part = event.get("part") or {}
                current_item["summary"].append(dict(part) if not isinstance(part, dict) else part)

        elif event_type == "response.reasoning_summary_text.delta":
            if current_item and current_item.get("type") == "reasoning" and isinstance(current_block, ThinkingContent):
                current_item.setdefault("summary", [])
                summary = current_item["summary"]
                if summary:
                    delta = event.get("delta", "") or ""
                    current_block.thinking += delta
                    summary[-1]["text"] = summary[-1].get("text", "") + delta
                    stream.push(
                        ThinkingDeltaEvent(
                            content_index=block_index(),
                            delta=delta,
                            partial=_copy_output(output),
                        )
                    )

        elif event_type == "response.reasoning_summary_part.done":
            if current_item and current_item.get("type") == "reasoning" and isinstance(current_block, ThinkingContent):
                current_item.setdefault("summary", [])
                summary = current_item["summary"]
                if summary:
                    current_block.thinking += "\n\n"
                    summary[-1]["text"] = summary[-1].get("text", "") + "\n\n"
                    stream.push(
                        ThinkingDeltaEvent(
                            content_index=block_index(),
                            delta="\n\n",
                            partial=_copy_output(output),
                        )
                    )

        elif event_type == "response.content_part.added":
            if current_item and current_item.get("type") == "message":
                part = event.get("part") or {}
                part_type = part.get("type") if isinstance(part, dict) else _get(part, "type")
                if part_type in ("output_text", "refusal"):
                    current_item.setdefault("content", [])
                    current_item["content"].append(dict(part) if not isinstance(part, dict) else part)

        elif event_type == "response.output_text.delta":
            if current_item and current_item.get("type") == "message" and isinstance(current_block, TextContent):
                content_list = current_item.get("content", [])
                if not content_list:
                    continue
                last_part = content_list[-1]
                last_type = last_part.get("type") if isinstance(last_part, dict) else _get(last_part, "type")
                if last_type == "output_text":
                    delta = event.get("delta", "") or ""
                    current_block.text += delta
                    last_part["text"] = last_part.get("text", "") + delta
                    stream.push(
                        TextDeltaEvent(
                            content_index=block_index(),
                            delta=delta,
                            partial=_copy_output(output),
                        )
                    )

        elif event_type == "response.refusal.delta":
            if current_item and current_item.get("type") == "message" and isinstance(current_block, TextContent):
                content_list = current_item.get("content", [])
                if not content_list:
                    continue
                last_part = content_list[-1]
                last_type = last_part.get("type") if isinstance(last_part, dict) else _get(last_part, "type")
                if last_type == "refusal":
                    delta = event.get("delta", "") or ""
                    current_block.text += delta
                    last_part["refusal"] = last_part.get("refusal", "") + delta
                    stream.push(
                        TextDeltaEvent(
                            content_index=block_index(),
                            delta=delta,
                            partial=_copy_output(output),
                        )
                    )

        elif event_type == "response.function_call_arguments.delta":
            if current_item and current_item.get("type") == "function_call" and isinstance(current_block, ToolCall):
                delta = event.get("delta", "") or ""
                partial_json += delta
                current_block.arguments = parse_streaming_json(partial_json) or {}
                stream.push(
                    ToolCallDeltaEvent(
                        content_index=block_index(),
                        delta=delta,
                        partial=_copy_output(output),
                    )
                )

        elif event_type == "response.function_call_arguments.done":
            if current_item and current_item.get("type") == "function_call" and isinstance(current_block, ToolCall):
                prev_partial = partial_json
                done_args: str = event.get("arguments", "") or ""
                partial_json = done_args
                current_block.arguments = parse_streaming_json(partial_json) or {}

                # Emit any tail delta not yet pushed
                if done_args.startswith(prev_partial):
                    tail = done_args[len(prev_partial) :]
                    if tail:
                        stream.push(
                            ToolCallDeltaEvent(
                                content_index=block_index(),
                                delta=tail,
                                partial=_copy_output(output),
                            )
                        )

        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            item_type = item.get("type") if isinstance(item, dict) else _get(item, "type")

            if item_type == "reasoning" and isinstance(current_block, ThinkingContent):
                summary = (item.get("summary") or []) if isinstance(item, dict) else (_get(item, "summary") or [])
                current_block.thinking = "\n\n".join(
                    (s.get("text") or "" if isinstance(s, dict) else _get(s, "text") or "") for s in summary
                )
                current_block.thinking_signature = json.dumps(
                    item if isinstance(item, dict) else _sdk_event_to_dict(item)
                )
                stream.push(
                    ThinkingEndEvent(
                        content_index=block_index(),
                        content=current_block.thinking,
                        partial=_copy_output(output),
                    )
                )
                current_block = None

            elif item_type == "message" and isinstance(current_block, TextContent):
                content_list = (item.get("content") or []) if isinstance(item, dict) else (_get(item, "content") or [])
                current_block.text = "".join(
                    (
                        (c.get("text") or "" if c.get("type") == "output_text" else c.get("refusal") or "")
                        if isinstance(c, dict)
                        else (_get(c, "text") or "" if _get(c, "type") == "output_text" else _get(c, "refusal") or "")
                    )
                    for c in content_list
                )
                msg_id = (item.get("id") or "") if isinstance(item, dict) else (_get(item, "id") or "")
                phase = (item.get("phase") or None) if isinstance(item, dict) else (_get(item, "phase") or None)
                current_block.text_signature = _encode_text_signature_v1(msg_id, phase)
                stream.push(
                    TextEndEvent(
                        content_index=block_index(),
                        content=current_block.text,
                        partial=_copy_output(output),
                    )
                )
                current_block = None

            elif item_type == "function_call":
                call_id = (item.get("call_id") or "") if isinstance(item, dict) else (_get(item, "call_id") or "")
                item_id_str = (item.get("id") or "") if isinstance(item, dict) else (_get(item, "id") or "")
                raw_args = (
                    (item.get("arguments") or "{}") if isinstance(item, dict) else (_get(item, "arguments") or "{}")
                )

                if isinstance(current_block, ToolCall) and partial_json:
                    args = parse_streaming_json(partial_json) or {}
                else:
                    args = parse_streaming_json(raw_args) or {}

                finalized = ToolCall(
                    id=f"{call_id}|{item_id_str}",
                    name=(item.get("name") or "") if isinstance(item, dict) else (_get(item, "name") or ""),
                    arguments=args,
                )
                current_block = None
                partial_json = ""
                stream.push(
                    ToolCallEndEvent(
                        content_index=block_index(),
                        tool_call=finalized,
                        partial=_copy_output(output),
                    )
                )

        elif event_type == "response.completed":
            response = event.get("response") or {}
            resp_id = response.get("id") if isinstance(response, dict) else _get(response, "id")
            if resp_id:
                output.response_id = resp_id

            usage_raw = response.get("usage") if isinstance(response, dict) else _get(response, "usage")
            if usage_raw is not None:
                if not isinstance(usage_raw, dict):
                    usage_raw = _to_dict(usage_raw) or {}
                input_tokens_details = usage_raw.get("input_tokens_details") or {}
                cached_tokens = (
                    input_tokens_details.get("cached_tokens") if isinstance(input_tokens_details, dict) else 0
                ) or 0
                output.usage = Usage(
                    # OpenAI includes cached tokens in input_tokens, subtract to get non-cached
                    input=max(0, (usage_raw.get("input_tokens") or 0) - cached_tokens),
                    output=usage_raw.get("output_tokens") or 0,
                    cache_read=cached_tokens,
                    cache_write=0,
                    total_tokens=usage_raw.get("total_tokens") or 0,
                    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                )
            calculate_cost(model, output.usage)

            resolved_service_tier = (
                response.get("service_tier") if isinstance(response, dict) else _get(response, "service_tier")
            ) or service_tier
            _apply_service_tier_pricing(output.usage, resolved_service_tier)

            status = response.get("status") if isinstance(response, dict) else _get(response, "status")
            output.stop_reason = map_stop_reason(status)
            if any(isinstance(b, ToolCall) for b in output.content) and output.stop_reason == "stop":
                output.stop_reason = "toolUse"

        elif event_type == "error":
            code = event.get("code", "unknown")
            message = event.get("message", "Unknown error")
            raise RuntimeError(f"Error Code {code}: {message}")

        elif event_type == "response.failed":
            response = event.get("response") or {}
            error = response.get("error") if isinstance(response, dict) else _get(response, "error")
            details = (
                response.get("incomplete_details")
                if isinstance(response, dict)
                else _get(response, "incomplete_details")
            )
            if error:
                err_code = (
                    (error.get("code") or "unknown") if isinstance(error, dict) else _get(error, "code") or "unknown"
                )
                err_msg = (
                    (error.get("message") or "no message")
                    if isinstance(error, dict)
                    else _get(error, "message") or "no message"
                )
                raise RuntimeError(f"{err_code}: {err_msg}")
            elif details:
                reason = (details.get("reason") or "") if isinstance(details, dict) else _get(details, "reason") or ""
                raise RuntimeError(f"incomplete: {reason}")
            else:
                raise RuntimeError("Unknown error (no error details in response)")


def _to_dict(obj: Any) -> dict[str, Any] | None:
    """Coerce an SDK object to a plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return None


def _sdk_event_to_dict(event: Any) -> dict[str, Any]:
    """Convert an SDK ResponseStreamEvent object to a plain dict."""
    d = _to_dict(event)
    return d if d is not None else {}


# ---------------------------------------------------------------------------
# SDK glue
# ---------------------------------------------------------------------------


def create_client(
    model: Model,
    context: Context,
    api_key: str | None = None,
    options_headers: dict[str, str] | None = None,
) -> Any:
    """Construct an :class:`openai.AsyncOpenAI` client configured for ``model``."""
    from openai import AsyncOpenAI  # noqa: PLC0415 — lazy import

    if not api_key:
        import os  # noqa: PLC0415

        env_key = os.environ.get("OPENAI_API_KEY")
        if not env_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument.",
            )
        api_key = env_key

    headers: dict[str, str] = dict(model.headers or {})
    if model.provider == "github-copilot":
        has_images = has_copilot_vision_input(context.messages)
        copilot_headers = build_copilot_dynamic_headers(
            messages=context.messages,
            has_images=has_images,
        )
        headers.update(copilot_headers)
    if options_headers:
        headers.update(options_headers)

    return AsyncOpenAI(
        api_key=api_key,
        base_url=model.base_url or None,
        default_headers=headers or None,
    )


# Hold strong references to in-flight background tasks so the event loop
# does not GC them while they are still running.
_background_tasks: set[asyncio.Task[None]] = set()


async def _run_openai_responses_stream(
    *,
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None,
    stream: AssistantMessageEventStream,
    output: AssistantMessage,
    client: Any | None,
) -> None:
    try:
        resolved_client: Any
        if client is not None:
            resolved_client = client
        else:
            api_key = (
                (options.api_key if options and options.api_key else None) or get_env_api_key(model.provider) or None
            )
            resolved_client = create_client(
                model,
                context,
                api_key=api_key,
                options_headers=options.headers if options else None,
            )

        params = build_params(model, context, options)
        stream.push(StartEvent(partial=_copy_output(output)))

        openai_stream = await resolved_client.responses.create(**params)
        service_tier = getattr(options, "service_tier", None) if options is not None else None

        await process_responses_stream(
            openai_stream,
            output,
            stream,
            model,
            service_tier=service_tier,
        )

        if output.stop_reason in ("aborted", "error"):
            raise RuntimeError(output.error_message or "An unknown error occurred")

        stream.push(DoneEvent(reason=output.stop_reason, message=_copy_output(output)))  # type: ignore[arg-type]
        stream.end()
    except asyncio.CancelledError:
        output.stop_reason = "aborted"
        output.error_message = "Request was aborted"
        stream.push(ErrorEvent(reason="aborted", error=_copy_output(output)))
        stream.end()
        raise
    except Exception as exc:
        output.stop_reason = "error"
        output.error_message = str(exc)
        stream.push(ErrorEvent(reason="error", error=_copy_output(output)))
        stream.end()


def stream_openai_responses(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the OpenAI Responses API (``/v1/responses``)."""
    stream = AssistantMessageEventStream()
    output = _new_output(model)
    task = asyncio.create_task(
        _run_openai_responses_stream(
            model=model,
            context=context,
            options=options,
            stream=stream,
            output=output,
            client=client,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return stream


def stream_simple_openai_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Simple-options variant: maps :class:`SimpleStreamOptions` into Responses API options.

    Lowers the unified ``reasoning`` level into ``reasoning_effort``, clamping
    ``xhigh`` down to ``high`` on models that don't support it.
    """
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider) or None
    base = build_base_options(model, options, api_key)

    reasoning_effort: ThinkingLevel | None = None
    if options is not None and options.reasoning is not None:
        reasoning_effort = options.reasoning if supports_xhigh(model) else clamp_reasoning(options.reasoning)

    merged = OpenAIResponsesOptions(
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        api_key=base.api_key,
        cache_retention=base.cache_retention,
        session_id=base.session_id,
        headers=base.headers,
        max_retry_delay_ms=base.max_retry_delay_ms,
        metadata=base.metadata,
        reasoning_effort=reasoning_effort,
    )
    return stream_openai_responses(model, context, merged, client=client)


__all__ = [
    "build_params",
    "convert_responses_messages",
    "convert_responses_tools",
    "create_client",
    "map_stop_reason",
    "process_responses_stream",
    "stream_openai_responses",
    "stream_simple_openai_responses",
]
