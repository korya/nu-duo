"""Mistral provider.

Direct port of ``packages/ai/src/providers/mistral.ts``. Uses the
``mistralai`` Python SDK (``Mistral`` client).

Two logical sections:

1. **Pure transforms** — ``to_function_tools``, ``to_chat_messages``,
   ``build_tool_result_text``, ``map_tool_choice``, ``map_stop_reason``,
   ``build_chat_payload``, ``derive_mistral_tool_call_id``,
   ``create_mistral_tool_call_id_normalizer``. All no-SDK so they can be
   unit-tested in isolation.
2. **SDK glue** — ``stream_mistral``, ``stream_simple_mistral`` built on the
   ``mistralai`` SDK.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Literal

from nu_ai.env_api_keys import get_env_api_key
from nu_ai.models import calculate_cost
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
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from nu_ai.utils.event_stream import AssistantMessageEventStream
from nu_ai.utils.hash import short_hash
from nu_ai.utils.json_parse import parse_streaming_json
from nu_ai.utils.sanitize_unicode import sanitize_surrogates

if TYPE_CHECKING:
    from nu_ai.types import (
        Context,
        Message,
        Model,
        SimpleStreamOptions,
        Tool,
    )

MISTRAL_TOOL_CALL_ID_LENGTH = 9
MAX_MISTRAL_ERROR_BODY_CHARS = 4000


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class MistralOptions(StreamOptions):
    """Mistral-specific extension of :class:`StreamOptions`.

    Mirrors ``MistralOptions`` from
    ``packages/ai/src/providers/mistral.ts``.
    """

    tool_choice: Literal["auto", "none", "any", "required"] | dict[str, Any] | None = None
    prompt_mode: Literal["reasoning"] | None = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def derive_mistral_tool_call_id(tool_call_id: str, attempt: int) -> str:
    """Derive a 9-character alphanumeric Mistral tool-call ID from ``tool_call_id``."""
    normalized = "".join(c for c in tool_call_id if c.isalnum())
    if attempt == 0 and len(normalized) == MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or tool_call_id
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return "".join(c for c in short_hash(seed) if c.isalnum())[:MISTRAL_TOOL_CALL_ID_LENGTH]


def create_mistral_tool_call_id_normalizer() -> Any:
    """Return a stateful normalizer function that maps arbitrary IDs to 9-char IDs.

    Collision-safe: tries successive attempt indices until it finds a unique
    9-char string, mirroring the upstream TS implementation.
    """
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def normalize(tool_call_id: str, *_: object) -> str:
        existing = id_map.get(tool_call_id)
        if existing:
            return existing
        attempt = 0
        while True:
            candidate = derive_mistral_tool_call_id(tool_call_id, attempt)
            owner = reverse_map.get(candidate)
            if owner is None or owner == tool_call_id:
                id_map[tool_call_id] = candidate
                reverse_map[candidate] = tool_call_id
                return candidate
            attempt += 1

    return normalize


def build_tool_result_text(
    text: str,
    has_images: bool,
    supports_images: bool,
    is_error: bool,
) -> str:
    """Build the text content of a tool result, including error/image annotations."""
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""

    if trimmed:
        image_suffix = (
            "\n[tool image omitted: model does not support images]" if has_images and not supports_images else ""
        )
        return f"{error_prefix}{trimmed}{image_suffix}"

    if has_images:
        if supports_images:
            return "[tool error] (see attached image)" if is_error else "(see attached image)"
        return (
            "[tool error] (image omitted: model does not support images)"
            if is_error
            else "(image omitted: model does not support images)"
        )

    return "[tool error] (no tool output)" if is_error else "(no tool output)"


def to_function_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert nu_ai :class:`Tool` list to the Mistral ``tools`` payload."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": False,
            },
        }
        for t in tools
    ]


def map_tool_choice(
    choice: MistralOptions | None,
) -> Any:
    """Convert :attr:`MistralOptions.tool_choice` to the Mistral wire value."""
    raw = choice.tool_choice if choice else None
    if raw is None:
        return None
    if raw in ("auto", "none", "any", "required"):
        return raw
    if isinstance(raw, dict) and raw.get("type") == "function":
        return {"type": "function", "function": {"name": raw["function"]["name"]}}
    return None


def to_chat_messages(
    messages: list[Message],
    supports_images: bool,
) -> list[dict[str, Any]]:
    """Convert nu_ai messages to the Mistral chat format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            # Shouldn't happen (transform_messages already filtered) but be safe
            pass

        from nu_ai.types import ToolResultMessage as ToolResult
        from nu_ai.types import UserMessage

        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": sanitize_surrogates(msg.content)})
                continue

            had_images = any(isinstance(item, ImageContent) for item in msg.content)
            content_chunks: list[dict[str, Any]] = []
            for item in msg.content:
                if isinstance(item, TextContent):
                    content_chunks.append({"type": "text", "text": sanitize_surrogates(item.text)})
                elif isinstance(item, ImageContent) and supports_images:
                    content_chunks.append(
                        {
                            "type": "image_url",
                            "imageUrl": f"data:{item.mime_type};base64,{item.data}",
                        }
                    )
            if content_chunks:
                result.append({"role": "user", "content": content_chunks})
            elif had_images and not supports_images:
                result.append(
                    {
                        "role": "user",
                        "content": "(image omitted: model does not support images)",
                    }
                )
            continue

        if isinstance(msg, AssistantMessage):
            content_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in msg.content:
                if isinstance(block, TextContent):
                    if block.text.strip():
                        content_parts.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(block.text),
                            }
                        )
                elif isinstance(block, ThinkingContent):
                    if block.thinking.strip():
                        content_parts.append(
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": sanitize_surrogates(block.thinking)}],
                            }
                        )
                elif isinstance(block, ToolCall):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments or {}),
                            },
                        }
                    )

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if content_parts:
                assistant_msg["content"] = content_parts
            if tool_calls:
                assistant_msg["toolCalls"] = tool_calls
            if content_parts or tool_calls:
                result.append(assistant_msg)
            continue

        if isinstance(msg, ToolResult):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            text_result = "\n".join(text_parts)
            has_images = any(isinstance(c, ImageContent) for c in msg.content)
            tool_text = build_tool_result_text(text_result, has_images, supports_images, msg.is_error)

            tool_content: list[dict[str, Any]] = [{"type": "text", "text": tool_text}]
            if supports_images:
                for part in msg.content:
                    if isinstance(part, ImageContent):
                        tool_content.append(
                            {
                                "type": "image_url",
                                "imageUrl": f"data:{part.mime_type};base64,{part.data}",
                            }
                        )

            result.append(
                {
                    "role": "tool",
                    "toolCallId": msg.tool_call_id,
                    "name": msg.tool_name,
                    "content": tool_content,
                }
            )

    return result


def map_stop_reason(reason: str | None) -> StopReason:
    """Map a Mistral ``finish_reason`` to nu_ai's :data:`StopReason`."""
    if reason is None:
        return "stop"
    if reason == "stop":
        return "stop"
    if reason in ("length", "model_length"):
        return "length"
    if reason == "tool_calls":
        return "toolUse"
    if reason == "error":
        return "error"
    return "stop"


def build_chat_payload(
    model: Model,
    context: Context,
    messages: list[Message],
    options: MistralOptions | None = None,
) -> dict[str, Any]:
    """Build the Mistral ``chat.stream`` payload."""
    supports_images = "image" in model.input
    chat_messages = to_chat_messages(messages, supports_images)

    if context.system_prompt:
        chat_messages.insert(
            0,
            {
                "role": "system",
                "content": sanitize_surrogates(context.system_prompt),
            },
        )

    payload: dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "messages": chat_messages,
    }

    if context.tools:
        payload["tools"] = to_function_tools(context.tools)
    if options and options.temperature is not None:
        payload["temperature"] = options.temperature
    if options and options.max_tokens is not None:
        payload["maxTokens"] = options.max_tokens
    tool_choice = map_tool_choice(options)
    if tool_choice is not None:
        payload["toolChoice"] = tool_choice
    if options and options.prompt_mode:
        payload["promptMode"] = options.prompt_mode

    return payload


def format_mistral_error(error: Exception) -> str:
    """Format a Mistral SDK error into a concise human-readable string."""
    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    body_text = body.strip() if isinstance(body, str) else None
    if isinstance(status_code, int) and body_text:
        truncated = (
            body_text
            if len(body_text) <= MAX_MISTRAL_ERROR_BODY_CHARS
            else f"{body_text[:MAX_MISTRAL_ERROR_BODY_CHARS]}... [truncated {len(body_text) - MAX_MISTRAL_ERROR_BODY_CHARS} chars]"
        )
        return f"Mistral API error ({status_code}): {truncated}"
    if isinstance(status_code, int):
        return f"Mistral API error ({status_code}): {error}"
    return str(error)


# ---------------------------------------------------------------------------
# SDK glue
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


# Hold strong references so GC doesn't kill in-flight background tasks.
_background_tasks: set[asyncio.Task[None]] = set()


async def _run_mistral_stream(
    *,
    model: Model,
    context: Context,
    options: MistralOptions | None,
    stream: AssistantMessageEventStream,
    output: AssistantMessage,
    client: Any | None,
) -> None:
    """Async coroutine driving the Mistral ``chat.stream`` call."""
    try:
        if client is None:
            from mistralai import Mistral

            api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
            if not api_key:
                raise ValueError(f"No API key for provider: {model.provider}")

            resolved_client = Mistral(
                api_key=api_key,
                server_url=model.base_url or None,
            )
        else:
            resolved_client = client

        normalizer = create_mistral_tool_call_id_normalizer()
        transformed = transform_messages(context.messages, model, normalizer)
        payload = build_chat_payload(model, context, transformed, options)

        stream.push(StartEvent(partial=_copy_output(output)))

        mistral_stream = await resolved_client.chat.stream_async(**payload)

        # Per-block tracking
        current_block: TextContent | ThinkingContent | None = None
        tool_blocks_by_key: dict[str, int] = {}  # "callId:index" -> content list index
        tool_partial_args: dict[int, str] = {}  # content list index -> partial JSON

        def finish_current_block() -> None:
            nonlocal current_block
            if current_block is None:
                return
            idx = len(output.content) - 1
            if isinstance(current_block, TextContent):
                stream.push(
                    TextEndEvent(
                        content_index=idx,
                        content=current_block.text,
                        partial=_copy_output(output),
                    )
                )
            elif isinstance(current_block, ThinkingContent):
                stream.push(
                    ThinkingEndEvent(
                        content_index=idx,
                        content=current_block.thinking,
                        partial=_copy_output(output),
                    )
                )
            current_block = None

        async for event in mistral_stream:
            chunk = event.data
            # Capture stable response ID from first non-empty chunk id.
            chunk_id = getattr(chunk, "id", None)
            if chunk_id and output.response_id is None:
                output.response_id = chunk_id

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                output.usage.input = getattr(usage, "prompt_tokens", None) or 0
                output.usage.output = getattr(usage, "completion_tokens", None) or 0
                output.usage.cache_read = 0
                output.usage.cache_write = 0
                total = getattr(usage, "total_tokens", None)
                output.usage.total_tokens = total if total is not None else output.usage.input + output.usage.output
                calculate_cost(model, output.usage)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]

            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason:
                output.stop_reason = map_stop_reason(finish_reason)

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            # --- Text / thinking content ---
            delta_content = getattr(delta, "content", None)
            if delta_content is not None:
                items = [delta_content] if isinstance(delta_content, str) else (delta_content or [])
                for item in items:
                    if isinstance(item, str):
                        text_delta = sanitize_surrogates(item)
                        if not isinstance(current_block, TextContent):
                            finish_current_block()
                            current_block = TextContent(text="")
                            output.content.append(current_block)
                            stream.push(
                                TextStartEvent(
                                    content_index=len(output.content) - 1,
                                    partial=_copy_output(output),
                                )
                            )
                        current_block.text += text_delta
                        stream.push(
                            TextDeltaEvent(
                                content_index=len(output.content) - 1,
                                delta=text_delta,
                                partial=_copy_output(output),
                            )
                        )
                        continue

                    item_type = getattr(item, "type", None)

                    if item_type == "thinking":
                        thinking_parts = getattr(item, "thinking", []) or []
                        delta_text = "".join(getattr(p, "text", "") for p in thinking_parts if hasattr(p, "text"))
                        thinking_delta = sanitize_surrogates(delta_text)
                        if not thinking_delta:
                            continue
                        if not isinstance(current_block, ThinkingContent):
                            finish_current_block()
                            current_block = ThinkingContent(thinking="")
                            output.content.append(current_block)
                            stream.push(
                                ThinkingStartEvent(
                                    content_index=len(output.content) - 1,
                                    partial=_copy_output(output),
                                )
                            )
                        current_block.thinking += thinking_delta
                        stream.push(
                            ThinkingDeltaEvent(
                                content_index=len(output.content) - 1,
                                delta=thinking_delta,
                                partial=_copy_output(output),
                            )
                        )
                        continue

                    if item_type == "text":
                        raw_text = getattr(item, "text", "") or ""
                        text_delta = sanitize_surrogates(raw_text)
                        if not isinstance(current_block, TextContent):
                            finish_current_block()
                            current_block = TextContent(text="")
                            output.content.append(current_block)
                            stream.push(
                                TextStartEvent(
                                    content_index=len(output.content) - 1,
                                    partial=_copy_output(output),
                                )
                            )
                        current_block.text += text_delta
                        stream.push(
                            TextDeltaEvent(
                                content_index=len(output.content) - 1,
                                delta=text_delta,
                                partial=_copy_output(output),
                            )
                        )

            # --- Tool calls ---
            tool_calls_delta = getattr(delta, "tool_calls", None) or []
            for tc in tool_calls_delta:
                if current_block is not None:
                    finish_current_block()

                tc_id_raw = getattr(tc, "id", None)
                tc_id = (
                    tc_id_raw
                    if tc_id_raw and tc_id_raw != "null"
                    else derive_mistral_tool_call_id(f"toolcall:{getattr(tc, 'index', 0) or 0}", 0)
                )
                tc_index = getattr(tc, "index", 0) or 0
                key = f"{tc_id}:{tc_index}"

                existing_idx = tool_blocks_by_key.get(key)
                block: ToolCall | None = None

                if existing_idx is not None:
                    candidate = output.content[existing_idx]
                    if isinstance(candidate, ToolCall):
                        block = candidate

                if block is None:
                    tc_func = getattr(tc, "function", None)
                    tc_name = getattr(tc_func, "name", "") or ""
                    block = ToolCall(id=tc_id, name=tc_name, arguments={})
                    output.content.append(block)
                    new_idx = len(output.content) - 1
                    tool_blocks_by_key[key] = new_idx
                    tool_partial_args[new_idx] = ""
                    stream.push(
                        ToolCallStartEvent(
                            content_index=new_idx,
                            partial=_copy_output(output),
                        )
                    )
                    existing_idx = new_idx

                assert existing_idx is not None
                tc_func = getattr(tc, "function", None)
                raw_args = getattr(tc_func, "arguments", None)
                if isinstance(raw_args, str):
                    args_delta = raw_args
                elif raw_args is not None:
                    args_delta = json.dumps(raw_args)
                else:
                    args_delta = ""

                tool_partial_args[existing_idx] = tool_partial_args.get(existing_idx, "") + args_delta
                block.arguments = parse_streaming_json(tool_partial_args[existing_idx]) or {}

                stream.push(
                    ToolCallDeltaEvent(
                        content_index=existing_idx,
                        delta=args_delta,
                        partial=_copy_output(output),
                    )
                )

        # Finalize current text/thinking block
        finish_current_block()

        # Finalize all tool call blocks
        for key, idx in tool_blocks_by_key.items():
            blk = output.content[idx]
            if not isinstance(blk, ToolCall):
                continue
            blk.arguments = parse_streaming_json(tool_partial_args.get(idx, "")) or {}
            stream.push(
                ToolCallEndEvent(
                    content_index=idx,
                    tool_call=blk.model_copy(deep=True),
                    partial=_copy_output(output),
                )
            )

        if output.stop_reason in ("aborted", "error"):
            raise RuntimeError("An unknown error occurred")

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
        output.error_message = format_mistral_error(exc)
        stream.push(ErrorEvent(reason="error", error=_copy_output(output)))
        stream.end()


def stream_mistral(
    model: Model,
    context: Context,
    options: MistralOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from Mistral via ``chat.stream``."""
    stream = AssistantMessageEventStream()
    output = _new_output(model)
    task = asyncio.create_task(
        _run_mistral_stream(
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


def stream_simple_mistral(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Simple-options variant: maps :class:`SimpleStreamOptions` to :class:`MistralOptions`.

    Sets ``prompt_mode="reasoning"`` on reasoning-capable models when a thinking
    level is requested.
    """
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)
    reasoning = clamp_reasoning(options.reasoning if options else None)
    prompt_mode: Literal["reasoning"] | None = "reasoning" if (model.reasoning and reasoning) else None

    return stream_mistral(
        model,
        context,
        MistralOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            max_retry_delay_ms=base.max_retry_delay_ms,
            metadata=base.metadata,
            prompt_mode=prompt_mode,
        ),
        client=client,
    )


__all__ = [
    "MistralOptions",
    "build_chat_payload",
    "build_tool_result_text",
    "create_mistral_tool_call_id_normalizer",
    "derive_mistral_tool_call_id",
    "format_mistral_error",
    "map_stop_reason",
    "map_tool_choice",
    "stream_mistral",
    "stream_simple_mistral",
    "to_chat_messages",
    "to_function_tools",
]
