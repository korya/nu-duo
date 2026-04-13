"""Amazon Bedrock Converse API provider.

Direct port of ``packages/ai/src/providers/amazon-bedrock.ts``. Uses the
``boto3`` ``bedrock-runtime`` client's ``converse_stream`` API.

Two logical sections:

1. **Pure transforms** — ``normalize_tool_call_id``, ``build_system_prompt``,
   ``convert_messages``, ``convert_tool_config``, ``map_stop_reason``,
   ``build_additional_model_request_fields``, ``create_image_block``.
   All no-SDK so they can be unit-tested in isolation.
2. **SDK glue** — ``create_bedrock_client``, ``stream_bedrock``,
   ``stream_simple_bedrock`` built on the ``boto3`` SDK.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import TYPE_CHECKING, Any, Literal

from nu_ai.models import calculate_cost
from nu_ai.providers.simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_reasoning,
)
from nu_ai.providers.transform_messages import transform_messages
from nu_ai.types import (
    AssistantMessage,
    CacheRetention,
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
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
)
from nu_ai.utils.event_stream import AssistantMessageEventStream
from nu_ai.utils.json_parse import parse_streaming_json
from nu_ai.utils.sanitize_unicode import sanitize_surrogates

if TYPE_CHECKING:
    from nu_ai.types import (
        Context,
        Model,
        SimpleStreamOptions,
        ThinkingLevel,
        Tool,
    )


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

type _ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]


class BedrockOptions(StreamOptions):
    """Bedrock-specific extension of :class:`StreamOptions`.

    Mirrors ``BedrockOptions`` from
    ``packages/ai/src/providers/amazon-bedrock.ts``.
    """

    region: str | None = None
    profile: str | None = None
    tool_choice: Literal["auto", "any", "none"] | dict[str, str] | None = None
    """``None`` = auto, ``"none"`` = no tools, ``{"type": "tool", "name": ...}`` = force."""
    reasoning: _ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = None
    interleaved_thinking: bool | None = None
    request_metadata: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

_ILLEGAL_TOOL_CALL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_tool_call_id(tool_call_id: str, *_: object) -> str:
    """Sanitize a tool call ID to match Bedrock's requirements."""
    sanitized = _ILLEGAL_TOOL_CALL_ID_CHARS.sub("_", tool_call_id)
    return sanitized[:64]


def supports_adaptive_thinking(model_id: str) -> bool:
    """Return ``True`` for Opus 4.6 and Sonnet 4.6 which support adaptive thinking."""
    return "opus-4-6" in model_id or "opus-4.6" in model_id or "sonnet-4-6" in model_id or "sonnet-4.6" in model_id


def supports_prompt_caching(model: Model) -> bool:
    """Return ``True`` for models that support explicit cache points.

    Supported: Claude 3.5 Haiku, Claude 3.7 Sonnet, Claude 4.x.
    Falls back to ``AWS_BEDROCK_FORCE_CACHE=1`` for application inference profiles.
    """
    model_id = model.id.lower()
    if "claude" not in model_id:
        return os.environ.get("AWS_BEDROCK_FORCE_CACHE") == "1"
    if "-4-" in model_id or "-4." in model_id:
        return True
    if "claude-3-7-sonnet" in model_id:
        return True
    return "claude-3-5-haiku" in model_id


def supports_thinking_signature(model: Model) -> bool:
    """Return ``True`` for models that accept the thinking signature field."""
    model_id = model.id.lower()
    return "anthropic.claude" in model_id or "anthropic/claude" in model_id


def resolve_cache_retention(cache_retention: CacheRetention | None) -> CacheRetention:
    """Resolve cache retention — defaults to short unless ``PI_CACHE_RETENTION=long``."""
    if cache_retention:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def map_thinking_level_to_effort(
    level: ThinkingLevel | None,
    model_id: str,
) -> str:
    """Map a nu_ai :data:`ThinkingLevel` to a Bedrock ``effort`` string."""
    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    if level == "high":
        return "high"
    if level == "xhigh":
        return "max" if ("opus-4-6" in model_id or "opus-4.6" in model_id) else "high"
    return "high"


def map_stop_reason(reason: str | None) -> StopReason:
    """Map a Bedrock stop reason to nu_ai's :data:`StopReason`."""
    if reason in ("end_turn", "stop_sequence"):
        return "stop"
    if reason in ("max_tokens", "model_context_window_exceeded"):
        return "length"
    if reason == "tool_use":
        return "toolUse"
    return "error"


def create_image_block(mime_type: str, data: str) -> dict[str, Any]:
    """Create a Bedrock image block from base64-encoded image data."""
    fmt_map = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    fmt = fmt_map.get(mime_type)
    if fmt is None:
        raise ValueError(f"Unknown image type: {mime_type}")
    import base64

    raw_bytes = base64.b64decode(data)
    return {"source": {"bytes": raw_bytes}, "format": fmt}


def build_system_prompt(
    system_prompt: str | None,
    model: Model,
    cache_retention: CacheRetention,
) -> list[dict[str, Any]] | None:
    """Build the Bedrock system prompt blocks, optionally with a cache point."""
    if not system_prompt:
        return None
    blocks: list[dict[str, Any]] = [{"text": sanitize_surrogates(system_prompt)}]
    if cache_retention != "none" and supports_prompt_caching(model):
        cache_point: dict[str, Any] = {"cachePoint": {"type": "default"}}
        if cache_retention == "long":
            cache_point["cachePoint"]["ttl"] = {"value": 3600, "unit": "seconds"}
        blocks.append(cache_point)
    return blocks


def convert_messages(
    context: Context,
    model: Model,
    cache_retention: CacheRetention,
) -> list[dict[str, Any]]:
    """Convert nu_ai context messages to the Bedrock Converse API format."""
    result: list[dict[str, Any]] = []
    transformed = transform_messages(
        context.messages,
        model,
        lambda tool_call_id, *_: normalize_tool_call_id(tool_call_id),
    )

    i = 0
    while i < len(transformed):
        msg = transformed[i]

        if isinstance(msg, ToolResultMessage):
            # Collect all consecutive tool results into a single user message
            tool_results: list[dict[str, Any]] = []

            while i < len(transformed) and isinstance(transformed[i], ToolResultMessage):
                tr = transformed[i]
                assert isinstance(tr, ToolResultMessage)
                content_blocks: list[dict[str, Any]] = []
                for c in tr.content:
                    if isinstance(c, ImageContent):
                        content_blocks.append({"image": create_image_block(c.mime_type, c.data)})
                    else:
                        assert isinstance(c, TextContent)
                        content_blocks.append({"text": sanitize_surrogates(c.text)})
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tr.tool_call_id,
                            "content": content_blocks,
                            "status": "error" if tr.is_error else "success",
                        }
                    }
                )
                i += 1

            result.append({"role": "user", "content": tool_results})
            continue

        if isinstance(msg, AssistantMessage):
            # Skip assistant messages with empty content
            if not msg.content:
                i += 1
                continue

            content_blocks: list[dict[str, Any]] = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    text = c.text.strip()
                    if not text:
                        continue
                    content_blocks.append({"text": sanitize_surrogates(c.text)})
                elif isinstance(c, ToolCall):
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": c.id,
                                "name": c.name,
                                "input": c.arguments,
                            }
                        }
                    )
                elif isinstance(c, ThinkingContent):
                    thinking = c.thinking.strip()
                    if not thinking:
                        continue
                    if supports_thinking_signature(model):
                        sig = c.thinking_signature or ""
                        if sig.strip():
                            content_blocks.append(
                                {
                                    "reasoningContent": {
                                        "reasoningText": {
                                            "text": sanitize_surrogates(c.thinking),
                                            "signature": sig,
                                        }
                                    }
                                }
                            )
                        else:
                            # No valid signature: fall back to plain text
                            content_blocks.append({"text": sanitize_surrogates(c.thinking)})
                    else:
                        content_blocks.append(
                            {"reasoningContent": {"reasoningText": {"text": sanitize_surrogates(c.thinking)}}}
                        )

            if not content_blocks:
                i += 1
                continue

            result.append({"role": "assistant", "content": content_blocks})
            i += 1
            continue

        # UserMessage
        from nu_ai.types import UserMessage

        assert isinstance(msg, UserMessage)
        if isinstance(msg.content, str):
            result.append(
                {
                    "role": "user",
                    "content": [{"text": sanitize_surrogates(msg.content)}],
                }
            )
        else:
            user_blocks: list[dict[str, Any]] = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    user_blocks.append({"text": sanitize_surrogates(c.text)})
                elif isinstance(c, ImageContent):
                    user_blocks.append({"image": create_image_block(c.mime_type, c.data)})
                else:
                    raise ValueError(f"Unknown user content type: {type(c)}")
            result.append({"role": "user", "content": user_blocks})
        i += 1

    # Add cache point to last user message for supported models
    if cache_retention != "none" and supports_prompt_caching(model) and result:
        last_msg = result[-1]
        if last_msg.get("role") == "user":
            cache_point: dict[str, Any] = {"cachePoint": {"type": "default"}}
            if cache_retention == "long":
                cache_point["cachePoint"]["ttl"] = {"value": 3600, "unit": "seconds"}
            last_msg["content"].append(cache_point)

    return result


def convert_tool_config(
    tools: list[Tool] | None,
    tool_choice: BedrockOptions | None = None,
) -> dict[str, Any] | None:
    """Build the Bedrock ``toolConfig`` block."""
    raw_tool_choice = tool_choice.tool_choice if tool_choice else None
    if not tools or raw_tool_choice == "none":
        return None

    bedrock_tools = [
        {
            "toolSpec": {
                "name": t.name,
                "description": t.description,
                "inputSchema": {"json": t.parameters},
            }
        }
        for t in tools
    ]

    bedrock_choice: dict[str, Any] | None = None
    if raw_tool_choice == "auto":
        bedrock_choice = {"auto": {}}
    elif raw_tool_choice == "any":
        bedrock_choice = {"any": {}}
    elif isinstance(raw_tool_choice, dict) and raw_tool_choice.get("type") == "tool":
        bedrock_choice = {"tool": {"name": raw_tool_choice["name"]}}

    config: dict[str, Any] = {"tools": bedrock_tools}
    if bedrock_choice is not None:
        config["toolChoice"] = bedrock_choice
    return config


def build_additional_model_request_fields(
    model: Model,
    options: BedrockOptions,
) -> dict[str, Any] | None:
    """Build ``additionalModelRequestFields`` for Anthropic reasoning models."""
    if not options.reasoning or not model.reasoning:
        return None

    model_id = model.id
    is_claude = "anthropic.claude" in model_id or "anthropic/claude" in model_id
    if not is_claude:
        return None

    if supports_adaptive_thinking(model_id):
        result: dict[str, Any] = {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": map_thinking_level_to_effort(options.reasoning, model_id)},
        }
    else:
        _DEFAULT_BUDGETS: dict[str, int] = {
            "minimal": 1024,
            "low": 2048,
            "medium": 8192,
            "high": 16384,
            "xhigh": 16384,
        }
        level = options.reasoning if options.reasoning != "xhigh" else "high"
        budget = _DEFAULT_BUDGETS[options.reasoning]
        if options.thinking_budgets is not None:
            custom = getattr(options.thinking_budgets, level, None)
            if custom is not None:
                budget = custom
        result = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget,
            }
        }

    if not supports_adaptive_thinking(model_id) and (options.interleaved_thinking is not False):
        result["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]

    return result


def build_params(
    model: Model,
    context: Context,
    options: BedrockOptions | None = None,
) -> dict[str, Any]:
    """Build the full request payload for a Bedrock ``converse_stream`` call."""
    opts = options or BedrockOptions()
    cache_retention = resolve_cache_retention(opts.cache_retention)

    params: dict[str, Any] = {
        "modelId": model.id,
        "messages": convert_messages(context, model, cache_retention),
    }

    system = build_system_prompt(context.system_prompt, model, cache_retention)
    if system is not None:
        params["system"] = system

    inference_config: dict[str, Any] = {}
    if opts.max_tokens is not None:
        inference_config["maxTokens"] = opts.max_tokens
    if opts.temperature is not None:
        inference_config["temperature"] = opts.temperature
    if inference_config:
        params["inferenceConfig"] = inference_config

    tool_config = convert_tool_config(context.tools, opts)
    if tool_config is not None:
        params["toolConfig"] = tool_config

    additional = build_additional_model_request_fields(model, opts)
    if additional is not None:
        params["additionalModelRequestFields"] = additional

    if opts.request_metadata is not None:
        params["requestMetadata"] = opts.request_metadata

    return params


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


def create_bedrock_client(options: BedrockOptions | None = None) -> Any:
    """Create a ``boto3`` Bedrock Runtime client with region / credentials resolved."""
    import boto3

    opts = options or BedrockOptions()
    kwargs: dict[str, Any] = {}

    if opts.profile:
        kwargs["profile_name"] = opts.profile

    # Region resolution: explicit option > env vars > SDK default > us-east-1
    explicit_region = opts.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if explicit_region:
        kwargs["region_name"] = explicit_region
    elif not os.environ.get("AWS_PROFILE"):
        kwargs["region_name"] = "us-east-1"

    if os.environ.get("AWS_BEDROCK_SKIP_AUTH") == "1":
        kwargs["aws_access_key_id"] = "dummy-access-key"
        kwargs["aws_secret_access_key"] = "dummy-secret-key"

    return boto3.client("bedrock-runtime", **kwargs)


# Hold strong references so GC doesn't kill in-flight background tasks.
_background_tasks: set[asyncio.Task[None]] = set()


def _format_bedrock_error(exc: Exception) -> str:
    """Format a boto3/botocore exception with a stable human-readable prefix."""
    _ERROR_PREFIXES: dict[str, str] = {
        "InternalServerException": "Internal server error",
        "ModelStreamErrorException": "Model stream error",
        "ValidationException": "Validation error",
        "ThrottlingException": "Throttling error",
        "ServiceUnavailableException": "Service unavailable",
    }
    name = type(exc).__name__
    message = str(exc)
    # botocore ClientError carries the code inside .response
    if hasattr(exc, "response"):
        try:
            code = exc.response["Error"]["Code"]  # type: ignore[union-attr]
            prefix = _ERROR_PREFIXES.get(code, code)
            return f"{prefix}: {message}"
        except (KeyError, TypeError):
            pass
    prefix = _ERROR_PREFIXES.get(name)
    return f"{prefix}: {message}" if prefix else message


async def _run_bedrock_stream(
    *,
    model: Model,
    context: Context,
    options: BedrockOptions | None,
    stream: AssistantMessageEventStream,
    output: AssistantMessage,
    client: Any | None,
) -> None:
    """Async coroutine driving the Bedrock ``converse_stream`` call."""
    try:
        resolved_client = client or create_bedrock_client(options)
        params = build_params(model, context, options)

        stream.push(StartEvent(partial=_copy_output(output)))

        # boto3 is synchronous; run in a thread pool to avoid blocking the loop.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: resolved_client.converse_stream(**params),
        )

        # Track active content blocks by their Bedrock contentBlockIndex.
        # Each entry is (content_list_index, block)
        block_map: dict[int, tuple[int, Any]] = {}
        # Partial JSON accumulation for tool calls: keyed by content_list_index
        partial_tool_json: dict[int, str] = {}

        for item in response.get("stream") or []:
            if "messageStart" in item:
                pass  # StartEvent already pushed
            elif "contentBlockStart" in item:
                cbs = item["contentBlockStart"]
                cb_index = cbs.get("contentBlockIndex", 0)
                start = cbs.get("start") or {}
                if "toolUse" in start:
                    tu = start["toolUse"]
                    block = ToolCall(
                        id=tu.get("toolUseId") or "",
                        name=tu.get("name") or "",
                        arguments={},
                    )
                    output.content.append(block)
                    list_idx = len(output.content) - 1
                    block_map[cb_index] = (list_idx, block)
                    partial_tool_json[list_idx] = ""
                    stream.push(
                        ToolCallStartEvent(
                            content_index=list_idx,
                            partial=_copy_output(output),
                        )
                    )
            elif "contentBlockDelta" in item:
                cbd = item["contentBlockDelta"]
                cb_index = cbd.get("contentBlockIndex", 0)
                delta = cbd.get("delta") or {}

                if "text" in delta:
                    text_delta = delta["text"]
                    # Text blocks may not have a prior contentBlockStart; create lazily.
                    if cb_index not in block_map:
                        block = TextContent(text="")
                        output.content.append(block)
                        list_idx = len(output.content) - 1
                        block_map[cb_index] = (list_idx, block)
                        stream.push(
                            TextStartEvent(
                                content_index=list_idx,
                                partial=_copy_output(output),
                            )
                        )
                    list_idx, block = block_map[cb_index]
                    if isinstance(block, TextContent):
                        block.text += text_delta
                        stream.push(
                            TextDeltaEvent(
                                content_index=list_idx,
                                delta=text_delta,
                                partial=_copy_output(output),
                            )
                        )

                elif "toolUse" in delta:
                    if cb_index in block_map:
                        list_idx, block = block_map[cb_index]
                        if isinstance(block, ToolCall):
                            args_delta = delta["toolUse"].get("input") or ""
                            partial_tool_json[list_idx] = partial_tool_json.get(list_idx, "") + args_delta
                            block.arguments = parse_streaming_json(partial_tool_json[list_idx]) or {}
                            stream.push(
                                ToolCallDeltaEvent(
                                    content_index=list_idx,
                                    delta=args_delta,
                                    partial=_copy_output(output),
                                )
                            )

                elif "reasoningContent" in delta:
                    rc = delta["reasoningContent"]
                    if cb_index not in block_map:
                        block = ThinkingContent(thinking="", thinking_signature="")
                        output.content.append(block)
                        list_idx = len(output.content) - 1
                        block_map[cb_index] = (list_idx, block)
                        stream.push(
                            ThinkingStartEvent(
                                content_index=list_idx,
                                partial=_copy_output(output),
                            )
                        )
                    list_idx, block = block_map[cb_index]
                    if isinstance(block, ThinkingContent):
                        if "text" in rc:
                            block.thinking += rc["text"]
                            stream.push(
                                ThinkingDeltaEvent(
                                    content_index=list_idx,
                                    delta=rc["text"],
                                    partial=_copy_output(output),
                                )
                            )
                        if "signature" in rc:
                            block.thinking_signature = (block.thinking_signature or "") + rc["signature"]

            elif "contentBlockStop" in item:
                cbs = item["contentBlockStop"]
                cb_index = cbs.get("contentBlockIndex", 0)
                if cb_index in block_map:
                    list_idx, block = block_map[cb_index]
                    if isinstance(block, TextContent):
                        stream.push(
                            TextEndEvent(
                                content_index=list_idx,
                                content=block.text,
                                partial=_copy_output(output),
                            )
                        )
                    elif isinstance(block, ThinkingContent):
                        stream.push(
                            ThinkingEndEvent(
                                content_index=list_idx,
                                content=block.thinking,
                                partial=_copy_output(output),
                            )
                        )
                    elif isinstance(block, ToolCall):
                        partial_json = partial_tool_json.pop(list_idx, "")
                        block.arguments = parse_streaming_json(partial_json) or {}
                        stream.push(
                            ToolCallEndEvent(
                                content_index=list_idx,
                                tool_call=block.model_copy(deep=True),
                                partial=_copy_output(output),
                            )
                        )

            elif "messageStop" in item:
                output.stop_reason = map_stop_reason(item["messageStop"].get("stopReason"))

            elif "metadata" in item:
                meta = item["metadata"]
                usage_data = meta.get("usage") or {}
                if usage_data:
                    output.usage.input = usage_data.get("inputTokens") or 0
                    output.usage.output = usage_data.get("outputTokens") or 0
                    output.usage.cache_read = usage_data.get("cacheReadInputTokens") or 0
                    output.usage.cache_write = usage_data.get("cacheWriteInputTokens") or 0
                    output.usage.total_tokens = (
                        usage_data.get("totalTokens") or output.usage.input + output.usage.output
                    )
                    calculate_cost(model, output.usage)

        if output.stop_reason in ("error", "aborted"):
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
        output.error_message = _format_bedrock_error(exc)
        stream.push(ErrorEvent(reason="error", error=_copy_output(output)))
        stream.end()


def stream_bedrock(
    model: Model,
    context: Context,
    options: BedrockOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from Amazon Bedrock via the Converse API."""
    stream = AssistantMessageEventStream()
    output = _new_output(model)
    task = asyncio.create_task(
        _run_bedrock_stream(
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


def stream_simple_bedrock(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Simple-options variant: maps :class:`SimpleStreamOptions` to :class:`BedrockOptions`.

    For Claude models, reasoning levels are mapped to budget tokens or adaptive
    thinking depending on model generation. Non-Claude models pass reasoning
    through unchanged if the model supports it.
    """
    base = build_base_options(model, options)
    reasoning = options.reasoning if options else None

    if not reasoning:
        return stream_bedrock(
            model,
            context,
            BedrockOptions(
                temperature=base.temperature,
                max_tokens=base.max_tokens,
                api_key=base.api_key,
                cache_retention=base.cache_retention,
                session_id=base.session_id,
                headers=base.headers,
                max_retry_delay_ms=base.max_retry_delay_ms,
                metadata=base.metadata,
            ),
            client=client,
        )

    model_id = model.id
    is_claude = "anthropic.claude" in model_id or "anthropic/claude" in model_id

    if is_claude and not supports_adaptive_thinking(model_id):
        adjusted = adjust_max_tokens_for_thinking(
            base.max_tokens or 0,
            model.max_tokens,
            reasoning,
            options.thinking_budgets if options else None,
        )
        clamped = clamp_reasoning(reasoning)
        budgets_dict: dict[str, int] = {}
        if clamped:
            budgets_dict[clamped] = adjusted["thinking_budget"]

        from nu_ai.types import ThinkingBudgets

        return stream_bedrock(
            model,
            context,
            BedrockOptions(
                temperature=base.temperature,
                max_tokens=adjusted["max_tokens"],
                api_key=base.api_key,
                cache_retention=base.cache_retention,
                session_id=base.session_id,
                headers=base.headers,
                max_retry_delay_ms=base.max_retry_delay_ms,
                metadata=base.metadata,
                reasoning=reasoning,
                thinking_budgets=ThinkingBudgets(**{k: v for k, v in budgets_dict.items()}),
            ),
            client=client,
        )

    return stream_bedrock(
        model,
        context,
        BedrockOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            max_retry_delay_ms=base.max_retry_delay_ms,
            metadata=base.metadata,
            reasoning=reasoning,
            thinking_budgets=options.thinking_budgets if options else None,
        ),
        client=client,
    )


__all__ = [
    "BedrockOptions",
    "build_additional_model_request_fields",
    "build_params",
    "build_system_prompt",
    "convert_messages",
    "convert_tool_config",
    "create_bedrock_client",
    "create_image_block",
    "map_stop_reason",
    "map_thinking_level_to_effort",
    "normalize_tool_call_id",
    "resolve_cache_retention",
    "stream_bedrock",
    "stream_simple_bedrock",
    "supports_adaptive_thinking",
    "supports_prompt_caching",
    "supports_thinking_signature",
]
