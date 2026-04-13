"""OpenAI Chat Completions provider.

Direct port of ``packages/ai/src/providers/openai-completions.ts``. Covers
the OpenAI Chat Completions wire format — which is also the format exposed
by Ollama (``/v1/chat/completions``), Groq, Cerebras, xAI, OpenRouter,
Together, Fireworks, DeepSeek, LMStudio, and many other providers. The
per-provider differences are folded into a single resolved
:class:`nu_ai.types.OpenAICompletionsCompat` object via
:func:`detect_compat` / :func:`get_compat`.

This module has two logical sections:

1. **Pure transforms** — ``convert_messages``, ``convert_tools``,
   ``build_params``, ``detect_compat``/``get_compat``, ``map_stop_reason``,
   ``parse_chunk_usage``. All no-SDK so they can be unit-tested in isolation.
2. **SDK glue** — ``create_client``, ``stream_openai_completions``,
   ``stream_simple_openai_completions`` built on the Python ``openai`` SDK.

The glue supports a ``client=`` injection point for tests — same pattern as
the Anthropic port.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

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
    OpenAICompletionsCompat,
    StartEvent,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
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
from nu_ai.utils.json_parse import parse_streaming_json
from nu_ai.utils.sanitize_unicode import sanitize_surrogates

if TYPE_CHECKING:
    from nu_ai.types import (
        Context,
        Message,
        Model,
        OpenAICompletionsOptions,
        SimpleStreamOptions,
        ThinkingLevel,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def has_tool_history(messages: list[Message]) -> bool:
    """Return ``True`` if any message carries tool calls or results.

    Used to force a ``tools: []`` param on OpenAI-compatible Anthropic proxies
    that reject tool-history messages without a ``tools`` array.
    """
    for msg in messages:
        if isinstance(msg, ToolResultMessage):
            return True
        if isinstance(msg, AssistantMessage) and any(isinstance(b, ToolCall) for b in msg.content):
            return True
    return False


class StopReasonResult(TypedDict):
    stop_reason: StopReason
    error_message: str | None


def map_stop_reason(reason: str | None) -> StopReasonResult:
    """Map an OpenAI ``finish_reason`` to nu_ai's :data:`StopReason`."""
    if reason is None:
        return {"stop_reason": "stop", "error_message": None}
    if reason in ("stop", "end"):
        return {"stop_reason": "stop", "error_message": None}
    if reason == "length":
        return {"stop_reason": "length", "error_message": None}
    if reason in ("function_call", "tool_calls"):
        return {"stop_reason": "toolUse", "error_message": None}
    if reason == "content_filter":
        return {
            "stop_reason": "error",
            "error_message": "Provider finish_reason: content_filter",
        }
    if reason == "network_error":
        return {
            "stop_reason": "error",
            "error_message": "Provider finish_reason: network_error",
        }
    return {
        "stop_reason": "error",
        "error_message": f"Provider finish_reason: {reason}",
    }


def map_reasoning_effort(
    effort: ThinkingLevel,
    reasoning_effort_map: dict[str, str],
) -> str:
    """Map a nu_ai reasoning level via a provider-specific override table."""
    return reasoning_effort_map.get(effort, effort)


def parse_chunk_usage(raw: dict[str, Any], model: Model) -> Usage:
    """Parse the ``usage`` object from an OpenAI stream chunk into :class:`Usage`.

    Mirrors the upstream normalization rules: reasoning tokens are added to
    ``output``; some providers (notably OpenRouter) report
    ``cached_tokens`` as ``prior-hits + current-writes`` so the parser
    subtracts ``cache_write_tokens`` to recover the true cache-read count.
    """
    prompt_tokens = raw.get("prompt_tokens") or 0
    prompt_details = raw.get("prompt_tokens_details") or {}
    reported_cached_tokens = prompt_details.get("cached_tokens") or 0
    cache_write_tokens = prompt_details.get("cache_write_tokens") or 0
    completion_details = raw.get("completion_tokens_details") or {}
    reasoning_tokens = completion_details.get("reasoning_tokens") or 0

    if cache_write_tokens > 0:
        cache_read_tokens = max(0, reported_cached_tokens - cache_write_tokens)
    else:
        cache_read_tokens = reported_cached_tokens

    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = (raw.get("completion_tokens") or 0) + reasoning_tokens

    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read_tokens,
        cache_write=cache_write_tokens,
        total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )
    calculate_cost(model, usage)
    return usage


# ---------------------------------------------------------------------------
# Compat detection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ResolvedCompat:
    """Fully-resolved compat settings used by the provider internals.

    Upstream uses ``Required<OpenAICompletionsCompat>`` — every field is
    populated. This mirrors that: nothing is ``None``.
    """

    supports_store: bool
    supports_developer_role: bool
    supports_reasoning_effort: bool
    reasoning_effort_map: dict[str, str]
    supports_usage_in_streaming: bool
    max_tokens_field: str
    requires_tool_result_name: bool
    requires_assistant_after_tool_result: bool
    requires_thinking_as_text: bool
    thinking_format: str
    zai_tool_stream: bool
    supports_strict_mode: bool

    def model_copy(self, *, update: dict[str, Any]) -> _ResolvedCompat:
        """Return a copy with the given fields updated."""
        fields: dict[str, Any] = {
            "supports_store": self.supports_store,
            "supports_developer_role": self.supports_developer_role,
            "supports_reasoning_effort": self.supports_reasoning_effort,
            "reasoning_effort_map": self.reasoning_effort_map,
            "supports_usage_in_streaming": self.supports_usage_in_streaming,
            "max_tokens_field": self.max_tokens_field,
            "requires_tool_result_name": self.requires_tool_result_name,
            "requires_assistant_after_tool_result": self.requires_assistant_after_tool_result,
            "requires_thinking_as_text": self.requires_thinking_as_text,
            "thinking_format": self.thinking_format,
            "zai_tool_stream": self.zai_tool_stream,
            "supports_strict_mode": self.supports_strict_mode,
        }
        fields.update(update)
        return _ResolvedCompat(**fields)


def detect_compat(model: Model) -> _ResolvedCompat:
    """Auto-detect compat settings from ``model.provider`` and ``model.base_url``."""
    provider = model.provider
    base_url = model.base_url

    is_zai = provider == "zai" or "api.z.ai" in base_url
    is_non_standard = (
        provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_zai
        or provider == "opencode"
        or "opencode.ai" in base_url
    )
    use_max_tokens = "chutes.ai" in base_url
    is_grok = provider == "xai" or "api.x.ai" in base_url

    is_groq_qwen = (provider == "groq" or "groq.com" in base_url) and model.id == "qwen/qwen3-32b"
    reasoning_effort_map: dict[str, str] = {}
    if is_groq_qwen:
        reasoning_effort_map = {
            "minimal": "default",
            "low": "default",
            "medium": "default",
            "high": "default",
            "xhigh": "default",
        }

    if is_zai:
        thinking_format = "zai"
    elif provider == "openrouter" or "openrouter.ai" in base_url:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    return _ResolvedCompat(
        supports_store=not is_non_standard,
        supports_developer_role=not is_non_standard,
        supports_reasoning_effort=not is_grok and not is_zai,
        reasoning_effort_map=reasoning_effort_map,
        supports_usage_in_streaming=True,
        max_tokens_field="max_tokens" if use_max_tokens else "max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        thinking_format=thinking_format,
        zai_tool_stream=False,
        supports_strict_mode=True,
    )


def get_compat(model: Model) -> _ResolvedCompat:
    """Resolve compat settings: explicit ``model.compat`` overrides detection."""
    detected = detect_compat(model)
    if model.compat is None or not isinstance(model.compat, OpenAICompletionsCompat):
        return detected
    overrides: dict[str, Any] = {}
    compat = model.compat
    if compat.supports_store is not None:
        overrides["supports_store"] = compat.supports_store
    if compat.supports_developer_role is not None:
        overrides["supports_developer_role"] = compat.supports_developer_role
    if compat.supports_reasoning_effort is not None:
        overrides["supports_reasoning_effort"] = compat.supports_reasoning_effort
    if compat.reasoning_effort_map is not None:
        overrides["reasoning_effort_map"] = dict(compat.reasoning_effort_map)
    if compat.supports_usage_in_streaming is not None:
        overrides["supports_usage_in_streaming"] = compat.supports_usage_in_streaming
    if compat.max_tokens_field is not None:
        overrides["max_tokens_field"] = compat.max_tokens_field
    if compat.requires_tool_result_name is not None:
        overrides["requires_tool_result_name"] = compat.requires_tool_result_name
    if compat.requires_assistant_after_tool_result is not None:
        overrides["requires_assistant_after_tool_result"] = compat.requires_assistant_after_tool_result
    if compat.requires_thinking_as_text is not None:
        overrides["requires_thinking_as_text"] = compat.requires_thinking_as_text
    if compat.thinking_format is not None:
        overrides["thinking_format"] = compat.thinking_format
    if compat.zai_tool_stream is not None:
        overrides["zai_tool_stream"] = compat.zai_tool_stream
    if compat.supports_strict_mode is not None:
        overrides["supports_strict_mode"] = compat.supports_strict_mode
    return detected.model_copy(update=overrides)


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def convert_tools(tools: list[Tool], compat: _ResolvedCompat) -> list[dict[str, Any]]:
    """Convert nu_ai :class:`Tool` list to OpenAI ``tools`` payload."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        function_def: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode:
            function_def["strict"] = False
        result.append({"type": "function", "function": function_def})
    return result


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


_ILLEGAL_TOOL_CALL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _normalize_tool_call_id(model: Model) -> Any:
    """Build a ``normalize_tool_call_id`` callback for :func:`transform_messages`."""

    def normalize(tool_call_id: str, *_: object) -> str:
        if "|" in tool_call_id:
            call_id = tool_call_id.split("|", 1)[0]
            return _ILLEGAL_TOOL_CALL_ID_CHARS.sub("_", call_id)[:40]
        if model.provider == "openai" and len(tool_call_id) > 40:
            return tool_call_id[:40]
        return tool_call_id

    return normalize


def _image_data_url(block: ImageContent) -> str:
    return f"data:{block.mime_type};base64,{block.data}"


def convert_messages(
    model: Model,
    context: Context,
    compat: _ResolvedCompat,
) -> list[dict[str, Any]]:
    """Convert nu_ai context messages to the OpenAI ``messages`` array.

    Handles:

    * System prompt prepending (using ``developer`` role for reasoning models
      that support it).
    * Text-only vs mixed user content.
    * Image filtering on text-only models.
    * Assistant messages with text, thinking, and tool calls.
    * Consecutive tool-result messages → tool role messages + optional
      follow-up user message when images are present.
    * Synthetic assistant message between tool result and user when the
      provider requires it.
    """
    params: list[dict[str, Any]] = []
    transformed = transform_messages(context.messages, model, _normalize_tool_call_id(model))

    if context.system_prompt:
        use_developer_role = model.reasoning and compat.supports_developer_role
        role = "developer" if use_developer_role else "system"
        params.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    last_role: str | None = None

    i = 0
    while i < len(transformed):
        msg = transformed[i]

        # Bridge tool_result → user with a synthetic assistant for providers that require it.
        if compat.requires_assistant_after_tool_result and last_role == "toolResult" and isinstance(msg, UserMessage):
            params.append({"role": "assistant", "content": "I have processed the tool results."})

        if isinstance(msg, UserMessage):
            _append_user_message(params, msg, model)
            last_role = "user"
            i += 1
            continue

        if isinstance(msg, AssistantMessage):
            if _append_assistant_message(params, msg, compat):
                last_role = "assistant"
            i += 1
            continue

        # ToolResultMessage — collapse consecutive results.
        image_blocks: list[dict[str, Any]] = []
        j = i
        while j < len(transformed) and isinstance(transformed[j], ToolResultMessage):
            tool_msg = transformed[j]
            assert isinstance(tool_msg, ToolResultMessage)
            text_parts = [c.text for c in tool_msg.content if isinstance(c, TextContent)]
            text_result = "\n".join(text_parts)
            has_text = bool(text_result)
            tool_result: dict[str, Any] = {
                "role": "tool",
                "content": sanitize_surrogates(text_result if has_text else "(see attached image)"),
                "tool_call_id": tool_msg.tool_call_id,
            }
            if compat.requires_tool_result_name and tool_msg.tool_name:
                tool_result["name"] = tool_msg.tool_name
            params.append(tool_result)

            if "image" in model.input:
                for block in tool_msg.content:
                    if isinstance(block, ImageContent):
                        image_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_data_url(block)},
                            }
                        )
            j += 1

        i = j

        if image_blocks:
            if compat.requires_assistant_after_tool_result:
                params.append({"role": "assistant", "content": "I have processed the tool results."})
            params.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Attached image(s) from tool result:"},
                        *image_blocks,
                    ],
                }
            )
            last_role = "user"
        else:
            last_role = "toolResult"

    return params


def _append_user_message(params: list[dict[str, Any]], msg: UserMessage, model: Model) -> None:
    if isinstance(msg.content, str):
        params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
        return

    content: list[dict[str, Any]] = []
    for item in msg.content:
        if isinstance(item, TextContent):
            content.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(item)},
                }
            )

    if "image" not in model.input:
        content = [c for c in content if c["type"] != "image_url"]
    if not content:
        return
    params.append({"role": "user", "content": content})


def _append_assistant_message(
    params: list[dict[str, Any]],
    msg: AssistantMessage,
    compat: _ResolvedCompat,
) -> bool:
    """Append an assistant message. Returns ``True`` when the message was kept."""
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "" if compat.requires_assistant_after_tool_result else None,
    }

    text_blocks = [b for b in msg.content if isinstance(b, TextContent) and b.text.strip()]
    if text_blocks:
        assistant_msg["content"] = "".join(sanitize_surrogates(b.text) for b in text_blocks)

    thinking_blocks = [b for b in msg.content if isinstance(b, ThinkingContent) and b.thinking.strip()]
    if thinking_blocks:
        if compat.requires_thinking_as_text:
            thinking_text = "\n\n".join(b.thinking for b in thinking_blocks)
            current = assistant_msg["content"]
            if isinstance(current, list):
                current.insert(0, {"type": "text", "text": thinking_text})
            else:
                assistant_msg["content"] = [{"type": "text", "text": thinking_text}]
        else:
            signature = thinking_blocks[0].thinking_signature
            if signature:
                assistant_msg[signature] = "\n".join(b.thinking for b in thinking_blocks)

    tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
    if tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ]
        reasoning_details: list[Any] = []
        for tc in tool_calls:
            if tc.thought_signature:
                with contextlib.suppress(ValueError, TypeError):
                    reasoning_details.append(json.loads(tc.thought_signature))
        if reasoning_details:
            assistant_msg["reasoning_details"] = reasoning_details

    content = assistant_msg.get("content")
    has_content = content is not None and (
        (isinstance(content, str) and len(content) > 0) or (isinstance(content, list) and len(content) > 0)
    )
    if not has_content and "tool_calls" not in assistant_msg:
        return False
    params.append(assistant_msg)
    return True


# ---------------------------------------------------------------------------
# Params builder
# ---------------------------------------------------------------------------


def build_params(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
) -> dict[str, Any]:
    """Build the JSON payload for an OpenAI Chat Completions streaming request."""
    compat = get_compat(model)
    messages = convert_messages(model, context, compat)

    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = False

    if options is not None and options.max_tokens is not None:
        params[compat.max_tokens_field] = options.max_tokens

    if options is not None and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = convert_tools(context.tools, compat)
        if compat.zai_tool_stream:
            params["tool_stream"] = True
    elif has_tool_history(context.messages):
        # OpenAI-compatible Anthropic proxies require ``tools: []`` when
        # the conversation carries tool calls/results.
        params["tools"] = []

    if options is not None and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    reasoning_effort = options.reasoning_effort if options is not None else None

    if (compat.thinking_format == "zai" and model.reasoning) or (compat.thinking_format == "qwen" and model.reasoning):
        params["enable_thinking"] = bool(reasoning_effort)
    elif compat.thinking_format == "qwen-chat-template" and model.reasoning:
        params["chat_template_kwargs"] = {"enable_thinking": bool(reasoning_effort)}
    elif compat.thinking_format == "openrouter" and model.reasoning:
        if reasoning_effort:
            params["reasoning"] = {
                "effort": map_reasoning_effort(reasoning_effort, compat.reasoning_effort_map),
            }
        else:
            params["reasoning"] = {"effort": "none"}
    elif reasoning_effort and model.reasoning and compat.supports_reasoning_effort:
        params["reasoning_effort"] = map_reasoning_effort(reasoning_effort, compat.reasoning_effort_map)

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


def create_client(
    model: Model,
    context: Context,
    api_key: str | None = None,
    options_headers: dict[str, str] | None = None,
) -> Any:
    """Construct an :class:`openai.AsyncOpenAI` client for ``model``.

    Imports ``openai`` lazily so tests that inject a fake client never pull
    in the SDK. For GitHub Copilot models the per-request dynamic headers
    (``X-Initiator``, vision toggle) are merged in here.
    """
    from openai import AsyncOpenAI

    if not api_key:
        import os

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


@dataclass(slots=True)
class _StreamState:
    """Mutable state carried across chunk-handling for a single stream."""

    current_block: TextContent | ThinkingContent | ToolCall | None = None
    partial_tool_args: str = ""


def _emit_end_for_current(
    state: _StreamState,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> None:
    block = state.current_block
    if block is None:
        return
    idx = len(output.content) - 1
    if isinstance(block, TextContent):
        stream.push(
            TextEndEvent(
                content_index=idx,
                content=block.text,
                partial=_copy_output(output),
            )
        )
    elif isinstance(block, ThinkingContent):
        stream.push(
            ThinkingEndEvent(
                content_index=idx,
                content=block.thinking,
                partial=_copy_output(output),
            )
        )
    else:  # ToolCall
        parsed = parse_streaming_json(state.partial_tool_args) or {}
        block.arguments = parsed
        state.partial_tool_args = ""
        stream.push(
            ToolCallEndEvent(
                content_index=idx,
                tool_call=block.model_copy(deep=True),
                partial=_copy_output(output),
            )
        )
    state.current_block = None


_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def _handle_chunk(
    chunk: Any,
    *,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    state: _StreamState,
    model: Model,
) -> None:
    if chunk is None:
        return

    chunk_id = _get_attr(chunk, "id")
    if chunk_id and output.response_id is None:
        output.response_id = chunk_id

    usage_payload = _to_dict(_get_attr(chunk, "usage"))
    if usage_payload:
        output.usage = parse_chunk_usage(usage_payload, model)

    choices = _get_attr(chunk, "choices") or []
    if not choices:
        return
    choice = choices[0]

    # Some providers return usage on the choice instead of the chunk.
    if not usage_payload:
        choice_usage = _to_dict(_get_attr(choice, "usage"))
        if choice_usage:
            output.usage = parse_chunk_usage(choice_usage, model)

    finish_reason = _get_attr(choice, "finish_reason")
    if finish_reason:
        mapped = map_stop_reason(finish_reason)
        output.stop_reason = mapped["stop_reason"]
        if mapped.get("error_message"):
            output.error_message = mapped["error_message"]

    delta = _get_attr(choice, "delta")
    if delta is None:
        return

    # Text content.
    delta_content = _get_attr(delta, "content")
    if delta_content:
        if not isinstance(state.current_block, TextContent):
            _emit_end_for_current(state, output, stream)
            state.current_block = TextContent(text="")
            output.content.append(state.current_block)
            stream.push(
                TextStartEvent(
                    content_index=len(output.content) - 1,
                    partial=_copy_output(output),
                )
            )
        state.current_block.text += delta_content
        stream.push(
            TextDeltaEvent(
                content_index=len(output.content) - 1,
                delta=delta_content,
                partial=_copy_output(output),
            )
        )

    # Reasoning content (llama.cpp, DeepSeek, etc.).
    reasoning_field: str | None = None
    for field in _REASONING_FIELDS:
        value = _get_attr(delta, field)
        if value:
            reasoning_field = field
            break

    if reasoning_field is not None:
        reasoning_delta = _get_attr(delta, reasoning_field) or ""
        if not isinstance(state.current_block, ThinkingContent):
            _emit_end_for_current(state, output, stream)
            state.current_block = ThinkingContent(thinking="", thinking_signature=reasoning_field)
            output.content.append(state.current_block)
            stream.push(
                ThinkingStartEvent(
                    content_index=len(output.content) - 1,
                    partial=_copy_output(output),
                )
            )
        state.current_block.thinking += reasoning_delta
        stream.push(
            ThinkingDeltaEvent(
                content_index=len(output.content) - 1,
                delta=reasoning_delta,
                partial=_copy_output(output),
            )
        )

    # Tool calls.
    tool_calls_delta = _get_attr(delta, "tool_calls") or []
    for tool_call_delta in tool_calls_delta:
        tc_id = _get_attr(tool_call_delta, "id") or ""
        function = _get_attr(tool_call_delta, "function")
        tc_name = _get_attr(function, "name") if function is not None else None
        tc_args = _get_attr(function, "arguments") if function is not None else None

        if not isinstance(state.current_block, ToolCall) or (tc_id and state.current_block.id != tc_id):
            _emit_end_for_current(state, output, stream)
            state.current_block = ToolCall(id=tc_id, name=tc_name or "", arguments={})
            state.partial_tool_args = ""
            output.content.append(state.current_block)
            stream.push(
                ToolCallStartEvent(
                    content_index=len(output.content) - 1,
                    partial=_copy_output(output),
                )
            )

        if tc_id:
            state.current_block.id = tc_id
        if tc_name:
            state.current_block.name = tc_name
        if tc_args:
            state.partial_tool_args += tc_args
            state.current_block.arguments = parse_streaming_json(state.partial_tool_args) or {}
            stream.push(
                ToolCallDeltaEvent(
                    content_index=len(output.content) - 1,
                    delta=tc_args,
                    partial=_copy_output(output),
                )
            )


def _get_attr(obj: Any, name: str) -> Any:
    """Duck-typed attribute accessor — supports both SDK objects and dicts."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _to_dict(obj: Any) -> dict[str, Any] | None:
    """Coerce an SDK object to a dict if it isn't one already."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return None


async def _run_openai_completions_stream(
    *,
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None,
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

        state = _StreamState()
        response = await resolved_client.chat.completions.create(**params)
        async for chunk in response:
            _handle_chunk(chunk, output=output, stream=stream, state=state, model=model)

        _emit_end_for_current(state, output, stream)

        if output.stop_reason == "error":
            raise RuntimeError(output.error_message or "Provider returned an error stop reason")

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


def stream_openai_completions(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from an OpenAI Chat Completions-compatible endpoint."""
    stream = AssistantMessageEventStream()
    output = _new_output(model)
    task = asyncio.create_task(
        _run_openai_completions_stream(
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


def stream_simple_openai_completions(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Simple-options variant: maps :class:`SimpleStreamOptions` into ``OpenAICompletionsOptions``.

    Lowers the unified ``reasoning`` level into ``reasoning_effort``, clamping
    ``xhigh`` down to ``high`` on models that don't support it (``supports_xhigh``).
    """
    from nu_ai.types import OpenAICompletionsOptions as _OpenAIOptions

    base = build_base_options(model, options)
    if options is not None and options.reasoning is not None:
        reasoning_effort = options.reasoning if supports_xhigh(model) else clamp_reasoning(options.reasoning)
    else:
        reasoning_effort = None

    merged = _OpenAIOptions(
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
    return stream_openai_completions(model, context, merged, client=client)


__all__ = [
    "build_params",
    "convert_messages",
    "convert_tools",
    "create_client",
    "detect_compat",
    "get_compat",
    "has_tool_history",
    "map_reasoning_effort",
    "map_stop_reason",
    "parse_chunk_usage",
    "stream_openai_completions",
    "stream_simple_openai_completions",
]
