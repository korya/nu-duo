"""Anthropic Messages API provider.

Direct port of ``packages/ai/src/providers/anthropic.ts``. This module hosts
two logical sections:

1. **Pure transforms** — cache control resolution, Claude Code tool name
   canonicalization, message conversion, tool conversion, param building,
   stop-reason mapping. These drive the outbound request payload and the
   inbound response decoding without touching the network.
2. **SDK glue** — client construction and async streaming over the official
   ``anthropic`` Python SDK. This section will be filled in by the second
   slice of the Anthropic port (abstraction 8b); transforms land first so
   the heavy logic can be unit-tested without replay fixtures.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import TYPE_CHECKING, Any, TypedDict

from nu_ai.env_api_keys import get_env_api_key
from nu_ai.models import calculate_cost
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
        AnthropicEffort,
        AnthropicOptions,
        Context,
        Message,
        Model,
        ThinkingLevel,
    )


# ---------------------------------------------------------------------------
# Cache control
# ---------------------------------------------------------------------------


def resolve_cache_retention(cache_retention: CacheRetention | None) -> CacheRetention:
    """Resolve the effective cache retention for an Anthropic request.

    Precedence:

    1. Explicit ``cache_retention`` argument.
    2. ``PI_CACHE_RETENTION=long`` environment variable (legacy opt-in).
    3. Default ``"short"``.
    """
    if cache_retention:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


class CacheControlResult(TypedDict):
    retention: CacheRetention
    cache_control: dict[str, str] | None


def get_cache_control(base_url: str, cache_retention: CacheRetention | None = None) -> CacheControlResult:
    """Return the resolved retention and the provider-shaped cache control header.

    Long-retention caching is only supported by the upstream Anthropic API
    (``api.anthropic.com``). Other base URLs fall back to the default 5 min TTL.
    """
    retention = resolve_cache_retention(cache_retention)
    if retention == "none":
        return {"retention": retention, "cache_control": None}

    cache_control: dict[str, str] = {"type": "ephemeral"}
    if retention == "long" and "api.anthropic.com" in base_url:
        cache_control["ttl"] = "1h"
    return {"retention": retention, "cache_control": cache_control}


# ---------------------------------------------------------------------------
# Claude Code tool name canonicalization (OAuth stealth mode)
# ---------------------------------------------------------------------------

# Canonical Claude Code 2.x tool names. Kept in lock-step with upstream
# ``claudeCodeTools`` — see the cchistory repo in the TS comment for updates.
_CLAUDE_CODE_TOOLS: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
)

_CC_TOOL_LOOKUP: dict[str, str] = {t.lower(): t for t in _CLAUDE_CODE_TOOLS}

# Claude Code identity header contents.
CLAUDE_CODE_VERSION = "2.1.75"
CLAUDE_CODE_IDENTITY_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."


def to_claude_code_name(name: str) -> str:
    """Upgrade ``name`` to Claude Code's canonical casing if it matches.

    Case-insensitive; unknown names pass through unchanged.
    """
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def from_claude_code_name(name: str, tools: list[Tool] | None) -> str:
    """Restore ``name`` to the caller's original tool casing.

    If ``tools`` is provided, a case-insensitive match on ``tool.name`` wins;
    otherwise ``name`` is returned as-is.
    """
    if tools:
        lower = name.lower()
        for tool in tools:
            if tool.name.lower() == lower:
                return tool.name
    return name


# ---------------------------------------------------------------------------
# Header merging
# ---------------------------------------------------------------------------


def merge_headers(*sources: dict[str, str] | None) -> dict[str, str]:
    """Merge header dicts, later sources overriding earlier ones."""
    merged: dict[str, str] = {}
    for src in sources:
        if src:
            merged.update(src)
    return merged


# ---------------------------------------------------------------------------
# Content block conversion (user messages → Anthropic content blocks)
# ---------------------------------------------------------------------------


_AnthropicContentBlock = dict[str, Any]


def convert_content_blocks(
    content: list[TextContent | ImageContent],
) -> str | list[_AnthropicContentBlock]:
    """Convert a nu_ai text/image content list to the Anthropic wire format.

    * Text-only content collapses to a single newline-joined string.
    * Mixed content becomes a list of Anthropic content block dicts. If no
      text block is present, a ``"(see attached image)"`` placeholder is
      prepended (Anthropic's image blocks must accompany a text block).
    """
    has_images = any(isinstance(c, ImageContent) for c in content)
    if not has_images:
        text = "\n".join(c.text for c in content if isinstance(c, TextContent))
        return sanitize_surrogates(text)

    blocks: list[_AnthropicContentBlock] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )

    has_text = any(b["type"] == "text" for b in blocks)
    if not has_text:
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


# ---------------------------------------------------------------------------
# Thinking support
# ---------------------------------------------------------------------------


def supports_adaptive_thinking(model_id: str) -> bool:
    """Return ``True`` iff ``model_id`` is Opus 4.6 or Sonnet 4.6.

    Adaptive thinking models decide internally when and how much to think;
    budget-based thinking parameters are ignored for them.
    """
    return "opus-4-6" in model_id or "opus-4.6" in model_id or "sonnet-4-6" in model_id or "sonnet-4.6" in model_id


def map_thinking_level_to_effort(level: ThinkingLevel | None, model_id: str) -> AnthropicEffort:
    """Map a nu_ai thinking level to an Anthropic adaptive effort level.

    ``xhigh`` promotes to ``max`` on Opus 4.6 only — everything else caps at
    ``high``.
    """
    if level == "minimal":
        return "low"
    if level == "low":
        return "low"
    if level == "medium":
        return "medium"
    if level == "high":
        return "high"
    if level == "xhigh":
        return "max" if ("opus-4-6" in model_id or "opus-4.6" in model_id) else "high"
    return "high"


# ---------------------------------------------------------------------------
# Tool call id normalization
# ---------------------------------------------------------------------------


_ILLEGAL_TOOL_CALL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_tool_call_id(tool_call_id: str, *_args: object) -> str:
    """Rewrite ``tool_call_id`` to Anthropic's ``^[a-zA-Z0-9_-]{1,64}$`` constraint.

    The signature accepts ``*_args`` so this function can be passed directly
    as the ``normalize_tool_call_id`` callback of
    :func:`nu_ai.providers.transform_messages.transform_messages`, which
    invokes ``(id, model, source_assistant_message)``.
    """
    return _ILLEGAL_TOOL_CALL_ID_CHARS.sub("_", tool_call_id)[:64]


# ---------------------------------------------------------------------------
# OAuth detection
# ---------------------------------------------------------------------------


def is_oauth_token(api_key: str) -> bool:
    """Return ``True`` iff ``api_key`` looks like a Claude Code OAuth token."""
    return "sk-ant-oat" in api_key


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------


def map_stop_reason(reason: str) -> StopReason:
    """Map an Anthropic stop reason to nu_ai's :data:`StopReason`."""
    if reason == "end_turn":
        return "stop"
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "toolUse"
    if reason == "refusal":
        return "error"
    if reason == "pause_turn":
        # pause_turn is treated as ``stop`` — caller decides whether to resume.
        return "stop"
    if reason == "stop_sequence":
        # nu_ai never supplies stop sequences, so this should not occur.
        return "stop"
    if reason == "sensitive":
        return "error"
    raise ValueError(f"Unhandled stop reason: {reason}")


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def convert_tools(tools: list[Tool], *, is_oauth_token: bool) -> list[dict[str, Any]]:
    """Convert nu_ai :class:`Tool` objects to Anthropic's ``tools`` payload."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        schema = tool.parameters
        result.append(
            {
                "name": to_claude_code_name(tool.name) if is_oauth_token else tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": schema.get("properties") or {},
                    "required": schema.get("required") or [],
                },
            }
        )
    return result


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


_MessageParam = dict[str, Any]


def _convert_user_message(
    msg: UserMessage,
    model: Model,
) -> _MessageParam | None:
    if isinstance(msg.content, str):
        sanitized = sanitize_surrogates(msg.content)
        if not sanitized.strip():
            return None
        return {"role": "user", "content": sanitized}

    blocks: list[_AnthropicContentBlock] = []
    for item in msg.content:
        if isinstance(item, TextContent):
            blocks.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": item.mime_type,
                        "data": item.data,
                    },
                }
            )

    if "image" not in model.input:
        blocks = [b for b in blocks if b["type"] != "image"]
    blocks = [b for b in blocks if b["type"] != "text" or b["text"].strip()]
    if not blocks:
        return None
    return {"role": "user", "content": blocks}


def _convert_assistant_message(
    msg: AssistantMessage,
    *,
    is_oauth_token: bool,
) -> _MessageParam | None:
    blocks: list[_AnthropicContentBlock] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            if not block.text.strip():
                continue
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        elif isinstance(block, ThinkingContent):
            if block.redacted:
                blocks.append({"type": "redacted_thinking", "data": block.thinking_signature or ""})
                continue
            if not block.thinking.strip():
                continue
            if not block.thinking_signature or not block.thinking_signature.strip():
                # Missing signature (e.g. from an aborted stream) → convert
                # to plain text so Anthropic doesn't reject the message.
                blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
            else:
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": sanitize_surrogates(block.thinking),
                        "signature": block.thinking_signature,
                    }
                )
        else:
            # ToolCall — exhaustive over AssistantContent.
            tool_name = to_claude_code_name(block.name) if is_oauth_token else block.name
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": tool_name,
                    "input": block.arguments or {},
                }
            )
    if not blocks:
        return None
    return {"role": "assistant", "content": blocks}


def _build_tool_result_block(msg: ToolResultMessage) -> _AnthropicContentBlock:
    return {
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": convert_content_blocks(msg.content),
        "is_error": msg.is_error,
    }


def convert_messages(
    messages: list[Message],
    model: Model,
    *,
    is_oauth_token: bool,
    cache_control: dict[str, str] | None = None,
) -> list[_MessageParam]:
    """Convert nu_ai messages to the Anthropic ``messages`` array.

    Responsibilities:

    * Runs :func:`transform_messages` with :func:`normalize_tool_call_id` so
      non-Anthropic tool call ids are rewritten to Anthropic's constraint.
    * Collapses consecutive ``toolResult`` messages into a single user turn
      carrying multiple ``tool_result`` blocks (z.ai Anthropic endpoint
      requirement).
    * Tags the final user message (or its last user-facing block) with
      ``cache_control`` to enable conversation-history caching.
    """
    params: list[_MessageParam] = []
    transformed = transform_messages(messages, model, normalize_tool_call_id)

    i = 0
    while i < len(transformed):
        msg = transformed[i]

        if isinstance(msg, UserMessage):
            param = _convert_user_message(msg, model)
            if param is not None:
                params.append(param)
            i += 1
            continue

        if isinstance(msg, AssistantMessage):
            param = _convert_assistant_message(msg, is_oauth_token=is_oauth_token)
            if param is not None:
                params.append(param)
            i += 1
            continue

        # ToolResultMessage — collapse consecutive tool results into one user turn.
        tool_results: list[_AnthropicContentBlock] = [_build_tool_result_block(msg)]
        j = i + 1
        while j < len(transformed) and isinstance(transformed[j], ToolResultMessage):
            tool_results.append(_build_tool_result_block(transformed[j]))  # type: ignore[arg-type]
            j += 1
        params.append({"role": "user", "content": tool_results})
        i = j

    # Attach cache_control to the last cache-eligible block in the last
    # user turn so conversation history is cached.
    if cache_control and params:
        last = params[-1]
        if last["role"] == "user":
            content = last["content"]
            if isinstance(content, list) and content:
                last_block = content[-1]
                if last_block.get("type") in {"text", "image", "tool_result"}:
                    last_block["cache_control"] = cache_control
            elif isinstance(content, str):
                last["content"] = [
                    {"type": "text", "text": content, "cache_control": cache_control},
                ]

    return params


# ---------------------------------------------------------------------------
# Request payload builder
# ---------------------------------------------------------------------------


def build_params(
    model: Model,
    context: Context,
    *,
    is_oauth_token: bool,
    options: AnthropicOptions | None = None,
) -> dict[str, Any]:
    """Build the JSON payload for an Anthropic Messages API streaming request.

    Mirrors ``buildParams`` from ``anthropic.ts`` field-for-field, including
    Claude Code identity injection for OAuth tokens, adaptive-vs-budget
    thinking selection, and the ``tool_choice``/``metadata`` passthroughs.
    """
    cache = get_cache_control(model.base_url, options.cache_retention if options else None)
    cache_control = cache["cache_control"]

    params: dict[str, Any] = {
        "model": model.id,
        "messages": convert_messages(
            context.messages,
            model,
            is_oauth_token=is_oauth_token,
            cache_control=cache_control,
        ),
        "max_tokens": (options.max_tokens if options and options.max_tokens else model.max_tokens // 3),
        "stream": True,
    }

    # System prompt handling.
    system_blocks: list[dict[str, Any]] = []
    if is_oauth_token:
        system_blocks.append(
            {
                "type": "text",
                "text": CLAUDE_CODE_IDENTITY_PROMPT,
                **({"cache_control": cache_control} if cache_control else {}),
            }
        )
        if context.system_prompt:
            system_blocks.append(
                {
                    "type": "text",
                    "text": sanitize_surrogates(context.system_prompt),
                    **({"cache_control": cache_control} if cache_control else {}),
                }
            )
    elif context.system_prompt:
        system_blocks.append(
            {
                "type": "text",
                "text": sanitize_surrogates(context.system_prompt),
                **({"cache_control": cache_control} if cache_control else {}),
            }
        )
    if system_blocks:
        params["system"] = system_blocks

    # Temperature is incompatible with extended thinking (adaptive or budget).
    if options and options.temperature is not None and not options.thinking_enabled:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = convert_tools(context.tools, is_oauth_token=is_oauth_token)

    # Thinking configuration.
    if model.reasoning and options is not None:
        if options.thinking_enabled:
            if supports_adaptive_thinking(model.id):
                params["thinking"] = {"type": "adaptive"}
                if options.effort is not None:
                    params["output_config"] = {"effort": options.effort}
            else:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": options.thinking_budget_tokens or 1024,
                }
        elif options.thinking_enabled is False:
            params["thinking"] = {"type": "disabled"}

    # Metadata passthrough — only the ``user_id`` string field is forwarded.
    if options and options.metadata:
        user_id = options.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    # Tool choice passthrough.
    if options and options.tool_choice is not None:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    return params


# ---------------------------------------------------------------------------
# SDK glue — client construction and async streaming
# ---------------------------------------------------------------------------


# Hold strong references to in-flight background tasks so the event loop
# does not GC them while they are still running.
_background_tasks: set[asyncio.Task[None]] = set()


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
    """Snapshot the working output for partial events.

    The upstream TS provider emits the *same mutable object* as ``partial``
    on every event. Pydantic models are mutable by default so we could do
    the same, but snapshotting keeps Python test assertions stable when a
    consumer stashes ``partial`` for later inspection.
    """
    return output.model_copy(deep=True)


def create_client(
    model: Model,
    api_key: str,
    *,
    interleaved_thinking: bool = True,
    options_headers: dict[str, str] | None = None,
    dynamic_headers: dict[str, str] | None = None,
) -> tuple[Any, bool]:
    """Construct an :class:`anthropic.AsyncAnthropic` client for ``model``.

    Returns ``(client, is_oauth_token)``. Imports the ``anthropic`` SDK
    lazily so unit tests that inject a fake client never pull it in.
    """
    from anthropic import AsyncAnthropic  # noqa: PLC0415 — lazy import

    needs_interleaved_beta = interleaved_thinking and not supports_adaptive_thinking(model.id)

    if model.provider == "github-copilot":
        beta_features: list[str] = []
        if needs_interleaved_beta:
            beta_features.append("interleaved-thinking-2025-05-14")
        headers = merge_headers(
            {
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                **({"anthropic-beta": ",".join(beta_features)} if beta_features else {}),
            },
            model.headers,
            dynamic_headers,
            options_headers,
        )
        client = AsyncAnthropic(
            api_key=None,
            auth_token=api_key,
            base_url=model.base_url,
            default_headers=headers,
        )
        return client, False

    beta_features = ["fine-grained-tool-streaming-2025-05-14"]
    if needs_interleaved_beta:
        beta_features.append("interleaved-thinking-2025-05-14")

    if is_oauth_token(api_key):
        headers = merge_headers(
            {
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                "anthropic-beta": f"claude-code-20250219,oauth-2025-04-20,{','.join(beta_features)}",
                "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                "x-app": "cli",
            },
            model.headers,
            options_headers,
        )
        client = AsyncAnthropic(
            api_key=None,
            auth_token=api_key,
            base_url=model.base_url,
            default_headers=headers,
        )
        return client, True

    headers = merge_headers(
        {
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
            "anthropic-beta": ",".join(beta_features),
        },
        model.headers,
        options_headers,
    )
    client = AsyncAnthropic(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=headers,
    )
    return client, False


def _handle_sdk_event(
    event: Any,
    *,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    partial_json_by_index: dict[int, str],
    index_to_content_pos: dict[int, int],
    model: Model,
    context: Context,
    is_oauth: bool,
) -> None:
    """Process one Anthropic SDK event, mutating ``output`` and pushing nu_ai events."""
    etype = event.type

    if etype == "message_start":
        message = event.message
        output.response_id = getattr(message, "id", None)
        usage = getattr(message, "usage", None)
        if usage is not None:
            output.usage.input = getattr(usage, "input_tokens", 0) or 0
            output.usage.output = getattr(usage, "output_tokens", 0) or 0
            output.usage.cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            output.usage.cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            output.usage.total_tokens = (
                output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
            )
            calculate_cost(model, output.usage)
        return

    if etype == "content_block_start":
        cb = event.content_block
        cb_type = getattr(cb, "type", None)
        if cb_type == "text":
            output.content.append(TextContent(text=""))
            index_to_content_pos[event.index] = len(output.content) - 1
            stream.push(TextStartEvent(content_index=len(output.content) - 1, partial=_copy_output(output)))
        elif cb_type == "thinking":
            output.content.append(ThinkingContent(thinking="", thinking_signature=""))
            index_to_content_pos[event.index] = len(output.content) - 1
            stream.push(ThinkingStartEvent(content_index=len(output.content) - 1, partial=_copy_output(output)))
        elif cb_type == "redacted_thinking":
            output.content.append(
                ThinkingContent(
                    thinking="[Reasoning redacted]",
                    thinking_signature=getattr(cb, "data", ""),
                    redacted=True,
                )
            )
            index_to_content_pos[event.index] = len(output.content) - 1
            stream.push(ThinkingStartEvent(content_index=len(output.content) - 1, partial=_copy_output(output)))
        elif cb_type == "tool_use":
            name = getattr(cb, "name", "")
            if is_oauth:
                name = from_claude_code_name(name, context.tools)
            output.content.append(
                ToolCall(
                    id=getattr(cb, "id", ""),
                    name=name,
                    arguments=getattr(cb, "input", None) or {},
                )
            )
            partial_json_by_index[event.index] = ""
            index_to_content_pos[event.index] = len(output.content) - 1
            stream.push(ToolCallStartEvent(content_index=len(output.content) - 1, partial=_copy_output(output)))
        return

    if etype == "content_block_delta":
        content_pos = index_to_content_pos.get(event.index)
        if content_pos is None:
            return
        block = output.content[content_pos]
        delta = event.delta
        dtype = getattr(delta, "type", None)

        if dtype == "text_delta" and isinstance(block, TextContent):
            block.text += delta.text
            stream.push(
                TextDeltaEvent(
                    content_index=content_pos,
                    delta=delta.text,
                    partial=_copy_output(output),
                )
            )
        elif dtype == "thinking_delta" and isinstance(block, ThinkingContent):
            block.thinking += delta.thinking
            stream.push(
                ThinkingDeltaEvent(
                    content_index=content_pos,
                    delta=delta.thinking,
                    partial=_copy_output(output),
                )
            )
        elif dtype == "input_json_delta" and isinstance(block, ToolCall):
            partial = partial_json_by_index.get(event.index, "") + delta.partial_json
            partial_json_by_index[event.index] = partial
            block.arguments = parse_streaming_json(partial) or {}
            stream.push(
                ToolCallDeltaEvent(
                    content_index=content_pos,
                    delta=delta.partial_json,
                    partial=_copy_output(output),
                )
            )
        elif dtype == "signature_delta" and isinstance(block, ThinkingContent):
            block.thinking_signature = (block.thinking_signature or "") + delta.signature
        return

    if etype == "content_block_stop":
        content_pos = index_to_content_pos.get(event.index)
        if content_pos is None:
            return
        block = output.content[content_pos]
        if isinstance(block, TextContent):
            stream.push(
                TextEndEvent(
                    content_index=content_pos,
                    content=block.text,
                    partial=_copy_output(output),
                )
            )
        elif isinstance(block, ThinkingContent):
            stream.push(
                ThinkingEndEvent(
                    content_index=content_pos,
                    content=block.thinking,
                    partial=_copy_output(output),
                )
            )
        else:
            # ToolCall — exhaustive over AssistantContent.
            partial = partial_json_by_index.pop(event.index, "")
            block.arguments = parse_streaming_json(partial) or {}
            stream.push(
                ToolCallEndEvent(
                    content_index=content_pos,
                    tool_call=block.model_copy(deep=True),
                    partial=_copy_output(output),
                )
            )
        return

    if etype == "message_delta":
        delta = event.delta
        stop_reason = getattr(delta, "stop_reason", None)
        if stop_reason:
            output.stop_reason = map_stop_reason(stop_reason)
        usage = getattr(event, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            if input_tokens is not None:
                output.usage.input = input_tokens
            output_tokens = getattr(usage, "output_tokens", None)
            if output_tokens is not None:
                output.usage.output = output_tokens
            cache_read = getattr(usage, "cache_read_input_tokens", None)
            if cache_read is not None:
                output.usage.cache_read = cache_read
            cache_write = getattr(usage, "cache_creation_input_tokens", None)
            if cache_write is not None:
                output.usage.cache_write = cache_write
            output.usage.total_tokens = (
                output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
            )
            calculate_cost(model, output.usage)


async def _run_anthropic_stream(
    *,
    model: Model,
    context: Context,
    options: AnthropicOptions | None,
    stream: AssistantMessageEventStream,
    output: AssistantMessage,
    client: Any | None,
) -> None:
    try:
        is_oauth: bool
        resolved_client: Any
        if client is not None:
            is_oauth = False
            resolved_client = client
        else:
            api_key = (
                (options.api_key if options and options.api_key else None) or get_env_api_key(model.provider) or ""
            )
            resolved_client, is_oauth = create_client(
                model,
                api_key,
                interleaved_thinking=(options.interleaved_thinking if options else True) is not False,
                options_headers=options.headers if options else None,
            )

        params = build_params(model, context, is_oauth_token=is_oauth, options=options)
        stream.push(StartEvent(partial=_copy_output(output)))

        partial_json_by_index: dict[int, str] = {}
        index_to_content_pos: dict[int, int] = {}

        async with resolved_client.messages.stream(**params) as sdk_stream:
            async for event in sdk_stream:
                _handle_sdk_event(
                    event,
                    output=output,
                    stream=stream,
                    partial_json_by_index=partial_json_by_index,
                    index_to_content_pos=index_to_content_pos,
                    model=model,
                    context=context,
                    is_oauth=is_oauth,
                )

        if output.stop_reason in {"error", "aborted"}:
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
        output.error_message = str(exc)
        stream.push(ErrorEvent(reason="error", error=_copy_output(output)))
        stream.end()


def stream_anthropic(
    model: Model,
    context: Context,
    options: AnthropicOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from Anthropic's Messages API.

    Returns an :class:`AssistantMessageEventStream` that resolves with the
    final :class:`AssistantMessage`. Any upstream error is encoded in the
    returned stream rather than raised — matches the TS
    ``StreamFunction`` contract.

    ``client`` is an out-of-band parameter (not on :class:`AnthropicOptions`
    because the SDK client isn't serializable). It lets callers inject a
    pre-built :class:`anthropic.AsyncAnthropic` (or a fake, for tests).
    """
    stream = AssistantMessageEventStream()
    output = _new_output(model)
    task = asyncio.create_task(
        _run_anthropic_stream(
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


__all__ = [
    "CLAUDE_CODE_IDENTITY_PROMPT",
    "CLAUDE_CODE_VERSION",
    "CacheControlResult",
    "build_params",
    "convert_content_blocks",
    "convert_messages",
    "convert_tools",
    "create_client",
    "from_claude_code_name",
    "get_cache_control",
    "is_oauth_token",
    "map_stop_reason",
    "map_thinking_level_to_effort",
    "merge_headers",
    "normalize_tool_call_id",
    "resolve_cache_retention",
    "stream_anthropic",
    "supports_adaptive_thinking",
    "to_claude_code_name",
]
