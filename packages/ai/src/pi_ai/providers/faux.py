"""Scriptable faux provider for testing.

Direct port of ``packages/ai/src/providers/faux.ts``. Pi's test harness
throughout the upstream repo (``test/suite/``, every regression test) uses
this provider: you register it, push a queue of :class:`AssistantMessage`
objects, and the standard ``stream`` / ``stream_simple`` API replays them
through the streaming pipeline with chunked deltas and estimated usage.

Differences from the TS version:

* ``tokensPerSecond`` scheduling uses :func:`asyncio.sleep` instead of
  ``setTimeout``; zero/``None`` causes an immediate yield via
  ``asyncio.sleep(0)``.
* The random-chunk sizer uses Python's :mod:`random` and is therefore
  deterministic only when the caller seeds the global RNG — same as TS.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, overload

from pi_ai.api_registry import (
    ApiProvider,
    register_api_provider,
    unregister_api_providers,
)
from pi_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    ModelCost,
    StartEvent,
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
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)

if TYPE_CHECKING:
    from pi_ai.types import Context, Message, StopReason, StreamOptions


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_API = "faux"
_DEFAULT_PROVIDER = "faux"
_DEFAULT_MODEL_ID = "faux-1"
_DEFAULT_MODEL_NAME = "Faux Model"
_DEFAULT_BASE_URL = "http://localhost:0"
_DEFAULT_MIN_TOKEN_SIZE = 3
_DEFAULT_MAX_TOKEN_SIZE = 5


def _default_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


# ---------------------------------------------------------------------------
# Model definitions + builders
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FauxModelDefinition:
    """Declarative model definition for :func:`register_faux_provider`."""

    id: str
    name: str | None = None
    reasoning: bool = False
    input: list[str] | None = None
    cost: ModelCost | None = None
    context_window: int = 128_000
    max_tokens: int = 16_384


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    id_: str | None = None,
) -> ToolCall:
    return ToolCall(id=id_ or _random_id("tool"), name=name, arguments=arguments)


FauxContentBlock = TextContent | ThinkingContent | ToolCall


def _normalize_faux_assistant_content(
    content: str | FauxContentBlock | list[FauxContentBlock],
) -> list[FauxContentBlock]:
    if isinstance(content, str):
        return [faux_text(content)]
    if isinstance(content, list):
        return list(content)
    return [content]


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: StopReason = "stop",
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    """Build an :class:`AssistantMessage` in the faux provider's tag space."""
    return AssistantMessage(
        content=_normalize_faux_assistant_content(content),
        api=_DEFAULT_API,
        provider=_DEFAULT_PROVIDER,
        model=_DEFAULT_MODEL_ID,
        usage=_default_usage(),
        stop_reason=stop_reason,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp or int(time.time() * 1000),
    )


# ---------------------------------------------------------------------------
# Response-step protocol
# ---------------------------------------------------------------------------


class FauxResponseFactory(Protocol):
    def __call__(
        self,
        context: Context,
        options: StreamOptions | None,
        state: _CallState,
        model: Model,
        /,
    ) -> AssistantMessage: ...


FauxResponseStep = AssistantMessage | FauxResponseFactory


@dataclass(slots=True)
class _CallState:
    call_count: int = 0


# ---------------------------------------------------------------------------
# Public registration API
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RegisterFauxProviderOptions:
    api: str | None = None
    provider: str | None = None
    models: list[FauxModelDefinition] | None = None
    tokens_per_second: float | None = None
    min_token_size: int = _DEFAULT_MIN_TOKEN_SIZE
    max_token_size: int = _DEFAULT_MAX_TOKEN_SIZE


@dataclass(slots=True)
class FauxProviderRegistration:
    """Handle returned from :func:`register_faux_provider`."""

    api: str
    models: list[Model]
    state: _CallState
    _pending: list[FauxResponseStep]
    _unregister_source_id: str
    _default_model: Model
    _models_by_id: dict[str, Model] = field(default_factory=dict)

    @overload
    def get_model(self) -> Model: ...
    @overload
    def get_model(self, model_id: str) -> Model | None: ...
    def get_model(self, model_id: str | None = None) -> Model | None:
        """Return the default model, or the named model if it exists.

        Calling without arguments always returns a :class:`Model`; calling
        with an explicit ``model_id`` returns ``Model | None`` (``None`` for
        unknown ids).
        """
        if model_id is None:
            return self._default_model
        return self._models_by_id.get(model_id)

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        self._pending.clear()
        self._pending.extend(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        self._pending.extend(responses)

    def get_pending_response_count(self) -> int:
        return len(self._pending)

    def unregister(self) -> None:
        unregister_api_providers(self._unregister_source_id)


def register_faux_provider(
    *,
    api: str | None = None,
    provider: str | None = None,
    models: list[FauxModelDefinition] | None = None,
    tokens_per_second: float | None = None,
    min_token_size: int = _DEFAULT_MIN_TOKEN_SIZE,
    max_token_size: int = _DEFAULT_MAX_TOKEN_SIZE,
) -> FauxProviderRegistration:
    """Register a faux streaming provider and return a handle.

    Every call returns a registration with a **fresh** ``api`` identifier (if
    ``api`` is not explicitly specified) so parallel tests don't collide.
    """
    resolved_api = api or _random_id(_DEFAULT_API)
    resolved_provider = provider or _DEFAULT_PROVIDER
    source_id = _random_id("faux-provider")
    min_size = max(1, min(min_token_size, max_token_size))
    max_size = max(min_size, max_token_size)
    pending: list[FauxResponseStep] = []
    state = _CallState()
    prompt_cache: dict[str, str] = {}

    model_defs = models or [
        FauxModelDefinition(
            id=_DEFAULT_MODEL_ID,
            name=_DEFAULT_MODEL_NAME,
            input=["text", "image"],
        ),
    ]
    faux_models: list[Model] = []
    for definition in model_defs:
        faux_models.append(
            Model(
                id=definition.id,
                name=definition.name or definition.id,
                api=resolved_api,
                provider=resolved_provider,
                base_url=_DEFAULT_BASE_URL,
                reasoning=definition.reasoning,
                input=definition.input or ["text", "image"],  # type: ignore[arg-type]
                cost=definition.cost or ModelCost(input=0, output=0, cache_read=0, cache_write=0),
                context_window=definition.context_window,
                max_tokens=definition.max_tokens,
            )
        )

    def stream_fn(
        request_model: Model,
        context: Context,
        stream_options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        outer = create_assistant_message_event_stream()
        step = pending.pop(0) if pending else None
        state.call_count += 1

        async def _run() -> None:
            try:
                if step is None:
                    error_msg = _create_error_message(
                        RuntimeError("No more faux responses queued"),
                        resolved_api,
                        resolved_provider,
                        request_model.id,
                    )
                    error_msg = _with_usage_estimate(error_msg, context, stream_options, prompt_cache)
                    outer.push(ErrorEvent(reason="error", error=error_msg))
                    outer.end(error_msg)
                    return

                resolved = step(context, stream_options, state, request_model) if callable(step) else step
                message = _clone_message(resolved, resolved_api, resolved_provider, request_model.id)
                message = _with_usage_estimate(message, context, stream_options, prompt_cache)
                await _stream_with_deltas(
                    outer,
                    message,
                    min_token_size=min_size,
                    max_token_size=max_size,
                    tokens_per_second=tokens_per_second,
                )
            except Exception as exc:
                error_msg = _create_error_message(exc, resolved_api, resolved_provider, request_model.id)
                outer.push(ErrorEvent(reason="error", error=error_msg))
                outer.end(error_msg)

        task = asyncio.create_task(_run())
        _background_faux_tasks.add(task)
        task.add_done_callback(_background_faux_tasks.discard)
        return outer

    def stream_simple_fn(
        request_model: Model,
        context: Context,
        stream_options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return stream_fn(request_model, context, stream_options)

    register_api_provider(
        ApiProvider(api=resolved_api, stream=stream_fn, stream_simple=stream_simple_fn),
        source_id=source_id,
    )

    return FauxProviderRegistration(
        api=resolved_api,
        models=faux_models,
        state=state,
        _pending=pending,
        _unregister_source_id=source_id,
        _default_model=faux_models[0],
        _models_by_id={m.id: m for m in faux_models},
    )


# Hold strong references so background faux tasks aren't GC'd mid-flight.
_background_faux_tasks: set[asyncio.Task[None]] = set()


# ---------------------------------------------------------------------------
# Delta streaming
# ---------------------------------------------------------------------------


async def _stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    *,
    min_token_size: int,
    max_token_size: int,
    tokens_per_second: float | None,
) -> None:
    # Start from an empty partial and populate block-by-block.
    partial = message.model_copy(update={"content": []}, deep=True)
    stream.push(StartEvent(partial=partial.model_copy(deep=True)))

    for index, block in enumerate(message.content):
        if isinstance(block, ThinkingContent):
            partial.content.append(ThinkingContent(thinking=""))
            stream.push(ThinkingStartEvent(content_index=index, partial=partial.model_copy(deep=True)))
            for chunk in _split_string_by_token_size(block.thinking, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                _tail_thinking(partial, index).thinking += chunk
                stream.push(
                    ThinkingDeltaEvent(
                        content_index=index,
                        delta=chunk,
                        partial=partial.model_copy(deep=True),
                    )
                )
            stream.push(
                ThinkingEndEvent(
                    content_index=index,
                    content=block.thinking,
                    partial=partial.model_copy(deep=True),
                )
            )
            continue

        if isinstance(block, TextContent):
            partial.content.append(TextContent(text=""))
            stream.push(TextStartEvent(content_index=index, partial=partial.model_copy(deep=True)))
            for chunk in _split_string_by_token_size(block.text, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                _tail_text(partial, index).text += chunk
                stream.push(
                    TextDeltaEvent(
                        content_index=index,
                        delta=chunk,
                        partial=partial.model_copy(deep=True),
                    )
                )
            stream.push(
                TextEndEvent(
                    content_index=index,
                    content=block.text,
                    partial=partial.model_copy(deep=True),
                )
            )
            continue

        # ToolCall
        partial.content.append(ToolCall(id=block.id, name=block.name, arguments={}))
        stream.push(ToolCallStartEvent(content_index=index, partial=partial.model_copy(deep=True)))
        for chunk in _split_string_by_token_size(json.dumps(block.arguments), min_token_size, max_token_size):
            await _schedule_chunk(chunk, tokens_per_second)
            stream.push(
                ToolCallDeltaEvent(
                    content_index=index,
                    delta=chunk,
                    partial=partial.model_copy(deep=True),
                )
            )
        _tail_tool_call(partial, index).arguments = dict(block.arguments)
        stream.push(
            ToolCallEndEvent(
                content_index=index,
                tool_call=block.model_copy(deep=True),
                partial=partial.model_copy(deep=True),
            )
        )

    if message.stop_reason in {"error", "aborted"}:
        stream.push(ErrorEvent(reason=message.stop_reason, error=message))  # type: ignore[arg-type]
        stream.end(message)
        return

    stream.push(DoneEvent(reason=message.stop_reason, message=message))  # type: ignore[arg-type]
    stream.end(message)


def _tail_thinking(partial: AssistantMessage, index: int) -> ThinkingContent:
    block = partial.content[index]
    assert isinstance(block, ThinkingContent)
    return block


def _tail_text(partial: AssistantMessage, index: int) -> TextContent:
    block = partial.content[index]
    assert isinstance(block, TextContent)
    return block


def _tail_tool_call(partial: AssistantMessage, index: int) -> ToolCall:
    block = partial.content[index]
    assert isinstance(block, ToolCall)
    return block


# ---------------------------------------------------------------------------
# Usage / chunk / prompt helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    return -(-len(text) // 4)  # ceil division


def _random_id(prefix: str) -> str:
    return f"{prefix}:{int(time.time() * 1000)}:{uuid.uuid4().hex[:10]}"


def _content_to_text(content: str | list[TextContent | ImageContent]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _assistant_content_to_text(content: list[TextContent | ThinkingContent | ToolCall]) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ThinkingContent):
            parts.append(block.thinking)
        else:
            parts.append(f"{block.name}:{json.dumps(block.arguments)}")
    return "\n".join(parts)


def _tool_result_to_text(msg: ToolResultMessage) -> str:
    parts: list[str] = [msg.tool_name]
    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _message_to_text(message: Message) -> str:
    if isinstance(message, UserMessage):
        return _content_to_text(message.content)
    if isinstance(message, AssistantMessage):
        return _assistant_content_to_text(message.content)
    return _tool_result_to_text(message)


def _serialize_context(context: Context) -> str:
    parts: list[str] = []
    if context.system_prompt:
        parts.append(f"system:{context.system_prompt}")
    for msg in context.messages:
        parts.append(f"{msg.role}:{_message_to_text(msg)}")
    if context.tools:
        parts.append(f"tools:{json.dumps([t.model_dump(by_alias=True, exclude_none=True) for t in context.tools])}")
    return "\n\n".join(parts)


def _common_prefix_length(a: str, b: str) -> int:
    length = min(len(a), len(b))
    i = 0
    while i < length and a[i] == b[i]:
        i += 1
    return i


def _with_usage_estimate(
    message: AssistantMessage,
    context: Context,
    options: StreamOptions | None,
    prompt_cache: dict[str, str],
) -> AssistantMessage:
    prompt_text = _serialize_context(context)
    prompt_tokens = _estimate_tokens(prompt_text)
    output_tokens = _estimate_tokens(_assistant_content_to_text(message.content))
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = options.session_id if options else None

    if session_id and (options is None or options.cache_retention != "none"):
        previous = prompt_cache.get(session_id)
        if previous is not None:
            cached_chars = _common_prefix_length(previous, prompt_text)
            cache_read = _estimate_tokens(previous[:cached_chars])
            cache_write = _estimate_tokens(prompt_text[cached_chars:])
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        prompt_cache[session_id] = prompt_text

    return message.model_copy(
        update={
            "usage": Usage(
                input=input_tokens,
                output=output_tokens,
                cache_read=cache_read,
                cache_write=cache_write,
                total_tokens=input_tokens + output_tokens + cache_read + cache_write,
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            ),
        },
        deep=True,
    )


def _split_string_by_token_size(text: str, min_token_size: int, max_token_size: int) -> list[str]:
    chunks: list[str] = []
    i = 0
    while i < len(text):
        tokens = random.randint(min_token_size, max_token_size)
        char_size = max(1, tokens * 4)
        chunks.append(text[i : i + char_size])
        i += char_size
    return chunks if chunks else [""]


def _clone_message(message: AssistantMessage, api: str, provider: str, model_id: str) -> AssistantMessage:
    cloned = copy.deepcopy(message)
    return cloned.model_copy(
        update={
            "api": api,
            "provider": provider,
            "model": model_id,
            "timestamp": cloned.timestamp or int(time.time() * 1000),
            "usage": cloned.usage or _default_usage(),
        },
    )


def _create_error_message(
    error: Exception | str,
    api: str,
    provider: str,
    model_id: str,
) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model_id,
        usage=_default_usage(),
        stop_reason="error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


async def _schedule_chunk(chunk: str, tokens_per_second: float | None) -> None:
    if tokens_per_second is None or tokens_per_second <= 0:
        await asyncio.sleep(0)
        return
    delay = _estimate_tokens(chunk) / tokens_per_second
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.sleep(delay)


__all__ = [
    "FauxContentBlock",
    "FauxModelDefinition",
    "FauxProviderRegistration",
    "FauxResponseFactory",
    "FauxResponseStep",
    "RegisterFauxProviderOptions",
    "faux_assistant_message",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "register_faux_provider",
]
