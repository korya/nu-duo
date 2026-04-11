"""Google Generative AI (Gemini) provider.

Direct port of ``packages/ai/src/providers/google.ts`` combined with
``google-shared.ts``. The upstream monorepo splits out three Google APIs
(``google-generative-ai``, ``google-gemini-cli``, ``google-vertex``); the
Python port currently ships only ``google-generative-ai`` — Vertex AI and
Cloud Code Assist are deferred but follow the same pattern.

Thinking / thought signatures: Gemini's protocol uses ``thought: true`` as
the definitive marker for thinking content, and ``thought_signature`` as an
encrypted context blob that can appear on *any* part type (text,
functionCall). The signature is preserved across streamed deltas via
:func:`retain_thought_signature` and validated as base64 before being
replayed.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
from nu_ai.utils.sanitize_unicode import sanitize_surrogates

if TYPE_CHECKING:
    from nu_ai.types import (
        Context,
        GoogleOptions,
        GoogleThinkingLevel,
        Model,
        SimpleStreamOptions,
        ThinkingBudgets,
        ThinkingLevel,
    )


# ---------------------------------------------------------------------------
# Thought signature helpers
# ---------------------------------------------------------------------------


def is_thinking_part(part: Any) -> bool:
    """Determine whether a streamed Gemini part is a thinking block.

    Protocol note: ``thought_signature`` can appear on any part type — it
    does NOT indicate the part itself is thinking content. Only
    ``thought: true`` does.
    """
    if isinstance(part, dict):
        return part.get("thought") is True
    return getattr(part, "thought", None) is True


def retain_thought_signature(
    existing: str | None,
    incoming: str | None,
) -> str | None:
    """Preserve the last non-empty signature across streamed deltas.

    Some backends emit ``thought_signature`` only on the first delta of a
    part; later deltas omit it. This helper prevents a signature from being
    overwritten with ``None`` within the same streamed block.
    """
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


# Signature validation: Gemini signatures are base64 (TYPE_BYTES).
_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

# Sentinel value telling the Gemini API to skip signature validation on
# unsigned function call parts (e.g. replayed from providers without
# thought signatures).
_SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE_PATTERN.match(signature))


def _resolve_thought_signature(
    is_same_provider_and_model: bool,
    signature: str | None,
) -> str | None:
    """Drop signatures from other models or with invalid base64."""
    if is_same_provider_and_model and _is_valid_thought_signature(signature):
        return signature
    return None


# ---------------------------------------------------------------------------
# Tool call id requirements
# ---------------------------------------------------------------------------


def requires_tool_call_id(model_id: str) -> bool:
    """Whether ``model_id`` requires explicit IDs on function calls/responses.

    Claude and gpt-oss models served via Google's APIs need explicit tool
    call IDs; native Gemini models rely on positional matching.
    """
    return model_id.startswith(("claude-", "gpt-oss-"))


def _get_gemini_major_version(model_id: str) -> int | None:
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    if not match:
        return None
    return int(match.group(1))


def _supports_multimodal_function_response(model_id: str) -> bool:
    major = _get_gemini_major_version(model_id)
    if major is not None:
        return major >= 3
    return True


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------


def is_gemma_4_model(model: Model) -> bool:
    return bool(re.search(r"gemma-?4", model.id.lower()))


def is_gemini_3_pro_model(model: Model) -> bool:
    return bool(re.search(r"gemini-3(?:\.\d+)?-pro", model.id.lower()))


def is_gemini_3_flash_model(model: Model) -> bool:
    return bool(re.search(r"gemini-3(?:\.\d+)?-flash", model.id.lower()))


# ---------------------------------------------------------------------------
# Thinking budget / level helpers
# ---------------------------------------------------------------------------


def get_thinking_level(
    effort: ThinkingLevel,
    model: Model,
) -> GoogleThinkingLevel:
    """Map a nu_ai effort level to Gemini's thinking level.

    Gemini 3 Pro collapses ``minimal``/``low`` → ``LOW`` and ``medium``/``high``
    → ``HIGH``. Gemma 4 collapses the low pair → ``MINIMAL`` and the high
    pair → ``HIGH``. Everything else uses the direct 1:1 mapping.
    """
    if is_gemini_3_pro_model(model):
        if effort in ("minimal", "low"):
            return "LOW"
        return "HIGH"
    if is_gemma_4_model(model):
        if effort in ("minimal", "low"):
            return "MINIMAL"
        return "HIGH"
    mapping: dict[str, GoogleThinkingLevel] = {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }
    return mapping.get(effort, "HIGH")  # type: ignore[return-value]


def get_google_budget(
    model: Model,
    effort: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> int:
    """Return the thinking token budget for ``effort`` on ``model``.

    ``-1`` is Gemini's sentinel for "dynamic budget" (model decides).
    """
    if custom_budgets is not None:
        custom_value = getattr(custom_budgets, effort, None) if effort != "xhigh" else None
        if custom_value is not None:
            return custom_value

    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[effort]  # type: ignore[index]
    if "2.5-flash-lite" in model.id:
        return {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}[effort]  # type: ignore[index]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[effort]  # type: ignore[index]

    return -1


def _get_disabled_thinking_config(model: Model) -> dict[str, Any]:
    """Return the ``thinkingConfig`` for disabling thinking on ``model``.

    Gemini 3 Pro can't fully disable thinking, so use the lowest supported
    level without ``includeThoughts``. Gemini 2.x supports ``thinkingBudget=0``.
    """
    if is_gemini_3_pro_model(model):
        return {"thinking_level": "LOW"}
    if is_gemini_3_flash_model(model):
        return {"thinking_level": "MINIMAL"}
    if is_gemma_4_model(model):
        return {"thinking_level": "MINIMAL"}
    return {"thinking_budget": 0}


# ---------------------------------------------------------------------------
# Stop reason / tool choice mapping
# ---------------------------------------------------------------------------


def map_stop_reason(reason: str | None) -> StopReason:
    """Map a Gemini ``FinishReason`` to nu_ai's :data:`StopReason`.

    Accepts either the string name (``"STOP"``, ``"MAX_TOKENS"``, …) or a
    google-genai enum value that stringifies to the same.
    """
    reason_str = reason if isinstance(reason, str) else str(reason or "")
    reason_str = reason_str.split(".")[-1]  # handle ``FinishReason.STOP``
    if reason_str == "STOP":
        return "stop"
    if reason_str == "MAX_TOKENS":
        return "length"
    return "error"


def map_tool_choice(choice: str) -> str:
    """Map nu_ai tool choice to Gemini ``FunctionCallingConfigMode`` name."""
    if choice == "auto":
        return "AUTO"
    if choice == "none":
        return "NONE"
    if choice == "any":
        return "ANY"
    return "AUTO"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def convert_tools(
    tools: list[Tool],
    *,
    use_parameters: bool = False,
) -> list[dict[str, Any]] | None:
    """Convert nu_ai tools to Gemini function declarations.

    Uses ``parameters_json_schema`` by default (full JSON Schema support).
    Set ``use_parameters=True`` for Cloud Code Assist with Claude, where
    the API translates ``parameters`` into Anthropic's ``input_schema``.
    """
    if not tools:
        return None
    function_declarations: list[dict[str, Any]] = []
    for tool in tools:
        declaration: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
        }
        if use_parameters:
            declaration["parameters"] = tool.parameters
        else:
            declaration["parameters_json_schema"] = tool.parameters
        function_declarations.append(declaration)
    return [{"function_declarations": function_declarations}]


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


_ILLEGAL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _make_tool_call_id_normalizer(model: Model) -> Any:
    def normalize(tool_call_id: str, *_: object) -> str:
        if not requires_tool_call_id(model.id):
            return tool_call_id
        return _ILLEGAL_ID_CHARS.sub("_", tool_call_id)[:64]

    return normalize


def convert_messages(
    model: Model,
    context: Context,
) -> list[dict[str, Any]]:
    """Convert nu_ai messages to the Gemini ``contents`` array.

    Runs the cross-provider :func:`transform_messages` with a Gemini-aware
    tool call id normalizer, then translates each message into the
    ``{role, parts: [...]}`` shape Gemini expects.
    """
    contents: list[dict[str, Any]] = []
    transformed = transform_messages(context.messages, model, _make_tool_call_id_normalizer(model))

    is_gemini_3 = "gemini-3" in model.id.lower()

    for msg in transformed:
        if isinstance(msg, UserMessage):
            _append_user_message(contents, msg, model)
            continue

        if isinstance(msg, AssistantMessage):
            _append_assistant_message(contents, msg, model, is_gemini_3=is_gemini_3)
            continue

        # ToolResultMessage
        _append_tool_result_message(contents, msg, model)

    return contents


def _append_user_message(
    contents: list[dict[str, Any]],
    msg: UserMessage,
    model: Model,
) -> None:
    if isinstance(msg.content, str):
        contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(msg.content)}]})
        return

    parts: list[dict[str, Any]] = []
    for item in msg.content:
        if isinstance(item, TextContent):
            parts.append({"text": sanitize_surrogates(item.text)})
        else:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": item.mime_type,
                        "data": item.data,
                    },
                }
            )

    if "image" not in model.input:
        parts = [p for p in parts if "text" in p]
    if not parts:
        return
    contents.append({"role": "user", "parts": parts})


def _append_assistant_message(
    contents: list[dict[str, Any]],
    msg: AssistantMessage,
    model: Model,
    *,
    is_gemini_3: bool,
) -> None:
    is_same_provider_and_model = msg.provider == model.provider and msg.model == model.id
    parts: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextContent):
            if not block.text.strip():
                continue
            thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.text_signature)
            part: dict[str, Any] = {"text": sanitize_surrogates(block.text)}
            if thought_signature:
                part["thought_signature"] = thought_signature
            parts.append(part)
        elif isinstance(block, ThinkingContent):
            if not block.thinking.strip():
                continue
            if is_same_provider_and_model:
                thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.thinking_signature)
                thinking_part: dict[str, Any] = {
                    "thought": True,
                    "text": sanitize_surrogates(block.thinking),
                }
                if thought_signature:
                    thinking_part["thought_signature"] = thought_signature
                parts.append(thinking_part)
            else:
                # Cross-model: fall back to plain text.
                parts.append({"text": sanitize_surrogates(block.thinking)})
        else:
            # ToolCall
            thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.thought_signature)
            effective_signature = thought_signature or (_SKIP_THOUGHT_SIGNATURE if is_gemini_3 else None)
            function_call: dict[str, Any] = {
                "name": block.name,
                "args": block.arguments or {},
            }
            if requires_tool_call_id(model.id):
                function_call["id"] = block.id
            part_dict: dict[str, Any] = {"function_call": function_call}
            if effective_signature:
                part_dict["thought_signature"] = effective_signature
            parts.append(part_dict)

    if not parts:
        return
    contents.append({"role": "model", "parts": parts})


def _append_tool_result_message(
    contents: list[dict[str, Any]],
    msg: ToolResultMessage,
    model: Model,
) -> None:
    text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
    text_result = "\n".join(text_parts)
    image_blocks: list[ImageContent] = (
        [c for c in msg.content if isinstance(c, ImageContent)] if "image" in model.input else []
    )
    has_text = bool(text_result)
    has_images = bool(image_blocks)

    model_supports_multimodal = _supports_multimodal_function_response(model.id)
    response_value = sanitize_surrogates(text_result) if has_text else ("(see attached image)" if has_images else "")

    image_parts: list[dict[str, Any]] = [
        {
            "inline_data": {
                "mime_type": block.mime_type,
                "data": block.data,
            },
        }
        for block in image_blocks
    ]

    include_id = requires_tool_call_id(model.id)
    function_response: dict[str, Any] = {
        "name": msg.tool_name,
        "response": {"error": response_value} if msg.is_error else {"output": response_value},
    }
    if has_images and model_supports_multimodal:
        function_response["parts"] = image_parts
    if include_id:
        function_response["id"] = msg.tool_call_id
    function_response_part = {"function_response": function_response}

    # Collapse consecutive tool results into a single user turn.
    if (
        contents
        and contents[-1]["role"] == "user"
        and any("function_response" in p for p in contents[-1].get("parts", []))
    ):
        contents[-1]["parts"].append(function_response_part)
    else:
        contents.append({"role": "user", "parts": [function_response_part]})

    # For Gemini < 3, add images as a separate user message.
    if has_images and not model_supports_multimodal:
        contents.append(
            {
                "role": "user",
                "parts": [{"text": "Tool result image:"}, *image_parts],
            }
        )


# ---------------------------------------------------------------------------
# Params builder
# ---------------------------------------------------------------------------


def build_params(
    model: Model,
    context: Context,
    options: GoogleOptions | None = None,
) -> dict[str, Any]:
    """Build the :func:`client.aio.models.generate_content_stream` kwargs."""
    contents = convert_messages(model, context)

    generation_config: dict[str, Any] = {}
    if options is not None:
        if options.temperature is not None:
            generation_config["temperature"] = options.temperature
        if options.max_tokens is not None:
            generation_config["max_output_tokens"] = options.max_tokens

    config: dict[str, Any] = {}
    if generation_config:
        config.update(generation_config)
    if context.system_prompt:
        config["system_instruction"] = sanitize_surrogates(context.system_prompt)
    if context.tools:
        tools_converted = convert_tools(context.tools)
        if tools_converted is not None:
            config["tools"] = tools_converted

    if context.tools and options is not None and options.tool_choice is not None:
        config["tool_config"] = {
            "function_calling_config": {"mode": map_tool_choice(options.tool_choice)},
        }

    if options is not None and options.thinking is not None and model.reasoning:
        if options.thinking.enabled:
            thinking_config: dict[str, Any] = {"include_thoughts": True}
            if options.thinking.level is not None:
                thinking_config["thinking_level"] = options.thinking.level
            elif options.thinking.budget_tokens is not None:
                thinking_config["thinking_budget"] = options.thinking.budget_tokens
            config["thinking_config"] = thinking_config
        else:
            config["thinking_config"] = _get_disabled_thinking_config(model)

    return {"model": model.id, "contents": contents, "config": config}


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
    api_key: str | None = None,
    options_headers: dict[str, str] | None = None,
) -> Any:
    """Construct a :class:`google.genai.Client` for ``model``.

    Imports ``google.genai`` lazily so tests that inject a fake client
    never pull in the SDK.
    """
    from google.genai import Client  # noqa: PLC0415 — lazy import

    http_options: dict[str, Any] = {}
    if model.base_url:
        http_options["base_url"] = model.base_url
        http_options["api_version"] = ""  # base URL already includes version path
    headers = {**(model.headers or {}), **(options_headers or {})}
    if headers:
        http_options["headers"] = headers

    kwargs: dict[str, Any] = {"api_key": api_key}
    if http_options:
        kwargs["http_options"] = http_options
    return Client(**kwargs)


_background_tasks: set[asyncio.Task[None]] = set()


@dataclass(slots=True)
class _GoogleStreamState:
    current_block: TextContent | ThinkingContent | None = None


def _finish_current_block(
    state: _GoogleStreamState,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> None:
    block = state.current_block
    if block is None:
        return
    idx = len(output.content) - 1
    # Find the actual content index (block may have been pushed earlier).
    for i, existing in enumerate(output.content):
        if existing is block:
            idx = i
            break
    if isinstance(block, TextContent):
        stream.push(
            TextEndEvent(
                content_index=idx,
                content=block.text,
                partial=_copy_output(output),
            )
        )
    else:
        stream.push(
            ThinkingEndEvent(
                content_index=idx,
                content=block.thinking,
                partial=_copy_output(output),
            )
        )
    state.current_block = None


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _get_any(obj: Any, *names: str) -> Any:
    """Accessor that tries both snake_case and camelCase attribute names."""
    for name in names:
        value = _get(obj, name)
        if value is not None:
            return value
    return None


def _handle_google_chunk(
    chunk: Any,
    *,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    state: _GoogleStreamState,
    model: Model,
) -> None:
    if chunk is None:
        return

    response_id = _get_any(chunk, "response_id", "responseId")
    if response_id and output.response_id is None:
        output.response_id = response_id

    candidates = _get(chunk, "candidates") or []
    candidate = candidates[0] if candidates else None
    if candidate is not None:
        content = _get(candidate, "content")
        parts = _get(content, "parts") or []
        for part in parts:
            _handle_part(part, output=output, stream=stream, state=state)

        finish_reason = _get_any(candidate, "finish_reason", "finishReason")
        if finish_reason:
            output.stop_reason = map_stop_reason(finish_reason)
            if any(isinstance(b, ToolCall) for b in output.content):
                output.stop_reason = "toolUse"

    usage_metadata = _get_any(chunk, "usage_metadata", "usageMetadata")
    if usage_metadata:
        prompt_tokens = _get_any(usage_metadata, "prompt_token_count", "promptTokenCount") or 0
        cached_tokens = _get_any(usage_metadata, "cached_content_token_count", "cachedContentTokenCount") or 0
        candidates_tokens = _get_any(usage_metadata, "candidates_token_count", "candidatesTokenCount") or 0
        thoughts_tokens = _get_any(usage_metadata, "thoughts_token_count", "thoughtsTokenCount") or 0
        total_tokens = _get_any(usage_metadata, "total_token_count", "totalTokenCount") or 0
        output.usage = Usage(
            input=prompt_tokens - cached_tokens,
            output=candidates_tokens + thoughts_tokens,
            cache_read=cached_tokens,
            cache_write=0,
            total_tokens=total_tokens,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )
        calculate_cost(model, output.usage)


def _handle_part(
    part: Any,
    *,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    state: _GoogleStreamState,
) -> None:
    text = _get(part, "text")
    function_call = _get_any(part, "function_call", "functionCall")
    thought_signature = _get_any(part, "thought_signature", "thoughtSignature")

    if text is not None:
        is_thinking = is_thinking_part(part)
        if (
            state.current_block is None
            or (is_thinking and not isinstance(state.current_block, ThinkingContent))
            or (not is_thinking and not isinstance(state.current_block, TextContent))
        ):
            _finish_current_block(state, output, stream)
            if is_thinking:
                state.current_block = ThinkingContent(thinking="")
                output.content.append(state.current_block)
                stream.push(
                    ThinkingStartEvent(
                        content_index=len(output.content) - 1,
                        partial=_copy_output(output),
                    )
                )
            else:
                state.current_block = TextContent(text="")
                output.content.append(state.current_block)
                stream.push(
                    TextStartEvent(
                        content_index=len(output.content) - 1,
                        partial=_copy_output(output),
                    )
                )

        # ``state.current_block`` is the right type for ``is_thinking`` after
        # the start branch above; the unnecessary-isinstance branch is
        # collapsed into a simple ``if/else`` to keep pyright happy.
        if isinstance(state.current_block, ThinkingContent):
            state.current_block.thinking += text
            state.current_block.thinking_signature = retain_thought_signature(
                state.current_block.thinking_signature, thought_signature
            )
            stream.push(
                ThinkingDeltaEvent(
                    content_index=len(output.content) - 1,
                    delta=text,
                    partial=_copy_output(output),
                )
            )
        else:  # TextContent — exhaustive over the start branches above.
            assert isinstance(state.current_block, TextContent)
            state.current_block.text += text
            state.current_block.text_signature = retain_thought_signature(
                state.current_block.text_signature, thought_signature
            )
            stream.push(
                TextDeltaEvent(
                    content_index=len(output.content) - 1,
                    delta=text,
                    partial=_copy_output(output),
                )
            )

    if function_call is not None:
        _finish_current_block(state, output, stream)

        provided_id = _get(function_call, "id")
        fc_name = _get(function_call, "name") or ""
        fc_args = _get(function_call, "args") or {}

        # Gemini may omit ids, or return duplicate ids across calls. Generate
        # a unique id whenever needed.
        needs_new_id = not provided_id or any(isinstance(b, ToolCall) and b.id == provided_id for b in output.content)
        tool_call_id = f"{fc_name}_{int(time.time() * 1000)}_{id(function_call)}" if needs_new_id else provided_id

        tool_call = ToolCall(
            id=tool_call_id,
            name=fc_name,
            arguments=fc_args if isinstance(fc_args, dict) else {},
            thought_signature=thought_signature if thought_signature else None,
        )
        output.content.append(tool_call)
        idx = len(output.content) - 1
        stream.push(ToolCallStartEvent(content_index=idx, partial=_copy_output(output)))
        stream.push(
            ToolCallDeltaEvent(
                content_index=idx,
                delta=json.dumps(tool_call.arguments),
                partial=_copy_output(output),
            )
        )
        stream.push(
            ToolCallEndEvent(
                content_index=idx,
                tool_call=tool_call.model_copy(deep=True),
                partial=_copy_output(output),
            )
        )


async def _run_google_stream(
    *,
    model: Model,
    context: Context,
    options: GoogleOptions | None,
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
                (options.api_key if options and options.api_key else None) or get_env_api_key(model.provider) or ""
            )
            resolved_client = create_client(
                model,
                api_key=api_key,
                options_headers=options.headers if options else None,
            )

        params = build_params(model, context, options)
        stream.push(StartEvent(partial=_copy_output(output)))

        state = _GoogleStreamState()
        async for chunk in await resolved_client.aio.models.generate_content_stream(**params):
            _handle_google_chunk(
                chunk,
                output=output,
                stream=stream,
                state=state,
                model=model,
            )

        _finish_current_block(state, output, stream)

        if output.stop_reason in {"error", "aborted"}:
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


def stream_google(
    model: Model,
    context: Context,
    options: GoogleOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from Google's Generative AI (Gemini) API."""
    stream = AssistantMessageEventStream()
    output = _new_output(model)
    task = asyncio.create_task(
        _run_google_stream(
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


def stream_simple_google(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Reasoning-aware wrapper over :func:`stream_google`.

    Maps the nu_ai unified reasoning level to Gemini's thinking config:

    * Non-reasoning or no reasoning requested → ``thinking.enabled = False``.
    * Gemini 3 Pro / Flash / Gemma 4 → use ``thinking_level`` (categorical).
    * Gemini 2.x → use ``thinking_budget`` (token budget).
    """
    from nu_ai.types import GoogleOptions as _GoogleOptions  # noqa: PLC0415
    from nu_ai.types import GoogleThinkingOptions  # noqa: PLC0415

    base = build_base_options(model, options)
    if options is None or options.reasoning is None:
        merged = _GoogleOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            thinking=GoogleThinkingOptions(enabled=False),
        )
        return stream_google(model, context, merged, client=client)

    effort = clamp_reasoning(options.reasoning)
    assert effort is not None

    if is_gemini_3_pro_model(model) or is_gemini_3_flash_model(model) or is_gemma_4_model(model):
        merged = _GoogleOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            thinking=GoogleThinkingOptions(
                enabled=True,
                level=get_thinking_level(effort, model),
            ),
        )
    else:
        merged = _GoogleOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            thinking=GoogleThinkingOptions(
                enabled=True,
                budget_tokens=get_google_budget(model, effort, options.thinking_budgets),
            ),
        )
    return stream_google(model, context, merged, client=client)


__all__ = [
    "build_params",
    "convert_messages",
    "convert_tools",
    "create_client",
    "get_google_budget",
    "get_thinking_level",
    "is_gemini_3_flash_model",
    "is_gemini_3_pro_model",
    "is_gemma_4_model",
    "is_thinking_part",
    "map_stop_reason",
    "map_tool_choice",
    "requires_tool_call_id",
    "retain_thought_signature",
    "stream_google",
    "stream_simple_google",
]
