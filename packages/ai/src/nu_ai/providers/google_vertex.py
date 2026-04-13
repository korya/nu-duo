"""Google Cloud Vertex AI provider.

Direct port of ``packages/ai/src/providers/google-vertex.ts``.

Auth modes
----------
* **API key** — pass ``options.api_key`` or set ``GOOGLE_CLOUD_API_KEY``.
  When an API key is present the ``project``/``location`` env vars are
  ignored and the key is used directly.
* **ADC / service account** — omit ``api_key``; pass ``project`` + ``location``
  in options or via ``GOOGLE_CLOUD_PROJECT`` / ``GCLOUD_PROJECT`` and
  ``GOOGLE_CLOUD_LOCATION`` env vars.

The wire format is identical to the standard Google Generative AI (Gemini) API.
Streaming is powered by the ``@google/genai`` SDK on the TS side; here we
delegate to the Python ``google.genai.Client`` which supports the same
``vertexai=True`` constructor parameter.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING, Any, Literal

from nu_ai.models import calculate_cost
from nu_ai.providers.google import (
    _copy_output,
    _finish_current_block,
    _GoogleStreamState,
    _handle_part,
    _new_output,
)
from nu_ai.providers.google_shared import (
    convert_messages,
    convert_tools,
    is_gemini_3_flash_model,
    is_gemini_3_pro_model,
    map_stop_reason,
    map_tool_choice,
)
from nu_ai.providers.simple_options import build_base_options, clamp_reasoning
from nu_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamOptions,
    Usage,
)
from nu_ai.utils.event_stream import AssistantMessageEventStream
from nu_ai.utils.sanitize_unicode import sanitize_surrogates

if TYPE_CHECKING:
    from nu_ai.types import (
        Context,
        GoogleThinkingLevel,
        Model,
        SimpleStreamOptions,
        ThinkingBudgets,
        ThinkingLevel,
    )

_GoogleThinkingLevelLiteral = Literal["THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"]

# ---------------------------------------------------------------------------
# Thinking level mapping
# ---------------------------------------------------------------------------

_THINKING_LEVEL_MAP: dict[str, str] = {
    "THINKING_LEVEL_UNSPECIFIED": "THINKING_LEVEL_UNSPECIFIED",
    "MINIMAL": "MINIMAL",
    "LOW": "LOW",
    "MEDIUM": "MEDIUM",
    "HIGH": "HIGH",
}


# ---------------------------------------------------------------------------
# Options type
# ---------------------------------------------------------------------------


class GoogleVertexThinkingOptions:
    """Nested thinking configuration for :class:`GoogleVertexOptions`."""

    def __init__(
        self,
        *,
        enabled: bool,
        budget_tokens: int | None = None,
        level: GoogleThinkingLevel | None = None,
    ) -> None:
        self.enabled = enabled
        self.budget_tokens = budget_tokens
        self.level = level


class GoogleVertexOptions(StreamOptions):
    """Vertex AI-specific extension of :class:`StreamOptions`.

    Mirrors ``GoogleVertexOptions`` from
    ``packages/ai/src/providers/google-vertex.ts``.
    """

    tool_choice: str | None = None
    """``"auto"``, ``"none"``, or ``"any"``."""

    project: str | None = None
    """Google Cloud project ID.  Falls back to ``GOOGLE_CLOUD_PROJECT``."""

    location: str | None = None
    """Google Cloud region (e.g. ``"us-central1"``).  Falls back to ``GOOGLE_CLOUD_LOCATION``."""

    # Thinking config is stored as flat fields rather than a nested sub-model.
    thinking_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    thinking_level: _GoogleThinkingLevelLiteral | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_api_key(options: GoogleVertexOptions | None) -> str | None:
    raw = (options.api_key if options else None) or os.environ.get("GOOGLE_CLOUD_API_KEY", "")
    api_key = (raw or "").strip()
    if not api_key:
        return None
    # Reject placeholder values like ``<YOUR_API_KEY>``
    if re.match(r"^<[^>]+>$", api_key):
        return None
    return api_key


def _resolve_project(options: GoogleVertexOptions | None) -> str:
    project = (
        (options.project if options else None)
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )
    if not project:
        raise ValueError(
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def _resolve_location(options: GoogleVertexOptions | None) -> str:
    location = (options.location if options else None) or os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not location:
        raise ValueError("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
    return location


def _create_vertex_client(
    model: Model,
    *,
    api_key: str | None,
    project: str | None,
    location: str | None,
    options_headers: dict[str, str] | None = None,
) -> Any:
    """Construct a ``google.genai.Client`` configured for Vertex AI.

    Imports the SDK lazily so tests that inject a fake client never pull it in.
    """
    from google.genai import Client

    http_options: dict[str, Any] = {}
    headers = {**(model.headers or {}), **(options_headers or {})}
    if headers:
        http_options["headers"] = headers

    kwargs: dict[str, Any] = {
        "vertexai": True,
        "api_version": "v1",
    }
    if api_key:
        kwargs["api_key"] = api_key
    else:
        kwargs["project"] = project
        kwargs["location"] = location
    if http_options:
        kwargs["http_options"] = http_options

    return Client(**kwargs)


def _get_disabled_thinking_config(model: Model) -> dict[str, Any]:
    """Return the ``thinkingConfig`` to disable (or minimise) thinking.

    Gemini 3.x Pro cannot be fully disabled; 3.x Flash and earlier get
    ``thinkingBudget=0``.
    """
    if is_gemini_3_pro_model(model.id):
        return {"thinking_level": "LOW"}
    if is_gemini_3_flash_model(model.id):
        return {"thinking_level": "MINIMAL"}
    return {"thinking_budget": 0}


def _get_gemini_3_thinking_level(
    effort: ThinkingLevel,
    model: Model,
) -> GoogleThinkingLevel:
    if is_gemini_3_pro_model(model.id):
        if effort in ("minimal", "low"):
            return "LOW"
        return "HIGH"
    mapping: dict[str, GoogleThinkingLevel] = {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }
    return mapping.get(effort, "HIGH")  # type: ignore[return-value]


def _get_google_budget(
    model: Model,
    effort: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> int:
    if custom_budgets is not None:
        value = getattr(custom_budgets, effort, None)
        if value is not None:
            return value
    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[effort]  # type: ignore[index]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[effort]  # type: ignore[index]
    return -1


def build_vertex_params(
    model: Model,
    context: Context,
    options: GoogleVertexOptions | None = None,
) -> dict[str, Any]:
    """Build the ``generate_content_stream`` kwargs for Vertex AI."""
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

    if options is not None and model.reasoning:
        if options.thinking_enabled:
            thinking_config: dict[str, Any] = {"include_thoughts": True}
            if options.thinking_level is not None:
                thinking_config["thinking_level"] = options.thinking_level
            elif options.thinking_budget_tokens is not None:
                thinking_config["thinking_budget"] = options.thinking_budget_tokens
            config["thinking_config"] = thinking_config
        elif options.thinking_enabled is False:
            config["thinking_config"] = _get_disabled_thinking_config(model)

    return {"model": model.id, "contents": contents, "config": config}


# ---------------------------------------------------------------------------
# Streaming implementation
# ---------------------------------------------------------------------------

_background_tasks: set[asyncio.Task[None]] = set()


def _handle_vertex_chunk(
    chunk: Any,
    *,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    state: _GoogleStreamState,
    model: Model,
) -> None:
    """Process a single ``GenerateContentResponse`` chunk from Vertex AI.

    The Vertex response shape is identical to the standard Gemini API.
    """
    if chunk is None:
        return

    # response_id
    response_id = getattr(chunk, "response_id", None) or (chunk.get("response_id") if isinstance(chunk, dict) else None)
    if response_id and output.response_id is None:
        output.response_id = response_id

    candidates = (
        getattr(chunk, "candidates", None) or (chunk.get("candidates") if isinstance(chunk, dict) else None) or []
    )
    candidate = candidates[0] if candidates else None
    if candidate is not None:
        content = getattr(candidate, "content", None) or (
            candidate.get("content") if isinstance(candidate, dict) else None
        )
        parts = getattr(content, "parts", None) or (content.get("parts") if isinstance(content, dict) else None) or []
        for part in parts:
            _handle_part(part, output=output, stream=stream, state=state)

        finish_reason = getattr(candidate, "finish_reason", None) or (
            candidate.get("finish_reason") or candidate.get("finishReason") if isinstance(candidate, dict) else None
        )
        if finish_reason:
            output.stop_reason = map_stop_reason(str(finish_reason))
            from nu_ai.types import ToolCall as _ToolCall

            if any(isinstance(b, _ToolCall) for b in output.content):
                output.stop_reason = "toolUse"

    usage_metadata = getattr(chunk, "usage_metadata", None) or (
        chunk.get("usage_metadata") or chunk.get("usageMetadata") if isinstance(chunk, dict) else None
    )
    if usage_metadata:

        def _get_u(obj: Any, *names: str) -> int:
            for n in names:
                v = getattr(obj, n, None) if not isinstance(obj, dict) else obj.get(n)
                if v is not None:
                    return int(v)
            return 0

        prompt_tokens = _get_u(usage_metadata, "prompt_token_count", "promptTokenCount")
        cached_tokens = _get_u(usage_metadata, "cached_content_token_count", "cachedContentTokenCount")
        candidates_tokens = _get_u(usage_metadata, "candidates_token_count", "candidatesTokenCount")
        thoughts_tokens = _get_u(usage_metadata, "thoughts_token_count", "thoughtsTokenCount")
        total_tokens = _get_u(usage_metadata, "total_token_count", "totalTokenCount")
        output.usage = Usage(
            input=prompt_tokens - cached_tokens,
            output=candidates_tokens + thoughts_tokens,
            cache_read=cached_tokens,
            cache_write=0,
            total_tokens=total_tokens,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )
        calculate_cost(model, output.usage)


async def _run_vertex_stream(
    *,
    model: Model,
    context: Context,
    options: GoogleVertexOptions | None,
    stream: AssistantMessageEventStream,
    output: AssistantMessage,
    client: Any | None,
) -> None:
    try:
        resolved_client: Any
        if client is not None:
            resolved_client = client
        else:
            api_key = _resolve_api_key(options)
            if api_key:
                resolved_client = _create_vertex_client(
                    model,
                    api_key=api_key,
                    project=None,
                    location=None,
                    options_headers=options.headers if options else None,
                )
            else:
                project = _resolve_project(options)
                location = _resolve_location(options)
                resolved_client = _create_vertex_client(
                    model,
                    api_key=None,
                    project=project,
                    location=location,
                    options_headers=options.headers if options else None,
                )

        params = build_vertex_params(model, context, options)
        stream.push(StartEvent(partial=_copy_output(output)))

        state = _GoogleStreamState()
        async for chunk in await resolved_client.aio.models.generate_content_stream(**params):
            _handle_vertex_chunk(
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


def stream_google_vertex(
    model: Model,
    context: Context,
    options: GoogleVertexOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from Google Cloud Vertex AI."""
    stream = AssistantMessageEventStream()
    output = _new_output(model)
    task = asyncio.create_task(
        _run_vertex_stream(
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


def stream_simple_google_vertex(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    client: Any | None = None,
) -> AssistantMessageEventStream:
    """Reasoning-aware wrapper over :func:`stream_google_vertex`.

    Maps the nu_ai unified reasoning level to Vertex AI's thinking config:

    * No reasoning requested → ``thinking_enabled = False``.
    * Gemini 3 Pro / Flash → use ``thinking_level`` (categorical).
    * Gemini 2.x → use ``thinking_budget`` (token budget).
    """
    base = build_base_options(model, options)

    if options is None or options.reasoning is None:
        vertex_opts = GoogleVertexOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            headers=base.headers,
            thinking_enabled=False,
        )
        return stream_google_vertex(model, context, vertex_opts, client=client)

    effort = clamp_reasoning(options.reasoning)
    assert effort is not None

    if is_gemini_3_pro_model(model.id) or is_gemini_3_flash_model(model.id):
        vertex_opts = GoogleVertexOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            headers=base.headers,
            thinking_enabled=True,
            thinking_level=_get_gemini_3_thinking_level(effort, model),
        )
    else:
        budget = _get_google_budget(model, effort, options.thinking_budgets if options else None)
        vertex_opts = GoogleVertexOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            headers=base.headers,
            thinking_enabled=True,
            thinking_budget_tokens=budget,
        )

    return stream_google_vertex(model, context, vertex_opts, client=client)


__all__ = [
    "GoogleVertexOptions",
    "build_vertex_params",
    "stream_google_vertex",
    "stream_simple_google_vertex",
]
