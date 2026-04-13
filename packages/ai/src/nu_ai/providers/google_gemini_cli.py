"""Google Gemini CLI / Cloud Code Assist provider.

Direct port of ``packages/ai/src/providers/google-gemini-cli.ts``.

This provider targets the Cloud Code Assist API endpoint used by the Gemini
CLI and the "Antigravity" variant.  Unlike the standard Gemini or Vertex
providers it uses **OAuth bearer tokens** (not API keys), the request body
is a ``CloudCodeAssistRequest`` wrapper around the familiar Gemini
``generateContent`` shape, and the response is delivered as a Server-Sent
Events stream that wraps each JSON chunk in a ``{response: ...}`` envelope.

Auth
----
``options.api_key`` is **not** a plain API key — it is a JSON-encoded object
``{"token": "<bearer-token>", "projectId": "<gcp-project>"}`` produced by
the Gemini CLI OAuth flow.

Endpoints
---------
Gemini CLI production: ``https://cloudcode-pa.googleapis.com``
Antigravity sandbox (tried in order on 403/404):
  1. ``https://daily-cloudcode-pa.sandbox.googleapis.com``
  2. ``https://autopush-cloudcode-pa.sandbox.googleapis.com``
  3. ``https://cloudcode-pa.googleapis.com`` (prod fallback)

Retry logic
-----------
* 403/404 → cascade to next endpoint immediately (no delay).
* 429/5xx / network patterns → exponential back-off, up to ``MAX_RETRIES``.
* Empty stream (no content) → short back-off + re-request, up to
  ``MAX_EMPTY_STREAM_RETRIES``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, Any, Literal

import httpx

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
    is_gemini_3_model,
    is_gemini_3_pro_model,
    map_stop_reason_string,
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
    ToolCall,
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
# Endpoints & headers
# ---------------------------------------------------------------------------

_DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
_ANTIGRAVITY_DAILY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
_ANTIGRAVITY_AUTOPUSH_ENDPOINT = "https://autopush-cloudcode-pa.sandbox.googleapis.com"
_ANTIGRAVITY_ENDPOINT_FALLBACKS = (
    _ANTIGRAVITY_DAILY_ENDPOINT,
    _ANTIGRAVITY_AUTOPUSH_ENDPOINT,
    _DEFAULT_ENDPOINT,
)

_GEMINI_CLI_HEADERS: dict[str, str] = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": json.dumps(
        {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    ),
}

_DEFAULT_ANTIGRAVITY_VERSION = "1.21.9"
_CLAUDE_THINKING_BETA_HEADER = "interleaved-thinking-2025-05-14"

_ANTIGRAVITY_SYSTEM_INSTRUCTION = (
    "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team "
    "working on Advanced Agentic Coding."
    "You are pair programming with a USER to solve their coding task. The task may require creating a "
    "new codebase, modifying or debugging an existing codebase, or simply answering a question."
    "**Absolute paths only**"
    "**Proactiveness**"
)

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BASE_DELAY_MS = 1_000
_MAX_EMPTY_STREAM_RETRIES = 2
_EMPTY_STREAM_BASE_DELAY_MS = 500

# ---------------------------------------------------------------------------
# Options type
# ---------------------------------------------------------------------------


class GoogleGeminiCliOptions(StreamOptions):
    """Cloud Code Assist-specific extension of :class:`StreamOptions`.

    Mirrors ``GoogleGeminiCliOptions`` from
    ``packages/ai/src/providers/google-gemini-cli.ts``.

    ``api_key`` is **not** a plain API key — it is a JSON-encoded object
    ``{"token": "<bearer-token>", "projectId": "<gcp-project>"}``.
    """

    tool_choice: str | None = None
    """``"auto"``, ``"none"``, or ``"any"``."""

    thinking_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    thinking_level: _GoogleThinkingLevelLiteral | None = None

    project_id: str | None = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _get_antigravity_headers() -> dict[str, str]:
    import os

    version = os.environ.get("PI_AI_ANTIGRAVITY_VERSION", _DEFAULT_ANTIGRAVITY_VERSION)
    return {"User-Agent": f"antigravity/{version} darwin/arm64"}


def _needs_claude_thinking_beta_header(model: Model) -> bool:
    return model.provider == "google-antigravity" and model.id.startswith("claude-") and model.reasoning


def _is_retryable_error(status: int, error_text: str) -> bool:
    if status in (429, 500, 502, 503, 504):
        return True
    return bool(
        re.search(
            r"resource.?exhausted|rate.?limit|overloaded|service.?unavailable|other.?side.?closed",
            error_text,
            re.IGNORECASE,
        )
    )


def _extract_error_message(error_text: str) -> str:
    """Return just the ``message`` field from a JSON error body, or the raw text."""
    try:
        parsed = json.loads(error_text)
        if isinstance(parsed, dict) and "error" in parsed:
            msg = parsed["error"].get("message")
            if msg:
                return str(msg)
    except (json.JSONDecodeError, AttributeError):
        pass
    return error_text


def extract_retry_delay(error_text: str, headers: httpx.Headers | None = None) -> int | None:
    """Extract the suggested retry delay (ms) from response headers / body.

    Checks (in order):
    1. ``Retry-After`` header (seconds or HTTP-date).
    2. ``X-RateLimit-Reset`` header (Unix seconds).
    3. ``X-RateLimit-Reset-After`` header (seconds).
    4. Body: ``"Your quota will reset after 18h31m10s"``
    5. Body: ``"Please retry in 5s"`` / ``"Please retry in 200ms"``
    6. Body: ``"retryDelay": "34.07s"``

    Returns the delay in milliseconds, always padded by 1 s, or ``None`` if
    nothing is found.
    """

    def _normalize(ms: float) -> int | None:
        return int(ms + 1_000) if ms > 0 else None

    if headers is not None:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                secs = float(retry_after)
                delay = _normalize(secs * 1_000)
                if delay is not None:
                    return delay
            except ValueError:
                pass
            # HTTP-date form
            from email.utils import parsedate_to_datetime

            try:
                dt = parsedate_to_datetime(retry_after)
                diff_ms = (dt.timestamp() - time.time()) * 1_000
                delay = _normalize(diff_ms)
                if delay is not None:
                    return delay
            except Exception:
                pass

        rate_reset = headers.get("x-ratelimit-reset")
        if rate_reset:
            try:
                delay = _normalize((int(rate_reset) * 1_000) - time.time() * 1_000)
                if delay is not None:
                    return delay
            except ValueError:
                pass

        rate_reset_after = headers.get("x-ratelimit-reset-after")
        if rate_reset_after:
            try:
                delay = _normalize(float(rate_reset_after) * 1_000)
                if delay is not None:
                    return delay
            except ValueError:
                pass

    # Body patterns
    duration_m = re.search(
        r"reset after (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s",
        error_text,
        re.IGNORECASE,
    )
    if duration_m:
        h = int(duration_m.group(1) or 0)
        m = int(duration_m.group(2) or 0)
        s = float(duration_m.group(3))
        total_ms = ((h * 60 + m) * 60 + s) * 1_000
        delay = _normalize(total_ms)
        if delay is not None:
            return delay

    retry_in_m = re.search(r"Please retry in ([0-9.]+)(ms|s)", error_text, re.IGNORECASE)
    if retry_in_m:
        value = float(retry_in_m.group(1))
        unit = retry_in_m.group(2).lower()
        ms = value if unit == "ms" else value * 1_000
        delay = _normalize(ms)
        if delay is not None:
            return delay

    retry_delay_m = re.search(r'"retryDelay":\s*"([0-9.]+)(ms|s)"', error_text, re.IGNORECASE)
    if retry_delay_m:
        value = float(retry_delay_m.group(1))
        unit = retry_delay_m.group(2).lower()
        ms = value if unit == "ms" else value * 1_000
        delay = _normalize(ms)
        if delay is not None:
            return delay

    return None


def _get_disabled_thinking_config(model_id: str) -> dict[str, Any]:
    if is_gemini_3_pro_model(model_id):
        return {"thinkingLevel": "LOW"}
    if is_gemini_3_flash_model(model_id):
        return {"thinkingLevel": "MINIMAL"}
    return {"thinkingBudget": 0}


def _get_gemini_cli_thinking_level(
    effort: ThinkingLevel,
    model_id: str,
) -> GoogleThinkingLevel:
    if is_gemini_3_pro_model(model_id):
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


# ---------------------------------------------------------------------------
# Request builder (pure / testable)
# ---------------------------------------------------------------------------


def build_request(
    model: Model,
    context: Context,
    project_id: str,
    options: GoogleGeminiCliOptions | None = None,
    *,
    is_antigravity: bool = False,
) -> dict[str, Any]:
    """Build the Cloud Code Assist request body.

    This is a pure function (no HTTP) so it can be unit-tested directly.
    """
    contents = convert_messages(model, context)

    generation_config: dict[str, Any] = {}
    if options is not None:
        if options.temperature is not None:
            generation_config["temperature"] = options.temperature
        if options.max_tokens is not None:
            generation_config["maxOutputTokens"] = options.max_tokens

    # Thinking config
    if options is not None and model.reasoning:
        if options.thinking_enabled:
            thinking_cfg: dict[str, Any] = {"includeThoughts": True}
            if options.thinking_level is not None:
                thinking_cfg["thinkingLevel"] = options.thinking_level
            elif options.thinking_budget_tokens is not None:
                thinking_cfg["thinkingBudget"] = options.thinking_budget_tokens
            generation_config["thinkingConfig"] = thinking_cfg
        elif options.thinking_enabled is False:
            generation_config["thinkingConfig"] = _get_disabled_thinking_config(model.id)

    request: dict[str, Any] = {"contents": contents}

    if options is not None and options.session_id:
        request["sessionId"] = options.session_id

    if context.system_prompt:
        request["systemInstruction"] = {
            "parts": [{"text": sanitize_surrogates(context.system_prompt)}],
        }

    if generation_config:
        request["generationConfig"] = generation_config

    if context.tools:
        # Claude models on Cloud Code Assist need the legacy ``parameters`` field.
        use_parameters = model.id.startswith("claude-")
        request["tools"] = convert_tools(context.tools, use_parameters=use_parameters)
        if options is not None and options.tool_choice:
            request["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": map_tool_choice(options.tool_choice),
                },
            }

    if is_antigravity:
        existing_parts = request.get("systemInstruction", {}).get("parts", [])
        request["systemInstruction"] = {
            "role": "user",
            "parts": [
                {"text": _ANTIGRAVITY_SYSTEM_INSTRUCTION},
                {"text": f"Please ignore following [ignore]{_ANTIGRAVITY_SYSTEM_INSTRUCTION}[/ignore]"},
                *existing_parts,
            ],
        }

    request_id = f"{'agent' if is_antigravity else 'pi'}-{int(time.time() * 1000)}-{id(model) % 0xFFFFFFFF:08x}"

    body: dict[str, Any] = {
        "project": project_id,
        "model": model.id,
        "request": request,
        "userAgent": "antigravity" if is_antigravity else "pi-coding-agent",
        "requestId": request_id,
    }
    if is_antigravity:
        body["requestType"] = "agent"
    return body


# ---------------------------------------------------------------------------
# SSE chunk processing
# ---------------------------------------------------------------------------


def _process_sse_chunk(
    chunk_data: dict[str, Any],
    *,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    state: _GoogleStreamState,
    model: Model,
) -> bool:
    """Process a single parsed SSE chunk.  Returns ``True`` if content was found."""
    response_data = chunk_data.get("response")
    if not response_data:
        return False

    response_id = response_data.get("responseId")
    if response_id and output.response_id is None:
        output.response_id = response_id

    has_content = False
    candidate = (response_data.get("candidates") or [None])[0]
    if candidate:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            _handle_part(part, output=output, stream=stream, state=state)
            if part.get("text") is not None or part.get("functionCall"):
                has_content = True

        finish_reason = candidate.get("finishReason")
        if finish_reason:
            output.stop_reason = map_stop_reason_string(finish_reason)
            if any(isinstance(b, ToolCall) for b in output.content):
                output.stop_reason = "toolUse"

    usage_metadata = response_data.get("usageMetadata")
    if usage_metadata:
        prompt_tokens = usage_metadata.get("promptTokenCount") or 0
        cache_read = usage_metadata.get("cachedContentTokenCount") or 0
        candidates_tokens = usage_metadata.get("candidatesTokenCount") or 0
        thoughts_tokens = usage_metadata.get("thoughtsTokenCount") or 0
        total_tokens = usage_metadata.get("totalTokenCount") or 0
        output.usage = Usage(
            input=prompt_tokens - cache_read,
            output=candidates_tokens + thoughts_tokens,
            cache_read=cache_read,
            cache_write=0,
            total_tokens=total_tokens,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )
        calculate_cost(model, output.usage)

    return has_content


# ---------------------------------------------------------------------------
# Streaming implementation
# ---------------------------------------------------------------------------

_background_tasks: set[asyncio.Task[None]] = set()


def _reset_output(output: AssistantMessage) -> None:
    output.content = []
    output.usage = Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )
    output.stop_reason = "stop"
    output.error_message = None
    output.response_id = None
    output.timestamp = int(time.time() * 1000)


async def _stream_response(
    response: httpx.Response,
    *,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    state: _GoogleStreamState,
    model: Model,
    started: list[bool],
) -> bool:
    """Drain one SSE response into *output* + *stream*. Returns ``True`` if content was emitted."""
    has_content = False

    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        json_str = line[5:].strip()
        if not json_str:
            continue
        try:
            chunk: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        if not started[0]:
            stream.push(StartEvent(partial=_copy_output(output)))
            started[0] = True

        found = _process_sse_chunk(
            chunk,
            output=output,
            stream=stream,
            state=state,
            model=model,
        )
        if found:
            has_content = True

    return has_content


async def _run_gemini_cli_stream(
    *,
    model: Model,
    context: Context,
    options: GoogleGeminiCliOptions | None,
    stream: AssistantMessageEventStream,
    output: AssistantMessage,
    client: httpx.AsyncClient | None,
) -> None:
    try:
        api_key_raw = options.api_key if options else None
        if not api_key_raw:
            raise ValueError("Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate.")

        try:
            parsed_creds = json.loads(api_key_raw)
            access_token: str = parsed_creds["token"]
            project_id: str = parsed_creds["projectId"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("Invalid Google Cloud Code Assist credentials. Use /login to re-authenticate.") from exc

        if not access_token or not project_id:
            raise ValueError("Missing token or projectId in Google Cloud credentials. Use /login to re-authenticate.")

        is_antigravity = model.provider == "google-antigravity"
        base_url = (model.base_url or "").strip()
        if base_url:
            endpoints: tuple[str, ...] = (base_url,)
        elif is_antigravity:
            endpoints = _ANTIGRAVITY_ENDPOINT_FALLBACKS
        else:
            endpoints = (_DEFAULT_ENDPOINT,)

        request_body = build_request(model, context, project_id, options, is_antigravity=is_antigravity)
        request_body_bytes = json.dumps(request_body).encode()

        provider_headers = _get_antigravity_headers() if is_antigravity else _GEMINI_CLI_HEADERS
        request_headers: dict[str, str] = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **provider_headers,
        }
        if _needs_claude_thinking_beta_header(model):
            request_headers["anthropic-beta"] = _CLAUDE_THINKING_BETA_HEADER
        if options and options.headers:
            request_headers.update(options.headers)

        request_url: str | None = None
        endpoint_index = 0
        last_error: Exception | None = None

        # Owned client if none injected
        _owned_client: httpx.AsyncClient | None = None
        if client is None:
            _owned_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

        active_client = client if client is not None else _owned_client
        assert active_client is not None

        try:
            # --- Fetch with retry + endpoint fallback ---
            response: httpx.Response | None = None
            for attempt in range(_MAX_RETRIES + 1):
                endpoint = endpoints[min(endpoint_index, len(endpoints) - 1)]
                request_url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
                try:
                    r = await active_client.post(
                        request_url,
                        content=request_body_bytes,
                        headers=request_headers,
                    )
                    if r.status_code == 200:
                        response = r
                        break

                    error_text = r.text

                    # 403/404 → try next endpoint immediately
                    if r.status_code in (403, 404) and endpoint_index < len(endpoints) - 1:
                        endpoint_index += 1
                        continue

                    if attempt < _MAX_RETRIES and _is_retryable_error(r.status_code, error_text):
                        if endpoint_index < len(endpoints) - 1:
                            endpoint_index += 1

                        server_delay = extract_retry_delay(error_text, r.headers)
                        delay_ms = server_delay if server_delay is not None else _BASE_DELAY_MS * (2**attempt)

                        max_delay = options.max_retry_delay_ms if options else None
                        if max_delay and max_delay > 0 and server_delay and server_delay > max_delay:
                            secs = (server_delay + 999) // 1000
                            raise RuntimeError(
                                f"Server requested {secs}s retry delay "
                                f"(max: {max_delay // 1000}s). "
                                f"{_extract_error_message(error_text)}"
                            )

                        await asyncio.sleep(delay_ms / 1_000)
                        continue

                    raise RuntimeError(
                        f"Cloud Code Assist API error ({r.status_code}): {_extract_error_message(error_text)}"
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_error = exc
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(_BASE_DELAY_MS * (2**attempt) / 1_000)
                        continue
                    raise

            if response is None:
                raise (last_error or RuntimeError("Failed to get response after retries"))

            # --- Empty-stream retry loop ---
            received_content = False
            current_response = response

            for empty_attempt in range(_MAX_EMPTY_STREAM_RETRIES + 1):
                if empty_attempt > 0:
                    backoff = _EMPTY_STREAM_BASE_DELAY_MS * (2 ** (empty_attempt - 1))
                    await asyncio.sleep(backoff / 1_000)

                    if not request_url:
                        raise RuntimeError("Missing request URL for retry")

                    current_response = await active_client.post(
                        request_url,
                        content=request_body_bytes,
                        headers=request_headers,
                    )
                    if current_response.status_code != 200:
                        err_text = current_response.text
                        raise RuntimeError(f"Cloud Code Assist API error ({current_response.status_code}): {err_text}")

                started: list[bool] = [False]
                state = _GoogleStreamState()
                streamed = await _stream_response(
                    current_response,
                    output=output,
                    stream=stream,
                    state=state,
                    model=model,
                    started=started,
                )
                _finish_current_block(state, output, stream)

                if streamed:
                    received_content = True
                    break

                if empty_attempt < _MAX_EMPTY_STREAM_RETRIES:
                    _reset_output(output)

            if not received_content:
                raise RuntimeError("Cloud Code Assist API returned an empty response")

        finally:
            if _owned_client is not None:
                await _owned_client.aclose()

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


def stream_google_gemini_cli(
    model: Model,
    context: Context,
    options: GoogleGeminiCliOptions | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from the Google Cloud Code Assist (Gemini CLI) API."""
    stream = AssistantMessageEventStream()
    output = _new_output(model)
    task = asyncio.create_task(
        _run_gemini_cli_stream(
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


def stream_simple_google_gemini_cli(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Reasoning-aware wrapper over :func:`stream_google_gemini_cli`.

    Maps the nu_ai unified reasoning level to the Cloud Code Assist thinking
    config:

    * No reasoning → ``thinking_enabled = False``.
    * Gemini 3 models → categorical ``thinking_level``.
    * Gemini 2.x → token ``thinking_budget``.
    """
    api_key = options.api_key if options else None
    if not api_key:
        raise ValueError("Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate.")

    base = build_base_options(model, options, api_key)

    if options is None or options.reasoning is None:
        cli_opts = GoogleGeminiCliOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            headers=base.headers,
            thinking_enabled=False,
        )
        return stream_google_gemini_cli(model, context, cli_opts, client=client)

    effort = clamp_reasoning(options.reasoning)
    assert effort is not None

    if is_gemini_3_model(model.id):
        cli_opts = GoogleGeminiCliOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            headers=base.headers,
            thinking_enabled=True,
            thinking_level=_get_gemini_cli_thinking_level(effort, model.id),
        )
        return stream_google_gemini_cli(model, context, cli_opts, client=client)

    # Gemini 2.x token-budget path (mirrors streamSimpleGoogleGeminiCli)
    default_budgets: dict[str, int] = {
        "minimal": 1024,
        "low": 2048,
        "medium": 8192,
        "high": 16384,
    }
    custom: ThinkingBudgets | None = options.thinking_budgets if options else None
    if custom is not None:
        for lvl in ("minimal", "low", "medium", "high"):
            val = getattr(custom, lvl, None)
            if val is not None:
                default_budgets[lvl] = val
    budgets = default_budgets

    min_output_tokens = 1024
    thinking_budget: int = budgets[effort]
    base_max = base.max_tokens or 0
    max_tokens = min(base_max + thinking_budget, model.max_tokens)
    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    cli_opts = GoogleGeminiCliOptions(
        temperature=base.temperature,
        max_tokens=max_tokens,
        api_key=base.api_key,
        headers=base.headers,
        thinking_enabled=True,
        thinking_budget_tokens=thinking_budget,
    )
    return stream_google_gemini_cli(model, context, cli_opts, client=client)


__all__ = [
    "GoogleGeminiCliOptions",
    "build_request",
    "extract_retry_delay",
    "stream_google_gemini_cli",
    "stream_simple_google_gemini_cli",
]
