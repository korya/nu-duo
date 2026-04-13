"""OpenAI Codex Responses provider — port of ``providers/openai-codex-responses.ts``.

This provider targets the OpenAI Codex/ChatGPT backend API
(``/backend-api/responses``) which uses a different auth flow
(JWT-based) and response format from the standard OpenAI API.
It reuses the shared Responses message/tool conversion from
:mod:`nu_ai.providers.openai_responses` but has its own:

* JWT account-ID extraction
* Retry logic with exponential backoff
* SSE streaming with custom headers
* Text verbosity parameter

The upstream is 929 LoC with WebSocket fallback; this port covers
the SSE path only (~400 LoC) since WebSocket fallback is a
browser-specific optimization.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import uuid
from typing import Any

import httpx

from nu_ai.providers.openai_responses import (
    convert_responses_messages,
    convert_responses_tools,
    map_stop_reason,
)
from nu_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    Usage,
)
from nu_ai.utils.event_stream import AssistantMessageEventStream

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
MAX_RETRIES = 3
BASE_DELAY_MS = 1000
CODEX_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _extract_account_id(api_key: str) -> str | None:
    """Extract account ID from a JWT token's claims."""
    parts = api_key.split(".")
    if len(parts) < 2:
        return None
    try:
        # Decode the payload (add padding)
        payload = parts[1]
        payload += "=" * (4 - len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        auth = claims.get("https://api.openai.com/auth", {})
        return auth.get("account_id") if isinstance(auth, dict) else None
    except Exception:
        return None


def _create_codex_request_id() -> str:
    return f"codex-{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _is_retryable_status(status: int) -> bool:
    return status in {429, 500, 502, 503, 504}


def _is_retryable_error(status: int, error_text: str) -> bool:
    if _is_retryable_status(status):
        return True
    return bool(
        re.search(
            r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
            error_text,
            re.IGNORECASE,
        )
    )


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _build_request_body(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> dict[str, Any]:
    """Build the Codex Responses API request body."""
    messages = convert_responses_messages(model, context)

    body: dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "input": messages,
    }

    if context.system_prompt:
        body["instructions"] = context.system_prompt

    if context.tools:
        body["tools"] = convert_responses_tools(context.tools)
        body["tool_choice"] = "auto"
        body["parallel_tool_calls"] = False

    opts = options or StreamOptions()
    if getattr(opts, "max_tokens", None):
        body["max_output_tokens"] = opts.max_tokens
    if getattr(opts, "temperature", None) is not None:
        body["temperature"] = opts.temperature

    # Reasoning
    if model.reasoning:
        effort = getattr(opts, "reasoning_effort", None)
        summary = getattr(opts, "reasoning_summary", None)
        if effort or summary:
            body["reasoning"] = {
                "effort": effort or "medium",
                "summary": summary or "auto",
            }
            body["include"] = ["reasoning.encrypted_content"]
        else:
            body["reasoning"] = {"effort": "none"}

    # Text verbosity (Codex-specific)
    verbosity = getattr(opts, "text_verbosity", None)
    if verbosity:
        body["text"] = {"verbosity": verbosity}

    # Session-based caching
    session_id = getattr(opts, "session_id", None)
    if session_id:
        body["prompt_cache_key"] = session_id

    return body


def _build_headers(
    model: Model,
    api_key: str,
    account_id: str | None,
    options: StreamOptions | None = None,
) -> dict[str, str]:
    """Build request headers for the Codex SSE endpoint."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {api_key}",
    }
    if account_id:
        headers["Openai-Organization"] = account_id
    # Merge model headers
    if hasattr(model, "headers") and model.headers:
        headers.update(model.headers)
    # Merge option headers
    opts = options or StreamOptions()
    if getattr(opts, "headers", None):
        headers.update(opts.headers)
    return headers


# ---------------------------------------------------------------------------
# Stream functions
# ---------------------------------------------------------------------------


def stream_openai_codex_responses(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream against the OpenAI Codex Responses endpoint with retries."""
    stream = AssistantMessageEventStream()

    async def _run() -> None:
        opts = options or StreamOptions()
        api_key = getattr(opts, "api_key", None) or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        account_id = _extract_account_id(api_key)
        body = _build_request_body(model, context, opts)
        headers = _build_headers(model, api_key, account_id, opts)
        base_url = (model.base_url or DEFAULT_CODEX_BASE_URL).rstrip("/")
        url = f"{base_url}/responses"

        output = AssistantMessage(
            content=[],
            api="openai-codex-responses",
            provider=model.provider,
            model=model.id,
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            ),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            # Retry loop
            last_error: Exception | None = None
            for attempt in range(MAX_RETRIES):
                try:
                    async with httpx.AsyncClient(timeout=120) as client:
                        async with client.stream("POST", url, json=body, headers=headers) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                error_str = error_text.decode("utf-8", errors="replace")
                                if _is_retryable_error(response.status_code, error_str) and attempt < MAX_RETRIES - 1:
                                    delay = BASE_DELAY_MS * (2**attempt) / 1000
                                    await asyncio.sleep(delay)
                                    continue
                                raise ValueError(f"Codex API error {response.status_code}: {error_str[:500]}")

                            stream.push({"type": "start", "partial": output})

                            # Process SSE stream
                            async for line in response.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    event = json.loads(data)
                                except json.JSONDecodeError:
                                    continue

                                event_type = event.get("type", "")

                                if event_type == "response.output_item.added":
                                    item = event.get("item", {})
                                    if item.get("type") == "message":
                                        output.content.append(TextContent(text=""))
                                        stream.push({"type": "content_start"})

                                elif event_type == "response.output_text.delta":
                                    delta = event.get("delta", "")
                                    if delta and output.content:
                                        last_block = output.content[-1]
                                        if isinstance(last_block, TextContent):
                                            last_block.text += delta
                                        stream.push({"type": "text_delta", "delta": delta})

                                elif event_type == "response.function_call_arguments.delta":
                                    # Tool call streaming
                                    delta = event.get("delta", "")
                                    if delta:
                                        stream.push({"type": "toolcall_delta", "delta": delta})

                                elif event_type == "response.completed":
                                    resp = event.get("response", {})
                                    status = resp.get("status", "completed")
                                    output.stop_reason = map_stop_reason(status)

                                    # Extract usage
                                    usage_data = resp.get("usage", {})
                                    if usage_data:
                                        output.usage = Usage(
                                            input=usage_data.get("input_tokens", 0),
                                            output=usage_data.get("output_tokens", 0),
                                            cache_read=usage_data.get("input_tokens_details", {}).get(
                                                "cached_tokens", 0
                                            ),
                                            cache_write=0,
                                            total_tokens=usage_data.get("total_tokens", 0),
                                            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                                        )

                            stream.push({"type": "done", "reason": output.stop_reason, "message": output})
                            stream.end()
                            return

                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES - 1:
                        delay = BASE_DELAY_MS * (2**attempt) / 1000
                        await asyncio.sleep(delay)
                        continue
                    raise

            if last_error:
                raise last_error

        except Exception as exc:
            output.stop_reason = "error"
            output.error_message = str(exc)
            stream.push({"type": "error", "reason": "error", "error": output})
            stream.end()

    asyncio.ensure_future(_run())  # noqa: RUF006
    return stream


def stream_simple_openai_codex_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simple-stream variant for OpenAI Codex Responses."""
    from nu_ai.providers.simple_options import build_base_options

    opts = options or SimpleStreamOptions()
    base = build_base_options(model, opts)
    return stream_openai_codex_responses(model, context, base)


__all__ = [
    "stream_openai_codex_responses",
    "stream_simple_openai_codex_responses",
]
