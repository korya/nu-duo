"""Proxy stream function — port of ``packages/agent/src/proxy.ts``.

For apps that route LLM calls through a server instead of calling providers
directly.  The server manages auth and proxies requests; it also strips the
``partial`` field from delta events to reduce bandwidth.  This module
reconstructs the partial :class:`~nu_ai.types.AssistantMessage` client-side
from the incoming delta events.

Typical usage::

    from nu_agent_core.proxy import stream_proxy
    from nu_agent_core import Agent

    agent = Agent(
        ...
        stream_fn=lambda model, context, opts: stream_proxy(
            model,
            context,
            ProxyStreamOptions(
                **opts.__dict__,
                auth_token=await get_token(),
                proxy_url="https://genai.example.com",
            ),
        ),
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from nu_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    ErrorEvent,
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
    Usage,
)
from nu_ai.utils.event_stream import EventStream
from nu_ai.utils.json_parse import parse_streaming_json

if TYPE_CHECKING:
    from nu_ai.types import AssistantMessageEvent, Model, StopReason


# ---------------------------------------------------------------------------
# Proxy event types (bandwidth-optimised: no ``partial`` field)
# ---------------------------------------------------------------------------

# These mirror ProxyAssistantMessageEvent in proxy.ts.
# We represent them as plain dicts since they are transient wire types.

ProxyEventType = str  # "start" | "text_start" | ... | "done" | "error"


# ---------------------------------------------------------------------------
# ProxyMessageEventStream
# ---------------------------------------------------------------------------


class ProxyMessageEventStream(EventStream["AssistantMessageEvent", "AssistantMessage"]):
    """EventStream subclass for proxy responses.

    Port of ``ProxyMessageEventStream`` in proxy.ts.
    """

    def __init__(self) -> None:
        super().__init__(
            is_complete=lambda ev: ev.type in ("done", "error"),
            extract_result=lambda ev: (
                ev.message if ev.type == "done" else ev.error  # type: ignore[union-attr]
            ),
        )


# ---------------------------------------------------------------------------
# ProxyStreamOptions
# ---------------------------------------------------------------------------


@dataclass
class ProxyStreamOptions:
    """Options for :func:`stream_proxy`.

    Mirrors ``ProxyStreamOptions`` from proxy.ts.  Implemented as a plain
    dataclass rather than a Pydantic model so it can carry the
    :class:`asyncio.Event` abort signal (Pydantic's ``extra="forbid"``
    would reject it on :class:`~nu_ai.types.SimpleStreamOptions`).
    """

    auth_token: str = ""
    """Bearer token for the proxy server."""

    proxy_url: str = ""
    """Base URL of the proxy server (e.g. ``https://genai.example.com``)."""

    # Standard stream options (mirrors SimpleStreamOptions fields)
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    reasoning: str | None = None
    headers: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    signal: Any | None = None
    """Optional :class:`asyncio.Event`; when set, the proxy request is cancelled."""


# ---------------------------------------------------------------------------
# Public stream_proxy function
# ---------------------------------------------------------------------------


def stream_proxy(
    model: Model[Any],
    context: Any,
    options: ProxyStreamOptions,
) -> ProxyMessageEventStream:
    """Stream via a proxy server; returns a :class:`ProxyMessageEventStream`.

    Port of ``streamProxy`` (proxy.ts).  Uses ``httpx`` (sync streaming in a
    thread via :func:`asyncio.to_thread` is not needed here because
    :class:`~nu_ai.utils.event_stream.EventStream` feeds events from a
    background asyncio task).
    """
    stream = ProxyMessageEventStream()

    import asyncio  # noqa: PLC0415

    asyncio.get_event_loop().create_task(_drive_proxy(model, context, options, stream))

    return stream


async def _drive_proxy(
    model: Model[Any],
    context: Any,
    options: ProxyStreamOptions,
    stream: ProxyMessageEventStream,
) -> None:
    """Background coroutine that drives the SSE proxy stream."""
    partial = _make_partial(model)
    # Tracks partial JSON strings for each in-progress tool call, keyed by content index.
    # We cannot attach extra state to Pydantic models (extra="forbid"), so we use a side dict.
    partial_tool_json: dict[int, str] = {}

    try:
        payload = {
            "model": _model_to_dict(model),
            "context": _context_to_dict(context),
            "options": {
                "temperature": options.temperature,
                "maxTokens": options.max_tokens,
                "reasoning": options.reasoning,
            },
        }

        async with httpx.AsyncClient() as client, client.stream(
            "POST",
            f"{options.proxy_url}/api/stream",
            headers={
                "Authorization": f"Bearer {options.auth_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=None,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                msg = f"Proxy error: {response.status_code} {response.reason_phrase}"
                try:
                    data = json.loads(body)
                    if isinstance(data, dict) and data.get("error"):
                        msg = f"Proxy error: {data['error']}"
                except Exception:
                    pass
                raise RuntimeError(msg)

            buffer = ""
            async for chunk in response.aiter_text():
                if options.signal is not None and options.signal.is_set():
                    raise RuntimeError("Request aborted by user")

                buffer += chunk
                *lines, buffer = buffer.split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str:
                            proxy_event: dict[str, Any] = json.loads(data_str)
                            event = _process_proxy_event(proxy_event, partial, partial_tool_json)
                            if event is not None:
                                stream.push(event)

        if options.signal is not None and options.signal.is_set():
            raise RuntimeError("Request aborted by user")

        stream.end()

    except Exception as exc:
        error_msg = str(exc)
        reason: StopReason = "aborted" if (options.signal is not None and options.signal.is_set()) else "error"
        partial.stop_reason = reason
        partial.error_message = error_msg
        stream.push({"type": "error", "reason": reason, "error": partial})  # type: ignore[arg-type]
        stream.end()


# ---------------------------------------------------------------------------
# Event processor — port of processProxyEvent()
# ---------------------------------------------------------------------------


def _process_proxy_event(
    proxy_event: dict[str, Any],
    partial: AssistantMessage,
    partial_tool_json: dict[int, str],
) -> AssistantMessageEvent | None:
    """Build a Pydantic AssistantMessageEvent from a raw proxy wire event."""
    etype = proxy_event.get("type")
    content = partial.content  # mutable list

    if etype == "start":
        return StartEvent(partial=partial)

    if etype == "text_start":
        idx: int = proxy_event["contentIndex"]
        content.insert(idx, TextContent(type="text", text=""))
        return TextStartEvent(content_index=idx, partial=partial)

    if etype == "text_delta":
        idx = proxy_event["contentIndex"]
        block = content[idx]
        if isinstance(block, TextContent):
            block.text += proxy_event["delta"]
            return TextDeltaEvent(content_index=idx, delta=proxy_event["delta"], partial=partial)
        raise RuntimeError("Received text_delta for non-text content")

    if etype == "text_end":
        idx = proxy_event["contentIndex"]
        block = content[idx]
        if isinstance(block, TextContent):
            block.text_signature = proxy_event.get("contentSignature")
            return TextEndEvent(content_index=idx, content=block.text, partial=partial)
        raise RuntimeError("Received text_end for non-text content")

    if etype == "thinking_start":
        idx = proxy_event["contentIndex"]
        content.insert(idx, ThinkingContent(type="thinking", thinking=""))
        return ThinkingStartEvent(content_index=idx, partial=partial)

    if etype == "thinking_delta":
        idx = proxy_event["contentIndex"]
        block = content[idx]
        if isinstance(block, ThinkingContent):
            block.thinking += proxy_event["delta"]
            return ThinkingDeltaEvent(content_index=idx, delta=proxy_event["delta"], partial=partial)
        raise RuntimeError("Received thinking_delta for non-thinking content")

    if etype == "thinking_end":
        idx = proxy_event["contentIndex"]
        block = content[idx]
        if isinstance(block, ThinkingContent):
            block.thinking_signature = proxy_event.get("contentSignature")
            return ThinkingEndEvent(content_index=idx, content=block.thinking, partial=partial)
        raise RuntimeError("Received thinking_end for non-thinking content")

    if etype == "toolcall_start":
        idx = proxy_event["contentIndex"]
        tc = ToolCall(type="toolCall", id=proxy_event["id"], name=proxy_event["toolName"], arguments={})
        partial_tool_json[idx] = ""
        content.insert(idx, tc)
        return ToolCallStartEvent(content_index=idx, partial=partial)

    if etype == "toolcall_delta":
        idx = proxy_event["contentIndex"]
        block = content[idx]
        if isinstance(block, ToolCall):
            partial_tool_json[idx] = partial_tool_json.get(idx, "") + proxy_event["delta"]
            block.arguments = parse_streaming_json(partial_tool_json[idx]) or {}
            return ToolCallDeltaEvent(content_index=idx, delta=proxy_event["delta"], partial=partial)
        raise RuntimeError("Received toolcall_delta for non-toolCall content")

    if etype == "toolcall_end":
        idx = proxy_event["contentIndex"]
        block = content[idx]
        partial_tool_json.pop(idx, None)
        if isinstance(block, ToolCall):
            return ToolCallEndEvent(content_index=idx, tool_call=block, partial=partial)
        return None

    if etype == "done":
        partial.stop_reason = proxy_event["reason"]
        partial.usage = _parse_usage(proxy_event.get("usage", {}))
        return DoneEvent(reason=proxy_event["reason"], message=partial)

    if etype == "error":
        partial.stop_reason = proxy_event["reason"]
        partial.error_message = proxy_event.get("errorMessage")
        partial.usage = _parse_usage(proxy_event.get("usage", {}))
        return ErrorEvent(reason=proxy_event["reason"], error=partial)

    import warnings  # noqa: PLC0415

    warnings.warn(f"Unhandled proxy event type: {etype}", stacklevel=2)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_partial(model: Model[Any]) -> AssistantMessage:
    zero_cost = Cost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0, total=0.0)
    return AssistantMessage(
        role="assistant",
        stop_reason="stop",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=zero_cost,
        ),
        timestamp=int(time.time() * 1000),
    )


def _parse_usage(raw: dict[str, Any]) -> Usage:
    zero_cost = Cost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0, total=0.0)
    return Usage(
        input=raw.get("input", 0),
        output=raw.get("output", 0),
        cache_read=raw.get("cacheRead", 0),
        cache_write=raw.get("cacheWrite", 0),
        total_tokens=raw.get("totalTokens", 0),
        cost=zero_cost,
    )


def _model_to_dict(model: Model[Any]) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return {
        "id": model.id,
        "provider": model.provider,
        "api": model.api,
    }


def _context_to_dict(context: Any) -> dict[str, Any]:
    if hasattr(context, "model_dump"):
        return context.model_dump()
    # Fallback: serialize messages + tools manually
    result: dict[str, Any] = {}
    if hasattr(context, "system_prompt"):
        result["systemPrompt"] = context.system_prompt
    if hasattr(context, "messages"):
        msgs = []
        for m in context.messages:
            msgs.append(m.model_dump() if hasattr(m, "model_dump") else dict(m))
        result["messages"] = msgs
    if hasattr(context, "tools") and context.tools:
        tools = []
        for t in context.tools:
            tools.append(t.model_dump() if hasattr(t, "model_dump") else dict(t))
        result["tools"] = tools
    return result


__all__ = [
    "ProxyMessageEventStream",
    "ProxyStreamOptions",
    "stream_proxy",
]
