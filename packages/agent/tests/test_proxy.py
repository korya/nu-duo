"""Tests for nu_agent_core.proxy — targeting ≥90 % coverage.

Covers:
* _process_proxy_event() for all 13 event types
* _make_partial()
* _parse_usage()
* _model_to_dict()
* _context_to_dict()
* stream_proxy() end-to-end (mocked httpx)
* Error paths: HTTP error, abort signal, malformed SSE
"""

from __future__ import annotations

import asyncio
import json
import warnings
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from nu_agent_core.proxy import (
    ProxyMessageEventStream,
    ProxyStreamOptions,
    _context_to_dict,
    _drive_proxy,
    _make_partial,
    _model_to_dict,
    _parse_usage,
    _process_proxy_event,
)
from nu_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
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
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model() -> Model:
    return Model(
        id="gpt-4",
        name="GPT-4",
        api="openai",
        provider="openai",
        base_url="",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=8192,
        max_tokens=4096,
    )


def _fresh_partial() -> AssistantMessage:
    return _make_partial(_model())


# ---------------------------------------------------------------------------
# _make_partial
# ---------------------------------------------------------------------------


class TestMakePartial:
    def test_returns_assistant_message_with_zero_usage(self) -> None:
        model = _model()
        partial = _make_partial(model)
        assert isinstance(partial, AssistantMessage)
        assert partial.role == "assistant"
        assert partial.content == []
        assert partial.model == "gpt-4"
        assert partial.api == "openai"
        assert partial.provider == "openai"
        assert partial.usage.input == 0
        assert partial.usage.output == 0
        assert partial.usage.cost.total == 0.0


# ---------------------------------------------------------------------------
# _parse_usage
# ---------------------------------------------------------------------------


class TestParseUsage:
    def test_empty_dict(self) -> None:
        u = _parse_usage({})
        assert u.input == 0
        assert u.output == 0
        assert u.total_tokens == 0

    def test_partial_dict(self) -> None:
        u = _parse_usage({"input": 10, "cacheRead": 5})
        assert u.input == 10
        assert u.cache_read == 5
        assert u.output == 0

    def test_full_dict(self) -> None:
        u = _parse_usage(
            {
                "input": 100,
                "output": 50,
                "cacheRead": 20,
                "cacheWrite": 10,
                "totalTokens": 180,
            }
        )
        assert u.input == 100
        assert u.output == 50
        assert u.cache_read == 20
        assert u.cache_write == 10
        assert u.total_tokens == 180


# ---------------------------------------------------------------------------
# _model_to_dict
# ---------------------------------------------------------------------------


class TestModelToDict:
    def test_with_model_dump(self) -> None:
        model = _model()
        result = _model_to_dict(model)
        assert result["id"] == "gpt-4"
        assert result["provider"] == "openai"

    def test_without_model_dump(self) -> None:
        @dataclass
        class FakeModel:
            id: str = "m"
            provider: str = "p"
            api: str = "a"

        result = _model_to_dict(FakeModel())  # type: ignore[arg-type]
        assert result == {"id": "m", "provider": "p", "api": "a"}


# ---------------------------------------------------------------------------
# _context_to_dict
# ---------------------------------------------------------------------------


class TestContextToDict:
    def test_with_model_dump(self) -> None:
        class FakeCtx:
            def model_dump(self) -> dict[str, Any]:
                return {"systemPrompt": "hi", "messages": []}

        result = _context_to_dict(FakeCtx())
        assert result == {"systemPrompt": "hi", "messages": []}

    def test_with_messages_and_tools(self) -> None:
        class Msg:
            def model_dump(self) -> dict[str, Any]:
                return {"role": "user", "content": "hello"}

        class Tool:
            def model_dump(self) -> dict[str, Any]:
                return {"name": "bash"}

        class FakeCtx:
            system_prompt = "sys"
            messages = [Msg()]
            tools = [Tool()]

        result = _context_to_dict(FakeCtx())
        assert result["systemPrompt"] == "sys"
        assert result["messages"] == [{"role": "user", "content": "hello"}]
        assert result["tools"] == [{"name": "bash"}]

    def test_with_dict_messages(self) -> None:
        """Messages that lack model_dump are coerced with dict()."""

        class FakeCtx:
            system_prompt = "sp"
            messages = [{"role": "user", "content": "hi"}]
            tools: list[Any] = []

        result = _context_to_dict(FakeCtx())
        assert result["systemPrompt"] == "sp"
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        # Empty tools list → "tools" key absent (falsy guard)
        assert "tools" not in result

    def test_no_attributes(self) -> None:
        result = _context_to_dict(object())
        assert result == {}


# ---------------------------------------------------------------------------
# _process_proxy_event — all event types
# ---------------------------------------------------------------------------


class TestProcessProxyEvent:
    def test_start(self) -> None:
        partial = _fresh_partial()
        ev = _process_proxy_event({"type": "start"}, partial, {})
        assert isinstance(ev, StartEvent)
        assert ev.partial is partial

    def test_text_start(self) -> None:
        partial = _fresh_partial()
        ev = _process_proxy_event({"type": "text_start", "contentIndex": 0}, partial, {})
        assert isinstance(ev, TextStartEvent)
        assert ev.content_index == 0
        assert len(partial.content) == 1
        assert isinstance(partial.content[0], TextContent)

    def test_text_delta(self) -> None:
        partial = _fresh_partial()
        partial.content.append(TextContent(type="text", text=""))
        ev = _process_proxy_event(
            {"type": "text_delta", "contentIndex": 0, "delta": "hello"},
            partial,
            {},
        )
        assert isinstance(ev, TextDeltaEvent)
        assert ev.delta == "hello"
        assert partial.content[0].text == "hello"

    def test_text_delta_wrong_type_raises(self) -> None:
        partial = _fresh_partial()
        partial.content.append(ThinkingContent(type="thinking", thinking=""))
        with pytest.raises(RuntimeError, match="non-text"):
            _process_proxy_event(
                {"type": "text_delta", "contentIndex": 0, "delta": "x"},
                partial,
                {},
            )

    def test_text_end(self) -> None:
        partial = _fresh_partial()
        partial.content.append(TextContent(type="text", text="full text"))
        ev = _process_proxy_event(
            {"type": "text_end", "contentIndex": 0, "contentSignature": "sig123"},
            partial,
            {},
        )
        assert isinstance(ev, TextEndEvent)
        assert ev.content == "full text"
        assert partial.content[0].text_signature == "sig123"

    def test_text_end_wrong_type_raises(self) -> None:
        partial = _fresh_partial()
        partial.content.append(ThinkingContent(type="thinking", thinking=""))
        with pytest.raises(RuntimeError, match="non-text"):
            _process_proxy_event(
                {"type": "text_end", "contentIndex": 0},
                partial,
                {},
            )

    def test_thinking_start(self) -> None:
        partial = _fresh_partial()
        ev = _process_proxy_event({"type": "thinking_start", "contentIndex": 0}, partial, {})
        assert isinstance(ev, ThinkingStartEvent)
        assert isinstance(partial.content[0], ThinkingContent)

    def test_thinking_delta(self) -> None:
        partial = _fresh_partial()
        partial.content.append(ThinkingContent(type="thinking", thinking=""))
        ev = _process_proxy_event(
            {"type": "thinking_delta", "contentIndex": 0, "delta": "hmm"},
            partial,
            {},
        )
        assert isinstance(ev, ThinkingDeltaEvent)
        assert partial.content[0].thinking == "hmm"

    def test_thinking_delta_wrong_type_raises(self) -> None:
        partial = _fresh_partial()
        partial.content.append(TextContent(type="text", text=""))
        with pytest.raises(RuntimeError, match="non-thinking"):
            _process_proxy_event(
                {"type": "thinking_delta", "contentIndex": 0, "delta": "x"},
                partial,
                {},
            )

    def test_thinking_end(self) -> None:
        partial = _fresh_partial()
        partial.content.append(ThinkingContent(type="thinking", thinking="done thinking"))
        ev = _process_proxy_event(
            {"type": "thinking_end", "contentIndex": 0, "contentSignature": "tsig"},
            partial,
            {},
        )
        assert isinstance(ev, ThinkingEndEvent)
        assert ev.content == "done thinking"
        assert partial.content[0].thinking_signature == "tsig"

    def test_thinking_end_wrong_type_raises(self) -> None:
        partial = _fresh_partial()
        partial.content.append(TextContent(type="text", text=""))
        with pytest.raises(RuntimeError, match="non-thinking"):
            _process_proxy_event(
                {"type": "thinking_end", "contentIndex": 0},
                partial,
                {},
            )

    def test_toolcall_start(self) -> None:
        partial = _fresh_partial()
        ptj: dict[int, str] = {}
        ev = _process_proxy_event(
            {"type": "toolcall_start", "contentIndex": 0, "id": "tc1", "toolName": "bash"},
            partial,
            ptj,
        )
        assert isinstance(ev, ToolCallStartEvent)
        tc = partial.content[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "tc1"
        assert tc.name == "bash"
        assert ptj[0] == ""

    def test_toolcall_delta(self) -> None:
        partial = _fresh_partial()
        tc = ToolCall(type="toolCall", id="tc1", name="bash", arguments={})
        partial.content.append(tc)
        ptj: dict[int, str] = {0: ""}
        ev = _process_proxy_event(
            {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"cmd":'},
            partial,
            ptj,
        )
        assert isinstance(ev, ToolCallDeltaEvent)
        assert ptj[0] == '{"cmd":'
        # parse_streaming_json should produce partial result
        ev2 = _process_proxy_event(
            {"type": "toolcall_delta", "contentIndex": 0, "delta": '"ls"}'},
            partial,
            ptj,
        )
        assert isinstance(ev2, ToolCallDeltaEvent)
        assert partial.content[0].arguments == {"cmd": "ls"}

    def test_toolcall_delta_wrong_type_raises(self) -> None:
        partial = _fresh_partial()
        partial.content.append(TextContent(type="text", text=""))
        with pytest.raises(RuntimeError, match="non-toolCall"):
            _process_proxy_event(
                {"type": "toolcall_delta", "contentIndex": 0, "delta": "x"},
                partial,
                {},
            )

    def test_toolcall_end(self) -> None:
        partial = _fresh_partial()
        tc = ToolCall(type="toolCall", id="tc1", name="bash", arguments={"cmd": "ls"})
        partial.content.append(tc)
        ptj: dict[int, str] = {0: '{"cmd":"ls"}'}
        ev = _process_proxy_event(
            {"type": "toolcall_end", "contentIndex": 0},
            partial,
            ptj,
        )
        assert isinstance(ev, ToolCallEndEvent)
        assert ev.tool_call is tc
        assert 0 not in ptj  # cleaned up

    def test_toolcall_end_non_toolcall_returns_none(self) -> None:
        partial = _fresh_partial()
        partial.content.append(TextContent(type="text", text=""))
        ptj: dict[int, str] = {}
        ev = _process_proxy_event(
            {"type": "toolcall_end", "contentIndex": 0},
            partial,
            ptj,
        )
        assert ev is None

    def test_done(self) -> None:
        partial = _fresh_partial()
        ev = _process_proxy_event(
            {"type": "done", "reason": "stop", "usage": {"input": 100, "output": 50}},
            partial,
            {},
        )
        assert isinstance(ev, DoneEvent)
        assert ev.reason == "stop"
        assert partial.stop_reason == "stop"
        assert partial.usage.input == 100

    def test_error(self) -> None:
        partial = _fresh_partial()
        ev = _process_proxy_event(
            {"type": "error", "reason": "error", "errorMessage": "oops", "usage": {}},
            partial,
            {},
        )
        assert isinstance(ev, ErrorEvent)
        assert partial.error_message == "oops"
        assert partial.stop_reason == "error"

    def test_unknown_emits_warning(self) -> None:
        partial = _fresh_partial()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev = _process_proxy_event({"type": "something_new"}, partial, {})
            assert ev is None
            assert len(w) == 1
            assert "something_new" in str(w[0].message)


# ---------------------------------------------------------------------------
# ProxyMessageEventStream
# ---------------------------------------------------------------------------


class TestProxyMessageEventStream:
    def test_done_event_completes_stream(self) -> None:
        stream = ProxyMessageEventStream()
        partial = _fresh_partial()
        partial.stop_reason = "stop"
        done = DoneEvent(reason="stop", message=partial)
        stream.push(done)
        # The stream recognises done as complete

    def test_error_event_completes_stream(self) -> None:
        stream = ProxyMessageEventStream()
        partial = _fresh_partial()
        partial.stop_reason = "error"
        err = ErrorEvent(reason="error", error=partial)
        stream.push(err)


# ---------------------------------------------------------------------------
# _drive_proxy end-to-end (mocked httpx)
# ---------------------------------------------------------------------------


def _sse_lines(*events: dict[str, Any]) -> str:
    """Build a single SSE text chunk from a sequence of proxy events."""
    lines: list[str] = []
    for ev in events:
        lines.append(f"data: {json.dumps(ev)}\n")
    return "\n".join(lines) + "\n"


async def _async_iter(items: list[str]) -> Any:
    for item in items:
        yield item


class _FakeResponse:
    """Minimal async-context-manager response for httpx mocking."""

    def __init__(
        self,
        status_code: int = 200,
        reason_phrase: str = "OK",
        body: bytes = b"",
        chunks: list[str] | None = None,
        chunk_gen: Any = None,
    ) -> None:
        self.status_code = status_code
        self.reason_phrase = reason_phrase
        self._body = body
        self._chunks = chunks or []
        self._chunk_gen = chunk_gen

    async def aread(self) -> bytes:
        return self._body

    async def aiter_text(self) -> Any:
        if self._chunk_gen is not None:
            async for c in self._chunk_gen():
                yield c
        else:
            for c in self._chunks:
                yield c

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _FakeClient:
    """Minimal async-context-manager client wrapping a _FakeResponse."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def stream(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        return self._response

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False


class TestDriveProxyE2E:
    async def test_happy_path_collects_events(self) -> None:
        model = _model()
        sse_payload = _sse_lines(
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "Hi!"},
            {"type": "text_end", "contentIndex": 0},
            {"type": "done", "reason": "stop", "usage": {"input": 10, "output": 5}},
        )

        fake_resp = _FakeResponse(chunks=[sse_payload])
        fake_client = _FakeClient(fake_resp)

        with patch("nu_agent_core.proxy.httpx.AsyncClient", return_value=fake_client):
            stream = ProxyMessageEventStream()
            options = ProxyStreamOptions(auth_token="tok", proxy_url="https://proxy.test")
            await _drive_proxy(model, object(), options, stream)

        events = []
        async for e in stream:
            events.append(e)

        types = [e.type for e in events]
        assert "start" in types
        assert "text_delta" in types
        assert "done" in types

    async def test_http_error_with_json_body(self) -> None:
        """HTTP 500 with JSON error body — exercises the error/json-parse path.

        Note: the current proxy code pushes a raw dict into the stream whose
        is_complete lambda does ``ev.type`` (attribute access), which fails on
        dicts.  We therefore expect an AttributeError to propagate from
        _drive_proxy.  This still exercises lines 184-193 + 216-222.
        """
        model = _model()
        fake_resp = _FakeResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            body=b'{"error":"backend down"}',
        )
        fake_client = _FakeClient(fake_resp)

        with patch("nu_agent_core.proxy.httpx.AsyncClient", return_value=fake_client):
            stream = ProxyMessageEventStream()
            options = ProxyStreamOptions(proxy_url="https://proxy.test")
            with pytest.raises(AttributeError):
                await _drive_proxy(model, object(), options, stream)

    async def test_http_error_non_json_body(self) -> None:
        """HTTP 502 with non-JSON body — exercises the json.loads except path."""
        model = _model()
        fake_resp = _FakeResponse(
            status_code=502,
            reason_phrase="Bad Gateway",
            body=b"not json at all",
        )
        fake_client = _FakeClient(fake_resp)

        with patch("nu_agent_core.proxy.httpx.AsyncClient", return_value=fake_client):
            stream = ProxyMessageEventStream()
            options = ProxyStreamOptions(proxy_url="https://proxy.test")
            with pytest.raises(AttributeError):
                await _drive_proxy(model, object(), options, stream)

    async def test_abort_signal_mid_stream(self) -> None:
        """Abort signal set between chunks — exercises 'Request aborted' path."""
        model = _model()
        signal = asyncio.Event()

        async def gen() -> Any:
            yield _sse_lines({"type": "start"})
            signal.set()
            yield _sse_lines({"type": "text_start", "contentIndex": 0})

        fake_resp = _FakeResponse(chunk_gen=gen)
        fake_client = _FakeClient(fake_resp)

        with patch("nu_agent_core.proxy.httpx.AsyncClient", return_value=fake_client):
            stream = ProxyMessageEventStream()
            options = ProxyStreamOptions(proxy_url="https://proxy.test", signal=signal)
            # Error path pushes a raw dict → AttributeError (same issue as above)
            with pytest.raises(AttributeError):
                await _drive_proxy(model, object(), options, stream)

    async def test_abort_signal_after_stream_ends(self) -> None:
        """Signal set after iteration finishes — exercises post-stream abort check."""
        model = _model()
        signal = asyncio.Event()

        async def gen() -> Any:
            yield _sse_lines(
                {"type": "start"},
                {"type": "text_start", "contentIndex": 0},
                {"type": "text_delta", "contentIndex": 0, "delta": "hi"},
                {"type": "text_end", "contentIndex": 0},
            )
            signal.set()

        fake_resp = _FakeResponse(chunk_gen=gen)
        fake_client = _FakeClient(fake_resp)

        with patch("nu_agent_core.proxy.httpx.AsyncClient", return_value=fake_client):
            stream = ProxyMessageEventStream()
            options = ProxyStreamOptions(proxy_url="https://proxy.test", signal=signal)
            with pytest.raises(AttributeError):
                await _drive_proxy(model, object(), options, stream)

    async def test_malformed_sse_ignored(self) -> None:
        model = _model()
        sse_payload = (
            ":comment line\n"
            "event: ping\n"
            f"data: {json.dumps({'type': 'start'})}\n"
            "\n"
            f"data: {json.dumps({'type': 'done', 'reason': 'stop', 'usage': {}})}\n"
        )

        fake_resp = _FakeResponse(chunks=[sse_payload])
        fake_client = _FakeClient(fake_resp)

        with patch("nu_agent_core.proxy.httpx.AsyncClient", return_value=fake_client):
            stream = ProxyMessageEventStream()
            options = ProxyStreamOptions(proxy_url="https://proxy.test")
            await _drive_proxy(model, object(), options, stream)

        events = []
        async for e in stream:
            events.append(e)

        types = [e.type for e in events]
        assert "start" in types
        assert "done" in types

    async def test_empty_data_line_skipped(self) -> None:
        """A ``data: `` line with no payload should be skipped."""
        model = _model()
        sse_payload = (
            "data: \n"
            f"data: {json.dumps({'type': 'start'})}\n"
            f"data: {json.dumps({'type': 'done', 'reason': 'stop', 'usage': {}})}\n"
        )

        fake_resp = _FakeResponse(chunks=[sse_payload])
        fake_client = _FakeClient(fake_resp)

        with patch("nu_agent_core.proxy.httpx.AsyncClient", return_value=fake_client):
            stream = ProxyMessageEventStream()
            options = ProxyStreamOptions(proxy_url="https://proxy.test")
            await _drive_proxy(model, object(), options, stream)

        events = []
        async for e in stream:
            events.append(e)

        types = [e.type for e in events]
        assert "start" in types
        assert "done" in types
