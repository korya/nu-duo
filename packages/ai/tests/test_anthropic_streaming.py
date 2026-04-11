"""Tests for nu_ai.providers.anthropic streaming (abstraction 8b).

Exercises :func:`stream_anthropic` end-to-end against a fake Anthropic SDK
client. The fake mimics the shape of
``anthropic.AsyncAnthropic().messages.stream(...)``: an object whose
``messages.stream(...)`` call returns an async context manager yielding
pre-scripted events with ``type`` tags matching the wire protocol.

No network calls are made. No API key is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from nu_ai.providers.anthropic import stream_anthropic
from nu_ai.types import (
    AnthropicOptions,
    AssistantMessage,
    Context,
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
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Fake Anthropic SDK stream manager
# ---------------------------------------------------------------------------


def _evt(type_: str, **fields: object) -> SimpleNamespace:
    """Build a fake SDK event with a ``.type`` attribute."""
    return SimpleNamespace(type=type_, **fields)


@dataclass
class _FakeStreamManager:
    events: list[SimpleNamespace]
    kwargs_holder: dict[str, Any] = field(default_factory=dict)

    async def __aenter__(self) -> _FakeStreamManager:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def __aiter__(self) -> _FakeStreamManager:
        self._iter = iter(self.events)
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@dataclass
class _FakeMessages:
    events: list[SimpleNamespace]
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    def stream(self, **kwargs: Any) -> _FakeStreamManager:
        self.last_kwargs = kwargs
        return _FakeStreamManager(self.events)


@dataclass
class _FakeAnthropic:
    events: list[SimpleNamespace]
    messages: _FakeMessages = field(init=False)

    def __post_init__(self) -> None:
        self.messages = _FakeMessages(self.events)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _model() -> Model:
    return Model(
        id="claude-sonnet-4-5",
        name="Claude Sonnet 4.5",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=200_000,
        max_tokens=64_000,
    )


def _context(text: str = "Hello") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=1)])


def _canonical_text_events() -> list[SimpleNamespace]:
    """Scripted events for a single text-only response."""
    return [
        _evt(
            "message_start",
            message=SimpleNamespace(
                id="msg_01",
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=0,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            ),
        ),
        _evt(
            "content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", text=""),
        ),
        _evt(
            "content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Hello"),
        ),
        _evt(
            "content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text=" there"),
        ),
        _evt("content_block_stop", index=0),
        _evt(
            "message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        ),
        _evt("message_stop"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamAnthropicText:
    async def test_canonical_text_response(self) -> None:
        fake = _FakeAnthropic(_canonical_text_events())
        stream = stream_anthropic(
            _model(),
            _context(),
            AnthropicOptions(thinking_enabled=False),
            client=fake,  # type: ignore[arg-type]
        )

        events = [e async for e in stream]
        types = [e.type for e in events]
        assert types == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]

        # Start event carries the empty partial.
        assert isinstance(events[0], StartEvent)
        assert events[0].partial.content == []

        # text_start → text_delta events accumulate.
        assert isinstance(events[1], TextStartEvent)
        assert isinstance(events[2], TextDeltaEvent)
        assert events[2].delta == "Hello"
        assert isinstance(events[3], TextDeltaEvent)
        assert events[3].delta == " there"

        # text_end carries the finalized content.
        assert isinstance(events[4], TextEndEvent)
        assert events[4].content == "Hello there"

        # Done carries the full message with accumulated usage.
        assert isinstance(events[5], DoneEvent)
        done = events[5]
        assert done.reason == "stop"
        assert isinstance(done.message, AssistantMessage)
        assert done.message.response_id == "msg_01"
        assert done.message.usage.input == 10
        assert done.message.usage.output == 5
        assert len(done.message.content) == 1
        assert isinstance(done.message.content[0], TextContent)
        assert done.message.content[0].text == "Hello there"

    async def test_result_resolves_with_final_message(self) -> None:
        fake = _FakeAnthropic(_canonical_text_events())
        stream = stream_anthropic(
            _model(),
            _context(),
            AnthropicOptions(thinking_enabled=False),
            client=fake,  # type: ignore[arg-type]
        )
        result = await stream.result()
        assert isinstance(result, AssistantMessage)
        assert result.stop_reason == "stop"


class TestStreamAnthropicToolCall:
    def _events(self) -> list[SimpleNamespace]:
        return [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_02",
                    usage=SimpleNamespace(
                        input_tokens=20,
                        output_tokens=0,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    ),
                ),
            ),
            _evt(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="bash", input={}),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"cmd"'),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json=': "ls"}'),
            ),
            _evt("content_block_stop", index=0),
            _evt(
                "message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(
                    input_tokens=20,
                    output_tokens=8,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            ),
            _evt("message_stop"),
        ]

    async def test_tool_call_accumulates_and_finalizes(self) -> None:
        fake = _FakeAnthropic(self._events())
        stream = stream_anthropic(
            _model(),
            _context("run ls"),
            AnthropicOptions(thinking_enabled=False),
            client=fake,  # type: ignore[arg-type]
        )
        events = [e async for e in stream]
        types = [e.type for e in events]
        assert types == [
            "start",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
            "done",
        ]

        assert isinstance(events[1], ToolCallStartEvent)
        assert isinstance(events[2], ToolCallDeltaEvent)
        assert events[2].delta == '{"cmd"'
        assert isinstance(events[4], ToolCallEndEvent)
        tool_call = events[4].tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id == "toolu_1"
        assert tool_call.name == "bash"
        assert tool_call.arguments == {"cmd": "ls"}

        done = events[5]
        assert isinstance(done, DoneEvent)
        assert done.reason == "toolUse"


class TestStreamAnthropicThinking:
    def _events(self) -> list[SimpleNamespace]:
        return [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_03",
                    usage=SimpleNamespace(
                        input_tokens=5,
                        output_tokens=0,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    ),
                ),
            ),
            _evt(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(type="thinking", thinking=""),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="hmm"),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="sig-part-1"),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="-end"),
            ),
            _evt("content_block_stop", index=0),
            _evt(
                "content_block_start",
                index=1,
                content_block=SimpleNamespace(type="text", text=""),
            ),
            _evt(
                "content_block_delta",
                index=1,
                delta=SimpleNamespace(type="text_delta", text="answer"),
            ),
            _evt("content_block_stop", index=1),
            _evt(
                "message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(
                    input_tokens=5,
                    output_tokens=3,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            ),
            _evt("message_stop"),
        ]

    async def test_thinking_captured_with_signature(self) -> None:
        fake = _FakeAnthropic(self._events())
        stream = stream_anthropic(
            _model(),
            _context(),
            AnthropicOptions(thinking_enabled=True),
            client=fake,  # type: ignore[arg-type]
        )
        events = [e async for e in stream]
        types = [e.type for e in events]
        assert types == [
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "done",
        ]

        assert isinstance(events[2], ThinkingDeltaEvent)
        assert events[2].delta == "hmm"

        assert isinstance(events[3], ThinkingEndEvent)
        assert events[3].content == "hmm"

        done = events[-1]
        assert isinstance(done, DoneEvent)
        thinking_block = next(b for b in done.message.content if isinstance(b, ThinkingContent))
        assert thinking_block.thinking == "hmm"
        assert thinking_block.thinking_signature == "sig-part-1-end"


class TestStreamAnthropicRedactedThinking:
    async def test_redacted_thinking_is_emitted(self) -> None:
        events = [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_04",
                    usage=SimpleNamespace(
                        input_tokens=1,
                        output_tokens=0,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    ),
                ),
            ),
            _evt(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(type="redacted_thinking", data="opaque-ciphertext"),
            ),
            _evt("content_block_stop", index=0),
            _evt(
                "message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            ),
            _evt("message_stop"),
        ]
        fake = _FakeAnthropic(events)
        stream = stream_anthropic(
            _model(),
            _context(),
            AnthropicOptions(thinking_enabled=True),
            client=fake,  # type: ignore[arg-type]
        )
        result = await stream.result()
        thinking = next(b for b in result.content if isinstance(b, ThinkingContent))
        assert thinking.redacted is True
        assert thinking.thinking_signature == "opaque-ciphertext"


class TestStreamAnthropicError:
    async def test_sdk_exception_becomes_error_event(self) -> None:
        class BoomMessages:
            def stream(self, **kwargs: Any) -> Any:
                raise RuntimeError("upstream failure")

        fake = SimpleNamespace(messages=BoomMessages())
        stream = stream_anthropic(
            _model(),
            _context(),
            AnthropicOptions(thinking_enabled=False),
            client=fake,  # type: ignore[arg-type]
        )
        events = [e async for e in stream]
        assert events[-1].type == "error"
        err = events[-1]
        assert isinstance(err, ErrorEvent)
        assert err.reason == "error"
        assert err.error.stop_reason == "error"
        assert err.error.error_message is not None
        assert "upstream failure" in err.error.error_message


class TestStreamAnthropicPayloadCapture:
    async def test_payload_sent_to_sdk_is_well_formed(self) -> None:
        fake = _FakeAnthropic(_canonical_text_events())
        stream = stream_anthropic(
            _model(),
            _context("Hello"),
            AnthropicOptions(thinking_enabled=False),
            client=fake,  # type: ignore[arg-type]
        )
        _ = [e async for e in stream]

        kwargs = fake.messages.last_kwargs
        assert kwargs["model"] == "claude-sonnet-4-5"
        assert kwargs["stream"] is True
        assert kwargs["thinking"] == {"type": "disabled"}
        assert kwargs["messages"][0]["role"] == "user"


class TestStreamAnthropicStopReasonMapping:
    @pytest.mark.parametrize(
        ("sdk_reason", "expected_pi_reason"),
        [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "toolUse"),
        ],
    )
    async def test_stop_reasons_mapped(self, sdk_reason: str, expected_pi_reason: str) -> None:
        events = [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_x",
                    usage=SimpleNamespace(
                        input_tokens=1,
                        output_tokens=0,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    ),
                ),
            ),
            _evt("content_block_start", index=0, content_block=SimpleNamespace(type="text", text="")),
            _evt("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="hi")),
            _evt("content_block_stop", index=0),
            _evt(
                "message_delta",
                delta=SimpleNamespace(stop_reason=sdk_reason),
                usage=SimpleNamespace(
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            ),
            _evt("message_stop"),
        ]
        fake = _FakeAnthropic(events)
        stream = stream_anthropic(
            _model(),
            _context(),
            AnthropicOptions(thinking_enabled=False),
            client=fake,  # type: ignore[arg-type]
        )
        result = await stream.result()
        assert result.stop_reason == expected_pi_reason
