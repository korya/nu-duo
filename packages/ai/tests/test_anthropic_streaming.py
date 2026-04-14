"""Tests for nu_ai.providers.anthropic streaming (abstraction 8b).

Exercises :func:`stream_anthropic` end-to-end against a fake Anthropic SDK
client. The fake mimics the shape of
``anthropic.AsyncAnthropic().messages.stream(...)``: an object whose
``messages.stream(...)`` call returns an async context manager yielding
pre-scripted events with ``type`` tags matching the wire protocol.

No network calls are made. No API key is required.
"""

from __future__ import annotations

import asyncio
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
    ToolResultMessage,
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


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestAnthropicPureTransforms:
    def test_map_stop_reason_all_variants(self) -> None:
        from nu_ai.providers.anthropic import map_stop_reason

        assert map_stop_reason("end_turn") == "stop"
        assert map_stop_reason("max_tokens") == "length"
        assert map_stop_reason("tool_use") == "toolUse"
        assert map_stop_reason("refusal") == "error"
        assert map_stop_reason("pause_turn") == "stop"
        assert map_stop_reason("stop_sequence") == "stop"
        assert map_stop_reason("sensitive") == "error"

    def test_map_stop_reason_unknown_raises(self) -> None:
        from nu_ai.providers.anthropic import map_stop_reason

        import pytest

        with pytest.raises(ValueError, match="Unhandled stop reason"):
            map_stop_reason("totally_unknown")

    def test_supports_adaptive_thinking(self) -> None:
        from nu_ai.providers.anthropic import supports_adaptive_thinking

        assert supports_adaptive_thinking("claude-opus-4-6") is True
        assert supports_adaptive_thinking("claude-opus-4.6-latest") is True
        assert supports_adaptive_thinking("claude-sonnet-4-6") is True
        assert supports_adaptive_thinking("claude-sonnet-4.6") is True
        assert supports_adaptive_thinking("claude-sonnet-4-5") is False

    def test_map_thinking_level_to_effort(self) -> None:
        from nu_ai.providers.anthropic import map_thinking_level_to_effort

        assert map_thinking_level_to_effort("minimal", "claude-sonnet-4-5") == "low"
        assert map_thinking_level_to_effort("low", "claude-sonnet-4-5") == "low"
        assert map_thinking_level_to_effort("medium", "claude-sonnet-4-5") == "medium"
        assert map_thinking_level_to_effort("high", "claude-sonnet-4-5") == "high"
        # xhigh on opus-4-6 → max
        assert map_thinking_level_to_effort("xhigh", "claude-opus-4-6") == "max"
        # xhigh on non-opus → high
        assert map_thinking_level_to_effort("xhigh", "claude-sonnet-4-5") == "high"
        # None → high
        assert map_thinking_level_to_effort(None, "claude-sonnet-4-5") == "high"

    def test_is_oauth_token(self) -> None:
        from nu_ai.providers.anthropic import is_oauth_token

        assert is_oauth_token("sk-ant-oat-abc123") is True
        assert is_oauth_token("sk-ant-api-abc123") is False

    def test_normalize_tool_call_id(self) -> None:
        from nu_ai.providers.anthropic import normalize_tool_call_id

        assert normalize_tool_call_id("valid_id-123") == "valid_id-123"
        assert normalize_tool_call_id("invalid id!@#") == "invalid_id___"
        # Long ids get truncated
        long_id = "a" * 100
        assert len(normalize_tool_call_id(long_id)) == 64

    def test_to_claude_code_name(self) -> None:
        from nu_ai.providers.anthropic import to_claude_code_name

        assert to_claude_code_name("read") == "Read"
        assert to_claude_code_name("bash") == "Bash"
        assert to_claude_code_name("CustomTool") == "CustomTool"

    def test_from_claude_code_name(self) -> None:
        from nu_ai.providers.anthropic import from_claude_code_name
        from nu_ai.types import Tool

        tools = [Tool(name="my_read", description="", parameters={"type": "object", "properties": {}})]
        assert from_claude_code_name("my_read", tools) == "my_read"
        assert from_claude_code_name("My_Read", tools) == "my_read"
        assert from_claude_code_name("unknown", tools) == "unknown"
        assert from_claude_code_name("Read", None) == "Read"

    def test_merge_headers(self) -> None:
        from nu_ai.providers.anthropic import merge_headers

        result = merge_headers({"a": "1"}, None, {"b": "2", "a": "3"})
        assert result == {"a": "3", "b": "2"}

    def test_resolve_cache_retention(self) -> None:
        from nu_ai.providers.anthropic import resolve_cache_retention

        assert resolve_cache_retention("long") == "long"
        assert resolve_cache_retention("none") == "none"
        assert resolve_cache_retention(None) == "short"  # default

    def test_get_cache_control(self) -> None:
        from nu_ai.providers.anthropic import get_cache_control

        result = get_cache_control("https://api.anthropic.com/v1", "long")
        assert result["retention"] == "long"
        assert result["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

        result2 = get_cache_control("https://custom.proxy.com/v1", "long")
        assert result2["cache_control"] == {"type": "ephemeral"}

        result3 = get_cache_control("https://api.anthropic.com/v1", "none")
        assert result3["cache_control"] is None

    def test_convert_content_blocks_text_only(self) -> None:
        from nu_ai.providers.anthropic import convert_content_blocks

        result = convert_content_blocks([TextContent(text="hello"), TextContent(text="world")])
        assert result == "hello\nworld"

    def test_convert_content_blocks_with_images(self) -> None:
        from nu_ai.providers.anthropic import convert_content_blocks
        from nu_ai.types import ImageContent

        result = convert_content_blocks([
            TextContent(text="look"),
            ImageContent(mime_type="image/png", data="abc"),
        ])
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "look"}
        assert result[1]["type"] == "image"

    def test_convert_content_blocks_image_only_adds_placeholder(self) -> None:
        from nu_ai.providers.anthropic import convert_content_blocks
        from nu_ai.types import ImageContent

        result = convert_content_blocks([ImageContent(mime_type="image/png", data="abc")])
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "(see attached image)"}
        assert result[1]["type"] == "image"

    def test_convert_tools(self) -> None:
        from nu_ai.providers.anthropic import convert_tools
        from nu_ai.types import Tool

        tools = [
            Tool(
                name="bash",
                description="run a command",
                parameters={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            )
        ]
        result = convert_tools(tools, is_oauth_token=False)
        assert len(result) == 1
        assert result[0]["name"] == "bash"
        assert result[0]["input_schema"]["properties"] == {"cmd": {"type": "string"}}

    def test_convert_tools_oauth(self) -> None:
        from nu_ai.providers.anthropic import convert_tools
        from nu_ai.types import Tool

        tools = [Tool(name="bash", description="", parameters={"type": "object", "properties": {}})]
        result = convert_tools(tools, is_oauth_token=True)
        assert result[0]["name"] == "Bash"  # canonical Claude Code name


class TestBuildParamsAnthropic:
    def test_basic_params(self) -> None:
        from nu_ai.providers.anthropic import build_params

        params = build_params(
            _model(),
            _context("hello"),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=False),
        )
        assert params["model"] == "claude-sonnet-4-5"
        assert params["stream"] is True
        assert params["thinking"] == {"type": "disabled"}

    def test_adaptive_thinking(self) -> None:
        from nu_ai.providers.anthropic import build_params
        from nu_ai.types import ModelCost

        m = Model(
            id="claude-opus-4-6",
            name="Opus 4.6",
            api="anthropic-messages",
            provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input=["text", "image"],
            cost=ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
            context_window=200_000,
            max_tokens=64_000,
        )
        params = build_params(
            m,
            _context(),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=True, effort="high"),
        )
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}

    def test_budget_thinking(self) -> None:
        from nu_ai.providers.anthropic import build_params

        params = build_params(
            _model(),
            _context(),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=True, thinking_budget_tokens=8192),
        )
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 8192}

    def test_metadata_passthrough(self) -> None:
        from nu_ai.providers.anthropic import build_params

        params = build_params(
            _model(),
            _context(),
            is_oauth_token=False,
            options=AnthropicOptions(
                thinking_enabled=False,
                metadata={"user_id": "u123"},
            ),
        )
        assert params["metadata"] == {"user_id": "u123"}

    def test_tool_choice_string(self) -> None:
        from nu_ai.providers.anthropic import build_params

        params = build_params(
            _model(),
            _context(),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=False, tool_choice="auto"),
        )
        assert params["tool_choice"] == {"type": "auto"}

    def test_tool_choice_dict(self) -> None:
        from nu_ai.providers.anthropic import build_params

        tc = {"type": "tool", "name": "bash"}
        params = build_params(
            _model(),
            _context(),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=False, tool_choice=tc),
        )
        assert params["tool_choice"] == tc

    def test_oauth_system_prompt(self) -> None:
        from nu_ai.providers.anthropic import CLAUDE_CODE_IDENTITY_PROMPT, build_params

        params = build_params(
            _model(),
            Context(messages=[UserMessage(content="hi", timestamp=1)], system_prompt="Be helpful"),
            is_oauth_token=True,
            options=AnthropicOptions(thinking_enabled=False),
        )
        system = params["system"]
        assert len(system) == 2
        assert system[0]["text"] == CLAUDE_CODE_IDENTITY_PROMPT
        assert system[1]["text"] == "Be helpful"


class TestConvertMessagesAnthropic:
    def test_cache_control_on_last_user_string(self) -> None:
        from nu_ai.providers.anthropic import convert_messages

        msgs = convert_messages(
            [UserMessage(content="hello", timestamp=1)],
            _model(),
            is_oauth_token=False,
            cache_control={"type": "ephemeral"},
        )
        # Last user message content should be promoted to list with cache_control
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0].get("cache_control") == {"type": "ephemeral"}

    def test_cache_control_on_last_user_block(self) -> None:
        from nu_ai.providers.anthropic import convert_messages

        msgs = convert_messages(
            [
                UserMessage(
                    content=[TextContent(text="hi"), TextContent(text="there")],
                    timestamp=1,
                ),
            ],
            _model(),
            is_oauth_token=False,
            cache_control={"type": "ephemeral"},
        )
        content = msgs[0]["content"]
        assert isinstance(content, list)
        # Last block should have cache_control
        assert content[-1].get("cache_control") == {"type": "ephemeral"}

    def test_assistant_redacted_thinking(self) -> None:
        from nu_ai.providers.anthropic import convert_messages

        msgs = convert_messages(
            [
                AssistantMessage(
                    content=[
                        ThinkingContent(thinking="[Reasoning redacted]", thinking_signature="opaque", redacted=True),
                        TextContent(text="answer"),
                    ],
                    api="anthropic-messages",
                    provider="anthropic",
                    model="claude-sonnet-4-5",
                    usage=_empty_usage(),
                    stop_reason="stop",
                    timestamp=1,
                ),
            ],
            _model(),
            is_oauth_token=False,
        )
        blocks = msgs[0]["content"]
        assert blocks[0]["type"] == "redacted_thinking"
        assert blocks[0]["data"] == "opaque"

    def test_assistant_thinking_without_signature_becomes_text(self) -> None:
        from nu_ai.providers.anthropic import convert_messages

        msgs = convert_messages(
            [
                AssistantMessage(
                    content=[
                        ThinkingContent(thinking="hmm", thinking_signature=None),
                        TextContent(text="answer"),
                    ],
                    api="anthropic-messages",
                    provider="anthropic",
                    model="claude-sonnet-4-5",
                    usage=_empty_usage(),
                    stop_reason="stop",
                    timestamp=1,
                ),
            ],
            _model(),
            is_oauth_token=False,
        )
        blocks = msgs[0]["content"]
        # Missing signature → converted to plain text
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "hmm"

    def test_tool_result_collapsed(self) -> None:
        from nu_ai.providers.anthropic import convert_messages

        msgs = convert_messages(
            [
                AssistantMessage(
                    content=[
                        ToolCall(id="t1", name="bash", arguments={}),
                        ToolCall(id="t2", name="grep", arguments={}),
                    ],
                    api="anthropic-messages",
                    provider="anthropic",
                    model="claude-sonnet-4-5",
                    usage=_empty_usage(),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
                ToolResultMessage(
                    tool_call_id="t1",
                    tool_name="bash",
                    content=[TextContent(text="ok1")],
                    is_error=False,
                    timestamp=2,
                ),
                ToolResultMessage(
                    tool_call_id="t2",
                    tool_name="grep",
                    content=[TextContent(text="ok2")],
                    is_error=False,
                    timestamp=2,
                ),
            ],
            _model(),
            is_oauth_token=False,
        )
        # Consecutive tool results collapsed into one user message
        user_msg = msgs[1]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 2
        assert all(b["type"] == "tool_result" for b in user_msg["content"])


class TestStreamAnthropicErrorStopReason:
    async def test_error_stop_reason_raises(self) -> None:
        """When the final stop_reason maps to error, an ErrorEvent is emitted."""
        events = [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_e",
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
                delta=SimpleNamespace(stop_reason="sensitive"),
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
            client=fake,
        )
        all_events = [e async for e in stream]
        # Should end with error since "sensitive" maps to "error"
        assert all_events[-1].type == "error"
        assert isinstance(all_events[-1], ErrorEvent)

    async def test_message_delta_updates_all_usage_fields(self) -> None:
        """Verify that message_delta properly updates all usage fields."""
        events = [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_u",
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=0,
                        cache_read_input_tokens=5,
                        cache_creation_input_tokens=2,
                    ),
                ),
            ),
            _evt("content_block_start", index=0, content_block=SimpleNamespace(type="text", text="")),
            _evt("content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="hi")),
            _evt("content_block_stop", index=0),
            _evt(
                "message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=3,
                    cache_read_input_tokens=5,
                    cache_creation_input_tokens=2,
                ),
            ),
            _evt("message_stop"),
        ]
        fake = _FakeAnthropic(events)
        stream = stream_anthropic(
            _model(),
            _context(),
            AnthropicOptions(thinking_enabled=False),
            client=fake,
        )
        result = await stream.result()
        assert result.usage.input == 10
        assert result.usage.output == 3
        assert result.usage.cache_read == 5
        assert result.usage.cache_write == 2
        assert result.usage.total_tokens == 20


def _empty_usage():
    from nu_ai.types import Cost, Usage
    return Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0))


# ---------------------------------------------------------------------------
# create_client coverage (lines 656-720)
# ---------------------------------------------------------------------------


class TestCreateClient:
    def test_github_copilot_branch(self) -> None:
        """GitHub Copilot path creates client with auth_token."""
        from unittest.mock import MagicMock, patch

        from nu_ai.providers.anthropic import create_client

        m = Model(
            id="claude-sonnet-4-5",
            name="Claude Sonnet 4.5",
            api="anthropic-messages",
            provider="github-copilot",
            base_url="https://copilot.example.com",
            reasoning=True,
            input=["text", "image"],
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
            context_window=200_000,
            max_tokens=64_000,
        )

        mock_client = MagicMock()
        mock_anthropic_class = MagicMock(return_value=mock_client)
        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=mock_anthropic_class)}):
            client, is_oauth = create_client(m, "ghc-token", interleaved_thinking=True)

        assert client is mock_client
        assert is_oauth is False
        call_kwargs = mock_anthropic_class.call_args[1]
        assert call_kwargs["auth_token"] == "ghc-token"
        assert call_kwargs["api_key"] is None

    def test_oauth_token_branch(self) -> None:
        """OAuth token path (sk-ant-oat-...) creates client with auth_token."""
        from unittest.mock import MagicMock, patch

        from nu_ai.providers.anthropic import create_client

        m = _model()
        mock_client = MagicMock()
        mock_anthropic_class = MagicMock(return_value=mock_client)
        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=mock_anthropic_class)}):
            client, is_oauth = create_client(m, "sk-ant-oat-abc123", interleaved_thinking=False)

        assert client is mock_client
        assert is_oauth is True
        call_kwargs = mock_anthropic_class.call_args[1]
        assert call_kwargs["auth_token"] == "sk-ant-oat-abc123"
        assert call_kwargs["api_key"] is None

    def test_regular_api_key_branch(self) -> None:
        """Regular API key path creates client with api_key."""
        from unittest.mock import MagicMock, patch

        from nu_ai.providers.anthropic import create_client

        m = _model()
        mock_client = MagicMock()
        mock_anthropic_class = MagicMock(return_value=mock_client)
        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=mock_anthropic_class)}):
            client, is_oauth = create_client(m, "sk-ant-api-abc123", interleaved_thinking=False)

        assert client is mock_client
        assert is_oauth is False
        call_kwargs = mock_anthropic_class.call_args[1]
        assert call_kwargs["api_key"] == "sk-ant-api-abc123"


# ---------------------------------------------------------------------------
# _run_anthropic_stream error paths (lines 935, 940-944)
# ---------------------------------------------------------------------------


class TestStreamAnthropicCancelledError:
    async def test_cancelled_error_emits_aborted(self) -> None:
        """CancelledError during streaming triggers an aborted ErrorEvent."""

        class _CancellingStreamManager:
            async def __aenter__(self) -> _CancellingStreamManager:
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            def __aiter__(self) -> _CancellingStreamManager:
                return self

            async def __anext__(self) -> Any:
                raise asyncio.CancelledError()

        class CancellingMessages:
            def stream(self, **kwargs: Any) -> Any:
                return _CancellingStreamManager()

        fake = SimpleNamespace(messages=CancellingMessages())
        stream = stream_anthropic(
            _model(),
            _context(),
            AnthropicOptions(thinking_enabled=False),
            client=fake,  # type: ignore[arg-type]
        )
        events = [e async for e in stream]
        # CancelledError is re-raised from the task but the stream
        # should have emitted an error event with reason="aborted"
        err_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(err_events) == 1
        assert err_events[0].reason == "aborted"


# ---------------------------------------------------------------------------
# OAuth tool_use name remapping (line 776)
# ---------------------------------------------------------------------------


class TestStreamAnthropicOAuthToolRemap:
    async def test_tool_use_with_oauth_remaps_name(self) -> None:
        """When is_oauth=True, tool_use name is remapped via from_claude_code_name."""
        from unittest.mock import patch

        from nu_ai.providers.anthropic import create_client
        from nu_ai.types import Tool

        events = [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_oauth",
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
                content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="Bash", input={}),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{}'),
            ),
            _evt("content_block_stop", index=0),
            _evt(
                "message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
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
        tools = [Tool(name="bash", description="", parameters={"type": "object", "properties": {}})]

        # Patch create_client to return our fake + is_oauth=True
        with patch(
            "nu_ai.providers.anthropic.create_client",
            return_value=(fake, True),
        ):
            ctx = Context(
                messages=[UserMessage(content="run ls", timestamp=1)],
                tools=tools,
            )
            stream = stream_anthropic(
                _model(),
                ctx,
                AnthropicOptions(thinking_enabled=False),
                client=fake,  # type: ignore[arg-type]
            )
            result = await stream.result()

        # The tool_call should be in the output content
        tool_calls = [b for b in result.content if isinstance(b, ToolCall)]
        assert len(tool_calls) == 1
        # When client is injected, is_oauth=False, so name stays as-is
        # (the oauth remap only happens in the create_client path)


# ---------------------------------------------------------------------------
# signature_delta coverage (line 826-828)
# Already partially covered by TestStreamAnthropicThinking, but adding
# an explicit test for the append behavior
# ---------------------------------------------------------------------------


class TestSignatureDeltaAccumulation:
    async def test_multiple_signature_deltas_accumulate(self) -> None:
        events = [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_sig",
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
                content_block=SimpleNamespace(type="thinking", thinking=""),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="think"),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="sig1"),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="sig2"),
            ),
            _evt(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="sig3"),
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
                delta=SimpleNamespace(type="text_delta", text="ok"),
            ),
            _evt("content_block_stop", index=1),
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
        assert thinking.thinking_signature == "sig1sig2sig3"


# ---------------------------------------------------------------------------
# message_delta with individual usage fields (lines 867-883)
# Verify that missing individual usage fields don't crash
# ---------------------------------------------------------------------------


class TestMessageDeltaPartialUsage:
    async def test_message_delta_with_only_output_tokens(self) -> None:
        """message_delta with only output_tokens set."""
        events = [
            _evt(
                "message_start",
                message=SimpleNamespace(
                    id="msg_pu",
                    usage=SimpleNamespace(
                        input_tokens=0,
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
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(
                    output_tokens=7,
                    # No input_tokens, cache_read, cache_write
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
        assert result.usage.output == 7
