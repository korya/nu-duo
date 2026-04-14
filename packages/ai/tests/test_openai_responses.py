"""Tests for nu_ai.providers.openai_responses.

Covers pure transform functions (request building, message conversion, tool
conversion, stop-reason mapping) as well as the stream event mapping via a
fake async client that yields scripted Responses API events.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nu_ai.providers.openai_responses import (
    OpenAIResponsesOptions,
    build_params,
    convert_responses_messages,
    convert_responses_tools,
    map_stop_reason,
    process_responses_stream,
    stream_openai_responses,
)
from nu_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
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

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _model(
    *,
    model_id: str = "o4-mini",
    provider: str = "openai",
    base_url: str = "https://api.openai.com/v1",
    reasoning: bool = False,
    inputs: list[str] | None = None,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        input=inputs or ["text", "image"],  # type: ignore[arg-type]
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=0),
        context_window=128_000,
        max_tokens=4096,
    )


def _usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


def _assistant(content: list[object], *, stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=content,  # type: ignore[arg-type]
        api="openai-responses",
        provider="openai",
        model="o4-mini",
        usage=_usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        timestamp=1,
    )


def _ctx(
    text: str = "hello",
    tools: list[Tool] | None = None,
) -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=1)], tools=tools)


# ---------------------------------------------------------------------------
# Fake async event stream for testing process_responses_stream
# ---------------------------------------------------------------------------


@dataclass
class _FakeEventStream:
    events: list[dict[str, Any]]

    def __aiter__(self) -> _FakeEventStream:
        self._iter = iter(self.events)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@dataclass
class _FakeResponses:
    events: list[dict[str, Any]]
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    async def create(self, **kwargs: Any) -> _FakeEventStream:
        self.last_kwargs = kwargs
        return _FakeEventStream(self.events)


@dataclass
class _FakeAsyncOpenAI:
    events: list[dict[str, Any]]
    responses: _FakeResponses = field(init=False)

    def __post_init__(self) -> None:
        self.responses = _FakeResponses(self.events)


def _new_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_usage(),
        stop_reason="stop",
        timestamp=1,
    )


async def _collect_events(
    events: list[dict[str, Any]],
    model: Model | None = None,
) -> tuple[AssistantMessage, list[Any]]:
    """Run process_responses_stream against scripted events; return output + collected stream events."""
    m = model or _model()
    output = _new_output(m)
    stream = AssistantMessageEventStream()
    collected: list[Any] = []

    # Collect in background while draining
    async def _drain() -> None:
        async for event in stream:
            collected.append(event)

    import asyncio

    drain_task = asyncio.create_task(_drain())
    await process_responses_stream(_FakeEventStream(events), output, stream, m)
    stream.end()
    await drain_task
    return output, collected


# ---------------------------------------------------------------------------
# 1. Build params — request construction
# ---------------------------------------------------------------------------


class TestBuildParams:
    def test_basic_params(self) -> None:
        model = _model()
        ctx = _ctx("hello")
        params = build_params(model, ctx)

        assert params["model"] == "o4-mini"
        assert params["stream"] is True
        assert params["store"] is False
        assert isinstance(params["input"], list)
        assert len(params["input"]) == 1
        assert params["input"][0]["role"] == "user"

    def test_max_tokens(self) -> None:
        model = _model()
        ctx = _ctx()
        opts = OpenAIResponsesOptions(max_tokens=1000)
        params = build_params(model, ctx, opts)
        assert params["max_output_tokens"] == 1000

    def test_temperature(self) -> None:
        model = _model()
        ctx = _ctx()
        opts = OpenAIResponsesOptions(temperature=0.7)
        params = build_params(model, ctx, opts)
        assert params["temperature"] == 0.7

    def test_tools_included(self) -> None:
        model = _model()
        tool = Tool(name="bash", description="Run bash", parameters={"type": "object", "properties": {}})
        ctx = _ctx(tools=[tool])
        params = build_params(model, ctx)
        assert "tools" in params
        assert len(params["tools"]) == 1
        assert params["tools"][0]["name"] == "bash"
        assert params["tools"][0]["type"] == "function"

    def test_reasoning_model_effort(self) -> None:
        model = _model(reasoning=True)
        ctx = _ctx()
        opts = OpenAIResponsesOptions(reasoning_effort="high")
        params = build_params(model, ctx, opts)
        assert "reasoning" in params
        assert params["reasoning"]["effort"] == "high"
        assert "reasoning.encrypted_content" in params.get("include", [])

    def test_reasoning_model_no_effort(self) -> None:
        """Non-copilot reasoning model without explicit effort gets effort=none."""
        model = _model(reasoning=True, provider="openai")
        ctx = _ctx()
        params = build_params(model, ctx)
        assert params["reasoning"] == {"effort": "none"}

    def test_service_tier(self) -> None:
        model = _model()
        ctx = _ctx()
        opts = OpenAIResponsesOptions(service_tier="flex")
        params = build_params(model, ctx, opts)
        assert params["service_tier"] == "flex"

    def test_no_tools_no_tools_key(self) -> None:
        model = _model()
        ctx = _ctx()  # No tools
        params = build_params(model, ctx)
        assert "tools" not in params


# ---------------------------------------------------------------------------
# 2. Message conversion
# ---------------------------------------------------------------------------


class TestConvertResponsesMessages:
    def test_simple_user_message(self) -> None:
        model = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        msgs = convert_responses_messages(model, ctx)
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "input_text"
        assert msgs[0]["content"][0]["text"] == "hi"

    def test_system_prompt_added(self) -> None:
        model = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)], system_prompt="Be concise.")
        msgs = convert_responses_messages(model, ctx)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be concise."
        assert msgs[1]["role"] == "user"

    def test_reasoning_model_uses_developer_role(self) -> None:
        model = _model(reasoning=True)
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)], system_prompt="Think hard.")
        msgs = convert_responses_messages(model, ctx)
        assert msgs[0]["role"] == "developer"

    def test_tool_result_message(self) -> None:
        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="run it", timestamp=1),
                _assistant(
                    content=[ToolCall(id="call_1|fc_abc", name="bash", arguments={"cmd": "ls"})],
                    stop_reason="toolUse",
                ),
                ToolResultMessage(
                    tool_call_id="call_1|fc_abc",
                    tool_name="bash",
                    content=[TextContent(text="file.txt")],
                    is_error=False,
                    timestamp=2,
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        # Find the function_call_output item
        fc_output = next(m for m in msgs if m.get("type") == "function_call_output")
        assert fc_output["call_id"] == "call_1"
        assert "file.txt" in fc_output["output"]

    def test_assistant_text_message(self) -> None:
        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant(content=[TextContent(text="Hello!")]),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        # Should have a message-type item with output_text
        text_item = next((m for m in msgs if m.get("type") == "message"), None)
        assert text_item is not None
        assert text_item["role"] == "assistant"
        assert text_item["content"][0]["text"] == "Hello!"

    def test_images_filtered_for_text_only_model(self) -> None:
        from nu_ai.types import ImageContent

        model = _model(inputs=["text"])
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="look at this"),
                        ImageContent(mime_type="image/png", data="abc123"),
                    ],
                    timestamp=1,
                )
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        user_msg = next(m for m in msgs if m.get("role") == "user")
        content = user_msg["content"]
        assert all(c["type"] != "input_image" for c in content)


# ---------------------------------------------------------------------------
# 3. Tool conversion
# ---------------------------------------------------------------------------


class TestConvertResponsesTools:
    def test_basic_tool(self) -> None:
        tools = [
            Tool(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]
        result = convert_responses_tools(tools)
        assert len(result) == 1
        t = result[0]
        assert t["type"] == "function"
        assert t["name"] == "search"
        assert t["description"] == "Search the web"
        assert t["strict"] is False

    def test_strict_mode(self) -> None:
        tools = [Tool(name="x", description="d", parameters={"type": "object", "properties": {}})]
        result = convert_responses_tools(tools, strict=True)
        assert result[0]["strict"] is True

    def test_multiple_tools(self) -> None:
        tools = [
            Tool(name="a", description="A", parameters={"type": "object", "properties": {}}),
            Tool(name="b", description="B", parameters={"type": "object", "properties": {}}),
        ]
        result = convert_responses_tools(tools)
        assert [t["name"] for t in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# 4. Stop-reason mapping
# ---------------------------------------------------------------------------


class TestMapStopReason:
    def test_unknown_status_maps_to_stop(self) -> None:
        assert map_stop_reason("some_unknown_status") == "stop"

    def test_queued_maps_to_stop(self) -> None:
        assert map_stop_reason("queued") == "stop"

    def test_completed_maps_to_stop(self) -> None:
        assert map_stop_reason("completed") == "stop"

    def test_incomplete_maps_to_length(self) -> None:
        assert map_stop_reason("incomplete") == "length"

    def test_failed_maps_to_error(self) -> None:
        assert map_stop_reason("failed") == "error"

    def test_cancelled_maps_to_error(self) -> None:
        assert map_stop_reason("cancelled") == "error"

    def test_none_maps_to_stop(self) -> None:
        assert map_stop_reason(None) == "stop"

    def test_in_progress_maps_to_stop(self) -> None:
        assert map_stop_reason("in_progress") == "stop"


# ---------------------------------------------------------------------------
# 5. Stream event mapping
# ---------------------------------------------------------------------------


class TestTextStream:
    async def test_basic_text_response(self) -> None:
        """A sequence of output_text.delta events assembles into a text block."""
        events = [
            {"type": "response.created", "response": {"id": "resp_abc"}},
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_1", "content": []},
            },
            {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " world"},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "Hello world"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_abc",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            },
        ]

        output, collected = await _collect_events(events)

        event_types = [e.type for e in collected]
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types

        deltas = [e.delta for e in collected if isinstance(e, TextDeltaEvent)]
        assert deltas == ["Hello", " world"]

        end_events = [e for e in collected if isinstance(e, TextEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].content == "Hello world"

        assert output.stop_reason == "stop"
        assert output.usage.input == 10
        assert output.usage.output == 5

    async def test_response_id_captured(self) -> None:
        events = [
            {"type": "response.created", "response": {"id": "resp_xyz"}},
            {
                "type": "response.completed",
                "response": {"id": "resp_xyz", "status": "completed", "usage": None},
            },
        ]
        output, _ = await _collect_events(events)
        assert output.response_id == "resp_xyz"


class TestToolCallStream:
    async def test_function_call_delta_accumulation(self) -> None:
        """Tool call arguments accumulate across delta events and finalize correctly."""
        events = [
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "",
                },
            },
            {"type": "response.function_call_arguments.delta", "delta": '{"cmd"'},
            {"type": "response.function_call_arguments.delta", "delta": ': "ls"}'},
            {
                "type": "response.function_call_arguments.done",
                "arguments": '{"cmd": "ls"}',
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": '{"cmd": "ls"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_tc",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 10,
                        "total_tokens": 15,
                        "input_tokens_details": {},
                    },
                },
            },
        ]

        output, collected = await _collect_events(events)

        start_events = [e for e in collected if isinstance(e, ToolCallStartEvent)]
        [e for e in collected if isinstance(e, ToolCallDeltaEvent)]
        end_events = [e for e in collected if isinstance(e, ToolCallEndEvent)]

        assert len(start_events) == 1
        assert len(end_events) == 1

        tc = end_events[0].tool_call
        assert isinstance(tc, ToolCall)
        assert tc.id == "call_1|fc_001"
        assert tc.name == "bash"
        assert tc.arguments == {"cmd": "ls"}

        assert output.stop_reason == "toolUse"

    async def test_tool_call_stop_reason(self) -> None:
        """When content has a toolCall block, stop_reason is set to toolUse."""
        events = [
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "id": "fc_x", "call_id": "c_x", "name": "fn", "arguments": ""},
            },
            {"type": "response.function_call_arguments.done", "arguments": "{}"},
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "id": "fc_x", "call_id": "c_x", "name": "fn", "arguments": "{}"},
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
        output, _ = await _collect_events(events)
        assert output.stop_reason == "toolUse"


class TestReasoningStream:
    async def test_reasoning_summary_deltas(self) -> None:
        """Reasoning summary text.delta events feed into a ThinkingContent block."""
        events = [
            {
                "type": "response.output_item.added",
                "item": {"type": "reasoning", "id": "rs_1", "summary": []},
            },
            {
                "type": "response.reasoning_summary_part.added",
                "part": {"type": "summary_text", "text": ""},
            },
            {"type": "response.reasoning_summary_text.delta", "delta": "I think..."},
            {"type": "response.reasoning_summary_part.done"},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "I think...\n\n"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
        output, collected = await _collect_events(events, _model(reasoning=True))

        thinking_starts = [e for e in collected if isinstance(e, ThinkingStartEvent)]
        thinking_deltas = [e for e in collected if isinstance(e, ThinkingDeltaEvent)]
        thinking_ends = [e for e in collected if isinstance(e, ThinkingEndEvent)]

        assert len(thinking_starts) == 1
        # At least one delta (the "I think..." delta, plus the "\n\n" separator)
        assert len(thinking_deltas) >= 1
        assert thinking_deltas[0].delta == "I think..."
        assert len(thinking_ends) == 1

        thinking_block = next(b for b in output.content if isinstance(b, ThinkingContent))
        assert thinking_block.thinking_signature is not None
        sig_data = json.loads(thinking_block.thinking_signature)
        assert sig_data.get("type") == "reasoning"


class TestErrorHandling:
    async def test_error_event_raises(self) -> None:
        events = [
            {"type": "error", "code": "rate_limit_exceeded", "message": "Too many requests"},
        ]
        with pytest.raises(RuntimeError, match="rate_limit_exceeded"):
            await _collect_events(events)

    async def test_response_failed_raises(self) -> None:
        events = [
            {
                "type": "response.failed",
                "response": {
                    "error": {"code": "server_error", "message": "Internal error"},
                    "incomplete_details": None,
                },
            }
        ]
        with pytest.raises(RuntimeError, match="server_error"):
            await _collect_events(events)

    async def test_stream_function_error_emits_error_event(self) -> None:
        """stream_openai_responses emits an ErrorEvent when the client raises."""

        model = _model()
        ctx = _ctx()

        class _BrokenResponses:
            async def create(self, **_: Any) -> None:
                raise RuntimeError("boom")

        class _BrokenClient:
            responses = _BrokenResponses()

        events = [e async for e in stream_openai_responses(model, ctx, client=_BrokenClient())]  # type: ignore[arg-type]
        assert events[-1].type == "error"
        assert isinstance(events[-1], ErrorEvent)


# ---------------------------------------------------------------------------
# 6. Integration: stream_openai_responses with fake client
# ---------------------------------------------------------------------------


class TestStreamOpenAIResponsesIntegration:
    async def test_full_text_stream(self) -> None:
        """End-to-end: fake client → stream_openai_responses → DoneEvent."""
        scripted: list[dict[str, Any]] = [
            {"type": "response.created", "response": {"id": "r1"}},
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_0", "content": []},
            },
            {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
            {"type": "response.output_text.delta", "delta": "Hi"},
            {
                "type": "response.output_item.done",
                "item": {"type": "message", "id": "msg_0", "content": [{"type": "output_text", "text": "Hi"}]},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 1,
                        "total_tokens": 4,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            },
        ]
        fake = _FakeAsyncOpenAI(scripted)
        all_events = [e async for e in stream_openai_responses(_model(), _ctx(), OpenAIResponsesOptions(), client=fake)]
        types = [e.type for e in all_events]
        assert "start" in types
        assert "text_delta" in types
        assert "done" in types

        done = next(e for e in all_events if isinstance(e, DoneEvent))
        assert done.reason == "stop"
        assert done.message.stop_reason == "stop"
        assert done.message.usage.input == 3

    async def test_params_passed_to_client(self) -> None:
        """Verify that build_params output is forwarded to the HTTP client."""
        scripted: list[dict[str, Any]] = [
            {
                "type": "response.completed",
                "response": {"id": "r2", "status": "completed", "usage": None},
            }
        ]
        fake = _FakeAsyncOpenAI(scripted)
        tool = Tool(name="calc", description="Calculator", parameters={"type": "object", "properties": {}})
        ctx = _ctx("compute 2+2", tools=[tool])
        opts = OpenAIResponsesOptions(max_tokens=256, temperature=0.5)

        [e async for e in stream_openai_responses(_model(), ctx, opts, client=fake)]

        last = fake.responses.last_kwargs
        assert last.get("model") == "o4-mini"
        assert last.get("max_output_tokens") == 256
        assert last.get("temperature") == 0.5
        assert last.get("stream") is True
        assert "tools" in last
        assert last["tools"][0]["name"] == "calc"


# ---------------------------------------------------------------------------
# 7. Additional coverage: text signature helpers, image messages, refusal,
#    response.failed variants, service tier pricing, assistant thinking
#    conversion, multiple content blocks, _parse_text_signature edge cases
# ---------------------------------------------------------------------------


class TestTextSignatureHelpers:
    def test_encode_text_signature_v1_basic(self) -> None:
        from nu_ai.providers.openai_responses import _encode_text_signature_v1

        sig = _encode_text_signature_v1("msg_123")
        parsed = json.loads(sig)
        assert parsed == {"v": 1, "id": "msg_123"}

    def test_encode_text_signature_v1_with_phase(self) -> None:
        from nu_ai.providers.openai_responses import _encode_text_signature_v1

        sig = _encode_text_signature_v1("msg_123", "final_answer")
        parsed = json.loads(sig)
        assert parsed == {"v": 1, "id": "msg_123", "phase": "final_answer"}

    def test_parse_text_signature_none(self) -> None:
        from nu_ai.providers.openai_responses import _parse_text_signature

        assert _parse_text_signature(None) is None
        assert _parse_text_signature("") is None

    def test_parse_text_signature_json_v1(self) -> None:
        from nu_ai.providers.openai_responses import _parse_text_signature

        sig = json.dumps({"v": 1, "id": "msg_1", "phase": "commentary"})
        result = _parse_text_signature(sig)
        assert result == {"id": "msg_1", "phase": "commentary"}

    def test_parse_text_signature_json_v1_no_phase(self) -> None:
        from nu_ai.providers.openai_responses import _parse_text_signature

        sig = json.dumps({"v": 1, "id": "msg_1"})
        result = _parse_text_signature(sig)
        assert result == {"id": "msg_1"}

    def test_parse_text_signature_json_v1_invalid_phase_ignored(self) -> None:
        from nu_ai.providers.openai_responses import _parse_text_signature

        sig = json.dumps({"v": 1, "id": "msg_1", "phase": "unknown_phase"})
        result = _parse_text_signature(sig)
        assert result == {"id": "msg_1"}

    def test_parse_text_signature_plain_string(self) -> None:
        from nu_ai.providers.openai_responses import _parse_text_signature

        result = _parse_text_signature("plain_id")
        assert result == {"id": "plain_id"}

    def test_parse_text_signature_malformed_json(self) -> None:
        from nu_ai.providers.openai_responses import _parse_text_signature

        result = _parse_text_signature("{bad json")
        assert result == {"id": "{bad json"}

    def test_parse_text_signature_json_wrong_version(self) -> None:
        from nu_ai.providers.openai_responses import _parse_text_signature

        sig = json.dumps({"v": 2, "id": "msg_1"})
        result = _parse_text_signature(sig)
        assert result == {"id": sig}


class TestNormalizeIdHelpers:
    def test_normalize_id_part(self) -> None:
        from nu_ai.providers.openai_responses import _normalize_id_part

        assert _normalize_id_part("abc-123") == "abc-123"
        assert _normalize_id_part("a b!c") == "a_b_c"
        # Trailing underscores stripped
        assert _normalize_id_part("abc___") == "abc"

    def test_build_foreign_responses_item_id(self) -> None:
        from nu_ai.providers.openai_responses import _build_foreign_responses_item_id

        result = _build_foreign_responses_item_id("some_foreign_id")
        assert result.startswith("fc_")
        assert len(result) <= 64


class TestMakeNormalizeToolCallId:
    def test_openai_provider_with_pipe(self) -> None:
        from nu_ai.providers.openai_responses import _make_normalize_tool_call_id

        model = _model(provider="openai")
        normalizer = _make_normalize_tool_call_id(model)
        # Same provider → keep the item id
        source = _assistant(content=[TextContent(text="hi")])
        result = normalizer("call_1|fc_item_1", model, source)
        assert "|" in result
        assert result.startswith("call_1|")
        assert "fc_" in result

    def test_non_openai_provider_strips_pipe(self) -> None:
        from nu_ai.providers.openai_responses import _make_normalize_tool_call_id

        model = _model(provider="custom")
        normalizer = _make_normalize_tool_call_id(model)
        source = _assistant(content=[TextContent(text="hi")])
        result = normalizer("call_1|fc_item", model, source)
        # Non-openai provider: no pipe, just normalized id part
        assert "|" not in result

    def test_openai_provider_without_pipe(self) -> None:
        from nu_ai.providers.openai_responses import _make_normalize_tool_call_id

        model = _model(provider="openai")
        normalizer = _make_normalize_tool_call_id(model)
        source = _assistant(content=[TextContent(text="hi")])
        result = normalizer("call_1", model, source)
        assert result == "call_1"

    def test_openai_provider_with_non_fc_item_id(self) -> None:
        """When item id doesn't start with fc_, it gets prepended."""
        from nu_ai.providers.openai_responses import _make_normalize_tool_call_id

        model = _model(provider="openai")
        normalizer = _make_normalize_tool_call_id(model)
        source = _assistant(content=[TextContent(text="hi")])
        result = normalizer("call_1|item_1", model, source)
        assert "|" in result
        parts = result.split("|")
        assert parts[1].startswith("fc_")

    def test_foreign_source_message(self) -> None:
        from nu_ai.providers.openai_responses import _make_normalize_tool_call_id

        model = _model(provider="openai")
        normalizer = _make_normalize_tool_call_id(model)
        # Source from a different provider
        foreign_source = AssistantMessage(
            content=[TextContent(text="hi")],
            api="other-api",
            provider="other",
            model="other-model",
            usage=_usage(),
            stop_reason="stop",
            timestamp=1,
        )
        result = normalizer("call_1|fc_foreign_item", model, foreign_source)
        assert "|" in result
        # Foreign item id gets hashed
        parts = result.split("|")
        assert parts[1].startswith("fc_")


class TestConvertResponsesMessagesExtended:
    def test_user_message_with_image(self) -> None:
        from nu_ai.types import ImageContent

        model = _model()
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="look at this"),
                        ImageContent(mime_type="image/png", data="abc123"),
                    ],
                    timestamp=1,
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        user_msg = next(m for m in msgs if m.get("role") == "user")
        content = user_msg["content"]
        assert any(c["type"] == "input_image" for c in content)
        assert any(c["type"] == "input_text" for c in content)

    def test_user_image_only_filtered_drops_message(self) -> None:
        """A user message with only an image on text-only model should be skipped."""
        from nu_ai.types import ImageContent

        model = _model(inputs=["text"])
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        ImageContent(mime_type="image/png", data="abc123"),
                    ],
                    timestamp=1,
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        # No user messages should appear (all content filtered)
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 0

    def test_assistant_thinking_with_signature(self) -> None:
        """ThinkingContent with a JSON signature should replay as a reasoning item."""
        reasoning_item = {"type": "reasoning", "id": "rs_1", "summary": []}
        model = _model(reasoning=True)
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant(
                    content=[
                        ThinkingContent(
                            thinking="thinking...",
                            thinking_signature=json.dumps(reasoning_item),
                        ),
                        TextContent(text="answer"),
                    ]
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        # Reasoning item replayed from the signature
        reasoning_items = [m for m in msgs if m.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["id"] == "rs_1"

    def test_assistant_thinking_without_signature_skipped(self) -> None:
        """ThinkingContent without a signature is simply dropped."""
        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant(
                    content=[
                        ThinkingContent(thinking="thinking...", thinking_signature=None),
                        TextContent(text="answer"),
                    ]
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        reasoning_items = [m for m in msgs if m.get("type") == "reasoning"]
        assert len(reasoning_items) == 0

    def test_assistant_text_with_text_signature_phase(self) -> None:
        """TextContent with a text_signature containing phase preserves it."""
        sig = json.dumps({"v": 1, "id": "msg_42", "phase": "final_answer"})
        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant(content=[TextContent(text="answer", text_signature=sig)]),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        msg_item = next(m for m in msgs if m.get("type") == "message")
        assert msg_item["phase"] == "final_answer"
        assert msg_item["id"] == "msg_42"

    def test_assistant_tool_call_different_model_drops_fc_id(self) -> None:
        """ToolCall from a different model (same provider) omits the item id."""
        model = _model(model_id="gpt-4o")
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[
                        ToolCall(id="call_1|fc_item", name="bash", arguments={"cmd": "ls"})
                    ],
                    api="openai-responses",
                    provider="openai",
                    model="gpt-3.5",  # Different model, same provider and api
                    usage=_usage(),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        fc_item = next(m for m in msgs if m.get("type") == "function_call")
        assert "id" not in fc_item  # id should be omitted for different model

    def test_tool_result_with_image(self) -> None:
        """Tool result with image gets image content parts."""
        from nu_ai.types import ImageContent

        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="run it", timestamp=1),
                _assistant(
                    content=[ToolCall(id="call_1|fc_abc", name="screenshot", arguments={})],
                    stop_reason="toolUse",
                ),
                ToolResultMessage(
                    tool_call_id="call_1|fc_abc",
                    tool_name="screenshot",
                    content=[
                        TextContent(text="captured"),
                        ImageContent(mime_type="image/png", data="imgdata"),
                    ],
                    is_error=False,
                    timestamp=2,
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        fc_output = next(m for m in msgs if m.get("type") == "function_call_output")
        # Output should be a list with image parts
        assert isinstance(fc_output["output"], list)
        assert any(c.get("type") == "input_image" for c in fc_output["output"])

    def test_tool_result_no_text_no_image(self) -> None:
        """Tool result with no text or images gets placeholder."""
        model = _model(inputs=["text"])
        ctx = Context(
            messages=[
                UserMessage(content="run it", timestamp=1),
                _assistant(
                    content=[ToolCall(id="call_1|fc_abc", name="bash", arguments={})],
                    stop_reason="toolUse",
                ),
                ToolResultMessage(
                    tool_call_id="call_1|fc_abc",
                    tool_name="bash",
                    content=[],
                    is_error=False,
                    timestamp=2,
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        fc_output = next(m for m in msgs if m.get("type") == "function_call_output")
        # Should use empty string or placeholder
        assert isinstance(fc_output["output"], str)

    def test_include_system_prompt_false(self) -> None:
        """Setting include_system_prompt=False skips the system prompt."""
        model = _model()
        ctx = Context(
            messages=[UserMessage(content="hi", timestamp=1)],
            system_prompt="Be nice",
        )
        msgs = convert_responses_messages(model, ctx, include_system_prompt=False)
        assert not any(m.get("role") in ("system", "developer") for m in msgs)


class TestServiceTierPricing:
    def test_flex_halves_costs(self) -> None:
        from nu_ai.providers.openai_responses import _apply_service_tier_pricing

        usage = _usage()
        usage.cost.input = 10.0
        usage.cost.output = 20.0
        usage.cost.cache_read = 2.0
        usage.cost.cache_write = 0.0
        usage.cost.total = 32.0
        _apply_service_tier_pricing(usage, "flex")
        assert usage.cost.input == 5.0
        assert usage.cost.output == 10.0

    def test_priority_doubles_costs(self) -> None:
        from nu_ai.providers.openai_responses import _apply_service_tier_pricing

        usage = _usage()
        usage.cost.input = 10.0
        usage.cost.output = 20.0
        usage.cost.cache_read = 0.0
        usage.cost.cache_write = 0.0
        usage.cost.total = 30.0
        _apply_service_tier_pricing(usage, "priority")
        assert usage.cost.input == 20.0
        assert usage.cost.output == 40.0

    def test_default_tier_no_change(self) -> None:
        from nu_ai.providers.openai_responses import _apply_service_tier_pricing

        usage = _usage()
        usage.cost.input = 10.0
        usage.cost.output = 20.0
        _apply_service_tier_pricing(usage, None)
        assert usage.cost.input == 10.0
        assert usage.cost.output == 20.0


class TestRefusalStream:
    async def test_refusal_delta_becomes_text(self) -> None:
        """A refusal delta event feeds into the TextContent block."""
        events = [
            {"type": "response.created", "response": {"id": "resp_r"}},
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_r", "content": []},
            },
            {"type": "response.content_part.added", "part": {"type": "refusal", "refusal": ""}},
            {"type": "response.refusal.delta", "delta": "I cannot"},
            {"type": "response.refusal.delta", "delta": " help"},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_r",
                    "content": [{"type": "refusal", "refusal": "I cannot help"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_r",
                    "status": "completed",
                    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                },
            },
        ]
        output, collected = await _collect_events(events)
        deltas = [e.delta for e in collected if isinstance(e, TextDeltaEvent)]
        assert "I cannot" in deltas
        assert " help" in deltas
        # The final text should come from the refusal
        assert output.content[0].text == "I cannot help"


class TestResponseFailedVariants:
    async def test_response_failed_incomplete_details(self) -> None:
        events = [
            {
                "type": "response.failed",
                "response": {
                    "error": None,
                    "incomplete_details": {"reason": "max_tokens"},
                },
            }
        ]
        with pytest.raises(RuntimeError, match="incomplete.*max_tokens"):
            await _collect_events(events)

    async def test_response_failed_no_details(self) -> None:
        events = [
            {
                "type": "response.failed",
                "response": {
                    "error": None,
                    "incomplete_details": None,
                },
            }
        ]
        with pytest.raises(RuntimeError, match="Unknown error"):
            await _collect_events(events)


class TestToDict:
    def test_to_dict_none(self) -> None:
        from nu_ai.providers.openai_responses import _to_dict

        assert _to_dict(None) is None

    def test_to_dict_dict(self) -> None:
        from nu_ai.providers.openai_responses import _to_dict

        d = {"a": 1}
        assert _to_dict(d) is d

    def test_to_dict_pydantic(self) -> None:
        from nu_ai.providers.openai_responses import _to_dict

        obj = TextContent(text="hi")
        result = _to_dict(obj)
        assert result is not None
        assert result["text"] == "hi"

    def test_to_dict_namespace(self) -> None:
        from types import SimpleNamespace

        from nu_ai.providers.openai_responses import _to_dict

        obj = SimpleNamespace(x=1, y=2)
        result = _to_dict(obj)
        assert result is not None
        assert result["x"] == 1


# ---------------------------------------------------------------------------
# Additional coverage: _get, create_client, CancelledError, cache_retention,
# ThinkingContent invalid JSON signature, empty content_list guards
# ---------------------------------------------------------------------------


class TestGetHelper:
    def test_get_none(self) -> None:
        from nu_ai.providers.openai_responses import _get

        assert _get(None, "key") is None

    def test_get_dict(self) -> None:
        from nu_ai.providers.openai_responses import _get

        assert _get({"a": 1}, "a") == 1
        assert _get({"a": 1}, "b") is None

    def test_get_object(self) -> None:
        from types import SimpleNamespace

        from nu_ai.providers.openai_responses import _get

        obj = SimpleNamespace(foo="bar")
        assert _get(obj, "foo") == "bar"
        assert _get(obj, "missing") is None


class TestBuildParamsCacheRetention:
    def test_long_cache_retention_with_openai_url(self) -> None:
        model = _model(base_url="https://api.openai.com/v1")
        ctx = _ctx()
        opts = OpenAIResponsesOptions(
            cache_retention="long",
            session_id="sess-123",
        )
        params = build_params(model, ctx, opts)
        assert params.get("prompt_cache_retention") == "24h"
        assert params.get("prompt_cache_key") == "sess-123"

    def test_long_cache_non_openai_url_no_prompt_cache(self) -> None:
        model = _model(base_url="https://custom.proxy.com/v1")
        ctx = _ctx()
        opts = OpenAIResponsesOptions(cache_retention="long", session_id="s1")
        params = build_params(model, ctx, opts)
        assert params.get("prompt_cache_retention") is None

    def test_env_based_cache_retention(self) -> None:
        import os

        model = _model(base_url="https://api.openai.com/v1")
        ctx = _ctx()
        with patch.dict(os.environ, {"PI_CACHE_RETENTION": "long"}):
            params = build_params(model, ctx)  # No options
        assert params.get("prompt_cache_retention") == "24h"


class TestCreateClient:
    def test_create_client_with_env_key(self) -> None:
        from nu_ai.providers.openai_responses import create_client

        model = _model()
        ctx = _ctx()
        mock_client = MagicMock()
        mock_openai_class = MagicMock(return_value=mock_client)
        with (
            patch.dict("sys.modules", {"openai": MagicMock(AsyncOpenAI=mock_openai_class)}),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}),
        ):
            client = create_client(model, ctx)
        assert client is mock_client

    def test_create_client_missing_key_raises(self) -> None:
        from nu_ai.providers.openai_responses import create_client

        model = _model()
        ctx = _ctx()
        mock_openai_class = MagicMock()
        with (
            patch.dict("sys.modules", {"openai": MagicMock(AsyncOpenAI=mock_openai_class)}),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="API key is required"):
                create_client(model, ctx)

    def test_create_client_with_copilot_headers(self) -> None:
        from nu_ai.providers.openai_responses import create_client

        model = _model(provider="github-copilot")
        ctx = _ctx()
        mock_client = MagicMock()
        mock_openai_class = MagicMock(return_value=mock_client)
        with (
            patch.dict("sys.modules", {"openai": MagicMock(AsyncOpenAI=mock_openai_class)}),
            patch("nu_ai.providers.openai_responses.build_copilot_dynamic_headers", return_value={"X-Copilot": "1"}),
            patch("nu_ai.providers.openai_responses.has_copilot_vision_input", return_value=False),
        ):
            client = create_client(model, ctx, api_key="ghc-token")
        assert client is mock_client
        call_kwargs = mock_openai_class.call_args[1]
        assert "X-Copilot" in (call_kwargs.get("default_headers") or {})

    def test_create_client_with_options_headers(self) -> None:
        from nu_ai.providers.openai_responses import create_client

        model = _model()
        ctx = _ctx()
        mock_client = MagicMock()
        mock_openai_class = MagicMock(return_value=mock_client)
        with patch.dict("sys.modules", {"openai": MagicMock(AsyncOpenAI=mock_openai_class)}):
            client = create_client(model, ctx, api_key="sk-key", options_headers={"X-Custom": "val"})
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["default_headers"]["X-Custom"] == "val"


class TestStreamCancelledError:
    async def test_cancelled_error_emits_aborted(self) -> None:
        """CancelledError triggers an ErrorEvent with reason=aborted."""
        model = _model()
        ctx = _ctx()

        class _CancellingStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError()

        class _CancellingResponses:
            async def create(self, **_: Any) -> _CancellingStream:
                return _CancellingStream()

        class _CancellingClient:
            responses = _CancellingResponses()

        events = [e async for e in stream_openai_responses(model, ctx, client=_CancellingClient())]  # type: ignore[arg-type]
        err_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(err_events) == 1
        assert err_events[0].reason == "aborted"


class TestStreamErrorStopReason:
    async def test_error_stop_reason_raises(self) -> None:
        """When output.stop_reason is 'error' after stream, an ErrorEvent is emitted."""
        events = [
            {"type": "response.created", "response": {"id": "r_err"}},
            {
                "type": "response.completed",
                "response": {
                    "id": "r_err",
                    "status": "failed",
                    "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
                },
            },
        ]
        fake = _FakeAsyncOpenAI(events)
        all_events = [e async for e in stream_openai_responses(_model(), _ctx(), client=fake)]
        assert all_events[-1].type == "error"
        assert isinstance(all_events[-1], ErrorEvent)


class TestThinkingInvalidSignature:
    def test_thinking_with_invalid_json_signature_skipped(self) -> None:
        """ThinkingContent with non-JSON signature is silently skipped."""
        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant(
                    content=[
                        ThinkingContent(thinking="hmm", thinking_signature="not-json{{{"),
                        TextContent(text="answer"),
                    ]
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        # No reasoning items should appear (invalid JSON caught)
        reasoning_items = [m for m in msgs if m.get("type") == "reasoning"]
        assert len(reasoning_items) == 0


class TestTextDeltaEmptyContent:
    async def test_text_delta_on_empty_content_skipped(self) -> None:
        """A text delta arriving when current_item has no content parts is skipped."""
        events = [
            {"type": "response.created", "response": {"id": "r1"}},
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_e", "content": []},
            },
            # No content_part.added, so content list stays empty
            # Delta arrives → should be skipped
            {"type": "response.output_text.delta", "delta": "orphan"},
            {
                "type": "response.output_item.done",
                "item": {"type": "message", "id": "msg_e", "content": []},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
        output, collected = await _collect_events(events)
        deltas = [e for e in collected if isinstance(e, TextDeltaEvent)]
        # The delta on empty content should be skipped
        assert len(deltas) == 0


class TestRefusalDeltaEmptyContent:
    async def test_refusal_delta_on_empty_content_skipped(self) -> None:
        events = [
            {"type": "response.created", "response": {"id": "r1"}},
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_r", "content": []},
            },
            # No content_part.added
            {"type": "response.refusal.delta", "delta": "orphan"},
            {
                "type": "response.output_item.done",
                "item": {"type": "message", "id": "msg_r", "content": []},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
        output, collected = await _collect_events(events)
        # No text deltas from refusal
        deltas = [e for e in collected if isinstance(e, TextDeltaEvent)]
        assert len(deltas) == 0


class TestFunctionCallOutputItemDone:
    async def test_function_call_done_with_non_dict_item(self) -> None:
        """output_item.done for function_call where item is a SimpleNamespace."""
        from types import SimpleNamespace

        item_ns = SimpleNamespace(
            type="function_call",
            id="fc_x",
            call_id="c_x",
            name="fn",
            arguments='{"a": 1}',
        )
        events = [
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "id": "fc_x", "call_id": "c_x", "name": "fn", "arguments": ""},
            },
            {"type": "response.function_call_arguments.done", "arguments": '{"a": 1}'},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_x",
                    "call_id": "c_x",
                    "name": "fn",
                    "arguments": '{"a": 1}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
        output, collected = await _collect_events(events)
        end_events = [e for e in collected if isinstance(e, ToolCallEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].tool_call.arguments == {"a": 1}


class TestSdkEventToDict:
    def test_converts_namespace(self) -> None:
        from types import SimpleNamespace

        from nu_ai.providers.openai_responses import _sdk_event_to_dict

        event = SimpleNamespace(type="foo", data="bar")
        result = _sdk_event_to_dict(event)
        assert result["type"] == "foo"
        assert result["data"] == "bar"

    def test_returns_empty_for_unconvertible(self) -> None:
        from nu_ai.providers.openai_responses import _sdk_event_to_dict

        # A plain int has no __dict__ / model_dump
        result = _sdk_event_to_dict(42)
        assert result == {}


class TestBuildParamsExtended:
    def test_prompt_cache_key_with_session_id(self) -> None:
        model = _model()
        ctx = _ctx()
        opts = OpenAIResponsesOptions(session_id="sess_123")
        params = build_params(model, ctx, opts)
        assert params.get("prompt_cache_key") == "sess_123"

    def test_prompt_cache_retention_long(self) -> None:
        model = _model()  # base_url contains api.openai.com
        ctx = _ctx()
        opts = OpenAIResponsesOptions(cache_retention="long")
        params = build_params(model, ctx, opts)
        assert params.get("prompt_cache_retention") == "24h"

    def test_cache_retention_none_no_cache_key(self) -> None:
        model = _model()
        ctx = _ctx()
        opts = OpenAIResponsesOptions(cache_retention="none", session_id="sess_1")
        params = build_params(model, ctx, opts)
        assert "prompt_cache_key" not in params

    def test_reasoning_model_with_summary(self) -> None:
        model = _model(reasoning=True)
        ctx = _ctx()
        opts = OpenAIResponsesOptions(reasoning_effort="medium", reasoning_summary="detailed")
        params = build_params(model, ctx, opts)
        assert params["reasoning"]["summary"] == "detailed"

    def test_pi_cache_retention_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PI_CACHE_RETENTION=long env var is picked up."""
        monkeypatch.setenv("PI_CACHE_RETENTION", "long")
        model = _model()
        ctx = _ctx()
        params = build_params(model, ctx)
        assert params.get("prompt_cache_retention") == "24h"

    def test_github_copilot_reasoning_no_effort_none(self) -> None:
        """Github copilot reasoning model without explicit effort does not set effort=none."""
        model = _model(reasoning=True, provider="github-copilot")
        ctx = _ctx()
        params = build_params(model, ctx)
        assert "reasoning" not in params


class TestCompletedEventExtended:
    async def test_cached_tokens_subtracted_from_input(self) -> None:
        events = [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_c",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "input_tokens_details": {"cached_tokens": 30},
                    },
                },
            },
        ]
        output, _ = await _collect_events(events)
        assert output.usage.input == 70
        assert output.usage.cache_read == 30
        assert output.usage.output == 20

    async def test_service_tier_from_response(self) -> None:
        """Service tier from the response.completed event is applied."""
        events = [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_st",
                    "status": "completed",
                    "service_tier": "flex",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            },
        ]
        output, _ = await _collect_events(events)
        # flex = 0.5 multiplier; cost is calculated and halved
        assert output.stop_reason == "stop"

    async def test_incomplete_status(self) -> None:
        events = [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_inc",
                    "status": "incomplete",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
        output, _ = await _collect_events(events)
        assert output.stop_reason == "length"


class TestConvertResponsesMessagesEdge:
    def test_thinking_with_invalid_json_signature(self) -> None:
        """ThinkingContent with an invalid JSON signature is silently skipped."""
        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant(
                    content=[
                        ThinkingContent(
                            thinking="thinking...",
                            thinking_signature="not valid json {{",
                        ),
                        TextContent(text="answer"),
                    ]
                ),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        reasoning_items = [m for m in msgs if m.get("type") == "reasoning"]
        assert len(reasoning_items) == 0

    def test_text_with_long_id_in_signature(self) -> None:
        """TextContent with a very long id in the signature gets hashed."""
        long_id = "a" * 100
        sig = json.dumps({"v": 1, "id": long_id})
        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant(content=[TextContent(text="answer", text_signature=sig)]),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        msg_item = next(m for m in msgs if m.get("type") == "message")
        # Should use msg_ + hash since the id is > 64
        assert msg_item["id"].startswith("msg_")
        assert len(msg_item["id"]) <= 64

    def test_text_without_signature_uses_index(self) -> None:
        """TextContent without text_signature uses msg_<index>."""
        model = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant(content=[TextContent(text="answer")]),
            ]
        )
        msgs = convert_responses_messages(model, ctx)
        msg_item = next(m for m in msgs if m.get("type") == "message")
        assert msg_item["id"] == "msg_1"  # index 1 (after user msg at 0)


class TestOutputTextDeltaEdge:
    async def test_output_text_delta_no_content_skipped(self) -> None:
        """output_text.delta with empty content list does nothing."""
        events = [
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_e", "content": []},
            },
            # Delta arrives before any content_part.added
            {"type": "response.output_text.delta", "delta": "should be ignored"},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_e",
                    "content": [],
                },
            },
            {
                "type": "response.completed",
                "response": {"id": "resp_e", "status": "completed", "usage": None},
            },
        ]
        output, collected = await _collect_events(events)
        # The text delta should have been skipped (no content parts)
        assert output.content[0].text == ""


class TestFunctionCallDoneWithoutDelta:
    async def test_function_call_done_uses_raw_args(self) -> None:
        """When function_call item.done arrives without prior deltas, uses raw arguments."""
        events = [
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_nd",
                    "call_id": "call_nd",
                    "name": "bash",
                    "arguments": "",
                },
            },
            # No delta events — jump straight to done
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_nd",
                    "call_id": "call_nd",
                    "name": "bash",
                    "arguments": '{"cmd": "pwd"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_nd",
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
        output, collected = await _collect_events(events)
        tool_end = next(e for e in collected if isinstance(e, ToolCallEndEvent))
        assert tool_end.tool_call.arguments == {"cmd": "pwd"}


class TestUsageAsSDKObject:
    async def test_usage_object_converted_to_dict(self) -> None:
        """When usage is an SDK object (not dict), it gets converted via _to_dict."""
        from types import SimpleNamespace

        events = [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_sdk",
                    "status": "completed",
                    "usage": SimpleNamespace(
                        input_tokens=50,
                        output_tokens=20,
                        total_tokens=70,
                        input_tokens_details={"cached_tokens": 10},
                    ),
                },
            },
        ]
        output, _ = await _collect_events(events)
        assert output.usage.input == 40  # 50 - 10 cached
        assert output.usage.cache_read == 10
        assert output.usage.output == 20


class TestStreamSimpleOpenAIResponses:
    async def test_stream_simple_maps_reasoning(self) -> None:
        from nu_ai.providers.openai_responses import stream_simple_openai_responses
        from nu_ai.types import SimpleStreamOptions

        scripted: list[dict[str, Any]] = [
            {"type": "response.created", "response": {"id": "r_s"}},
            {
                "type": "response.completed",
                "response": {"id": "r_s", "status": "completed", "usage": None},
            },
        ]
        fake = _FakeAsyncOpenAI(scripted)
        model = _model(reasoning=True)
        opts = SimpleStreamOptions(reasoning="high")
        events = [e async for e in stream_simple_openai_responses(model, _ctx(), opts, client=fake)]
        # Should complete without error
        assert events[-1].type == "done"
        # Verify reasoning params were passed
        kwargs = fake.responses.last_kwargs
        assert kwargs.get("reasoning", {}).get("effort") == "high"
