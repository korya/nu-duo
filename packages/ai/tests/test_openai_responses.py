"""Tests for nu_ai.providers.openai_responses.

Covers pure transform functions (request building, message conversion, tool
conversion, stop-reason mapping) as well as the stream event mapping via a
fake async client that yields scripted Responses API events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

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
