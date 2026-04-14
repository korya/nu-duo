"""Tests for nu_ai.providers.google streaming.

Uses a fake ``google.genai.Client``-shaped client whose
``client.aio.models.generate_content_stream(...)`` returns an async iterator
yielding scripted chunks with the same shape as the real
``GenerateContentResponse`` (``.candidates[0].content.parts``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from nu_ai.providers.google import stream_google
from nu_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    DoneEvent,
    ErrorEvent,
    GoogleOptions,
    Model,
    ModelCost,
    TextContent,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _ns(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


@dataclass
class _FakeAsyncStream:
    chunks: list[SimpleNamespace]

    def __aiter__(self) -> _FakeAsyncStream:
        self._iter = iter(self.chunks)
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@dataclass
class _FakeAsyncModels:
    chunks: list[SimpleNamespace]
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    async def generate_content_stream(self, **kwargs: Any) -> _FakeAsyncStream:
        self.last_kwargs = kwargs
        return _FakeAsyncStream(self.chunks)


@dataclass
class _FakeAio:
    models: _FakeAsyncModels


@dataclass
class _FakeGoogleClient:
    chunks: list[SimpleNamespace]
    aio: _FakeAio = field(init=False)

    def __post_init__(self) -> None:
        self.aio = _FakeAio(models=_FakeAsyncModels(self.chunks))


def _model(model_id: str = "gemini-2.5-flash") -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="google-generative-ai",
        provider="google",
        base_url="",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0.3, output=2.5, cache_read=0.075, cache_write=0),
        context_window=1_048_576,
        max_tokens=8192,
    )


def _ctx(text: str = "hello") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=1)])


def _text_part(text: str, *, thought: bool = False) -> SimpleNamespace:
    return _ns(text=text, thought=thought, function_call=None, thought_signature=None)


def _function_call_part(name: str, args: dict[str, Any], call_id: str | None = None) -> SimpleNamespace:
    return _ns(
        text=None,
        thought=None,
        thought_signature=None,
        function_call=_ns(name=name, args=args, id=call_id),
    )


def _chunk(parts: list[SimpleNamespace], *, finish_reason: str | None = None) -> SimpleNamespace:
    return _ns(
        response_id="resp_x",
        candidates=[
            _ns(
                content=_ns(parts=parts),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=None,
    )


def _usage_chunk(prompt: int, candidates: int, cached: int = 0, thoughts: int = 0) -> SimpleNamespace:
    return _ns(
        response_id="resp_x",
        candidates=[],
        usage_metadata=_ns(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            cached_content_token_count=cached,
            thoughts_token_count=thoughts,
            total_token_count=prompt + candidates + thoughts,
        ),
    )


class TestTextStream:
    async def test_basic_text(self) -> None:
        chunks = [
            _chunk([_text_part("Hello")]),
            _chunk([_text_part(" world")]),
            _chunk([], finish_reason="STOP"),
            _usage_chunk(prompt=10, candidates=5),
        ]
        fake = _FakeGoogleClient(chunks)
        events = [e async for e in stream_google(_model(), _ctx(), GoogleOptions(), client=fake)]
        types = [e.type for e in events]
        assert types == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]

        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert [e.delta for e in deltas] == ["Hello", " world"]

        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.message.usage.input == 10
        assert done.message.usage.output == 5
        assert done.message.response_id == "resp_x"


class TestThinkingStream:
    async def test_thought_part_becomes_thinking(self) -> None:
        chunks = [
            _chunk([_text_part("reasoning", thought=True)]),
            _chunk([_text_part("final answer")]),
            _chunk([], finish_reason="STOP"),
            _usage_chunk(prompt=5, candidates=3, thoughts=2),
        ]
        fake = _FakeGoogleClient(chunks)
        events = [
            e
            async for e in stream_google(
                _model(),
                _ctx(),
                GoogleOptions(),
                client=fake,
            )
        ]
        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_end" in types
        assert "text_start" in types

        thinking_delta = next(e for e in events if isinstance(e, ThinkingDeltaEvent))
        assert thinking_delta.delta == "reasoning"

        done = events[-1]
        assert isinstance(done, DoneEvent)
        thinking = next(b for b in done.message.content if isinstance(b, ThinkingContent))
        assert thinking.thinking == "reasoning"
        # Thoughts go to output token count.
        assert done.message.usage.output == 5  # candidates(3) + thoughts(2)


class TestToolCallStream:
    async def test_function_call_emits_tool_call_events(self) -> None:
        chunks = [
            _chunk([_function_call_part("bash", {"cmd": "ls"}, call_id="call_1")]),
            _chunk([], finish_reason="STOP"),  # tool_use detected via content
            _usage_chunk(prompt=8, candidates=4),
        ]
        fake = _FakeGoogleClient(chunks)
        events = [e async for e in stream_google(_model(), _ctx("run ls"), GoogleOptions(), client=fake)]
        tool_end = next(e for e in events if isinstance(e, ToolCallEndEvent))
        assert isinstance(tool_end.tool_call, ToolCall)
        assert tool_end.tool_call.name == "bash"
        assert tool_end.tool_call.arguments == {"cmd": "ls"}
        assert tool_end.tool_call.id == "call_1"

        done = events[-1]
        assert isinstance(done, DoneEvent)
        # Provider promotes stop → toolUse when content has a tool call.
        assert done.reason == "toolUse"

    async def test_function_call_without_id_generates_one(self) -> None:
        chunks = [
            _chunk([_function_call_part("bash", {"cmd": "ls"}, call_id=None)]),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeGoogleClient(chunks)
        events = [e async for e in stream_google(_model(), _ctx(), GoogleOptions(), client=fake)]
        tool_end = next(e for e in events if isinstance(e, ToolCallEndEvent))
        assert tool_end.tool_call.id  # non-empty
        assert tool_end.tool_call.id.startswith("bash_")


class TestErrorHandling:
    async def test_sdk_exception_becomes_error_event(self) -> None:
        class BoomModels:
            def __init__(self) -> None:
                self.last_kwargs: dict[str, Any] = {}

            async def generate_content_stream(self, **_: Any) -> None:
                raise RuntimeError("boom")

        fake = SimpleNamespace(aio=SimpleNamespace(models=BoomModels()))
        events = [e async for e in stream_google(_model(), _ctx(), GoogleOptions(), client=fake)]
        err = events[-1]
        assert isinstance(err, ErrorEvent)
        assert err.reason == "error"
        assert "boom" in (err.error.error_message or "")


class TestPayloadCapture:
    async def test_payload_shape(self) -> None:
        chunks = [
            _chunk([_text_part("ok")]),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeGoogleClient(chunks)
        events = [
            e
            async for e in stream_google(
                _model(),
                Context(
                    system_prompt="be helpful",
                    messages=[UserMessage(content="ping", timestamp=1)],
                ),
                GoogleOptions(temperature=0.5, max_tokens=200),
                client=fake,
            )
        ]
        assert events[-1].type == "done"
        kwargs = fake.aio.models.last_kwargs
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["contents"][0]["role"] == "user"
        config = kwargs["config"]
        assert config["temperature"] == 0.5
        assert config["max_output_tokens"] == 200
        assert config["system_instruction"] == "be helpful"


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestGoogleHelpers:
    def test_is_thinking_part_dict(self) -> None:
        from nu_ai.providers.google import is_thinking_part

        assert is_thinking_part({"thought": True, "text": "hi"}) is True
        assert is_thinking_part({"thought": False, "text": "hi"}) is False
        assert is_thinking_part({"text": "hi"}) is False

    def test_is_thinking_part_object(self) -> None:
        from nu_ai.providers.google import is_thinking_part

        assert is_thinking_part(_ns(thought=True)) is True
        assert is_thinking_part(_ns(thought=False)) is False
        assert is_thinking_part(_ns()) is False

    def test_retain_thought_signature(self) -> None:
        from nu_ai.providers.google import retain_thought_signature

        assert retain_thought_signature(None, "new_sig") == "new_sig"
        assert retain_thought_signature("old", None) == "old"
        assert retain_thought_signature("old", "") == "old"
        assert retain_thought_signature("old", "new") == "new"

    def test_is_valid_thought_signature(self) -> None:
        from nu_ai.providers.google import _is_valid_thought_signature

        assert _is_valid_thought_signature(None) is False
        assert _is_valid_thought_signature("") is False
        assert _is_valid_thought_signature("abc") is False  # len not multiple of 4
        assert _is_valid_thought_signature("YWJj") is True  # valid base64 "abc"
        assert _is_valid_thought_signature("YWJj==") is False  # wrong padding

    def test_resolve_thought_signature(self) -> None:
        from nu_ai.providers.google import _resolve_thought_signature

        # Same provider with valid signature
        assert _resolve_thought_signature(True, "YWJj") == "YWJj"
        # Same provider with invalid signature
        assert _resolve_thought_signature(True, "abc") is None
        # Different provider
        assert _resolve_thought_signature(False, "YWJj") is None

    def test_requires_tool_call_id(self) -> None:
        from nu_ai.providers.google import requires_tool_call_id

        assert requires_tool_call_id("claude-3.5-sonnet") is True
        assert requires_tool_call_id("gpt-oss-mini") is True
        assert requires_tool_call_id("gemini-2.5-flash") is False

    def test_get_gemini_major_version(self) -> None:
        from nu_ai.providers.google import _get_gemini_major_version

        assert _get_gemini_major_version("gemini-2.5-flash") == 2
        assert _get_gemini_major_version("gemini-3-pro") == 3
        assert _get_gemini_major_version("gemini-live-2") == 2
        assert _get_gemini_major_version("llama-3") is None

    def test_supports_multimodal_function_response(self) -> None:
        from nu_ai.providers.google import _supports_multimodal_function_response

        assert _supports_multimodal_function_response("gemini-3-pro") is True
        assert _supports_multimodal_function_response("gemini-2.5-flash") is False
        assert _supports_multimodal_function_response("claude-3.5") is True  # non-gemini defaults to True

    def test_model_family_detection(self) -> None:
        from nu_ai.providers.google import (
            is_gemini_3_flash_model,
            is_gemini_3_pro_model,
            is_gemma_4_model,
        )
        from nu_ai.types import ModelCost

        def _mk(model_id: str) -> Model:
            return Model(
                id=model_id,
                name=model_id,
                api="google",
                provider="google",
                base_url="",
                reasoning=True,
                input=["text"],
                cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
                context_window=1000,
                max_tokens=1000,
            )

        assert is_gemini_3_pro_model(_mk("gemini-3-pro")) is True
        assert is_gemini_3_pro_model(_mk("gemini-3.5-pro")) is True
        assert is_gemini_3_flash_model(_mk("gemini-3-flash")) is True
        assert is_gemini_3_flash_model(_mk("gemini-3.5-flash")) is True
        assert is_gemma_4_model(_mk("gemma-4")) is True
        assert is_gemma_4_model(_mk("gemma4")) is True
        assert is_gemma_4_model(_mk("gemini-3-pro")) is False


class TestGoogleThinkingHelpers:
    def test_get_thinking_level_gemini_3_pro(self) -> None:
        from nu_ai.providers.google import get_thinking_level
        from nu_ai.types import ModelCost

        m = Model(
            id="gemini-3-pro",
            name="gemini-3-pro",
            api="google",
            provider="google",
            base_url="",
            reasoning=True,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        assert get_thinking_level("minimal", m) == "LOW"
        assert get_thinking_level("low", m) == "LOW"
        assert get_thinking_level("medium", m) == "HIGH"
        assert get_thinking_level("high", m) == "HIGH"

    def test_get_thinking_level_gemma4(self) -> None:
        from nu_ai.providers.google import get_thinking_level
        from nu_ai.types import ModelCost

        m = Model(
            id="gemma-4",
            name="gemma-4",
            api="google",
            provider="google",
            base_url="",
            reasoning=True,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        assert get_thinking_level("minimal", m) == "MINIMAL"
        assert get_thinking_level("low", m) == "MINIMAL"
        assert get_thinking_level("high", m) == "HIGH"

    def test_get_thinking_level_default_mapping(self) -> None:
        from nu_ai.providers.google import get_thinking_level

        m = _model("gemini-2.5-flash")
        assert get_thinking_level("minimal", m) == "MINIMAL"
        assert get_thinking_level("low", m) == "LOW"
        assert get_thinking_level("medium", m) == "MEDIUM"
        assert get_thinking_level("high", m) == "HIGH"

    def test_get_google_budget(self) -> None:
        from nu_ai.providers.google import get_google_budget

        m25p = _model("gemini-2.5-pro")
        assert get_google_budget(m25p, "minimal") == 128
        assert get_google_budget(m25p, "high") == 32768

        m25f = _model("gemini-2.5-flash")
        assert get_google_budget(m25f, "low") == 2048

        m25fl = _model("gemini-2.5-flash-lite")
        assert get_google_budget(m25fl, "minimal") == 512

        # Unknown model returns -1
        other = _model("gemini-3-pro")
        assert get_google_budget(other, "high") == -1

    def test_get_google_budget_custom(self) -> None:
        from nu_ai.providers.google import get_google_budget
        from nu_ai.types import ThinkingBudgets

        m = _model("gemini-2.5-flash")
        budgets = ThinkingBudgets(low=999)
        assert get_google_budget(m, "low", budgets) == 999
        # Non-overridden level falls back to default
        assert get_google_budget(m, "high", budgets) == 24576

    def test_get_disabled_thinking_config(self) -> None:
        from nu_ai.providers.google import _get_disabled_thinking_config
        from nu_ai.types import ModelCost

        def _mk(model_id: str) -> Model:
            return Model(
                id=model_id,
                name=model_id,
                api="google",
                provider="google",
                base_url="",
                reasoning=True,
                input=["text"],
                cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
                context_window=1000,
                max_tokens=1000,
            )

        assert _get_disabled_thinking_config(_mk("gemini-3-pro")) == {"thinking_level": "LOW"}
        assert _get_disabled_thinking_config(_mk("gemini-3-flash")) == {"thinking_level": "MINIMAL"}
        assert _get_disabled_thinking_config(_mk("gemma-4")) == {"thinking_level": "MINIMAL"}
        assert _get_disabled_thinking_config(_mk("gemini-2.5-flash")) == {"thinking_budget": 0}


class TestMapStopReason:
    def test_stop(self) -> None:
        from nu_ai.providers.google import map_stop_reason

        assert map_stop_reason("STOP") == "stop"

    def test_max_tokens(self) -> None:
        from nu_ai.providers.google import map_stop_reason

        assert map_stop_reason("MAX_TOKENS") == "length"

    def test_unknown(self) -> None:
        from nu_ai.providers.google import map_stop_reason

        assert map_stop_reason("SAFETY") == "error"

    def test_none(self) -> None:
        from nu_ai.providers.google import map_stop_reason

        assert map_stop_reason(None) == "error"

    def test_enum_style(self) -> None:
        from nu_ai.providers.google import map_stop_reason

        assert map_stop_reason("FinishReason.STOP") == "stop"


class TestMapToolChoice:
    def test_known_choices(self) -> None:
        from nu_ai.providers.google import map_tool_choice

        assert map_tool_choice("auto") == "AUTO"
        assert map_tool_choice("none") == "NONE"
        assert map_tool_choice("any") == "ANY"
        assert map_tool_choice("unknown") == "AUTO"


class TestConvertTools:
    def test_basic_conversion(self) -> None:
        from nu_ai.providers.google import convert_tools
        from nu_ai.types import Tool

        tools = [Tool(name="bash", description="Run bash", parameters={"type": "object", "properties": {}})]
        result = convert_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function_declarations"][0]["name"] == "bash"

    def test_empty_tools(self) -> None:
        from nu_ai.providers.google import convert_tools

        assert convert_tools([]) is None

    def test_use_parameters_flag(self) -> None:
        from nu_ai.providers.google import convert_tools
        from nu_ai.types import Tool

        tools = [
            Tool(name="t", description="d", parameters={"type": "object", "properties": {"a": {"type": "string"}}})
        ]
        result = convert_tools(tools, use_parameters=True)
        assert result is not None
        assert "parameters" in result[0]["function_declarations"][0]
        assert "parameters_json_schema" not in result[0]["function_declarations"][0]


class TestConvertMessagesBasic:
    def test_tool_call_id_normalizer_claude(self) -> None:
        """Claude models require tool call ids to be normalized."""
        from nu_ai.providers.google import _make_tool_call_id_normalizer
        from nu_ai.types import ModelCost

        m = Model(
            id="claude-3.5-sonnet",
            name="claude",
            api="google",
            provider="google",
            base_url="",
            reasoning=False,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        normalizer = _make_tool_call_id_normalizer(m)
        # Claude requires normalized ids
        result = normalizer("invalid id!@#")
        assert "!" not in result
        assert "@" not in result

    def test_assistant_with_tool_call_same_provider(self) -> None:
        """ToolCall in same-provider assistant message."""
        from nu_ai.providers.google import convert_messages

        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-2.5-flash",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        model_msg = contents[1]
        assert model_msg["role"] == "model"
        fc = model_msg["parts"][0]["function_call"]
        assert fc["name"] == "bash"

    def test_assistant_tool_call_gemini3_skip_signature(self) -> None:
        """Gemini 3 tool calls without thought_signature get skip_thought_signature."""
        from nu_ai.providers.google import convert_messages
        from nu_ai.types import ModelCost

        m = Model(
            id="gemini-3-pro",
            name="gemini-3-pro",
            api="google-generative-ai",
            provider="google",
            base_url="",
            reasoning=True,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[ToolCall(id="c1", name="bash", arguments={})],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-3-pro",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        model_msg = contents[1]
        # Should have thought_signature = skip_thought_signature_validator
        assert model_msg["parts"][0].get("thought_signature") == "skip_thought_signature_validator"

    def test_assistant_same_provider_thinking_with_valid_signature(self) -> None:
        """Thinking block from same provider with valid base64 signature."""
        import base64

        from nu_ai.providers.google import convert_messages

        valid_sig = base64.b64encode(b"test signature data here").decode()
        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[
                        ThinkingContent(thinking="pondering...", thinking_signature=valid_sig),
                        TextContent(text="answer"),
                    ],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-2.5-flash",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="stop",
                    timestamp=1,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        model_msg = contents[1]
        parts = model_msg["parts"]
        # First part should be a thinking part with signature
        thinking_part = parts[0]
        assert thinking_part["thought"] is True
        assert thinking_part.get("thought_signature") == valid_sig

    def test_assistant_empty_text_skipped(self) -> None:
        """Empty text blocks in assistant messages are skipped."""
        from nu_ai.providers.google import convert_messages

        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[TextContent(text="   ")],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-2.5-flash",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="stop",
                    timestamp=1,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        # Only user message should be present, empty assistant dropped
        assert len(contents) == 1

    def test_tool_result_error(self) -> None:
        """Tool result with is_error=True uses error key."""
        from nu_ai.providers.google import convert_messages

        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[ToolCall(id="c1", name="bash", arguments={})],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-2.5-flash",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="bash",
                    content=[TextContent(text="command not found")],
                    is_error=True,
                    timestamp=2,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        tool_result_msg = [
            c for c in contents if c["role"] == "user" and any("function_response" in p for p in c.get("parts", []))
        ]
        assert len(tool_result_msg) == 1
        fr = tool_result_msg[0]["parts"][0]["function_response"]
        assert "error" in fr["response"]
        assert fr["response"]["error"] == "command not found"

    def test_tool_result_gemini3_multimodal(self) -> None:
        """Gemini 3 supports multimodal function responses — images go inline."""
        from nu_ai.providers.google import convert_messages
        from nu_ai.types import ImageContent, ModelCost

        m = Model(
            id="gemini-3-pro",
            name="gemini-3-pro",
            api="google-generative-ai",
            provider="google",
            base_url="",
            reasoning=True,
            input=["text", "image"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[ToolCall(id="c1", name="shot", arguments={})],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-3-pro",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="shot",
                    content=[
                        TextContent(text="captured"),
                        ImageContent(mime_type="image/png", data="imgdata"),
                    ],
                    is_error=False,
                    timestamp=2,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        # Gemini 3 multimodal: images go as parts on the function_response
        tool_result_msgs = [
            c for c in contents if c["role"] == "user" and any("function_response" in p for p in c.get("parts", []))
        ]
        fr = tool_result_msgs[0]["parts"][0]["function_response"]
        assert "parts" in fr

    def test_tool_result_claude_model_includes_id(self) -> None:
        """Claude model tool results include the tool call id."""
        from nu_ai.providers.google import convert_messages
        from nu_ai.types import ModelCost

        m = Model(
            id="claude-3.5-sonnet",
            name="claude",
            api="google",
            provider="google",
            base_url="",
            reasoning=False,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[ToolCall(id="c1", name="bash", arguments={})],
                    api="google",
                    provider="google",
                    model="claude-3.5-sonnet",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="bash",
                    content=[TextContent(text="ok")],
                    is_error=False,
                    timestamp=2,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        tool_result_msgs = [
            c for c in contents if c["role"] == "user" and any("function_response" in p for p in c.get("parts", []))
        ]
        fr = tool_result_msgs[0]["parts"][0]["function_response"]
        assert "id" in fr

    def test_consecutive_tool_results_collapsed(self) -> None:
        """Two consecutive tool results go into the same user turn."""
        from nu_ai.providers.google import convert_messages

        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[
                        ToolCall(id="c1", name="bash", arguments={}),
                        ToolCall(id="c2", name="grep", arguments={}),
                    ],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-2.5-flash",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="bash",
                    content=[TextContent(text="ok1")],
                    is_error=False,
                    timestamp=2,
                ),
                ToolResultMessage(
                    tool_call_id="c2",
                    tool_name="grep",
                    content=[TextContent(text="ok2")],
                    is_error=False,
                    timestamp=2,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        # Tool results should be collapsed into one user turn
        tool_user_msgs = [
            c for c in contents if c["role"] == "user" and any("function_response" in p for p in c.get("parts", []))
        ]
        assert len(tool_user_msgs) == 1
        assert len(tool_user_msgs[0]["parts"]) == 2

    def test_user_image_only_drops_on_text_model(self) -> None:
        """User message with only images on text-only model is dropped."""
        from nu_ai.providers.google import convert_messages
        from nu_ai.types import ImageContent, ModelCost

        m = Model(
            id="gemini-2.5-flash",
            name="gemini",
            api="google",
            provider="google",
            base_url="",
            reasoning=True,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        ctx = Context(
            messages=[
                UserMessage(
                    content=[ImageContent(mime_type="image/png", data="abc")],
                    timestamp=1,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        assert len(contents) == 0


class TestConvertMessages:
    def test_user_message_with_image(self) -> None:
        from nu_ai.providers.google import convert_messages
        from nu_ai.types import ImageContent

        m = _model()
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="look"),
                        ImageContent(mime_type="image/png", data="abc"),
                    ],
                    timestamp=1,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        parts = contents[0]["parts"]
        assert any("inline_data" in p for p in parts)

    def test_user_image_filtered_on_text_model(self) -> None:
        from nu_ai.providers.google import convert_messages
        from nu_ai.types import ImageContent, ModelCost

        m = Model(
            id="gemini-2.5-flash",
            name="gemini",
            api="google",
            provider="google",
            base_url="",
            reasoning=True,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="hi"),
                        ImageContent(mime_type="image/png", data="abc"),
                    ],
                    timestamp=1,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        parts = contents[0]["parts"]
        assert all("inline_data" not in p for p in parts)

    def test_assistant_thinking_cross_model(self) -> None:
        """Thinking from a different model falls back to plain text."""
        from nu_ai.providers.google import convert_messages
        from nu_ai.types import Cost, ThinkingContent, Usage

        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[
                        ThinkingContent(thinking="thinking..."),
                        TextContent(text="answer"),
                    ],
                    api="other",
                    provider="other",
                    model="other-model",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="stop",
                    timestamp=1,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        model_parts = contents[1]["parts"]
        # Thinking should be plain text, not a thought part
        for p in model_parts:
            assert "thought" not in p or p.get("thought") is not True

    def test_tool_result_with_image_gemini_2(self) -> None:
        """Gemini 2 doesn't support multimodal function response, images go to separate user msg."""
        from nu_ai.providers.google import convert_messages
        from nu_ai.types import Cost, ImageContent, Usage

        m = _model("gemini-2.5-flash")
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    content=[ToolCall(id="c1", name="shot", arguments={})],
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-2.5-flash",
                    usage=Usage(
                        input=0,
                        output=0,
                        cache_read=0,
                        cache_write=0,
                        total_tokens=0,
                        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                    ),
                    stop_reason="toolUse",
                    timestamp=1,
                ),
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="shot",
                    content=[
                        TextContent(text="captured"),
                        ImageContent(mime_type="image/png", data="imgdata"),
                    ],
                    is_error=False,
                    timestamp=2,
                ),
            ]
        )
        contents = convert_messages(m, ctx)
        # Should have a separate user message with image
        last_user = [c for c in contents if c["role"] == "user"]
        assert any("Tool result image:" in str(u.get("parts", "")) for u in last_user)


class TestBuildParamsGoogle:
    def test_thinking_config_enabled_with_level(self) -> None:
        from nu_ai.providers.google import build_params
        from nu_ai.types import GoogleOptions, GoogleThinkingOptions

        m = _model()
        ctx = _ctx()
        opts = GoogleOptions(thinking=GoogleThinkingOptions(enabled=True, level="HIGH"))
        params = build_params(m, ctx, opts)
        assert params["config"]["thinking_config"]["include_thoughts"] is True
        assert params["config"]["thinking_config"]["thinking_level"] == "HIGH"

    def test_thinking_config_enabled_with_budget(self) -> None:
        from nu_ai.providers.google import build_params
        from nu_ai.types import GoogleOptions, GoogleThinkingOptions

        m = _model()
        ctx = _ctx()
        opts = GoogleOptions(thinking=GoogleThinkingOptions(enabled=True, budget_tokens=4096))
        params = build_params(m, ctx, opts)
        assert params["config"]["thinking_config"]["thinking_budget"] == 4096

    def test_thinking_config_disabled(self) -> None:
        from nu_ai.providers.google import build_params
        from nu_ai.types import GoogleOptions, GoogleThinkingOptions

        m = _model()
        ctx = _ctx()
        opts = GoogleOptions(thinking=GoogleThinkingOptions(enabled=False))
        params = build_params(m, ctx, opts)
        # For gemini-2.5, disabled thinking = budget 0
        assert params["config"]["thinking_config"]["thinking_budget"] == 0

    def test_tool_choice(self) -> None:
        from nu_ai.providers.google import build_params
        from nu_ai.types import GoogleOptions, Tool

        m = _model()
        tool = Tool(name="t", description="d", parameters={"type": "object", "properties": {}})
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)], tools=[tool])
        opts = GoogleOptions(tool_choice="any")
        params = build_params(m, ctx, opts)
        assert params["config"]["tool_config"]["function_calling_config"]["mode"] == "ANY"


class TestFunctionCallWithThoughtSignature:
    async def test_function_call_with_signature(self) -> None:
        """Function call part with thought_signature preserves it."""
        fc_part = _ns(
            text=None,
            thought=None,
            thought_signature="sig_abc",
            function_call=_ns(name="bash", args={"cmd": "ls"}, id="call_1"),
        )
        chunks = [
            _chunk([fc_part]),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeGoogleClient(chunks)
        events = [e async for e in stream_google(_model(), _ctx(), GoogleOptions(), client=fake)]
        tool_end = next(e for e in events if isinstance(e, ToolCallEndEvent))
        assert tool_end.tool_call.thought_signature == "sig_abc"


class TestThinkingWithSignature:
    async def test_thought_signature_retained(self) -> None:
        """Thought signature on text parts is retained across deltas."""
        from nu_ai.types import ThinkingContent as _TC

        chunks = [
            _ns(
                response_id="resp_ts",
                candidates=[
                    _ns(
                        content=_ns(
                            parts=[_ns(text="think", thought=True, function_call=None, thought_signature="sig1")]
                        ),
                        finish_reason=None,
                    )
                ],
                usage_metadata=None,
            ),
            _ns(
                response_id="resp_ts",
                candidates=[
                    _ns(
                        content=_ns(parts=[_ns(text="ing", thought=True, function_call=None, thought_signature=None)]),
                        finish_reason=None,
                    )
                ],
                usage_metadata=None,
            ),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeGoogleClient(chunks)
        events = [e async for e in stream_google(_model(), _ctx(), GoogleOptions(), client=fake)]
        done = events[-1]
        assert isinstance(done, DoneEvent)
        thinking = next(b for b in done.message.content if isinstance(b, _TC))
        assert thinking.thinking == "thinking"
        assert thinking.thinking_signature == "sig1"


class TestTextWithSignature:
    async def test_text_signature_retained(self) -> None:
        chunks = [
            _ns(
                response_id="resp_txs",
                candidates=[
                    _ns(
                        content=_ns(
                            parts=[_ns(text="hello", thought=False, function_call=None, thought_signature="txt_sig")]
                        ),
                        finish_reason=None,
                    )
                ],
                usage_metadata=None,
            ),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeGoogleClient(chunks)
        events = [e async for e in stream_google(_model(), _ctx(), GoogleOptions(), client=fake)]
        done = events[-1]
        assert isinstance(done, DoneEvent)
        text_block = next(b for b in done.message.content if isinstance(b, TextContent))
        assert text_block.text_signature == "txt_sig"


class TestStreamSimpleGoogle:
    async def test_stream_simple_with_reasoning(self) -> None:
        from nu_ai.providers.google import stream_simple_google
        from nu_ai.types import SimpleStreamOptions

        chunks = [
            _chunk([_text_part("ok")]),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeGoogleClient(chunks)
        opts = SimpleStreamOptions(reasoning="high")
        events = [e async for e in stream_simple_google(_model(), _ctx(), opts, client=fake)]
        assert events[-1].type == "done"
        kwargs = fake.aio.models.last_kwargs
        config = kwargs.get("config", {})
        assert "thinking_config" in config
        assert config["thinking_config"].get("include_thoughts") is True

    async def test_stream_simple_without_reasoning(self) -> None:
        from nu_ai.providers.google import stream_simple_google
        from nu_ai.types import SimpleStreamOptions

        chunks = [
            _chunk([_text_part("ok")]),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeGoogleClient(chunks)
        opts = SimpleStreamOptions()
        events = [e async for e in stream_simple_google(_model(), _ctx(), opts, client=fake)]
        assert events[-1].type == "done"
        kwargs = fake.aio.models.last_kwargs
        config = kwargs.get("config", {})
        # Thinking disabled
        thinking = config.get("thinking_config", {})
        assert thinking.get("enabled") is not True or thinking.get("thinking_budget") == 0

    async def test_stream_simple_gemini3_pro(self) -> None:
        from nu_ai.providers.google import stream_simple_google
        from nu_ai.types import ModelCost, SimpleStreamOptions

        m = Model(
            id="gemini-3-pro",
            name="gemini-3-pro",
            api="google-generative-ai",
            provider="google",
            base_url="",
            reasoning=True,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=1000,
        )
        chunks = [
            _chunk([_text_part("ok")]),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeGoogleClient(chunks)
        opts = SimpleStreamOptions(reasoning="high")
        events = [e async for e in stream_simple_google(m, _ctx(), opts, client=fake)]
        assert events[-1].type == "done"
        kwargs = fake.aio.models.last_kwargs
        config = kwargs.get("config", {})
        # Gemini 3 pro uses thinking_level instead of budget
        assert "thinking_level" in config.get("thinking_config", {})
