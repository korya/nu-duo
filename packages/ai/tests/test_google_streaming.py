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
    Context,
    DoneEvent,
    ErrorEvent,
    GoogleOptions,
    Model,
    ModelCost,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolCallEndEvent,
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
