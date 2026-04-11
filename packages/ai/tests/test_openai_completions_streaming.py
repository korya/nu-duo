"""Tests for nu_ai.providers.openai_completions streaming.

Uses a fake ``AsyncOpenAI``-shaped client whose
``chat.completions.create(...)`` returns an async iterator yielding
scripted chunks with ``.choices[0].delta`` and ``.usage`` attributes —
matching the shape of ``openai.types.chat.ChatCompletionChunk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from nu_ai.providers.openai_completions import stream_openai_completions
from nu_ai.types import (
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    OpenAICompletionsOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
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
class _FakeCompletions:
    chunks: list[SimpleNamespace]
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    async def create(self, **kwargs: Any) -> _FakeAsyncStream:
        self.last_kwargs = kwargs
        return _FakeAsyncStream(self.chunks)


@dataclass
class _FakeChat:
    completions: _FakeCompletions


@dataclass
class _FakeAsyncOpenAI:
    chunks: list[SimpleNamespace]
    chat: _FakeChat = field(init=False)

    def __post_init__(self) -> None:
        self.chat = _FakeChat(completions=_FakeCompletions(self.chunks))


def _model() -> Model:
    return Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=0),
        context_window=128_000,
        max_tokens=4096,
    )


def _ctx(text: str = "hello") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=1)])


def _text_chunk(delta_text: str | None = None, finish_reason: str | None = None) -> SimpleNamespace:
    return _ns(
        id="chatcmpl-x",
        choices=[
            _ns(
                delta=_ns(content=delta_text, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _usage_chunk(
    prompt: int,
    completion: int,
    cached: int = 0,
    reasoning: int = 0,
) -> SimpleNamespace:
    return _ns(
        id="chatcmpl-x",
        choices=[],
        usage=_ns(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=_ns(cached_tokens=cached) if cached else None,
            completion_tokens_details=_ns(reasoning_tokens=reasoning) if reasoning else None,
        ),
    )


class TestTextStream:
    async def test_basic_text_response(self) -> None:
        chunks = [
            _text_chunk(delta_text="Hello"),
            _text_chunk(delta_text=" world"),
            _text_chunk(finish_reason="stop"),
            _usage_chunk(prompt=10, completion=5),
        ]
        fake = _FakeAsyncOpenAI(chunks)
        events = [e async for e in stream_openai_completions(_model(), _ctx(), OpenAICompletionsOptions(), client=fake)]
        types = [e.type for e in events]
        assert types == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]

        delta_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert [e.delta for e in delta_events] == ["Hello", " world"]

        end_event = next(e for e in events if isinstance(e, TextEndEvent))
        assert end_event.content == "Hello world"

        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.reason == "stop"
        assert done.message.usage.input == 10
        assert done.message.usage.output == 5


class TestToolCallStream:
    async def test_tool_call_accumulation(self) -> None:
        chunks = [
            _ns(
                id="chatcmpl-tc",
                choices=[
                    _ns(
                        delta=_ns(
                            content=None,
                            tool_calls=[
                                _ns(
                                    id="call_1",
                                    function=_ns(name="bash", arguments='{"cmd"'),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            _ns(
                id="chatcmpl-tc",
                choices=[
                    _ns(
                        delta=_ns(
                            content=None,
                            tool_calls=[
                                _ns(
                                    id=None,
                                    function=_ns(name=None, arguments=': "ls"}'),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            _ns(
                id="chatcmpl-tc",
                choices=[_ns(delta=_ns(content=None, tool_calls=None), finish_reason="tool_calls")],
                usage=None,
            ),
        ]
        fake = _FakeAsyncOpenAI(chunks)
        events = [
            e
            async for e in stream_openai_completions(_model(), _ctx("run ls"), OpenAICompletionsOptions(), client=fake)
        ]
        tool_end = next(e for e in events if isinstance(e, ToolCallEndEvent))
        assert isinstance(tool_end.tool_call, ToolCall)
        assert tool_end.tool_call.id == "call_1"
        assert tool_end.tool_call.name == "bash"
        assert tool_end.tool_call.arguments == {"cmd": "ls"}

        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.reason == "toolUse"


class TestReasoningStream:
    async def test_reasoning_content_field_becomes_thinking(self) -> None:
        # llama.cpp and DeepSeek expose reasoning via ``reasoning_content``.
        chunks = [
            _ns(
                id="c",
                choices=[
                    _ns(
                        delta=_ns(
                            content=None,
                            tool_calls=None,
                            reasoning_content="thinking...",
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            _ns(
                id="c",
                choices=[
                    _ns(
                        delta=_ns(content="answer", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            _ns(
                id="c",
                choices=[_ns(delta=_ns(content=None, tool_calls=None), finish_reason="stop")],
                usage=None,
            ),
        ]
        fake = _FakeAsyncOpenAI(chunks)
        events = [e async for e in stream_openai_completions(_model(), _ctx(), OpenAICompletionsOptions(), client=fake)]
        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_end" in types
        assert "text_start" in types

        thinking_delta = next(e for e in events if isinstance(e, ThinkingDeltaEvent))
        assert thinking_delta.delta == "thinking..."

        done = events[-1]
        assert isinstance(done, DoneEvent)
        thinking_block = next(b for b in done.message.content if isinstance(b, ThinkingContent))
        assert thinking_block.thinking == "thinking..."


class TestErrorHandling:
    async def test_sdk_exception_becomes_error_event(self) -> None:
        class BoomCompletions:
            def __init__(self) -> None:
                self.last_kwargs: dict[str, Any] = {}

            async def create(self, **_: Any) -> None:
                raise RuntimeError("upstream failure")

        fake = SimpleNamespace(chat=SimpleNamespace(completions=BoomCompletions()))
        events = [e async for e in stream_openai_completions(_model(), _ctx(), OpenAICompletionsOptions(), client=fake)]
        err = events[-1]
        assert isinstance(err, ErrorEvent)
        assert err.reason == "error"
        assert "upstream failure" in (err.error.error_message or "")


class TestPayloadCapture:
    async def test_sent_payload_has_expected_shape(self) -> None:
        chunks = [
            _text_chunk(delta_text="ok"),
            _text_chunk(finish_reason="stop"),
            _usage_chunk(prompt=1, completion=1),
        ]
        fake = _FakeAsyncOpenAI(chunks)
        events = [
            e
            async for e in stream_openai_completions(
                _model(),
                _ctx("ping"),
                OpenAICompletionsOptions(max_tokens=100, temperature=0.5),
                client=fake,
            )
        ]
        assert events[-1].type == "done"
        kwargs = fake.chat.completions.last_kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        assert kwargs["store"] is False
        assert kwargs["max_completion_tokens"] == 100
        assert kwargs["temperature"] == 0.5
        assert kwargs["messages"][0]["role"] == "user"


class TestOllamaIntegration:
    async def test_ollama_base_url_with_no_api_key_does_not_crash(self) -> None:
        # Ollama uses the openai-completions API but has no API key. Verify the
        # whole pipeline runs through with a fake client (which bypasses the
        # real api_key requirement in create_client).
        ollama_model = Model(
            id="llama3.2",
            name="Llama 3.2",
            api="openai-completions",
            provider="ollama",
            base_url="http://localhost:11434/v1",
            reasoning=False,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=128_000,
            max_tokens=4096,
        )
        chunks = [
            _text_chunk(delta_text="hi from llama"),
            _text_chunk(finish_reason="stop"),
            _usage_chunk(prompt=2, completion=4),
        ]
        fake = _FakeAsyncOpenAI(chunks)
        events = [
            e
            async for e in stream_openai_completions(
                ollama_model,
                _ctx("hey"),
                OpenAICompletionsOptions(),
                client=fake,
            )
        ]
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert isinstance(done.message.content[0], TextContent)
        assert done.message.content[0].text == "hi from llama"
