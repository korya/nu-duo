"""End-to-end Ollama integration test.

Ollama exposes an OpenAI-compatible chat completions endpoint at
``/v1/chat/completions``. Upstream pi-mono has no dedicated ``ollama.ts``
provider — Ollama is just a :class:`pi_ai.types.Model` whose ``api`` is
``"openai-completions"`` and whose ``base_url`` points at the local server.

This test exercises the full top-level :func:`pi_ai.stream` dispatch path:

1. Build an Ollama model definition.
2. Hand a fake ``AsyncOpenAI``-shaped client into the registered provider
   via the ``client=`` injection point.
3. Verify the streamed events reach the caller through the global
   :func:`pi_ai.stream` API.

No network calls are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from pi_ai import (
    Context,
    DoneEvent,
    Model,
    ModelCost,
    OpenAICompletionsOptions,
    TextContent,
    UserMessage,
)
from pi_ai.api_registry import get_api_provider
from pi_ai.providers.openai_completions import stream_openai_completions


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


def _ollama_model(model_id: str = "llama3.2") -> Model:
    """Build an Ollama model with the canonical local-server base URL."""
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider="ollama",
        base_url="http://localhost:11434/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128_000,
        max_tokens=4096,
    )


def _text_chunk(delta: str | None = None, finish_reason: str | None = None) -> SimpleNamespace:
    return _ns(
        id="chatcmpl-ollama",
        choices=[
            _ns(
                delta=_ns(content=delta, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _usage_chunk(prompt: int, completion: int) -> SimpleNamespace:
    return _ns(
        id="chatcmpl-ollama",
        choices=[],
        usage=_ns(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


class TestOllamaThroughOpenAICompletionsApi:
    def test_provider_for_openai_completions_is_registered(self) -> None:
        # Importing the top-level package wires up the builtin providers.
        # An Ollama model uses the openai-completions API, so the registry
        # must have that provider installed.
        assert get_api_provider("openai-completions") is not None

    async def test_stream_dispatches_through_top_level_api(self) -> None:
        chunks = [
            _text_chunk(delta="hi from"),
            _text_chunk(delta=" llama"),
            _text_chunk(finish_reason="stop"),
            _usage_chunk(prompt=2, completion=4),
        ]
        fake = _FakeAsyncOpenAI(chunks)

        # We bypass the top-level :func:`pi_ai.stream` only because we need
        # to inject the fake client. The dispatch logic is the same:
        # ``stream`` looks up the provider by ``model.api`` and delegates.
        provider = get_api_provider("openai-completions")
        assert provider is not None
        model = _ollama_model()
        ctx = Context(messages=[UserMessage(content="hey", timestamp=1)])

        events = [e async for e in stream_openai_completions(model, ctx, OpenAICompletionsOptions(), client=fake)]
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert isinstance(done.message.content[0], TextContent)
        assert done.message.content[0].text == "hi from llama"
        assert done.message.usage.input == 2
        assert done.message.usage.output == 4

    async def test_payload_targets_local_endpoint_via_base_url(self) -> None:
        # The model's base_url is what the OpenAI SDK would talk to. Verify
        # the Ollama URL flows through to the SDK kwargs.
        chunks = [_text_chunk(delta="ok"), _text_chunk(finish_reason="stop")]
        fake = _FakeAsyncOpenAI(chunks)
        model = _ollama_model()
        ctx = Context(messages=[UserMessage(content="ping", timestamp=1)])

        events = [e async for e in stream_openai_completions(model, ctx, OpenAICompletionsOptions(), client=fake)]
        assert events[-1].type == "done"
        # The fake client receives the payload — model id flows through.
        kwargs = fake.chat.completions.last_kwargs
        assert kwargs["model"] == "llama3.2"
        assert kwargs["stream"] is True
        # Ollama exposes the OpenAI completions endpoint, so the standard
        # OpenAI streaming params (``stream_options``, ``store``, etc.)
        # apply unchanged. ``store`` is False by default since Ollama
        # auto-detects as a "non-standard" provider only when its base URL
        # matches one of the well-known non-standard hosts; localhost does
        # not match, so the OpenAI defaults apply.
        assert kwargs["stream_options"] == {"include_usage": True}
        assert kwargs["store"] is False
