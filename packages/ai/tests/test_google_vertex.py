"""Tests for nu_ai.providers.google_vertex.

Uses a fake ``google.genai.Client``-shaped client whose
``client.aio.models.generate_content_stream(...)`` returns an async iterator
yielding scripted chunks — same shape as the real Vertex AI response.

The tests cover:
1. Basic text streaming
2. Thinking (reasoning) blocks
3. Tool calls
4. Usage metadata
5. Error handling
6. ``build_vertex_params`` (pure, no client needed)
7. ``_resolve_api_key`` / ``_resolve_project`` / ``_resolve_location`` helpers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from nu_ai.providers.google_vertex import (
    GoogleVertexOptions,
    _resolve_api_key,
    _resolve_location,
    _resolve_project,
    build_vertex_params,
    stream_google_vertex,
)
from nu_ai.types import (
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolCallEndEvent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Fake client infrastructure
# ---------------------------------------------------------------------------


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
class _FakeVertexClient:
    chunks: list[SimpleNamespace]
    aio: _FakeAio = field(init=False)

    def __post_init__(self) -> None:
        self.aio = _FakeAio(models=_FakeAsyncModels(self.chunks))


# ---------------------------------------------------------------------------
# Test model / context helpers
# ---------------------------------------------------------------------------


def _model(model_id: str = "gemini-2.5-flash", *, reasoning: bool = True) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="google-vertex",
        provider="google-vertex",
        base_url="",
        reasoning=reasoning,
        input=["text", "image"],
        cost=ModelCost(input=0.3, output=2.5, cache_read=0.075, cache_write=0),
        context_window=1_048_576,
        max_tokens=8192,
    )


def _ctx(text: str = "hello") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=1)])


# ---------------------------------------------------------------------------
# Chunk factory helpers
# ---------------------------------------------------------------------------


def _text_part(text: str, *, thought: bool = False) -> SimpleNamespace:
    return _ns(text=text, thought=thought, function_call=None, thought_signature=None)


def _function_call_part(name: str, args: dict[str, Any], call_id: str | None = None) -> SimpleNamespace:
    return _ns(
        text=None,
        thought=None,
        thought_signature=None,
        function_call=_ns(name=name, args=args, id=call_id),
    )


def _chunk(
    parts: list[SimpleNamespace],
    *,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return _ns(
        response_id="vtx_resp_1",
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
        response_id=None,
        candidates=[],
        usage_metadata=_ns(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            cached_content_token_count=cached,
            thoughts_token_count=thoughts,
            total_token_count=prompt + candidates + thoughts,
        ),
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestTextStreamVertex:
    async def test_basic_text(self) -> None:
        chunks = [
            _chunk([_text_part("Hello")]),
            _chunk([_text_part(" Vertex")]),
            _chunk([], finish_reason="STOP"),
            _usage_chunk(prompt=10, candidates=5),
        ]
        fake = _FakeVertexClient(chunks)
        events = [e async for e in stream_google_vertex(_model(), _ctx(), GoogleVertexOptions(), client=fake)]
        types = [e.type for e in events]
        assert types == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]

        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert [e.delta for e in deltas] == ["Hello", " Vertex"]

        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.message.usage.input == 10
        assert done.message.usage.output == 5
        assert done.message.response_id == "vtx_resp_1"

    async def test_stop_reason_length(self) -> None:
        chunks = [
            _chunk([_text_part("truncated")], finish_reason="MAX_TOKENS"),
        ]
        fake = _FakeVertexClient(chunks)
        events = [e async for e in stream_google_vertex(_model(), _ctx(), None, client=fake)]
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.reason == "length"


class TestThinkingStreamVertex:
    async def test_thought_part_becomes_thinking_block(self) -> None:
        chunks = [
            _chunk([_text_part("reasoning...", thought=True)]),
            _chunk([_text_part("answer")]),
            _chunk([], finish_reason="STOP"),
            _usage_chunk(prompt=5, candidates=3, thoughts=4),
        ]
        fake = _FakeVertexClient(chunks)
        events = [e async for e in stream_google_vertex(_model(), _ctx(), None, client=fake)]
        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types
        assert "text_start" in types

        thinking_delta = next(e for e in events if isinstance(e, ThinkingDeltaEvent))
        assert thinking_delta.delta == "reasoning..."

        done = events[-1]
        assert isinstance(done, DoneEvent)
        thinking = next(b for b in done.message.content if isinstance(b, ThinkingContent))
        assert thinking.thinking == "reasoning..."
        # Thoughts go to output token count
        assert done.message.usage.output == 7  # 3 candidates + 4 thoughts


class TestToolCallStreamVertex:
    async def test_function_call_emits_tool_call_events(self) -> None:
        chunks = [
            _chunk([_function_call_part("search", {"q": "python"}, call_id="tc_1")]),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeVertexClient(chunks)
        events = [e async for e in stream_google_vertex(_model(), _ctx(), None, client=fake)]
        tool_end = next(e for e in events if isinstance(e, ToolCallEndEvent))
        assert isinstance(tool_end.tool_call, ToolCall)
        assert tool_end.tool_call.name == "search"
        assert tool_end.tool_call.arguments == {"q": "python"}

        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.reason == "toolUse"

    async def test_function_call_without_id_gets_generated_id(self) -> None:
        chunks = [
            _chunk([_function_call_part("bash", {"cmd": "ls"}, call_id=None)]),
            _chunk([], finish_reason="STOP"),
        ]
        fake = _FakeVertexClient(chunks)
        events = [e async for e in stream_google_vertex(_model(), _ctx(), None, client=fake)]
        tool_end = next(e for e in events if isinstance(e, ToolCallEndEvent))
        assert tool_end.tool_call.id
        assert tool_end.tool_call.id.startswith("bash_")


class TestErrorHandlingVertex:
    async def test_sdk_exception_becomes_error_event(self) -> None:
        class _BrokenModels:
            async def generate_content_stream(self, **_: Any) -> None:
                raise RuntimeError("Vertex exploded")

        class _BrokenAio:
            models = _BrokenModels()

        class _BrokenClient:
            aio = _BrokenAio()

        events = [e async for e in stream_google_vertex(_model(), _ctx(), None, client=_BrokenClient())]
        error_ev = next(e for e in events if isinstance(e, ErrorEvent))
        assert error_ev.reason == "error"
        assert "Vertex exploded" in error_ev.error.error_message


class TestBuildVertexParams:
    def test_includes_system_prompt(self) -> None:
        ctx = Context(
            messages=[UserMessage(content="hi", timestamp=1)],
            system_prompt="You are helpful.",
        )
        params = build_vertex_params(_model(), ctx)
        assert params["config"]["system_instruction"] == "You are helpful."

    def test_temperature_and_max_tokens(self) -> None:
        opts = GoogleVertexOptions(temperature=0.5, max_tokens=512)
        params = build_vertex_params(_model(), _ctx(), opts)
        assert params["config"]["temperature"] == 0.5
        assert params["config"]["max_output_tokens"] == 512

    def test_thinking_enabled_sets_include_thoughts(self) -> None:
        opts = GoogleVertexOptions(thinking_enabled=True, thinking_budget_tokens=2048)
        params = build_vertex_params(_model(reasoning=True), _ctx(), opts)
        tc = params["config"]["thinking_config"]
        assert tc["include_thoughts"] is True
        assert tc["thinking_budget"] == 2048

    def test_thinking_disabled_for_gemini2(self) -> None:
        opts = GoogleVertexOptions(thinking_enabled=False)
        params = build_vertex_params(_model("gemini-2.5-flash", reasoning=True), _ctx(), opts)
        tc = params["config"]["thinking_config"]
        assert tc == {"thinking_budget": 0}

    def test_no_thinking_config_when_reasoning_false(self) -> None:
        opts = GoogleVertexOptions(thinking_enabled=True, thinking_budget_tokens=1000)
        params = build_vertex_params(_model("gemini-2.5-flash", reasoning=False), _ctx(), opts)
        assert "thinking_config" not in params["config"]


class TestResolveHelpers:
    def test_resolve_api_key_placeholder_returns_none(self) -> None:
        opts = GoogleVertexOptions(api_key="<YOUR_API_KEY>")
        assert _resolve_api_key(opts) is None

    def test_resolve_api_key_real_value(self) -> None:
        opts = GoogleVertexOptions(api_key="real-key-123")
        assert _resolve_api_key(opts) == "real-key-123"

    def test_resolve_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "env-key")
        assert _resolve_api_key(None) == "env-key"

    def test_resolve_project_raises_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError, match="project ID"):
            _resolve_project(None)

    def test_resolve_project_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        assert _resolve_project(None) == "my-project"

    def test_resolve_location_raises_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        with pytest.raises(ValueError, match="location"):
            _resolve_location(None)

    def test_resolve_location_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        assert _resolve_location(None) == "us-central1"
