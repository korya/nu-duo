"""Tests for nu_ai.providers.google_gemini_cli.

Uses a fake ``httpx.AsyncClient`` that returns scripted SSE payloads instead
of making real HTTP requests.

The tests cover:
1. Basic text streaming via ``stream_google_gemini_cli``
2. Tool call streaming
3. Missing credentials → ``ErrorEvent``
4. ``build_request`` pure function (no HTTP)
5. ``extract_retry_delay`` header/body parsing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from nu_ai.providers.google_gemini_cli import (
    GoogleGeminiCliOptions,
    build_request,
    extract_retry_delay,
    stream_google_gemini_cli,
)
from nu_ai.types import (
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    TextDeltaEvent,
    Tool,
    ToolCall,
    ToolCallEndEvent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(
    model_id: str = "gemini-2.5-flash",
    *,
    provider: str = "google-gemini-cli",
    reasoning: bool = False,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="google-gemini-cli",
        provider=provider,
        base_url="",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(input=0.075, output=0.30, cache_read=0.0, cache_write=0.0),
        context_window=1_048_576,
        max_tokens=8192,
    )


def _ctx(text: str = "hello") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=1)])


def _creds(token: str = "tok-abc", project: str = "my-proj") -> str:
    return json.dumps({"token": token, "projectId": project})


def _sse_lines(response: dict[str, Any]) -> bytes:
    """Encode a single SSE event."""
    return f"data: {json.dumps(response)}\n\n".encode()


def _candidate_chunk(
    parts: list[dict[str, Any]],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "content": {"role": "model", "parts": parts},
    }
    if finish_reason:
        candidate["finishReason"] = finish_reason
    return {"response": {"candidates": [candidate], "responseId": "cli_resp_1"}}


def _usage_chunk(prompt: int, candidates: int) -> dict[str, Any]:
    return {
        "response": {
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": prompt,
                "candidatesTokenCount": candidates,
                "totalTokenCount": prompt + candidates,
            },
        }
    }


# ---------------------------------------------------------------------------
# Fake httpx.AsyncClient
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    """Minimal fake for ``httpx.Response`` supporting ``aiter_lines``."""

    chunks: list[dict[str, Any]]
    status_code: int = 200
    headers: httpx.Headers = field(default_factory=lambda: httpx.Headers({}))

    async def aiter_lines(self):  # type: ignore[override]
        for chunk in self.chunks:
            encoded = json.dumps(chunk)
            yield f"data: {encoded}"

    @property
    def text(self) -> str:
        return ""


@dataclass
class _FakeAsyncClient:
    """Fake ``httpx.AsyncClient`` that returns scripted responses for POST."""

    responses: list[_FakeResponse]
    _call_index: int = field(default=0, init=False)
    last_post_kwargs: dict[str, Any] = field(default_factory=dict, init=False)

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.last_post_kwargs = {"url": url, **kwargs}
        resp = self.responses[self._call_index]
        self._call_index = min(self._call_index + 1, len(self.responses) - 1)
        return resp

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicTextStream:
    async def test_text_streaming(self) -> None:
        chunks = [
            _candidate_chunk([{"text": "Hello"}]),
            _candidate_chunk([{"text": " world"}]),
            _candidate_chunk([], finish_reason="STOP"),
            _usage_chunk(prompt=10, candidates=5),
        ]
        fake = _FakeAsyncClient([_FakeResponse(chunks)])
        opts = GoogleGeminiCliOptions(api_key=_creds())

        events = [e async for e in stream_google_gemini_cli(_model(), _ctx(), opts, client=fake)]
        types = [e.type for e in events]
        assert "start" in types
        assert "text_start" in types
        assert "done" in types

        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert [e.delta for e in deltas] == ["Hello", " world"]

        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.reason == "stop"
        assert done.message.usage.input == 10
        assert done.message.usage.output == 5

    async def test_response_id_propagated(self) -> None:
        chunks = [
            _candidate_chunk([{"text": "Hi"}]),
            _candidate_chunk([], finish_reason="STOP"),
        ]
        fake = _FakeAsyncClient([_FakeResponse(chunks)])
        opts = GoogleGeminiCliOptions(api_key=_creds())

        events = [e async for e in stream_google_gemini_cli(_model(), _ctx(), opts, client=fake)]
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.message.response_id == "cli_resp_1"


class TestToolCallStream:
    async def test_function_call_emits_tool_events(self) -> None:
        chunks = [
            _candidate_chunk([{"functionCall": {"name": "get_weather", "args": {"city": "London"}, "id": "fc_1"}}]),
            _candidate_chunk([], finish_reason="STOP"),
        ]
        fake = _FakeAsyncClient([_FakeResponse(chunks)])
        opts = GoogleGeminiCliOptions(api_key=_creds())

        events = [e async for e in stream_google_gemini_cli(_model(), _ctx(), opts, client=fake)]
        tool_end = next((e for e in events if isinstance(e, ToolCallEndEvent)), None)
        assert tool_end is not None
        assert isinstance(tool_end.tool_call, ToolCall)
        assert tool_end.tool_call.name == "get_weather"
        assert tool_end.tool_call.arguments == {"city": "London"}

        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.reason == "toolUse"


class TestErrorHandling:
    async def test_missing_credentials_produces_error_event(self) -> None:
        # No api_key → should get an ErrorEvent immediately.
        events = [e async for e in stream_google_gemini_cli(_model(), _ctx(), None, client=None)]
        # We won't have an httpx client so the async task will error early.
        # Catch the ErrorEvent.
        error_ev = next((e for e in events if isinstance(e, ErrorEvent)), None)
        assert error_ev is not None
        assert error_ev.reason == "error"
        assert "OAuth" in error_ev.error.error_message or "authenticate" in error_ev.error.error_message

    async def test_invalid_credentials_json_produces_error_event(self) -> None:
        opts = GoogleGeminiCliOptions(api_key="not-json")
        events = [e async for e in stream_google_gemini_cli(_model(), _ctx(), opts, client=None)]
        error_ev = next((e for e in events if isinstance(e, ErrorEvent)), None)
        assert error_ev is not None
        assert (
            "credentials" in error_ev.error.error_message.lower()
            or "authenticate" in error_ev.error.error_message.lower()
        )

    async def test_http_500_becomes_error_event(self) -> None:
        class _Error500Response:
            status_code = 500
            headers = httpx.Headers({})

            @property
            def text(self) -> str:
                return '{"error": {"message": "Internal server error"}}'

            async def aiter_lines(self):
                return
                yield  # make it a generator

        class _Error500Client:
            async def post(self, url: str, **_: Any) -> _Error500Response:
                return _Error500Response()

            async def aclose(self) -> None:
                pass

        opts = GoogleGeminiCliOptions(api_key=_creds())
        events = [
            e
            async for e in stream_google_gemini_cli(_model(), _ctx(), opts, client=_Error500Client())  # type: ignore[arg-type]
        ]
        error_ev = next((e for e in events if isinstance(e, ErrorEvent)), None)
        assert error_ev is not None
        assert error_ev.reason == "error"


class TestBuildRequest:
    def test_project_and_model_in_body(self) -> None:
        body = build_request(_model(), _ctx(), "test-project")
        assert body["project"] == "test-project"
        assert body["model"] == "gemini-2.5-flash"

    def test_system_instruction_wraps_parts(self) -> None:
        ctx = Context(
            messages=[UserMessage(content="hi", timestamp=1)],
            system_prompt="Be concise.",
        )
        body = build_request(_model(), ctx, "p")
        assert body["request"]["systemInstruction"]["parts"][0]["text"] == "Be concise."

    def test_antigravity_prepends_system_instruction(self) -> None:
        ctx = Context(
            messages=[UserMessage(content="hi", timestamp=1)],
            system_prompt="My prompt.",
        )
        body = build_request(_model(), ctx, "p", is_antigravity=True)
        parts = body["request"]["systemInstruction"]["parts"]
        # First two parts are the Antigravity boilerplate
        assert "Antigravity" in parts[0]["text"]
        # Original system prompt is preserved at the end
        assert parts[-1]["text"] == "My prompt."

    def test_request_type_set_for_antigravity(self) -> None:
        body = build_request(_model(), _ctx(), "p", is_antigravity=True)
        assert body.get("requestType") == "agent"
        assert body["userAgent"] == "antigravity"

    def test_tool_conversion_uses_parameters_for_claude(self) -> None:
        model = _model("claude-3-5-sonnet")
        tool = Tool(
            name="search",
            description="Search",
            parameters={"type": "object", "properties": {}},
        )
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)], tools=[tool])
        body = build_request(model, ctx, "p")
        fn_decls = body["request"]["tools"][0]["function_declarations"]
        # Claude → legacy ``parameters`` key
        assert "parameters" in fn_decls[0]
        assert "parameters_json_schema" not in fn_decls[0]

    def test_tool_conversion_uses_json_schema_for_gemini(self) -> None:
        tool = Tool(
            name="search",
            description="Search",
            parameters={"type": "object", "properties": {}},
        )
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)], tools=[tool])
        body = build_request(_model(), ctx, "p")
        fn_decls = body["request"]["tools"][0]["function_declarations"]
        assert "parameters_json_schema" in fn_decls[0]


class TestExtractRetryDelay:
    def test_quota_reset_seconds(self) -> None:
        delay = extract_retry_delay("Your quota will reset after 39s")
        assert delay is not None
        assert delay >= 40_000  # 39s + 1s padding

    def test_quota_reset_hours_minutes_seconds(self) -> None:
        delay = extract_retry_delay("Your quota will reset after 1h2m30s")
        expected_ms = (1 * 3600 + 2 * 60 + 30) * 1000
        assert delay is not None
        assert delay >= expected_ms

    def test_please_retry_seconds(self) -> None:
        delay = extract_retry_delay("Please retry in 5s")
        assert delay is not None
        assert delay >= 6_000

    def test_please_retry_ms(self) -> None:
        delay = extract_retry_delay("Please retry in 500ms")
        assert delay is not None
        assert delay >= 1_500  # 500ms + 1000ms padding

    def test_retry_delay_json_field(self) -> None:
        delay = extract_retry_delay('{"retryDelay": "10s"}')
        assert delay is not None
        assert delay >= 11_000

    def test_retry_after_header_seconds(self) -> None:
        headers = httpx.Headers({"retry-after": "30"})
        delay = extract_retry_delay("", headers)
        assert delay is not None
        assert delay >= 31_000

    def test_no_match_returns_none(self) -> None:
        delay = extract_retry_delay("Unknown error occurred")
        assert delay is None
