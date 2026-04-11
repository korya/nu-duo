"""Tests for pi_ai.providers.faux.

Ported from the documented behaviour of
``packages/ai/src/providers/faux.ts``. The faux provider is the test harness
used throughout upstream pi-mono; its contract is stable and exercised by
every downstream test suite. These tests verify:

* Message factories produce wire-valid content.
* ``register_faux_provider`` installs the provider in the global registry
  and returns a handle with ``set_responses`` / ``unregister``.
* Queued responses are replayed through the streaming pipeline with
  content deltas, usage estimates, and session-based prompt caching.
* Empty queues yield an error event; callable steps can observe context.
* Unregister removes the provider.
"""

from __future__ import annotations

from pi_ai.api_registry import get_api_provider
from pi_ai.providers.faux import (
    FauxModelDefinition,
    _CallState,  # pyright: ignore[reportPrivateUsage]
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
    register_faux_provider,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    StreamOptions,
    TextContent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ToolCall,
    ToolCallEndEvent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


class TestFactories:
    def test_faux_text(self) -> None:
        tc = faux_text("hi")
        assert isinstance(tc, TextContent)
        assert tc.text == "hi"

    def test_faux_thinking(self) -> None:
        th = faux_thinking("ponder")
        assert isinstance(th, ThinkingContent)
        assert th.thinking == "ponder"

    def test_faux_tool_call_with_explicit_id(self) -> None:
        tc = faux_tool_call("bash", {"cmd": "ls"}, id_="my-id")
        assert isinstance(tc, ToolCall)
        assert tc.id == "my-id"
        assert tc.name == "bash"
        assert tc.arguments == {"cmd": "ls"}

    def test_faux_tool_call_default_id_is_non_empty(self) -> None:
        tc = faux_tool_call("bash", {})
        assert tc.id != ""

    def test_faux_assistant_message_from_string(self) -> None:
        msg = faux_assistant_message("hello")
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "hello"
        assert msg.stop_reason == "stop"
        assert msg.provider == "faux"

    def test_faux_assistant_message_from_block(self) -> None:
        msg = faux_assistant_message(faux_text("hi"))
        assert len(msg.content) == 1

    def test_faux_assistant_message_from_list(self) -> None:
        msg = faux_assistant_message([faux_thinking("let me think"), faux_text("answer")])
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], ThinkingContent)
        assert isinstance(msg.content[1], TextContent)

    def test_faux_assistant_message_with_options(self) -> None:
        msg = faux_assistant_message(
            "partial",
            stop_reason="error",
            error_message="boom",
            response_id="msg_x",
        )
        assert msg.stop_reason == "error"
        assert msg.error_message == "boom"
        assert msg.response_id == "msg_x"


# ---------------------------------------------------------------------------
# Registration lifecycle
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registration_installs_provider_in_registry(self) -> None:
        registration = register_faux_provider()
        try:
            provider = get_api_provider(registration.api)
            assert provider is not None
            assert provider.api == registration.api
        finally:
            registration.unregister()

    def test_unregister_removes_provider(self) -> None:
        registration = register_faux_provider()
        api = registration.api
        registration.unregister()
        assert get_api_provider(api) is None

    def test_default_model_fields(self) -> None:
        registration = register_faux_provider()
        try:
            model = registration.get_model()
            assert model.id == "faux-1"
            assert model.name == "Faux Model"
            assert model.provider == "faux"
        finally:
            registration.unregister()

    def test_custom_models(self) -> None:
        registration = register_faux_provider(
            models=[
                FauxModelDefinition(id="m1"),
                FauxModelDefinition(id="m2", reasoning=True, context_window=1000),
            ]
        )
        try:
            assert {m.id for m in registration.models} == {"m1", "m2"}
            m2 = registration.get_model("m2")
            assert m2 is not None
            assert m2.reasoning is True
            assert m2.context_window == 1000
            assert registration.get_model("missing") is None
        finally:
            registration.unregister()

    def test_get_pending_response_count_tracks_queue(self) -> None:
        registration = register_faux_provider()
        try:
            assert registration.get_pending_response_count() == 0
            registration.set_responses(
                [
                    faux_assistant_message("first"),
                    faux_assistant_message("second"),
                ]
            )
            assert registration.get_pending_response_count() == 2
            registration.append_responses([faux_assistant_message("third")])
            assert registration.get_pending_response_count() == 3
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------------


def _ctx(text: str = "hello") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=1)])


class TestStreaming:
    async def test_queued_text_response_streams(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("hello there")])
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            stream = provider.stream(model, _ctx("say hi"))
            events = [e async for e in stream]
            types = [e.type for e in events]

            assert types[0] == "start"
            assert types[-1] == "done"
            assert "text_start" in types
            assert "text_end" in types
            assert "text_delta" in types

            assert isinstance(events[0], StartEvent)
            text_start = next(e for e in events if isinstance(e, TextStartEvent))
            assert text_start.content_index == 0
            text_end = next(e for e in events if isinstance(e, TextEndEvent))
            assert text_end.content == "hello there"

            done = events[-1]
            assert isinstance(done, DoneEvent)
            assert done.reason == "stop"
            assert isinstance(done.message.content[0], TextContent)
            assert done.message.content[0].text == "hello there"
            # Usage estimate is non-zero.
            assert done.message.usage.output > 0
        finally:
            registration.unregister()

    async def test_tool_call_streams_with_final_arguments(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="call_1")],
                        stop_reason="toolUse",
                    ),
                ]
            )
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            stream = provider.stream(model, _ctx("run ls"))
            events = [e async for e in stream]

            tool_end = next(e for e in events if isinstance(e, ToolCallEndEvent))
            assert tool_end.tool_call.id == "call_1"
            assert tool_end.tool_call.name == "bash"
            assert tool_end.tool_call.arguments == {"cmd": "ls"}

            done = events[-1]
            assert isinstance(done, DoneEvent)
            assert done.reason == "toolUse"
        finally:
            registration.unregister()

    async def test_error_response_emits_error_event(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        "partial",
                        stop_reason="error",
                        error_message="boom",
                    ),
                ]
            )
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            stream = provider.stream(model, _ctx())
            events = [e async for e in stream]
            assert any(isinstance(e, ErrorEvent) for e in events)
            result = await stream.result()
            assert result.stop_reason == "error"
            assert result.error_message == "boom"
        finally:
            registration.unregister()

    async def test_empty_queue_raises_via_error_event(self) -> None:
        registration = register_faux_provider()
        try:
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            stream = provider.stream(model, _ctx())
            events = [e async for e in stream]
            assert events[-1].type == "error"
            result = await stream.result()
            assert result.stop_reason == "error"
            assert result.error_message is not None
            assert "faux" in result.error_message.lower()
        finally:
            registration.unregister()

    async def test_factory_step_observes_context_and_state(self) -> None:
        registration = register_faux_provider()
        try:
            captured: dict[str, object] = {}

            def factory(
                context: Context,
                options: StreamOptions | None,
                state: _CallState,
                model: Model,
                /,
            ) -> AssistantMessage:
                captured["messages"] = context.messages
                captured["call_count"] = state.call_count
                return faux_assistant_message("factory reply")

            registration.set_responses([factory])
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            stream = provider.stream(model, _ctx("ping"))
            _ = [e async for e in stream]

            assert registration.state.call_count == 1
            assert captured["call_count"] == 1
            messages = captured["messages"]
            assert isinstance(messages, list)
            assert len(messages) == 1
        finally:
            registration.unregister()

    async def test_multiple_responses_consumed_in_order(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message("first"),
                    faux_assistant_message("second"),
                ]
            )
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            out1 = await provider.stream(model, _ctx()).result()
            out2 = await provider.stream(model, _ctx()).result()
            assert isinstance(out1.content[0], TextContent)
            assert isinstance(out2.content[0], TextContent)
            assert out1.content[0].text == "first"
            assert out2.content[0].text == "second"
            assert registration.state.call_count == 2
        finally:
            registration.unregister()

    async def test_thinking_block_streams_with_delta(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_thinking("let me think about this"), faux_text("done")],
                    ),
                ]
            )
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            events = [e async for e in provider.stream(model, _ctx())]
            types = [e.type for e in events]
            assert "thinking_start" in types
            assert "thinking_delta" in types
            assert "thinking_end" in types

            done = events[-1]
            assert isinstance(done, DoneEvent)
            thinking = next(b for b in done.message.content if isinstance(b, ThinkingContent))
            assert thinking.thinking == "let me think about this"
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Session-based prompt cache
# ---------------------------------------------------------------------------


class TestPromptCache:
    async def test_session_id_enables_cache_read_on_repeat(self) -> None:
        # First call with a session id seeds the cache (all tokens written),
        # second call with the same session and overlapping prompt has cache_read > 0.
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message("one"),
                    faux_assistant_message("two"),
                ]
            )
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            # Same context both times → second call should have cache_read > 0.
            ctx = _ctx("long prompt that we expect to cache deterministically")
            first = await provider.stream(model, ctx, StreamOptions(session_id="s1")).result()
            second = await provider.stream(model, ctx, StreamOptions(session_id="s1")).result()
            assert first.usage.cache_write > 0
            assert second.usage.cache_read > 0
        finally:
            registration.unregister()

    async def test_cache_retention_none_disables_cache(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message("one"),
                    faux_assistant_message("two"),
                ]
            )
            provider = get_api_provider(registration.api)
            assert provider is not None
            model = registration.get_model()

            ctx = _ctx("prompt")
            opts = StreamOptions(session_id="s1", cache_retention="none")
            first = await provider.stream(model, ctx, opts).result()
            second = await provider.stream(model, ctx, opts).result()
            assert first.usage.cache_read == 0
            assert second.usage.cache_read == 0
        finally:
            registration.unregister()
