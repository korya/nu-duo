"""End-to-end tests for the top-level ``nu_ai.stream`` entry points.

Uses the faux provider as the driver: registers it, queues a scripted
response, and calls the module-level :func:`nu_ai.stream`, :func:`nu_ai.complete`,
:func:`nu_ai.stream_simple`, :func:`nu_ai.complete_simple` — the exact
integration points that downstream consumers (nu_agent_core, nu_coding_agent)
will rely on.
"""

from __future__ import annotations

import nu_ai
import pytest
from nu_ai import (
    AssistantMessage,
    Context,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    UserMessage,
)
from nu_ai.providers.faux import (
    faux_assistant_message,
    register_faux_provider,
)


def _ctx(text: str = "hello") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=1)])


class TestTopLevelStream:
    async def test_stream_dispatches_to_registered_provider(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("pong")])
            model = registration.get_model()
            events = [e async for e in nu_ai.stream(model, _ctx("ping"))]
            done = events[-1]
            assert isinstance(done, DoneEvent)
            assert isinstance(done.message.content[0], TextContent)
            assert done.message.content[0].text == "pong"
        finally:
            registration.unregister()

    async def test_complete_awaits_final_message(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("answer")])
            model = registration.get_model()
            result = await nu_ai.complete(model, _ctx("question"))
            assert isinstance(result, AssistantMessage)
            assert result.stop_reason == "stop"
            assert isinstance(result.content[0], TextContent)
            assert result.content[0].text == "answer"
        finally:
            registration.unregister()

    async def test_stream_simple_and_complete_simple(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message("simple-answer"),
                    faux_assistant_message("simple-complete"),
                ]
            )
            model = registration.get_model()
            events = [e async for e in nu_ai.stream_simple(model, _ctx())]
            assert events[-1].type == "done"
            result = await nu_ai.complete_simple(model, _ctx())
            assert isinstance(result.content[0], TextContent)
            assert result.content[0].text == "simple-complete"
        finally:
            registration.unregister()

    async def test_unknown_api_raises(self) -> None:
        # A model with an unregistered api is a programming error, not a
        # stream-level failure.
        model = Model(
            id="x",
            name="X",
            api="does-not-exist",
            provider="nowhere",
            base_url="",
            reasoning=False,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=0,
            max_tokens=0,
        )
        with pytest.raises(ValueError, match="No API provider registered"):
            nu_ai.stream(model, _ctx())


class TestBuiltinProvidersRegistered:
    def test_anthropic_builtin_registered(self) -> None:
        assert nu_ai.get_api_provider("anthropic-messages") is not None
