"""Tests for ``nu_coding_agent.core.agent_session``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from nu_agent_core.agent import Agent, AgentOptions
from nu_ai.api_registry import get_api_provider
from nu_ai.providers.faux import (
    faux_assistant_message,
    register_faux_provider,
)
from nu_ai.types import AssistantMessage, Message, ToolResultMessage, UserMessage
from nu_coding_agent.core.agent_session import AgentSession, AgentSessionConfig
from nu_coding_agent.core.auth_storage import ApiKeyCredential, AuthStorage
from nu_coding_agent.core.model_registry import ModelRegistry
from nu_coding_agent.core.session_manager import SessionManager

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nu_ai.utils.event_stream import AssistantMessageEventStream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_fn(api: str):
    provider = get_api_provider(api)
    assert provider is not None

    def stream_fn(model: Any, context: Any, options: Any | None = None) -> AssistantMessageEventStream:
        return provider.stream_simple(model, context, options)

    return stream_fn


async def _convert_to_llm(messages: list[Any]) -> list[Message]:
    return [m for m in messages if isinstance(m, UserMessage | AssistantMessage | ToolResultMessage)]


@pytest.fixture
def faux_setup() -> Iterator[tuple[Agent, SessionManager, ModelRegistry, AuthStorage]]:
    """Spin up a faux-provider Agent + the four AgentSession collaborators."""
    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
        # Seed AuthStorage by *provider* (not by `registration.api` — that's
        # a unique tag like "faux:1234:abcd"; has_auth() looks up by provider).
        storage = AuthStorage.in_memory({faux_model.provider: ApiKeyCredential(type="api_key", key="test-key")})
        registry = ModelRegistry.in_memory(storage)
        # Inject the faux model into the registry so has_configured_auth
        # walks the right code path.
        registry._models.append(faux_model)  # type: ignore[attr-defined]

        agent = Agent(
            AgentOptions(
                initial_state={
                    "model": faux_model,
                    "system_prompt": "",
                    "tools": [],
                },
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        session_manager = SessionManager.in_memory("/work")
        yield agent, session_manager, registry, storage
    finally:
        registration.unregister()


def _make_session(setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage]) -> AgentSession:
    agent, sm, registry, storage = setup
    return AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=storage,
            cwd="/work",
        )
    )


# ---------------------------------------------------------------------------
# Construction + accessors
# ---------------------------------------------------------------------------


def test_create_classmethod_builds_session(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    agent, sm, registry, storage = faux_setup
    session = AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=storage,
        cwd="/work",
    )
    assert session.agent is agent
    assert session.session_manager is sm
    assert session.model_registry is registry
    assert session.auth_storage is storage
    assert session.cwd == "/work"
    assert session.model is not None
    session.close()


def test_close_is_idempotent(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    session = _make_session(faux_setup)
    session.close()
    session.close()  # second call must be a no-op


# ---------------------------------------------------------------------------
# prompt() — happy path persists user + assistant messages
# ---------------------------------------------------------------------------


async def test_prompt_round_trip_with_session_persistence() -> None:
    """End-to-end: prompt → assistant message lands in the session."""
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("hello world")])
        storage = AuthStorage.in_memory({registration.get_model().provider: ApiKeyCredential(type="api_key", key="x")})
        registry = ModelRegistry.in_memory(storage)
        faux_model = registration.get_model()
        registry._models.append(faux_model)  # type: ignore[attr-defined]

        agent = Agent(
            AgentOptions(
                initial_state={"model": faux_model, "system_prompt": "", "tools": []},
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        sm = SessionManager.in_memory("/work")
        session = AgentSession.create(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=storage,
            cwd="/work",
        )
        try:
            await session.prompt("hi")
        finally:
            session.close()

        entries = sm.get_entries()
        # Entry types in the session: message (user) → message (assistant).
        types = [e.get("type") for e in entries]
        assert "message" in types
        message_entries = [e for e in entries if e.get("type") == "message"]
        roles = [e["message"].get("role") for e in message_entries]
        assert "user" in roles
        assert "assistant" in roles
    finally:
        registration.unregister()


async def test_prompt_rejects_when_no_model() -> None:
    """An Agent constructed without a model uses the placeholder; prompt() must reject."""
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    agent = Agent(AgentOptions(initial_state={"system_prompt": "", "tools": []}))
    sm = SessionManager.in_memory("/work")
    session = AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=storage,
        cwd="/work",
    )
    try:
        with pytest.raises(ValueError, match="No model selected"):
            await session.prompt("hi")
    finally:
        session.close()


async def test_prompt_rejects_when_no_credentials(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    agent, sm, _registry, _storage = faux_setup
    # Build a fresh storage with no credential at all.
    empty_storage = AuthStorage.in_memory()
    session = AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=sm,
            model_registry=ModelRegistry.in_memory(empty_storage),
            auth_storage=empty_storage,
            cwd="/work",
        )
    )
    try:
        with pytest.raises(ValueError, match="No API key configured"):
            await session.prompt("hi")
    finally:
        session.close()


async def test_prompt_after_close_raises(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    session = _make_session(faux_setup)
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        await session.prompt("hi")


# ---------------------------------------------------------------------------
# subscribe() — listeners receive events after persistence
# ---------------------------------------------------------------------------


async def test_subscribe_receives_events_after_persistence() -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("ack")])
        storage = AuthStorage.in_memory({registration.get_model().provider: ApiKeyCredential(type="api_key", key="x")})
        registry = ModelRegistry.in_memory(storage)
        faux_model = registration.get_model()
        registry._models.append(faux_model)  # type: ignore[attr-defined]

        agent = Agent(
            AgentOptions(
                initial_state={"model": faux_model, "system_prompt": "", "tools": []},
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        sm = SessionManager.in_memory("/work")
        session = AgentSession.create(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=storage,
            cwd="/work",
        )
        try:
            received: list[str] = []

            def listener(event: Any) -> None:
                received.append(event["type"])

            session.subscribe(listener)
            await session.prompt("hi")
            assert "message_end" in received
            assert "agent_end" in received
        finally:
            session.close()
    finally:
        registration.unregister()


async def test_subscribe_unsubscribe_stops_delivery() -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("ack")])
        storage = AuthStorage.in_memory({registration.get_model().provider: ApiKeyCredential(type="api_key", key="x")})
        registry = ModelRegistry.in_memory(storage)
        faux_model = registration.get_model()
        registry._models.append(faux_model)  # type: ignore[attr-defined]

        agent = Agent(
            AgentOptions(
                initial_state={"model": faux_model, "system_prompt": "", "tools": []},
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        sm = SessionManager.in_memory("/work")
        session = AgentSession.create(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=storage,
            cwd="/work",
        )
        try:
            received: list[str] = []
            unsubscribe = session.subscribe(lambda event: received.append(event["type"]))
            unsubscribe()
            await session.prompt("hi")
            assert received == []
        finally:
            session.close()
    finally:
        registration.unregister()


# ---------------------------------------------------------------------------
# set_model() persists a model_change entry
# ---------------------------------------------------------------------------


def test_set_model_appends_session_entry(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    session = _make_session(faux_setup)
    try:
        agent, sm, _registry, _storage = faux_setup
        original_id = session.model.id  # type: ignore[union-attr]
        # Build a synthetic Model object for the swap.
        from nu_ai.types import Model as NuModel  # noqa: PLC0415
        from nu_ai.types import ModelCost  # noqa: PLC0415

        new_model = NuModel(
            id="some-other-id",
            name="some-other-id",
            api=session.model.api,  # type: ignore[union-attr]
            provider=session.model.provider,  # type: ignore[union-attr]
            base_url=session.model.base_url,  # type: ignore[union-attr]
            reasoning=False,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=10000,
            max_tokens=1000,
        )
        session.set_model(new_model)
        assert session.model is new_model
        assert agent.state.model is new_model
        change_entries = [e for e in sm.get_entries() if e.get("type") == "model_change"]
        assert len(change_entries) == 1
        assert change_entries[0]["modelId"] == "some-other-id"
        assert change_entries[0]["provider"] == new_model.provider
        # Original id was different.
        assert original_id != "some-other-id"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------


async def test_get_stats_after_one_turn() -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("hello")])
        storage = AuthStorage.in_memory({registration.get_model().provider: ApiKeyCredential(type="api_key", key="x")})
        registry = ModelRegistry.in_memory(storage)
        faux_model = registration.get_model()
        registry._models.append(faux_model)  # type: ignore[attr-defined]

        agent = Agent(
            AgentOptions(
                initial_state={"model": faux_model, "system_prompt": "", "tools": []},
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        sm = SessionManager.in_memory("/work")
        session = AgentSession.create(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=storage,
            cwd="/work",
        )
        try:
            await session.prompt("hi")
            stats = session.get_stats()
            assert stats.user_messages == 1
            assert stats.assistant_messages == 1
            assert stats.total_messages >= 2
            assert stats.session_id
        finally:
            session.close()
    finally:
        registration.unregister()


def test_get_stats_empty_session(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    session = _make_session(faux_setup)
    try:
        stats = session.get_stats()
        assert stats.user_messages == 0
        assert stats.assistant_messages == 0
        assert stats.tool_calls == 0
        assert stats.tool_results == 0
        assert stats.tokens_total == 0
        assert stats.cost == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Compaction surface
# ---------------------------------------------------------------------------


def test_should_compact_returns_false_without_assistant(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    session = _make_session(faux_setup)
    try:
        # No usage info → can't decide → returns False.
        assert session.should_compact() is False
    finally:
        session.close()


def test_estimate_context_tokens_for_empty_session(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    session = _make_session(faux_setup)
    try:
        assert session.estimate_context_tokens() == 0
    finally:
        session.close()


def test_prepare_compaction_returns_none_for_empty_session(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    session = _make_session(faux_setup)
    try:
        assert session.prepare_compaction() is None
    finally:
        session.close()


async def test_compact_returns_none_when_nothing_to_compact(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    session = _make_session(faux_setup)
    try:
        assert await session.compact() is None
    finally:
        session.close()


async def test_listener_exception_swallowed(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A broken listener must not stop event delivery to other listeners."""
    session = _make_session(faux_setup)
    try:
        crashed: list[bool] = []
        ok_count: list[int] = [0]

        def boom(_event: Any) -> None:
            crashed.append(True)
            raise RuntimeError("kaboom")

        def ok(_event: Any) -> None:
            ok_count[0] += 1

        session.subscribe(boom)
        session.subscribe(ok)
        # Manually drive the event handler with a fake message_end so we
        # don't need a faux response set up here.
        fake_event = {
            "type": "message_end",
            "message": UserMessage(content="hi", timestamp=1),
        }
        await session._handle_agent_event(fake_event, asyncio.Event())  # type: ignore[arg-type]
        assert crashed
        assert ok_count[0] == 1
        captured = capsys.readouterr()
        assert "AgentSession listener error" in captured.err
    finally:
        session.close()


def test_session_property_returns_none_for_placeholder() -> None:
    """``session.model`` returns None when the agent has the placeholder model."""
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    agent = Agent(AgentOptions(initial_state={"system_prompt": "", "tools": []}))
    sm = SessionManager.in_memory("/work")
    session = AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=storage,
        cwd="/work",
    )
    try:
        assert session.model is None
    finally:
        session.close()


async def test_prompt_with_images_raises_not_implemented(
    faux_setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
) -> None:
    from nu_ai.types import ImageContent  # noqa: PLC0415

    session = _make_session(faux_setup)
    try:
        with pytest.raises(NotImplementedError):
            await session.prompt(
                "hi",
                images=[ImageContent(data="aGVsbG8=", mime_type="image/png")],
            )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# compact() end-to-end against the faux provider
# ---------------------------------------------------------------------------


async def test_compact_e2e_with_faux_provider() -> None:
    """Drive an actual compaction round with a stubbed LLM response."""
    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
        # Pre-populate the session with enough user messages that find_cut_point
        # has something to summarise.
        sm = SessionManager.in_memory("/work")
        for i in range(20):
            sm.append_message({"role": "user", "content": f"prompt {i}", "timestamp": 1})
            sm.append_message(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "x" * 5000}],
                    "provider": faux_model.provider,
                    "model": faux_model.id,
                    "api": faux_model.api,
                    "stopReason": "stop",
                    "usage": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 0,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                    },
                    "timestamp": 1,
                }
            )

        # Compaction can do a history summary AND a turn-prefix summary,
        # so queue two responses; the second is consumed only if needed.
        registration.set_responses(
            [
                faux_assistant_message("STRUCTURED CHECKPOINT"),
                faux_assistant_message("TURN PREFIX"),
            ]
        )
        storage = AuthStorage.in_memory({faux_model.provider: ApiKeyCredential(type="api_key", key="x")})
        registry = ModelRegistry.in_memory(storage)
        registry._models.append(faux_model)  # type: ignore[attr-defined]

        agent = Agent(
            AgentOptions(
                initial_state={"model": faux_model, "system_prompt": "", "tools": []},
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        from nu_coding_agent.core.compaction import CompactionSettings  # noqa: PLC0415

        session = AgentSession.create(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=storage,
            cwd="/work",
            compaction_settings=CompactionSettings(enabled=True, reserve_tokens=1000, keep_recent_tokens=200),
        )
        try:
            result = await session.compact()
            assert result is not None
            assert "STRUCTURED CHECKPOINT" in result.summary
            # Compaction entry was appended.
            comp_entries = [e for e in sm.get_entries() if e.get("type") == "compaction"]
            assert len(comp_entries) == 1
        finally:
            session.close()
    finally:
        registration.unregister()


async def test_compact_raises_when_no_credentials() -> None:
    """compact() must reject when the model's provider has no key."""
    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
        sm = SessionManager.in_memory("/work")
        sm.append_message({"role": "user", "content": "hi", "timestamp": 1})
        sm.append_message(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "x" * 5000}],
                "provider": faux_model.provider,
                "model": faux_model.id,
                "api": faux_model.api,
                "stopReason": "stop",
                "usage": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 0,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                },
                "timestamp": 1,
            }
        )
        sm.append_message({"role": "user", "content": "more", "timestamp": 1})

        empty_storage = AuthStorage.in_memory()
        registry = ModelRegistry.in_memory(empty_storage)
        registry._models.append(faux_model)  # type: ignore[attr-defined]

        agent = Agent(
            AgentOptions(
                initial_state={"model": faux_model, "system_prompt": "", "tools": []},
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        from nu_coding_agent.core.compaction import CompactionSettings  # noqa: PLC0415

        session = AgentSession.create(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=empty_storage,
            cwd="/work",
            compaction_settings=CompactionSettings(enabled=True, reserve_tokens=1000, keep_recent_tokens=10),
        )
        try:
            with pytest.raises(ValueError, match="No API key"):
                await session.compact()
        finally:
            session.close()
    finally:
        registration.unregister()


def test_get_stats_with_assistant_usage_and_cost() -> None:
    """Build a session manually with usage data to exercise the cost path."""
    sm = SessionManager.in_memory("/work")
    sm.append_message({"role": "user", "content": "hi", "timestamp": 1})
    sm.append_message(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ack"},
                {"type": "toolCall", "name": "read", "arguments": {"path": "/x"}},
            ],
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api": "openai-completions",
            "stopReason": "stop",
            "usage": {
                "input": 10,
                "output": 20,
                "cacheRead": 5,
                "cacheWrite": 3,
                "totalTokens": 38,
                "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0, "total": 3},
            },
            "timestamp": 1,
        }
    )
    sm.append_message(
        {
            "role": "toolResult",
            "toolCallId": "tc-1",
            "toolName": "read",
            "content": [{"type": "text", "text": "ok"}],
            "details": None,
            "isError": False,
            "timestamp": 1,
        }
    )

    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    agent = Agent(AgentOptions(initial_state={"system_prompt": "", "tools": []}))
    session = AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=storage,
        cwd="/work",
    )
    try:
        stats = session.get_stats()
        assert stats.user_messages == 1
        assert stats.assistant_messages == 1
        assert stats.tool_calls == 1
        assert stats.tool_results == 1
        assert stats.tokens_input == 10
        assert stats.tokens_output == 20
        assert stats.tokens_cache_read == 5
        assert stats.tokens_cache_write == 3
        assert stats.cost == 3
    finally:
        session.close()
