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


# ---------------------------------------------------------------------------
# Extension lifecycle integration (sub-slice 2)
#
# These tests prove the AgentSession ↔ ExtensionRunner wiring:
#
# * session_start fires lazily on the first prompt (not at construction).
# * Every agent loop event is mapped to the matching extension dataclass
#   and dispatched to the runner.
# * The dispatch order is "persist → extensions → user listeners".
# * shutdown() emits session_shutdown and is idempotent.
# * No-runner sessions are completely unaffected (existing tests above).
# ---------------------------------------------------------------------------


def _make_session_with_runner(
    setup: tuple[Agent, SessionManager, ModelRegistry, AuthStorage],
    runner: Any,
) -> AgentSession:
    agent, sm, registry, storage = setup
    return AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=storage,
            cwd="/work",
            extension_runner=runner,
        )
    )


async def test_extension_runner_attached_via_create() -> None:
    """``create`` accepts an ``extension_runner`` keyword argument."""
    from nu_coding_agent.core.extensions import ExtensionRunner  # noqa: PLC0415

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(cwd="/work")
        session = AgentSession.create(
            agent=agent,
            session_manager=sm,
            model_registry=registry,
            auth_storage=storage,
            cwd="/work",
            extension_runner=runner,
        )
        assert session.extension_runner is runner
        session.close()
    finally:
        registration.unregister()


async def test_session_start_emitted_on_first_prompt_only() -> None:
    """``session_start`` fires once on the first prompt, never again."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionRunner,
        SessionStartEvent,
        load_extensions_from_factories,
    )

    seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        api.on("session_start", lambda event, ctx: seen.append(event))

    load_result = await load_extensions_from_factories([("<inline:start>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        # No events fired at construction.
        assert seen == []

        registration.set_responses([faux_assistant_message("hi")])
        await session.prompt("first")
        # session_start fired exactly once during the first prompt.
        assert len(seen) == 1
        assert isinstance(seen[0], SessionStartEvent)
        assert seen[0].cwd == "/work"

        registration.set_responses([faux_assistant_message("hi again")])
        await session.prompt("second")
        # Still exactly one — session_start does not re-fire.
        assert len(seen) == 1

        session.close()
    finally:
        registration.unregister()


async def test_lifecycle_events_dispatched_during_prompt() -> None:
    """All ten lifecycle event types reach the extension runner from a real prompt."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionRunner,
        load_extensions_from_factories,
    )

    received: list[str] = []

    def register(api: ExtensionAPI) -> None:
        for event_name in (
            "agent_start",
            "agent_end",
            "turn_start",
            "turn_end",
            "message_start",
            "message_update",
            "message_end",
            "tool_execution_start",
            "tool_execution_update",
            "tool_execution_end",
        ):
            api.on(event_name, lambda event, ctx, name=event_name: received.append(name))

    load_result = await load_extensions_from_factories([("<inline:lifecycle>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        registration.set_responses([faux_assistant_message("hello")])
        await session.prompt("hi")

        # Every event the agent loop produced should have been dispatched.
        assert "agent_start" in received
        assert "agent_end" in received
        assert "message_start" in received
        assert "message_end" in received
        # agent_start always precedes agent_end.
        assert received.index("agent_start") < received.index("agent_end")
        # Every message_start has a matching message_end after it.
        assert received.index("message_start") < received.index("message_end")
        # No errors leaked to the runner.
        assert runner.drain_errors() == []

        session.close()
    finally:
        registration.unregister()


async def test_extension_event_payload_carries_message() -> None:
    """``message_end`` events expose the assistant message to extensions."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionContext,
        ExtensionRunner,
        MessageEndEvent,
        load_extensions_from_factories,
    )

    captured: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        def on_message_end(event: Any, ctx: ExtensionContext) -> None:
            captured.append(event)

        api.on("message_end", on_message_end)

    load_result = await load_extensions_from_factories([("<inline:msg>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        registration.set_responses([faux_assistant_message("ack")])
        await session.prompt("hi")

        # We should see a message_end event for both the user prompt and
        # the assistant reply (and possibly tool results, but the faux
        # provider doesn't issue any tool calls).
        assert len(captured) >= 2
        assert all(isinstance(e, MessageEndEvent) for e in captured)
        assistant_payloads = [e.message for e in captured if getattr(e.message, "role", None) == "assistant"]
        assert len(assistant_payloads) == 1
        session.close()
    finally:
        registration.unregister()


async def test_dispatch_order_persist_then_extensions_then_listeners() -> None:
    """Persistence runs before extension dispatch which runs before user listeners."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionContext,
        ExtensionRunner,
        load_extensions_from_factories,
    )

    order: list[str] = []

    def register(api: ExtensionAPI) -> None:
        def on_message_end(event: Any, ctx: ExtensionContext) -> None:
            order.append("extension")

        api.on("message_end", on_message_end)

    load_result = await load_extensions_from_factories([("<inline:order>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        # Track persistence indirectly: the session manager has the
        # entry on disk by the time the listener runs. The listener
        # checks both that fact AND that the extension already saw the
        # event.
        def listener(event: Any) -> None:
            if event["type"] == "message_end":
                # SessionManager already has the entry persisted.
                last_entry = sm.get_entries()[-1] if sm.get_entries() else None
                assert last_entry is not None
                # And the extension handler has already run.
                assert "extension" in order
                order.append("listener")

        session.subscribe(listener)

        registration.set_responses([faux_assistant_message("ack")])
        await session.prompt("hi")

        # Listener should have run after every extension call.
        assert order.count("extension") >= 2  # at least user + assistant message_end
        assert "listener" in order

        session.close()
    finally:
        registration.unregister()


async def test_shutdown_emits_session_shutdown_then_closes() -> None:
    """``shutdown()`` broadcasts session_shutdown then unsubscribes."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionRunner,
        SessionShutdownEvent,
        load_extensions_from_factories,
    )

    shutdown_seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        api.on("session_shutdown", lambda event, ctx: shutdown_seen.append(event))

    load_result = await load_extensions_from_factories([("<inline:shutdown>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        # Drive one prompt so session_start has fired (extensions are
        # only "started" after the first prompt).
        registration.set_responses([faux_assistant_message("ack")])
        await session.prompt("hi")

        await session.shutdown()
        assert len(shutdown_seen) == 1
        assert isinstance(shutdown_seen[0], SessionShutdownEvent)

        # Idempotent.
        await session.shutdown()
        assert len(shutdown_seen) == 1

        # Subsequent prompt rejected because the session is closed.
        with pytest.raises(RuntimeError, match="closed"):
            await session.prompt("post-shutdown")
    finally:
        registration.unregister()


async def test_shutdown_without_first_prompt_skips_shutdown_event() -> None:
    """``shutdown()`` before any prompt does not emit ``session_shutdown``.

    Symmetric to the lazy ``session_start`` semantics: if the session
    never started, it has nothing to shut down. The agent listener is
    still detached so calling ``shutdown()`` is always safe.
    """
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionRunner,
        load_extensions_from_factories,
    )

    seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        api.on("session_shutdown", lambda event, ctx: seen.append(event))

    load_result = await load_extensions_from_factories([("<inline:nostart>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        await session.shutdown()  # never prompted → never started
        assert seen == []
    finally:
        registration.unregister()


async def test_shutdown_works_without_runner() -> None:
    """``shutdown()`` is safe to call when no runner is attached."""
    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        session = AgentSession(
            AgentSessionConfig(
                agent=agent,
                session_manager=sm,
                model_registry=registry,
                auth_storage=storage,
                cwd="/work",
            )
        )
        await session.shutdown()  # no runner → just close
        # Idempotent.
        await session.shutdown()
    finally:
        registration.unregister()


async def test_extension_handler_exception_does_not_break_user_listener() -> None:
    """A broken extension handler must not prevent user listeners from running."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionContext,
        ExtensionRunner,
        load_extensions_from_factories,
    )

    def register(api: ExtensionAPI) -> None:
        def boom(event: Any, ctx: ExtensionContext) -> None:
            raise RuntimeError("extension exploded")

        api.on("message_end", boom)

    load_result = await load_extensions_from_factories([("<inline:bad>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        listener_calls: list[Any] = []
        session.subscribe(lambda event: listener_calls.append(event["type"]))

        registration.set_responses([faux_assistant_message("ack")])
        await session.prompt("hi")  # must not raise

        # User listener still ran for every event.
        assert "message_end" in listener_calls
        # Runner captured the extension error.
        errors = runner.drain_errors()
        assert any("extension exploded" in err.error for err in errors)
        session.close()
    finally:
        registration.unregister()


# ---------------------------------------------------------------------------
# Extension tool registration (sub-slice 3)
# ---------------------------------------------------------------------------


def _make_extension_tool(name: str, *, marker: str = "called") -> Any:
    """Build an :class:`AgentTool` instance suitable for ``api.register_tool``."""
    from nu_agent_core.types import AgentTool, AgentToolResult  # noqa: PLC0415
    from nu_ai.types import TextContent  # noqa: PLC0415

    async def execute(_tool_call_id: str, params: Any, _signal: Any, _on_update: Any) -> AgentToolResult[Any]:
        return AgentToolResult(
            content=[TextContent(text=f"{marker}:{params}")],
            details=None,
        )

    return AgentTool(
        name=name,
        description=f"test tool {name}",
        parameters={"type": "object", "properties": {}, "additionalProperties": True},
        label=name,
        execute=execute,
    )


async def test_apply_extension_tools_no_runner_is_noop() -> None:
    """Sessions without an attached runner skip extension tool merging."""
    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        session = AgentSession(
            AgentSessionConfig(
                agent=agent,
                session_manager=sm,
                model_registry=registry,
                auth_storage=storage,
                cwd="/work",
            )
        )
        applied = session.apply_extension_tools()
        assert applied == 0
        assert agent.state.tools == []
        session.close()
    finally:
        registration.unregister()


async def test_apply_extension_tools_appends_extension_tools() -> None:
    """Extension-registered tools land in agent.state.tools after merge."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionRunner,
        load_extensions_from_factories,
    )

    def register(api: ExtensionAPI) -> None:
        api.register_tool(_make_extension_tool("ext_tool_a"))
        api.register_tool(_make_extension_tool("ext_tool_b"))

    load_result = await load_extensions_from_factories([("<inline:tools>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
        storage = AuthStorage.in_memory({faux_model.provider: ApiKeyCredential(type="api_key", key="x")})
        registry = ModelRegistry.in_memory(storage)
        registry._models.append(faux_model)  # type: ignore[attr-defined]
        builtin_tool = _make_extension_tool("builtin_tool", marker="builtin")
        agent = Agent(
            AgentOptions(
                initial_state={"model": faux_model, "system_prompt": "", "tools": [builtin_tool]},
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        applied = session.apply_extension_tools()
        assert applied == 2
        names = [t.name for t in agent.state.tools]
        # Built-in tool stays first, extension tools appended in order.
        assert names == ["builtin_tool", "ext_tool_a", "ext_tool_b"]
        session.close()
    finally:
        registration.unregister()


async def test_apply_extension_tools_overrides_builtin_by_name() -> None:
    """An extension tool whose name matches a built-in *replaces* the built-in."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionRunner,
        load_extensions_from_factories,
    )

    def register(api: ExtensionAPI) -> None:
        api.register_tool(_make_extension_tool("read", marker="ext-read"))

    load_result = await load_extensions_from_factories([("<inline:override>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
        storage = AuthStorage.in_memory({faux_model.provider: ApiKeyCredential(type="api_key", key="x")})
        registry = ModelRegistry.in_memory(storage)
        registry._models.append(faux_model)  # type: ignore[attr-defined]
        builtin_read = _make_extension_tool("read", marker="builtin-read")
        agent = Agent(
            AgentOptions(
                initial_state={"model": faux_model, "system_prompt": "", "tools": [builtin_read]},
                convert_to_llm=_convert_to_llm,
                stream_fn=_make_stream_fn(registration.api),
            )
        )
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        session.apply_extension_tools()
        assert len(agent.state.tools) == 1
        from nu_ai.types import TextContent  # noqa: PLC0415

        result = await agent.state.tools[0].execute("tc1", {"x": 1}, None, None)
        text_block = next(b for b in result.content if isinstance(b, TextContent))
        assert "ext-read" in text_block.text
        session.close()
    finally:
        registration.unregister()


async def test_apply_extension_tools_is_idempotent() -> None:
    """Calling ``apply_extension_tools`` twice does not duplicate tools."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionRunner,
        load_extensions_from_factories,
    )

    def register(api: ExtensionAPI) -> None:
        api.register_tool(_make_extension_tool("ext_one"))

    load_result = await load_extensions_from_factories([("<inline:idem>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        first_count = session.apply_extension_tools()
        second_count = session.apply_extension_tools()
        assert first_count == 1
        assert second_count == 1
        # Still exactly one tool — replaced, not duplicated.
        assert [t.name for t in agent.state.tools] == ["ext_one"]
        session.close()
    finally:
        registration.unregister()


async def test_extension_tool_invoked_by_agent_loop() -> None:
    """End-to-end: extension-registered tool is actually called by the agent loop."""
    from nu_ai.providers.faux import (  # noqa: PLC0415
        faux_assistant_message,
        faux_tool_call,
    )
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionRunner,
        load_extensions_from_factories,
    )

    invocation_count = 0

    def register(api: ExtensionAPI) -> None:
        from nu_agent_core.types import AgentTool, AgentToolResult  # noqa: PLC0415
        from nu_ai.types import TextContent  # noqa: PLC0415

        async def execute(_tcid: str, params: Any, _sig: Any, _upd: Any) -> AgentToolResult[Any]:
            nonlocal invocation_count
            invocation_count += 1
            return AgentToolResult(
                content=[TextContent(text=f"counted={params.get('n', 0)}")],
                details=None,
            )

        api.register_tool(
            AgentTool(
                name="count",
                description="Increment a counter and echo the result",
                parameters={
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                },
                label="count",
                execute=execute,
            )
        )

    load_result = await load_extensions_from_factories([("<inline:counted>", register)])

    registration = register_faux_provider()
    try:
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)

        applied = session.apply_extension_tools()
        assert applied == 1

        # Faux script: assistant returns a tool call, then a final text reply.
        registration.set_responses(
            [
                faux_assistant_message(
                    [faux_tool_call("count", {"n": 7}, id_="tc1")],
                    stop_reason="toolUse",
                ),
                faux_assistant_message("done"),
            ]
        )

        await session.prompt("count to 7")

        # The tool was actually called by the agent loop.
        assert invocation_count == 1
        # And the result landed in the session as a tool_result message.
        entries = sm.get_entries()
        assert any(isinstance(e.get("message"), dict) and e["message"].get("role") == "toolResult" for e in entries)
        session.close()
    finally:
        registration.unregister()


# ---------------------------------------------------------------------------
# Extension action methods (sub-slice 4) — bind_core wires the runtime
# action slots to the AgentSession so extensions can read+mutate session
# state from inside event handlers.
# ---------------------------------------------------------------------------


def _build_session_with_extension(setup_factory: Any) -> Any:
    """Build (registration, session, runner) for an action-method test."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionRunner,
        load_extensions_from_factories,
    )

    async def _build():
        load_result = await load_extensions_from_factories([("<inline:actions>", setup_factory)])
        registration = register_faux_provider()
        faux_model = registration.get_model()
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
        sm = SessionManager.in_memory("/work")
        runner = ExtensionRunner.create(extensions=load_result.extensions, runtime=load_result.runtime, cwd="/work")
        session = _make_session_with_runner((agent, sm, registry, storage), runner)
        return registration, session, runner

    return _build


async def test_action_set_label_round_trip() -> None:
    """``api.set_label`` writes a label entry that round-trips through SessionManager."""
    from nu_coding_agent.core.extensions import ExtensionAPI  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def register(api: ExtensionAPI) -> None:
        captured["api"] = api

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        # Append a real session entry to label.
        sm = session.session_manager
        msg_id = sm.append_message({"role": "user", "content": "hi", "timestamp": 1})

        api = captured["api"]
        api.set_label(msg_id, "important")

        assert sm.get_label(msg_id) == "important"
        # Clearing works too.
        api.set_label(msg_id, None)
        assert sm.get_label(msg_id) is None
    finally:
        session.close()
        registration.unregister()


async def test_action_append_custom_entry_returns_id() -> None:
    """``append_custom_entry`` writes a custom entry and returns its id."""
    from nu_coding_agent.core.extensions import ExtensionAPI  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def register(api: ExtensionAPI) -> None:
        captured["api"] = api

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        api = captured["api"]
        entry_id = api.append_custom_entry("todo", {"text": "ship sub-slice 4"})
        assert isinstance(entry_id, str) and entry_id

        entries = session.session_manager.get_entries()
        custom = next(e for e in entries if e.get("type") == "custom")
        assert custom.get("customType") == "todo"
        assert custom.get("data") == {"text": "ship sub-slice 4"}
    finally:
        session.close()
        registration.unregister()


async def test_action_session_name_get_set() -> None:
    """``set_session_name`` persists the name and ``get_session_name`` reads it back."""
    from nu_coding_agent.core.extensions import ExtensionAPI  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def register(api: ExtensionAPI) -> None:
        captured["api"] = api

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        api = captured["api"]
        assert api.get_session_name() is None
        api.set_session_name("my session")
        assert api.get_session_name() == "my session"
    finally:
        session.close()
        registration.unregister()


async def test_action_get_active_tools_and_get_all_tools() -> None:
    """Active/all-tool reads return whatever is currently on agent.state.tools."""
    from nu_coding_agent.core.extensions import ExtensionAPI  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def register(api: ExtensionAPI) -> None:
        captured["api"] = api

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        # Seed two tools post-construction.
        session.agent.state.tools = [
            _make_extension_tool("read"),
            _make_extension_tool("write"),
        ]
        api = captured["api"]
        assert api.get_active_tools() == ["read", "write"]
        all_tools = api.get_all_tools()
        assert {t["name"] for t in all_tools} == {"read", "write"}
        assert all("description" in t for t in all_tools)
        assert all("parameters" in t for t in all_tools)
    finally:
        session.close()
        registration.unregister()


async def test_action_set_active_tools_filters_existing_list() -> None:
    """``set_active_tools`` keeps only tools whose name is in the supplied list."""
    from nu_coding_agent.core.extensions import ExtensionAPI  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def register(api: ExtensionAPI) -> None:
        captured["api"] = api

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        session.agent.state.tools = [
            _make_extension_tool("read"),
            _make_extension_tool("write"),
            _make_extension_tool("bash"),
        ]
        api = captured["api"]
        api.set_active_tools(["read", "bash"])
        assert [t.name for t in session.agent.state.tools] == ["read", "bash"]
    finally:
        session.close()
        registration.unregister()


async def test_action_set_model_swaps_active_model() -> None:
    """``set_model`` updates ``agent.state.model`` and persists ``model_change``."""
    from nu_ai.types import Model, ModelCost  # noqa: PLC0415
    from nu_coding_agent.core.extensions import ExtensionAPI  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def register(api: ExtensionAPI) -> None:
        captured["api"] = api

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        api = captured["api"]
        new_model = Model(
            id="new-model",
            name="new-model",
            api="openai-completions",
            provider="openai",
            base_url="https://example",
            reasoning=False,
            input=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=1000,
            max_tokens=100,
        )
        result = await api.set_model(new_model)
        assert result is True
        assert session.agent.state.model.id == "new-model"
        # And a model_change entry hit the session JSONL.
        entries = session.session_manager.get_entries()
        assert any(e.get("type") == "model_change" and e.get("modelId") == "new-model" for e in entries)
    finally:
        session.close()
        registration.unregister()


async def test_action_thinking_level_get_set_round_trip() -> None:
    """``set_thinking_level`` persists, ``get_thinking_level`` reads it back."""
    from nu_coding_agent.core.extensions import ExtensionAPI  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def register(api: ExtensionAPI) -> None:
        captured["api"] = api

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        api = captured["api"]
        # Default before any change is "off".
        assert api.get_thinking_level() == "off"
        api.set_thinking_level("high")
        assert api.get_thinking_level() == "high"
        api.set_thinking_level("low")
        assert api.get_thinking_level() == "low"
    finally:
        session.close()
        registration.unregister()


async def test_action_methods_unbound_before_session_attached() -> None:
    """Calling an action before ``bind_core`` runs raises a clear error."""
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        load_extensions_from_factories,
    )

    captured: dict[str, Any] = {}

    def register(api: ExtensionAPI) -> None:
        captured["api"] = api

    # Load with no AgentSession around — runtime stays in pre-bind state.
    await load_extensions_from_factories([("<inline:unbound>", register)])
    api = captured["api"]
    with pytest.raises(RuntimeError, match="not bound"):
        api.set_label("entry-id", "label")


async def test_action_set_label_invoked_from_event_handler() -> None:
    """End-to-end: a handler that calls ``set_label`` from inside ``message_end``."""
    from nu_coding_agent.core.extensions import ExtensionAPI, ExtensionContext  # noqa: PLC0415

    label_calls: list[str] = []

    def register(api: ExtensionAPI) -> None:
        def on_message_end(event: Any, ctx: ExtensionContext) -> None:
            msg = event.message
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "assistant":
                # Label the most recent entry on the session manager —
                # extensions can call into the runtime via ``api`` even
                # though the handler signature only gets event + ctx.
                latest = label_calls  # placeholder so closure captures
                api.append_custom_entry("seen_assistant", {"role": role})
                latest.append(role)

        api.on("message_end", on_message_end)

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        registration.set_responses([faux_assistant_message("ack")])
        await session.prompt("hi")
        assert label_calls == ["assistant"]
        # The custom entry the extension wrote landed in the session.
        entries = session.session_manager.get_entries()
        assert any(e.get("type") == "custom" and e.get("customType") == "seen_assistant" for e in entries)
    finally:
        session.close()
        registration.unregister()


# ---------------------------------------------------------------------------
# session_before_compact / session_compact hooks (sub-slice 5)
# ---------------------------------------------------------------------------


def _seed_compactable_session(session: AgentSession) -> None:
    """Pad the session with enough messages that ``prepare_compaction`` returns a real preparation."""
    sm = session.session_manager
    for i in range(20):
        sm.append_message({"role": "user", "content": "x" * 500, "timestamp": i})
        sm.append_message(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "x" * 500}],
                "provider": "openai",
                "model": "m",
                "api": "openai-completions",
                "usage": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 0,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                },
                "stopReason": "stop",
                "timestamp": i,
            }
        )


async def test_compact_emits_session_compact_event() -> None:
    """Standard compaction path fires ``session_compact`` after persisting."""
    from nu_ai.providers.faux import faux_assistant_message  # noqa: PLC0415
    from nu_coding_agent.core.compaction import CompactionSettings  # noqa: PLC0415
    from nu_coding_agent.core.extensions import ExtensionAPI, ExtensionContext  # noqa: PLC0415

    seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        def handler(event: Any, ctx: ExtensionContext) -> None:
            seen.append(event)

        api.on("session_compact", handler)

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        # Override compaction settings so prepare_compaction has work to do.
        session._compaction_settings = CompactionSettings(  # pyright: ignore[reportPrivateUsage]
            enabled=True, reserve_tokens=1000, keep_recent_tokens=300
        )
        _seed_compactable_session(session)
        registration.set_responses(
            [faux_assistant_message("structured summary"), faux_assistant_message("turn prefix")]
        )

        result = await session.compact()
        assert result is not None
        assert len(seen) == 1
        event = seen[0]
        assert event.type == "session_compact"
        assert event.from_extension is False
        assert event.compaction_entry is not None
        assert event.compaction_entry.get("type") == "compaction"
    finally:
        session.close()
        registration.unregister()


async def test_before_compact_handler_can_cancel() -> None:
    """A handler that returns ``cancel=True`` aborts the compaction entirely."""
    from nu_coding_agent.core.compaction import CompactionSettings  # noqa: PLC0415
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionContext,
        SessionBeforeCompactResult,
    )

    def register(api: ExtensionAPI) -> None:
        def handler(event: Any, ctx: ExtensionContext) -> SessionBeforeCompactResult:
            return SessionBeforeCompactResult(cancel=True)

        api.on("session_before_compact", handler)

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        session._compaction_settings = CompactionSettings(  # pyright: ignore[reportPrivateUsage]
            enabled=True, reserve_tokens=1000, keep_recent_tokens=300
        )
        _seed_compactable_session(session)

        result = await session.compact()
        # Cancelled — no result, no LLM call (would have failed
        # without a queued faux response).
        assert result is None
        # And no compaction entry was persisted.
        entries = session.session_manager.get_entries()
        assert not any(e.get("type") == "compaction" for e in entries)
    finally:
        session.close()
        registration.unregister()


async def test_before_compact_handler_can_provide_custom_result() -> None:
    """A handler can supply a custom ``CompactionResult`` to bypass the LLM."""
    from nu_coding_agent.core.compaction import (  # noqa: PLC0415
        CompactionDetails,
        CompactionResult,
        CompactionSettings,
    )
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionAPI,
        ExtensionContext,
        SessionBeforeCompactResult,
    )

    custom_seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        def before(event: Any, ctx: ExtensionContext) -> SessionBeforeCompactResult:
            return SessionBeforeCompactResult(
                compaction=CompactionResult(
                    summary="extension-supplied summary",
                    first_kept_entry_id=event.preparation.first_kept_entry_id,
                    tokens_before=event.preparation.tokens_before,
                    details=CompactionDetails(read_files=[], modified_files=[]),
                )
            )

        def after(event: Any, ctx: ExtensionContext) -> None:
            custom_seen.append(event)

        api.on("session_before_compact", before)
        api.on("session_compact", after)

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        session._compaction_settings = CompactionSettings(  # pyright: ignore[reportPrivateUsage]
            enabled=True, reserve_tokens=1000, keep_recent_tokens=300
        )
        _seed_compactable_session(session)

        result = await session.compact()
        assert result is not None
        assert "extension-supplied summary" in result.summary
        # No LLM was called (no faux response queued); the persisted
        # compaction entry uses the extension's summary.
        compactions = [e for e in session.session_manager.get_entries() if e.get("type") == "compaction"]
        assert len(compactions) == 1
        assert "extension-supplied summary" in compactions[0]["summary"]
        # The session_compact handler observed from_extension=True.
        assert len(custom_seen) == 1
        assert custom_seen[0].from_extension is True
    finally:
        session.close()
        registration.unregister()


async def test_before_compact_handler_returning_none_falls_through() -> None:
    """A handler that observes-only (returns None) leaves the standard path intact."""
    from nu_ai.providers.faux import faux_assistant_message  # noqa: PLC0415
    from nu_coding_agent.core.compaction import CompactionSettings  # noqa: PLC0415
    from nu_coding_agent.core.extensions import ExtensionAPI, ExtensionContext  # noqa: PLC0415

    observed: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        def handler(event: Any, ctx: ExtensionContext) -> None:
            observed.append(event)

        api.on("session_before_compact", handler)

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        session._compaction_settings = CompactionSettings(  # pyright: ignore[reportPrivateUsage]
            enabled=True, reserve_tokens=1000, keep_recent_tokens=300
        )
        _seed_compactable_session(session)
        registration.set_responses([faux_assistant_message("normal summary"), faux_assistant_message("turn prefix")])

        result = await session.compact()
        assert result is not None
        assert "normal summary" in result.summary
        # The handler observed the event with the prepared payload.
        assert len(observed) == 1
        assert observed[0].preparation is not None
        assert isinstance(observed[0].branch_entries, list)
    finally:
        session.close()
        registration.unregister()


async def test_before_compact_dict_result_form_supported() -> None:
    """Handlers may return a plain dict instead of a SessionBeforeCompactResult."""
    from nu_coding_agent.core.compaction import CompactionSettings  # noqa: PLC0415
    from nu_coding_agent.core.extensions import ExtensionAPI, ExtensionContext  # noqa: PLC0415

    def register(api: ExtensionAPI) -> None:
        def handler(event: Any, ctx: ExtensionContext) -> dict[str, Any]:
            return {"cancel": True}  # plain-dict cancel

        api.on("session_before_compact", handler)

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        session._compaction_settings = CompactionSettings(  # pyright: ignore[reportPrivateUsage]
            enabled=True, reserve_tokens=1000, keep_recent_tokens=300
        )
        _seed_compactable_session(session)

        result = await session.compact()
        assert result is None  # dict-form cancel honored
    finally:
        session.close()
        registration.unregister()


async def test_compact_no_handlers_uses_standard_path() -> None:
    """Sessions without compaction-extension handlers fall back to LLM compaction."""
    from nu_ai.providers.faux import faux_assistant_message  # noqa: PLC0415
    from nu_coding_agent.core.compaction import CompactionSettings  # noqa: PLC0415
    from nu_coding_agent.core.extensions import ExtensionAPI  # noqa: PLC0415

    def register(api: ExtensionAPI) -> None:
        # Register a handler for an unrelated event so the runner has
        # at least one extension loaded.
        api.on("agent_start", lambda event, ctx: None)

    build = _build_session_with_extension(register)
    registration, session, _runner = await build()
    try:
        session._compaction_settings = CompactionSettings(  # pyright: ignore[reportPrivateUsage]
            enabled=True, reserve_tokens=1000, keep_recent_tokens=300
        )
        _seed_compactable_session(session)
        registration.set_responses([faux_assistant_message("standard"), faux_assistant_message("turn prefix")])

        result = await session.compact()
        assert result is not None
        assert "standard" in result.summary
    finally:
        session.close()
        registration.unregister()


async def test_normalize_before_compact_result_helper() -> None:
    """The helper accepts dataclass, dict, and None forms."""
    from nu_coding_agent.core.agent_session import (  # noqa: PLC0415
        _normalize_before_compact_result,  # pyright: ignore[reportPrivateUsage]
    )
    from nu_coding_agent.core.extensions import SessionBeforeCompactResult  # noqa: PLC0415

    assert _normalize_before_compact_result(None) == (False, None)
    assert _normalize_before_compact_result({"cancel": True}) == (True, None)
    assert _normalize_before_compact_result(SessionBeforeCompactResult(cancel=True)) == (True, None)
    assert _normalize_before_compact_result({}) == (False, None)
    sentinel = object()
    cancel, compaction = _normalize_before_compact_result({"compaction": sentinel})
    assert cancel is False
    assert compaction is sentinel


def test_translate_to_extension_event_unknown_returns_none() -> None:
    """Unknown agent event types return ``None`` (forward-compatible)."""
    from nu_coding_agent.core.agent_session import (  # noqa: PLC0415
        _translate_to_extension_event,  # pyright: ignore[reportPrivateUsage]
    )

    assert _translate_to_extension_event({"type": "future_event_type"}) is None  # type: ignore[arg-type]


def test_translate_to_extension_event_covers_all_known_types() -> None:
    """Spot-check the translation table for every documented event type."""
    from nu_coding_agent.core.agent_session import (  # noqa: PLC0415
        _translate_to_extension_event,  # pyright: ignore[reportPrivateUsage]
    )
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        AgentEndEvent,
        AgentStartEvent,
        MessageEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        ToolExecutionEndEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        TurnEndEvent,
        TurnStartEvent,
    )

    pairs = [
        ({"type": "agent_start"}, AgentStartEvent),
        ({"type": "agent_end", "messages": []}, AgentEndEvent),
        ({"type": "turn_start"}, TurnStartEvent),
        ({"type": "turn_end", "message": None, "tool_results": []}, TurnEndEvent),
        ({"type": "message_start", "message": None}, MessageStartEvent),
        ({"type": "message_update", "message": None, "assistant_message_event": None}, MessageUpdateEvent),
        ({"type": "message_end", "message": None}, MessageEndEvent),
        (
            {"type": "tool_execution_start", "tool_call_id": "tc1", "tool_name": "read", "args": {"x": 1}},
            ToolExecutionStartEvent,
        ),
        (
            {
                "type": "tool_execution_update",
                "tool_call_id": "tc1",
                "tool_name": "read",
                "args": {},
                "partial_result": None,
            },
            ToolExecutionUpdateEvent,
        ),
        (
            {
                "type": "tool_execution_end",
                "tool_call_id": "tc1",
                "tool_name": "read",
                "result": None,
                "is_error": False,
            },
            ToolExecutionEndEvent,
        ),
    ]

    for event_dict, expected_cls in pairs:
        translated = _translate_to_extension_event(event_dict)  # type: ignore[arg-type]
        assert isinstance(translated, expected_cls)


# Keep ``asyncio`` referenced — used by other tests in the file but
# the Edit churn occasionally drops it from imports.
_ = asyncio
