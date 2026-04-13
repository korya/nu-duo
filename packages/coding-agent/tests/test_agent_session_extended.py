"""Tests for extended AgentSession properties and methods.

Tests cover the new properties and methods added in Tier 5:
- Session info (session_id, session_name, session_file, is_streaming, is_compacting, etc.)
- Thinking level management
- Queue modes (steering, follow-up)
- Auto-compaction / auto-retry settings
- Steer / follow-up / abort
- get_last_assistant_text
- get_user_messages_for_forking
"""

from __future__ import annotations

import time

from nu_agent_core.agent import Agent, AgentOptions
from nu_ai import AssistantMessage, Cost, Model, ModelCost, TextContent, Usage, UserMessage
from nu_coding_agent.core.agent_session import AgentSession
from nu_coding_agent.core.auth_storage import AuthStorage
from nu_coding_agent.core.model_registry import ModelRegistry
from nu_coding_agent.core.session_manager import SessionManager


def _make_model() -> Model:
    return Model(
        id="test-model",
        name="test-model",
        api="openai-completions",
        provider="openai",
        base_url="http://localhost",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128_000,
        max_tokens=16_384,
    )


def _make_assistant_msg(text: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        api="openai-completions",
        provider="openai",
        model="test-model",
        usage=Usage(
            input=10,
            output=5,
            cache_read=0,
            cache_write=0,
            total_tokens=15,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def _make_session() -> AgentSession:
    model = _make_model()
    auth = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(auth)
    agent = Agent(AgentOptions(initial_state={"model": model, "system_prompt": "", "tools": []}))
    sm = SessionManager.in_memory()
    return AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=auth,
        cwd="/tmp",
    )


# ---------------------------------------------------------------------------
# Session info
# ---------------------------------------------------------------------------


class TestSessionInfo:
    def test_session_id(self) -> None:
        session = _make_session()
        assert isinstance(session.session_id, str)
        assert len(session.session_id) > 0

    def test_session_name_initially_none(self) -> None:
        session = _make_session()
        assert session.session_name is None

    def test_set_session_name(self) -> None:
        session = _make_session()
        session.set_session_name("My Session")
        assert session.session_name == "My Session"

    def test_session_file(self) -> None:
        session = _make_session()
        # In-memory session manager has no file
        assert session.session_file is None

    def test_is_streaming_initially_false(self) -> None:
        session = _make_session()
        assert session.is_streaming is False

    def test_is_compacting_initially_false(self) -> None:
        session = _make_session()
        assert session.is_compacting is False

    def test_pending_message_count(self) -> None:
        session = _make_session()
        assert session.pending_message_count == 0

    def test_messages_initially_empty(self) -> None:
        session = _make_session()
        assert session.messages == []


# ---------------------------------------------------------------------------
# Thinking level
# ---------------------------------------------------------------------------


class TestThinkingLevel:
    def test_default_is_off(self) -> None:
        session = _make_session()
        assert session.thinking_level == "off"

    def test_set_thinking_level(self) -> None:
        session = _make_session()
        session.set_thinking_level("high")
        assert session.thinking_level == "high"

    def test_cycle_thinking_level(self) -> None:
        session = _make_session()
        assert session.thinking_level == "off"
        assert session.cycle_thinking_level() == "low"
        assert session.cycle_thinking_level() == "medium"
        assert session.cycle_thinking_level() == "high"
        assert session.cycle_thinking_level() == "off"

    def test_cycle_from_unknown_starts_at_low(self) -> None:
        session = _make_session()
        session.set_thinking_level("unknown")
        # Unknown index raises ValueError, defaults to 0
        result = session.cycle_thinking_level()
        assert result == "low"


# ---------------------------------------------------------------------------
# Queue modes
# ---------------------------------------------------------------------------


class TestQueueModes:
    def test_default_steering_mode(self) -> None:
        session = _make_session()
        assert session.steering_mode == "all"

    def test_set_steering_mode(self) -> None:
        session = _make_session()
        session.set_steering_mode("one-at-a-time")
        assert session.steering_mode == "one-at-a-time"

    def test_default_follow_up_mode(self) -> None:
        session = _make_session()
        assert session.follow_up_mode == "all"

    def test_set_follow_up_mode(self) -> None:
        session = _make_session()
        session.set_follow_up_mode("one-at-a-time")
        assert session.follow_up_mode == "one-at-a-time"


# ---------------------------------------------------------------------------
# Auto-compaction / auto-retry
# ---------------------------------------------------------------------------


class TestAutoSettings:
    def test_auto_compaction_enabled_by_default(self) -> None:
        session = _make_session()
        assert session.auto_compaction_enabled is True

    def test_toggle_auto_compaction(self) -> None:
        session = _make_session()
        session.set_auto_compaction_enabled(False)
        assert session.auto_compaction_enabled is False
        session.set_auto_compaction_enabled(True)
        assert session.auto_compaction_enabled is True

    def test_auto_retry_enabled_by_default(self) -> None:
        session = _make_session()
        assert session.auto_retry_enabled is True

    def test_toggle_auto_retry(self) -> None:
        session = _make_session()
        session.set_auto_retry_enabled(False)
        assert session.auto_retry_enabled is False

    def test_abort_retry_no_op(self) -> None:
        session = _make_session()
        # Should not raise
        session.abort_retry()


# ---------------------------------------------------------------------------
# Steer / follow-up / abort
# ---------------------------------------------------------------------------


class TestSteerFollowUpAbort:
    def test_steer(self) -> None:
        session = _make_session()
        # Should not raise (message gets queued on the agent)
        session.steer("Hey, focus on this!")

    def test_follow_up(self) -> None:
        session = _make_session()
        # Should not raise
        session.follow_up("What about this?")

    def test_abort(self) -> None:
        session = _make_session()
        # Should not raise even when not streaming
        session.abort()

    def test_abort_bash_no_op(self) -> None:
        session = _make_session()
        session.abort_bash()


# ---------------------------------------------------------------------------
# get_last_assistant_text
# ---------------------------------------------------------------------------


class TestGetLastAssistantText:
    def test_returns_none_when_no_messages(self) -> None:
        session = _make_session()
        assert session.get_last_assistant_text() is None

    def test_returns_last_assistant_text(self) -> None:
        session = _make_session()
        # Add messages to agent state
        session.agent.state.messages = [
            UserMessage(role="user", content="hello", timestamp=int(time.time() * 1000)),
            _make_assistant_msg("Hi there!"),
        ]
        assert session.get_last_assistant_text() == "Hi there!"

    def test_returns_last_when_multiple(self) -> None:
        session = _make_session()
        ts = int(time.time() * 1000)
        session.agent.state.messages = [
            UserMessage(role="user", content="first", timestamp=ts),
            _make_assistant_msg("First reply"),
            UserMessage(role="user", content="second", timestamp=ts),
            _make_assistant_msg("Second reply"),
        ]
        assert session.get_last_assistant_text() == "Second reply"


# ---------------------------------------------------------------------------
# get_user_messages_for_forking
# ---------------------------------------------------------------------------


class TestGetUserMessagesForForking:
    def test_returns_empty_when_no_entries(self) -> None:
        session = _make_session()
        assert session.get_user_messages_for_forking() == []

    def test_returns_user_messages_from_session(self) -> None:
        session = _make_session()
        # Manually add entries to session manager
        ts = int(time.time() * 1000)
        user_msg = {"role": "user", "content": [{"type": "text", "text": "Hello world"}], "timestamp": ts}
        session.session_manager.append_message(user_msg)
        result = session.get_user_messages_for_forking()
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"
        assert "entryId" in result[0]


# ---------------------------------------------------------------------------
# reload / bind_extensions
# ---------------------------------------------------------------------------


class TestReloadBindExtensions:
    async def test_reload_no_op(self) -> None:
        session = _make_session()
        await session.reload()  # Should not raise

    async def test_bind_extensions_no_op(self) -> None:
        session = _make_session()
        await session.bind_extensions(ui_context={})  # Should not raise
