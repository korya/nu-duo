"""Tests for nu_agent_core.agent.Agent.

Covers the documented contracts in ``packages/agent/src/agent.ts``:

* ``prompt`` accepts a string, a single message, or a list of messages.
* State (``messages``, ``is_streaming``, ``streaming_message``,
  ``pending_tool_calls``, ``error_message``) reflects the loop's lifecycle.
* ``subscribe`` listeners are called for every event in subscription order.
* ``steer`` and ``follow_up`` queue messages with ``"all"`` and
  ``"one-at-a-time"`` drain modes.
* ``abort`` cancels in flight.
* ``wait_for_idle`` resolves only after the run finishes.
* ``reset`` clears transcript and queues.
* ``continue_run`` works after appending a tool result; rejects an
  assistant-last context unless a steering or follow-up message is queued.
* Concurrent ``prompt`` calls raise.
* Run failure becomes a synthetic assistant error message.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from nu_agent_core.agent import Agent, AgentOptions
from nu_agent_core.types import AgentEvent, AgentTool, AgentToolResult
from nu_ai.api_registry import get_api_provider
from nu_ai.providers.faux import (
    faux_assistant_message,
    faux_tool_call,
    register_faux_provider,
)
from nu_ai.types import (
    AssistantMessage,
    Cost,
    ImageContent,
    Message,
    TextContent,
    Usage,
    UserMessage,
)


def _stream_fn_for(api: str) -> Any:
    provider = get_api_provider(api)
    assert provider is not None

    def stream_fn(model: Any, context: Any, options: Any | None = None) -> Any:
        return provider.stream_simple(model, context, options)

    return stream_fn


def _bash_tool() -> AgentTool[dict[str, Any], dict[str, Any]]:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[dict[str, Any]]:
        return AgentToolResult(
            content=[TextContent(text="ok")],
            details={"tool_call_id": tool_call_id, "params": params},
        )

    return AgentTool[dict[str, Any], dict[str, Any]](
        name="bash",
        description="run a command",
        parameters={"type": "object", "properties": {}},
        label="Bash",
        execute=execute,
    )


# ---------------------------------------------------------------------------
# prompt() — happy paths
# ---------------------------------------------------------------------------


class TestPrompt:
    async def test_prompt_string(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("hello back")])
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            await agent.prompt("hi")
            messages = agent.state.messages
            assert len(messages) == 2
            assert isinstance(messages[0], UserMessage)
            assert isinstance(messages[0].content, list)
            assert isinstance(messages[0].content[0], TextContent)
            assert messages[0].content[0].text == "hi"
            assert isinstance(messages[1], AssistantMessage)
            assert isinstance(messages[1].content[0], TextContent)
            assert messages[1].content[0].text == "hello back"
            assert agent.state.is_streaming is False
            assert agent.state.streaming_message is None
        finally:
            registration.unregister()

    async def test_prompt_string_with_images(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("got it")])
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            await agent.prompt("look", images=[ImageContent(data="d", mime_type="image/png")])
            user = agent.state.messages[0]
            assert isinstance(user, UserMessage)
            assert isinstance(user.content, list)
            assert any(isinstance(c, ImageContent) for c in user.content)
        finally:
            registration.unregister()

    async def test_prompt_message_list(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("ack")])
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            await agent.prompt(
                [
                    UserMessage(content="first", timestamp=1),
                    UserMessage(content="second", timestamp=2),
                ]
            )
            roles = [m.role for m in agent.state.messages]
            assert roles == ["user", "user", "assistant"]
        finally:
            registration.unregister()

    async def test_prompt_during_active_run_raises(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message("first"),
                    faux_assistant_message("second"),
                ]
            )
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )

            first_task = asyncio.create_task(agent.prompt("hi"))
            # Yield so the task starts running before we call prompt() again.
            await asyncio.sleep(0)
            with pytest.raises(RuntimeError, match="already processing"):
                await agent.prompt("hi again")
            await first_task
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Subscribe lifecycle
# ---------------------------------------------------------------------------


class TestSubscribe:
    async def test_listener_receives_events_in_order(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("hi")])
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            received: list[str] = []

            async def listener(event: AgentEvent, signal: Any) -> None:
                received.append(event["type"])

            unsubscribe = agent.subscribe(listener)
            try:
                await agent.prompt("hello")
            finally:
                unsubscribe()
            assert received[0] == "agent_start"
            assert received[-1] == "agent_end"
            assert "turn_start" in received
            assert "message_start" in received
            assert "message_end" in received
        finally:
            registration.unregister()

    async def test_unsubscribe_stops_listener(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message("first"),
                    faux_assistant_message("second"),
                ]
            )
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            received: list[str] = []

            async def listener(event: AgentEvent, signal: Any) -> None:
                received.append(event["type"])

            unsubscribe = agent.subscribe(listener)
            await agent.prompt("first")
            unsubscribe()
            received.clear()
            await agent.prompt("second")
            assert received == []
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Steering and follow-up queues
# ---------------------------------------------------------------------------


class TestQueues:
    async def test_steer_injects_after_assistant_turn(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("acknowledged"),
                    faux_assistant_message("final"),
                ]
            )
            agent = Agent(
                AgentOptions(
                    initial_state={
                        "model": registration.get_model(),
                        "tools": [_bash_tool()],
                    },
                    stream_fn=_stream_fn_for(registration.api),
                )
            )

            # Subscribe a listener that calls steer() after the first turn ends.
            steered = False

            async def listener(event: AgentEvent, signal: Any) -> None:
                nonlocal steered
                if event["type"] == "turn_end" and not steered:
                    steered = True
                    agent.steer(UserMessage(content="steering message", timestamp=999))

            agent.subscribe(listener)
            await agent.prompt("start")

            user_messages = [m for m in agent.state.messages if isinstance(m, UserMessage)]
            assert any(isinstance(m.content, str) and m.content == "steering message" for m in user_messages)
        finally:
            registration.unregister()

    async def test_follow_up_drains_after_stop(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message("first"),
                    faux_assistant_message("after follow-up"),
                ]
            )
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            agent.follow_up(UserMessage(content="actually one more thing", timestamp=2))
            await agent.prompt("hi")
            roles = [m.role for m in agent.state.messages]
            # original user, first assistant, follow-up user, second assistant
            assert roles == ["user", "assistant", "user", "assistant"]
        finally:
            registration.unregister()

    async def test_clear_queues_removes_pending(self) -> None:
        agent = Agent()
        agent.steer(UserMessage(content="a", timestamp=1))
        agent.follow_up(UserMessage(content="b", timestamp=1))
        assert agent.has_queued_messages() is True
        agent.clear_all_queues()
        assert agent.has_queued_messages() is False

    async def test_steering_mode_all_drains_everything(self) -> None:
        agent = Agent(AgentOptions(steering_mode="all"))
        agent.steer(UserMessage(content="a", timestamp=1))
        agent.steer(UserMessage(content="b", timestamp=1))
        # Internal drain method is the only way to verify directly.
        drained = agent._steering_queue.drain()  # pyright: ignore[reportPrivateUsage]
        assert len(drained) == 2

    async def test_steering_mode_one_at_a_time(self) -> None:
        agent = Agent(AgentOptions(steering_mode="one-at-a-time"))
        agent.steer(UserMessage(content="a", timestamp=1))
        agent.steer(UserMessage(content="b", timestamp=1))
        drained = agent._steering_queue.drain()  # pyright: ignore[reportPrivateUsage]
        assert len(drained) == 1


# ---------------------------------------------------------------------------
# Abort and waitForIdle
# ---------------------------------------------------------------------------


class TestAbort:
    async def test_abort_with_no_active_run_is_noop(self) -> None:
        agent = Agent()
        agent.abort()  # should not raise

    async def test_signal_is_none_when_idle(self) -> None:
        agent = Agent()
        assert agent.signal is None

    async def test_wait_for_idle_returns_immediately_when_idle(self) -> None:
        agent = Agent()
        await agent.wait_for_idle()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    async def test_reset_clears_messages_and_queues(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("hi")])
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            await agent.prompt("hello")
            agent.steer(UserMessage(content="x", timestamp=1))
            agent.reset()
            assert agent.state.messages == []
            assert agent.state.is_streaming is False
            assert agent.state.streaming_message is None
            assert agent.state.error_message is None
            assert agent.has_queued_messages() is False
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# continue_run
# ---------------------------------------------------------------------------


class TestContinue:
    async def test_continue_after_user_message(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("answer")])
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            agent.state.messages.append(UserMessage(content="ping", timestamp=1))
            await agent.continue_run()
            assert isinstance(agent.state.messages[-1], AssistantMessage)
        finally:
            registration.unregister()

    async def test_continue_with_no_messages_raises(self) -> None:
        agent = Agent()
        with pytest.raises(RuntimeError, match="No messages"):
            await agent.continue_run()

    async def test_continue_with_assistant_last_and_no_queue_raises(self) -> None:
        registration = register_faux_provider()
        try:
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_stream_fn_for(registration.api),
                )
            )
            agent.state.messages.append(
                AssistantMessage(
                    content=[TextContent(text="prior")],
                    api="faux",
                    provider="faux",
                    model="faux-1",
                    usage=_empty_usage(),
                    stop_reason="stop",
                    timestamp=1,
                )
            )
            with pytest.raises(RuntimeError, match="assistant"):
                await agent.continue_run()
        finally:
            registration.unregister()


def _empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


# ---------------------------------------------------------------------------
# Run failure
# ---------------------------------------------------------------------------


class TestRunFailure:
    async def test_stream_fn_raising_creates_error_assistant_message(self) -> None:
        registration = register_faux_provider()
        try:
            agent = Agent(
                AgentOptions(
                    initial_state={"model": registration.get_model()},
                    stream_fn=_failing_stream_fn,
                )
            )
            await agent.prompt("hi")
            last = agent.state.messages[-1]
            assert isinstance(last, AssistantMessage)
            assert last.stop_reason == "error"
            assert last.error_message is not None
            assert "boom" in last.error_message
            assert agent.state.is_streaming is False
            assert agent.state.error_message is not None
        finally:
            registration.unregister()


def _failing_stream_fn(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# State views
# ---------------------------------------------------------------------------


class TestStateViews:
    async def test_state_tools_assignment_copies_array(self) -> None:
        agent = Agent()
        original = [_bash_tool()]
        agent.state.tools = original  # type: ignore[misc]
        # Mutating the original list must not affect the agent's tools.
        original.clear()
        assert len(agent.state.tools) == 1

    async def test_state_messages_assignment_copies_array(self) -> None:
        agent = Agent()
        msgs: list[Message] = [UserMessage(content="x", timestamp=1)]
        agent.state.messages = msgs  # type: ignore[misc]
        msgs.clear()
        assert len(agent.state.messages) == 1
