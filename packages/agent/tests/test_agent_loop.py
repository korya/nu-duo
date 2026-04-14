"""Tests for nu_agent_core.agent_loop.

The faux provider drives every test — no real model traffic. Tests cover
the documented contracts in ``packages/agent/src/agent-loop.ts``:

* Single-turn text response → emits agent_start, turn_start, message_*,
  turn_end, agent_end in order.
* Tool call → tool execution → second turn → final answer.
* Sequential vs parallel tool execution preserves source order.
* ``before_tool_call`` hook can block execution with a custom reason.
* ``after_tool_call`` hook can override content / details / is_error.
* Unknown tool → synthetic error tool result.
* Steering messages injected mid-run.
* Follow-up messages after the agent would otherwise stop.
* Errored / aborted assistant terminates the loop without further turns.
* :func:`agent_loop_continue` rejects empty context / assistant-last context.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from nu_agent_core.agent_loop import (
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
)
from nu_agent_core.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from nu_ai.api_registry import get_api_provider
from nu_ai.providers.faux import (
    faux_assistant_message,
    faux_tool_call,
    register_faux_provider,
)
from nu_ai.types import (
    AssistantMessage,
    Cost,
    Message,
    Model,
    ModelCost,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)

if TYPE_CHECKING:
    from nu_ai.utils.event_stream import AssistantMessageEventStream

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Default convert_to_llm: pass through Message-typed entries."""
    return [m for m in messages if isinstance(m, UserMessage | AssistantMessage | ToolResultMessage)]


def _bash_tool(
    *,
    raise_on_call: Exception | None = None,
    return_text: str = "ran ok",
    sleep: float = 0,
) -> AgentTool[dict[str, Any], dict[str, Any]]:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[dict[str, Any]]:
        if sleep:
            await asyncio.sleep(sleep)
        if raise_on_call is not None:
            raise raise_on_call
        return AgentToolResult(
            content=[TextContent(text=return_text)],
            details={"tool_call_id": tool_call_id, "params": params},
        )

    return AgentTool[dict[str, Any], dict[str, Any]](
        name="bash",
        description="run a bash command",
        parameters={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
        label="Bash",
        execute=execute,
    )


def _make_stream_fn(api: str) -> Any:
    """Build a StreamFn that calls into the registered faux provider."""
    provider = get_api_provider(api)
    assert provider is not None

    def stream_fn(model: Any, context: Any, options: Any | None = None) -> AssistantMessageEventStream:
        # The faux provider exposes the same callable as ``stream_simple``.
        return provider.stream_simple(model, context, options)

    return stream_fn


# ---------------------------------------------------------------------------
# Single-turn text response
# ---------------------------------------------------------------------------


class TestSingleTurnText:
    async def test_text_response_emits_lifecycle(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("hello back")])

            events: list[AgentEvent] = []

            async def emit(event: AgentEvent) -> None:
                events.append(event)

            cfg = AgentLoopConfig(
                model=registration.get_model(),
                convert_to_llm=_convert_to_llm,
            )
            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(system_prompt="", messages=[]),
                config=cfg,
                emit=emit,
                stream_fn=_make_stream_fn(registration.api),
            )

            types = [e["type"] for e in events]
            # Lifecycle: agent_start, turn_start, prompt message_start/end,
            # streamed message_start/updates/end, turn_end, agent_end.
            assert types[0] == "agent_start"
            assert types[1] == "turn_start"
            assert types[-1] == "agent_end"
            assert "message_start" in types
            assert "message_end" in types
            assert "turn_end" in types

            # The returned messages list contains the user prompt + assistant reply.
            assert len(messages) == 2
            assert isinstance(messages[0], UserMessage)
            assert isinstance(messages[1], AssistantMessage)
            assert isinstance(messages[1].content[0], TextContent)
            assert messages[1].content[0].text == "hello back"
        finally:
            registration.unregister()

    async def test_agent_loop_event_stream_resolves_with_messages(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("done")])

            stream = agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                stream_fn=_make_stream_fn(registration.api),
            )
            events = [e async for e in stream]
            assert events[-1]["type"] == "agent_end"
            messages = await stream.result()
            assert len(messages) == 2
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


class TestToolExecution:
    async def test_tool_call_then_final_response(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="call_1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("here is the listing"),
                ]
            )

            events: list[AgentEvent] = []

            async def emit(event: AgentEvent) -> None:
                events.append(event)

            messages = await run_agent_loop(
                prompts=[UserMessage(content="ls please", timestamp=1)],
                context=AgentContext(tools=[_bash_tool()]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=emit,
                stream_fn=_make_stream_fn(registration.api),
            )

            types = [e["type"] for e in events]
            assert types.count("turn_start") == 2
            assert "tool_execution_start" in types
            assert "tool_execution_end" in types

            # Final message list: user, assistant(tool call), tool result, assistant(text).
            roles = [m.role for m in messages]
            assert roles == ["user", "assistant", "toolResult", "assistant"]

            tool_result = messages[2]
            assert isinstance(tool_result, ToolResultMessage)
            assert tool_result.tool_name == "bash"
            assert tool_result.is_error is False
            assert isinstance(tool_result.content[0], TextContent)
            assert tool_result.content[0].text == "ran ok"

            final = messages[3]
            assert isinstance(final, AssistantMessage)
            assert isinstance(final.content[0], TextContent)
            assert final.content[0].text == "here is the listing"
        finally:
            registration.unregister()

    async def test_unknown_tool_returns_error_tool_result(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("nonexistent", {}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("recovered"),
                ]
            )
            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(tools=[_bash_tool()]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            tool_result = messages[2]
            assert isinstance(tool_result, ToolResultMessage)
            # The TS contract returns the synthetic result with the message text.
            assert isinstance(tool_result.content[0], TextContent)
            assert "nonexistent" in tool_result.content[0].text
        finally:
            registration.unregister()

    async def test_tool_execution_failure_becomes_error_result(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("recovered"),
                ]
            )
            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(
                    tools=[_bash_tool(raise_on_call=RuntimeError("boom"))],
                ),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            tool_result = messages[2]
            assert isinstance(tool_result, ToolResultMessage)
            assert isinstance(tool_result.content[0], TextContent)
            assert "boom" in tool_result.content[0].text
        finally:
            registration.unregister()

    @pytest.mark.parametrize("mode", ["sequential", "parallel"])
    async def test_multiple_tool_calls_preserve_source_order(self, mode: str) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [
                            faux_tool_call("bash", {"cmd": "a"}, id_="call_a"),
                            faux_tool_call("bash", {"cmd": "b"}, id_="call_b"),
                        ],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("done"),
                ]
            )
            messages = await run_agent_loop(
                prompts=[UserMessage(content="run both", timestamp=1)],
                context=AgentContext(tools=[_bash_tool(sleep=0.01)]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                    tool_execution=mode,  # type: ignore[arg-type]
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            tool_results = [m for m in messages if isinstance(m, ToolResultMessage)]
            assert [r.tool_call_id for r in tool_results] == ["call_a", "call_b"]
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Hook results
# ---------------------------------------------------------------------------


class TestHooks:
    async def test_before_tool_call_blocks_with_reason(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "rm -rf /"}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("understood"),
                ]
            )

            async def before(
                ctx: BeforeToolCallContext,
                signal: Any = None,
            ) -> BeforeToolCallResult | None:
                return BeforeToolCallResult(block=True, reason="too dangerous")

            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(tools=[_bash_tool()]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                    before_tool_call=before,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            tool_result = messages[2]
            assert isinstance(tool_result, ToolResultMessage)
            assert isinstance(tool_result.content[0], TextContent)
            assert tool_result.content[0].text == "too dangerous"
            # Note: blocked tools currently aren't marked is_error=True (matches TS).
        finally:
            registration.unregister()

    async def test_after_tool_call_overrides_content_and_error_flag(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("done"),
                ]
            )

            async def after(
                ctx: AfterToolCallContext,
                signal: Any = None,
            ) -> AfterToolCallResult | None:
                return AfterToolCallResult(
                    content=[TextContent(text="overridden")],
                    is_error=True,
                )

            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(tools=[_bash_tool()]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                    after_tool_call=after,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            tool_result = messages[2]
            assert isinstance(tool_result, ToolResultMessage)
            assert tool_result.is_error is True
            assert isinstance(tool_result.content[0], TextContent)
            assert tool_result.content[0].text == "overridden"
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Steering and follow-up
# ---------------------------------------------------------------------------


class TestSteeringAndFollowUp:
    async def test_steering_messages_injected_after_turn(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("acknowledged steering"),
                    faux_assistant_message("final"),
                ]
            )

            steering_payloads: list[list[AgentMessage]] = [
                [UserMessage(content="actually do this instead", timestamp=2)],
                [],  # second call after the steering injection
                [],  # third call
            ]

            async def get_steering() -> list[AgentMessage]:
                return steering_payloads.pop(0) if steering_payloads else []

            messages = await run_agent_loop(
                prompts=[UserMessage(content="ls please", timestamp=1)],
                context=AgentContext(tools=[_bash_tool()]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                    get_steering_messages=get_steering,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            roles = [m.role for m in messages]
            # The steering message should appear in the middle of the run.
            assert roles.count("user") == 2  # original + steering
        finally:
            registration.unregister()

    async def test_follow_up_messages_continue_after_stop(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message("initial answer"),
                    faux_assistant_message("follow-up answer"),
                ]
            )

            follow_up_payloads: list[list[AgentMessage]] = [
                [UserMessage(content="actually one more thing", timestamp=2)],
                [],  # subsequent calls return empty → loop exits
            ]

            async def get_follow_up() -> list[AgentMessage]:
                return follow_up_payloads.pop(0) if follow_up_payloads else []

            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                    get_follow_up_messages=get_follow_up,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            # original user, first assistant, follow-up user, second assistant
            assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Error termination
# ---------------------------------------------------------------------------


class TestErrorTermination:
    async def test_errored_assistant_stops_loop_immediately(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        "boom",
                        stop_reason="error",
                        error_message="upstream failure",
                    ),
                    faux_assistant_message("never reached"),
                ]
            )
            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            # User + the errored assistant only — loop exited.
            assert len(messages) == 2
            errored = messages[1]
            assert isinstance(errored, AssistantMessage)
            assert errored.stop_reason == "error"
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# agent_loop_continue
# ---------------------------------------------------------------------------


class TestAgentLoopContinue:
    async def test_continue_from_existing_user_message(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("answer")])
            ctx = AgentContext(messages=[UserMessage(content="ping", timestamp=1)])
            new_messages = await run_agent_loop_continue(
                context=ctx,
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            # Continue does NOT add a new prompt; it just runs another turn.
            # The new_messages list contains only what was added during this run.
            assert len(new_messages) == 1
            assert isinstance(new_messages[0], AssistantMessage)
        finally:
            registration.unregister()

    async def test_continue_rejects_empty_context(self) -> None:
        with pytest.raises(ValueError, match="no messages"):
            agent_loop_continue(
                context=AgentContext(),
                config=AgentLoopConfig(
                    model_dummy(),
                    convert_to_llm=_convert_to_llm,
                ),
            )

    async def test_continue_streams_via_event_stream(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("answer")])
            ctx = AgentContext(messages=[UserMessage(content="ping", timestamp=1)])
            stream = agent_loop_continue(
                context=ctx,
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                stream_fn=_make_stream_fn(registration.api),
            )
            events: list[AgentEvent] = []
            async for event in stream:
                events.append(event)
            assert len(events) > 0
            messages = await stream.result()
            assert any(isinstance(m, AssistantMessage) for m in messages)
        finally:
            registration.unregister()

    async def test_continue_rejects_assistant_last_context(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="prior")],
            api="faux",
            provider="faux",
            model="m",
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            ),
            stop_reason="stop",
            timestamp=1,
        )
        with pytest.raises(ValueError, match="assistant"):
            agent_loop_continue(
                context=AgentContext(messages=[msg]),
                config=AgentLoopConfig(
                    model_dummy(),
                    convert_to_llm=_convert_to_llm,
                ),
            )


# ---------------------------------------------------------------------------
# run_agent_loop_continue validation (imperative driver)
# ---------------------------------------------------------------------------


class TestRunAgentLoopContinueValidation:
    async def test_rejects_empty_context(self) -> None:
        with pytest.raises(ValueError, match="no messages"):
            await run_agent_loop_continue(
                context=AgentContext(),
                config=AgentLoopConfig(
                    model=model_dummy(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=_noop_emit,
            )

    async def test_rejects_assistant_last(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="x")],
            api="faux",
            provider="faux",
            model="m",
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            ),
            stop_reason="stop",
            timestamp=1,
        )
        with pytest.raises(ValueError, match="assistant"):
            await run_agent_loop_continue(
                context=AgentContext(messages=[msg]),
                config=AgentLoopConfig(
                    model=model_dummy(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=_noop_emit,
            )


# ---------------------------------------------------------------------------
# Abort signal mid-turn
# ---------------------------------------------------------------------------


class TestAbortSignal:
    async def test_aborted_assistant_stops_loop(self) -> None:
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        "partial",
                        stop_reason="aborted",
                        error_message="cancelled",
                    ),
                ]
            )
            events: list[AgentEvent] = []

            async def emit(event: AgentEvent) -> None:
                events.append(event)

            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            assert len(messages) == 2
            assert messages[1].stop_reason == "aborted"
            types = [e["type"] for e in events]
            assert "agent_end" in types
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Tool with prepare_arguments
# ---------------------------------------------------------------------------


class TestPrepareArguments:
    async def test_prepare_arguments_modifies_args(self) -> None:
        """Cover _prepare_tool_call_arguments with a non-None prepare_arguments."""
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("done"),
                ]
            )

            def prepare_arguments(args: dict[str, Any]) -> dict[str, Any]:
                return {**args, "extra": "injected"}

            tool = _bash_tool()
            tool.prepare_arguments = prepare_arguments

            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(tools=[tool]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            tool_result = messages[2]
            assert isinstance(tool_result, ToolResultMessage)
            assert tool_result.is_error is False
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Tool execution with on_update callback
# ---------------------------------------------------------------------------


class TestToolExecutionOnUpdate:
    async def test_on_update_emits_tool_execution_update(self) -> None:
        """Cover lines 692-706 and 715-716: on_update + gather."""
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("done"),
                ]
            )

            async def execute_with_updates(
                tool_call_id: str,
                params: dict[str, Any],
                signal: Any = None,
                on_update: Any = None,
            ) -> AgentToolResult[dict[str, Any]]:
                if on_update:
                    on_update(AgentToolResult(
                        content=[TextContent(text="partial")],
                        details={},
                    ))
                return AgentToolResult(
                    content=[TextContent(text="final")],
                    details={},
                )

            tool = _bash_tool()
            tool.execute = execute_with_updates

            events: list[AgentEvent] = []

            async def emit(event: AgentEvent) -> None:
                events.append(event)

            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(tools=[tool]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            types = [e["type"] for e in events]
            assert "tool_execution_update" in types
        finally:
            registration.unregister()

    async def test_exception_with_pending_updates(self) -> None:
        """Cover lines 718-720: exception path with pending update emits."""
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("recovered"),
                ]
            )

            async def execute_with_updates_then_fail(
                tool_call_id: str,
                params: dict[str, Any],
                signal: Any = None,
                on_update: Any = None,
            ) -> AgentToolResult[dict[str, Any]]:
                if on_update:
                    on_update(AgentToolResult(
                        content=[TextContent(text="partial")],
                        details={},
                    ))
                raise RuntimeError("boom after update")

            tool = _bash_tool()
            tool.execute = execute_with_updates_then_fail

            events: list[AgentEvent] = []

            async def emit(event: AgentEvent) -> None:
                events.append(event)

            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(tools=[tool]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            types = [e["type"] for e in events]
            assert "tool_execution_update" in types
            assert "tool_execution_end" in types
            # Tool result should be an error
            tool_result = [m for m in messages if isinstance(m, ToolResultMessage)][0]
            assert tool_result.is_error is True
            assert "boom after update" in tool_result.content[0].text
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Sync emit (non-awaitable callback)
# ---------------------------------------------------------------------------


class TestSyncEmit:
    async def test_sync_emit_callback(self) -> None:
        """Cover the non-awaitable branch of _emit (line 817)."""
        registration = register_faux_provider()
        try:
            registration.set_responses([faux_assistant_message("hi")])

            events: list[AgentEvent] = []

            def sync_emit(event: AgentEvent) -> None:  # note: not async
                events.append(event)

            messages = await run_agent_loop(
                prompts=[UserMessage(content="hi", timestamp=1)],
                context=AgentContext(),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                ),
                emit=sync_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            assert len(messages) == 2
            assert len(events) > 0
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Sync get_steering_messages (non-awaitable)
# ---------------------------------------------------------------------------


class TestSyncSteeringMessages:
    async def test_sync_get_steering_messages(self) -> None:
        """Cover the non-awaitable branch in _maybe_call_get_messages (line 829)."""
        registration = register_faux_provider()
        try:
            registration.set_responses(
                [
                    faux_assistant_message(
                        [faux_tool_call("bash", {"cmd": "ls"}, id_="c1")],
                        stop_reason="toolUse",
                    ),
                    faux_assistant_message("done"),
                ]
            )
            call_count = 0

            def sync_get_steering() -> list[AgentMessage]:  # not async
                nonlocal call_count
                call_count += 1
                return []

            messages = await run_agent_loop(
                prompts=[UserMessage(content="go", timestamp=1)],
                context=AgentContext(tools=[_bash_tool()]),
                config=AgentLoopConfig(
                    model=registration.get_model(),
                    convert_to_llm=_convert_to_llm,
                    get_steering_messages=sync_get_steering,
                ),
                emit=_noop_emit,
                stream_fn=_make_stream_fn(registration.api),
            )
            assert call_count > 0
            assert len(messages) >= 2
        finally:
            registration.unregister()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


async def _noop_emit(event: AgentEvent) -> None:
    pass


def model_dummy() -> Any:
    return Model(
        id="m",
        name="m",
        api="faux",
        provider="faux",
        base_url="",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=0,
        max_tokens=0,
    )
