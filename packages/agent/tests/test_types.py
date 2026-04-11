"""Tests for pi_agent_core.types.

Ports the documented contracts from ``packages/agent/src/types.ts``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pi_agent_core.types import (
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentState,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
    ThinkingLevel,
    ToolExecutionMode,
)
from pi_ai.types import (
    AssistantMessage,
    Cost,
    Message,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model() -> Model:
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


def _echo_tool() -> AgentTool[dict[str, Any], None]:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: object | None = None,
        on_update: object | None = None,
    ) -> AgentToolResult[None]:
        return AgentToolResult(content=[TextContent(text=str(params))], details=None)

    return AgentTool[dict[str, Any], None](
        name="echo",
        description="echo parameters back",
        parameters={"type": "object", "properties": {}},
        label="Echo",
        execute=execute,
    )


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


class TestAgentContext:
    def test_empty_context(self) -> None:
        ctx = AgentContext(system_prompt="", messages=[])
        assert ctx.messages == []
        assert ctx.tools is None

    def test_with_messages_and_tools(self) -> None:
        user = UserMessage(content="hi", timestamp=1)
        ctx = AgentContext(system_prompt="sys", messages=[user], tools=[_echo_tool()])
        assert ctx.tools is not None
        assert len(ctx.tools) == 1
        assert ctx.tools[0].name == "echo"


# ---------------------------------------------------------------------------
# AgentTool
# ---------------------------------------------------------------------------


class TestAgentTool:
    def test_tool_shape(self) -> None:
        tool = _echo_tool()
        assert tool.name == "echo"
        assert tool.label == "Echo"
        assert tool.description == "echo parameters back"
        assert "type" in tool.parameters

    def test_tool_is_also_base_tool_shape(self) -> None:
        tool = _echo_tool()
        # AgentTool extends pi_ai Tool — name/description/parameters fields match.
        base_tool = Tool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )
        assert base_tool.name == tool.name


# ---------------------------------------------------------------------------
# AgentToolResult
# ---------------------------------------------------------------------------


class TestAgentToolResult:
    def test_text_only_result(self) -> None:
        r = AgentToolResult[dict[str, int]](
            content=[TextContent(text="ok")],
            details={"exit_code": 0},
        )
        assert r.details == {"exit_code": 0}
        assert r.content[0].type == "text"

    def test_generic_details(self) -> None:
        r = AgentToolResult[str](content=[], details="stringly typed")
        assert r.details == "stringly typed"


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


class TestAgentEventDiscriminatedUnion:
    def test_agent_start(self) -> None:
        evt: AgentEvent = {"type": "agent_start"}
        assert evt["type"] == "agent_start"

    def test_agent_end(self) -> None:
        msgs: list[Message] = [UserMessage(content="hi", timestamp=1)]
        evt: AgentEvent = {"type": "agent_end", "messages": msgs}
        assert evt["type"] == "agent_end"
        assert len(evt["messages"]) == 1

    def test_turn_end_carries_tool_results(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="done")],
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
        tr = ToolResultMessage(
            tool_call_id="c",
            tool_name="t",
            content=[TextContent(text="ok")],
            is_error=False,
            timestamp=1,
        )
        evt: AgentEvent = {"type": "turn_end", "message": msg, "tool_results": [tr]}
        assert evt["type"] == "turn_end"
        assert len(evt["tool_results"]) == 1


# ---------------------------------------------------------------------------
# BeforeToolCallResult / AfterToolCallResult
# ---------------------------------------------------------------------------


class TestHookResults:
    def test_before_tool_call_block(self) -> None:
        r = BeforeToolCallResult(block=True, reason="denied")
        assert r.block is True
        assert r.reason == "denied"

    def test_before_tool_call_noop(self) -> None:
        r = BeforeToolCallResult()
        assert r.block is None
        assert r.reason is None

    def test_after_tool_call_partial_override(self) -> None:
        r = AfterToolCallResult(content=[TextContent(text="override")])
        assert r.details is None
        assert r.is_error is None
        assert r.content is not None


# ---------------------------------------------------------------------------
# AgentLoopConfig
# ---------------------------------------------------------------------------


class TestAgentLoopConfig:
    def test_minimal_config(self) -> None:
        async def convert_to_llm(messages: list[Message]) -> list[Message]:
            return list(messages)

        cfg = AgentLoopConfig(model=_model(), convert_to_llm=convert_to_llm)
        assert cfg.model.id == "m"
        assert cfg.tool_execution == "parallel"  # default

    def test_sequential_tool_execution(self) -> None:
        async def convert_to_llm(messages: list[Message]) -> list[Message]:
            return list(messages)

        cfg = AgentLoopConfig(
            model=_model(),
            convert_to_llm=convert_to_llm,
            tool_execution="sequential",
        )
        assert cfg.tool_execution == "sequential"


# ---------------------------------------------------------------------------
# ThinkingLevel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["off", "minimal", "low", "medium", "high", "xhigh"])
def test_thinking_level_values(level: ThinkingLevel) -> None:
    assert level in {"off", "minimal", "low", "medium", "high", "xhigh"}


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_tool_execution_mode_values(mode: ToolExecutionMode) -> None:
    assert mode in {"sequential", "parallel"}


# ---------------------------------------------------------------------------
# AgentState (Protocol shape)
# ---------------------------------------------------------------------------


class TestAgentStateShape:
    def test_protocol_accepts_minimal_implementation(self) -> None:
        class _Impl:
            def __init__(self) -> None:
                self.system_prompt = "sys"
                self.model = _model()
                self.thinking_level: ThinkingLevel = "off"
                self.tools: list[AgentTool[Any, Any]] = []
                self.messages: list[Message] = []
                self.is_streaming = False
                self.streaming_message: Message | None = None
                self.pending_tool_calls: frozenset[str] = frozenset()
                self.error_message: str | None = None

        impl = _Impl()
        # runtime_checkable not used — we just verify the attributes match.
        assert isinstance(impl.system_prompt, str)
        assert impl.is_streaming is False
        _state: AgentState = impl  # type: ignore[assignment]
