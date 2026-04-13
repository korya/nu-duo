"""nu_agent_core — stateful agent loop.

Port of ``packages/agent/src/index.ts``.  Re-exports the full public surface
in a flat namespace so callers can do::

    from nu_agent_core import Agent, agent_loop, stream_proxy, AgentLoopConfig
"""

# Core Agent class
from nu_agent_core.agent import Agent, AgentListener, AgentOptions

# Loop functions
from nu_agent_core.agent_loop import (
    AgentEventSink,
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
)

# Proxy utilities
from nu_agent_core.proxy import ProxyMessageEventStream, ProxyStreamOptions, stream_proxy

# Types
from nu_agent_core.types import (
    AfterToolCallContext,
    AfterToolCallHook,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    BeforeToolCallContext,
    BeforeToolCallHook,
    ConvertToLlmFn,
    GetApiKeyFn,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
    TransformContextFn,
)

__all__ = [
    "AfterToolCallContext",
    "AfterToolCallHook",
    "Agent",
    "AgentContext",
    "AgentEvent",
    "AgentEventSink",
    "AgentListener",
    "AgentLoopConfig",
    "AgentMessage",
    "AgentOptions",
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "BeforeToolCallContext",
    "BeforeToolCallHook",
    "ConvertToLlmFn",
    "GetApiKeyFn",
    "ProxyMessageEventStream",
    "ProxyStreamOptions",
    "StreamFn",
    "ThinkingLevel",
    "ToolExecutionMode",
    "TransformContextFn",
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
    "stream_proxy",
]
