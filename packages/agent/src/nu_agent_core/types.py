"""Core types for nu_agent_core.

Direct port of ``packages/agent/src/types.ts`` from the upstream TypeScript
monorepo.

Python adaptations:

* ``Message`` union (from nu_ai) + ``CustomAgentMessages`` extension point →
  in Python we expose ``AgentMessage`` as an open ``Message | BaseModel``
  type. Apps can still add their own message kinds; there is no
  declaration-merging equivalent in Python, so custom messages are added by
  subclassing a registered base and relying on runtime isinstance checks.
* ``AgentEvent`` is a :class:`typing.TypedDict` union keyed by ``type``.
  Using ``TypedDict`` (rather than Pydantic) keeps the agent loop's
  dispatch semantics identical to the upstream ``event.type === ...``
  switch while avoiding runtime validation overhead on every event.
* Abort signals: TypeScript ``AbortSignal`` maps to :class:`asyncio.Event`
  in Python. Callers check ``signal.is_set()`` instead of ``signal.aborted``.
* ``SimpleStreamOptions`` inheritance: :class:`AgentLoopConfig` is a
  dataclass that stores the raw config; when the agent loop calls into
  nu_ai it constructs a fresh :class:`SimpleStreamOptions` from the config's
  fields. This avoids forcing callers to subclass a Pydantic model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypedDict,
)

from nu_ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ToolResultMessage,
)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable

    from nu_ai.types import AssistantMessageEvent, Model, ThinkingBudgets
    from nu_ai.utils.event_stream import AssistantMessageEventStream


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

type ToolExecutionMode = Literal["sequential", "parallel"]
"""Sequential: one tool at a time. Parallel: preflight, then concurrent.

Default in :class:`AgentLoopConfig` is ``"parallel"`` to match upstream.
"""

type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
"""Thinking/reasoning level for models that support it.

``xhigh`` is only honoured by OpenAI gpt-5.1-codex-max, gpt-5.2, gpt-5.2-codex,
gpt-5.3, and gpt-5.3-codex. Other models treat it as ``high``.
"""


# ---------------------------------------------------------------------------
# AgentMessage — unions with apps' custom message kinds.
# ---------------------------------------------------------------------------

# In TypeScript apps extend ``CustomAgentMessages`` via declaration merging.
# Python has no direct equivalent, so ``AgentMessage`` stays permissive at
# the type level (``Message | Any``) while the agent loop branches on
# ``role`` / ``type`` for dispatch. Downstream apps may subclass
# ``nu_ai.types._Model`` for their own messages — ``convert_to_llm`` is the
# choke point that normalises them to LLM-compatible messages.
type AgentMessage = Message | Any


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentContext:
    """Snapshot of state passed into the low-level agent loop."""

    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool[Any, Any]] | None = None


# ---------------------------------------------------------------------------
# AgentToolResult / AgentTool
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentToolResult[TDetails]:
    """Final or partial result produced by a tool."""

    content: list[TextContent | ImageContent]
    details: TDetails


type AgentToolUpdateCallback[TDetails] = Callable[[AgentToolResult[TDetails]], None]
"""Callback used by tools to stream partial execution updates."""


@dataclass(slots=True)
class AgentTool[TParams, TDetails]:
    """Tool definition used by the agent runtime.

    Unlike :class:`nu_ai.types.Tool` this includes a UI label, optional
    prompt-snippet metadata used by system-prompt builders, an optional
    ``prepare_arguments`` shim, and the actual ``execute`` callable.

    Upstream pi-coding-agent splits these fields across two wrapper
    types (``AgentTool`` for the runtime, ``ToolDefinition`` for the
    coding-agent extras). The Python port collapses both into a single
    dataclass — the optional fields default to ``None``/``[]`` so any
    code that only cares about the runtime contract can ignore them.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema for the tool arguments."""
    label: str
    """Human-readable label for UI display."""
    execute: Callable[..., Awaitable[AgentToolResult[TDetails]]]
    """Async callable: ``(tool_call_id, params, signal, on_update) -> AgentToolResult``."""
    prepare_arguments: Callable[[Any], TParams] | None = None
    """Optional compatibility shim applied before schema validation."""
    prompt_snippet: str | None = None
    """One-line description used in the system prompt's tool list.

    ``nu_coding_agent.core.system_prompt.build_system_prompt`` only
    surfaces tools that have a non-empty snippet.
    """
    prompt_guidelines: list[str] | None = None
    """Optional bullet-point guidelines appended to the system prompt."""


# ---------------------------------------------------------------------------
# AgentEvent (TypedDict discriminated union keyed by ``type``)
# ---------------------------------------------------------------------------


class _AgentStartEvent(TypedDict):
    type: Literal["agent_start"]


class _AgentEndEvent(TypedDict):
    type: Literal["agent_end"]
    messages: list[AgentMessage]


class _TurnStartEvent(TypedDict):
    type: Literal["turn_start"]


class _TurnEndEvent(TypedDict):
    type: Literal["turn_end"]
    message: AgentMessage
    tool_results: list[ToolResultMessage]


class _MessageStartEvent(TypedDict):
    type: Literal["message_start"]
    message: AgentMessage


class _MessageUpdateEvent(TypedDict):
    type: Literal["message_update"]
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent


class _MessageEndEvent(TypedDict):
    type: Literal["message_end"]
    message: AgentMessage


class _ToolExecutionStartEvent(TypedDict):
    type: Literal["tool_execution_start"]
    tool_call_id: str
    tool_name: str
    args: Any


class _ToolExecutionUpdateEvent(TypedDict):
    type: Literal["tool_execution_update"]
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


class _ToolExecutionEndEvent(TypedDict):
    type: Literal["tool_execution_end"]
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


type AgentEvent = (
    _AgentStartEvent
    | _AgentEndEvent
    | _TurnStartEvent
    | _TurnEndEvent
    | _MessageStartEvent
    | _MessageUpdateEvent
    | _MessageEndEvent
    | _ToolExecutionStartEvent
    | _ToolExecutionUpdateEvent
    | _ToolExecutionEndEvent
)


# ---------------------------------------------------------------------------
# Hook results and contexts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BeforeToolCallResult:
    """Return value from the ``before_tool_call`` hook.

    Setting ``block=True`` prevents execution; ``reason`` becomes the text
    shown in the synthetic error tool result.
    """

    block: bool | None = None
    reason: str | None = None


@dataclass(slots=True)
class AfterToolCallResult:
    """Partial override returned from the ``after_tool_call`` hook.

    Field-by-field merge semantics — omitted fields keep the original
    executed tool result values.
    """

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None


@dataclass(slots=True)
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: Any
    """The ``ToolCall`` content block from ``assistant_message.content``."""
    args: Any
    context: AgentContext


@dataclass(slots=True)
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: Any
    args: Any
    result: AgentToolResult[Any]
    is_error: bool
    context: AgentContext


# ---------------------------------------------------------------------------
# AgentLoopConfig
# ---------------------------------------------------------------------------


type ConvertToLlmFn = Callable[[list[AgentMessage]], Awaitable[list[Message]] | list[Message]]
type TransformContextFn = Callable[
    [list[AgentMessage], "asyncio.Event | None"],
    Awaitable[list[AgentMessage]],
]
type GetApiKeyFn = Callable[[str], Awaitable[str | None] | str | None]
type GetMessagesFn = Callable[[], Awaitable[list[AgentMessage]]]
type BeforeToolCallHook = Callable[
    [BeforeToolCallContext, "asyncio.Event | None"],
    Awaitable[BeforeToolCallResult | None],
]
type AfterToolCallHook = Callable[
    [AfterToolCallContext, "asyncio.Event | None"],
    Awaitable[AfterToolCallResult | None],
]


@dataclass(slots=True)
class AgentLoopConfig:
    """Configuration passed to :func:`nu_agent_core.agent_loop.run_agent_loop`.

    Stores raw fields rather than inheriting from
    :class:`nu_ai.types.SimpleStreamOptions` because Pydantic models don't
    naturally carry callable fields; the agent loop rebuilds a
    ``SimpleStreamOptions`` for each LLM call from
    ``temperature``/``max_tokens``/``reasoning``/``thinking_budgets`` etc.
    """

    model: Model
    convert_to_llm: ConvertToLlmFn

    # SimpleStreamOptions-style knobs, forwarded to nu_ai.stream_simple.
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    reasoning: ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = None
    cache_retention: Literal["none", "short", "long"] | None = None
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] | None = None

    transform_context: TransformContextFn | None = None
    get_api_key: GetApiKeyFn | None = None
    get_steering_messages: GetMessagesFn | None = None
    get_follow_up_messages: GetMessagesFn | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None


# ---------------------------------------------------------------------------
# StreamFn
# ---------------------------------------------------------------------------


class StreamFn(Protocol):
    """Stream function signature used by the agent loop.

    Defaults to :func:`nu_ai.stream_simple`; tests supply a scripted
    alternative (e.g. one that drives a faux provider). Must not raise —
    failures are encoded in the returned stream as an error event.
    """

    def __call__(
        self,
        model: Model,
        context: Any,  # nu_ai.Context, but kept loose to match stream_simple's signature
        options: Any | None = None,
        /,
    ) -> AssistantMessageEventStream: ...


# ---------------------------------------------------------------------------
# AgentState (public read-only view of the stateful Agent)
# ---------------------------------------------------------------------------


class AgentState(Protocol):
    """Public agent state — implemented by :class:`nu_agent_core.agent.Agent`."""

    system_prompt: str
    model: Model
    thinking_level: ThinkingLevel
    tools: list[AgentTool[Any, Any]]
    messages: list[AgentMessage]
    is_streaming: bool
    streaming_message: AgentMessage | None
    pending_tool_calls: frozenset[str]
    error_message: str | None


# ---------------------------------------------------------------------------
# Re-exports that agent_loop expects
# ---------------------------------------------------------------------------


__all__ = [
    "AfterToolCallContext",
    "AfterToolCallHook",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentMessage",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "BeforeToolCallContext",
    "BeforeToolCallHook",
    "BeforeToolCallResult",
    "ConvertToLlmFn",
    "GetApiKeyFn",
    "GetMessagesFn",
    "StreamFn",
    "ThinkingLevel",
    "ToolExecutionMode",
    "TransformContextFn",
]
