"""Stateful Agent wrapper.

Direct port of ``packages/agent/src/agent.ts``. The :class:`Agent` class
owns the current transcript, emits lifecycle events, executes tools, and
exposes queueing APIs for steering and follow-up messages — a thin
stateful skin over :func:`pi_agent_core.agent_loop.run_agent_loop`.

Python adaptations:

* TypeScript ``AbortController`` → :class:`asyncio.Event` (the agent's
  ``signal`` is set on :meth:`abort`).
* TypeScript ``continue()`` method → renamed to :meth:`continue_run` to
  avoid the Python keyword conflict.
* TypeScript getters/setters on ``state`` → a small ``_MutableAgentState``
  shim with copy-on-assign for ``tools``/``messages``.
* TypeScript overloaded ``prompt()`` → a single Python signature that
  accepts ``str`` / single message / list of messages, with an optional
  ``images`` keyword for the string variant.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pi_ai.types import (
    AssistantMessage,
    Cost,
    ImageContent,
    Message,
    Model,
    ModelCost,
    TextContent,
    Usage,
    UserMessage,
)

from pi_agent_core.agent_loop import run_agent_loop, run_agent_loop_continue
from pi_agent_core.types import (
    AfterToolCallHook,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    BeforeToolCallHook,
    ConvertToLlmFn,
    GetApiKeyFn,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
    TransformContextFn,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pi_ai.types import ThinkingBudgets, Transport


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


_EMPTY_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
)


_DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="unknown",
    provider="unknown",
    base_url="",
    reasoning=False,
    input=[],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=0,
    max_tokens=0,
)


type _QueueMode = Literal["all", "one-at-a-time"]
"""Steering / follow-up queue drain mode."""


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    return [
        m
        for m in messages
        if isinstance(m, UserMessage | AssistantMessage) or (hasattr(m, "role") and m.role == "toolResult")
    ]


# ---------------------------------------------------------------------------
# Mutable state shim
# ---------------------------------------------------------------------------


class _MutableAgentState:
    """Backing state for :class:`Agent` with copy-on-assign for collections."""

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: Model | None = None,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool[Any, Any]] | None = None,
        messages: list[AgentMessage] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model or _DEFAULT_MODEL
        self.thinking_level: ThinkingLevel = thinking_level
        self._tools: list[AgentTool[Any, Any]] = list(tools or [])
        self._messages: list[AgentMessage] = list(messages or [])
        self.is_streaming = False
        self.streaming_message: AgentMessage | None = None
        self.pending_tool_calls: frozenset[str] = frozenset()
        self.error_message: str | None = None

    @property
    def tools(self) -> list[AgentTool[Any, Any]]:
        return self._tools

    @tools.setter
    def tools(self, next_tools: list[AgentTool[Any, Any]]) -> None:
        self._tools = list(next_tools)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, next_messages: list[AgentMessage]) -> None:
        self._messages = list(next_messages)


# ---------------------------------------------------------------------------
# Pending message queue
# ---------------------------------------------------------------------------


class _PendingMessageQueue:
    """A small queue with two drain modes — ports the upstream behaviour."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._messages: list[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return bool(self._messages)

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            drained = list(self._messages)
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


# ---------------------------------------------------------------------------
# AgentOptions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentOptions:
    """Options for constructing an :class:`Agent`.

    Mirrors ``AgentOptions`` from ``packages/agent/src/agent.ts``. Field
    naming follows Python conventions (snake_case).
    """

    initial_state: dict[str, Any] | None = None
    convert_to_llm: ConvertToLlmFn | None = None
    transform_context: TransformContextFn | None = None
    stream_fn: StreamFn | None = None
    get_api_key: GetApiKeyFn | None = None
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    steering_mode: str = "one-at-a-time"
    follow_up_mode: str = "one-at-a-time"
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport | None = None
    max_retry_delay_ms: int | None = None
    tool_execution: ToolExecutionMode = "parallel"


# ---------------------------------------------------------------------------
# Active run record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ActiveRun:
    future: asyncio.Future[None]
    abort_event: asyncio.Event


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


type AgentListener = Callable[[AgentEvent, asyncio.Event], "Awaitable[None] | None"]
"""Listener signature: receives every event plus the active abort event."""


class Agent:
    """Stateful wrapper around the low-level agent loop.

    Owns the current transcript, emits lifecycle events, executes tools,
    and exposes queueing APIs for steering and follow-up messages.
    """

    def __init__(self, options: AgentOptions | None = None) -> None:
        opts = options or AgentOptions()
        initial_state = opts.initial_state or {}
        self._state = _MutableAgentState(
            system_prompt=initial_state.get("system_prompt", ""),
            model=initial_state.get("model"),
            thinking_level=initial_state.get("thinking_level", "off"),
            tools=initial_state.get("tools"),
            messages=initial_state.get("messages"),
        )

        self.convert_to_llm: ConvertToLlmFn = opts.convert_to_llm or _default_convert_to_llm
        self.transform_context: TransformContextFn | None = opts.transform_context
        self.stream_fn: StreamFn | None = opts.stream_fn
        self.get_api_key: GetApiKeyFn | None = opts.get_api_key
        self.before_tool_call: BeforeToolCallHook | None = opts.before_tool_call
        self.after_tool_call: AfterToolCallHook | None = opts.after_tool_call
        self.session_id: str | None = opts.session_id
        self.thinking_budgets: ThinkingBudgets | None = opts.thinking_budgets
        self.transport: Transport | None = opts.transport
        self.max_retry_delay_ms: int | None = opts.max_retry_delay_ms
        self.tool_execution: ToolExecutionMode = opts.tool_execution

        self._steering_queue = _PendingMessageQueue(opts.steering_mode)
        self._follow_up_queue = _PendingMessageQueue(opts.follow_up_mode)
        self._listeners: list[AgentListener] = []
        self._active_run: _ActiveRun | None = None

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> _MutableAgentState:
        """Live state object — assigning ``tools``/``messages`` copies the array."""
        return self._state

    # ------------------------------------------------------------------
    # Subscribe / event listeners
    # ------------------------------------------------------------------

    def subscribe(self, listener: AgentListener) -> Callable[[], None]:
        """Register a listener; returns an unsubscribe callable."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    # ------------------------------------------------------------------
    # Queue mode
    # ------------------------------------------------------------------

    @property
    def steering_mode(self) -> str:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: str) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> str:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: str) -> None:
        self._follow_up_queue.mode = mode

    # ------------------------------------------------------------------
    # Queue mutators
    # ------------------------------------------------------------------

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to inject after the current assistant turn finishes."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    # ------------------------------------------------------------------
    # Abort / wait
    # ------------------------------------------------------------------

    @property
    def signal(self) -> asyncio.Event | None:
        """The active abort event for the current run, if any."""
        return self._active_run.abort_event if self._active_run is not None else None

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        if self._active_run is not None:
            self._active_run.abort_event.set()

    async def wait_for_idle(self) -> None:
        """Resolve when the current run has fully settled."""
        if self._active_run is not None:
            await self._active_run.future

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear transcript, runtime state, and queued messages."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = frozenset()
        self._state.error_message = None
        self.clear_all_queues()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    async def prompt(
        self,
        input_: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new prompt from text, a message, or a batch of messages."""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, "
                "or wait for completion.",
            )
        messages = self._normalize_prompt_input(input_, images)
        await self._run_prompt_messages(messages)

    async def continue_run(self) -> None:
        """Continue from the current transcript.

        The last message must be a user or tool-result message **unless** a
        steering/follow-up message has been queued, in which case those are
        promoted to the next prompt automatically.
        """
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")
        if not self._state.messages:
            raise RuntimeError("No messages to continue from")

        last_message = self._state.messages[-1]
        if getattr(last_message, "role", None) == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt_messages(queued_steering, skip_initial_steering_poll=True)
                return
            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                await self._run_prompt_messages(queued_follow_ups)
                return
            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_prompt_input(
        self,
        input_: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(input_, list):
            return input_
        if not isinstance(input_, str):
            return [input_]

        content: list[Any] = [TextContent(text=input_)]
        if images:
            content.extend(images)
        return [UserMessage(content=content, timestamp=int(time.time() * 1000))]

    async def _run_prompt_messages(
        self,
        messages: list[AgentMessage],
        *,
        skip_initial_steering_poll: bool = False,
    ) -> None:
        async def executor(signal: asyncio.Event) -> None:
            await run_agent_loop(
                prompts=messages,
                context=self._create_context_snapshot(),
                config=self._create_loop_config(skip_initial_steering_poll=skip_initial_steering_poll),
                emit=self._process_event,
                signal=signal,
                stream_fn=self.stream_fn,
            )

        await self._run_with_lifecycle(executor)

    async def _run_continuation(self) -> None:
        async def executor(signal: asyncio.Event) -> None:
            await run_agent_loop_continue(
                context=self._create_context_snapshot(),
                config=self._create_loop_config(),
                emit=self._process_event,
                signal=signal,
                stream_fn=self.stream_fn,
            )

        await self._run_with_lifecycle(executor)

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _create_loop_config(self, *, skip_initial_steering_poll: bool = False) -> AgentLoopConfig:
        skip_remaining = skip_initial_steering_poll

        async def get_steering_messages() -> list[AgentMessage]:
            nonlocal skip_remaining
            if skip_remaining:
                skip_remaining = False
                return []
            return self._steering_queue.drain()

        async def get_follow_up_messages() -> list[AgentMessage]:
            return self._follow_up_queue.drain()

        return AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self.convert_to_llm,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self.session_id,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        )

    async def _run_with_lifecycle(
        self,
        executor: Callable[[asyncio.Event], Awaitable[None]],
    ) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        abort_event = asyncio.Event()
        self._active_run = _ActiveRun(future=future, abort_event=abort_event)

        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            try:
                await executor(abort_event)
            except Exception as exc:
                await self._handle_run_failure(exc, aborted=abort_event.is_set())
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: Exception, *, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            usage=_EMPTY_USAGE.model_copy(deep=True),
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=int(time.time() * 1000),
        )
        self._state.messages.append(failure_message)
        self._state.error_message = failure_message.error_message
        await self._process_event(
            {"type": "agent_end", "messages": [failure_message]},
        )

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = frozenset()
        if self._active_run is not None and not self._active_run.future.done():
            self._active_run.future.set_result(None)
        self._active_run = None

    async def _process_event(self, event: AgentEvent) -> None:
        """Reduce internal state for ``event``, then dispatch listeners.

        ``event`` is a discriminated :class:`typing.TypedDict` union keyed by
        ``"type"``. Pyright narrows on inline ``event["type"] == ...`` checks,
        but not on a local-variable copy — so the dispatch is written as a
        plain if-chain that keeps the narrowing context.
        """
        # Two near-identical branches, kept separate so pyright narrows the
        # discriminated TypedDict correctly on each ``event["type"]`` check.
        if event["type"] == "message_start":  # noqa: SIM114
            self._state.streaming_message = event["message"]
        elif event["type"] == "message_update":
            self._state.streaming_message = event["message"]
        elif event["type"] == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event["message"])
        elif event["type"] == "tool_execution_start":
            new_pending = set(self._state.pending_tool_calls)
            new_pending.add(event["tool_call_id"])
            self._state.pending_tool_calls = frozenset(new_pending)
        elif event["type"] == "tool_execution_end":
            new_pending = set(self._state.pending_tool_calls)
            new_pending.discard(event["tool_call_id"])
            self._state.pending_tool_calls = frozenset(new_pending)
        elif event["type"] == "turn_end":
            msg = event["message"]
            error_message = getattr(msg, "error_message", None)
            if error_message:
                self._state.error_message = error_message
        elif event["type"] == "agent_end":
            self._state.streaming_message = None

        signal = self._active_run.abort_event if self._active_run is not None else None
        if signal is None:
            raise RuntimeError("Agent listener invoked outside active run")
        for listener in list(self._listeners):
            result = listener(event, signal)
            if inspect.isawaitable(result):
                await result


__all__ = ["Agent", "AgentListener", "AgentOptions"]
