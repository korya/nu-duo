"""Agent loop that drives an LLM through tool-execution turns.

Direct port of ``packages/agent/src/agent-loop.ts``. Operates entirely on
:data:`nu_agent_core.types.AgentMessage` and only crosses into nu_ai's
LLM-compatible :class:`nu_ai.types.Message` shape at the boundary via
``config.convert_to_llm``.

Public surface:

* :func:`run_agent_loop` / :func:`run_agent_loop_continue` — imperative
  drivers; consumers pass an ``emit`` callback that receives every
  :data:`AgentEvent`.
* :func:`agent_loop` / :func:`agent_loop_continue` — wrap the imperative
  drivers in an :class:`nu_ai.utils.event_stream.EventStream` for
  ``async for`` iteration.

Python adaptations:

* TypeScript ``AbortSignal`` → :class:`asyncio.Event`. Hooks check
  ``signal.is_set()`` instead of ``signal.aborted``.
* TypeScript ``Promise<...> | ...`` return-type-or-direct unions → ``await``
  with :func:`inspect.isawaitable` so callers can return either a value
  directly or an awaitable.
* The TypeScript ``streamFn`` parameter defaults to ``streamSimple``;
  Python defaults to :func:`nu_ai.stream_simple` (resolved lazily so a
  test can override the default by passing ``stream_fn=...`` directly).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Any

from nu_ai.types import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from nu_ai.utils.event_stream import EventStream
from nu_ai.utils.validation import validate_tool_arguments

from nu_agent_core.types import (
    AfterToolCallContext,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nu_ai.utils.event_stream import AssistantMessageEventStream

    from nu_agent_core.types import StreamFn


type AgentEventSink = Callable[[AgentEvent], "Awaitable[None] | None"]


# Hold strong references to in-flight background tasks so the event loop
# doesn't GC them while they're still running.
_background_tasks: set[asyncio.Task[Any]] = set()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Start an agent loop with a new prompt list.

    Returns an :class:`EventStream` whose ``result()`` resolves to the new
    messages produced during this run (the prompts plus everything the
    loop appended).
    """
    stream = _create_agent_stream()

    async def driver() -> None:
        async def emit(event: AgentEvent) -> None:
            stream.push(event)

        messages = await run_agent_loop(
            prompts=prompts,
            context=context,
            config=config,
            emit=emit,
            signal=signal,
            stream_fn=stream_fn,
        )
        stream.end(messages)

    task = asyncio.create_task(driver())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Continue an agent loop from the current context without a new prompt.

    Used for retries after the user has appended a tool result or follow-up
    user message to the context. Raises :class:`ValueError` if the context
    is empty or if the last message is an assistant message.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    last = context.messages[-1]
    if _role_of(last) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def driver() -> None:
        async def emit(event: AgentEvent) -> None:
            stream.push(event)

        messages = await run_agent_loop_continue(
            context=context,
            config=config,
            emit=emit,
            signal=signal,
            stream_fn=stream_fn,
        )
        stream.end(messages)

    task = asyncio.create_task(driver())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return stream


# ---------------------------------------------------------------------------
# Imperative drivers
# ---------------------------------------------------------------------------


async def run_agent_loop(
    *,
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Drive an agent loop with new prompts. Returns ``prompts + appended``."""
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})
    for prompt in prompts:
        await _emit(emit, {"type": "message_start", "message": prompt})
        await _emit(emit, {"type": "message_end", "message": prompt})

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        config=config,
        signal=signal,
        emit=emit,
        stream_fn=stream_fn,
    )
    return new_messages


async def run_agent_loop_continue(
    *,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Continue from the current context. Same validation as :func:`agent_loop_continue`."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if _role_of(context.messages[-1]) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        config=config,
        signal=signal,
        emit=emit,
        stream_fn=stream_fn,
    )
    return new_messages


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------


async def _run_loop(
    *,
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
    stream_fn: StreamFn | None,
) -> None:
    first_turn = True
    pending_messages: list[AgentMessage] = await _maybe_call_get_messages(config.get_steering_messages)

    while True:  # outer loop: follow-up message support
        has_more_tool_calls = True
        while has_more_tool_calls or pending_messages:
            if first_turn:
                first_turn = False
            else:
                await _emit(emit, {"type": "turn_start"})

            if pending_messages:
                for message in pending_messages:
                    await _emit(emit, {"type": "message_start", "message": message})
                    await _emit(emit, {"type": "message_end", "message": message})
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            assistant_message = await _stream_assistant_response(
                context=current_context,
                config=config,
                signal=signal,
                emit=emit,
                stream_fn=stream_fn,
            )
            new_messages.append(assistant_message)

            if assistant_message.stop_reason in {"error", "aborted"}:
                await _emit(
                    emit,
                    {
                        "type": "turn_end",
                        "message": assistant_message,
                        "tool_results": [],
                    },
                )
                await _emit(emit, {"type": "agent_end", "messages": new_messages})
                return

            tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]
            has_more_tool_calls = bool(tool_calls)

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_results.extend(
                    await _execute_tool_calls(
                        current_context=current_context,
                        assistant_message=assistant_message,
                        config=config,
                        signal=signal,
                        emit=emit,
                    )
                )
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(
                emit,
                {
                    "type": "turn_end",
                    "message": assistant_message,
                    "tool_results": tool_results,
                },
            )

            pending_messages = await _maybe_call_get_messages(config.get_steering_messages)

        # Agent would stop. Check follow-up messages.
        follow_up_messages = await _maybe_call_get_messages(config.get_follow_up_messages)
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue
        break

    await _emit(emit, {"type": "agent_end", "messages": new_messages})


# ---------------------------------------------------------------------------
# Stream the assistant response
# ---------------------------------------------------------------------------


async def _stream_assistant_response(
    *,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    """Run one assistant turn through the LLM and emit message events."""
    messages = context.messages
    if config.transform_context is not None:
        transformed = await config.transform_context(messages, signal)
        messages = transformed

    converted = config.convert_to_llm(messages)
    if inspect.isawaitable(converted):
        llm_messages = await converted
    else:
        llm_messages = converted

    llm_context = Context(
        system_prompt=context.system_prompt or None,
        messages=llm_messages,
        tools=_to_pi_ai_tools(context.tools),
    )

    resolved_api_key: str | None = None
    if config.get_api_key is not None:
        maybe = config.get_api_key(config.model.provider)
        resolved_api_key = await maybe if inspect.isawaitable(maybe) else maybe  # type: ignore[assignment]
    if resolved_api_key is None:
        resolved_api_key = config.api_key

    # Map pi_agent's ``"off"`` reasoning level to nu_ai's ``None`` (no
    # reasoning). pi_agent has the extra ``"off"`` literal so apps can
    # distinguish "explicitly disabled" from "not configured"; nu_ai
    # collapses both into ``None``.
    pi_ai_reasoning = (
        None if config.reasoning is None or config.reasoning == "off" else config.reasoning  # type: ignore[assignment]
    )

    options = SimpleStreamOptions(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=resolved_api_key,
        cache_retention=config.cache_retention,
        session_id=config.session_id,
        headers=config.headers,
        max_retry_delay_ms=config.max_retry_delay_ms,
        metadata=config.metadata,
        reasoning=pi_ai_reasoning,
        thinking_budgets=config.thinking_budgets,
    )

    response = _resolve_stream_fn(stream_fn)(config.model, llm_context, options)
    response_stream: AssistantMessageEventStream = (
        await response if inspect.isawaitable(response) else response  # type: ignore[assignment]
    )

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response_stream:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await _emit(
                emit,
                {
                    "type": "message_start",
                    "message": partial_message.model_copy(deep=True),
                },
            )
        elif event.type in {
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        }:
            if partial_message is not None:
                partial_message = event.partial  # type: ignore[union-attr]
                context.messages[-1] = partial_message
                await _emit(
                    emit,
                    {
                        "type": "message_update",
                        "message": partial_message.model_copy(deep=True),
                        "assistant_message_event": event,
                    },
                )
        elif event.type in {"done", "error"}:
            final_message = await response_stream.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await _emit(
                    emit,
                    {
                        "type": "message_start",
                        "message": final_message.model_copy(deep=True),
                    },
                )
            await _emit(emit, {"type": "message_end", "message": final_message})
            return final_message

    # Stream ended without an explicit done/error event — fall back to result().
    final_message = await response_stream.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(
            emit,
            {"type": "message_start", "message": final_message.model_copy(deep=True)},
        )
    await _emit(emit, {"type": "message_end", "message": final_message})
    return final_message


def _resolve_stream_fn(stream_fn: StreamFn | None) -> StreamFn:
    """Return ``stream_fn`` if provided, else :func:`nu_ai.stream_simple`.

    Resolved lazily so importing nu_agent_core doesn't pull nu_ai.stream
    (and its provider registrations) until the first call.
    """
    if stream_fn is not None:
        return stream_fn
    from nu_ai.stream import stream_simple  # noqa: PLC0415 — lazy import

    return stream_simple


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _execute_tool_calls(
    *,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]
    if config.tool_execution == "sequential":
        return await _execute_tool_calls_sequential(
            current_context=current_context,
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            config=config,
            signal=signal,
            emit=emit,
        )
    return await _execute_tool_calls_parallel(
        current_context=current_context,
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        config=config,
        signal=signal,
        emit=emit,
    )


async def _execute_tool_calls_sequential(
    *,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            {
                "type": "tool_execution_start",
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "args": tool_call.arguments,
            },
        )
        prep = await _prepare_tool_call(
            current_context=current_context,
            assistant_message=assistant_message,
            tool_call=tool_call,
            config=config,
            signal=signal,
        )
        if isinstance(prep, _ImmediateToolCallOutcome):
            results.append(await _emit_tool_call_outcome(tool_call, prep.result, prep.is_error, emit))
            continue
        executed = await _execute_prepared_tool_call(prep, signal, emit)
        results.append(
            await _finalize_executed_tool_call(
                current_context=current_context,
                assistant_message=assistant_message,
                prepared=prep,
                executed=executed,
                config=config,
                signal=signal,
                emit=emit,
            )
        )
    return results


async def _execute_tool_calls_parallel(
    *,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []
    prepared: list[_PreparedToolCall] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            {
                "type": "tool_execution_start",
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "args": tool_call.arguments,
            },
        )
        prep = await _prepare_tool_call(
            current_context=current_context,
            assistant_message=assistant_message,
            tool_call=tool_call,
            config=config,
            signal=signal,
        )
        if isinstance(prep, _ImmediateToolCallOutcome):
            results.append(await _emit_tool_call_outcome(tool_call, prep.result, prep.is_error, emit))
        else:
            prepared.append(prep)

    running = [(p, asyncio.create_task(_execute_prepared_tool_call(p, signal, emit))) for p in prepared]

    for prep_item, task in running:
        executed = await task
        results.append(
            await _finalize_executed_tool_call(
                current_context=current_context,
                assistant_message=assistant_message,
                prepared=prep_item,
                executed=executed,
                config=config,
                signal=signal,
                emit=emit,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Tool call lifecycle helpers
# ---------------------------------------------------------------------------


from dataclasses import dataclass  # noqa: E402 — keep close to its consumers


@dataclass(slots=True)
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool[Any, Any]
    args: Any


@dataclass(slots=True)
class _ImmediateToolCallOutcome:
    result: AgentToolResult[Any]
    is_error: bool


@dataclass(slots=True)
class _ExecutedToolCallOutcome:
    result: AgentToolResult[Any]
    is_error: bool


def _prepare_tool_call_arguments(tool: AgentTool[Any, Any], tool_call: ToolCall) -> ToolCall:
    if tool.prepare_arguments is None:
        return tool_call
    prepared = tool.prepare_arguments(tool_call.arguments)
    if prepared is tool_call.arguments:
        return tool_call
    return tool_call.model_copy(update={"arguments": dict(prepared) if isinstance(prepared, dict) else prepared})


async def _prepare_tool_call(
    *,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> _PreparedToolCall | _ImmediateToolCallOutcome:
    tool = next(
        (t for t in (current_context.tools or []) if t.name == tool_call.name),
        None,
    )
    if tool is None:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        prepared_tool_call = _prepare_tool_call_arguments(tool, tool_call)
        # validate_tool_arguments expects a nu_ai Tool — wrap our AgentTool.
        from nu_ai.types import Tool as _PiAiTool  # noqa: PLC0415

        validated_args = validate_tool_arguments(
            _PiAiTool(name=tool.name, description=tool.description, parameters=tool.parameters),
            prepared_tool_call,
        )

        if config.before_tool_call is not None:
            before_result = await config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=validated_args,
                    context=current_context,
                ),
                signal,
            )
            if before_result is not None and before_result.block:
                return _ImmediateToolCallOutcome(
                    result=_create_error_tool_result(
                        before_result.reason or "Tool execution was blocked",
                    ),
                    is_error=True,
                )

        return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as exc:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(str(exc)),
            is_error=True,
        )


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallOutcome:
    update_emits: list[Any] = []

    def on_update(partial_result: AgentToolResult[Any]) -> None:
        update_emits.append(
            asyncio.ensure_future(
                _emit(
                    emit,
                    {
                        "type": "tool_execution_update",
                        "tool_call_id": prepared.tool_call.id,
                        "tool_name": prepared.tool_call.name,
                        "args": prepared.tool_call.arguments,
                        "partial_result": partial_result,
                    },
                ),
            ),
        )

    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            signal,
            on_update,
        )
        if update_emits:
            await asyncio.gather(*update_emits)
        return _ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as exc:
        if update_emits:
            await asyncio.gather(*update_emits)
        return _ExecutedToolCallOutcome(
            result=_create_error_tool_result(str(exc)),
            is_error=True,
        )


async def _finalize_executed_tool_call(
    *,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> ToolResultMessage:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call is not None:
        after_result = await config.after_tool_call(
            AfterToolCallContext(
                assistant_message=assistant_message,
                tool_call=prepared.tool_call,
                args=prepared.args,
                result=result,
                is_error=is_error,
                context=current_context,
            ),
            signal,
        )
        if after_result is not None:
            result = AgentToolResult(
                content=after_result.content if after_result.content is not None else result.content,
                details=after_result.details if after_result.details is not None else result.details,
            )
            if after_result.is_error is not None:
                is_error = after_result.is_error

    return await _emit_tool_call_outcome(prepared.tool_call, result, is_error, emit)


async def _emit_tool_call_outcome(
    tool_call: ToolCall,
    result: AgentToolResult[Any],
    is_error: bool,
    emit: AgentEventSink,
) -> ToolResultMessage:
    await _emit(
        emit,
        {
            "type": "tool_execution_end",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "result": result,
            "is_error": is_error,
        },
    )

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
        timestamp=int(time.time() * 1000),
    )

    await _emit(emit, {"type": "message_start", "message": tool_result_message})
    await _emit(emit, {"type": "message_end", "message": tool_result_message})
    return tool_result_message


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _create_error_tool_result(message: str) -> AgentToolResult[Any]:
    return AgentToolResult(content=[TextContent(text=message)], details={})


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    def is_complete(event: AgentEvent) -> bool:
        return event["type"] == "agent_end"

    def extract(event: AgentEvent) -> list[AgentMessage]:
        if event["type"] == "agent_end":
            return event["messages"]
        return []

    return EventStream(is_complete=is_complete, extract_result=extract)


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result


async def _maybe_call_get_messages(
    fn: Callable[[], Awaitable[list[AgentMessage]]] | None,
) -> list[AgentMessage]:
    if fn is None:
        return []
    result = fn()
    if inspect.isawaitable(result):
        return await result
    return list(result)  # type: ignore[arg-type]


def _role_of(message: AgentMessage) -> str | None:
    return getattr(message, "role", None)


def _to_pi_ai_tools(tools: list[AgentTool[Any, Any]] | None) -> list[Any] | None:
    """Convert :class:`AgentTool` list to nu_ai :class:`Tool` list for the LLM."""
    if not tools:
        return None
    from nu_ai.types import Tool as _PiAiTool  # noqa: PLC0415

    return [_PiAiTool(name=t.name, description=t.description, parameters=t.parameters) for t in tools]


__all__ = [
    "AgentEventSink",
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
]
