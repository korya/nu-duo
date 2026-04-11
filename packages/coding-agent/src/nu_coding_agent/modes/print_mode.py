"""Print mode (single-shot) — port of ``packages/coding-agent/src/modes/print-mode.ts``.

Sends a sequence of prompts to the agent and prints the result. Two
output modes:

* ``"text"`` — only the final assistant text content
* ``"json"`` — every :class:`AgentEvent` as a JSON line on stdout

The upstream version goes through ``AgentSessionRuntime`` which exposes
the full extension lifecycle (``newSession``, ``fork``, ``navigateTree``,
``switchSession``, ``reload``). The Python port doesn't have those yet
(they live in :mod:`nu_coding_agent.core.agent_session`, still pending),
so this module operates directly on a :class:`nu_agent_core.agent.Agent`
instance plus a callable that pumps the prompts. When the runtime port
lands, swap the ``send_prompts`` callback for a runtime adapter without
changing the rest of the surface.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nu_ai.types import AssistantMessage, ImageContent, TextContent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nu_agent_core.agent import Agent
    from nu_agent_core.types import AgentEvent


type PrintMode = Literal["text", "json"]


@dataclass(slots=True)
class PrintModeOptions:
    """Knobs for :func:`run_print_mode`."""

    mode: PrintMode
    initial_message: str | None = None
    initial_images: list[ImageContent] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


type SendPrompt = Callable[[str, list[ImageContent] | None], Awaitable[None]]


def _write(line: str) -> None:
    sys.stdout.write(line)
    sys.stdout.flush()


def _format_event_for_json(event: Any) -> str:
    """Render an :class:`AgentEvent` as a single JSON line.

    The agent event payloads contain Pydantic models; we coerce them
    via ``model_dump`` so the line stays JSON-serializable.
    """

    def _coerce(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(by_alias=True)
        if isinstance(value, dict):
            return {k: _coerce(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_coerce(item) for item in value]
        return value

    return json.dumps(_coerce(event), default=str)


def _print_text_result(agent: Agent) -> int:
    """Render the final assistant message in ``text`` mode and return an exit code."""
    state = agent.state
    if not state.messages:
        return 0
    last = state.messages[-1]
    if not isinstance(last, AssistantMessage):
        return 0
    if last.stop_reason in ("error", "aborted"):
        message = last.error_message or f"Request {last.stop_reason}"
        sys.stderr.write(message + "\n")
        return 1
    for content in last.content:
        if isinstance(content, TextContent):
            _write(content.text + "\n")
    return 0


async def run_print_mode(
    agent: Agent,
    send_prompt: SendPrompt,
    options: PrintModeOptions,
) -> int:
    """Drive ``agent`` through the supplied prompts and print the result.

    ``send_prompt(text, images)`` is the indirection that lets callers
    plug a session manager (or the eventual ``AgentSessionRuntime``)
    between this function and the agent. It must:

    * append the user message,
    * call ``agent.prompt(...)``,
    * await the turn to completion (so we can read the final state).
    """
    exit_code = 0
    unsubscribe = None
    try:
        if options.mode == "json":

            async def emit(event: AgentEvent, _signal: Any) -> None:
                _write(_format_event_for_json(event) + "\n")

            unsubscribe = agent.subscribe(emit)

        if options.initial_message is not None:
            await send_prompt(options.initial_message, options.initial_images or None)

        for message in options.messages:
            await send_prompt(message, None)

        if options.mode == "text":
            exit_code = _print_text_result(agent)

        return exit_code
    except Exception as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    finally:
        if unsubscribe is not None:
            unsubscribe()
        sys.stdout.flush()


__all__ = [
    "PrintMode",
    "PrintModeOptions",
    "SendPrompt",
    "run_print_mode",
]
