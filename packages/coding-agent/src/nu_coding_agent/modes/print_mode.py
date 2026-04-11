"""Print mode (single-shot) — port of ``packages/coding-agent/src/modes/print-mode.ts``.

Sends a sequence of prompts to an :class:`AgentSession` and prints the
result. Two output modes:

* ``"text"`` — only the final assistant text content
* ``"json"`` — every :class:`AgentEvent` as a JSON line on stdout

The upstream version goes through ``AgentSessionRuntime``, which adds
``newSession`` / ``fork`` / ``navigateTree`` / ``switchSession`` /
``reload`` plumbing the Python port hasn't reached yet. The simplified
:class:`nu_coding_agent.core.agent_session.AgentSession` is sufficient
for the print-mode contract — it owns session persistence and
credential validation, and it's stable enough to swap underneath
``run_print_mode`` once the runtime layer lands.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nu_ai.types import AssistantMessage, ImageContent, TextContent

if TYPE_CHECKING:
    from nu_agent_core.types import AgentEvent

    from nu_coding_agent.core.agent_session import AgentSession


type PrintMode = Literal["text", "json"]


@dataclass(slots=True)
class PrintModeOptions:
    """Knobs for :func:`run_print_mode`."""

    mode: PrintMode
    initial_message: str | None = None
    initial_images: list[ImageContent] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


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


def _print_text_result(session: AgentSession) -> int:
    """Render the final assistant message in ``text`` mode and return an exit code."""
    state = session.agent.state
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
    session: AgentSession,
    options: PrintModeOptions,
) -> int:
    """Drive ``session`` through the supplied prompts and print the result.

    ``session`` is an :class:`AgentSession` — it already wires up
    credential validation, session-file persistence, and event
    forwarding to subscribers. We just need to push prompts and
    render the output.
    """
    exit_code = 0
    unsubscribe = None
    try:
        if options.mode == "json":

            def emit(event: AgentEvent) -> None:
                _write(_format_event_for_json(event) + "\n")

            unsubscribe = session.subscribe(emit)

        if options.initial_message is not None:
            await session.prompt(
                options.initial_message,
                images=options.initial_images or None,
            )

        for message in options.messages:
            await session.prompt(message)

        if options.mode == "text":
            exit_code = _print_text_result(session)

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
    "run_print_mode",
]
