"""Custom message types and ``convert_to_llm`` — direct port of ``packages/coding-agent/src/core/messages.ts``.

The coding agent extends the base :class:`nu_ai.types.Message` union with
four extra message kinds that live in the session JSONL file but never go
to a provider verbatim. :func:`convert_to_llm` is THE choke point that
flattens those kinds back into plain ``Message`` objects before each LLM
call.

The TS upstream uses declaration merging on a ``CustomAgentMessages``
interface to make ``AgentMessage`` exhaustive over the extra kinds.
Python has no equivalent, so :data:`nu_agent_core.types.AgentMessage` is
``Message | Any`` and we branch on ``role`` here at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from nu_ai.types import ImageContent, TextContent, UserMessage

if TYPE_CHECKING:
    from nu_agent_core.types import AgentMessage
    from nu_ai.types import Message


COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
BRANCH_SUMMARY_SUFFIX = "</summary>"


# ---------------------------------------------------------------------------
# Custom message dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BashExecutionMessage:
    """Output of a ``!``-prefixed bash run typed straight into the REPL prompt."""

    role: Literal["bashExecution"]
    command: str
    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    timestamp: int
    full_output_path: str | None = None
    exclude_from_context: bool = False
    """When ``True`` (the ``!!`` prefix), the message is omitted from LLM context."""


@dataclass(slots=True)
class CustomMessage:
    """Extension-injected message produced via ``sendMessage()``."""

    role: Literal["custom"]
    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool
    timestamp: int
    details: Any = None


@dataclass(slots=True)
class BranchSummaryMessage:
    """Summary inserted when the user reverts to an earlier branch of the session tree."""

    role: Literal["branchSummary"]
    summary: str
    from_id: str
    timestamp: int


@dataclass(slots=True)
class CompactionSummaryMessage:
    """Summary inserted when the session is compacted to free up context tokens."""

    role: Literal["compactionSummary"]
    summary: str
    tokens_before: int
    timestamp: int


# ---------------------------------------------------------------------------
# Constructors — keep parity with the TS factory functions.
# ---------------------------------------------------------------------------


def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    """Render a :class:`BashExecutionMessage` as the user-message text the LLM sees."""
    text = f"Ran `{msg.command}`\n"
    text += f"```\n{msg.output}\n```" if msg.output else "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated and msg.full_output_path:
        text += f"\n\n[Output truncated. Full output: {msg.full_output_path}]"
    return text


def _parse_iso_timestamp(timestamp: str) -> int:
    """Convert an ISO-8601 timestamp string to a unix-millisecond integer."""
    # ``datetime.fromisoformat`` accepts ``Z`` suffixes from Python 3.11+.
    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)


def create_branch_summary_message(summary: str, from_id: str, timestamp: str) -> BranchSummaryMessage:
    return BranchSummaryMessage(
        role="branchSummary",
        summary=summary,
        from_id=from_id,
        timestamp=_parse_iso_timestamp(timestamp),
    )


def create_compaction_summary_message(
    summary: str,
    tokens_before: int,
    timestamp: str,
) -> CompactionSummaryMessage:
    return CompactionSummaryMessage(
        role="compactionSummary",
        summary=summary,
        tokens_before=tokens_before,
        timestamp=_parse_iso_timestamp(timestamp),
    )


def create_custom_message(
    custom_type: str,
    content: str | list[TextContent | ImageContent],
    display: bool,
    details: Any,
    timestamp: str,
) -> CustomMessage:
    return CustomMessage(
        role="custom",
        custom_type=custom_type,
        content=content,
        display=display,
        details=details,
        timestamp=_parse_iso_timestamp(timestamp),
    )


# ---------------------------------------------------------------------------
# convert_to_llm
# ---------------------------------------------------------------------------


def _custom_role(msg: Any) -> str | None:
    """Return ``msg.role`` for both Pydantic models and dataclasses, else ``None``."""
    role = getattr(msg, "role", None)
    return role if isinstance(role, str) else None


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Flatten coding-agent custom messages into plain :class:`nu_ai.types.Message` objects.

    Mirrors ``packages/coding-agent/src/core/messages.ts`` ``convertToLlm``.
    Pass-through for the standard ``user`` / ``assistant`` / ``toolResult``
    roles; rewrite the four custom roles into ``user`` text messages; drop
    bash executions marked ``exclude_from_context``.
    """
    out: list[Message] = []
    for msg in messages:
        role = _custom_role(msg)
        if role == "bashExecution":
            assert isinstance(msg, BashExecutionMessage)
            if msg.exclude_from_context:
                continue
            out.append(
                UserMessage(
                    content=[TextContent(text=bash_execution_to_text(msg))],
                    timestamp=msg.timestamp,
                )
            )
        elif role == "custom":
            assert isinstance(msg, CustomMessage)
            content: list[TextContent | ImageContent] = (
                [TextContent(text=msg.content)] if isinstance(msg.content, str) else list(msg.content)
            )
            out.append(UserMessage(content=content, timestamp=msg.timestamp))
        elif role == "branchSummary":
            assert isinstance(msg, BranchSummaryMessage)
            out.append(
                UserMessage(
                    content=[
                        TextContent(text=BRANCH_SUMMARY_PREFIX + msg.summary + BRANCH_SUMMARY_SUFFIX),
                    ],
                    timestamp=msg.timestamp,
                )
            )
        elif role == "compactionSummary":
            assert isinstance(msg, CompactionSummaryMessage)
            out.append(
                UserMessage(
                    content=[
                        TextContent(text=COMPACTION_SUMMARY_PREFIX + msg.summary + COMPACTION_SUMMARY_SUFFIX),
                    ],
                    timestamp=msg.timestamp,
                )
            )
        elif role in ("user", "assistant", "toolResult"):
            out.append(msg)
        # Anything else is silently dropped, matching the TS exhaustive default.
    return out


__all__ = [
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BashExecutionMessage",
    "BranchSummaryMessage",
    "CompactionSummaryMessage",
    "CustomMessage",
    "bash_execution_to_text",
    "convert_to_llm",
    "create_branch_summary_message",
    "create_compaction_summary_message",
    "create_custom_message",
]
