"""GitHub Copilot-specific request headers.

Direct port of ``packages/ai/src/providers/github-copilot-headers.ts``.
Copilot uses two extra request headers beyond the normal auth setup:

* ``X-Initiator`` — tells Copilot whether a request is user-initiated or a
  follow-up agent turn. Inferred from the role of the last message.
* ``Copilot-Vision-Request`` — required when any user or tool-result
  message carries an image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_ai.types import ImageContent, ToolResultMessage, UserMessage

if TYPE_CHECKING:
    from pi_ai.types import Message


def infer_copilot_initiator(messages: list[Message]) -> str:
    """Return ``"user"`` when the last message is a user turn, else ``"agent"``."""
    if not messages:
        return "user"
    last = messages[-1]
    return "user" if last.role == "user" else "agent"


def has_copilot_vision_input(messages: list[Message]) -> bool:
    """Return ``True`` iff any user or tool-result message carries an image."""
    for msg in messages:
        eligible = (isinstance(msg, UserMessage) and not isinstance(msg.content, str)) or isinstance(
            msg, ToolResultMessage
        )
        if eligible and any(isinstance(c, ImageContent) for c in msg.content):
            return True
    return False


def build_copilot_dynamic_headers(*, messages: list[Message], has_images: bool) -> dict[str, str]:
    """Build the per-request Copilot headers for ``messages``."""
    headers: dict[str, str] = {
        "X-Initiator": infer_copilot_initiator(messages),
        "Openai-Intent": "conversation-edits",
    }
    if has_images:
        headers["Copilot-Vision-Request"] = "true"
    return headers


__all__ = [
    "build_copilot_dynamic_headers",
    "has_copilot_vision_input",
    "infer_copilot_initiator",
]
