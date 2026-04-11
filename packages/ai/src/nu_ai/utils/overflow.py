"""Context-overflow detection.

Port of ``packages/ai/src/utils/overflow.ts``. Matches provider error
messages against a battery of known patterns and, where ``context_window``
is supplied, detects silent overflow (non-erroring successful responses
whose input token count exceeds the model's context window).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nu_ai.types import AssistantMessage

# Patterns that identify context-overflow errors from various providers.
# Kept in sync with upstream ``overflow.ts``.
_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"request_too_large", re.IGNORECASE),
    re.compile(r"input is too long for requested model", re.IGNORECASE),
    re.compile(r"exceeds the context window", re.IGNORECASE),
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),
    re.compile(r"reduce the length of the messages", re.IGNORECASE),
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),
    re.compile(r"exceeds the available context size", re.IGNORECASE),
    re.compile(r"greater than the context length", re.IGNORECASE),
    re.compile(r"context window exceeds limit", re.IGNORECASE),
    re.compile(r"exceeded model token limit", re.IGNORECASE),
    re.compile(r"too large for model with \d+ maximum context length", re.IGNORECASE),
    re.compile(r"model_context_window_exceeded", re.IGNORECASE),
    re.compile(r"prompt too long; exceeded (?:max )?context length", re.IGNORECASE),
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"token limit exceeded", re.IGNORECASE),
    re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE),
]

# Patterns that indicate a *non-overflow* error even though they'd otherwise
# match one of the overflow patterns (rate limiting, throttling, etc.).
_NON_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
]


def is_context_overflow(message: AssistantMessage, context_window: int | None = None) -> bool:
    """Return ``True`` iff ``message`` indicates a context-overflow condition.

    Handles two cases:

    1. **Error-based**: ``stop_reason == "error"`` and ``error_message``
       matches one of ``_OVERFLOW_PATTERNS`` but none of
       ``_NON_OVERFLOW_PATTERNS``.
    2. **Silent overflow**: when ``context_window`` is supplied, a
       successful response whose ``usage.input + usage.cache_read``
       exceeds the supplied context window (z.ai-style).
    """
    if message.stop_reason == "error" and message.error_message:
        msg = message.error_message
        if not any(p.search(msg) for p in _NON_OVERFLOW_PATTERNS) and any(p.search(msg) for p in _OVERFLOW_PATTERNS):
            return True

    if context_window and message.stop_reason == "stop":
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens > context_window:
            return True

    return False


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """Return a copy of the overflow patterns (for testing)."""
    return list(_OVERFLOW_PATTERNS)


__all__ = ["get_overflow_patterns", "is_context_overflow"]
