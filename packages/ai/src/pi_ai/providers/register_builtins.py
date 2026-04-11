"""Register pi_ai's built-in API providers.

Port of ``packages/ai/src/providers/register-builtins.ts``. The upstream
module lazy-loads every built-in provider to avoid pulling in large SDKs
at startup. The Python port currently registers:

* ``anthropic-messages`` — Anthropic Messages API.
* ``openai-completions`` — OpenAI Chat Completions API. Also covers Ollama,
  Groq, Cerebras, xAI, OpenRouter, LMStudio, and any other provider that
  speaks the OpenAI Chat Completions wire format. The provider auto-detects
  per-vendor compat differences from ``model.provider`` / ``model.base_url``.
* ``google-generative-ai`` — Google Gemini.

This module is imported for its side-effect by :mod:`pi_ai.stream`.
"""

from __future__ import annotations

from pi_ai.api_registry import ApiProvider, register_api_provider
from pi_ai.providers.anthropic import stream_anthropic
from pi_ai.providers.google import stream_google, stream_simple_google
from pi_ai.providers.openai_completions import (
    stream_openai_completions,
    stream_simple_openai_completions,
)

_BUILTIN_SOURCE_ID = "pi-ai/builtins"


def _register_anthropic() -> None:
    # ``stream_simple_anthropic`` not yet ported; alias ``stream`` for both so
    # callers using ``stream_simple`` get the same behaviour as ``stream``.
    # The reasoning-aware path lives in the TS version's
    # ``streamSimpleAnthropic``; it'll land in a follow-up slice.
    register_api_provider(
        ApiProvider(
            api="anthropic-messages",
            stream=stream_anthropic,  # type: ignore[arg-type]
            stream_simple=stream_anthropic,  # type: ignore[arg-type]
        ),
        source_id=_BUILTIN_SOURCE_ID,
    )


def _register_openai_completions() -> None:
    register_api_provider(
        ApiProvider(
            api="openai-completions",
            stream=stream_openai_completions,  # type: ignore[arg-type]
            stream_simple=stream_simple_openai_completions,  # type: ignore[arg-type]
        ),
        source_id=_BUILTIN_SOURCE_ID,
    )


def _register_google() -> None:
    register_api_provider(
        ApiProvider(
            api="google-generative-ai",
            stream=stream_google,  # type: ignore[arg-type]
            stream_simple=stream_simple_google,  # type: ignore[arg-type]
        ),
        source_id=_BUILTIN_SOURCE_ID,
    )


def register_builtin_providers() -> None:
    """Install every built-in provider into the global API registry.

    Safe to call multiple times — :func:`pi_ai.api_registry.register_api_provider`
    replaces any existing entry for the same ``api`` identifier, so this is
    a true idempotent re-register rather than a one-shot guard.
    """
    _register_anthropic()
    _register_openai_completions()
    _register_google()


# Side-effect: register on import, same as the TS
# ``import "./providers/register-builtins.js";`` line in ``stream.ts``.
register_builtin_providers()


__all__ = ["register_builtin_providers"]
