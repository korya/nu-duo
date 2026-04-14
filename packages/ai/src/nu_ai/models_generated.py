"""Auto-generated model catalog (snapshot).

Contains a hardcoded snapshot of popular models from each provider, plus logic
to load the full catalog from ``resources/models_generated.json`` when available.

Regenerate the JSON file via::

    python scripts/generate_models.py

Do not edit the snapshot below by hand -- it exists so that nu_ai works out of
the box even before the full catalog JSON is generated.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Final

from nu_ai.types import Model

# ---------------------------------------------------------------------------
# Hardcoded snapshot of popular models (camelCase wire format).
# ---------------------------------------------------------------------------

_SNAPSHOT: list[dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # Anthropic
    # -----------------------------------------------------------------------
    {
        "id": "claude-sonnet-4-20250514",
        "name": "Claude Sonnet 4",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.anthropic.com",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75},
        "contextWindow": 200000,
        "maxTokens": 64000,
    },
    {
        "id": "claude-haiku-4-5-20251001",
        "name": "Claude Haiku 4.5",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.anthropic.com",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 0.8, "output": 4, "cacheRead": 0.08, "cacheWrite": 1},
        "contextWindow": 200000,
        "maxTokens": 8192,
    },
    {
        "id": "claude-opus-4-20250514",
        "name": "Claude Opus 4",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.anthropic.com",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 15, "output": 75, "cacheRead": 1.5, "cacheWrite": 18.75},
        "contextWindow": 200000,
        "maxTokens": 32000,
    },
    # -----------------------------------------------------------------------
    # OpenAI
    # -----------------------------------------------------------------------
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "api": "openai-responses",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 2.5, "output": 10, "cacheRead": 1.25, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 16384,
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "api": "openai-responses",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 0.15, "output": 0.6, "cacheRead": 0.075, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 16384,
    },
    {
        "id": "gpt-4.1",
        "name": "GPT-4.1",
        "api": "openai-responses",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 2, "output": 8, "cacheRead": 0.5, "cacheWrite": 0},
        "contextWindow": 1047576,
        "maxTokens": 32768,
    },
    {
        "id": "gpt-4.1-mini",
        "name": "GPT-4.1 Mini",
        "api": "openai-responses",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 0.4, "output": 1.6, "cacheRead": 0.1, "cacheWrite": 0},
        "contextWindow": 1047576,
        "maxTokens": 32768,
    },
    {
        "id": "gpt-4.1-nano",
        "name": "GPT-4.1 Nano",
        "api": "openai-responses",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 0.1, "output": 0.4, "cacheRead": 0.025, "cacheWrite": 0},
        "contextWindow": 1047576,
        "maxTokens": 32768,
    },
    {
        "id": "o3",
        "name": "o3",
        "api": "openai-responses",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 2, "output": 8, "cacheRead": 0.5, "cacheWrite": 0},
        "contextWindow": 200000,
        "maxTokens": 100000,
    },
    {
        "id": "o4-mini",
        "name": "o4-mini",
        "api": "openai-responses",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 1.1, "output": 4.4, "cacheRead": 0.275, "cacheWrite": 0},
        "contextWindow": 200000,
        "maxTokens": 100000,
    },
    # -----------------------------------------------------------------------
    # Google
    # -----------------------------------------------------------------------
    {
        "id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "api": "google-generative-ai",
        "provider": "google",
        "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 1.25, "output": 10, "cacheRead": 0.3125, "cacheWrite": 0},
        "contextWindow": 1048576,
        "maxTokens": 65536,
    },
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "api": "google-generative-ai",
        "provider": "google",
        "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 0.15, "output": 0.6, "cacheRead": 0.0375, "cacheWrite": 0},
        "contextWindow": 1048576,
        "maxTokens": 65536,
    },
    {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "api": "google-generative-ai",
        "provider": "google",
        "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 0.1, "output": 0.4, "cacheRead": 0.025, "cacheWrite": 0},
        "contextWindow": 1048576,
        "maxTokens": 8192,
    },
    # -----------------------------------------------------------------------
    # Mistral
    # -----------------------------------------------------------------------
    {
        "id": "mistral-large-latest",
        "name": "Mistral Large",
        "api": "mistral-conversations",
        "provider": "mistral",
        "baseUrl": "https://api.mistral.ai",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 2, "output": 6, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 8192,
    },
    {
        "id": "codestral-latest",
        "name": "Codestral",
        "api": "mistral-conversations",
        "provider": "mistral",
        "baseUrl": "https://api.mistral.ai",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0.3, "output": 0.9, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 256000,
        "maxTokens": 8192,
    },
    # -----------------------------------------------------------------------
    # DeepSeek
    # -----------------------------------------------------------------------
    {
        "id": "deepseek-chat",
        "name": "DeepSeek Chat",
        "api": "openai-completions",
        "provider": "openai",
        "baseUrl": "https://api.deepseek.com/v1",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0.14, "output": 0.28, "cacheRead": 0.014, "cacheWrite": 0},
        "contextWindow": 64000,
        "maxTokens": 8192,
    },
    {
        "id": "deepseek-reasoner",
        "name": "DeepSeek Reasoner",
        "api": "openai-completions",
        "provider": "openai",
        "baseUrl": "https://api.deepseek.com/v1",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0.55, "output": 2.19, "cacheRead": 0.14, "cacheWrite": 0},
        "contextWindow": 64000,
        "maxTokens": 8192,
    },
]

# ---------------------------------------------------------------------------
# Build catalog: try JSON first, fall back to snapshot.
# ---------------------------------------------------------------------------


def _load_json() -> dict[str, dict[str, Model]] | None:
    """Attempt to load the full catalog from the JSON resource file."""
    try:
        raw = json.loads(
            files("nu_ai.resources")
            .joinpath("models_generated.json")
            .read_text(encoding="utf-8")
        )
        catalog: dict[str, dict[str, Model]] = {}
        for provider, models in raw.items():
            catalog[provider] = {
                model_id: Model.model_validate(record)
                for model_id, record in models.items()
            }
        return catalog
    except Exception:
        return None


def _load_snapshot() -> dict[str, dict[str, Model]]:
    """Build catalog from the hardcoded snapshot above."""
    catalog: dict[str, dict[str, Model]] = {}
    for record in _SNAPSHOT:
        provider = record["provider"]
        model_id = record["id"]
        catalog.setdefault(provider, {})
        catalog[provider][model_id] = Model.model_validate(record)
    return catalog


def _load() -> dict[str, dict[str, Model]]:
    """Load from JSON when available, otherwise fall back to snapshot."""
    return _load_json() or _load_snapshot()


MODELS: Final[dict[str, dict[str, Model]]] = _load()


__all__ = ["MODELS"]
