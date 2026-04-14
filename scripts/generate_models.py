#!/usr/bin/env python3
"""Generate the model catalog for nu_ai.

Port of ``packages/ai/scripts/generate-models.ts`` from the upstream TS
monorepo.  Fetches model metadata from:

* models.dev (Anthropic, Google, OpenAI, Groq, Cerebras, xAI, Mistral, ...)
* OpenRouter  (openrouter.ai/api/v1/models)
* Vercel AI Gateway (ai-gateway.vercel.sh/v1/models)

Normalises every entry into a ``Model``-compatible dict and writes the result
to ``packages/ai/src/nu_ai/resources/models_generated.json``.

Usage::

    python scripts/generate_models.py           # write to models_generated.json
    python scripts/generate_models.py --dry-run  # print JSON to stdout only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "packages" / "ai"
OUTPUT_PATH = PACKAGE_ROOT / "src" / "nu_ai" / "resources" / "models_generated.json"

AI_GATEWAY_MODELS_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh"

COPILOT_STATIC_HEADERS: dict[str, str] = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

ZAI_TOOL_STREAM_UNSUPPORTED_MODELS = {"glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4.5v"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ModelDict = dict[str, Any]


def _to_number(value: str | int | float | None) -> float:
    """Safely parse a numeric value, returning 0 for anything unparseable."""
    import math  # noqa: PLC0415

    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else 0.0
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    except (ValueError, TypeError):
        return 0.0


def _make_model(
    *,
    id: str,
    name: str,
    api: str,
    provider: str,
    base_url: str,
    reasoning: bool,
    input: list[str],
    cost: dict[str, float],
    context_window: int,
    max_tokens: int,
    headers: dict[str, str] | None = None,
    compat: dict[str, Any] | None = None,
) -> ModelDict:
    """Build a model dict in the wire (camelCase) format used by models_generated.json."""
    m: ModelDict = {
        "id": id,
        "name": name,
        "api": api,
        "provider": provider,
        "baseUrl": base_url,
        "reasoning": reasoning,
        "input": input,
        "cost": cost,
        "contextWindow": context_window,
        "maxTokens": max_tokens,
    }
    if headers is not None:
        m["headers"] = headers
    if compat is not None:
        m["compat"] = compat
    return m


def _cost(
    inp: float = 0.0,
    out: float = 0.0,
    cache_read: float = 0.0,
    cache_write: float = 0.0,
) -> dict[str, float]:
    return {"input": inp, "output": out, "cacheRead": cache_read, "cacheWrite": cache_write}


def _input_modalities(modalities_input: list[str] | None) -> list[str]:
    if modalities_input and "image" in modalities_input:
        return ["text", "image"]
    return ["text"]


# ---------------------------------------------------------------------------
# Provider fetchers — mirror the TS script's data sources.
# ---------------------------------------------------------------------------


def _fetch_models_dev(client: httpx.Client) -> list[ModelDict]:
    """Fetch from https://models.dev/api.json (Anthropic, Google, OpenAI, etc.)."""
    print("Fetching models from models.dev API...")
    try:
        resp = client.get("https://models.dev/api.json", timeout=30)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except Exception as exc:
        print(f"Failed to load models.dev data: {exc}", file=sys.stderr)
        return []

    models: list[ModelDict] = []

    # --- provider configs: (key_in_json, provider, api, base_url) ---
    provider_configs: list[tuple[str, str, str, str]] = [
        ("anthropic", "anthropic", "anthropic-messages", "https://api.anthropic.com"),
        ("google", "google", "google-generative-ai", "https://generativelanguage.googleapis.com/v1beta"),
        ("openai", "openai", "openai-responses", "https://api.openai.com/v1"),
        ("groq", "groq", "openai-completions", "https://api.groq.com/openai/v1"),
        ("cerebras", "cerebras", "openai-completions", "https://api.cerebras.ai/v1"),
        ("xai", "xai", "openai-completions", "https://api.x.ai/v1"),
        ("mistral", "mistral", "mistral-conversations", "https://api.mistral.ai"),
        ("huggingface", "huggingface", "openai-completions", "https://router.huggingface.co/v1"),
    ]

    for json_key, provider, api, base_url in provider_configs:
        section = data.get(json_key, {}).get("models", {})
        for model_id, m in section.items():
            if m.get("tool_call") is not True:
                continue

            # Skip specific problematic models (same filters as TS)
            if provider == "amazon-bedrock":
                if model_id.startswith("ai21.jamba"):
                    continue
                if model_id.startswith("mistral.mistral-7b-instruct-v0"):
                    continue

            compat_val = None
            if provider == "huggingface":
                compat_val = {"supportsDeveloperRole": False}

            models.append(_make_model(
                id=model_id,
                name=m.get("name") or model_id,
                api=api,
                provider=provider,
                base_url=base_url,
                reasoning=m.get("reasoning") is True,
                input=_input_modalities(m.get("modalities", {}).get("input")),
                cost=_cost(
                    m.get("cost", {}).get("input", 0),
                    m.get("cost", {}).get("output", 0),
                    m.get("cost", {}).get("cache_read", 0),
                    m.get("cost", {}).get("cache_write", 0),
                ),
                context_window=m.get("limit", {}).get("context", 4096),
                max_tokens=m.get("limit", {}).get("output", 4096),
                compat=compat_val,
            ))

    # --- Amazon Bedrock ---
    bedrock = data.get("amazon-bedrock", {}).get("models", {})
    for model_id, m in bedrock.items():
        if m.get("tool_call") is not True:
            continue
        if model_id.startswith("ai21.jamba"):
            continue
        if model_id.startswith("mistral.mistral-7b-instruct-v0"):
            continue
        models.append(_make_model(
            id=model_id,
            name=m.get("name") or model_id,
            api="bedrock-converse-stream",
            provider="amazon-bedrock",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            reasoning=m.get("reasoning") is True,
            input=_input_modalities(m.get("modalities", {}).get("input")),
            cost=_cost(
                m.get("cost", {}).get("input", 0),
                m.get("cost", {}).get("output", 0),
                m.get("cost", {}).get("cache_read", 0),
                m.get("cost", {}).get("cache_write", 0),
            ),
            context_window=m.get("limit", {}).get("context", 4096),
            max_tokens=m.get("limit", {}).get("output", 4096),
        ))

    # --- zAi (zai-coding-plan) ---
    zai = data.get("zai-coding-plan", {}).get("models", {})
    for model_id, m in zai.items():
        if m.get("tool_call") is not True:
            continue
        compat_val: dict[str, Any] = {"supportsDeveloperRole": False, "thinkingFormat": "zai"}
        if model_id not in ZAI_TOOL_STREAM_UNSUPPORTED_MODELS:
            compat_val["zaiToolStream"] = True
        models.append(_make_model(
            id=model_id,
            name=m.get("name") or model_id,
            api="openai-completions",
            provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4",
            reasoning=m.get("reasoning") is True,
            input=_input_modalities(m.get("modalities", {}).get("input")),
            cost=_cost(
                m.get("cost", {}).get("input", 0),
                m.get("cost", {}).get("output", 0),
                m.get("cost", {}).get("cache_read", 0),
                m.get("cost", {}).get("cache_write", 0),
            ),
            context_window=m.get("limit", {}).get("context", 4096),
            max_tokens=m.get("limit", {}).get("output", 4096),
            compat=compat_val,
        ))

    # --- OpenCode (Zen / Go) ---
    opencode_variants = [
        ("opencode", "opencode", "https://opencode.ai/zen"),
        ("opencode-go", "opencode-go", "https://opencode.ai/zen/go"),
    ]
    for json_key, provider, base_path in opencode_variants:
        section = data.get(json_key, {}).get("models", {})
        for model_id, m in section.items():
            if m.get("tool_call") is not True:
                continue
            if m.get("status") == "deprecated":
                continue

            npm = (m.get("provider") or {}).get("npm")
            if npm == "@ai-sdk/openai":
                api = "openai-responses"
                base_url = f"{base_path}/v1"
            elif npm == "@ai-sdk/anthropic":
                api = "anthropic-messages"
                base_url = base_path
            elif npm == "@ai-sdk/google":
                api = "google-generative-ai"
                base_url = f"{base_path}/v1"
            else:
                api = "openai-completions"
                base_url = f"{base_path}/v1"

            models.append(_make_model(
                id=model_id,
                name=m.get("name") or model_id,
                api=api,
                provider=provider,
                base_url=base_url,
                reasoning=m.get("reasoning") is True,
                input=_input_modalities(m.get("modalities", {}).get("input")),
                cost=_cost(
                    m.get("cost", {}).get("input", 0),
                    m.get("cost", {}).get("output", 0),
                    m.get("cost", {}).get("cache_read", 0),
                    m.get("cost", {}).get("cache_write", 0),
                ),
                context_window=m.get("limit", {}).get("context", 4096),
                max_tokens=m.get("limit", {}).get("output", 4096),
            ))

    # --- GitHub Copilot ---
    copilot = data.get("github-copilot", {}).get("models", {})
    for model_id, m in copilot.items():
        if m.get("tool_call") is not True:
            continue
        if m.get("status") == "deprecated":
            continue

        is_copilot_claude4 = bool(re.match(r"^claude-(haiku|sonnet|opus)-4([.\-]|$)", model_id))
        needs_responses_api = model_id.startswith(("gpt-5", "oswe"))
        if is_copilot_claude4:
            api = "anthropic-messages"
        elif needs_responses_api:
            api = "openai-responses"
        else:
            api = "openai-completions"

        copilot_model = _make_model(
            id=model_id,
            name=m.get("name") or model_id,
            api=api,
            provider="github-copilot",
            base_url="https://api.individual.githubcopilot.com",
            reasoning=m.get("reasoning") is True,
            input=_input_modalities(m.get("modalities", {}).get("input")),
            cost=_cost(
                m.get("cost", {}).get("input", 0),
                m.get("cost", {}).get("output", 0),
                m.get("cost", {}).get("cache_read", 0),
                m.get("cost", {}).get("cache_write", 0),
            ),
            context_window=m.get("limit", {}).get("context", 128000),
            max_tokens=m.get("limit", {}).get("output", 8192),
            headers=dict(COPILOT_STATIC_HEADERS),
            compat={"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False}
            if api == "openai-completions"
            else None,
        )
        models.append(copilot_model)

    # --- MiniMax ---
    minimax_variants = [
        ("minimax", "minimax", "https://api.minimax.io/anthropic"),
        ("minimax-cn", "minimax-cn", "https://api.minimaxi.com/anthropic"),
    ]
    for json_key, provider, base_url in minimax_variants:
        section = data.get(json_key, {}).get("models", {})
        for model_id, m in section.items():
            if m.get("tool_call") is not True:
                continue
            models.append(_make_model(
                id=model_id,
                name=m.get("name") or model_id,
                api="anthropic-messages",
                provider=provider,
                base_url=base_url,
                reasoning=m.get("reasoning") is True,
                input=_input_modalities(m.get("modalities", {}).get("input")),
                cost=_cost(
                    m.get("cost", {}).get("input", 0),
                    m.get("cost", {}).get("output", 0),
                    m.get("cost", {}).get("cache_read", 0),
                    m.get("cost", {}).get("cache_write", 0),
                ),
                context_window=m.get("limit", {}).get("context", 4096),
                max_tokens=m.get("limit", {}).get("output", 4096),
            ))

    # --- Kimi For Coding ---
    kimi = data.get("kimi-for-coding", {}).get("models", {})
    for model_id, m in kimi.items():
        if m.get("tool_call") is not True:
            continue
        models.append(_make_model(
            id=model_id,
            name=m.get("name") or model_id,
            api="anthropic-messages",
            provider="kimi-coding",
            base_url="https://api.kimi.com/coding",
            reasoning=m.get("reasoning") is True,
            input=_input_modalities(m.get("modalities", {}).get("input")),
            cost=_cost(
                m.get("cost", {}).get("input", 0),
                m.get("cost", {}).get("output", 0),
                m.get("cost", {}).get("cache_read", 0),
                m.get("cost", {}).get("cache_write", 0),
            ),
            context_window=m.get("limit", {}).get("context", 4096),
            max_tokens=m.get("limit", {}).get("output", 4096),
        ))

    print(f"Loaded {len(models)} tool-capable models from models.dev")
    return models


def _fetch_openrouter(client: httpx.Client) -> list[ModelDict]:
    """Fetch from OpenRouter /api/v1/models."""
    print("Fetching models from OpenRouter API...")
    try:
        resp = client.get("https://openrouter.ai/api/v1/models", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"Failed to fetch OpenRouter models: {exc}", file=sys.stderr)
        return []

    models: list[ModelDict] = []
    for model in data.get("data", []):
        supported = model.get("supported_parameters") or []
        if "tools" not in supported:
            continue

        input_mods: list[str] = ["text"]
        arch_modality = (model.get("architecture") or {}).get("modality", "")
        if "image" in arch_modality:
            input_mods.append("image")

        pricing = model.get("pricing") or {}
        input_cost = float(pricing.get("prompt", "0")) * 1_000_000
        output_cost = float(pricing.get("completion", "0")) * 1_000_000
        cache_read_cost = float(pricing.get("input_cache_read", "0")) * 1_000_000
        cache_write_cost = float(pricing.get("input_cache_write", "0")) * 1_000_000

        top_prov = model.get("top_provider") or {}
        models.append(_make_model(
            id=model["id"],
            name=model.get("name", model["id"]),
            api="openai-completions",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            reasoning="reasoning" in supported,
            input=input_mods,
            cost=_cost(input_cost, output_cost, cache_read_cost, cache_write_cost),
            context_window=model.get("context_length", 4096),
            max_tokens=top_prov.get("max_completion_tokens", 4096),
        ))

    print(f"Fetched {len(models)} tool-capable models from OpenRouter")
    return models


def _fetch_ai_gateway(client: httpx.Client) -> list[ModelDict]:
    """Fetch from Vercel AI Gateway."""
    print("Fetching models from Vercel AI Gateway API...")
    try:
        resp = client.get(f"{AI_GATEWAY_MODELS_URL}/models", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"Failed to fetch Vercel AI Gateway models: {exc}", file=sys.stderr)
        return []

    models: list[ModelDict] = []
    items = data.get("data", [])
    if not isinstance(items, list):
        items = []

    for model in items:
        tags = model.get("tags") or []
        if "tool-use" not in tags:
            continue

        input_mods: list[str] = ["text"]
        if "vision" in tags:
            input_mods.append("image")

        pricing = model.get("pricing") or {}
        input_cost = _to_number(pricing.get("input")) * 1_000_000
        output_cost = _to_number(pricing.get("output")) * 1_000_000
        cache_read_cost = _to_number(pricing.get("input_cache_read")) * 1_000_000
        cache_write_cost = _to_number(pricing.get("input_cache_write")) * 1_000_000

        models.append(_make_model(
            id=model["id"],
            name=model.get("name") or model["id"],
            api="anthropic-messages",
            provider="vercel-ai-gateway",
            base_url=AI_GATEWAY_BASE_URL,
            reasoning="reasoning" in tags,
            input=input_mods,
            cost=_cost(input_cost, output_cost, cache_read_cost, cache_write_cost),
            context_window=model.get("context_window", 4096),
            max_tokens=model.get("max_tokens", 4096),
        ))

    print(f"Fetched {len(models)} tool-capable models from Vercel AI Gateway")
    return models


# ---------------------------------------------------------------------------
# Post-processing: overrides, static additions, dedup — mirroring TS logic.
# ---------------------------------------------------------------------------

def _has(models: list[ModelDict], provider: str, model_id: str) -> bool:
    return any(m["provider"] == provider and m["id"] == model_id for m in models)


def _apply_overrides_and_additions(all_models: list[ModelDict]) -> list[ModelDict]:
    """Apply the same fixups the TS script does after fetching."""

    # Filter out gpt-5.3-codex-spark from opencode/opencode-go
    all_models = [
        m for m in all_models
        if not (m["provider"] in ("opencode", "opencode-go") and m["id"] == "gpt-5.3-codex-spark")
    ]

    # Fix Opus 4.5 cache pricing
    for m in all_models:
        if m["provider"] == "anthropic" and m["id"] == "claude-opus-4-5":
            m["cost"]["cacheRead"] = 0.5
            m["cost"]["cacheWrite"] = 6.25

    # Temporary overrides (same as TS)
    for m in all_models:
        if m["provider"] == "amazon-bedrock" and "anthropic.claude-opus-4-6-v1" in m["id"]:
            m["cost"]["cacheRead"] = 0.5
            m["cost"]["cacheWrite"] = 6.25

        if (
            m["provider"] in ("anthropic", "opencode", "opencode-go", "github-copilot")
            and m["id"] in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4.6", "claude-sonnet-4.6")
        ):
            m["contextWindow"] = 1_000_000

        if m["provider"] == "google-antigravity" and m["id"] in (
            "claude-opus-4-6-thinking",
            "claude-sonnet-4-6",
        ):
            m["contextWindow"] = 1_000_000

        if m["provider"] in ("opencode", "opencode-go") and m["id"] in (
            "claude-sonnet-4-5",
            "claude-sonnet-4",
        ):
            m["contextWindow"] = 200_000

        if m["provider"] in ("opencode", "opencode-go") and m["id"] == "gpt-5.4":
            m["contextWindow"] = 272_000
            m["maxTokens"] = 128_000

        if m["provider"] == "openai" and m["id"] == "gpt-5.4":
            m["contextWindow"] = 272_000
            m["maxTokens"] = 128_000

        if m["provider"] == "openrouter" and m["id"] == "moonshotai/kimi-k2.5":
            m["cost"]["input"] = 0.41
            m["cost"]["output"] = 2.06
            m["cost"]["cacheRead"] = 0.07
            m["maxTokens"] = 4096

        if m["provider"] == "openrouter" and m["id"] == "z-ai/glm-5":
            m["cost"]["input"] = 0.6
            m["cost"]["output"] = 1.9
            m["cost"]["cacheRead"] = 0.119

    # --- Static additions (missing models) ---

    if not _has(all_models, "amazon-bedrock", "eu.anthropic.claude-opus-4-6-v1"):
        all_models.append(_make_model(
            id="eu.anthropic.claude-opus-4-6-v1", name="Claude Opus 4.6 (EU)",
            api="bedrock-converse-stream", provider="amazon-bedrock",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            reasoning=True, input=["text", "image"],
            cost=_cost(5, 25, 0.5, 6.25), context_window=200_000, max_tokens=128_000,
        ))

    if not _has(all_models, "anthropic", "claude-opus-4-6"):
        all_models.append(_make_model(
            id="claude-opus-4-6", name="Claude Opus 4.6",
            api="anthropic-messages", provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True, input=["text", "image"],
            cost=_cost(5, 25, 0.5, 6.25), context_window=1_000_000, max_tokens=128_000,
        ))

    if not _has(all_models, "anthropic", "claude-sonnet-4-6"):
        all_models.append(_make_model(
            id="claude-sonnet-4-6", name="Claude Sonnet 4.6",
            api="anthropic-messages", provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True, input=["text", "image"],
            cost=_cost(3, 15, 0.3, 3.75), context_window=1_000_000, max_tokens=64_000,
        ))

    if not _has(all_models, "google", "gemini-3.1-flash-lite-preview"):
        all_models.append(_make_model(
            id="gemini-3.1-flash-lite-preview", name="Gemini 3.1 Flash Lite Preview",
            api="google-generative-ai", provider="google",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            reasoning=True, input=["text", "image"],
            cost=_cost(), context_window=1_048_576, max_tokens=65_536,
        ))

    if not _has(all_models, "openai", "gpt-5-chat-latest"):
        all_models.append(_make_model(
            id="gpt-5-chat-latest", name="GPT-5 Chat Latest",
            api="openai-responses", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=False, input=["text", "image"],
            cost=_cost(1.25, 10, 0.125), context_window=128_000, max_tokens=16_384,
        ))

    if not _has(all_models, "openai", "gpt-5.1-codex"):
        all_models.append(_make_model(
            id="gpt-5.1-codex", name="GPT-5.1 Codex",
            api="openai-responses", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=True, input=["text", "image"],
            cost=_cost(1.25, 5, 0.125, 1.25), context_window=400_000, max_tokens=128_000,
        ))

    if not _has(all_models, "openai", "gpt-5.1-codex-max"):
        all_models.append(_make_model(
            id="gpt-5.1-codex-max", name="GPT-5.1 Codex Max",
            api="openai-responses", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=True, input=["text", "image"],
            cost=_cost(1.25, 10, 0.125), context_window=400_000, max_tokens=128_000,
        ))

    if not _has(all_models, "openai", "gpt-5.3-codex-spark"):
        all_models.append(_make_model(
            id="gpt-5.3-codex-spark", name="GPT-5.3 Codex Spark",
            api="openai-responses", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=True, input=["text"],
            cost=_cost(), context_window=128_000, max_tokens=16_384,
        ))

    if not _has(all_models, "openai", "gpt-5.4"):
        all_models.append(_make_model(
            id="gpt-5.4", name="GPT-5.4",
            api="openai-responses", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=True, input=["text", "image"],
            cost=_cost(2.5, 15, 0.25), context_window=272_000, max_tokens=128_000,
        ))

    # GitHub Copilot GPT-5.3 Codex
    copilot_base = next(
        (m for m in all_models if m["provider"] == "github-copilot" and m["id"] == "gpt-5.2-codex"),
        None,
    )
    if copilot_base and not _has(all_models, "github-copilot", "gpt-5.3-codex"):
        entry = dict(copilot_base)
        entry["id"] = "gpt-5.3-codex"
        entry["name"] = "GPT-5.3 Codex"
        all_models.append(entry)

    # MiniMax context window overrides + filter unsupported
    minimax_direct = {"MiniMax-M2.7", "MiniMax-M2.7-highspeed"}
    for m in all_models:
        if m["provider"] in ("minimax", "minimax-cn") and m["id"] in minimax_direct:
            m["contextWindow"] = 204_800
            m["maxTokens"] = 131_072
    all_models = [
        m for m in all_models
        if not (m["provider"] in ("minimax", "minimax-cn") and m["id"] not in minimax_direct)
    ]

    # --- OpenAI Codex (ChatGPT OAuth) ---
    codex_base_url = "https://chatgpt.com/backend-api"
    codex_ctx = 272_000
    codex_max = 128_000
    codex_models = [
        _make_model(id="gpt-5.1", name="GPT-5.1", api="openai-codex-responses", provider="openai-codex",
                     base_url=codex_base_url, reasoning=True, input=["text", "image"],
                     cost=_cost(1.25, 10, 0.125), context_window=codex_ctx, max_tokens=codex_max),
        _make_model(id="gpt-5.1-codex-max", name="GPT-5.1 Codex Max", api="openai-codex-responses",
                     provider="openai-codex", base_url=codex_base_url, reasoning=True, input=["text", "image"],
                     cost=_cost(1.25, 10, 0.125), context_window=codex_ctx, max_tokens=codex_max),
        _make_model(id="gpt-5.1-codex-mini", name="GPT-5.1 Codex Mini", api="openai-codex-responses",
                     provider="openai-codex", base_url=codex_base_url, reasoning=True, input=["text", "image"],
                     cost=_cost(0.25, 2, 0.025), context_window=codex_ctx, max_tokens=codex_max),
        _make_model(id="gpt-5.2", name="GPT-5.2", api="openai-codex-responses", provider="openai-codex",
                     base_url=codex_base_url, reasoning=True, input=["text", "image"],
                     cost=_cost(1.75, 14, 0.175), context_window=codex_ctx, max_tokens=codex_max),
        _make_model(id="gpt-5.2-codex", name="GPT-5.2 Codex", api="openai-codex-responses",
                     provider="openai-codex", base_url=codex_base_url, reasoning=True, input=["text", "image"],
                     cost=_cost(1.75, 14, 0.175), context_window=codex_ctx, max_tokens=codex_max),
        _make_model(id="gpt-5.3-codex", name="GPT-5.3 Codex", api="openai-codex-responses",
                     provider="openai-codex", base_url=codex_base_url, reasoning=True, input=["text", "image"],
                     cost=_cost(1.75, 14, 0.175), context_window=codex_ctx, max_tokens=codex_max),
        _make_model(id="gpt-5.4", name="GPT-5.4", api="openai-codex-responses", provider="openai-codex",
                     base_url=codex_base_url, reasoning=True, input=["text", "image"],
                     cost=_cost(2.5, 15, 0.25), context_window=codex_ctx, max_tokens=codex_max),
        _make_model(id="gpt-5.4-mini", name="GPT-5.4 Mini", api="openai-codex-responses",
                     provider="openai-codex", base_url=codex_base_url, reasoning=True, input=["text", "image"],
                     cost=_cost(0.75, 4.5, 0.075), context_window=codex_ctx, max_tokens=codex_max),
        _make_model(id="gpt-5.3-codex-spark", name="GPT-5.3 Codex Spark", api="openai-codex-responses",
                     provider="openai-codex", base_url=codex_base_url, reasoning=True, input=["text"],
                     cost=_cost(), context_window=128_000, max_tokens=codex_max),
    ]
    all_models.extend(codex_models)

    # --- Grok ---
    if not _has(all_models, "xai", "grok-code-fast-1"):
        all_models.append(_make_model(
            id="grok-code-fast-1", name="Grok Code Fast 1",
            api="openai-completions", provider="xai",
            base_url="https://api.x.ai/v1",
            reasoning=False, input=["text"],
            cost=_cost(0.2, 1.5, 0.02), context_window=32_768, max_tokens=8_192,
        ))

    # --- OpenRouter auto ---
    if not _has(all_models, "openrouter", "auto"):
        all_models.append(_make_model(
            id="auto", name="Auto",
            api="openai-completions", provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            reasoning=True, input=["text", "image"],
            cost=_cost(), context_window=2_000_000, max_tokens=30_000,
        ))

    # --- Google Cloud Code Assist (Gemini CLI) ---
    cca_url = "https://cloudcode-pa.googleapis.com"
    cca_models = [
        _make_model(id="gemini-2.5-pro", name="Gemini 2.5 Pro (Cloud Code Assist)",
                     api="google-gemini-cli", provider="google-gemini-cli", base_url=cca_url,
                     reasoning=True, input=["text", "image"], cost=_cost(),
                     context_window=1_048_576, max_tokens=65_535),
        _make_model(id="gemini-2.5-flash", name="Gemini 2.5 Flash (Cloud Code Assist)",
                     api="google-gemini-cli", provider="google-gemini-cli", base_url=cca_url,
                     reasoning=True, input=["text", "image"], cost=_cost(),
                     context_window=1_048_576, max_tokens=65_535),
        _make_model(id="gemini-2.0-flash", name="Gemini 2.0 Flash (Cloud Code Assist)",
                     api="google-gemini-cli", provider="google-gemini-cli", base_url=cca_url,
                     reasoning=False, input=["text", "image"], cost=_cost(),
                     context_window=1_048_576, max_tokens=8_192),
        _make_model(id="gemini-3-pro-preview", name="Gemini 3 Pro Preview (Cloud Code Assist)",
                     api="google-gemini-cli", provider="google-gemini-cli", base_url=cca_url,
                     reasoning=True, input=["text", "image"], cost=_cost(),
                     context_window=1_048_576, max_tokens=65_535),
        _make_model(id="gemini-3-flash-preview", name="Gemini 3 Flash Preview (Cloud Code Assist)",
                     api="google-gemini-cli", provider="google-gemini-cli", base_url=cca_url,
                     reasoning=True, input=["text", "image"], cost=_cost(),
                     context_window=1_048_576, max_tokens=65_535),
        _make_model(id="gemini-3.1-pro-preview", name="Gemini 3.1 Pro Preview (Cloud Code Assist)",
                     api="google-gemini-cli", provider="google-gemini-cli", base_url=cca_url,
                     reasoning=True, input=["text", "image"], cost=_cost(),
                     context_window=1_048_576, max_tokens=65_535),
    ]
    all_models.extend(cca_models)

    # --- Antigravity ---
    ag_url = "https://daily-cloudcode-pa.sandbox.googleapis.com"
    ag_models = [
        _make_model(id="gemini-3.1-pro-high", name="Gemini 3.1 Pro High (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=True, input=["text", "image"], cost=_cost(2, 12, 0.2, 2.375),
                     context_window=1_048_576, max_tokens=65_535),
        _make_model(id="gemini-3.1-pro-low", name="Gemini 3.1 Pro Low (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=True, input=["text", "image"], cost=_cost(2, 12, 0.2, 2.375),
                     context_window=1_048_576, max_tokens=65_535),
        _make_model(id="gemini-3-flash", name="Gemini 3 Flash (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=True, input=["text", "image"], cost=_cost(0.5, 3, 0.5),
                     context_window=1_048_576, max_tokens=65_535),
        _make_model(id="claude-sonnet-4-5", name="Claude Sonnet 4.5 (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=False, input=["text", "image"], cost=_cost(3, 15, 0.3, 3.75),
                     context_window=200_000, max_tokens=64_000),
        _make_model(id="claude-sonnet-4-5-thinking", name="Claude Sonnet 4.5 Thinking (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=True, input=["text", "image"], cost=_cost(3, 15, 0.3, 3.75),
                     context_window=200_000, max_tokens=64_000),
        _make_model(id="claude-opus-4-5-thinking", name="Claude Opus 4.5 Thinking (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=True, input=["text", "image"], cost=_cost(5, 25, 0.5, 6.25),
                     context_window=200_000, max_tokens=64_000),
        _make_model(id="claude-opus-4-6-thinking", name="Claude Opus 4.6 Thinking (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=True, input=["text", "image"], cost=_cost(5, 25, 0.5, 6.25),
                     context_window=200_000, max_tokens=128_000),
        _make_model(id="claude-sonnet-4-6", name="Claude Sonnet 4.6 (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=True, input=["text", "image"], cost=_cost(3, 15, 0.3, 3.75),
                     context_window=200_000, max_tokens=64_000),
        _make_model(id="gpt-oss-120b-medium", name="GPT-OSS 120B Medium (Antigravity)",
                     api="google-gemini-cli", provider="google-antigravity", base_url=ag_url,
                     reasoning=False, input=["text"], cost=_cost(0.09, 0.36),
                     context_window=131_072, max_tokens=32_768),
    ]
    all_models.extend(ag_models)

    # --- Google Vertex ---
    vertex_url = "https://{location}-aiplatform.googleapis.com"
    vertex_models = [
        _make_model(id="gemini-3-pro-preview", name="Gemini 3 Pro Preview (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(2, 12, 0.2),
                     context_window=1_000_000, max_tokens=64_000),
        _make_model(id="gemini-3.1-pro-preview", name="Gemini 3.1 Pro Preview (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(2, 12, 0.2),
                     context_window=1_048_576, max_tokens=65_536),
        _make_model(id="gemini-3.1-pro-preview-customtools", name="Gemini 3.1 Pro Preview Custom Tools (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(2, 12, 0.2),
                     context_window=1_048_576, max_tokens=65_536),
        _make_model(id="gemini-3-flash-preview", name="Gemini 3 Flash Preview (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(0.5, 3, 0.05),
                     context_window=1_048_576, max_tokens=65_536),
        _make_model(id="gemini-2.0-flash", name="Gemini 2.0 Flash (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=False, input=["text", "image"], cost=_cost(0.15, 0.6, 0.0375),
                     context_window=1_048_576, max_tokens=8_192),
        _make_model(id="gemini-2.0-flash-lite", name="Gemini 2.0 Flash Lite (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(0.075, 0.3, 0.01875),
                     context_window=1_048_576, max_tokens=65_536),
        _make_model(id="gemini-2.5-pro", name="Gemini 2.5 Pro (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(1.25, 10, 0.125),
                     context_window=1_048_576, max_tokens=65_536),
        _make_model(id="gemini-2.5-flash", name="Gemini 2.5 Flash (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(0.3, 2.5, 0.03),
                     context_window=1_048_576, max_tokens=65_536),
        _make_model(id="gemini-2.5-flash-lite-preview-09-2025", name="Gemini 2.5 Flash Lite Preview 09-25 (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(0.1, 0.4, 0.01),
                     context_window=1_048_576, max_tokens=65_536),
        _make_model(id="gemini-2.5-flash-lite", name="Gemini 2.5 Flash Lite (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=True, input=["text", "image"], cost=_cost(0.1, 0.4, 0.01),
                     context_window=1_048_576, max_tokens=65_536),
        _make_model(id="gemini-1.5-pro", name="Gemini 1.5 Pro (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=False, input=["text", "image"], cost=_cost(1.25, 5, 0.3125),
                     context_window=1_000_000, max_tokens=8_192),
        _make_model(id="gemini-1.5-flash", name="Gemini 1.5 Flash (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=False, input=["text", "image"], cost=_cost(0.075, 0.3, 0.01875),
                     context_window=1_000_000, max_tokens=8_192),
        _make_model(id="gemini-1.5-flash-8b", name="Gemini 1.5 Flash-8B (Vertex)",
                     api="google-vertex", provider="google-vertex", base_url=vertex_url,
                     reasoning=False, input=["text", "image"], cost=_cost(0.0375, 0.15, 0.01),
                     context_window=1_000_000, max_tokens=8_192),
    ]
    all_models.extend(vertex_models)

    # --- Kimi For Coding static fallback ---
    kimi_url = "https://api.kimi.com/coding"
    kimi_fallback = [
        _make_model(id="kimi-k2-thinking", name="Kimi K2 Thinking",
                     api="anthropic-messages", provider="kimi-coding", base_url=kimi_url,
                     reasoning=True, input=["text"], cost=_cost(),
                     context_window=262_144, max_tokens=32_768),
        _make_model(id="k2p5", name="Kimi K2.5",
                     api="anthropic-messages", provider="kimi-coding", base_url=kimi_url,
                     reasoning=True, input=["text"], cost=_cost(),
                     context_window=262_144, max_tokens=32_768),
    ]
    for km in kimi_fallback:
        if not _has(all_models, "kimi-coding", km["id"]):
            all_models.append(km)

    # --- Azure OpenAI (mirror of openai responses models) ---
    azure_models = [
        {**m, "api": "azure-openai-responses", "provider": "azure-openai-responses", "baseUrl": ""}
        for m in all_models
        if m["provider"] == "openai" and m["api"] == "openai-responses"
    ]
    all_models.extend(azure_models)

    return all_models


# ---------------------------------------------------------------------------
# Dedup & output
# ---------------------------------------------------------------------------


def _group_and_dedup(all_models: list[ModelDict]) -> dict[str, dict[str, ModelDict]]:
    """Group by provider, dedup by model ID (first occurrence wins)."""
    providers: dict[str, dict[str, ModelDict]] = {}
    for model in all_models:
        prov = model["provider"]
        mid = model["id"]
        providers.setdefault(prov, {})
        if mid not in providers[prov]:
            providers[prov][mid] = model
    return providers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the nu_ai model catalog.")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON to stdout; do not write file.")
    args = parser.parse_args()

    client = httpx.Client(follow_redirects=True)
    try:
        models_dev = _fetch_models_dev(client)
        openrouter = _fetch_openrouter(client)
        ai_gateway = _fetch_ai_gateway(client)
    finally:
        client.close()

    # models.dev has priority (listed first)
    all_models = models_dev + openrouter + ai_gateway
    all_models = _apply_overrides_and_additions(all_models)
    catalog = _group_and_dedup(all_models)

    # Sort providers and model IDs for deterministic output
    sorted_catalog: dict[str, dict[str, ModelDict]] = {}
    for prov_id in sorted(catalog):
        sorted_catalog[prov_id] = dict(sorted(catalog[prov_id].items()))

    output = json.dumps(sorted_catalog, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(output)
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote {OUTPUT_PATH}")

    # Statistics
    total = sum(len(models) for models in catalog.values())
    reasoning = sum(
        1 for models in catalog.values() for m in models.values() if m["reasoning"]
    )
    print("\nModel Statistics:")
    print(f"  Total tool-capable models: {total}")
    print(f"  Reasoning-capable models: {reasoning}")
    for prov_id in sorted(catalog):
        print(f"  {prov_id}: {len(catalog[prov_id])} models")


if __name__ == "__main__":
    main()
