"""Model discovery — detect local LLM servers and enumerate their models.

Mirrors the detection logic from utils/model-discovery.ts but uses
httpx.AsyncClient instead of browser fetch.  LM Studio SDK calls are
replaced with a plain OpenAI-compatible /v1/models HTTP request.
"""

from __future__ import annotations

import logging

import httpx

from nu_web_ui.types import ModelCostInfo, ModelInfo

logger = logging.getLogger(__name__)

_EMPTY_COST = ModelCostInfo()

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


async def detect_ollama(
    base_url: str = "http://localhost:11434",
    *,
    timeout: float = 5.0,
) -> list[ModelInfo]:
    """Discover models from an Ollama server.

    Only returns models that report ``tools`` in their capabilities list,
    matching the upstream filter.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("Ollama not available at %s: %s", base_url, exc)
            return []

        data = resp.json()
        raw_models: list[dict] = data.get("models", [])

    models: list[ModelInfo] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for m in raw_models:
            name: str = m.get("name", "")
            try:
                detail_resp = await client.post(f"{base_url}/api/show", json={"model": name})
                detail_resp.raise_for_status()
                detail = detail_resp.json()
            except Exception as exc:
                logger.debug("Failed to get details for Ollama model %s: %s", name, exc)
                continue

            capabilities: list[str] = detail.get("capabilities", [])
            if "tools" not in capabilities:
                logger.debug("Skipping %s: no tool support", name)
                continue

            model_info: dict = detail.get("model_info", {})
            architecture: str = model_info.get("general.architecture", "")
            context_key = f"{architecture}.context_length"
            context_window = int(model_info.get(context_key, 8192))
            max_tokens = context_window * 10

            models.append(
                ModelInfo(
                    id=name,
                    name=name,
                    api="openai-completions",
                    provider="",
                    base_url=f"{base_url}/v1",
                    reasoning="thinking" in capabilities,
                    input=["text"],
                    cost=_EMPTY_COST,
                    context_window=context_window,
                    max_tokens=max_tokens,
                )
            )

    return models


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------


async def detect_lmstudio(
    base_url: str = "http://localhost:1234",
    *,
    timeout: float = 5.0,
) -> list[ModelInfo]:
    """Discover models from an LM Studio server via /v1/models."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(f"{base_url}/v1/models")
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("LM Studio not available at %s: %s", base_url, exc)
            return []

        data = resp.json()

    raw_models: list[dict] = data.get("data", [])
    models: list[ModelInfo] = []
    for m in raw_models:
        model_id: str = m.get("id", "")
        # LM Studio returns context_length in the model object when available.
        context_window = int(m.get("context_length") or m.get("max_context_length") or 8192)
        max_tokens = context_window
        models.append(
            ModelInfo(
                id=model_id,
                name=m.get("name") or m.get("display_name") or model_id,
                api="openai-completions",
                provider="",
                base_url=f"{base_url}/v1",
                reasoning=False,
                input=["text"],
                cost=_EMPTY_COST,
                context_window=context_window,
                max_tokens=max_tokens,
            )
        )
    return models


# ---------------------------------------------------------------------------
# llama.cpp / vLLM  (generic OpenAI-compatible /v1/models)
# ---------------------------------------------------------------------------


async def detect_openai_compatible(
    base_url: str,
    provider_name: str = "",
    *,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> list[ModelInfo]:
    """Generic discovery for any OpenAI-compatible /v1/models endpoint."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        try:
            resp = await client.get(f"{base_url}/v1/models")
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("Server not available at %s: %s", base_url, exc)
            return []

        data = resp.json()

    raw_models: list[dict] = data.get("data", [])
    models: list[ModelInfo] = []
    for m in raw_models:
        model_id: str = m.get("id", "")
        context_window = int(m.get("max_model_len") or m.get("context_length") or 8192)
        max_tokens = min(context_window, 4096)
        models.append(
            ModelInfo(
                id=model_id,
                name=model_id,
                api="openai-completions",
                provider=provider_name,
                base_url=f"{base_url}/v1",
                reasoning=False,
                input=["text"],
                cost=_EMPTY_COST,
                context_window=context_window,
                max_tokens=max_tokens,
            )
        )
    return models


# ---------------------------------------------------------------------------
# All-at-once discovery
# ---------------------------------------------------------------------------


async def discover_all_local_models() -> list[ModelInfo]:
    """Run Ollama and LM Studio discovery concurrently and merge the results."""
    import asyncio

    ollama_task = asyncio.create_task(detect_ollama())
    lmstudio_task = asyncio.create_task(detect_lmstudio())

    ollama_models, lmstudio_models = await asyncio.gather(ollama_task, lmstudio_task, return_exceptions=False)

    return list(ollama_models) + list(lmstudio_models)


__all__ = [
    "detect_lmstudio",
    "detect_ollama",
    "detect_openai_compatible",
    "discover_all_local_models",
]
