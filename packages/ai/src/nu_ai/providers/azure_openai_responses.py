"""Azure OpenAI Responses provider — port of ``providers/azure-openai-responses.ts``.

Thin adapter over the OpenAI Responses provider that resolves
Azure-specific configuration (resource name, deployment name, API
version) before delegating to the shared Responses stream processor.
Uses ``httpx`` against the Azure OpenAI endpoint rather than the
official ``openai`` SDK's ``AzureOpenAI`` class (same HTTP calls,
no extra dependency).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from nu_ai.providers.openai_responses import (
    stream_openai_responses,
)
from nu_ai.types import (
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)

if TYPE_CHECKING:
    from nu_ai.utils.event_stream import AssistantMessageEventStream

DEFAULT_AZURE_API_VERSION = "v1"


def _parse_deployment_name_map(value: str | None) -> dict[str, str]:
    """Parse ``AZURE_OPENAI_DEPLOYMENT_NAME_MAP`` env var."""
    if not value:
        return {}
    result: dict[str, str] = {}
    for entry in value.split(","):
        entry = entry.strip()
        if "=" not in entry:
            continue
        model_id, deployment = entry.split("=", 1)
        if model_id.strip() and deployment.strip():
            result[model_id.strip()] = deployment.strip()
    return result


def _resolve_deployment_name(model: Model, azure_deployment_name: str | None = None) -> str:
    if azure_deployment_name:
        return azure_deployment_name
    mapped = _parse_deployment_name_map(os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_MAP"))
    return mapped.get(model.id, model.id)


def _resolve_azure_config(
    model: Model,
    *,
    azure_base_url: str | None = None,
    azure_resource_name: str | None = None,
    azure_api_version: str | None = None,
) -> tuple[str, str]:
    """Resolve (base_url, api_version) for Azure OpenAI."""
    api_version = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_API_VERSION

    base_url = (azure_base_url or "").strip() or os.environ.get("AZURE_OPENAI_BASE_URL", "").strip() or None
    resource_name = azure_resource_name or os.environ.get("AZURE_OPENAI_RESOURCE_NAME")

    if not base_url and resource_name:
        base_url = f"https://{resource_name}.openai.azure.com/openai/v1"
    if not base_url and model.base_url:
        base_url = model.base_url
    if not base_url:
        raise ValueError(
            "Azure OpenAI base URL required. Set AZURE_OPENAI_BASE_URL or "
            "AZURE_OPENAI_RESOURCE_NAME, or pass azure_base_url."
        )

    return base_url.rstrip("/"), api_version


def stream_azure_openai_responses(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream against the Azure OpenAI Responses endpoint.

    Resolves Azure-specific config then delegates to the shared
    OpenAI Responses stream processor via ``stream_openai_responses``.
    """
    opts = options or StreamOptions()

    # Extract Azure-specific options
    azure_base_url = getattr(opts, "azure_base_url", None)
    azure_resource_name = getattr(opts, "azure_resource_name", None)
    azure_api_version = getattr(opts, "azure_api_version", None)
    azure_deployment_name = getattr(opts, "azure_deployment_name", None)

    base_url, _api_version = _resolve_azure_config(
        model,
        azure_base_url=azure_base_url,
        azure_resource_name=azure_resource_name,
        azure_api_version=azure_api_version,
    )

    deployment = _resolve_deployment_name(model, azure_deployment_name)

    # Build an Azure-flavored model with the resolved base URL
    azure_model = Model(
        id=deployment,
        name=model.name,
        api="azure-openai-responses",
        provider=model.provider,
        base_url=base_url,
        reasoning=model.reasoning,
        input=model.input,
        cost=model.cost,
        context_window=model.context_window,
        max_tokens=model.max_tokens,
    )

    return stream_openai_responses(azure_model, context, opts)


def stream_simple_azure_openai_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simple-stream variant for Azure OpenAI Responses."""
    from nu_ai.providers.simple_options import build_base_options

    opts = options or SimpleStreamOptions()
    base = build_base_options(model, opts)
    return stream_azure_openai_responses(model, context, base)


__all__ = [
    "stream_azure_openai_responses",
    "stream_simple_azure_openai_responses",
]
