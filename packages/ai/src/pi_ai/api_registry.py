"""API provider registry.

Direct port of ``packages/ai/src/api-registry.ts``. Providers register a
``stream`` and ``stream_simple`` function for a given :data:`pi_ai.types.Api`
identifier; ``stream``/``complete`` at the top level look up providers through
this registry.

Differences from upstream:

* TypeScript generics ``<TApi, TOptions>`` → Python keeps the public API
  nominal (``ApiProvider.api: str``) and validates the model-to-api match at
  call time, same as the TS ``wrapStream`` guard. No runtime type variance
  is lost because providers always construct the wrappers themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pi_ai.types import Api, Context, Model, SimpleStreamOptions, StreamOptions
    from pi_ai.utils.event_stream import AssistantMessageEventStream


class ApiStreamFunction(Protocol):
    def __call__(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
        /,
    ) -> AssistantMessageEventStream: ...


class ApiStreamSimpleFunction(Protocol):
    def __call__(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
        /,
    ) -> AssistantMessageEventStream: ...


@dataclass(frozen=True, slots=True)
class ApiProvider:
    """Public registration record for an API provider."""

    api: Api
    stream: ApiStreamFunction
    stream_simple: ApiStreamSimpleFunction


@dataclass(frozen=True, slots=True)
class _RegisteredApiProvider:
    provider: ApiProvider
    source_id: str | None = None


_api_provider_registry: dict[str, _RegisteredApiProvider] = {}


def _wrap_stream(api: Api, fn: ApiStreamFunction) -> ApiStreamFunction:
    def wrapped(
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return fn(model, context, options)

    return wrapped


def _wrap_stream_simple(api: Api, fn: ApiStreamSimpleFunction) -> ApiStreamSimpleFunction:
    def wrapped(
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return fn(model, context, options)

    return wrapped


def register_api_provider(provider: ApiProvider, source_id: str | None = None) -> None:
    """Register (or replace) an API provider."""
    wrapped = ApiProvider(
        api=provider.api,
        stream=_wrap_stream(provider.api, provider.stream),
        stream_simple=_wrap_stream_simple(provider.api, provider.stream_simple),
    )
    _api_provider_registry[provider.api] = _RegisteredApiProvider(provider=wrapped, source_id=source_id)


def get_api_provider(api: Api) -> ApiProvider | None:
    entry = _api_provider_registry.get(api)
    return entry.provider if entry is not None else None


def get_api_providers() -> list[ApiProvider]:
    return [entry.provider for entry in _api_provider_registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """Remove every provider that was registered with ``source_id``."""
    to_remove = [api for api, entry in _api_provider_registry.items() if entry.source_id == source_id]
    for api in to_remove:
        del _api_provider_registry[api]


def clear_api_providers() -> None:
    _api_provider_registry.clear()


__all__ = [
    "ApiProvider",
    "ApiStreamFunction",
    "ApiStreamSimpleFunction",
    "clear_api_providers",
    "get_api_provider",
    "get_api_providers",
    "register_api_provider",
    "unregister_api_providers",
]
