"""Top-level ``stream``, ``stream_simple``, ``complete``, ``complete_simple``.

Direct port of ``packages/ai/src/stream.ts``. Resolves a provider from the
registry by ``model.api`` and delegates streaming. Importing this module
triggers :func:`nu_ai.providers.register_builtins.register_builtin_providers`
as a side-effect so the registry has every built-in provider wired up
before the first call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nu_ai.api_registry import get_api_provider

# Side-effect: register every built-in provider before anyone calls stream().
from nu_ai.providers import register_builtins as _register_builtins  # noqa: F401  # pyright: ignore[reportUnusedImport]

if TYPE_CHECKING:
    from nu_ai.api_registry import ApiProvider
    from nu_ai.types import (
        Api,
        AssistantMessage,
        Context,
        Model,
        SimpleStreamOptions,
        StreamOptions,
    )
    from nu_ai.utils.event_stream import AssistantMessageEventStream


def _resolve_provider(api: Api) -> ApiProvider:
    provider = get_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response from ``model`` using its registered provider.

    Returns an :class:`AssistantMessageEventStream` — errors are encoded in
    the returned stream, not raised.
    """
    return _resolve_provider(model.api).stream(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """Convenience wrapper over :func:`stream` that awaits the final result."""
    return await stream(model, context, options).result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Reasoning-aware variant of :func:`stream`.

    Accepts :class:`SimpleStreamOptions` (with a ``reasoning`` level) and
    delegates to the provider's ``stream_simple`` hook.
    """
    return _resolve_provider(model.api).stream_simple(model, context, options)


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """Convenience wrapper over :func:`stream_simple`."""
    return await stream_simple(model, context, options).result()


__all__ = ["complete", "complete_simple", "stream", "stream_simple"]
