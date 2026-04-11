"""Helpers for building provider ``StreamOptions`` from ``SimpleStreamOptions``.

Port of ``packages/ai/src/providers/simple-options.ts``. Every provider's
``stream_simple`` implementation funnels through :func:`build_base_options`
to get a uniform ``StreamOptions`` instance, then tweaks it with provider-
specific knobs; :func:`clamp_reasoning` and :func:`adjust_max_tokens_for_thinking`
encapsulate the two tweaks most providers share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from pi_ai.types import StreamOptions

if TYPE_CHECKING:
    from pi_ai.types import Model, SimpleStreamOptions, ThinkingBudgets, ThinkingLevel


def build_base_options(
    model: Model,
    options: SimpleStreamOptions | None,
    api_key: str | None = None,
) -> StreamOptions:
    """Lower ``SimpleStreamOptions`` into the provider-level ``StreamOptions``.

    ``max_tokens`` defaults to ``min(model.max_tokens, 32000)`` when the
    caller does not override it — mirrors the upstream clamp.
    """
    resolved_max_tokens = options.max_tokens if options and options.max_tokens else min(model.max_tokens, 32_000)
    resolved_api_key = api_key or (options.api_key if options else None)

    return StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=resolved_max_tokens,
        api_key=resolved_api_key,
        cache_retention=options.cache_retention if options else None,
        session_id=options.session_id if options else None,
        headers=options.headers if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else None,
        metadata=options.metadata if options else None,
    )


def clamp_reasoning(effort: ThinkingLevel | None) -> ThinkingLevel | None:
    """Demote ``xhigh`` to ``high`` for providers that don't support it.

    Providers that support ``xhigh`` natively bypass this helper; those that
    accept only the original ``minimal``/``low``/``medium``/``high`` levels
    use it to fold the extra level into the closest supported one.
    """
    return "high" if effort == "xhigh" else effort


class AdjustedTokenBudget(TypedDict):
    max_tokens: int
    thinking_budget: int


_DEFAULT_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}
"""Default token budget per thinking level."""

_MIN_OUTPUT_TOKENS = 1024


def adjust_max_tokens_for_thinking(
    base_max_tokens: int,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> AdjustedTokenBudget:
    """Compute ``max_tokens``/``thinking_budget`` for a thinking-enabled request.

    The budget starts from the per-level default (or ``custom_budgets``
    override) and is reserved on top of ``base_max_tokens``. The final
    ``max_tokens`` is clamped to ``model_max_tokens``. If that clamp pushes
    the total down to or below the thinking budget, the budget shrinks so
    at least :data:`_MIN_OUTPUT_TOKENS` remain for the response body.
    """
    budgets: dict[str, int] = dict(_DEFAULT_BUDGETS)
    if custom_budgets is not None:
        for level in ("minimal", "low", "medium", "high"):
            value = getattr(custom_budgets, level, None)
            if value is not None:
                budgets[level] = value

    level = clamp_reasoning(reasoning_level)
    assert level is not None
    thinking_budget = budgets[level]
    max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - _MIN_OUTPUT_TOKENS)

    return {"max_tokens": max_tokens, "thinking_budget": thinking_budget}


__all__ = [
    "AdjustedTokenBudget",
    "adjust_max_tokens_for_thinking",
    "build_base_options",
    "clamp_reasoning",
]
