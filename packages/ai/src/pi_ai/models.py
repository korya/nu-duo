"""Model registry and cost helpers.

Direct port of ``packages/ai/src/models.ts``. The TS conditional type
``ModelApi<TProvider, TModelId>`` is deliberately not reproduced — Python
cannot express "the api of the model whose id is X in provider Y" as a
dependent type, so :func:`get_model` simply returns ``Model`` (or ``None``
when absent). Callers branch on ``model.api`` at runtime like every other
pi_ai consumer already does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_ai.models_generated import MODELS

if TYPE_CHECKING:
    from pi_ai.types import Cost, Model, Provider, Usage


def get_model(provider: Provider, model_id: str) -> Model | None:
    """Look up a model by ``provider`` and ``model_id`` in the generated catalog."""
    provider_models = MODELS.get(provider)
    if provider_models is None:
        return None
    return provider_models.get(model_id)


def get_providers() -> list[Provider]:
    """Return every provider present in the generated catalog."""
    return list(MODELS.keys())


def get_models(provider: Provider) -> list[Model]:
    """Return every model defined for ``provider``."""
    provider_models = MODELS.get(provider)
    return list(provider_models.values()) if provider_models is not None else []


def calculate_cost(model: Model, usage: Usage) -> Cost:
    """Populate ``usage.cost`` from ``model.cost`` and token counts.

    Mutates ``usage.cost`` **in place** and returns it, matching the TS
    behaviour. All per-million rates are converted to per-token before
    multiplying by the observed usage.
    """
    usage.cost.input = (model.cost.input / 1_000_000) * usage.input
    usage.cost.output = (model.cost.output / 1_000_000) * usage.output
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000) * usage.cache_write
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage.cost


def supports_xhigh(model: Model) -> bool:
    """Whether the model supports the ``xhigh`` thinking level.

    Supported families (mirror of upstream):

    * GPT-5.2 / GPT-5.3 / GPT-5.4
    * Claude Opus 4.6 (``opus-4-6`` / ``opus-4.6``)
    """
    if "gpt-5.2" in model.id or "gpt-5.3" in model.id or "gpt-5.4" in model.id:
        return True
    return "opus-4-6" in model.id or "opus-4.6" in model.id


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Whether ``a`` and ``b`` refer to the same model (id + provider match).

    Returns ``False`` if either argument is ``None`` — same contract as the
    TS ``modelsAreEqual``.
    """
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider


__all__ = [
    "MODELS",
    "calculate_cost",
    "get_model",
    "get_models",
    "get_providers",
    "models_are_equal",
    "supports_xhigh",
]
