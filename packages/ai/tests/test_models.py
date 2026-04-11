"""Tests for pi_ai.models.

Ported from the documented contract in ``packages/ai/src/models.ts``. The
upstream has no dedicated models test file; these cover the five exported
helpers plus the integrity of ``models_generated``.
"""

from __future__ import annotations

import pytest
from pi_ai.models import (
    MODELS,
    calculate_cost,
    get_model,
    get_models,
    get_providers,
    models_are_equal,
    supports_xhigh,
)
from pi_ai.types import Cost, Model, ModelCost, Usage


def _mk_model(model_id: str = "x", provider: str = "anthropic", cost: ModelCost | None = None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="anthropic-messages",
        provider=provider,
        base_url="",
        reasoning=False,
        input=["text"],
        cost=cost or ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=0,
        max_tokens=0,
    )


# ---------------------------------------------------------------------------
# Generated catalog integrity
# ---------------------------------------------------------------------------


class TestModelsGenerated:
    def test_catalog_is_non_empty(self) -> None:
        assert len(MODELS) > 0
        assert sum(len(v) for v in MODELS.values()) > 0

    def test_all_entries_are_valid_models(self) -> None:
        for provider, models in MODELS.items():
            assert len(models) > 0, f"{provider} has no models"
            for _mid, model in models.items():
                assert isinstance(model, Model)
                assert model.provider == provider

    def test_known_providers_present(self) -> None:
        # Spot-check a few known upstream providers survive the import.
        for expected in ("anthropic", "openai", "google", "amazon-bedrock"):
            assert expected in MODELS


# ---------------------------------------------------------------------------
# get_model / get_models / get_providers
# ---------------------------------------------------------------------------


class TestGetProviders:
    def test_returns_all_catalog_providers(self) -> None:
        providers = get_providers()
        assert "anthropic" in providers
        assert set(providers) == set(MODELS.keys())


class TestGetModels:
    def test_returns_list_of_models_for_provider(self) -> None:
        models = get_models("anthropic")
        assert len(models) > 0
        assert all(m.provider == "anthropic" for m in models)

    def test_unknown_provider_returns_empty(self) -> None:
        assert get_models("no-such-provider") == []


class TestGetModel:
    def test_known_model_returns_instance(self) -> None:
        anth = get_models("anthropic")
        assert anth, "anthropic catalog unexpectedly empty"
        sample = anth[0]
        fetched = get_model("anthropic", sample.id)
        assert fetched is not None
        assert fetched.id == sample.id

    def test_unknown_model_returns_none(self) -> None:
        assert get_model("anthropic", "no-such-model") is None

    def test_unknown_provider_returns_none(self) -> None:
        assert get_model("no-such-provider", "no-such-model") is None


# ---------------------------------------------------------------------------
# calculate_cost
# ---------------------------------------------------------------------------


class TestCalculateCost:
    def test_basic_cost_arithmetic(self) -> None:
        model = _mk_model(cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75))
        usage = Usage(
            input=1_000_000,
            output=2_000_000,
            cache_read=100_000,
            cache_write=50_000,
            total_tokens=3_150_000,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )
        cost = calculate_cost(model, usage)
        assert cost.input == pytest.approx(3.0)
        assert cost.output == pytest.approx(30.0)
        assert cost.cache_read == pytest.approx(0.03)
        assert cost.cache_write == pytest.approx(0.1875)
        assert cost.total == pytest.approx(3.0 + 30.0 + 0.03 + 0.1875)

    def test_zero_usage_yields_zero_cost(self) -> None:
        model = _mk_model(cost=ModelCost(input=10, output=10, cache_read=10, cache_write=10))
        usage = Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )
        cost = calculate_cost(model, usage)
        assert cost.total == 0

    def test_mutates_usage_cost_in_place(self) -> None:
        model = _mk_model(cost=ModelCost(input=1.0, output=0.0, cache_read=0.0, cache_write=0.0))
        usage = Usage(
            input=1_000_000,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=1_000_000,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )
        calculate_cost(model, usage)
        assert usage.cost.input == pytest.approx(1.0)
        assert usage.cost.total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# supports_xhigh
# ---------------------------------------------------------------------------


class TestSupportsXhigh:
    @pytest.mark.parametrize(
        "model_id",
        ["gpt-5.2-foo", "gpt-5.3-bar", "gpt-5.4", "claude-opus-4-6", "opus-4.6"],
    )
    def test_supports(self, model_id: str) -> None:
        assert supports_xhigh(_mk_model(model_id)) is True

    @pytest.mark.parametrize(
        "model_id",
        ["gpt-4", "gpt-5.1", "claude-opus-4-5", "claude-3-5-sonnet"],
    )
    def test_does_not_support(self, model_id: str) -> None:
        assert supports_xhigh(_mk_model(model_id)) is False


# ---------------------------------------------------------------------------
# models_are_equal
# ---------------------------------------------------------------------------


class TestModelsAreEqual:
    def test_same_id_and_provider(self) -> None:
        a = _mk_model("x", "anthropic")
        b = _mk_model("x", "anthropic")
        assert models_are_equal(a, b) is True

    def test_different_id(self) -> None:
        a = _mk_model("x", "anthropic")
        b = _mk_model("y", "anthropic")
        assert models_are_equal(a, b) is False

    def test_different_provider(self) -> None:
        a = _mk_model("x", "anthropic")
        b = _mk_model("x", "openai")
        assert models_are_equal(a, b) is False

    def test_none_returns_false(self) -> None:
        a = _mk_model("x", "anthropic")
        assert models_are_equal(a, None) is False
        assert models_are_equal(None, a) is False
        assert models_are_equal(None, None) is False
