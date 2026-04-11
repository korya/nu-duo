"""Tests for pi_ai.providers.simple_options.

Port of the documented contract in ``packages/ai/src/providers/simple-options.ts``.
"""

from __future__ import annotations

import pytest
from pi_ai.providers.simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_reasoning,
)
from pi_ai.types import (
    Model,
    ModelCost,
    SimpleStreamOptions,
    ThinkingBudgets,
)


def _model(max_tokens: int = 8192) -> Model:
    return Model(
        id="m",
        name="m",
        api="anthropic-messages",
        provider="anthropic",
        base_url="",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128_000,
        max_tokens=max_tokens,
    )


class TestBuildBaseOptions:
    def test_defaults_to_min_of_model_and_32k(self) -> None:
        opts = build_base_options(_model(max_tokens=16_000), None)
        assert opts.max_tokens == 16_000

    def test_clamps_to_32k_cap(self) -> None:
        opts = build_base_options(_model(max_tokens=128_000), None)
        assert opts.max_tokens == 32_000

    def test_explicit_max_tokens_passes_through(self) -> None:
        opts = build_base_options(_model(), SimpleStreamOptions(max_tokens=500))
        assert opts.max_tokens == 500

    def test_forwards_all_option_fields(self) -> None:
        src = SimpleStreamOptions(
            temperature=0.7,
            max_tokens=1024,
            cache_retention="long",
            session_id="s",
            headers={"x-foo": "bar"},
            max_retry_delay_ms=5_000,
            metadata={"user_id": "u"},
        )
        opts = build_base_options(_model(), src, api_key="explicit")
        assert opts.temperature == 0.7
        assert opts.cache_retention == "long"
        assert opts.session_id == "s"
        assert opts.headers == {"x-foo": "bar"}
        assert opts.max_retry_delay_ms == 5_000
        assert opts.metadata == {"user_id": "u"}
        assert opts.api_key == "explicit"

    def test_explicit_api_key_overrides_options_api_key(self) -> None:
        src = SimpleStreamOptions(api_key="from-options")
        opts = build_base_options(_model(), src, api_key="explicit")
        assert opts.api_key == "explicit"

    def test_options_api_key_used_when_no_explicit(self) -> None:
        src = SimpleStreamOptions(api_key="from-options")
        opts = build_base_options(_model(), src)
        assert opts.api_key == "from-options"


class TestClampReasoning:
    @pytest.mark.parametrize(
        ("level", "expected"),
        [("minimal", "minimal"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "high")],
    )
    def test_clamp(self, level: str, expected: str) -> None:
        assert clamp_reasoning(level) == expected  # type: ignore[arg-type]

    def test_none_passthrough(self) -> None:
        assert clamp_reasoning(None) is None


class TestAdjustMaxTokensForThinking:
    def test_default_budgets(self) -> None:
        out = adjust_max_tokens_for_thinking(
            base_max_tokens=1000,
            model_max_tokens=32_000,
            reasoning_level="medium",
        )
        # default medium budget is 8192
        assert out["thinking_budget"] == 8192
        assert out["max_tokens"] == 9192  # 1000 + 8192

    def test_clamps_to_model_max_tokens(self) -> None:
        out = adjust_max_tokens_for_thinking(
            base_max_tokens=30_000,
            model_max_tokens=32_000,
            reasoning_level="high",
        )
        assert out["max_tokens"] == 32_000

    def test_xhigh_clamped_to_high(self) -> None:
        out = adjust_max_tokens_for_thinking(
            base_max_tokens=1000,
            model_max_tokens=32_000,
            reasoning_level="xhigh",
        )
        # clamp_reasoning demotes xhigh to high → budget is 16384
        assert out["thinking_budget"] == 16384

    def test_custom_budgets_override_defaults(self) -> None:
        out = adjust_max_tokens_for_thinking(
            base_max_tokens=2_000,
            model_max_tokens=32_000,
            reasoning_level="low",
            custom_budgets=ThinkingBudgets(low=500),
        )
        # 500 is well below the clamp threshold (max_tokens=2500 > 1024 output reserve)
        assert out["thinking_budget"] == 500
        assert out["max_tokens"] == 2_500

    def test_budget_reduced_when_max_tokens_too_small(self) -> None:
        # If clamped max_tokens <= thinking_budget, budget shrinks to leave
        # at least 1024 for output tokens.
        out = adjust_max_tokens_for_thinking(
            base_max_tokens=100,
            model_max_tokens=5_000,
            reasoning_level="high",  # default high = 16384
        )
        assert out["max_tokens"] == 5_000
        assert out["thinking_budget"] == 5_000 - 1024  # room reserved for output
