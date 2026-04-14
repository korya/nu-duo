"""Tests for ``nu_coding_agent.list_models``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from nu_coding_agent.list_models import _format_token_count, list_models

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeModel:
    id: str
    name: str
    provider: str
    context_window: int
    max_tokens: int
    reasoning: bool = False
    input: list[str] = field(default_factory=lambda: ["text"])
    api: str = "completions"
    base_url: str = "https://example.com"
    cost: Any = None


def _make_registry(models: list[FakeModel]) -> MagicMock:
    registry = MagicMock()
    registry.get_available.return_value = models
    return registry


# ---------------------------------------------------------------------------
# _format_token_count
# ---------------------------------------------------------------------------


class TestFormatTokenCount:
    def test_millions(self) -> None:
        assert _format_token_count(1_000_000) == "1M"

    def test_millions_fractional(self) -> None:
        assert _format_token_count(1_500_000) == "1.5M"

    def test_thousands(self) -> None:
        assert _format_token_count(200_000) == "200K"

    def test_thousands_fractional(self) -> None:
        assert _format_token_count(4_500) == "4.5K"

    def test_small(self) -> None:
        assert _format_token_count(500) == "500"

    def test_exact_thousand(self) -> None:
        assert _format_token_count(1_000) == "1K"


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_no_models(self, capsys: Any) -> None:
        registry = _make_registry([])
        list_models(registry)
        out = capsys.readouterr().out
        assert "No models available" in out

    def test_basic_output(self, capsys: Any) -> None:
        models = [
            FakeModel(
                id="gpt-4",
                name="GPT-4",
                provider="openai",
                context_window=128_000,
                max_tokens=4096,
                reasoning=True,
                input=["text", "image"],
            ),
        ]
        registry = _make_registry(models)
        list_models(registry)
        out = capsys.readouterr().out
        assert "openai" in out
        assert "gpt-4" in out
        assert "128K" in out
        assert "yes" in out  # thinking and images

    def test_multiple_models_sorted(self, capsys: Any) -> None:
        models = [
            FakeModel(id="z-model", name="Z", provider="b-provider", context_window=1000, max_tokens=500),
            FakeModel(id="a-model", name="A", provider="a-provider", context_window=2000, max_tokens=1000),
        ]
        registry = _make_registry(models)
        list_models(registry)
        out = capsys.readouterr().out
        lines = [line for line in out.strip().splitlines() if line.strip()]
        # Header + 2 rows
        assert len(lines) == 3
        # a-provider should come first
        assert lines[1].strip().startswith("a-provider")

    def test_search_filter(self, capsys: Any) -> None:
        models = [
            FakeModel(id="claude-3", name="Claude", provider="anthropic", context_window=200_000, max_tokens=4096),
            FakeModel(id="gpt-4", name="GPT-4", provider="openai", context_window=128_000, max_tokens=4096),
        ]
        registry = _make_registry(models)
        list_models(registry, search="claude")
        out = capsys.readouterr().out
        assert "claude" in out
        # openai/gpt-4 should be filtered out
        assert "gpt-4" not in out

    def test_search_no_match(self, capsys: Any) -> None:
        models = [
            FakeModel(id="gpt-4", name="GPT-4", provider="openai", context_window=128_000, max_tokens=4096),
        ]
        registry = _make_registry(models)
        list_models(registry, search="zzzzz-no-match")
        out = capsys.readouterr().out
        assert "No models matching" in out

    def test_no_thinking_no_images(self, capsys: Any) -> None:
        models = [
            FakeModel(
                id="basic",
                name="Basic",
                provider="test",
                context_window=4_000,
                max_tokens=1_000,
                reasoning=False,
                input=["text"],
            ),
        ]
        registry = _make_registry(models)
        list_models(registry)
        out = capsys.readouterr().out
        lines = out.strip().splitlines()
        data_line = lines[1]
        # Both thinking and images should be "no"
        assert data_line.count("no") >= 2
