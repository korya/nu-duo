"""Tests for pi_ai.api_registry.

Ported from the documented contract in
``packages/ai/src/api-registry.ts`` — no dedicated upstream test file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pi_ai.api_registry import (
    ApiProvider,
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)
from pi_ai.providers.register_builtins import register_builtin_providers
from pi_ai.types import Context, Model, ModelCost
from pi_ai.utils.event_stream import AssistantMessageEventStream

if TYPE_CHECKING:
    from collections.abc import Iterator


def _mk_model(api: str = "anthropic-messages", provider: str = "anthropic") -> Model:
    return Model(
        id="x",
        name="X",
        api=api,
        provider=provider,
        base_url="https://example.test",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=1000,
        max_tokens=100,
    )


def _mk_provider(api: str = "anthropic-messages") -> ApiProvider:
    def stream(model: Model, context: Context, options: object = None) -> AssistantMessageEventStream:
        s = AssistantMessageEventStream()
        s.end()
        return s

    def stream_simple(model: Model, context: Context, options: object = None) -> AssistantMessageEventStream:
        s = AssistantMessageEventStream()
        s.end()
        return s

    return ApiProvider(api=api, stream=stream, stream_simple=stream_simple)


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    # Clear before each test so tests start with a known-empty registry, then
    # restore the built-in providers on teardown so downstream test modules
    # in the same pytest session don't see a wiped registry.
    clear_api_providers()
    yield
    clear_api_providers()
    register_builtin_providers()


class TestRegisterAndLookup:
    def test_register_and_get(self) -> None:
        register_api_provider(_mk_provider("anthropic-messages"))
        entry = get_api_provider("anthropic-messages")
        assert entry is not None
        assert entry.api == "anthropic-messages"

    def test_get_unknown_returns_none(self) -> None:
        assert get_api_provider("no-such-api") is None

    def test_get_all(self) -> None:
        register_api_provider(_mk_provider("anthropic-messages"))
        register_api_provider(_mk_provider("openai-completions"))
        all_apis = sorted(p.api for p in get_api_providers())
        assert all_apis == ["anthropic-messages", "openai-completions"]


class TestApiMismatchGuard:
    def test_stream_rejects_mismatched_model(self) -> None:
        register_api_provider(_mk_provider("anthropic-messages"))
        entry = get_api_provider("anthropic-messages")
        assert entry is not None
        mismatched = _mk_model(api="openai-completions")
        with pytest.raises(ValueError, match="Mismatched api"):
            entry.stream(mismatched, Context(messages=[]))

    def test_stream_accepts_matching_model(self) -> None:
        register_api_provider(_mk_provider("anthropic-messages"))
        entry = get_api_provider("anthropic-messages")
        assert entry is not None
        matching = _mk_model(api="anthropic-messages")
        # Should not raise.
        entry.stream(matching, Context(messages=[]))


class TestUnregister:
    def test_unregister_by_source_id(self) -> None:
        register_api_provider(_mk_provider("anthropic-messages"), source_id="ext-a")
        register_api_provider(_mk_provider("openai-completions"), source_id="ext-b")
        unregister_api_providers("ext-a")
        assert get_api_provider("anthropic-messages") is None
        assert get_api_provider("openai-completions") is not None

    def test_unregister_unknown_source_is_noop(self) -> None:
        register_api_provider(_mk_provider("anthropic-messages"), source_id="ext-a")
        unregister_api_providers("ext-unknown")
        assert get_api_provider("anthropic-messages") is not None


class TestClear:
    def test_clear_removes_all(self) -> None:
        register_api_provider(_mk_provider("anthropic-messages"))
        register_api_provider(_mk_provider("openai-completions"))
        clear_api_providers()
        assert get_api_providers() == []


class TestReRegistration:
    def test_re_registering_same_api_replaces_prior(self) -> None:
        register_api_provider(_mk_provider("anthropic-messages"), source_id="first")
        register_api_provider(_mk_provider("anthropic-messages"), source_id="second")
        # Only one entry for this API.
        assert len(get_api_providers()) == 1
        # Only "second" should remain after unregistering "first".
        unregister_api_providers("first")
        assert get_api_provider("anthropic-messages") is not None
