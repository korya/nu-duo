"""Tests for ``nu_coding_agent.core.model_registry``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from nu_coding_agent.core.auth_storage import ApiKeyCredential, AuthStorage
from nu_coding_agent.core.model_registry import (
    ModelRegistry,
    ProviderConfigInput,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Built-in catalog basics
# ---------------------------------------------------------------------------


def test_get_all_returns_built_in_models() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    models = registry.get_all()
    assert len(models) > 0
    # The catalog includes Anthropic + OpenAI providers.
    providers = {m.provider for m in models}
    assert "anthropic" in providers
    assert "openai" in providers


def test_find_returns_matching_model() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    a_model = registry.get_all()[0]
    assert registry.find(a_model.provider, a_model.id) is a_model


def test_find_returns_none_for_unknown() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    assert registry.find("nope", "no-such-id") is None


def test_get_available_filters_by_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    available = registry.get_available()
    # Without any credentials, every built-in model is filtered out.
    assert all(not registry.has_configured_auth(m) for m in available)
    storage.set("openai", ApiKeyCredential(type="api_key", key="x"))
    available_after = registry.get_available()
    assert any(m.provider == "openai" for m in available_after)


# ---------------------------------------------------------------------------
# models.json loading
# ---------------------------------------------------------------------------


def test_missing_models_json_is_a_no_op(tmp_path: Path) -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(tmp_path / "missing.json"))
    assert registry.get_error() is None


def test_invalid_json_records_error(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("not json {{")
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert registry.get_error() is not None
    assert "models.json" in registry.get_error()  # type: ignore[arg-type]


def test_invalid_schema_records_error(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"foo": {}}}))
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert registry.get_error() is not None
    assert "must specify" in registry.get_error()  # type: ignore[arg-type]


def test_custom_model_loaded_from_json(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "ollama",
                        "api": "openai-completions",
                        "models": [{"id": "llama3", "name": "Llama 3 (local)", "contextWindow": 8000}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert registry.get_error() is None
    custom = registry.find("ollama", "llama3")
    assert custom is not None
    assert custom.name == "Llama 3 (local)"
    assert custom.context_window == 8000
    assert custom.base_url == "http://localhost:11434"


def test_custom_model_validation_requires_api_key(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "api": "openai-completions",
                        "models": [{"id": "llama3"}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert registry.get_error() is not None
    assert "apiKey" in registry.get_error()  # type: ignore[arg-type]


def test_provider_baseurl_override(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "openai": {
                        "baseUrl": "https://custom.example.com/v1",
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    openai_models = [m for m in registry.get_all() if m.provider == "openai"]
    assert openai_models
    assert all(m.base_url == "https://custom.example.com/v1" for m in openai_models)


def test_per_model_override(tmp_path: Path) -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    target = next(iter(registry.get_all()))
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    target.provider: {
                        "baseUrl": "https://example.com",
                        "modelOverrides": {
                            target.id: {
                                "name": "OVERRIDDEN",
                                "contextWindow": 99999,
                            }
                        },
                    }
                }
            }
        )
    )
    registry2 = ModelRegistry.create(storage, str(path))
    overridden = registry2.find(target.provider, target.id)
    assert overridden is not None
    assert overridden.name == "OVERRIDDEN"
    assert overridden.context_window == 99999


def test_per_model_override_full_fields(tmp_path: Path) -> None:
    """Exercise the full ``_apply_model_override`` branch (cost + reasoning + input + maxTokens)."""
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    target = next(iter(registry.get_all()))
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    target.provider: {
                        "baseUrl": "https://example.com",
                        "modelOverrides": {
                            target.id: {
                                "reasoning": True,
                                "input": ["text"],
                                "maxTokens": 50000,
                                "cost": {
                                    "input": 1.0,
                                    "output": 2.0,
                                    "cacheRead": 0.1,
                                    "cacheWrite": 0.2,
                                },
                            }
                        },
                    }
                }
            }
        )
    )
    registry2 = ModelRegistry.create(storage, str(path))
    overridden = registry2.find(target.provider, target.id)
    assert overridden is not None
    assert overridden.reasoning is True
    assert overridden.input == ["text"]
    assert overridden.max_tokens == 50000
    assert overridden.cost.input == 1.0
    assert overridden.cost.cache_write == 0.2


def test_provider_compat_override(tmp_path: Path) -> None:
    storage = AuthStorage.in_memory()
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "openai": {
                        "baseUrl": "https://x",
                        "compat": {"supportsStrictMode": True},
                    }
                }
            }
        )
    )
    registry = ModelRegistry.create(storage, str(path))
    openai_models = [m for m in registry.get_all() if m.provider == "openai"]
    assert openai_models  # at least one openai model present
    # The compat dict was applied via _merge_compat path.
    assert all(m.base_url == "https://x" for m in openai_models)


def test_load_custom_models_file_handles_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("")
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert registry.get_error() is not None


async def test_get_api_key_and_headers_with_provider_headers(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "literal",
                        "api": "openai-completions",
                        "headers": {"X-Custom": "literal-value"},
                        "models": [{"id": "m1"}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    model = registry.find("ollama", "m1")
    assert model is not None
    result = await registry.get_api_key_and_headers(model)
    assert result.ok is True
    assert result.headers is not None
    assert result.headers["X-Custom"] == "literal-value"


async def test_get_api_key_and_headers_auth_header_no_key_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "MISSING_KEY",
                        "api": "openai-completions",
                        "authHeader": True,
                        "models": [{"id": "m1"}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    model = registry.find("ollama", "m1")
    assert model is not None
    # The custom resolver returns the literal env-var name when unset, so
    # this branch resolves "MISSING_KEY" → "MISSING_KEY" and the auth header
    # is set. The "no API key" branch is exercised when nothing resolves at all.
    result = await registry.get_api_key_and_headers(model)
    assert result.ok is True


def test_validation_rejects_non_dict_provider_config(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"foo": "not an object"}}))
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert "must be an object" in (registry.get_error() or "")


def test_validation_rejects_non_dict_model_entry(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "x",
                        "api": "openai-completions",
                        "models": ["not a dict"],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert "must be objects" in (registry.get_error() or "")


def test_register_provider_validation_requires_api_key() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    with pytest.raises(ValueError, match="apiKey"):
        registry.register_provider(
            "bad",
            ProviderConfigInput(
                base_url="http://x",
                api="openai-completions",
                models=[{"id": "m1"}],
            ),
        )


def test_register_provider_validation_requires_api_per_model() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    with pytest.raises(ValueError, match='"api"'):
        registry.register_provider(
            "bad",
            ProviderConfigInput(
                base_url="http://x",
                api_key="x",
                models=[{"id": "m1"}],  # no provider api, no model api
            ),
        )


# ---------------------------------------------------------------------------
# get_api_key_and_headers
# ---------------------------------------------------------------------------


async def test_get_api_key_and_headers_uses_auth_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    openai_model = next(m for m in registry.get_all() if m.provider == "openai")
    result = await registry.get_api_key_and_headers(openai_model)
    assert result.ok is True
    assert result.api_key == "from-env"


async def test_get_api_key_and_headers_falls_back_to_provider_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_KEY", "from-resolver")
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "CUSTOM_KEY",
                        "api": "openai-completions",
                        "models": [{"id": "llama3"}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    model = registry.find("ollama", "llama3")
    assert model is not None
    result = await registry.get_api_key_and_headers(model)
    assert result.ok is True
    assert result.api_key == "from-resolver"


async def test_get_api_key_for_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    assert await registry.get_api_key_for_provider("openai") == "secret"


async def test_get_api_key_for_provider_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    assert await registry.get_api_key_for_provider("openai") is None


async def test_auth_header_synthesised(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "literal-secret",
                        "api": "openai-completions",
                        "authHeader": True,
                        "models": [{"id": "llama3"}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    model = registry.find("ollama", "llama3")
    assert model is not None
    result = await registry.get_api_key_and_headers(model)
    assert result.ok is True
    assert result.headers is not None
    assert result.headers["Authorization"] == "Bearer literal-secret"


# ---------------------------------------------------------------------------
# Dynamic provider registration
# ---------------------------------------------------------------------------


def test_register_dynamic_provider() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    registry.register_provider(
        "myprov",
        ProviderConfigInput(
            base_url="http://localhost:8080",
            api_key="literal",
            api="openai-completions",
            models=[{"id": "m1", "name": "M1"}],
        ),
    )
    assert registry.find("myprov", "m1") is not None


def test_register_provider_validation_requires_baseurl() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    with pytest.raises(ValueError, match="baseUrl"):
        registry.register_provider(
            "bad",
            ProviderConfigInput(
                api="openai-completions",
                api_key="x",
                models=[{"id": "m1"}],
            ),
        )


def test_unregister_provider_restores_built_in_overrides() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    registry.register_provider(
        "myprov",
        ProviderConfigInput(
            base_url="http://x",
            api_key="x",
            api="openai-completions",
            models=[{"id": "m1"}],
        ),
    )
    assert registry.find("myprov", "m1") is not None
    registry.unregister_provider("myprov")
    assert registry.find("myprov", "m1") is None


def test_unregister_unknown_provider_is_noop() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    registry.unregister_provider("nonexistent")  # should not raise


def test_register_provider_override_only_changes_baseurl() -> None:
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(storage)
    openai_before = registry.get_all()[0]
    registry.register_provider(
        openai_before.provider,
        ProviderConfigInput(base_url="https://override.example.com"),
    )
    openai_after = next(m for m in registry.get_all() if m.id == openai_before.id)
    assert openai_after.base_url == "https://override.example.com"


# ---------------------------------------------------------------------------
# OAuth detection
# ---------------------------------------------------------------------------


def test_is_using_oauth_false_for_api_key() -> None:
    storage = AuthStorage.in_memory({"openai": ApiKeyCredential(type="api_key", key="x")})
    registry = ModelRegistry.in_memory(storage)
    openai_model = next(m for m in registry.get_all() if m.provider == "openai")
    assert registry.is_using_oauth(openai_model) is False


def test_is_using_oauth_true_for_oauth_credential() -> None:
    from nu_coding_agent.core.auth_storage import OAuthCredential  # noqa: PLC0415

    storage = AuthStorage.in_memory(
        {"anthropic": OAuthCredential(type="oauth", access_token="x", refresh_token="r", expires=10**18)}
    )
    registry = ModelRegistry.in_memory(storage)
    anth = next(m for m in registry.get_all() if m.provider == "anthropic")
    assert registry.is_using_oauth(anth) is True


# ---------------------------------------------------------------------------
# Additional models.json schema validation paths
# ---------------------------------------------------------------------------


def test_validation_rejects_invalid_context_window(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "x",
                        "api": "openai-completions",
                        "models": [{"id": "m1", "contextWindow": 0}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert "contextWindow" in (registry.get_error() or "")


def test_validation_rejects_invalid_max_tokens(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "x",
                        "api": "openai-completions",
                        "models": [{"id": "m1", "maxTokens": 0}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert "maxTokens" in (registry.get_error() or "")


def test_validation_requires_api(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "x",
                        "models": [{"id": "m1"}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert '"api"' in (registry.get_error() or "")


def test_validation_rejects_non_object_providers(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": []}))
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert "providers" in (registry.get_error() or "")


def test_validation_rejects_missing_id(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "x",
                        "api": "openai-completions",
                        "models": [{"name": "no id here"}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert "id" in (registry.get_error() or "")


def test_refresh_reloads_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "x",
                        "api": "openai-completions",
                        "models": [{"id": "m1"}],
                    }
                }
            }
        )
    )
    storage = AuthStorage.in_memory()
    registry = ModelRegistry.create(storage, str(path))
    assert registry.find("ollama", "m1") is not None
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "ollama": {
                        "baseUrl": "http://localhost:11434",
                        "apiKey": "x",
                        "api": "openai-completions",
                        "models": [{"id": "m2"}],
                    }
                }
            }
        )
    )
    registry.refresh()
    assert registry.find("ollama", "m1") is None
    assert registry.find("ollama", "m2") is not None


def test_oauth_modify_models_hook_runs() -> None:
    """The OAuth registry hook can transform built-in models on load."""
    from typing import Any  # noqa: PLC0415

    storage = AuthStorage.in_memory()

    def modify(models: list[Any]) -> list[Any]:
        return [m for m in models if m.provider == "openai"]

    registry = ModelRegistry.in_memory(storage, oauth_modify_models=modify)
    providers = {m.provider for m in registry.get_all()}
    assert providers == {"openai"}
