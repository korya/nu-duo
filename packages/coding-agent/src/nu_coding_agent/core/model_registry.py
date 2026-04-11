"""Model registry — direct port of ``packages/coding-agent/src/core/model-registry.ts``.

Loads built-in models (from :func:`nu_ai.models.get_models` /
:func:`nu_ai.models.get_providers`), merges them with user-defined custom
models loaded from ``<agent_dir>/models.json``, and resolves API keys +
request headers via :class:`AuthStorage`.

The TS upstream uses TypeBox + AJV for ``models.json`` validation. The
Python port replaces that with a hand-rolled validator that mirrors the
exact error messages so existing config files behave identically.

OAuth provider integration is gated behind an optional callable
(:attr:`oauth_modify_models`) so this module doesn't have to wait for the
:mod:`nu_ai.oauth` port. Once OAuth lands, callers can pass
``oauth_modify_models=oauth_registry.apply_modify`` and the registry will
let OAuth providers tweak baseUrl/headers before serving them.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_ai.models import get_models, get_providers
from nu_ai.types import Model, ModelCost

from nu_coding_agent.config import get_agent_dir
from nu_coding_agent.core.resolve_config_value import (
    clear_config_value_cache,
    resolve_config_value_or_throw,
    resolve_config_value_uncached,
    resolve_headers_or_throw,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nu_coding_agent.core.auth_storage import AuthStorage


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProviderOverride:
    """``baseUrl`` / ``compat`` overrides for built-in models of a provider."""

    base_url: str | None = None
    compat: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelOverride:
    """Per-model field overrides applied on top of a built-in model."""

    name: str | None = None
    reasoning: bool | None = None
    input: list[str] | None = None
    cost: dict[str, float] | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None


@dataclass(slots=True)
class ProviderRequestConfig:
    """Per-provider request-time auth/headers config from models.json."""

    api_key: str | None = None
    headers: dict[str, str] | None = None
    auth_header: bool | None = None


@dataclass(slots=True)
class ResolvedRequestAuth:
    """Result of :meth:`ModelRegistry.get_api_key_and_headers`."""

    ok: bool
    api_key: str | None = None
    headers: dict[str, str] | None = None
    error: str | None = None


@dataclass(slots=True)
class ProviderConfigInput:
    """Input for :meth:`ModelRegistry.register_provider` (extension hook)."""

    base_url: str | None = None
    api_key: str | None = None
    api: str | None = None
    headers: dict[str, str] | None = None
    auth_header: bool | None = None
    models: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CustomModelsResult:
    models: list[Model]
    overrides: dict[str, ProviderOverride]
    model_overrides: dict[str, dict[str, ModelOverride]]
    error: str | None


def _empty_custom_result(error: str | None = None) -> _CustomModelsResult:
    return _CustomModelsResult(models=[], overrides={}, model_overrides={}, error=error)


def _merge_compat(
    base: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if override is None:
        return base
    merged: dict[str, Any] = {**(base or {}), **override}
    if (base or {}).get("openRouterRouting") or override.get("openRouterRouting"):
        merged["openRouterRouting"] = {
            **((base or {}).get("openRouterRouting") or {}),
            **(override.get("openRouterRouting") or {}),
        }
    if (base or {}).get("vercelGatewayRouting") or override.get("vercelGatewayRouting"):
        merged["vercelGatewayRouting"] = {
            **((base or {}).get("vercelGatewayRouting") or {}),
            **(override.get("vercelGatewayRouting") or {}),
        }
    return merged


def _apply_model_override(model: Model, override: ModelOverride) -> Model:
    """Return a copy of ``model`` with the override fields applied."""
    data = model.model_dump(by_alias=True)
    if override.name is not None:
        data["name"] = override.name
    if override.reasoning is not None:
        data["reasoning"] = override.reasoning
    if override.input is not None:
        data["input"] = override.input
    if override.context_window is not None:
        data["contextWindow"] = override.context_window
    if override.max_tokens is not None:
        data["maxTokens"] = override.max_tokens
    if override.cost is not None:
        base_cost = data.get("cost") or {}
        data["cost"] = {
            "input": override.cost.get("input", base_cost.get("input", 0)),
            "output": override.cost.get("output", base_cost.get("output", 0)),
            "cacheRead": override.cost.get("cacheRead", base_cost.get("cacheRead", 0)),
            "cacheWrite": override.cost.get("cacheWrite", base_cost.get("cacheWrite", 0)),
        }
    if override.compat is not None or "compat" in data:
        merged = _merge_compat(data.get("compat"), override.compat)
        if merged is not None:
            data["compat"] = merged
    return Model.model_validate(data)


def _parse_provider_override(provider_config: dict[str, Any]) -> ProviderOverride | None:
    base_url = provider_config.get("baseUrl")
    compat = provider_config.get("compat")
    if base_url is None and compat is None:
        return None
    return ProviderOverride(base_url=base_url, compat=compat)


def _parse_model_overrides(provider_config: dict[str, Any]) -> dict[str, ModelOverride]:
    raw = provider_config.get("modelOverrides") or {}
    out: dict[str, ModelOverride] = {}
    for model_id, override_dict in raw.items():
        out[model_id] = ModelOverride(
            name=override_dict.get("name"),
            reasoning=override_dict.get("reasoning"),
            input=override_dict.get("input"),
            cost=override_dict.get("cost"),
            context_window=override_dict.get("contextWindow"),
            max_tokens=override_dict.get("maxTokens"),
            headers=override_dict.get("headers"),
            compat=override_dict.get("compat"),
        )
    return out


def _validate_models_config(config: dict[str, Any]) -> None:
    """Apply the upstream's structural checks to a parsed ``models.json``."""
    providers = config.get("providers")
    if not isinstance(providers, dict):
        raise ValueError("models.json: top-level `providers` must be an object")
    for provider_name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            raise ValueError(f"Provider {provider_name}: config must be an object")
        has_provider_api = bool(provider_config.get("api"))
        models = provider_config.get("models") or []
        has_model_overrides = bool(provider_config.get("modelOverrides"))
        if not models:
            if not provider_config.get("baseUrl") and not provider_config.get("compat") and not has_model_overrides:
                raise ValueError(
                    f'Provider {provider_name}: must specify "baseUrl", "compat", "modelOverrides", or "models".'
                )
        else:
            if not provider_config.get("baseUrl"):
                raise ValueError(f'Provider {provider_name}: "baseUrl" is required when defining custom models.')
            if not provider_config.get("apiKey"):
                raise ValueError(f'Provider {provider_name}: "apiKey" is required when defining custom models.')
        for model_def in models:
            if not isinstance(model_def, dict):
                raise ValueError(f"Provider {provider_name}: model entries must be objects")
            has_model_api = bool(model_def.get("api"))
            if not has_provider_api and not has_model_api:
                raise ValueError(
                    f"Provider {provider_name}, model {model_def.get('id')}: "
                    f'no "api" specified. Set at provider or model level.'
                )
            if not model_def.get("id"):
                raise ValueError(f'Provider {provider_name}: model missing "id"')
            ctx_window = model_def.get("contextWindow")
            if ctx_window is not None and ctx_window <= 0:
                raise ValueError(f"Provider {provider_name}, model {model_def.get('id')}: invalid contextWindow")
            max_tokens = model_def.get("maxTokens")
            if max_tokens is not None and max_tokens <= 0:
                raise ValueError(f"Provider {provider_name}, model {model_def.get('id')}: invalid maxTokens")


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


# Re-exported for parity with the upstream ``clearApiKeyCache`` symbol.
clear_api_key_cache = clear_config_value_cache


class ModelRegistry:
    """Registry of built-in + custom models, with API key resolution."""

    def __init__(
        self,
        auth_storage: AuthStorage,
        models_json_path: str | None,
        *,
        oauth_modify_models: Callable[[list[Model]], list[Model]] | None = None,
    ) -> None:
        self._auth_storage = auth_storage
        self._models_json_path = models_json_path
        self._oauth_modify_models = oauth_modify_models
        self._models: list[Model] = []
        self._provider_request_configs: dict[str, ProviderRequestConfig] = {}
        self._model_request_headers: dict[str, dict[str, str]] = {}
        self._registered_providers: dict[str, ProviderConfigInput] = {}
        self._load_error: str | None = None
        self._load_models()

    @property
    def auth_storage(self) -> AuthStorage:
        return self._auth_storage

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        auth_storage: AuthStorage,
        models_json_path: str | None = None,
        *,
        oauth_modify_models: Callable[[list[Model]], list[Model]] | None = None,
    ) -> ModelRegistry:
        path = models_json_path or str(Path(get_agent_dir()) / "models.json")
        return cls(auth_storage, path, oauth_modify_models=oauth_modify_models)

    @classmethod
    def in_memory(
        cls,
        auth_storage: AuthStorage,
        *,
        oauth_modify_models: Callable[[list[Model]], list[Model]] | None = None,
    ) -> ModelRegistry:
        return cls(auth_storage, None, oauth_modify_models=oauth_modify_models)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload models from disk and reapply registered dynamic providers."""
        self._provider_request_configs.clear()
        self._model_request_headers.clear()
        self._load_error = None
        self._load_models()
        for provider_name, config in list(self._registered_providers.items()):
            self._apply_provider_config(provider_name, config)

    def get_error(self) -> str | None:
        return self._load_error

    def _load_models(self) -> None:
        custom = self._load_custom_models(self._models_json_path) if self._models_json_path else _empty_custom_result()
        if custom.error is not None:
            self._load_error = custom.error
        built_in = self._load_built_in_models(custom.overrides, custom.model_overrides)
        combined = self._merge_custom_models(built_in, custom.models)
        if self._oauth_modify_models is not None:
            combined = self._oauth_modify_models(combined)
        self._models = combined

    def _load_built_in_models(
        self,
        overrides: dict[str, ProviderOverride],
        model_overrides: dict[str, dict[str, ModelOverride]],
    ) -> list[Model]:
        out: list[Model] = []
        for provider in get_providers():
            provider_override = overrides.get(provider)
            per_model_overrides = model_overrides.get(provider, {})
            for model in get_models(provider):
                current = model
                if provider_override is not None:
                    data = current.model_dump(by_alias=True)
                    if provider_override.base_url is not None:
                        data["baseUrl"] = provider_override.base_url
                    if provider_override.compat is not None:
                        data["compat"] = _merge_compat(data.get("compat"), provider_override.compat)
                    current = Model.model_validate(data)
                if model.id in per_model_overrides:
                    current = _apply_model_override(current, per_model_overrides[model.id])
                out.append(current)
        return out

    @staticmethod
    def _merge_custom_models(built_in: list[Model], custom: list[Model]) -> list[Model]:
        merged = list(built_in)
        for custom_model in custom:
            existing_index = next(
                (i for i, m in enumerate(merged) if m.provider == custom_model.provider and m.id == custom_model.id),
                None,
            )
            if existing_index is not None:
                merged[existing_index] = custom_model
            else:
                merged.append(custom_model)
        return merged

    def _load_custom_models(self, models_json_path: str) -> _CustomModelsResult:
        path = Path(models_json_path)
        if not path.exists():
            return _empty_custom_result()
        try:
            content = path.read_text(encoding="utf-8")
            config = json.loads(content)
        except json.JSONDecodeError as exc:
            return _empty_custom_result(f"Failed to parse models.json: {exc.msg}\n\nFile: {models_json_path}")
        except OSError as exc:
            return _empty_custom_result(f"Failed to load models.json: {exc}\n\nFile: {models_json_path}")
        try:
            _validate_models_config(config)
        except ValueError as exc:
            return _empty_custom_result(f"Invalid models.json: {exc}\n\nFile: {models_json_path}")
        overrides: dict[str, ProviderOverride] = {}
        model_overrides: dict[str, dict[str, ModelOverride]] = {}
        for provider_name, provider_config in config["providers"].items():
            override = _parse_provider_override(provider_config)
            if override is not None:
                overrides[provider_name] = override
            self._store_provider_request_config(provider_name, provider_config)
            per_model = _parse_model_overrides(provider_config)
            if per_model:
                model_overrides[provider_name] = per_model
                for model_id, model_override in per_model.items():
                    self._store_model_headers(provider_name, model_id, model_override.headers)
        return _CustomModelsResult(
            models=self._parse_models(config),
            overrides=overrides,
            model_overrides=model_overrides,
            error=None,
        )

    def _parse_models(self, config: dict[str, Any]) -> list[Model]:
        out: list[Model] = []
        for provider_name, provider_config in config["providers"].items():
            model_defs = provider_config.get("models") or []
            if not model_defs:
                continue
            for model_def in model_defs:
                api = model_def.get("api") or provider_config.get("api")
                if api is None:
                    continue
                compat = _merge_compat(provider_config.get("compat"), model_def.get("compat"))
                self._store_model_headers(provider_name, model_def["id"], model_def.get("headers"))
                cost_dict = model_def.get("cost") or {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                }
                out.append(
                    Model(
                        id=model_def["id"],
                        name=model_def.get("name") or model_def["id"],
                        api=api,  # type: ignore[arg-type]
                        provider=provider_name,
                        base_url=model_def.get("baseUrl") or provider_config["baseUrl"],
                        reasoning=bool(model_def.get("reasoning", False)),
                        input=model_def.get("input") or ["text"],
                        cost=ModelCost(
                            input=cost_dict["input"],
                            output=cost_dict["output"],
                            cache_read=cost_dict["cacheRead"],
                            cache_write=cost_dict["cacheWrite"],
                        ),
                        context_window=model_def.get("contextWindow", 128000),
                        max_tokens=model_def.get("maxTokens", 16384),
                        headers=None,
                        compat=compat,  # type: ignore[arg-type]
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    def get_all(self) -> list[Model]:
        return list(self._models)

    def get_available(self) -> list[Model]:
        return [m for m in self._models if self.has_configured_auth(m)]

    def find(self, provider: str, model_id: str) -> Model | None:
        return next((m for m in self._models if m.provider == provider and m.id == model_id), None)

    def has_configured_auth(self, model: Model) -> bool:
        if self._auth_storage.has_auth(model.provider):
            return True
        config = self._provider_request_configs.get(model.provider)
        return config is not None and config.api_key is not None

    # ------------------------------------------------------------------
    # Per-provider request config
    # ------------------------------------------------------------------

    @staticmethod
    def _model_request_key(provider: str, model_id: str) -> str:
        return f"{provider}:{model_id}"

    def _store_provider_request_config(self, provider_name: str, config: dict[str, Any]) -> None:
        if not config.get("apiKey") and not config.get("headers") and not config.get("authHeader"):
            return
        self._provider_request_configs[provider_name] = ProviderRequestConfig(
            api_key=config.get("apiKey"),
            headers=config.get("headers"),
            auth_header=config.get("authHeader"),
        )

    def _store_model_headers(
        self,
        provider_name: str,
        model_id: str,
        headers: dict[str, str] | None,
    ) -> None:
        key = self._model_request_key(provider_name, model_id)
        if not headers:
            self._model_request_headers.pop(key, None)
            return
        self._model_request_headers[key] = headers

    # ------------------------------------------------------------------
    # API key resolution
    # ------------------------------------------------------------------

    async def get_api_key_and_headers(self, model: Model) -> ResolvedRequestAuth:
        try:
            provider_config = self._provider_request_configs.get(model.provider)
            api_key_from_storage = await self._auth_storage.get_api_key(model.provider, include_fallback=False)
            api_key = api_key_from_storage
            if api_key is None and provider_config is not None and provider_config.api_key:
                api_key = resolve_config_value_or_throw(
                    provider_config.api_key,
                    f'API key for provider "{model.provider}"',
                )
            provider_headers = (
                resolve_headers_or_throw(provider_config.headers, f'provider "{model.provider}"')
                if provider_config is not None
                else None
            )
            model_headers = resolve_headers_or_throw(
                self._model_request_headers.get(self._model_request_key(model.provider, model.id)),
                f'model "{model.provider}/{model.id}"',
            )
            headers: dict[str, str] | None
            if model.headers or provider_headers or model_headers:
                headers = {**(model.headers or {}), **(provider_headers or {}), **(model_headers or {})}
            else:
                headers = None
            if provider_config is not None and provider_config.auth_header:
                if api_key is None:
                    return ResolvedRequestAuth(
                        ok=False,
                        error=f'No API key found for "{model.provider}"',
                    )
                headers = {**(headers or {}), "Authorization": f"Bearer {api_key}"}
            return ResolvedRequestAuth(
                ok=True,
                api_key=api_key,
                headers=headers if headers else None,
            )
        except ValueError as exc:
            return ResolvedRequestAuth(ok=False, error=str(exc))

    async def get_api_key_for_provider(self, provider: str) -> str | None:
        api_key = await self._auth_storage.get_api_key(provider, include_fallback=False)
        if api_key is not None:
            return api_key
        config = self._provider_request_configs.get(provider)
        if config is None or config.api_key is None:
            return None
        return resolve_config_value_uncached(config.api_key)

    def is_using_oauth(self, model: Model) -> bool:
        from nu_coding_agent.core.auth_storage import OAuthCredential  # noqa: PLC0415

        cred = self._auth_storage.get(model.provider)
        return isinstance(cred, OAuthCredential)

    # ------------------------------------------------------------------
    # Dynamic provider registration
    # ------------------------------------------------------------------

    def register_provider(self, provider_name: str, config: ProviderConfigInput) -> None:
        self._validate_provider_config(provider_name, config)
        self._apply_provider_config(provider_name, config)
        self._registered_providers[provider_name] = config

    def unregister_provider(self, provider_name: str) -> None:
        if provider_name not in self._registered_providers:
            return
        del self._registered_providers[provider_name]
        self.refresh()

    @staticmethod
    def _validate_provider_config(provider_name: str, config: ProviderConfigInput) -> None:
        if not config.models:
            return
        if not config.base_url:
            raise ValueError(f'Provider {provider_name}: "baseUrl" is required when defining models.')
        if not config.api_key:
            raise ValueError(f'Provider {provider_name}: "apiKey" is required when defining models.')
        for model_def in config.models:
            api = model_def.get("api") or config.api
            if not api:
                raise ValueError(f'Provider {provider_name}, model {model_def.get("id")}: no "api" specified.')

    def _apply_provider_config(self, provider_name: str, config: ProviderConfigInput) -> None:
        request_config_dict = {
            "apiKey": config.api_key,
            "headers": config.headers,
            "authHeader": config.auth_header,
        }
        self._store_provider_request_config(provider_name, request_config_dict)
        if config.models:
            self._models = [m for m in self._models if m.provider != provider_name]
            for model_def in config.models:
                api = model_def.get("api") or config.api
                if api is None:
                    continue
                self._store_model_headers(provider_name, model_def["id"], model_def.get("headers"))
                cost_dict = model_def.get("cost") or {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                }
                self._models.append(
                    Model(
                        id=model_def["id"],
                        name=model_def.get("name") or model_def["id"],
                        api=api,  # type: ignore[arg-type]
                        provider=provider_name,
                        base_url=config.base_url or "",
                        reasoning=bool(model_def.get("reasoning", False)),
                        input=model_def.get("input") or ["text"],
                        cost=ModelCost(
                            input=cost_dict["input"],
                            output=cost_dict["output"],
                            cache_read=cost_dict["cacheRead"],
                            cache_write=cost_dict["cacheWrite"],
                        ),
                        context_window=model_def.get("contextWindow", 128000),
                        max_tokens=model_def.get("maxTokens", 16384),
                        headers=None,
                        compat=model_def.get("compat"),
                    )
                )
        elif config.base_url or config.headers:
            updated: list[Model] = []
            for model in self._models:
                if model.provider != provider_name:
                    updated.append(model)
                    continue
                copied = copy.deepcopy(model)
                if config.base_url:
                    copied.base_url = config.base_url
                updated.append(copied)
            self._models = updated


__all__ = [
    "ModelOverride",
    "ModelRegistry",
    "ProviderConfigInput",
    "ProviderOverride",
    "ProviderRequestConfig",
    "ResolvedRequestAuth",
    "clear_api_key_cache",
]
