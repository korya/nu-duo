"""Tests for ``nu_coding_agent.core.auth_storage``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nu_coding_agent.core.auth_storage import (
    ApiKeyCredential,
    AuthCredential,
    AuthStorage,
    FileAuthStorageBackend,
    OAuthCredential,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# In-memory backend basics
# ---------------------------------------------------------------------------


def test_in_memory_set_get_remove() -> None:
    storage = AuthStorage.in_memory()
    storage.set("openai", ApiKeyCredential(type="api_key", key="sk-foo"))
    cred = storage.get("openai")
    assert isinstance(cred, ApiKeyCredential)
    assert cred.key == "sk-foo"
    storage.remove("openai")
    assert storage.get("openai") is None


def test_in_memory_seed_data() -> None:
    seed: dict[str, AuthCredential] = {"anthropic": ApiKeyCredential(type="api_key", key="sk-bar")}
    storage = AuthStorage.in_memory(seed)
    cred = storage.get("anthropic")
    assert isinstance(cred, ApiKeyCredential)
    assert cred.key == "sk-bar"


def test_list_and_has() -> None:
    storage = AuthStorage.in_memory({"openai": ApiKeyCredential(type="api_key", key="x")})
    assert "openai" in storage.list()
    assert storage.has("openai") is True
    assert storage.has("anthropic") is False


def test_get_all_returns_copy() -> None:
    storage = AuthStorage.in_memory({"openai": ApiKeyCredential(type="api_key", key="x")})
    snapshot = storage.get_all()
    snapshot.pop("openai")
    assert "openai" in storage.list()  # internal state untouched


# ---------------------------------------------------------------------------
# Runtime overrides + fallback
# ---------------------------------------------------------------------------


async def test_runtime_override_takes_priority() -> None:
    storage = AuthStorage.in_memory({"openai": ApiKeyCredential(type="api_key", key="from-disk")})
    storage.set_runtime_api_key("openai", "from-cli")
    assert await storage.get_api_key("openai") == "from-cli"
    storage.remove_runtime_api_key("openai")
    assert await storage.get_api_key("openai") == "from-disk"


async def test_resolves_env_var_when_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
    storage = AuthStorage.in_memory()
    assert await storage.get_api_key("openai") == "env-secret"


async def test_fallback_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    storage = AuthStorage.in_memory()
    storage.set_fallback_resolver(lambda provider: "custom-key" if provider == "custom" else None)
    assert await storage.get_api_key("custom") == "custom-key"
    assert await storage.get_api_key("custom", include_fallback=False) is None


def test_has_auth_runtime_override() -> None:
    storage = AuthStorage.in_memory()
    storage.set_runtime_api_key("openai", "x")
    assert storage.has_auth("openai") is True


def test_has_auth_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    storage = AuthStorage.in_memory()
    assert storage.has_auth("openai") is True


def test_has_auth_fallback() -> None:
    storage = AuthStorage.in_memory()
    storage.set_fallback_resolver(lambda _provider: "z")
    assert storage.has_auth("anything") is True


def test_has_auth_returns_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    storage = AuthStorage.in_memory()
    assert storage.has_auth("openai") is False


# ---------------------------------------------------------------------------
# Resolution priority for ApiKeyCredential entries
# ---------------------------------------------------------------------------


async def test_api_key_resolved_via_resolve_config_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDIRECT_KEY", "real-secret")
    storage = AuthStorage.in_memory({"openai": ApiKeyCredential(type="api_key", key="INDIRECT_KEY")})
    assert await storage.get_api_key("openai") == "real-secret"


# ---------------------------------------------------------------------------
# OAuth handling via the registry protocol
# ---------------------------------------------------------------------------


class _StubProvider:
    def __init__(self, prefix: str = "tok-") -> None:
        self._prefix = prefix

    def get_api_key(self, credential: OAuthCredential) -> str:
        return f"{self._prefix}{credential.access_token}"


class _StubRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, _StubProvider] = {"anthropic": _StubProvider()}
        self.refresh_calls = 0

    def get_provider(self, provider_id: str):
        return self._providers.get(provider_id)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    async def refresh(
        self,
        provider_id: str,
        credentials: dict[str, OAuthCredential],
    ):
        self.refresh_calls += 1
        cred = credentials.get(provider_id)
        if cred is None:
            return None
        new_cred = OAuthCredential(
            type="oauth",
            access_token="refreshed",
            refresh_token=cred.refresh_token,
            expires=10**18,
        )
        return ("tok-refreshed", new_cred)


async def test_oauth_unexpired_returns_token() -> None:
    registry = _StubRegistry()
    storage = AuthStorage.in_memory(
        {
            "anthropic": OAuthCredential(
                type="oauth",
                access_token="abc",
                refresh_token="r",
                expires=10**18,  # far future
            )
        },
        oauth_registry=registry,
    )
    assert await storage.get_api_key("anthropic") == "tok-abc"
    assert registry.refresh_calls == 0


async def test_oauth_expired_triggers_refresh() -> None:
    registry = _StubRegistry()
    storage = AuthStorage.in_memory(
        {
            "anthropic": OAuthCredential(
                type="oauth",
                access_token="stale",
                refresh_token="r",
                expires=0,
            )
        },
        oauth_registry=registry,
    )
    key = await storage.get_api_key("anthropic")
    assert key == "tok-refreshed"
    assert registry.refresh_calls == 1


async def test_oauth_without_registry_returns_none() -> None:
    storage = AuthStorage.in_memory(
        {
            "anthropic": OAuthCredential(
                type="oauth",
                access_token="abc",
                refresh_token="r",
                expires=10**18,
            )
        }
    )
    assert await storage.get_api_key("anthropic") is None


def test_get_oauth_providers_empty_without_registry() -> None:
    storage = AuthStorage.in_memory()
    assert storage.get_oauth_providers() == []


def test_get_oauth_providers_lists_registry() -> None:
    storage = AuthStorage.in_memory(oauth_registry=_StubRegistry())
    assert "anthropic" in storage.get_oauth_providers()


# ---------------------------------------------------------------------------
# File backend round-trip
# ---------------------------------------------------------------------------


def test_file_backend_round_trip(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    backend = FileAuthStorageBackend(str(auth_file))
    storage = AuthStorage.from_storage(backend)
    storage.set("openai", ApiKeyCredential(type="api_key", key="from-file"))
    # Reload from disk via a new instance.
    backend2 = FileAuthStorageBackend(str(auth_file))
    storage2 = AuthStorage.from_storage(backend2)
    cred = storage2.get("openai")
    assert isinstance(cred, ApiKeyCredential)
    assert cred.key == "from-file"


def test_file_backend_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "auth.json"
    backend = FileAuthStorageBackend(str(nested))
    storage = AuthStorage.from_storage(backend)
    storage.set("openai", ApiKeyCredential(type="api_key", key="x"))
    assert nested.exists()


def test_file_backend_writes_json(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    backend = FileAuthStorageBackend(str(auth_file))
    storage = AuthStorage.from_storage(backend)
    storage.set("openai", ApiKeyCredential(type="api_key", key="alpha"))
    parsed = json.loads(auth_file.read_text())
    assert parsed["openai"]["type"] == "api_key"
    assert parsed["openai"]["key"] == "alpha"


# ---------------------------------------------------------------------------
# Async login plumbing
# ---------------------------------------------------------------------------


async def test_login_persists_credential() -> None:
    storage = AuthStorage.in_memory()

    async def login() -> OAuthCredential:
        return OAuthCredential(
            type="oauth",
            access_token="fresh",
            refresh_token="r",
            expires=10**18,
        )

    await storage.login("anthropic", login)
    cred = storage.get("anthropic")
    assert isinstance(cred, OAuthCredential)
    assert cred.access_token == "fresh"


def test_logout_removes_credential() -> None:
    storage = AuthStorage.in_memory({"openai": ApiKeyCredential(type="api_key", key="x")})
    storage.logout("openai")
    assert storage.get("openai") is None


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


async def test_in_memory_async_lock() -> None:
    """Exercise ``InMemoryAuthStorageBackend.with_lock_async``."""
    storage = AuthStorage.in_memory({"openai": ApiKeyCredential(type="api_key", key="x")})

    async def fake_login() -> OAuthCredential:
        return OAuthCredential(
            type="oauth",
            access_token="async-token",
            refresh_token="r",
            expires=10**18,
        )

    await storage.login("anthropic", fake_login)
    assert storage.has("anthropic")


async def test_runtime_override_persists_across_get_calls() -> None:
    storage = AuthStorage.in_memory()
    storage.set_runtime_api_key("custom", "abc")
    assert await storage.get_api_key("custom") == "abc"
    assert await storage.get_api_key("custom") == "abc"


async def test_oauth_unknown_provider_returns_none() -> None:
    """An OAuth credential whose provider isn't in the registry → ``None``."""

    from typing import Any  # noqa: PLC0415

    class _EmptyRegistry:
        def get_provider(self, _provider_id: str) -> Any:
            return None

        def list_providers(self) -> list[str]:
            return []

        async def refresh(self, _provider_id: str, _credentials: Any) -> Any:
            return None

    storage = AuthStorage.in_memory(
        {
            "anthropic": OAuthCredential(
                type="oauth",
                access_token="abc",
                refresh_token="r",
                expires=10**18,
            )
        },
        oauth_registry=_EmptyRegistry(),  # type: ignore[arg-type]
    )
    assert await storage.get_api_key("anthropic") is None


def test_oauth_credential_round_trips_through_disk(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    backend = FileAuthStorageBackend(str(auth_file))
    storage = AuthStorage.from_storage(backend)
    storage.set(
        "anthropic",
        OAuthCredential(
            type="oauth",
            access_token="aa",
            refresh_token="rr",
            expires=12345,
            raw={"id_token": "idtok"},
        ),
    )
    backend2 = FileAuthStorageBackend(str(auth_file))
    storage2 = AuthStorage.from_storage(backend2)
    cred = storage2.get("anthropic")
    assert isinstance(cred, OAuthCredential)
    assert cred.access_token == "aa"
    assert cred.refresh_token == "rr"
    assert cred.expires == 12345
    assert cred.raw.get("id_token") == "idtok"


def test_file_backend_path_property(tmp_path: Path) -> None:
    backend = FileAuthStorageBackend(str(tmp_path / "x.json"))
    assert backend.path.endswith("x.json")


def test_drain_errors_captures_backend_failures() -> None:
    """Trigger a real recorded error by swapping in a backend that always raises."""
    from typing import Any  # noqa: PLC0415

    class _BoomBackend:
        def with_lock(self, fn: Any) -> Any:
            raise RuntimeError("backend down")

        async def with_lock_async(self, fn: Any) -> Any:  # pragma: no cover
            raise RuntimeError("backend down")

    storage = AuthStorage.in_memory()
    storage._storage = _BoomBackend()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
    storage.reload()  # forces _record_error via the failing backend
    drained = storage.drain_errors()
    assert any("backend down" in str(e) for e in drained)
    assert storage.drain_errors() == []
