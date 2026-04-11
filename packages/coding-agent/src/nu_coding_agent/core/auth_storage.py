"""Credential storage — direct port of ``packages/coding-agent/src/core/auth-storage.ts``.

Loads, saves, and refreshes credentials from ``<agent_dir>/auth.json``.
File access is serialised through :mod:`filelock` so multiple ``nu``
instances refreshing OAuth tokens at the same time can't clobber each
other.

The upstream module pulls in ``getOAuthProvider`` / ``getOAuthApiKey``
from ``@mariozechner/pi-ai/oauth``. The Python OAuth port hasn't landed
yet, so this module ships with a pluggable :class:`OAuthRegistry`
:class:`typing.Protocol` — :func:`AuthStorage.create` accepts an
optional registry, and :meth:`get_api_key` and :meth:`login` defer the
provider lookup to it. When the OAuth port lands it will satisfy the
protocol without changes here.
"""

from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from filelock import FileLock, Timeout
from nu_ai.env_api_keys import get_env_api_key

from nu_coding_agent.config import get_agent_dir
from nu_coding_agent.core.resolve_config_value import resolve_config_value

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping


# ---------------------------------------------------------------------------
# Credential value types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ApiKeyCredential:
    type: Literal["api_key"]
    key: str


@dataclass(slots=True)
class OAuthCredential:
    """OAuth credential payload — fields mirror the upstream ``OAuthCredentials``.

    ``expires`` is a unix-millisecond timestamp (parity with the TS
    upstream). ``raw`` carries any extra provider-specific fields so
    the OAuth provider can round-trip them.
    """

    type: Literal["oauth"]
    access_token: str
    refresh_token: str | None
    expires: int
    raw: dict[str, Any] = field(default_factory=dict)


type AuthCredential = ApiKeyCredential | OAuthCredential


# ---------------------------------------------------------------------------
# OAuth provider registry — pluggable until ``nu_ai/oauth`` ships.
# ---------------------------------------------------------------------------


class OAuthProviderLike(Protocol):
    """Structural shape of upstream ``getOAuthProvider``'s return value."""

    def get_api_key(self, credential: OAuthCredential) -> str: ...


class OAuthRegistry(Protocol):
    """Structural protocol that ``nu_ai/oauth`` will satisfy once ported."""

    def get_provider(self, provider_id: str) -> OAuthProviderLike | None: ...
    def list_providers(self) -> list[str]: ...

    async def refresh(
        self,
        provider_id: str,
        credentials: dict[str, OAuthCredential],
    ) -> tuple[str, OAuthCredential] | None: ...


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LockResult:
    """Return value of the ``with_lock`` callback — text to write back, plus payload."""

    result: Any = None
    next: str | None = None


class AuthStorageBackend(Protocol):
    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any: ...
    async def with_lock_async(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
    ) -> Any: ...


class FileAuthStorageBackend:
    """Real on-disk backend, with :mod:`filelock`-based exclusion."""

    def __init__(self, auth_path: str | None = None) -> None:
        self._auth_path = auth_path or str(Path(get_agent_dir()) / "auth.json")
        self._lock = FileLock(f"{self._auth_path}.lock")

    @property
    def path(self) -> str:
        return self._auth_path

    def _ensure_parent_dir(self) -> None:
        parent = Path(self._auth_path).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            os.chmod(parent, stat.S_IRWXU)

    def _ensure_file_exists(self) -> None:
        path = Path(self._auth_path)
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    def _read_current(self) -> str | None:
        path = Path(self._auth_path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _write_next(self, content: str) -> None:
        path = Path(self._auth_path)
        path.write_text(content, encoding="utf-8")
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any:
        self._ensure_parent_dir()
        self._ensure_file_exists()
        try:
            with self._lock.acquire(timeout=10):
                current = self._read_current()
                outcome = fn(current)
                if outcome.next is not None:
                    self._write_next(outcome.next)
                return outcome.result
        except Timeout as exc:
            raise RuntimeError(f"Failed to acquire auth storage lock: {self._auth_path}") from exc

    async def with_lock_async(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
    ) -> Any:
        self._ensure_parent_dir()
        self._ensure_file_exists()
        try:
            with self._lock.acquire(timeout=30):
                current = self._read_current()
                outcome = await fn(current)
                if outcome.next is not None:
                    self._write_next(outcome.next)
                return outcome.result
        except Timeout as exc:
            raise RuntimeError(f"Failed to acquire auth storage lock: {self._auth_path}") from exc


class InMemoryAuthStorageBackend:
    """Test-friendly backend that holds the auth blob in memory."""

    def __init__(self) -> None:
        self._value: str | None = None

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any:
        outcome = fn(self._value)
        if outcome.next is not None:
            self._value = outcome.next
        return outcome.result

    async def with_lock_async(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
    ) -> Any:
        outcome = await fn(self._value)
        if outcome.next is not None:
            self._value = outcome.next
        return outcome.result


# ---------------------------------------------------------------------------
# (de)serialisation helpers
# ---------------------------------------------------------------------------


def _credential_to_jsonable(cred: AuthCredential) -> dict[str, Any]:
    if isinstance(cred, ApiKeyCredential):
        return {"type": "api_key", "key": cred.key}
    payload: dict[str, Any] = {
        "type": "oauth",
        "access_token": cred.access_token,
        "refresh_token": cred.refresh_token,
        "expires": cred.expires,
    }
    payload.update(cred.raw)
    return payload


def _credential_from_jsonable(value: Any) -> AuthCredential | None:
    if not isinstance(value, dict):
        return None
    cred_type = value.get("type")
    if cred_type == "api_key":
        key = value.get("key")
        if isinstance(key, str):
            return ApiKeyCredential(type="api_key", key=key)
        return None
    if cred_type == "oauth":
        access_token = value.get("access_token")
        expires = value.get("expires")
        if not isinstance(access_token, str) or not isinstance(expires, int):
            return None
        refresh_token = value.get("refresh_token")
        if refresh_token is not None and not isinstance(refresh_token, str):
            refresh_token = None
        raw = {k: v for k, v in value.items() if k not in {"type", "access_token", "refresh_token", "expires"}}
        return OAuthCredential(
            type="oauth",
            access_token=access_token,
            refresh_token=refresh_token,
            expires=expires,
            raw=raw,
        )
    return None


def _data_to_json(data: dict[str, AuthCredential]) -> str:
    return json.dumps(
        {provider: _credential_to_jsonable(cred) for provider, cred in data.items()},
        indent=2,
    )


def _parse_storage_data(content: str | None) -> dict[str, AuthCredential]:
    if not content:
        return {}
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, AuthCredential] = {}
    for provider, value in parsed.items():
        cred = _credential_from_jsonable(value)
        if cred is not None:
            out[provider] = cred
    return out


# ---------------------------------------------------------------------------
# AuthStorage
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


class AuthStorage:
    """Credential store backed by an :class:`AuthStorageBackend`."""

    def __init__(
        self,
        storage: AuthStorageBackend,
        *,
        oauth_registry: OAuthRegistry | None = None,
    ) -> None:
        self._storage = storage
        self._oauth_registry = oauth_registry
        self._data: dict[str, AuthCredential] = {}
        self._runtime_overrides: dict[str, str] = {}
        self._fallback_resolver: Callable[[str], str | None] | None = None
        self._load_error: Exception | None = None
        self._errors: list[Exception] = []
        self.reload()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        auth_path: str | None = None,
        *,
        oauth_registry: OAuthRegistry | None = None,
    ) -> AuthStorage:
        backend_path = auth_path or str(Path(get_agent_dir()) / "auth.json")
        return cls(FileAuthStorageBackend(backend_path), oauth_registry=oauth_registry)

    @classmethod
    def from_storage(
        cls,
        storage: AuthStorageBackend,
        *,
        oauth_registry: OAuthRegistry | None = None,
    ) -> AuthStorage:
        return cls(storage, oauth_registry=oauth_registry)

    @classmethod
    def in_memory(
        cls,
        data: Mapping[str, AuthCredential] | None = None,
        *,
        oauth_registry: OAuthRegistry | None = None,
    ) -> AuthStorage:
        backend = InMemoryAuthStorageBackend()
        if data:
            seeded = dict(data)
            backend.with_lock(lambda _current: LockResult(next=_data_to_json(seeded)))
        return cls.from_storage(backend, oauth_registry=oauth_registry)

    # ------------------------------------------------------------------
    # Runtime overrides / fallback
    # ------------------------------------------------------------------

    def set_runtime_api_key(self, provider: str, api_key: str) -> None:
        self._runtime_overrides[provider] = api_key

    def remove_runtime_api_key(self, provider: str) -> None:
        self._runtime_overrides.pop(provider, None)

    def set_fallback_resolver(self, resolver: Callable[[str], str | None]) -> None:
        self._fallback_resolver = resolver

    # ------------------------------------------------------------------
    # Error tracking
    # ------------------------------------------------------------------

    def _record_error(self, error: Exception) -> None:
        self._errors.append(error)

    def drain_errors(self) -> list[Exception]:
        drained = list(self._errors)
        self._errors.clear()
        return drained

    # ------------------------------------------------------------------
    # Storage I/O
    # ------------------------------------------------------------------

    def reload(self) -> None:
        content_holder: dict[str, str | None] = {"value": None}

        def _capture(current: str | None) -> LockResult:
            content_holder["value"] = current
            return LockResult()

        try:
            self._storage.with_lock(_capture)
            self._data = _parse_storage_data(content_holder["value"])
            self._load_error = None
        except Exception as exc:
            self._load_error = exc
            self._record_error(exc)

    def _persist_provider_change(self, provider: str, credential: AuthCredential | None) -> None:
        if self._load_error is not None:
            return

        def _update(current: str | None) -> LockResult:
            current_data = _parse_storage_data(current)
            if credential is not None:
                current_data[provider] = credential
            else:
                current_data.pop(provider, None)
            return LockResult(next=_data_to_json(current_data))

        try:
            self._storage.with_lock(_update)
        except Exception as exc:
            self._record_error(exc)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, provider: str) -> AuthCredential | None:
        return self._data.get(provider)

    def set(self, provider: str, credential: AuthCredential) -> None:
        self._data[provider] = credential
        self._persist_provider_change(provider, credential)

    def remove(self, provider: str) -> None:
        self._data.pop(provider, None)
        self._persist_provider_change(provider, None)

    def list(self) -> list[str]:
        return list(self._data.keys())

    def has(self, provider: str) -> bool:
        return provider in self._data

    def has_auth(self, provider: str) -> bool:
        """Return ``True`` iff *some* form of credential exists for ``provider``.

        Unlike :meth:`get_api_key` this never refreshes OAuth tokens.
        """
        if provider in self._runtime_overrides:
            return True
        if provider in self._data:
            return True
        if get_env_api_key(provider):
            return True
        return self._fallback_resolver is not None and bool(self._fallback_resolver(provider))

    def get_all(self) -> dict[str, AuthCredential]:
        return dict(self._data)

    # ------------------------------------------------------------------
    # OAuth lifecycle
    # ------------------------------------------------------------------

    async def login(self, provider_id: str, login_callback: Callable[[], Awaitable[OAuthCredential]]) -> None:
        """Run an OAuth login flow and persist the resulting credential.

        The upstream signature took an ``OAuthLoginCallbacks`` object that
        the registered provider drove. Until the OAuth port lands the
        ``login_callback`` argument lets the caller drive the flow
        directly — it returns the freshly minted credential.
        """
        cred = await login_callback()
        self.set(provider_id, cred)

    def logout(self, provider: str) -> None:
        self.remove(provider)

    async def _refresh_oauth_token_with_lock(
        self,
        provider_id: str,
    ) -> tuple[str, OAuthCredential] | None:
        registry = self._oauth_registry
        if registry is None:
            return None
        provider = registry.get_provider(provider_id)
        if provider is None:
            return None

        async def _update(current: str | None) -> LockResult:
            current_data = _parse_storage_data(current)
            self._data = current_data
            self._load_error = None

            cred = current_data.get(provider_id)
            if not isinstance(cred, OAuthCredential):
                return LockResult(result=None)

            if _now_ms() < cred.expires:
                return LockResult(result=(provider.get_api_key(cred), cred))

            oauth_creds = {k: v for k, v in current_data.items() if isinstance(v, OAuthCredential)}
            refreshed = await registry.refresh(provider_id, oauth_creds)
            if refreshed is None:
                return LockResult(result=None)

            api_key, new_cred = refreshed
            merged = {**current_data, provider_id: new_cred}
            self._data = merged
            self._load_error = None
            return LockResult(result=(api_key, new_cred), next=_data_to_json(merged))

        return await self._storage.with_lock_async(_update)

    async def get_api_key(
        self,
        provider_id: str,
        *,
        include_fallback: bool = True,
    ) -> str | None:
        """Resolve the API key for ``provider_id`` using the same priority order as upstream:

        1. Runtime override (set via :meth:`set_runtime_api_key`).
        2. ``api_key`` credential in ``auth.json``.
        3. ``oauth`` credential in ``auth.json`` (auto-refreshed via the
           :class:`OAuthRegistry` if expired).
        4. Environment variable from :func:`nu_ai.env_api_keys.get_env_api_key`.
        5. Fallback resolver registered via :meth:`set_fallback_resolver`
           (used for custom-provider keys from ``models.json``).
        """
        runtime_key = self._runtime_overrides.get(provider_id)
        if runtime_key:
            return runtime_key

        cred = self._data.get(provider_id)

        if isinstance(cred, ApiKeyCredential):
            return resolve_config_value(cred.key)

        if isinstance(cred, OAuthCredential):
            registry = self._oauth_registry
            if registry is None:
                return None
            provider = registry.get_provider(provider_id)
            if provider is None:
                return None
            if _now_ms() < cred.expires:
                return provider.get_api_key(cred)
            try:
                refreshed = await self._refresh_oauth_token_with_lock(provider_id)
                if refreshed is not None:
                    return refreshed[0]
            except Exception as exc:
                self._record_error(exc)
                self.reload()
                updated = self._data.get(provider_id)
                if isinstance(updated, OAuthCredential) and _now_ms() < updated.expires:
                    return provider.get_api_key(updated)
                return None

        env_key = get_env_api_key(provider_id)
        if env_key:
            return env_key

        if include_fallback and self._fallback_resolver is not None:
            return self._fallback_resolver(provider_id)
        return None

    def get_oauth_providers(self) -> list[str]:
        if self._oauth_registry is None:
            return []
        return self._oauth_registry.list_providers()


__all__ = [
    "ApiKeyCredential",
    "AuthCredential",
    "AuthStorage",
    "AuthStorageBackend",
    "FileAuthStorageBackend",
    "InMemoryAuthStorageBackend",
    "LockResult",
    "OAuthCredential",
    "OAuthProviderLike",
    "OAuthRegistry",
]
