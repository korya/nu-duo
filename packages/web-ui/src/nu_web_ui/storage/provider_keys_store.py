"""Provider keys store — API keys for LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nu_web_ui.storage.sqlite_backend import SQLiteBackend


class ProviderKeysStore:
    """Stores provider API keys keyed by provider name.

    Values are plain strings (the key itself), serialised as JSON by the
    SQLiteBackend.
    """

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend

    async def get_key(self, provider: str) -> str | None:
        """Return the stored API key for *provider*, or ``None``."""
        value: str | None = await self._backend.get("provider-keys", provider)
        return value

    async def set_key(self, provider: str, key: str) -> None:
        await self._backend.set("provider-keys", provider, key)

    async def delete_key(self, provider: str) -> None:
        await self._backend.delete("provider-keys", provider)

    async def list_keys(self) -> list[str]:
        """Return the list of provider names that have stored keys."""
        return await self._backend.keys("provider-keys")

    async def has_key(self, provider: str) -> bool:
        return await self._backend.has("provider-keys", provider)


__all__ = ["ProviderKeysStore"]
