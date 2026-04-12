"""Custom providers store — user-defined LLM provider configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nu_web_ui.types import CustomProvider

if TYPE_CHECKING:
    from nu_web_ui.storage.sqlite_backend import SQLiteBackend


class CustomProvidersStore:
    """Manages custom LLM provider configurations (Ollama, LM Studio, etc.)."""

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend

    async def get_provider(self, provider_id: str) -> CustomProvider | None:
        raw = await self._backend.get("custom-providers", provider_id)
        if raw is None:
            return None
        return CustomProvider.model_validate(raw)

    async def set_provider(self, provider: CustomProvider) -> None:
        await self._backend.set(
            "custom-providers",
            provider.id,
            provider.model_dump(by_alias=True),
        )

    async def delete_provider(self, provider_id: str) -> None:
        await self._backend.delete("custom-providers", provider_id)

    async def list_providers(self) -> list[CustomProvider]:
        keys = await self._backend.keys("custom-providers")
        providers: list[CustomProvider] = []
        for key in keys:
            provider = await self.get_provider(key)
            if provider is not None:
                providers.append(provider)
        return providers

    async def has_provider(self, provider_id: str) -> bool:
        return await self._backend.has("custom-providers", provider_id)


__all__ = ["CustomProvidersStore"]
