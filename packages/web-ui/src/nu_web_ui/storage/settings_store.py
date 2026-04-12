"""Settings store — key/value persistence for application settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nu_web_ui.types import Settings

if TYPE_CHECKING:
    from nu_web_ui.storage.sqlite_backend import SQLiteBackend

_SETTINGS_KEY = "app"


class SettingsStore:
    """Persists application settings as a single JSON blob keyed by ``app``."""

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend

    async def get_settings(self) -> Settings:
        """Return current settings, falling back to defaults if not stored yet."""
        raw: Any = await self._backend.get("settings", _SETTINGS_KEY)
        if raw is None:
            return Settings()
        return Settings.model_validate(raw)

    async def save_settings(self, settings: Settings) -> None:
        await self._backend.set("settings", _SETTINGS_KEY, settings.model_dump(by_alias=True))

    # ------------------------------------------------------------------
    # Low-level helpers (mirrors TS SettingsStore.get/set/delete/list)
    # ------------------------------------------------------------------

    async def get_raw(self, key: str) -> Any | None:
        return await self._backend.get("settings", key)

    async def set_raw(self, key: str, value: Any) -> None:
        await self._backend.set("settings", key, value)

    async def delete_raw(self, key: str) -> None:
        await self._backend.delete("settings", key)

    async def list_keys(self) -> list[str]:
        return await self._backend.keys("settings")


__all__ = ["SettingsStore"]
