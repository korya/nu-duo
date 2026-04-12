"""Sessions store — CRUD for chat sessions and their metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nu_web_ui.types import SessionData, SessionInfo

if TYPE_CHECKING:
    from nu_web_ui.storage.sqlite_backend import SQLiteBackend


class SessionsStore:
    """Manages session data and metadata backed by SQLiteBackend.

    Two logical stores are used:
    - ``sessions``          — full session data including the message transcript.
    - ``sessions-metadata`` — lightweight metadata used for listing/searching.

    Both stores are always kept in sync via transactions.
    """

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    async def list_sessions(self) -> list[SessionInfo]:
        """Return all session metadata sorted by lastModified descending."""
        raw: list[dict[str, Any]] = await self._backend.get_all_from_index(
            "sessions-metadata", "last_modified", direction="desc"
        )
        result: list[SessionInfo] = []
        for item in raw:
            try:
                result.append(SessionInfo.model_validate(item))
            except Exception:
                pass  # skip corrupt rows
        return result

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> SessionData | None:
        """Return full session data, or ``None`` if not found."""
        raw = await self._backend.get("sessions", session_id)
        if raw is None:
            return None
        return SessionData.model_validate(raw)

    async def get_metadata(self, session_id: str) -> SessionInfo | None:
        raw = await self._backend.get("sessions-metadata", session_id)
        if raw is None:
            return None
        return SessionInfo.model_validate(raw)

    # ------------------------------------------------------------------
    # Save (upsert)
    # ------------------------------------------------------------------

    async def save_session(self, data: SessionData, metadata: SessionInfo) -> None:
        """Atomically persist both full data and metadata."""
        data_dict = data.model_dump(by_alias=True)
        meta_dict = metadata.model_dump(by_alias=True)
        await self._backend.transaction(
            ["sessions", "sessions-metadata"],
            "readwrite",
            lambda tx: _save_both(tx, data.id, data_dict, metadata.id, meta_dict),
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_session(self, session_id: str) -> None:
        """Atomically remove both full data and metadata."""
        await self._backend.transaction(
            ["sessions", "sessions-metadata"],
            "readwrite",
            lambda tx: _delete_both(tx, session_id),
        )

    # ------------------------------------------------------------------
    # Update title
    # ------------------------------------------------------------------

    async def update_title(self, session_id: str, title: str) -> None:
        meta = await self.get_metadata(session_id)
        if meta is not None:
            meta.title = title
            await self._backend.set("sessions-metadata", session_id, meta.model_dump(by_alias=True))
        data = await self.get_session(session_id)
        if data is not None:
            data.title = title
            await self._backend.set("sessions", session_id, data.model_dump(by_alias=True))


# ---------------------------------------------------------------------------
# Transaction helpers (must be plain async functions for the lambda to work)
# ---------------------------------------------------------------------------


async def _save_both(
    tx: Any,
    data_id: str,
    data_dict: dict[str, Any],
    meta_id: str,
    meta_dict: dict[str, Any],
) -> None:
    await tx.set("sessions", data_id, data_dict)
    await tx.set("sessions-metadata", meta_id, meta_dict)


async def _delete_both(tx: Any, session_id: str) -> None:
    await tx.delete("sessions", session_id)
    await tx.delete("sessions-metadata", session_id)


__all__ = ["SessionsStore"]
