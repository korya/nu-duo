"""FastAPI application factory for nu_web_ui.

Wires up all routes, dependency injection, and static file serving.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from nu_web_ui.chat import chat_websocket
from nu_web_ui.model_discovery import discover_all_local_models
from nu_web_ui.proxy import proxy_request
from nu_web_ui.storage import (
    CustomProvidersStore,
    ProviderKeysStore,
    SessionsStore,
    SettingsStore,
    SQLiteBackend,
)
from nu_web_ui.types import CustomProvider, ModelInfo, SessionData, SessionInfo, Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (set during lifespan)
# ---------------------------------------------------------------------------

_backend: SQLiteBackend | None = None


def _get_db_path() -> str:
    return os.environ.get("NU_WEB_UI_DB", str(Path.home() / ".nu-web-ui" / "data.db"))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[return]
    global _backend
    db_path = _get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _backend = SQLiteBackend(db_path)
    await _backend.initialize()
    logger.info("Database initialised at %s", db_path)
    yield
    if _backend is not None:
        await _backend.close()
        _backend = None


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def get_backend() -> SQLiteBackend:
    if _backend is None:
        raise RuntimeError("Backend not initialised")
    return _backend


def get_sessions_store(
    backend: Annotated[SQLiteBackend, Depends(get_backend)],
) -> SessionsStore:
    return SessionsStore(backend)


def get_settings_store(
    backend: Annotated[SQLiteBackend, Depends(get_backend)],
) -> SettingsStore:
    return SettingsStore(backend)


def get_provider_keys_store(
    backend: Annotated[SQLiteBackend, Depends(get_backend)],
) -> ProviderKeysStore:
    return ProviderKeysStore(backend)


def get_custom_providers_store(
    backend: Annotated[SQLiteBackend, Depends(get_backend)],
) -> CustomProvidersStore:
    return CustomProvidersStore(backend)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    db_path: str | None = None,
    frontend_dir: str | None = None,
    allow_origins: list[str] | None = None,
) -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        db_path: Override the SQLite database path.
        frontend_dir: Path to the compiled frontend static files.
        allow_origins: CORS allowed origins.  Defaults to ``["*"]`` for
            convenience during development.
    """
    if db_path is not None:
        os.environ["NU_WEB_UI_DB"] = db_path

    app = FastAPI(
        title="nu-web-ui",
        description="Pi web UI backend — FastAPI + SQLite + WebSocket streaming.",
        version="0.0.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @app.get("/api/sessions", response_model=list[SessionInfo])
    async def list_sessions(
        store: Annotated[SessionsStore, Depends(get_sessions_store)],
    ) -> list[SessionInfo]:
        return await store.list_sessions()

    @app.get("/api/sessions/{session_id}", response_model=SessionData)
    async def get_session(
        session_id: str,
        store: Annotated[SessionsStore, Depends(get_sessions_store)],
    ) -> SessionData:
        data = await store.get_session(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return data

    @app.put("/api/sessions/{session_id}", response_model=SessionData)
    async def put_session(
        session_id: str,
        body: SessionData,
        sessions: Annotated[SessionsStore, Depends(get_sessions_store)],
    ) -> SessionData:
        if body.id != session_id:
            raise HTTPException(status_code=400, detail="session_id mismatch")
        from nu_web_ui.types import UsageStats

        existing_meta = await sessions.get_metadata(session_id)
        now_iso = _now_iso()
        meta = SessionInfo(
            id=session_id,
            title=body.title,
            createdAt=existing_meta.created_at if existing_meta else (body.created_at or now_iso),
            lastModified=now_iso,
            messageCount=len(body.messages),
            usage=existing_meta.usage if existing_meta else UsageStats(),
            thinkingLevel=body.thinking_level,
            preview="",
        )
        await sessions.save_session(body, meta)
        return body

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(
        session_id: str,
        store: Annotated[SessionsStore, Depends(get_sessions_store)],
    ) -> None:
        await store.delete_session(session_id)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    @app.get("/api/settings", response_model=Settings)
    async def get_settings(
        store: Annotated[SettingsStore, Depends(get_settings_store)],
    ) -> Settings:
        return await store.get_settings()

    @app.put("/api/settings", response_model=Settings)
    async def put_settings(
        body: Settings,
        store: Annotated[SettingsStore, Depends(get_settings_store)],
    ) -> Settings:
        await store.save_settings(body)
        return body

    # ------------------------------------------------------------------
    # Provider keys
    # ------------------------------------------------------------------

    @app.get("/api/provider-keys", response_model=list[str])
    async def list_provider_keys(
        store: Annotated[ProviderKeysStore, Depends(get_provider_keys_store)],
    ) -> list[str]:
        return await store.list_keys()

    @app.get("/api/provider-keys/{provider}")
    async def get_provider_key(
        provider: str,
        store: Annotated[ProviderKeysStore, Depends(get_provider_keys_store)],
    ) -> dict:
        key = await store.get_key(provider)
        if key is None:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"provider": provider, "key": key}

    @app.put("/api/provider-keys/{provider}", status_code=204)
    async def put_provider_key(
        provider: str,
        body: dict,
        store: Annotated[ProviderKeysStore, Depends(get_provider_keys_store)],
    ) -> None:
        key_value: str = body.get("key", "")
        if not key_value:
            raise HTTPException(status_code=400, detail="Missing 'key' field")
        await store.set_key(provider, key_value)

    @app.delete("/api/provider-keys/{provider}", status_code=204)
    async def delete_provider_key(
        provider: str,
        store: Annotated[ProviderKeysStore, Depends(get_provider_keys_store)],
    ) -> None:
        await store.delete_key(provider)

    # ------------------------------------------------------------------
    # Custom providers
    # ------------------------------------------------------------------

    @app.get("/api/custom-providers", response_model=list[CustomProvider])
    async def list_custom_providers(
        store: Annotated[CustomProvidersStore, Depends(get_custom_providers_store)],
    ) -> list[CustomProvider]:
        return await store.list_providers()

    @app.get("/api/custom-providers/{provider_name}", response_model=CustomProvider)
    async def get_custom_provider(
        provider_name: str,
        store: Annotated[CustomProvidersStore, Depends(get_custom_providers_store)],
    ) -> CustomProvider:
        provider = await store.get_provider(provider_name)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")
        return provider

    @app.put("/api/custom-providers/{provider_name}", response_model=CustomProvider)
    async def put_custom_provider(
        provider_name: str,
        body: CustomProvider,
        store: Annotated[CustomProvidersStore, Depends(get_custom_providers_store)],
    ) -> CustomProvider:
        await store.set_provider(body)
        return body

    @app.delete("/api/custom-providers/{provider_name}", status_code=204)
    async def delete_custom_provider(
        provider_name: str,
        store: Annotated[CustomProvidersStore, Depends(get_custom_providers_store)],
    ) -> None:
        await store.delete_provider(provider_name)

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    @app.get("/api/models/discover", response_model=list[ModelInfo])
    async def discover_models() -> list[ModelInfo]:
        return await discover_all_local_models()

    # ------------------------------------------------------------------
    # CORS proxy
    # ------------------------------------------------------------------

    from fastapi import Request as _Request

    @app.post("/api/proxy")
    async def proxy_handler(request: _Request) -> None:  # type: ignore[return]
        return await proxy_request(request)

    # ------------------------------------------------------------------
    # WebSocket chat
    # ------------------------------------------------------------------

    @app.websocket("/api/chat")
    async def ws_chat(
        websocket: WebSocket,
        sessions: Annotated[SessionsStore, Depends(get_sessions_store)],
        settings: Annotated[SettingsStore, Depends(get_settings_store)],
        provider_keys: Annotated[ProviderKeysStore, Depends(get_provider_keys_store)],
        session_id: str | None = None,
    ) -> None:
        await chat_websocket(
            websocket,
            sessions_store=sessions,
            settings_store=settings,
            provider_keys_store=provider_keys,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Static frontend
    # ------------------------------------------------------------------

    _mount_frontend(app, frontend_dir)

    return app


def _mount_frontend(app: FastAPI, frontend_dir: str | None) -> None:
    """Mount the compiled frontend if the directory exists."""
    candidates = [
        frontend_dir,
        str(Path(__file__).parent / "frontend"),
        str(Path(__file__).parent.parent.parent / "frontend" / "dist"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            app.mount("/", StaticFiles(directory=candidate, html=True), name="frontend")
            logger.info("Serving frontend from %s", candidate)
            return
    logger.info("No frontend directory found — serving API only.")


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


__all__ = ["create_app"]
