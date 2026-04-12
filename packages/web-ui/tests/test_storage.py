"""Tests for the SQLite storage backend and store classes."""

from __future__ import annotations

import pytest
import pytest_asyncio
from nu_web_ui.storage import (
    CustomProvidersStore,
    ProviderKeysStore,
    SessionsStore,
    SettingsStore,
    SQLiteBackend,
)
from nu_web_ui.types import (
    CostBreakdown,
    CustomProvider,
    SessionData,
    SessionInfo,
    UsageStats,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def backend() -> SQLiteBackend:
    """In-memory SQLite backend, initialised fresh for each test."""
    b = SQLiteBackend(":memory:")
    await b.initialize()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def sessions_store(backend: SQLiteBackend) -> SessionsStore:
    return SessionsStore(backend)


@pytest_asyncio.fixture
async def settings_store(backend: SQLiteBackend) -> SettingsStore:
    return SettingsStore(backend)


@pytest_asyncio.fixture
async def provider_keys_store(backend: SQLiteBackend) -> ProviderKeysStore:
    return ProviderKeysStore(backend)


@pytest_asyncio.fixture
async def custom_providers_store(backend: SQLiteBackend) -> CustomProvidersStore:
    return CustomProvidersStore(backend)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "test-session-1") -> tuple[SessionData, SessionInfo]:
    data = SessionData(
        id=session_id,
        title="Test session",
        model=None,
        thinkingLevel="off",
        messages=[{"role": "user", "content": "Hello"}],
        createdAt="2026-01-01T00:00:00+00:00",
        lastModified="2026-01-01T00:01:00+00:00",
    )
    meta = SessionInfo(
        id=session_id,
        title="Test session",
        createdAt="2026-01-01T00:00:00+00:00",
        lastModified="2026-01-01T00:01:00+00:00",
        messageCount=1,
        usage=UsageStats(
            input=10,
            output=20,
            cacheRead=0,
            cacheWrite=0,
            totalTokens=30,
            cost=CostBreakdown(input=0.001, output=0.002, cacheRead=0, cacheWrite=0, total=0.003),
        ),
        thinkingLevel="off",
        preview="Hello",
    )
    return data, meta


# ---------------------------------------------------------------------------
# Test 1 — backend initialises tables without error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_initialises_tables(backend: SQLiteBackend) -> None:
    """SQLiteBackend.initialize() should create all expected tables."""
    # Verify we can get/set/delete without errors — a proxy for tables existing.
    await backend.set("settings", "smoke", {"value": 42})
    result = await backend.get("settings", "smoke")
    assert result == {"value": 42}
    await backend.delete("settings", "smoke")
    assert await backend.get("settings", "smoke") is None


# ---------------------------------------------------------------------------
# Test 2 — sessions CRUD round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessions_crud(sessions_store: SessionsStore) -> None:
    data, meta = _make_session("sess-001")

    # Create
    await sessions_store.save_session(data, meta)

    # Read
    loaded_data = await sessions_store.get_session("sess-001")
    assert loaded_data is not None
    assert loaded_data.id == "sess-001"
    assert loaded_data.title == "Test session"

    loaded_meta = await sessions_store.get_metadata("sess-001")
    assert loaded_meta is not None
    assert loaded_meta.message_count == 1

    # Update title
    await sessions_store.update_title("sess-001", "Updated title")
    updated = await sessions_store.get_session("sess-001")
    assert updated is not None
    assert updated.title == "Updated title"

    # Delete
    await sessions_store.delete_session("sess-001")
    assert await sessions_store.get_session("sess-001") is None
    assert await sessions_store.get_metadata("sess-001") is None


# ---------------------------------------------------------------------------
# Test 3 — settings round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_round_trip(settings_store: SettingsStore) -> None:
    # Default settings
    settings = await settings_store.get_settings()
    assert settings.theme == "system"
    assert settings.proxy.enabled is False

    # Save and reload
    settings.theme = "dark"
    settings.proxy.enabled = True
    settings.proxy.url = "https://proxy.example.com"
    await settings_store.save_settings(settings)

    reloaded = await settings_store.get_settings()
    assert reloaded.theme == "dark"
    assert reloaded.proxy.enabled is True
    assert reloaded.proxy.url == "https://proxy.example.com"


# ---------------------------------------------------------------------------
# Test 4 — provider keys CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_keys_crud(provider_keys_store: ProviderKeysStore) -> None:
    # Initially empty
    keys = await provider_keys_store.list_keys()
    assert keys == []

    # Set a couple of keys
    await provider_keys_store.set_key("anthropic", "sk-ant-api-xxxx")
    await provider_keys_store.set_key("openai", "sk-openai-yyyy")

    # List
    keys = await provider_keys_store.list_keys()
    assert set(keys) == {"anthropic", "openai"}

    # Get
    assert await provider_keys_store.get_key("anthropic") == "sk-ant-api-xxxx"
    assert await provider_keys_store.has_key("openai") is True
    assert await provider_keys_store.get_key("google") is None

    # Delete
    await provider_keys_store.delete_key("openai")
    assert await provider_keys_store.has_key("openai") is False
    keys = await provider_keys_store.list_keys()
    assert keys == ["anthropic"]


# ---------------------------------------------------------------------------
# Test 5 — custom providers CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_providers_crud(custom_providers_store: CustomProvidersStore) -> None:
    provider = CustomProvider(
        id="prov-1",
        name="My Ollama",
        type="ollama",
        baseUrl="http://localhost:11434",
        apiKey=None,
        models=None,
    )

    # Create
    await custom_providers_store.set_provider(provider)

    # Read
    loaded = await custom_providers_store.get_provider("prov-1")
    assert loaded is not None
    assert loaded.name == "My Ollama"
    assert loaded.base_url == "http://localhost:11434"

    # List
    all_providers = await custom_providers_store.list_providers()
    assert len(all_providers) == 1
    assert all_providers[0].id == "prov-1"

    # has
    assert await custom_providers_store.has_provider("prov-1") is True
    assert await custom_providers_store.has_provider("non-existent") is False

    # Delete
    await custom_providers_store.delete_provider("prov-1")
    assert await custom_providers_store.get_provider("prov-1") is None
    assert await custom_providers_store.list_providers() == []


# ---------------------------------------------------------------------------
# Test 6 — list_sessions returns newest-first ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_sessions_ordering(sessions_store: SessionsStore) -> None:
    for i in range(3):
        ts_data = f"2026-01-0{i + 1}T00:00:00+00:00"
        ts_meta = f"2026-01-0{i + 1}T00:01:00+00:00"
        data = SessionData(
            id=f"sess-{i}",
            title=f"Session {i}",
            model=None,
            thinkingLevel="off",
            messages=[],
            createdAt=ts_data,
            lastModified=ts_meta,
        )
        meta = SessionInfo(
            id=f"sess-{i}",
            title=f"Session {i}",
            createdAt=ts_data,
            lastModified=ts_meta,
            messageCount=0,
            usage=UsageStats(),
            thinkingLevel="off",
            preview="",
        )
        await sessions_store.save_session(data, meta)

    sessions = await sessions_store.list_sessions()
    assert len(sessions) == 3
    # Newest (index 2, lastModified 2026-01-03) should come first.
    assert sessions[0].id == "sess-2"
    assert sessions[-1].id == "sess-0"


# ---------------------------------------------------------------------------
# Test 7 — backend transaction atomicity on failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transaction_rollback_on_error(backend: SQLiteBackend) -> None:
    """A transaction that raises should not commit any partial writes."""

    async def _failing_op(tx):  # type: ignore[return]
        await tx.set("settings", "partial", {"written": True})
        raise ValueError("deliberate failure")

    with pytest.raises(ValueError, match="deliberate failure"):
        await backend.transaction(["settings"], "readwrite", _failing_op)

    # The partial write should have been rolled back.
    result = await backend.get("settings", "partial")
    assert result is None
