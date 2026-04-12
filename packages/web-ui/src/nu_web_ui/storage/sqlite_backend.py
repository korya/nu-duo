"""SQLite storage backend using aiosqlite.

Implements a multi-store key-value abstraction backed by a single SQLite
database.  Each logical "store" maps to a table named ``kv_<store_name>``
with a TEXT primary key and a TEXT value (JSON-encoded).

The ``sessions-metadata`` store additionally maintains a ``last_modified``
column so that list queries can be sorted without deserialising every row.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager, suppress
from typing import Any

import aiosqlite


# Map store names to safe SQL table names (replace hyphens with underscores).
def _table(store_name: str) -> str:
    return "kv_" + re.sub(r"[^a-zA-Z0-9_]", "_", store_name)


# Stores that need an extra indexed column for ordered listing.
_INDEXED_STORES: dict[str, str] = {
    "sessions": "last_modified",
    "sessions-metadata": "last_modified",
}


class SQLiteBackend:
    """Async SQLite key-value backend.

    Args:
        db_path: Path to the SQLite database file.  Use ``":memory:"`` for
            in-memory databases (useful in tests).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the database and create all required tables."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    async def _create_tables(self) -> None:
        assert self._db is not None
        stores = [
            "sessions",
            "sessions-metadata",
            "settings",
            "provider-keys",
            "custom-providers",
        ]
        for store in stores:
            table = _table(store)
            if store in _INDEXED_STORES:
                idx_col = _INDEXED_STORES[store]
                await self._db.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        {idx_col} TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                await self._db.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{idx_col} ON {table}({idx_col})")
            else:
                await self._db.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteBackend not initialized — call initialize() first.")
        return self._db

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def get(self, store_name: str, key: str) -> Any | None:
        table = _table(store_name)
        async with self._conn().execute(f"SELECT value FROM {table} WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    async def set(self, store_name: str, key: str, value: Any) -> None:
        table = _table(store_name)
        encoded = json.dumps(value)
        if store_name in _INDEXED_STORES:
            idx_col = _INDEXED_STORES[store_name]
            # Extract the indexed column value from the object (if it's a dict).
            idx_val = ""
            if isinstance(value, dict):
                idx_val = value.get(idx_col) or value.get(_camel(idx_col)) or ""
            await self._conn().execute(
                f"""
                INSERT INTO {table}(key, value, {idx_col})
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, {idx_col}=excluded.{idx_col}
                """,
                (key, encoded, idx_val),
            )
        else:
            await self._conn().execute(
                f"""
                INSERT INTO {table}(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, encoded),
            )
        await self._conn().commit()

    async def delete(self, store_name: str, key: str) -> None:
        table = _table(store_name)
        await self._conn().execute(f"DELETE FROM {table} WHERE key = ?", (key,))
        await self._conn().commit()

    async def keys(self, store_name: str, prefix: str | None = None) -> list[str]:
        table = _table(store_name)
        if prefix:
            async with self._conn().execute(f"SELECT key FROM {table} WHERE key LIKE ?", (prefix + "%",)) as cur:
                rows = await cur.fetchall()
        else:
            async with self._conn().execute(f"SELECT key FROM {table}") as cur:
                rows = await cur.fetchall()
        return [row["key"] for row in rows]

    async def get_all_from_index(
        self,
        store_name: str,
        index_name: str,
        direction: str = "asc",
    ) -> list[Any]:
        table = _table(store_name)
        order = "DESC" if direction == "desc" else "ASC"
        # Sanitise index_name to prevent SQL injection (only allow word chars).
        safe_col = re.sub(r"[^a-zA-Z0-9_]", "_", index_name)
        async with self._conn().execute(f"SELECT value FROM {table} ORDER BY {safe_col} {order}") as cur:
            rows = await cur.fetchall()
        return [json.loads(row["value"]) for row in rows]

    async def clear(self, store_name: str) -> None:
        table = _table(store_name)
        await self._conn().execute(f"DELETE FROM {table}")
        await self._conn().commit()

    async def has(self, store_name: str, key: str) -> bool:
        table = _table(store_name)
        async with self._conn().execute(f"SELECT 1 FROM {table} WHERE key = ? LIMIT 1", (key,)) as cur:
            row = await cur.fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Transaction helper
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _transaction(self):  # type: ignore[return]
        """Context manager that wraps operations in a BEGIN/COMMIT block."""
        conn = self._conn()
        await conn.execute("BEGIN")
        try:
            yield conn
            await conn.execute("COMMIT")
        except Exception:
            await conn.execute("ROLLBACK")
            raise

    async def transaction(
        self,
        store_names: list[str],
        mode: str,
        operation: Any,
    ) -> Any:
        """Execute *operation* atomically across the given stores.

        ``operation`` receives a ``_TxProxy`` that exposes get/set/delete
        without individual commits, all flushed together at the end.
        """
        conn = self._conn()
        await conn.execute("BEGIN")
        try:
            proxy = _TxProxy(conn, store_names)
            result = await operation(proxy)
            await conn.execute("COMMIT")
            return result
        except Exception:
            await conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Quota / persistence (stubs — SQLite has no browser quota concept)
    # ------------------------------------------------------------------

    async def get_quota_info(self) -> dict[str, Any]:
        import os

        usage = 0
        if self._db_path != ":memory:":
            with suppress(OSError):
                usage = os.path.getsize(self._db_path)
        return {"usage": usage, "quota": 0, "percent": 0}

    async def request_persistence(self) -> bool:
        return True  # SQLite files are already persistent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _camel(snake: str) -> str:
    """Convert simple snake_case to camelCase."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class _TxProxy:
    """Lightweight proxy used inside a transaction callback.

    Performs reads and writes directly on the connection without individual
    COMMITs — the outer transaction() manager handles commit/rollback.
    """

    def __init__(self, conn: aiosqlite.Connection, store_names: list[str]) -> None:
        self._conn = conn
        self._store_names = store_names

    def _check(self, store_name: str) -> None:
        if store_name not in self._store_names:
            raise ValueError(f"Store '{store_name}' not included in transaction stores.")

    async def get(self, store_name: str, key: str) -> Any | None:
        self._check(store_name)
        table = _table(store_name)
        async with self._conn.execute(f"SELECT value FROM {table} WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    async def set(self, store_name: str, key: str, value: Any) -> None:
        self._check(store_name)
        table = _table(store_name)
        encoded = json.dumps(value)
        if store_name in _INDEXED_STORES:
            idx_col = _INDEXED_STORES[store_name]
            idx_val = ""
            if isinstance(value, dict):
                idx_val = value.get(idx_col) or value.get(_camel(idx_col)) or ""
            await self._conn.execute(
                f"""
                INSERT INTO {table}(key, value, {idx_col})
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, {idx_col}=excluded.{idx_col}
                """,
                (key, encoded, idx_val),
            )
        else:
            await self._conn.execute(
                f"""
                INSERT INTO {table}(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, encoded),
            )

    async def delete(self, store_name: str, key: str) -> None:
        self._check(store_name)
        table = _table(store_name)
        await self._conn.execute(f"DELETE FROM {table} WHERE key = ?", (key,))


__all__ = ["SQLiteBackend"]
