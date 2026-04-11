"""Session manager — direct port of ``packages/coding-agent/src/core/session-manager.ts``.

Sessions are append-only trees stored as JSONL. Every entry has an
``id`` and ``parent_id`` so the same file can hold multiple branches;
the manager tracks a "leaf" pointer that determines where the next
:meth:`SessionManager.append_message` call attaches its child.

The on-disk JSONL format is byte-compatible with the TypeScript
upstream so a session written by ``nu`` can be resumed by ``pi`` and
vice versa. To preserve that, all entry payloads use the upstream's
camelCase keys (``firstKeptEntryId``, ``parentSession``, etc.) and the
JSON encoder is the stdlib :mod:`json` with the same defaults.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_coding_agent.config import get_agent_dir as _get_default_agent_dir
from nu_coding_agent.config import get_sessions_dir
from nu_coding_agent.core.messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)

if TYPE_CHECKING:
    from collections.abc import Callable


CURRENT_SESSION_VERSION = 3


# ---------------------------------------------------------------------------
# Entry payloads — kept as plain ``dict[str, Any]`` so the on-disk format
# is byte-compatible with the upstream's ``JSON.stringify`` output. Strongly
# typed dataclasses for the most-used surface types are exposed alongside
# for ergonomics.
# ---------------------------------------------------------------------------


type FileEntry = dict[str, Any]
type SessionEntry = dict[str, Any]


@dataclass(slots=True)
class SessionHeader:
    """Decoded view of the first entry of a session JSONL file."""

    id: str
    timestamp: str
    cwd: str
    version: int | None = None
    parent_session: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionHeader:
        return cls(
            id=str(data.get("id", "")),
            timestamp=str(data.get("timestamp", "")),
            cwd=str(data.get("cwd", "")),
            version=data.get("version"),
            parent_session=data.get("parentSession"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "session",
            "version": self.version if self.version is not None else CURRENT_SESSION_VERSION,
            "id": self.id,
            "timestamp": self.timestamp,
            "cwd": self.cwd,
        }
        if self.parent_session is not None:
            out["parentSession"] = self.parent_session
        return out


@dataclass(slots=True)
class SessionContext:
    """Resolved view of a session for LLM context: messages + thinking + model."""

    messages: list[Any]
    thinking_level: str
    model: dict[str, str] | None


@dataclass(slots=True)
class SessionInfo:
    """Listing entry returned by :meth:`SessionManager.list`."""

    path: str
    id: str
    cwd: str
    created: datetime
    modified: datetime
    message_count: int
    first_message: str
    all_messages_text: str
    name: str | None = None
    parent_session_path: str | None = None


@dataclass(slots=True)
class SessionTreeNode:
    entry: SessionEntry
    children: list[SessionTreeNode] = field(default_factory=list)
    label: str | None = None
    label_timestamp: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as ``2026-01-02T03:04:05.123Z`` (matches Node)."""
    now = datetime.now(UTC)
    # Node's ``new Date().toISOString()`` always uses ``Z`` and 3 decimal digits.
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _generate_id(taken: set[str] | dict[str, Any]) -> str:
    """Generate an 8-hex-char id, retrying on collision."""
    for _ in range(100):
        candidate = uuid.uuid4().hex[:8]
        if candidate not in taken:
            return candidate
    return uuid.uuid4().hex  # pragma: no cover — extremely unlikely fallback


def _migrate_v1_to_v2(entries: list[FileEntry]) -> None:
    """Add ``id``/``parentId`` to v1 entries (mutates in place)."""
    ids: set[str] = set()
    prev_id: str | None = None
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 2
            continue
        new_id = _generate_id(ids)
        ids.add(new_id)
        entry["id"] = new_id
        entry["parentId"] = prev_id
        prev_id = new_id
        if entry.get("type") == "compaction" and "firstKeptEntryIndex" in entry:
            idx = entry["firstKeptEntryIndex"]
            if isinstance(idx, int) and 0 <= idx < len(entries):
                target = entries[idx]
                if target.get("type") != "session" and "id" in target:
                    entry["firstKeptEntryId"] = target["id"]
            del entry["firstKeptEntryIndex"]


def _migrate_v2_to_v3(entries: list[FileEntry]) -> None:
    """Rename the legacy ``hookMessage`` role to ``custom`` (mutates in place)."""
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 3
            continue
        if entry.get("type") == "message":
            message = entry.get("message")
            if isinstance(message, dict) and message.get("role") == "hookMessage":
                message["role"] = "custom"


def _migrate_to_current_version(entries: list[FileEntry]) -> bool:
    header = next((e for e in entries if e.get("type") == "session"), None)
    version = header.get("version", 1) if isinstance(header, dict) else 1
    if not isinstance(version, int):
        version = 1
    if version >= CURRENT_SESSION_VERSION:
        return False
    if version < 2:
        _migrate_v1_to_v2(entries)
    if version < 3:
        _migrate_v2_to_v3(entries)
    return True


def migrate_session_entries(entries: list[FileEntry]) -> None:
    """Run all migrations to bring ``entries`` to ``CURRENT_SESSION_VERSION``."""
    _migrate_to_current_version(entries)


def parse_session_entries(content: str) -> list[FileEntry]:
    """Parse a JSONL session blob, dropping malformed lines."""
    entries: list[FileEntry] = []
    for line in content.strip().splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


def get_latest_compaction_entry(entries: list[SessionEntry]) -> SessionEntry | None:
    """Return the most recent compaction entry, or ``None`` if there isn't one."""
    return next(
        (entry for entry in reversed(entries) if entry.get("type") == "compaction"),
        None,
    )


def build_session_context(
    entries: list[SessionEntry],
    leaf_id: str | None = None,
    by_id: dict[str, SessionEntry] | None = None,
) -> SessionContext:
    """Walk from ``leaf_id`` to root, returning the resolved LLM context."""
    if by_id is None:
        by_id = {entry["id"]: entry for entry in entries if "id" in entry}

    if not entries:
        return SessionContext(messages=[], thinking_level="off", model=None)

    leaf: SessionEntry | None = by_id.get(leaf_id) if leaf_id else None
    if leaf is None:
        leaf = entries[-1]

    path: list[SessionEntry] = []
    current: SessionEntry | None = leaf
    while current is not None:
        path.insert(0, current)
        parent_id = current.get("parentId")
        current = by_id.get(parent_id) if parent_id else None

    thinking_level = "off"
    model: dict[str, str] | None = None
    compaction: SessionEntry | None = None

    for entry in path:
        entry_type = entry.get("type")
        if entry_type == "thinking_level_change":
            thinking_level = entry.get("thinkingLevel", thinking_level)
        elif entry_type == "model_change":
            model = {"provider": entry.get("provider", ""), "modelId": entry.get("modelId", "")}
        elif entry_type == "message":
            message = entry.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                model = {
                    "provider": message.get("provider", ""),
                    "modelId": message.get("model", ""),
                }
        elif entry_type == "compaction":
            compaction = entry

    messages: list[Any] = []

    def append_message(entry: SessionEntry) -> None:
        entry_type = entry.get("type")
        if entry_type == "message":
            message = entry.get("message")
            if message is not None:
                messages.append(message)
        elif entry_type == "custom_message":
            messages.append(
                create_custom_message(
                    entry.get("customType", ""),
                    entry.get("content", ""),
                    bool(entry.get("display", False)),
                    entry.get("details"),
                    entry.get("timestamp", _now_iso()),
                )
            )
        elif entry_type == "branch_summary" and entry.get("summary"):
            messages.append(
                create_branch_summary_message(
                    entry["summary"],
                    entry.get("fromId", ""),
                    entry.get("timestamp", _now_iso()),
                )
            )

    if compaction is not None:
        messages.append(
            create_compaction_summary_message(
                compaction.get("summary", ""),
                int(compaction.get("tokensBefore", 0)),
                compaction.get("timestamp", _now_iso()),
            )
        )
        compaction_idx = next(
            (
                i
                for i, entry in enumerate(path)
                if entry.get("type") == "compaction" and entry.get("id") == compaction.get("id")
            ),
            -1,
        )
        first_kept_id = compaction.get("firstKeptEntryId")
        found_first_kept = False
        for i in range(compaction_idx):
            entry = path[i]
            if entry.get("id") == first_kept_id:
                found_first_kept = True
            if found_first_kept:
                append_message(entry)
        for i in range(compaction_idx + 1, len(path)):
            append_message(path[i])
    else:
        for entry in path:
            append_message(entry)

    return SessionContext(messages=messages, thinking_level=thinking_level, model=model)


def get_default_session_dir(cwd: str, agent_dir: str | None = None) -> str:
    """Return ``<agent_dir>/sessions/<encoded-cwd>/`` (creating it if needed)."""
    resolved_agent_dir = agent_dir or _get_default_agent_dir()
    safe_path = "--" + cwd.lstrip("/\\").replace("/", "-").replace("\\", "-").replace(":", "-") + "--"
    session_dir = str(Path(resolved_agent_dir) / "sessions" / safe_path)
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    return session_dir


def load_entries_from_file(file_path: str) -> list[FileEntry]:
    """Load JSONL entries from ``file_path``, returning ``[]`` for missing/invalid files."""
    path = Path(file_path)
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    entries = parse_session_entries(content)
    if not entries:
        return entries
    header = entries[0]
    if header.get("type") != "session" or not isinstance(header.get("id"), str):
        return []
    return entries


def _is_valid_session_file(file_path: str) -> bool:
    try:
        with Path(file_path).open("r", encoding="utf-8") as fp:
            first_line = fp.readline()
        if not first_line.strip():
            return False
        header = json.loads(first_line)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(header, dict) and header.get("type") == "session" and isinstance(header.get("id"), str)


def find_most_recent_session(session_dir: str) -> str | None:
    """Return the most-recently-modified valid ``.jsonl`` in ``session_dir``."""
    dir_path = Path(session_dir)
    if not dir_path.exists():
        return None
    try:
        files = [
            (str(p), p.stat().st_mtime)
            for p in dir_path.iterdir()
            if p.suffix == ".jsonl" and _is_valid_session_file(str(p))
        ]
    except OSError:
        return None
    if not files:
        return None
    files.sort(key=lambda item: item[1], reverse=True)
    return files[0][0]


# ---------------------------------------------------------------------------
# Session listing helpers
# ---------------------------------------------------------------------------


def _is_message_with_content(message: Any) -> bool:
    return isinstance(message, dict) and isinstance(message.get("role"), str) and "content" in message


def _extract_text_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""  # pragma: no cover


def _get_session_modified_date(entries: list[FileEntry], header: dict[str, Any], stat_mtime: float) -> datetime:
    last_activity_time: float | None = None
    for entry in entries:
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if not _is_message_with_content(message):
            continue
        assert isinstance(message, dict)
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        msg_ts = message.get("timestamp")
        if isinstance(msg_ts, (int, float)):
            ts_seconds = msg_ts / 1000
            last_activity_time = ts_seconds if last_activity_time is None else max(last_activity_time, ts_seconds)
            continue
        # Fallback: messages without their own timestamp use the entry timestamp.
        entry_ts = entry.get("timestamp")
        if isinstance(entry_ts, str):  # pragma: no cover — exercised only by very old sessions
            try:
                ts_seconds = datetime.fromisoformat(entry_ts.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            last_activity_time = ts_seconds if last_activity_time is None else max(last_activity_time, ts_seconds)
    if last_activity_time is not None and last_activity_time > 0:
        return datetime.fromtimestamp(last_activity_time, tz=UTC)
    header_ts = header.get("timestamp")
    if isinstance(header_ts, str):
        try:
            return datetime.fromisoformat(header_ts.replace("Z", "+00:00"))
        except ValueError:  # pragma: no cover — header timestamps are written in ISO form
            pass
    return datetime.fromtimestamp(stat_mtime, tz=UTC)  # pragma: no cover — only when header lacks a valid ts


def _build_session_info(file_path: str) -> SessionInfo | None:
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8")
        stat = path.stat()
    except OSError:  # pragma: no cover — file disappeared between listing and read
        return None
    entries = parse_session_entries(content)
    if not entries:  # pragma: no cover — empty files are filtered by the listing walker
        return None
    header = entries[0]
    if header.get("type") != "session":  # pragma: no cover — listing walker checks this first
        return None

    message_count = 0
    first_message = ""
    all_messages: list[str] = []
    name: str | None = None

    for entry in entries:
        if entry.get("type") == "session_info":
            raw_name = entry.get("name")
            stripped = raw_name.strip() if isinstance(raw_name, str) else ""
            name = stripped or None
        if entry.get("type") != "message":
            continue
        message_count += 1
        message = entry.get("message")
        if not _is_message_with_content(message):
            continue
        assert isinstance(message, dict)
        if message.get("role") not in ("user", "assistant"):
            continue
        text = _extract_text_content(message)
        if not text:
            continue
        all_messages.append(text)
        if not first_message and message.get("role") == "user":
            first_message = text

    raw_cwd = header.get("cwd")
    cwd: str = raw_cwd if isinstance(raw_cwd, str) else ""
    raw_parent = header.get("parentSession")
    parent_session_path: str | None = raw_parent if isinstance(raw_parent, str) else None

    try:
        created = datetime.fromisoformat(str(header.get("timestamp", "")).replace("Z", "+00:00"))
    except ValueError:
        created = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    modified = _get_session_modified_date(entries, header, stat.st_mtime)

    return SessionInfo(
        path=file_path,
        id=str(header.get("id", "")),
        cwd=cwd,
        name=name,
        parent_session_path=parent_session_path,
        created=created,
        modified=modified,
        message_count=message_count,
        first_message=first_message or "(no messages)",
        all_messages_text=" ".join(all_messages),
    )


type SessionListProgress = Callable[[int, int], None]


def _list_all_sync(on_progress: SessionListProgress | None) -> list[SessionInfo]:
    sessions_dir = get_sessions_dir()
    if not Path(sessions_dir).exists():
        return []
    try:
        project_dirs = [str(p) for p in Path(sessions_dir).iterdir() if p.is_dir()]
    except OSError:
        return []
    all_files: list[str] = []
    for project_dir in project_dirs:
        try:
            files = [str(p) for p in Path(project_dir).iterdir() if p.suffix == ".jsonl"]
        except OSError:
            continue
        all_files.extend(files)
    total = len(all_files)
    sessions: list[SessionInfo] = []
    for loaded, file in enumerate(all_files, start=1):
        info = _build_session_info(file)
        if on_progress is not None:
            on_progress(loaded, total)
        if info is not None:
            sessions.append(info)
    sessions.sort(key=lambda s: s.modified, reverse=True)
    return sessions


def _list_sessions_from_dir(
    dir_path: str,
    on_progress: SessionListProgress | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    path = Path(dir_path)
    if not path.exists():
        return sessions
    try:
        files = sorted(p for p in path.iterdir() if p.suffix == ".jsonl")
    except OSError:
        return sessions
    total = progress_total if progress_total is not None else len(files)
    for loaded, file in enumerate(files, start=1):
        info = _build_session_info(str(file))
        if on_progress is not None:
            on_progress(progress_offset + loaded, total)
        if info is not None:
            sessions.append(info)
    return sessions


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """Append-only session store backed by a JSONL file."""

    def __init__(
        self,
        cwd: str,
        session_dir: str,
        session_file: str | None,
        persist: bool,
    ) -> None:
        self._cwd = cwd
        self._session_dir = session_dir
        self._session_file: str | None = None
        self._persist = persist
        self._flushed = False
        self._file_entries: list[FileEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self._labels_by_id: dict[str, str] = {}
        self._label_timestamps_by_id: dict[str, str] = {}
        self._leaf_id: str | None = None
        self._session_id = ""

        if persist and session_dir and not Path(session_dir).exists():
            Path(session_dir).mkdir(parents=True, exist_ok=True)

        if session_file:
            self.set_session_file(session_file)
        else:
            self.new_session()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, cwd: str, session_dir: str | None = None) -> SessionManager:
        dir_path = session_dir or get_default_session_dir(cwd)
        return cls(cwd=cwd, session_dir=dir_path, session_file=None, persist=True)

    @classmethod
    def open(
        cls,
        path: str,
        session_dir: str | None = None,
        cwd_override: str | None = None,
    ) -> SessionManager:
        entries = load_entries_from_file(path)
        header = next((e for e in entries if e.get("type") == "session"), None)
        header_cwd = header.get("cwd") if isinstance(header, dict) else None
        if not isinstance(header_cwd, str):
            header_cwd = None
        cwd = cwd_override or header_cwd or os.getcwd()
        dir_path = session_dir or str(Path(path).resolve().parent)
        return cls(cwd=cwd, session_dir=dir_path, session_file=path, persist=True)

    @classmethod
    def continue_recent(cls, cwd: str, session_dir: str | None = None) -> SessionManager:
        dir_path = session_dir or get_default_session_dir(cwd)
        most_recent = find_most_recent_session(dir_path)
        if most_recent is not None:
            return cls(cwd=cwd, session_dir=dir_path, session_file=most_recent, persist=True)
        return cls(cwd=cwd, session_dir=dir_path, session_file=None, persist=True)

    @classmethod
    def in_memory(cls, cwd: str | None = None) -> SessionManager:
        resolved_cwd = cwd or os.getcwd()
        return cls(cwd=resolved_cwd, session_dir="", session_file=None, persist=False)

    @classmethod
    def fork_from(
        cls,
        source_path: str,
        target_cwd: str,
        session_dir: str | None = None,
    ) -> SessionManager:
        source_entries = load_entries_from_file(source_path)
        if not source_entries:
            raise ValueError(f"Cannot fork: source session file is empty or invalid: {source_path}")
        source_header = next((e for e in source_entries if e.get("type") == "session"), None)
        if not isinstance(source_header, dict):
            raise ValueError(f"Cannot fork: source session has no header: {source_path}")

        dir_path = session_dir or get_default_session_dir(target_cwd)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        new_session_id = uuid.uuid4().hex
        timestamp = _now_iso()
        file_timestamp = timestamp.replace(":", "-").replace(".", "-")
        new_session_file = str(Path(dir_path) / f"{file_timestamp}_{new_session_id}.jsonl")

        new_header = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": new_session_id,
            "timestamp": timestamp,
            "cwd": target_cwd,
            "parentSession": source_path,
        }
        with Path(new_session_file).open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(new_header) + "\n")
            for entry in source_entries:
                if entry.get("type") != "session":
                    fp.write(json.dumps(entry) + "\n")

        return cls(cwd=target_cwd, session_dir=dir_path, session_file=new_session_file, persist=True)

    @classmethod
    async def list(
        cls,
        cwd: str,
        session_dir: str | None = None,
        on_progress: SessionListProgress | None = None,
    ) -> list[SessionInfo]:
        dir_path = session_dir or get_default_session_dir(cwd)
        sessions = _list_sessions_from_dir(dir_path, on_progress)
        sessions.sort(key=lambda s: s.modified, reverse=True)
        return sessions

    @classmethod
    async def list_all(cls, on_progress: SessionListProgress | None = None) -> list[SessionInfo]:
        # The upstream version is async because it parallelises file reads
        # via Promise.all. We do the work synchronously inside a worker
        # thread so the public signature stays compatible without forcing
        # callers to introduce a new sync entry point.
        return await asyncio.to_thread(_list_all_sync, on_progress)

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def set_session_file(self, session_file: str) -> None:
        """Switch to a different session file (resume / branch)."""
        self._session_file = str(Path(session_file).resolve())
        path = Path(self._session_file)
        if path.exists():
            self._file_entries = load_entries_from_file(self._session_file)
            if not self._file_entries:
                explicit_path = self._session_file
                self.new_session()
                self._session_file = explicit_path
                self._rewrite_file()
                self._flushed = True
                return
            header = next((e for e in self._file_entries if e.get("type") == "session"), None)
            header_id = header.get("id") if isinstance(header, dict) else None
            self._session_id = header_id if isinstance(header_id, str) and header_id else uuid.uuid4().hex
            if _migrate_to_current_version(self._file_entries):
                self._rewrite_file()
            self._build_index()
            self._flushed = True
        else:
            explicit_path = self._session_file
            self.new_session()
            self._session_file = explicit_path

    def new_session(self, options: dict[str, Any] | None = None) -> str | None:
        opts = options or {}
        opt_id = opts.get("id")
        self._session_id = opt_id if isinstance(opt_id, str) and opt_id else uuid.uuid4().hex
        timestamp = _now_iso()
        header: FileEntry = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": self._session_id,
            "timestamp": timestamp,
            "cwd": self._cwd,
        }
        if opts.get("parentSession"):
            header["parentSession"] = opts["parentSession"]
        self._file_entries = [header]
        self._by_id.clear()
        self._labels_by_id.clear()
        self._label_timestamps_by_id.clear()
        self._leaf_id = None
        self._flushed = False
        if self._persist:
            file_timestamp = timestamp.replace(":", "-").replace(".", "-")
            self._session_file = str(Path(self.get_session_dir()) / f"{file_timestamp}_{self._session_id}.jsonl")
        return self._session_file

    def _build_index(self) -> None:
        self._by_id.clear()
        self._labels_by_id.clear()
        self._label_timestamps_by_id.clear()
        self._leaf_id = None
        for entry in self._file_entries:
            if entry.get("type") == "session":
                continue
            entry_id = entry.get("id")
            if not isinstance(entry_id, str):
                continue
            self._by_id[entry_id] = entry
            self._leaf_id = entry_id
            if entry.get("type") == "label":
                target_id = entry.get("targetId")
                label = entry.get("label")
                if isinstance(target_id, str):
                    if label:
                        self._labels_by_id[target_id] = str(label)
                        self._label_timestamps_by_id[target_id] = str(entry.get("timestamp", ""))
                    else:
                        self._labels_by_id.pop(target_id, None)
                        self._label_timestamps_by_id.pop(target_id, None)

    def _rewrite_file(self) -> None:
        if not self._persist or not self._session_file:
            return
        content = "\n".join(json.dumps(e) for e in self._file_entries) + "\n"
        Path(self._session_file).write_text(content, encoding="utf-8")

    def is_persisted(self) -> bool:
        return self._persist

    def get_cwd(self) -> str:
        return self._cwd

    def get_session_dir(self) -> str:
        return self._session_dir

    def get_session_id(self) -> str:
        return self._session_id

    def get_session_file(self) -> str | None:
        return self._session_file

    def _persist_entry(self, entry: SessionEntry) -> None:
        if not self._persist or not self._session_file:
            return
        has_assistant = any(
            e.get("type") == "message"
            and isinstance(e.get("message"), dict)
            and e["message"].get("role") == "assistant"
            for e in self._file_entries
        )
        if not has_assistant:
            self._flushed = False
            return
        path = Path(self._session_file)
        if not self._flushed:
            with path.open("a", encoding="utf-8") as fp:
                for e in self._file_entries:
                    fp.write(json.dumps(e) + "\n")
            self._flushed = True
        else:
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry) + "\n")

    def _append_entry(self, entry: SessionEntry) -> None:
        self._file_entries.append(entry)
        entry_id = entry.get("id")
        if isinstance(entry_id, str):
            self._by_id[entry_id] = entry
            self._leaf_id = entry_id
        self._persist_entry(entry)

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------

    def append_message(self, message: Any) -> str:
        entry: SessionEntry = {
            "type": "message",
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "message": message.model_dump(by_alias=True) if hasattr(message, "model_dump") else message,
        }
        self._append_entry(entry)
        return entry["id"]

    def append_thinking_level_change(self, thinking_level: str) -> str:
        entry: SessionEntry = {
            "type": "thinking_level_change",
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "thinkingLevel": thinking_level,
        }
        self._append_entry(entry)
        return entry["id"]

    def append_model_change(self, provider: str, model_id: str) -> str:
        entry: SessionEntry = {
            "type": "model_change",
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "provider": provider,
            "modelId": model_id,
        }
        self._append_entry(entry)
        return entry["id"]

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Any = None,
        from_hook: bool | None = None,
    ) -> str:
        entry: SessionEntry = {
            "type": "compaction",
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        }
        if details is not None:
            entry["details"] = details
        if from_hook is not None:
            entry["fromHook"] = from_hook
        self._append_entry(entry)
        return entry["id"]

    def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        entry: SessionEntry = {
            "type": "custom",
            "customType": custom_type,
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
        }
        if data is not None:
            entry["data"] = data
        self._append_entry(entry)
        return entry["id"]

    def append_session_info(self, name: str) -> str:
        entry: SessionEntry = {
            "type": "session_info",
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "name": name.strip(),
        }
        self._append_entry(entry)
        return entry["id"]

    def get_session_name(self) -> str | None:
        for entry in reversed(self.get_entries()):
            if entry.get("type") == "session_info":
                raw = entry.get("name")
                stripped = raw.strip() if isinstance(raw, str) else ""
                return stripped or None
        return None

    def append_custom_message_entry(
        self,
        custom_type: str,
        content: Any,
        display: bool,
        details: Any = None,
    ) -> str:
        entry: SessionEntry = {
            "type": "custom_message",
            "customType": custom_type,
            "content": content,
            "display": display,
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
        }
        if details is not None:
            entry["details"] = details
        self._append_entry(entry)
        return entry["id"]

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def get_leaf_entry(self) -> SessionEntry | None:
        return self._by_id.get(self._leaf_id) if self._leaf_id else None

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self._by_id.get(entry_id)

    def get_children(self, parent_id: str) -> list[SessionEntry]:
        return [entry for entry in self._by_id.values() if entry.get("parentId") == parent_id]

    def get_label(self, entry_id: str) -> str | None:
        return self._labels_by_id.get(entry_id)

    def append_label_change(self, target_id: str, label: str | None) -> str:
        if target_id not in self._by_id:
            raise ValueError(f"Entry {target_id} not found")
        entry: SessionEntry = {
            "type": "label",
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "targetId": target_id,
            "label": label,
        }
        self._append_entry(entry)
        if label:
            self._labels_by_id[target_id] = label
            self._label_timestamps_by_id[target_id] = entry["timestamp"]
        else:
            self._labels_by_id.pop(target_id, None)
            self._label_timestamps_by_id.pop(target_id, None)
        return entry["id"]

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        path: list[SessionEntry] = []
        start_id = from_id if from_id is not None else self._leaf_id
        current = self._by_id.get(start_id) if start_id else None
        while current is not None:
            path.insert(0, current)
            parent_id = current.get("parentId")
            current = self._by_id.get(parent_id) if parent_id else None
        return path

    def build_session_context(self) -> SessionContext:
        return build_session_context(self.get_entries(), self._leaf_id, self._by_id)

    def get_header(self) -> SessionHeader | None:
        header = next((e for e in self._file_entries if e.get("type") == "session"), None)
        return SessionHeader.from_dict(header) if isinstance(header, dict) else None

    def get_entries(self) -> list[SessionEntry]:
        return [e for e in self._file_entries if e.get("type") != "session"]

    def get_tree(self) -> list[SessionTreeNode]:
        entries = self.get_entries()
        node_map: dict[str, SessionTreeNode] = {}
        roots: list[SessionTreeNode] = []
        for entry in entries:
            entry_id = entry.get("id")
            if not isinstance(entry_id, str):
                continue
            node_map[entry_id] = SessionTreeNode(
                entry=entry,
                label=self._labels_by_id.get(entry_id),
                label_timestamp=self._label_timestamps_by_id.get(entry_id),
            )
        for entry in entries:
            entry_id = entry.get("id")
            if not isinstance(entry_id, str):
                continue
            node = node_map[entry_id]
            parent_id = entry.get("parentId")
            if parent_id is None or parent_id == entry_id:
                roots.append(node)
                continue
            parent = node_map.get(parent_id)
            if parent is not None:
                parent.children.append(node)
            else:
                roots.append(node)
        # Sort children by timestamp ascending.
        stack: list[SessionTreeNode] = list(roots)
        while stack:
            node = stack.pop()
            node.children.sort(key=lambda n: n.entry.get("timestamp", ""))
            stack.extend(node.children)
        return roots

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def branch(self, branch_from_id: str) -> None:
        if branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        self._leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: str | None,
        summary: str,
        details: Any = None,
        from_hook: bool | None = None,
    ) -> str:
        if branch_from_id is not None and branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id
        entry: SessionEntry = {
            "type": "branch_summary",
            "id": _generate_id(self._by_id),
            "parentId": branch_from_id,
            "timestamp": _now_iso(),
            "fromId": branch_from_id or "root",
            "summary": summary,
        }
        if details is not None:
            entry["details"] = details
        if from_hook is not None:
            entry["fromHook"] = from_hook
        self._append_entry(entry)
        return entry["id"]

    def create_branched_session(self, leaf_id: str) -> str | None:
        """Fork the path ending at ``leaf_id`` into a brand-new session.

        Direct port of TS ``createBranchedSession``. Walks up from
        ``leaf_id`` to the root, drops any intermediate ``label`` entries
        from the path itself, then assembles a new session whose entries
        are ``[header, ...path_without_labels, ...recreated_label_entries]``.
        Labels are recreated only for entries that survive in the new
        path (matching TS behaviour).

        In persisted mode the new file is only flushed eagerly when the
        forked path already contains an assistant message — otherwise we
        defer the first write to ``_persist_entry`` so a forked-but-empty
        session does not leave a duplicate header on disk. Returns the
        new file path (persisted mode) or ``None`` (in-memory).
        """
        previous_session_file = self._session_file
        path = self.get_branch(leaf_id)
        if not path:
            raise ValueError(f"Entry {leaf_id} not found")

        path_without_labels = [e for e in path if e.get("type") != "label"]

        new_session_id = uuid.uuid4().hex
        timestamp = _now_iso()
        file_timestamp = timestamp.replace(":", "-").replace(".", "-")

        header: FileEntry = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": new_session_id,
            "timestamp": timestamp,
            "cwd": self._cwd,
        }
        if self._persist and previous_session_file:
            header["parentSession"] = previous_session_file

        # Collect labels for entries that survive in the new path.
        path_entry_ids: set[str] = {e["id"] for e in path_without_labels if isinstance(e.get("id"), str)}
        labels_to_write: list[tuple[str, str, str]] = []
        for target_id, label in self._labels_by_id.items():
            if target_id in path_entry_ids:
                labels_to_write.append((target_id, label, self._label_timestamps_by_id.get(target_id, "")))

        # Recreate the label entries chained off the last path entry.
        last_entry_id: str | None = path_without_labels[-1].get("id") if path_without_labels else None
        parent_id = last_entry_id if isinstance(last_entry_id, str) else None
        label_entries: list[SessionEntry] = []
        used_ids: set[str] = set(path_entry_ids)
        for target_id, label, label_timestamp in labels_to_write:
            label_id = _generate_id(used_ids)
            label_entry: SessionEntry = {
                "type": "label",
                "id": label_id,
                "parentId": parent_id,
                "timestamp": label_timestamp,
                "targetId": target_id,
                "label": label,
            }
            used_ids.add(label_id)
            label_entries.append(label_entry)
            parent_id = label_id

        if self._persist:
            new_session_file = str(Path(self.get_session_dir()) / f"{file_timestamp}_{new_session_id}.jsonl")
            self._file_entries = [header, *path_without_labels, *label_entries]
            self._session_id = new_session_id
            self._session_file = new_session_file
            self._build_index()

            has_assistant = any(
                e.get("type") == "message"
                and isinstance(e.get("message"), dict)
                and e["message"].get("role") == "assistant"
                for e in self._file_entries
            )
            if has_assistant:
                self._rewrite_file()
                self._flushed = True
            else:
                self._flushed = False
            return new_session_file

        # In-memory: just swap the entries in place.
        self._file_entries = [header, *path_without_labels, *label_entries]
        self._session_id = new_session_id
        self._build_index()
        return None


__all__ = [
    "CURRENT_SESSION_VERSION",
    "SessionContext",
    "SessionHeader",
    "SessionInfo",
    "SessionListProgress",
    "SessionManager",
    "SessionTreeNode",
    "build_session_context",
    "find_most_recent_session",
    "get_default_session_dir",
    "get_latest_compaction_entry",
    "load_entries_from_file",
    "migrate_session_entries",
    "parse_session_entries",
]
