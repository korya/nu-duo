"""Per-channel file storage for nu-mom.

Port of ``packages/mom/src/store.ts``. Manages a working-directory tree where
each Slack channel gets its own sub-directory containing:

  - ``log.jsonl``     — append-only human-readable message history
  - ``attachments/``  — downloaded Slack file attachments

The class is intentionally *not* async-heavy at the surface: message logging
uses ``aiofiles``-style async append but attachment downloads happen via an
internal background asyncio task so callers never block on network I/O.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from nu_mom import log

__all__ = [
    "Attachment",
    "ChannelStore",
    "LoggedMessage",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Attachment:
    """Metadata for a Slack file attachment."""

    original: str
    """Original filename as supplied by the uploader."""
    local: str
    """Path relative to the working directory (e.g. ``C12345/attachments/ts_file.png``)."""


@dataclass(slots=True)
class LoggedMessage:
    """One line in a channel's ``log.jsonl``."""

    ts: str
    user: str
    text: str
    is_bot: bool
    attachments: list[Attachment] = field(default_factory=list)
    date: str = ""
    user_name: str | None = None
    display_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "date": self.date,
            "ts": self.ts,
            "user": self.user,
            "text": self.text,
            "attachments": [{"original": a.original, "local": a.local} for a in self.attachments],
            "isBot": self.is_bot,
        }
        if self.user_name is not None:
            d["userName"] = self.user_name
        if self.display_name is not None:
            d["displayName"] = self.display_name
        return d


# ---------------------------------------------------------------------------
# ChannelStore
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PendingDownload:
    channel_id: str
    local_path: str
    url: str


class ChannelStore:
    """Manages per-channel file storage in a working directory.

    Thread-safety note: all public methods are safe to call from a single
    asyncio event loop.  Parallel downloads are serialised internally.
    """

    def __init__(self, working_dir: str, bot_token: str) -> None:
        self._working_dir = working_dir
        self._bot_token = bot_token
        self._pending: list[_PendingDownload] = []
        self._downloading = False
        # Dedup: "channelId:ts" → epoch-ms when first logged.
        self._recently_logged: dict[str, float] = {}

        Path(working_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_channel_dir(self, channel_id: str) -> str:
        """Return (and create if needed) the directory for *channel_id*."""
        d = Path(self._working_dir) / channel_id
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def get_attachments_dir(self, channel_id: str) -> str:
        """Return (and create if needed) the attachments sub-directory."""
        d = Path(self._working_dir) / channel_id / "attachments"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def generate_local_filename(self, original_name: str, timestamp: str) -> str:
        """Build a unique local filename from the Slack file metadata."""
        ts_ms = math.floor(float(timestamp) * 1000)
        sanitized = "".join(c if c.isalnum() or c in "._-" else "_" for c in original_name)
        return f"{ts_ms}_{sanitized}"

    def process_attachments(
        self,
        channel_id: str,
        files: list[dict[str, Any]],
        timestamp: str,
    ) -> list[Attachment]:
        """Process Slack file metadata, queue background downloads, and return attachment list."""
        attachments: list[Attachment] = []
        for f in files:
            url = f.get("url_private_download") or f.get("url_private")
            if not url:
                continue
            name = f.get("name")
            if not name:
                log.log_warning("Attachment missing name, skipping", url)
                continue
            filename = self.generate_local_filename(name, timestamp)
            local_path = f"{channel_id}/attachments/{filename}"
            attachments.append(Attachment(original=name, local=local_path))
            self._pending.append(_PendingDownload(channel_id, local_path, url))

        # Kick off background download without awaiting
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon(self._maybe_start_downloads)
            else:
                asyncio.ensure_future(self._process_download_queue())
        except RuntimeError:
            pass  # No event loop in this context — downloads will be skipped
        return attachments

    async def log_user_message(
        self,
        channel_id: str,
        message: LoggedMessage,
    ) -> bool:
        """Append a user message to the channel log.  Returns ``False`` on duplicate."""
        return await self._log_message(channel_id, message)

    async def log_bot_response(
        self,
        channel_id: str,
        text: str,
        ts: str,
    ) -> None:
        """Append a bot response entry to the channel log."""
        import datetime

        msg = LoggedMessage(
            date=datetime.datetime.now(datetime.UTC).isoformat(),
            ts=ts,
            user="bot",
            text=text,
            attachments=[],
            is_bot=True,
        )
        await self._log_message(channel_id, msg)

    def get_last_timestamp(self, channel_id: str) -> str | None:
        """Return the ``ts`` of the last message in log.jsonl, or ``None``."""
        log_path = Path(self._working_dir) / channel_id / "log.jsonl"
        if not log_path.exists():
            return None
        try:
            content = log_path.read_text(encoding="utf-8")
            lines = [line for line in content.strip().split("\n") if line]
            if not lines:
                return None
            entry = json.loads(lines[-1])
            return entry.get("ts")
        except Exception:
            return None

    async def download_attachment(self, local_path: str, url: str) -> None:
        """Download a single Slack attachment via httpx."""
        file_path = Path(self._working_dir) / local_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {self._bot_token}"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            file_path.write_bytes(resp.content)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _log_message(self, channel_id: str, message: LoggedMessage) -> bool:
        dedupe_key = f"{channel_id}:{message.ts}"
        if dedupe_key in self._recently_logged:
            return False

        now = time.time() * 1000
        self._recently_logged[dedupe_key] = now
        # Schedule cleanup after 60 s (best-effort; no-op if loop not running)
        try:
            loop = asyncio.get_running_loop()
            loop.call_later(60.0, lambda: self._recently_logged.pop(dedupe_key, None))
        except RuntimeError:
            pass

        if not message.date:
            import datetime

            if "." in message.ts:
                dt = datetime.datetime.fromtimestamp(float(message.ts), tz=datetime.UTC)
            else:
                dt = datetime.datetime.fromtimestamp(int(message.ts) / 1000, tz=datetime.UTC)
            message.date = dt.isoformat()

        log_path = Path(self.get_channel_dir(channel_id)) / "log.jsonl"
        line = json.dumps(message.to_dict()) + "\n"
        await asyncio.to_thread(_append_text, str(log_path), line)
        return True

    def _maybe_start_downloads(self) -> None:
        if not self._downloading and self._pending:
            asyncio.ensure_future(self._process_download_queue())

    async def _process_download_queue(self) -> None:
        if self._downloading:
            return
        self._downloading = True
        try:
            while self._pending:
                item = self._pending.pop(0)
                try:
                    await self.download_attachment(item.local_path, item.url)
                except Exception as exc:
                    log.log_warning(
                        "Failed to download attachment",
                        f"{item.local_path}: {exc}",
                    )
        finally:
            self._downloading = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_text(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)
