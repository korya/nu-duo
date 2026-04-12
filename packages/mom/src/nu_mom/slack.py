"""SlackBot — Socket Mode client that drives nu-mom.

Port of ``packages/mom/src/slack.ts``.  Uses ``slack_bolt`` for the Socket
Mode app lifecycle and ``slack_sdk`` for the Web API.  All heavy I/O
(user/channel fetch, backfill) runs in asyncio; the Bolt event handlers
dispatch into an asyncio event loop via :func:`asyncio.run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from nu_mom import log

if TYPE_CHECKING:
    from nu_mom.store import Attachment, ChannelStore

__all__ = [
    "MomHandler",
    "SlackBot",
    "SlackChannel",
    "SlackEvent",
    "SlackUser",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SlackUser:
    id: str
    user_name: str
    display_name: str


@dataclass(slots=True)
class SlackChannel:
    id: str
    name: str


@dataclass(slots=True)
class SlackEvent:
    type: str  # "mention" or "dm"
    channel: str
    ts: str
    user: str
    text: str
    files: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)


class MomHandler:
    """Protocol interface implemented by the main event handler."""

    def is_running(self, channel_id: str) -> bool:
        """Return ``True`` if the agent is currently running for *channel_id*."""
        raise NotImplementedError

    async def handle_event(self, event: SlackEvent, slack: SlackBot, is_event: bool = False) -> None:
        """Process an incoming Slack or synthetic event."""
        raise NotImplementedError

    async def handle_stop(self, channel_id: str, slack: SlackBot) -> None:
        """Abort the running agent for *channel_id*."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Per-channel async queue
# ---------------------------------------------------------------------------

type _QueuedWork = Callable[[], Any]


class _ChannelQueue:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._queue: list[_QueuedWork] = []
        self._processing = False
        self._loop = loop

    def enqueue(self, work: _QueuedWork) -> None:
        self._queue.append(work)
        asyncio.run_coroutine_threadsafe(self._process_next(), self._loop)

    def size(self) -> int:
        return len(self._queue)

    async def _process_next(self) -> None:
        if self._processing or not self._queue:
            return
        self._processing = True
        work = self._queue.pop(0)
        try:
            result = work()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log.log_warning("Queue error", str(exc))
        finally:
            self._processing = False
            if self._queue:
                await self._process_next()


# ---------------------------------------------------------------------------
# SlackBot
# ---------------------------------------------------------------------------


class SlackBot:
    """Slack Socket Mode bot for nu-mom."""

    def __init__(
        self,
        handler: MomHandler,
        app_token: str,
        bot_token: str,
        working_dir: str,
        store: ChannelStore,
    ) -> None:
        self._handler = handler
        self._working_dir = working_dir
        self._store = store
        self._bot_token = bot_token
        self._app_token = app_token

        self._web = WebClient(token=bot_token)
        self._app = App(token=bot_token)

        self._bot_user_id: str | None = None
        self._startup_ts: str | None = None

        self._users: dict[str, SlackUser] = {}
        self._channels: dict[str, SlackChannel] = {}
        self._queues: dict[str, _ChannelQueue] = {}

        # asyncio loop used by Bolt event handlers (set in start())
        self._loop: asyncio.AbstractEventLoop | None = None

        self._setup_event_handlers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Slack and begin processing events."""
        self._loop = asyncio.get_event_loop()

        auth = self._web.auth_test()
        self._bot_user_id = auth["user_id"]

        await asyncio.gather(self._fetch_users(), self._fetch_channels())
        log.log_info(f"Loaded {len(self._channels)} channels, {len(self._users)} users")

        await self._backfill_all_channels()

        self._startup_ts = f"{time.time():.6f}"

        # Start Bolt socket mode in a background thread
        handler = SocketModeHandler(self._app, self._app_token)
        t = threading.Thread(target=handler.start, daemon=True)
        t.start()

        log.log_connected()

    def get_user(self, user_id: str) -> SlackUser | None:
        return self._users.get(user_id)

    def get_channel(self, channel_id: str) -> SlackChannel | None:
        return self._channels.get(channel_id)

    def get_all_users(self) -> list[SlackUser]:
        return list(self._users.values())

    def get_all_channels(self) -> list[SlackChannel]:
        return list(self._channels.values())

    async def post_message(self, channel: str, text: str) -> str:
        result = await asyncio.to_thread(self._web.chat_postMessage, channel=channel, text=text)
        return result["ts"]

    async def update_message(self, channel: str, ts: str, text: str) -> None:
        await asyncio.to_thread(self._web.chat_update, channel=channel, ts=ts, text=text)

    async def delete_message(self, channel: str, ts: str) -> None:
        await asyncio.to_thread(self._web.chat_delete, channel=channel, ts=ts)

    async def post_in_thread(self, channel: str, thread_ts: str, text: str) -> str:
        result = await asyncio.to_thread(
            self._web.chat_postMessage,
            channel=channel,
            thread_ts=thread_ts,
            text=text,
        )
        return result["ts"]

    async def upload_file(self, channel: str, file_path: str, title: str | None = None) -> None:
        name = title or Path(file_path).name
        with open(file_path, "rb") as fh:
            content = fh.read()
        await asyncio.to_thread(
            self._web.files_uploadV2,
            channel_id=channel,
            file=content,
            filename=name,
            title=name,
        )

    def log_to_file(self, channel: str, entry: dict[str, Any]) -> None:
        """Synchronously append *entry* to channel's log.jsonl."""
        d = Path(self._working_dir) / channel
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def log_bot_response(self, channel: str, text: str, ts: str) -> None:
        self.log_to_file(
            channel,
            {
                "date": _iso_now(),
                "ts": ts,
                "user": "bot",
                "text": text,
                "attachments": [],
                "isBot": True,
            },
        )

    def log_user_message(self, event: SlackEvent) -> list[Attachment]:
        """Log a user message synchronously and queue attachment downloads."""
        user = self._users.get(event.user)
        attachments = self._store.process_attachments(event.channel, event.files, event.ts) if event.files else []
        self.log_to_file(
            event.channel,
            {
                "date": _slack_ts_to_iso(event.ts),
                "ts": event.ts,
                "user": event.user,
                "userName": user.user_name if user else None,
                "displayName": user.display_name if user else None,
                "text": event.text,
                "attachments": [{"original": a.original, "local": a.local} for a in attachments],
                "isBot": False,
            },
        )
        return attachments

    def enqueue_event(self, event: SlackEvent) -> bool:
        """Enqueue a synthetic event.  Returns ``False`` if the queue is full."""
        queue = self._get_queue(event.channel)
        if queue.size() >= 5:
            log.log_warning(f"Event queue full for {event.channel}, discarding: {event.text[:50]}")
            return False
        log.log_info(f"Enqueueing event for {event.channel}: {event.text[:50]}")
        queue.enqueue(lambda: self._handler.handle_event(event, self, True))
        return True

    # ------------------------------------------------------------------
    # Private — Bolt event handlers
    # ------------------------------------------------------------------

    def _setup_event_handlers(self) -> None:
        @self._app.event("app_mention")
        def on_mention(event: dict, ack: Callable) -> None:  # type: ignore[type-arg]
            ack()
            if not self._loop:
                return

            channel = event.get("channel", "")
            if channel.startswith("D"):
                return  # handled by message event

            text = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", ""), flags=re.IGNORECASE).strip()
            ts = event.get("ts", "")
            user = event.get("user", "")

            slack_event = SlackEvent(
                type="mention",
                channel=channel,
                ts=ts,
                user=user,
                text=text,
                files=event.get("files", []),
            )
            slack_event.attachments = self.log_user_message(slack_event)

            if self._startup_ts and ts < self._startup_ts:
                log.log_info(f"[{channel}] Logged old message (pre-startup), not triggering: {text[:30]}")
                return

            if text.lower().strip() == "stop":
                if self._handler.is_running(channel):
                    asyncio.run_coroutine_threadsafe(self._handler.handle_stop(channel, self), self._loop)
                else:
                    asyncio.run_coroutine_threadsafe(self.post_message(channel, "_Nothing running_"), self._loop)
                return

            if self._handler.is_running(channel):
                asyncio.run_coroutine_threadsafe(
                    self.post_message(channel, "_Already working. Say `@mom stop` to cancel._"),
                    self._loop,
                )
            else:
                self._get_queue(channel).enqueue(lambda e=slack_event: self._handler.handle_event(e, self))

        @self._app.event("message")
        def on_message(event: dict, ack: Callable) -> None:  # type: ignore[type-arg]
            ack()
            if not self._loop:
                return

            # Skip bots, edits, etc.
            if event.get("bot_id") or not event.get("user"):
                return
            if event.get("user") == self._bot_user_id:
                return
            subtype = event.get("subtype")
            if subtype is not None and subtype != "file_share":
                return
            if not event.get("text") and not event.get("files"):
                return

            channel = event.get("channel", "")
            is_dm = event.get("channel_type") == "im"
            text = re.sub(r"<@[A-Z0-9]+>", "", event.get("text") or "", flags=re.IGNORECASE).strip()
            is_bot_mention = f"<@{self._bot_user_id}>" in (event.get("text") or "")

            # Channel @mentions handled by app_mention
            if not is_dm and is_bot_mention:
                return

            ts = event.get("ts", "")
            slack_event = SlackEvent(
                type="dm" if is_dm else "mention",
                channel=channel,
                ts=ts,
                user=event.get("user", ""),
                text=text,
                files=event.get("files", []),
            )
            slack_event.attachments = self.log_user_message(slack_event)

            if self._startup_ts and ts < self._startup_ts:
                log.log_info(f"[{channel}] Skipping old message (pre-startup): {text[:30]}")
                return

            if is_dm:
                if text.lower().strip() == "stop":
                    if self._handler.is_running(channel):
                        asyncio.run_coroutine_threadsafe(self._handler.handle_stop(channel, self), self._loop)
                    else:
                        asyncio.run_coroutine_threadsafe(self.post_message(channel, "_Nothing running_"), self._loop)
                    return

                if self._handler.is_running(channel):
                    asyncio.run_coroutine_threadsafe(
                        self.post_message(channel, "_Already working. Say `stop` to cancel._"),
                        self._loop,
                    )
                else:
                    self._get_queue(channel).enqueue(lambda e=slack_event: self._handler.handle_event(e, self))

    # ------------------------------------------------------------------
    # Private — Backfill
    # ------------------------------------------------------------------

    def _get_existing_timestamps(self, channel_id: str) -> set[str]:
        log_path = Path(self._working_dir) / channel_id / "log.jsonl"
        timestamps: set[str] = set()
        if not log_path.exists():
            return timestamps
        for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                try:
                    entry = json.loads(line)
                    if entry.get("ts"):
                        timestamps.add(entry["ts"])
                except Exception:
                    pass
        return timestamps

    async def _backfill_channel(self, channel_id: str) -> int:
        existing_ts = self._get_existing_timestamps(channel_id)
        latest_ts = max(existing_ts, key=float) if existing_ts else None

        all_messages: list[dict] = []
        cursor: str | None = None
        page_count = 0

        while True:
            kwargs: dict[str, Any] = {
                "channel": channel_id,
                "limit": 1000,
                "inclusive": False,
            }
            if latest_ts:
                kwargs["oldest"] = latest_ts
            if cursor:
                kwargs["cursor"] = cursor

            result = await asyncio.to_thread(self._web.conversations_history, **kwargs)
            if result.get("messages"):
                all_messages.extend(result["messages"])
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            page_count += 1
            if not cursor or page_count >= 3:
                break

        def is_relevant(msg: dict) -> bool:
            if not msg.get("ts") or msg["ts"] in existing_ts:
                return False
            if msg.get("user") == self._bot_user_id:
                return True
            if msg.get("bot_id"):
                return False
            sub = msg.get("subtype")
            if sub is not None and sub != "file_share":
                return False
            if not msg.get("user"):
                return False
            return not (not msg.get("text") and not msg.get("files"))

        relevant = [m for m in all_messages if is_relevant(m)]
        relevant.reverse()  # chronological

        for msg in relevant:
            is_mom = msg.get("user") == self._bot_user_id
            user = self._users.get(msg.get("user", ""))
            text = re.sub(r"<@[A-Z0-9]+>", "", msg.get("text") or "", flags=re.IGNORECASE).strip()
            attachments = (
                self._store.process_attachments(channel_id, msg["files"], msg["ts"]) if msg.get("files") else []
            )
            self.log_to_file(
                channel_id,
                {
                    "date": _slack_ts_to_iso(msg["ts"]),
                    "ts": msg["ts"],
                    "user": "bot" if is_mom else msg.get("user", ""),
                    "userName": None if is_mom else (user.user_name if user else None),
                    "displayName": None if is_mom else (user.display_name if user else None),
                    "text": text,
                    "attachments": [{"original": a.original, "local": a.local} for a in attachments],
                    "isBot": is_mom,
                },
            )

        return len(relevant)

    async def _backfill_all_channels(self) -> None:
        start = time.time()
        to_backfill = [
            (cid, ch) for cid, ch in self._channels.items() if (Path(self._working_dir) / cid / "log.jsonl").exists()
        ]
        log.log_backfill_start(len(to_backfill))
        total = 0
        for cid, ch in to_backfill:
            try:
                count = await self._backfill_channel(cid)
                if count > 0:
                    log.log_backfill_channel(ch.name, count)
                total += count
            except Exception as exc:
                log.log_warning(f"Failed to backfill #{ch.name}", str(exc))
        log.log_backfill_complete(total, (time.time() - start) * 1000)

    # ------------------------------------------------------------------
    # Private — Fetch users / channels
    # ------------------------------------------------------------------

    async def _fetch_users(self) -> None:
        cursor: str | None = None
        while True:
            result = await asyncio.to_thread(self._web.users_list, limit=200, cursor=cursor)
            for u in result.get("members") or []:
                if u.get("id") and u.get("name") and not u.get("deleted"):
                    self._users[u["id"]] = SlackUser(
                        id=u["id"],
                        user_name=u["name"],
                        display_name=u.get("real_name") or u["name"],
                    )
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break

    async def _fetch_channels(self) -> None:
        # Public + private channels
        cursor: str | None = None
        while True:
            result = await asyncio.to_thread(
                self._web.conversations_list,
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=200,
                cursor=cursor,
            )
            for c in result.get("channels") or []:
                if c.get("id") and c.get("name") and c.get("is_member"):
                    self._channels[c["id"]] = SlackChannel(id=c["id"], name=c["name"])
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break

        # DMs
        cursor = None
        while True:
            result = await asyncio.to_thread(self._web.conversations_list, types="im", limit=200, cursor=cursor)
            for im in result.get("channels") or []:
                if im.get("id"):
                    user = self._users.get(im.get("user", ""))
                    name = f"DM:{user.user_name}" if user else f"DM:{im['id']}"
                    self._channels[im["id"]] = SlackChannel(id=im["id"], name=name)
            cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break

    def _get_queue(self, channel_id: str) -> _ChannelQueue:
        if channel_id not in self._queues:
            assert self._loop
            self._queues[channel_id] = _ChannelQueue(self._loop)
        return self._queues[channel_id]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat()


def _slack_ts_to_iso(ts: str) -> str:
    import datetime

    dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.UTC)
    return dt.isoformat()
