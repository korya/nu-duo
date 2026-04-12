"""SlackContext — per-event context passed to the agent runner.

Port of the SlackContext shape defined in ``packages/mom/src/slack.ts`` and
instantiated in ``packages/mom/src/main.ts``.  In the Python port the context
is a dataclass whose mutable Slack-posting state is encapsulated here rather
than closing over local variables in ``main.py``.

The :class:`SlackContext` delegates all Slack API calls to the :class:`SlackBot`
it receives so that the agent runner and tools remain fully testable without
a live Slack connection.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nu_mom.slack import SlackBot
    from nu_mom.store import ChannelStore

__all__ = [
    "ChannelInfo",
    "MessageInfo",
    "SlackContext",
    "UserInfo",
]


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MessageInfo:
    """Metadata about the incoming Slack message that triggered this run."""

    text: str
    raw_text: str
    user: str
    channel: str
    ts: str
    attachments: list[dict[str, str]] = field(default_factory=list)
    user_name: str | None = None


@dataclass(slots=True)
class ChannelInfo:
    """Minimal channel descriptor used in the system prompt."""

    id: str
    name: str


@dataclass(slots=True)
class UserInfo:
    """Minimal user descriptor used in the system prompt."""

    id: str
    user_name: str
    display_name: str


# ---------------------------------------------------------------------------
# SlackContext
# ---------------------------------------------------------------------------

WORKING_INDICATOR = " ..."
MAX_MAIN_LENGTH = 35_000
MAX_THREAD_LENGTH = 20_000
_TRUNCATION_NOTE = "\n\n_(message truncated, ask me to elaborate on specific parts)_"


class SlackContext:
    """Wraps Slack posting state for a single event/run.

    The ``respond``/``replace_message``/``respond_in_thread`` methods form a
    serialised promise chain so that concurrent calls (from the event
    subscription) never race on the same Slack message.
    """

    def __init__(
        self,
        message: MessageInfo,
        channel_name: str | None,
        store: ChannelStore,
        channels: list[ChannelInfo],
        users: list[UserInfo],
        bot: SlackBot,
        is_event: bool = False,
    ) -> None:
        self.message = message
        self.channel_name = channel_name
        self.store = store
        self.channels = channels
        self.users = users
        self._bot = bot
        self._is_event = is_event

        self._message_ts: str | None = None
        self._thread_ts_list: list[str] = []
        self._accumulated_text = ""
        self._is_working = True
        self._update_chain: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._update_chain.set_result(None)

        # For the "Starting event" status message
        self._event_filename: str | None = None
        if is_event:
            import re

            m = re.match(r"^\[EVENT:([^:]+):", message.text)
            if m:
                self._event_filename = m.group(1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def respond(self, text: str, should_log: bool = True) -> None:
        """Post or update the main message with *text* appended."""
        await self._serial(self._do_respond(text, should_log))

    async def replace_message(self, text: str) -> None:
        """Replace the entire accumulated text with *text*."""
        await self._serial(self._do_replace(text))

    async def respond_in_thread(self, text: str) -> None:
        """Post a reply in the thread under the main message."""
        await self._serial(self._do_thread(text))

    async def set_typing(self, is_typing: bool) -> None:
        """Post an initial "_Thinking_" status message if needed."""
        if is_typing and self._message_ts is None:
            await self._serial(self._do_set_typing())

    async def upload_file(self, file_path: str, title: str | None = None) -> None:
        """Upload a file to the channel."""
        await self._bot.upload_file(self.message.channel, file_path, title)

    async def set_working(self, working: bool) -> None:
        """Toggle the " ..." working indicator on the main message."""
        await self._serial(self._do_set_working(working))

    async def delete_message(self) -> None:
        """Delete the main message and all thread replies."""
        await self._serial(self._do_delete())

    # ------------------------------------------------------------------
    # Private implementation coroutines
    # ------------------------------------------------------------------

    async def _do_respond(self, text: str, should_log: bool) -> None:
        self._accumulated_text = f"{self._accumulated_text}\n{text}" if self._accumulated_text else text
        if len(self._accumulated_text) > MAX_MAIN_LENGTH:
            cut = MAX_MAIN_LENGTH - len(_TRUNCATION_NOTE)
            self._accumulated_text = self._accumulated_text[:cut] + _TRUNCATION_NOTE

        display = self._accumulated_text + (WORKING_INDICATOR if self._is_working else "")

        if self._message_ts:
            await self._bot.update_message(self.message.channel, self._message_ts, display)
        else:
            self._message_ts = await self._bot.post_message(self.message.channel, display)

        if should_log and self._message_ts:
            self._bot.log_bot_response(self.message.channel, text, self._message_ts)

    async def _do_replace(self, text: str) -> None:
        if len(text) > MAX_MAIN_LENGTH:
            cut = MAX_MAIN_LENGTH - len(_TRUNCATION_NOTE)
            self._accumulated_text = text[:cut] + _TRUNCATION_NOTE
        else:
            self._accumulated_text = text

        display = self._accumulated_text + (WORKING_INDICATOR if self._is_working else "")

        if self._message_ts:
            await self._bot.update_message(self.message.channel, self._message_ts, display)
        else:
            self._message_ts = await self._bot.post_message(self.message.channel, display)

    async def _do_thread(self, text: str) -> None:
        if self._message_ts is None:
            return
        thread_text = text
        if len(thread_text) > MAX_THREAD_LENGTH:
            thread_text = thread_text[: MAX_THREAD_LENGTH - 50] + "\n\n_(truncated)_"
        ts = await self._bot.post_in_thread(self.message.channel, self._message_ts, thread_text)
        self._thread_ts_list.append(ts)

    async def _do_set_typing(self) -> None:
        if self._message_ts is None:
            label = f"_Starting event: {self._event_filename}_" if self._event_filename else "_Thinking_"
            self._accumulated_text = label
            self._message_ts = await self._bot.post_message(
                self.message.channel,
                self._accumulated_text + WORKING_INDICATOR,
            )

    async def _do_set_working(self, working: bool) -> None:
        self._is_working = working
        if self._message_ts:
            display = self._accumulated_text + (WORKING_INDICATOR if self._is_working else "")
            await self._bot.update_message(self.message.channel, self._message_ts, display)

    async def _do_delete(self) -> None:
        for ts in reversed(self._thread_ts_list):
            with contextlib.suppress(Exception):
                await self._bot.delete_message(self.message.channel, ts)
        self._thread_ts_list.clear()
        if self._message_ts:
            await self._bot.delete_message(self.message.channel, self._message_ts)
            self._message_ts = None

    # ------------------------------------------------------------------
    # Serialisation helper
    # ------------------------------------------------------------------

    async def _serial(self, coro: Any) -> None:
        """Await *coro* after the current update chain completes."""
        fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        prev = self._update_chain
        self._update_chain = fut

        await prev
        try:
            await coro
            fut.set_result(None)
        except Exception as exc:
            fut.set_exception(exc)
            raise
