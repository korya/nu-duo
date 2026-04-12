"""Slack channel history downloader.

Port of ``packages/mom/src/download.ts``.  Downloads a channel's full
message history (including thread replies) and prints it to stdout in a
human-readable plain-text format, with progress info written to stderr.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from slack_sdk import WebClient

__all__ = ["download_channel"]


def _format_ts(ts: str) -> str:
    """Convert a Slack timestamp to a readable date string."""
    import datetime

    dt = datetime.datetime.fromtimestamp(float(ts))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_message(ts: str, user: str, text: str, indent: str = "") -> str:
    prefix = f"[{_format_ts(ts)}] {user}: "
    lines = text.split("\n")
    first_line = f"{indent}{prefix}{lines[0]}"
    if len(lines) == 1:
        return first_line
    content_indent = indent + " " * len(prefix)
    rest = "\n".join(content_indent + line for line in lines[1:])
    return f"{first_line}\n{rest}"


async def download_channel(channel_id: str, bot_token: str) -> None:
    """Download all messages from *channel_id* and print them to stdout."""
    client = WebClient(token=bot_token)

    print(f"Fetching channel info for {channel_id}...", file=sys.stderr)
    channel_name = channel_id
    try:
        info = await asyncio.to_thread(client.conversations_info, channel=channel_id)
        channel_name = (info.get("channel") or {}).get("name") or channel_id
    except Exception:
        pass  # DM channels have no name

    print(f"Downloading history for #{channel_name} ({channel_id})...", file=sys.stderr)

    messages: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        result = await asyncio.to_thread(
            client.conversations_history,
            channel=channel_id,
            limit=200,
            cursor=cursor,
        )
        if result.get("messages"):
            messages.extend(result["messages"])
        cursor = (result.get("response_metadata") or {}).get("next_cursor")
        print(f"  Fetched {len(messages)} messages...", file=sys.stderr)
        if not cursor:
            break

    messages.reverse()  # chronological

    threads_to_fetch = [m for m in messages if (m.get("reply_count") or 0) > 0]
    print(f"Fetching {len(threads_to_fetch)} threads...", file=sys.stderr)

    thread_replies: dict[str, list[dict]] = {}
    for i, parent in enumerate(threads_to_fetch):
        print(
            f"  Thread {i + 1}/{len(threads_to_fetch)} ({parent.get('reply_count')} replies)...",
            file=sys.stderr,
        )
        replies: list[dict] = []
        thread_cursor: str | None = None
        while True:
            result = await asyncio.to_thread(
                client.conversations_replies,
                channel=channel_id,
                ts=parent["ts"],
                limit=200,
                cursor=thread_cursor,
            )
            if result.get("messages"):
                replies.extend(result["messages"][1:])  # skip parent
            thread_cursor = (result.get("response_metadata") or {}).get("next_cursor")
            if not thread_cursor:
                break
        thread_replies[parent["ts"]] = replies

    total_replies = 0
    for msg in messages:
        print(_format_message(msg["ts"], msg.get("user") or "unknown", msg.get("text") or ""))
        replies = thread_replies.get(msg["ts"]) or []
        for reply in replies:
            print(
                _format_message(
                    reply["ts"],
                    reply.get("user") or "unknown",
                    reply.get("text") or "",
                    indent="  ",
                )
            )
            total_replies += 1

    print(f"Done! {len(messages)} messages, {total_replies} thread replies", file=sys.stderr)
