"""``nu-mom`` command-line entry point.

Port of ``packages/mom/src/main.ts``.

Usage::

    nu-mom [--sandbox host|docker:<name>] <working-directory>
    nu-mom --download <channel-id>

Environment variables required for bot mode:
    MOM_SLACK_APP_TOKEN
    MOM_SLACK_BOT_TOKEN
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["main"]

USAGE = """nu-mom — Slack bot delegating to pi agent

Usage:
  nu-mom [--sandbox host|docker:<name>] <working-directory>
  nu-mom --download <channel-id>
"""


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> dict:
    sandbox_str = "host"
    working_dir: str | None = None
    download_channel: str | None = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--sandbox="):
            sandbox_str = arg[len("--sandbox=") :]
        elif arg == "--sandbox":
            if i + 1 >= len(argv):
                print("--sandbox requires a value", file=sys.stderr)
                sys.exit(1)
            sandbox_str = argv[i + 1]
            i += 1
        elif arg.startswith("--download="):
            download_channel = arg[len("--download=") :]
        elif arg == "--download":
            if i + 1 >= len(argv):
                print("--download requires a value", file=sys.stderr)
                sys.exit(1)
            download_channel = argv[i + 1]
            i += 1
        elif arg in ("-h", "--help"):
            print(USAGE)
            sys.exit(0)
        elif not arg.startswith("-"):
            working_dir = str(Path(arg).resolve())
        i += 1

    return {
        "sandbox_str": sandbox_str,
        "working_dir": working_dir,
        "download_channel": download_channel,
    }


# ---------------------------------------------------------------------------
# Per-channel state
# ---------------------------------------------------------------------------


class _ChannelState:
    __slots__ = ("runner", "running", "stop_message_ts", "stop_requested", "store")

    def __init__(self, runner: object, store: object) -> None:
        self.running = False
        self.runner = runner
        self.store = store
        self.stop_requested = False
        self.stop_message_ts: str | None = None


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------


async def _run(argv: list[str]) -> int:
    from nu_mom import log
    from nu_mom.agent import get_or_create_runner
    from nu_mom.context import ChannelInfo, MessageInfo, SlackContext, UserInfo
    from nu_mom.download import download_channel
    from nu_mom.events import create_events_watcher
    from nu_mom.sandbox import parse_sandbox_arg, validate_sandbox
    from nu_mom.slack import MomHandler, SlackBot, SlackEvent
    from nu_mom.store import ChannelStore

    parsed = _parse_args(argv)
    sandbox_str: str = parsed["sandbox_str"]
    working_dir: str | None = parsed["working_dir"]
    download_ch: str | None = parsed["download_channel"]

    app_token = os.environ.get("MOM_SLACK_APP_TOKEN")
    bot_token = os.environ.get("MOM_SLACK_BOT_TOKEN")

    # --download mode
    if download_ch:
        if not bot_token:
            print("Missing env: MOM_SLACK_BOT_TOKEN", file=sys.stderr)
            return 1
        await download_channel(download_ch, bot_token)
        return 0

    # Normal bot mode
    if not working_dir:
        print(USAGE, file=sys.stderr)
        print("Usage: nu-mom [--sandbox=host|docker:<name>] <working-directory>", file=sys.stderr)
        return 1

    if not app_token or not bot_token:
        print("Missing env: MOM_SLACK_APP_TOKEN, MOM_SLACK_BOT_TOKEN", file=sys.stderr)
        return 1

    sandbox = parse_sandbox_arg(sandbox_str)
    await validate_sandbox(sandbox)

    log.log_startup(
        working_dir,
        "host" if sandbox.type == "host" else f"docker:{sandbox.container}",  # type: ignore[union-attr]
    )

    # Shared store
    store = ChannelStore(working_dir=working_dir, bot_token=bot_token)

    # Per-channel state cache
    channel_states: dict[str, _ChannelState] = {}

    def get_state(channel_id: str) -> _ChannelState:
        if channel_id not in channel_states:
            channel_dir = str(Path(working_dir) / channel_id)
            state = _ChannelState(
                runner=get_or_create_runner(sandbox, channel_id, channel_dir),
                store=store,
            )
            channel_states[channel_id] = state
        return channel_states[channel_id]

    # Handler implementation
    class _Handler(MomHandler):
        def is_running(self, channel_id: str) -> bool:
            state = channel_states.get(channel_id)
            return state.running if state else False

        async def handle_stop(self, channel_id: str, slack: SlackBot) -> None:
            state = channel_states.get(channel_id)
            if state and state.running:
                state.stop_requested = True
                state.runner.abort()  # type: ignore[union-attr]
                ts = await slack.post_message(channel_id, "_Stopping..._")
                state.stop_message_ts = ts
            else:
                await slack.post_message(channel_id, "_Nothing running_")

        async def handle_event(self, event: SlackEvent, slack: SlackBot, is_event: bool = False) -> None:
            state = get_state(event.channel)
            state.running = True
            state.stop_requested = False

            log.log_info(f"[{event.channel}] Starting run: {event.text[:50]}")

            try:
                user = slack.get_user(event.user)
                channel_obj = slack.get_channel(event.channel)

                msg = MessageInfo(
                    text=event.text,
                    raw_text=event.text,
                    user=event.user,
                    channel=event.channel,
                    ts=event.ts,
                    attachments=[{"local": a.local} for a in (event.attachments or [])],
                    user_name=user.user_name if user else None,
                )
                channels = [ChannelInfo(id=c.id, name=c.name) for c in slack.get_all_channels()]
                users = [
                    UserInfo(id=u.id, user_name=u.user_name, display_name=u.display_name) for u in slack.get_all_users()
                ]

                ctx = SlackContext(
                    message=msg,
                    channel_name=channel_obj.name if channel_obj else None,
                    store=state.store,  # type: ignore[arg-type]
                    channels=channels,
                    users=users,
                    bot=slack,
                    is_event=is_event,
                )

                await ctx.set_typing(True)
                await ctx.set_working(True)
                result = await state.runner.run(ctx, state.store)  # type: ignore[union-attr]
                await ctx.set_working(False)

                stop_reason = result.get("stop_reason", "stop") if isinstance(result, dict) else "stop"
                if stop_reason == "aborted" and state.stop_requested:
                    if state.stop_message_ts:
                        await slack.update_message(event.channel, state.stop_message_ts, "_Stopped_")
                        state.stop_message_ts = None
                    else:
                        await slack.post_message(event.channel, "_Stopped_")

            except Exception as exc:
                log.log_warning(f"[{event.channel}] Run error", str(exc))
            finally:
                state.running = False

    handler = _Handler()
    bot = SlackBot(
        handler=handler,
        app_token=app_token,
        bot_token=bot_token,
        working_dir=working_dir,
        store=store,
    )

    # Events watcher
    watcher = create_events_watcher(working_dir, bot)

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def _on_signal() -> None:
        log.log_info("Shutting down...")
        watcher.stop()
        shutdown_event.set()

    loop.add_signal_handler(signal.SIGINT, _on_signal)
    loop.add_signal_handler(signal.SIGTERM, _on_signal)

    await bot.start()
    watcher.start()

    await shutdown_event.wait()
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point registered as ``nu-mom`` in pyproject.toml."""
    if argv is None:
        argv = sys.argv[1:]
    return asyncio.run(_run(list(argv)))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
