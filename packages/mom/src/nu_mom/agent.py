"""AgentRunner — bridges SlackContext to nu_coding_agent.

Port of ``packages/mom/src/agent.ts``.  Creates and caches one
:class:`nu_coding_agent.core.agent_session.AgentSession` per channel, feeds
each user turn through ``session.prompt()``, and streams tool/response events
back to Slack via the :class:`~nu_mom.context.SlackContext`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from nu_coding_agent.core.session_manager import SessionManager

from nu_mom import log
from nu_mom.tools import create_mom_tools

if TYPE_CHECKING:
    from nu_agent_core.types import AgentEvent

    from nu_mom.context import SlackContext
    from nu_mom.sandbox import SandboxConfig
    from nu_mom.store import ChannelStore

__all__ = [
    "AgentRunner",
    "get_or_create_runner",
]

SLACK_MAX_LENGTH = 40_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_tool_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and "content" in result:
        parts = result["content"]
        texts = [p["text"] for p in parts if isinstance(p, dict) and p.get("type") == "text" and p.get("text")]
        if texts:
            return "\n".join(texts)
    return str(result)


def _format_tool_args_for_slack(args: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in args.items():
        if key == "label":
            continue
        if key == "path" and isinstance(value, str):
            offset = args.get("offset")
            limit = args.get("limit")
            if offset is not None and limit is not None:
                lines.append(f"{value}:{offset}-{offset + limit}")
            else:
                lines.append(value)
            continue
        if key in ("offset", "limit"):
            continue
        lines.append(value if isinstance(value, str) else str(value))
    return "\n".join(lines)


def _get_memory(channel_dir: str) -> str:
    parts: list[str] = []
    workspace_memory = Path(channel_dir).parent / "MEMORY.md"
    if workspace_memory.exists():
        try:
            content = workspace_memory.read_text("utf-8").strip()
            if content:
                parts.append(f"### Global Workspace Memory\n{content}")
        except Exception as exc:
            log.log_warning("Failed to read workspace memory", str(exc))

    channel_memory = Path(channel_dir) / "MEMORY.md"
    if channel_memory.exists():
        try:
            content = channel_memory.read_text("utf-8").strip()
            if content:
                parts.append(f"### Channel-Specific Memory\n{content}")
        except Exception as exc:
            log.log_warning("Failed to read channel memory", str(exc))

    return "\n\n".join(parts) if parts else "(no working memory yet)"


def _build_system_prompt(
    workspace_path: str,
    channel_id: str,
    memory: str,
    sandbox_config: SandboxConfig,
    channels: list[Any],
    users: list[Any],
) -> str:
    import zoneinfo

    channel_path = f"{workspace_path}/{channel_id}"
    is_docker = sandbox_config.type == "docker"

    channel_mappings = "\n".join(f"{c.id}\t#{c.name}" for c in channels) if channels else "(no channels loaded)"
    user_mappings = (
        "\n".join(f"{u.id}\t@{u.user_name}\t{u.display_name}" for u in users) if users else "(no users loaded)"
    )

    try:
        local_tz = str(zoneinfo.ZoneInfo(os.environ.get("TZ", "UTC")).key)
    except Exception:
        local_tz = "UTC"

    if is_docker:
        env_desc = (
            "You are running inside a Docker container (Alpine Linux).\n"
            "- Bash working directory: / (use cd or absolute paths)\n"
            "- Install tools with: apk add <package>\n"
            "- Your changes persist across sessions"
        )
    else:
        env_desc = (
            f"You are running directly on the host machine.\n"
            f"- Bash working directory: {os.getcwd()}\n"
            "- Be careful with system modifications"
        )

    return f"""You are mom, a Slack bot assistant. Be concise. No emojis.

## Context
- For current date/time, use: date
- You have access to previous conversation context including tool results from prior turns.
- For older history beyond your context, search log.jsonl (contains user messages and your final responses, but not tool results).

## Slack Formatting (mrkdwn, NOT Markdown)
Bold: *text*, Italic: _text_, Code: `code`, Block: ```code```, Links: <url|text>
Do NOT use **double asterisks** or [markdown](links).

## Slack IDs
Channels: {channel_mappings}

Users: {user_mappings}

When mentioning users, use <@username> format (e.g., <@mario>).

## Environment
{env_desc}

## Workspace Layout
{workspace_path}/
├── MEMORY.md                    # Global memory (all channels)
├── skills/                      # Global CLI tools you create
└── {channel_id}/                # This channel
    ├── MEMORY.md                # Channel-specific memory
    ├── log.jsonl                # Message history (no tool results)
    ├── attachments/             # User-shared files
    ├── scratch/                 # Your working directory
    └── skills/                  # Channel-specific tools

## Memory
Write to MEMORY.md files to persist context across conversations.
- Global ({workspace_path}/MEMORY.md): skills, preferences, project info
- Channel ({channel_path}/MEMORY.md): channel-specific decisions, ongoing work
Update when you learn something important or when asked to remember something.

### Current Memory
{memory}

## Events
Schedule events in {workspace_path}/events/ (JSON files).
Types: immediate, one-shot (at: ISO8601), periodic (schedule: cron, timezone: IANA).
The harness runs in {local_tz}.

## Log Queries (for older history)
Format: {{"date":"...","ts":"...","user":"...","userName":"...","text":"...","isBot":false}}
```bash
tail -30 log.jsonl | jq -c '{{date: .date[0:19], user: (.userName // .user), text}}'
grep -i "topic" log.jsonl | jq -c '{{date: .date[0:19], user: (.userName // .user), text}}'
```

## Tools
- bash: Run shell commands (primary tool).
- read: Read files
- write: Create/overwrite files
- edit: Surgical file edits
- attach: Share files to Slack

Each tool requires a "label" parameter (shown to user).
"""


def _split_for_slack(text: str) -> list[str]:
    if len(text) <= SLACK_MAX_LENGTH:
        return [text]
    parts: list[str] = []
    remaining = text
    part_num = 1
    while remaining:
        chunk = remaining[: SLACK_MAX_LENGTH - 50]
        remaining = remaining[SLACK_MAX_LENGTH - 50 :]
        suffix = f"\n_(continued {part_num}...)_" if remaining else ""
        parts.append(chunk + suffix)
        part_num += 1
    return parts


def _get_image_mime_type(filename: str) -> str | None:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext)


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------


class AgentRunner:
    """Wraps a single-channel agent session with Slack-aware event handling."""

    def __init__(
        self,
        sandbox_config: SandboxConfig,
        channel_id: str,
        channel_dir: str,
    ) -> None:
        self._sandbox_config = sandbox_config
        self._channel_id = channel_id
        self._channel_dir = channel_dir
        self._session: Any | None = None
        self._abort_event = asyncio.Event()

        # Per-run state (reset each call to run())
        self._ctx: SlackContext | None = None
        self._log_ctx: log.LogContext | None = None
        self._pending_tools: dict[str, dict] = {}
        self._total_usage: dict = {}
        self._stop_reason = "stop"
        self._error_message: str | None = None
        self._upload_fn: Any = None

    async def _ensure_session(self) -> Any:
        """Lazily create the agent session on first use."""
        if self._session is not None:
            return self._session

        from nu_mom.sandbox import create_executor

        executor = create_executor(self._sandbox_config)
        workspace_path = executor.get_workspace_path(str(Path(self._channel_dir).parent))
        tools = create_mom_tools(executor, self)

        context_file = str(Path(self._channel_dir) / "context.jsonl")
        session_manager = SessionManager.open(context_file, self._channel_dir)

        opts = CreateAgentSessionOptions(
            cwd=self._channel_dir,
            tools=tools,
            session_manager=session_manager,
        )
        result = await create_agent_session(opts)
        self._session = result.session
        self._workspace_path = workspace_path

        self._session.subscribe(self._on_event)
        return self._session

    async def run(
        self,
        ctx: SlackContext,
        store: ChannelStore,
        pending_messages: list | None = None,
    ) -> dict[str, Any]:
        """Run one agent turn for *ctx*, streaming updates to Slack."""
        from nu_mom.sandbox import create_executor

        session = await self._ensure_session()
        Path(self._channel_dir).mkdir(parents=True, exist_ok=True)

        executor = create_executor(self._sandbox_config)
        workspace_path = executor.get_workspace_path(str(Path(self._channel_dir).parent))

        # Update system prompt with fresh context
        memory = _get_memory(self._channel_dir)
        system_prompt = _build_system_prompt(
            workspace_path,
            self._channel_id,
            memory,
            self._sandbox_config,
            ctx.channels,
            ctx.users,
        )
        session._agent.state.system_prompt = system_prompt

        # Set upload function for attach tool
        self._upload_fn = ctx.upload_file

        # Reset per-run state
        self._ctx = ctx
        self._log_ctx = log.LogContext(
            channel_id=ctx.message.channel,
            user_name=ctx.message.user_name,
            channel_name=ctx.channel_name,
        )
        self._pending_tools.clear()
        self._total_usage = {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        }
        self._stop_reason = "stop"
        self._error_message = None
        self._abort_event.clear()

        # Build queued message pipeline
        queue_future: asyncio.Future = asyncio.get_event_loop().create_future()
        queue_future.set_result(None)

        def enqueue_fn(fn: Any, _ctx: str = "") -> None:
            nonlocal queue_future
            prev = queue_future
            new_fut: asyncio.Future = asyncio.get_event_loop().create_future()
            queue_future = new_fut

            async def _run() -> None:
                await prev
                try:
                    await fn()
                    new_fut.set_result(None)
                except Exception as exc:
                    log.log_warning(f"Slack API error ({_ctx})", str(exc))
                    new_fut.set_result(None)

            asyncio.ensure_future(_run())

        def enqueue_msg(text: str, target: str, error_ctx: str, do_log: bool = True) -> None:
            for part in _split_for_slack(text):
                if target == "main":
                    enqueue_fn(lambda p=part, dl=do_log: ctx.respond(p, dl), error_ctx)
                else:
                    enqueue_fn(lambda p=part: ctx.respond_in_thread(p), error_ctx)

        self._enqueue_fn = enqueue_fn
        self._enqueue_msg = enqueue_msg

        log.log_info(f"Context sizes - system: {len(system_prompt)} chars, memory: {len(memory)} chars")
        log.log_info(f"Channels: {len(ctx.channels)}, Users: {len(ctx.users)}")

        # Build user message with timestamp prefix
        now_obj = __import__("datetime").datetime.now()
        offset_min = -now_obj.utcoffset().seconds // 60 if now_obj.utcoffset() else 0
        offset_sign = "+" if offset_min >= 0 else "-"
        offset_h = abs(offset_min) // 60
        offset_m = abs(offset_min) % 60

        import datetime

        now = datetime.datetime.now()
        ts_str = (
            f"{now.year}-{now.month:02d}-{now.day:02d} "
            f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
            f"{offset_sign}{offset_h:02d}:{offset_m:02d}"
        )
        user_message = f"[{ts_str}] [{ctx.message.user_name or 'unknown'}]: {ctx.message.text}"

        # Attach non-image files as path references
        non_image_paths: list[str] = []
        for a in ctx.message.attachments or []:
            full_path = f"{workspace_path}/{a['local']}"
            if not _get_image_mime_type(a["local"]):
                non_image_paths.append(full_path)

        if non_image_paths:
            user_message += "\n\n<slack_attachments>\n" + "\n".join(non_image_paths) + "\n</slack_attachments>"

        await session.prompt(user_message)
        await queue_future

        # Handle error
        if self._stop_reason == "error" and self._error_message:
            try:
                await ctx.replace_message("_Sorry, something went wrong_")
                await ctx.respond_in_thread(f"_Error: {self._error_message}_")
            except Exception as exc:
                log.log_warning("Failed to post error message", str(exc))
        else:
            # Get final text from last assistant message
            messages = session._agent.state.messages
            final_text = ""
            for msg in reversed(messages):
                if hasattr(msg, "role") and msg.role == "assistant":
                    parts = [
                        c.text if hasattr(c, "text") else c.get("text", "")
                        for c in (msg.content or [])
                        if (hasattr(c, "type") and c.type == "text")
                        or (isinstance(c, dict) and c.get("type") == "text")
                    ]
                    final_text = "\n".join(parts)
                    break

            stripped = final_text.strip()
            if stripped == "[SILENT]" or stripped.startswith("[SILENT]"):
                try:
                    await ctx.delete_message()
                    log.log_info("Silent response - deleted message and thread")
                except Exception as exc:
                    log.log_warning("Failed to delete message for silent response", str(exc))
            elif stripped:
                try:
                    main_text = (
                        f"{final_text[: SLACK_MAX_LENGTH - 50]}\n\n_(see thread for full response)_"
                        if len(final_text) > SLACK_MAX_LENGTH
                        else final_text
                    )
                    await ctx.replace_message(main_text)
                except Exception as exc:
                    log.log_warning("Failed to replace message with final text", str(exc))

        # Usage summary
        total_cost = self._total_usage.get("cost", {}).get("total", 0)
        if total_cost > 0:
            summary = log.log_usage_summary(self._log_ctx, self._total_usage)
            enqueue_fn(lambda s=summary: ctx.respond_in_thread(s), "usage summary")
            await queue_future

        self._ctx = None
        self._log_ctx = None

        return {"stop_reason": self._stop_reason, "error_message": self._error_message}

    def abort(self) -> None:
        """Signal the running session to abort."""
        self._abort_event.set()
        if self._session:
            with contextlib.suppress(Exception):
                self._session.abort()

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    async def _on_event(self, event: AgentEvent) -> None:
        ctx = self._ctx
        log_ctx = self._log_ctx
        if ctx is None or log_ctx is None:
            return

        etype = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

        if etype == "tool_execution_start":
            tool_name = event.get("tool_name") if isinstance(event, dict) else getattr(event, "tool_name", "")
            args = event.get("args") if isinstance(event, dict) else getattr(event, "args", {})
            tool_call_id = event.get("tool_call_id") if isinstance(event, dict) else getattr(event, "tool_call_id", "")
            label = (args or {}).get("label", tool_name)
            self._pending_tools[tool_call_id] = {"tool_name": tool_name, "args": args, "start_time": time.time() * 1000}
            log.log_tool_start(log_ctx, tool_name, label, args or {})
            self._enqueue_fn(lambda lb=label: ctx.respond(f"_→ {lb}_", False), "tool label")

        elif etype == "tool_execution_end":
            tool_name = event.get("tool_name") if isinstance(event, dict) else getattr(event, "tool_name", "")
            tool_call_id = event.get("tool_call_id") if isinstance(event, dict) else getattr(event, "tool_call_id", "")
            result = event.get("result") if isinstance(event, dict) else getattr(event, "result", None)
            is_error = event.get("is_error") if isinstance(event, dict) else getattr(event, "is_error", False)
            result_str = _extract_tool_result_text(result)
            pending = self._pending_tools.pop(tool_call_id, None)
            duration_ms = time.time() * 1000 - pending["start_time"] if pending else 0

            if is_error:
                log.log_tool_error(log_ctx, tool_name, duration_ms, result_str)
            else:
                log.log_tool_success(log_ctx, tool_name, duration_ms, result_str)

            label = (pending["args"] or {}).get("label") if pending else None
            args_formatted = _format_tool_args_for_slack(pending["args"] or {}) if pending else "(args not found)"
            duration_s = duration_ms / 1000
            mark = "✗" if is_error else "✓"
            thread_msg = f"*{mark} {tool_name}*"
            if label:
                thread_msg += f": {label}"
            thread_msg += f" ({duration_s:.1f}s)\n"
            if args_formatted:
                thread_msg += f"```\n{args_formatted}\n```\n"
            thread_msg += f"*Result:*\n```\n{result_str}\n```"

            self._enqueue_msg(thread_msg, "thread", "tool result thread", False)
            if is_error:
                self._enqueue_fn(
                    lambda rs=result_str: ctx.respond(f"_Error: {_truncate(rs, 200)}_", False),
                    "tool error",
                )

        elif etype == "message_end":
            message = event.get("message") if isinstance(event, dict) else getattr(event, "message", None)
            if message is None:
                return
            role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
            if role != "assistant":
                return

            log.log_response_start(log_ctx)

            # Accumulate usage
            usage = getattr(message, "usage", None) or (message.get("usage") if isinstance(message, dict) else None)
            if usage:
                self._total_usage["input"] = self._total_usage.get("input", 0) + getattr(
                    usage, "input", usage.get("input", 0) if isinstance(usage, dict) else 0
                )
                self._total_usage["output"] = self._total_usage.get("output", 0) + getattr(
                    usage, "output", usage.get("output", 0) if isinstance(usage, dict) else 0
                )

            content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else [])
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            for part in content or []:
                ptype = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
                if ptype == "thinking":
                    thinking_parts.append(
                        getattr(part, "thinking", None) or (part.get("thinking", "") if isinstance(part, dict) else "")
                    )
                elif ptype == "text":
                    text_parts.append(
                        getattr(part, "text", None) or (part.get("text", "") if isinstance(part, dict) else "")
                    )

            text = "\n".join(text_parts)

            for thinking in thinking_parts:
                log.log_thinking(log_ctx, thinking)
                self._enqueue_msg(f"_{thinking}_", "main", "thinking main")
                self._enqueue_msg(f"_{thinking}_", "thread", "thinking thread", False)

            if text.strip():
                log.log_response(log_ctx, text)
                self._enqueue_msg(text, "main", "response main")
                self._enqueue_msg(text, "thread", "response thread", False)


# ---------------------------------------------------------------------------
# Channel runner cache
# ---------------------------------------------------------------------------

_channel_runners: dict[str, AgentRunner] = {}


def get_or_create_runner(
    sandbox_config: SandboxConfig,
    channel_id: str,
    channel_dir: str,
) -> AgentRunner:
    """Return the cached runner for *channel_id*, creating it if necessary."""
    if channel_id not in _channel_runners:
        _channel_runners[channel_id] = AgentRunner(sandbox_config, channel_id, channel_dir)
    return _channel_runners[channel_id]
