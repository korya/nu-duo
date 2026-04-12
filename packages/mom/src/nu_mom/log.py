"""Logging utilities for nu-mom.

Port of ``packages/mom/src/log.ts``. Provides module-level functions that
mirror the TS surface: ``log_startup``, ``log_info``, ``log_warning``,
``log_error``, ``log_agent_message``, ``log_slack_event``, and a rich set of
contextual helpers for tool execution, response streaming, etc.

All output goes through Python's ``logging`` module rather than ``chalk``
so it integrates naturally with the host application's log config.
"""

from __future__ import annotations

import logging

__all__ = [
    "LogContext",
    "log_agent_error",
    "log_agent_message",
    "log_backfill_channel",
    "log_backfill_complete",
    "log_backfill_start",
    "log_connected",
    "log_disconnected",
    "log_download_error",
    "log_download_start",
    "log_download_success",
    "log_error",
    "log_info",
    "log_response",
    "log_response_start",
    "log_slack_event",
    "log_startup",
    "log_stop_request",
    "log_thinking",
    "log_tool_error",
    "log_tool_start",
    "log_tool_success",
    "log_usage_summary",
    "log_user_message",
    "log_warning",
]

logger = logging.getLogger("nu_mom")


# ---------------------------------------------------------------------------
# LogContext
# ---------------------------------------------------------------------------


class LogContext:
    """Carries per-event channel/user metadata for log formatting."""

    __slots__ = ("channel_id", "channel_name", "user_name")

    def __init__(
        self,
        channel_id: str,
        user_name: str | None = None,
        channel_name: str | None = None,
    ) -> None:
        self.channel_id = channel_id
        self.user_name = user_name
        self.channel_name = channel_name


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_ctx(ctx: LogContext) -> str:
    if ctx.channel_id.startswith("D"):
        return f"[DM:{ctx.user_name or ctx.channel_id}]"
    channel = ctx.channel_name or ctx.channel_id
    user = ctx.user_name or "unknown"
    prefix = channel if channel.startswith("#") else f"#{channel}"
    return f"[{prefix}:{user}]"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}\n(truncated at {max_len} chars)"


def _fmt_tool_args(args: dict) -> str:
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


def _indent(text: str) -> str:
    return "\n".join(f"           {line}" for line in text.split("\n"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_startup(working_dir: str, sandbox: str) -> None:
    """Log startup banner with working dir and sandbox type."""
    logger.info("Starting mom bot...")
    logger.info("  Working directory: %s", working_dir)
    logger.info("  Sandbox: %s", sandbox)


def log_connected() -> None:
    """Log successful Slack connection."""
    logger.info("Mom bot connected and listening!")


def log_disconnected() -> None:
    """Log Slack disconnection."""
    logger.info("Mom bot disconnected.")


def log_info(message: str) -> None:
    """Log a system-level info message."""
    logger.info("[system] %s", message)


def log_warning(message: str, details: str | None = None) -> None:
    """Log a system-level warning, with optional detail block."""
    logger.warning("[system] %s", message)
    if details:
        logger.warning(_indent(details))


def log_error(message: str, details: str | None = None) -> None:
    """Log a system-level error, with optional detail block."""
    logger.error("[system] %s", message)
    if details:
        logger.error(_indent(details))


def log_slack_event(ctx: LogContext, event_type: str) -> None:
    """Log a raw Slack event receipt."""
    logger.debug("%s [slack_event:%s]", _fmt_ctx(ctx), event_type)


def log_agent_message(ctx: LogContext, text: str) -> None:
    """Log an agent message (outgoing response)."""
    logger.info("%s [agent] %s", _fmt_ctx(ctx), _truncate(text, 200))


def log_user_message(ctx: LogContext, text: str) -> None:
    """Log an incoming user message."""
    logger.info("%s %s", _fmt_ctx(ctx), text)


def log_tool_start(
    ctx: LogContext,
    tool_name: str,
    label: str,
    args: dict,
) -> None:
    """Log tool execution start."""
    logger.info("%s → %s: %s", _fmt_ctx(ctx), tool_name, label)
    formatted = _fmt_tool_args(args)
    if formatted:
        logger.debug(_indent(formatted))


def log_tool_success(
    ctx: LogContext,
    tool_name: str,
    duration_ms: float,
    result: str,
) -> None:
    """Log successful tool execution."""
    duration = duration_ms / 1000
    logger.info("%s ✓ %s (%.1fs)", _fmt_ctx(ctx), tool_name, duration)
    truncated = _truncate(result, 1000)
    if truncated:
        logger.debug(_indent(truncated))


def log_tool_error(
    ctx: LogContext,
    tool_name: str,
    duration_ms: float,
    error: str,
) -> None:
    """Log a tool execution failure."""
    duration = duration_ms / 1000
    logger.info("%s ✗ %s (%.1fs)", _fmt_ctx(ctx), tool_name, duration)
    logger.debug(_indent(_truncate(error, 1000)))


def log_response_start(ctx: LogContext) -> None:
    """Log that the agent has started streaming a response."""
    logger.info("%s → Streaming response...", _fmt_ctx(ctx))


def log_thinking(ctx: LogContext, thinking: str) -> None:
    """Log an extended-thinking block."""
    logger.info("%s 💭 Thinking", _fmt_ctx(ctx))
    logger.debug(_indent(_truncate(thinking, 1000)))


def log_response(ctx: LogContext, text: str) -> None:
    """Log a completed assistant response."""
    logger.info("%s 💬 Response", _fmt_ctx(ctx))
    logger.debug(_indent(_truncate(text, 1000)))


def log_download_start(ctx: LogContext, filename: str, local_path: str) -> None:
    """Log attachment download start."""
    logger.info("%s ↓ Downloading attachment", _fmt_ctx(ctx))
    logger.debug("           %s → %s", filename, local_path)


def log_download_success(ctx: LogContext, size_kb: int) -> None:
    """Log successful attachment download."""
    logger.info("%s ✓ Downloaded (%s KB)", _fmt_ctx(ctx), f"{size_kb:,}")


def log_download_error(ctx: LogContext, filename: str, error: str) -> None:
    """Log attachment download failure."""
    logger.info("%s ✗ Download failed", _fmt_ctx(ctx))
    logger.debug("           %s: %s", filename, error)


def log_stop_request(ctx: LogContext) -> None:
    """Log that the user requested a stop."""
    logger.info("%s stop", _fmt_ctx(ctx))
    logger.info("%s ⊗ Stop requested - aborting", _fmt_ctx(ctx))


def log_agent_error(ctx: LogContext | None, error: str) -> None:
    """Log an agent-level error (ctx=None for system-level errors)."""
    context = "[system]" if ctx is None else _fmt_ctx(ctx)
    logger.error("%s ✗ Agent error", context)
    logger.debug(_indent(error))


def log_usage_summary(
    ctx: LogContext,
    usage: dict,
    context_tokens: int | None = None,
    context_window: int | None = None,
) -> str:
    """Format and log a usage summary.  Returns the Slack-formatted string."""

    def fmt_tokens(count: int) -> str:
        if count < 1000:
            return str(count)
        if count < 10000:
            return f"{count / 1000:.1f}k"
        if count < 1_000_000:
            return f"{round(count / 1000)}k"
        return f"{count / 1_000_000:.1f}M"

    inp = usage.get("input", 0)
    out = usage.get("output", 0)
    cache_read = usage.get("cache_read", 0)
    cache_write = usage.get("cache_write", 0)
    cost = usage.get("cost", {})

    lines: list[str] = ["*Usage Summary*"]
    lines.append(f"Tokens: {inp:,} in, {out:,} out")
    if cache_read > 0 or cache_write > 0:
        lines.append(f"Cache: {cache_read:,} read, {cache_write:,} write")
    if context_tokens and context_window:
        pct = (context_tokens / context_window) * 100
        lines.append(f"Context: {fmt_tokens(context_tokens)} / {fmt_tokens(context_window)} ({pct:.1f}%)")
    cost_line = f"Cost: ${cost.get('input', 0):.4f} in, ${cost.get('output', 0):.4f} out"
    if cache_read > 0 or cache_write > 0:
        cost_line += f", ${cost.get('cache_read', 0):.4f} cache read, ${cost.get('cache_write', 0):.4f} cache write"
    lines.append(cost_line)
    lines.append(f"*Total: ${cost.get('total', 0):.4f}*")

    summary = "\n".join(lines)

    total_cost = cost.get("total", 0)
    logger.info("%s 💰 Usage", _fmt_ctx(ctx))
    cache_info = (
        f" ({cache_read:,} cache read, {cache_write:,} cache write)" if cache_read > 0 or cache_write > 0 else ""
    )
    logger.debug(
        "           %s in + %s out%s = $%.4f",
        f"{inp:,}",
        f"{out:,}",
        cache_info,
        total_cost,
    )

    return summary


def log_backfill_start(channel_count: int) -> None:
    """Log start of channel backfill."""
    logger.info("[system] Backfilling %d channels...", channel_count)


def log_backfill_channel(channel_name: str, message_count: int) -> None:
    """Log per-channel backfill result."""
    logger.info("[system]   #%s: %d messages", channel_name, message_count)


def log_backfill_complete(total_messages: int, duration_ms: float) -> None:
    """Log completion of the full backfill pass."""
    logger.info(
        "[system] Backfill complete: %d messages in %.1fs",
        total_messages,
        duration_ms / 1000,
    )
