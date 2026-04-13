"""RPC protocol types for headless operation.

Port of ``modes/rpc/rpc-types.ts``.

Commands are sent as JSON lines on stdin.
Responses and events are emitted as JSON lines on stdout.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

# ============================================================================
# RPC Commands (stdin)
# ============================================================================


class RpcCommandBase(BaseModel):
    """Base for all RPC commands."""

    id: str | None = None
    type: str

    model_config = {"extra": "allow"}


# Individual command types — we use the base class with extra="allow"
# rather than 30+ subclasses, since commands are parsed dynamically.
# The ``type`` field determines the command.

RpcCommandType = Literal[
    # Prompting
    "prompt",
    "steer",
    "follow_up",
    "abort",
    "new_session",
    # State
    "get_state",
    # Model
    "set_model",
    "cycle_model",
    "get_available_models",
    # Thinking
    "set_thinking_level",
    "cycle_thinking_level",
    # Queue modes
    "set_steering_mode",
    "set_follow_up_mode",
    # Compaction
    "compact",
    "set_auto_compaction",
    # Retry
    "set_auto_retry",
    "abort_retry",
    # Bash
    "bash",
    "abort_bash",
    # Session
    "get_session_stats",
    "export_html",
    "switch_session",
    "fork",
    "get_fork_messages",
    "get_last_assistant_text",
    "set_session_name",
    # Messages
    "get_messages",
    # Commands
    "get_commands",
]

# For type checking: the union is represented as dict[str, Any] at runtime.
RpcCommand = dict[str, Any]


# ============================================================================
# RPC State
# ============================================================================


class RpcSessionState(BaseModel):
    """Current session state."""

    model: dict[str, Any] | None = None
    thinking_level: str = "off"
    is_streaming: bool = False
    is_compacting: bool = False
    steering_mode: str = "all"
    follow_up_mode: str = "all"
    session_file: str | None = None
    session_id: str = ""
    session_name: str | None = None
    auto_compaction_enabled: bool = True
    message_count: int = 0
    pending_message_count: int = 0

    model_config = {"populate_by_name": True}


# ============================================================================
# RPC Responses (stdout)
# ============================================================================

# Responses are dicts with: id, type="response", command, success, data/error
RpcResponse = dict[str, Any]


# ============================================================================
# Extension UI Events (stdout)
# ============================================================================

RpcExtensionUIRequest = dict[str, Any]

# ============================================================================
# Extension UI Commands (stdin)
# ============================================================================

RpcExtensionUIResponse = dict[str, Any]


# ============================================================================
# RPC Slash Command
# ============================================================================


class RpcSlashCommand(BaseModel):
    """A command available for invocation via prompt."""

    name: str
    description: str | None = None
    source: Literal["extension", "prompt", "skill"]
    source_info: dict[str, Any] = {}
