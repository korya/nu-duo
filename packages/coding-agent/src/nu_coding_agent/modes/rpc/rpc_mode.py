"""RPC mode: headless operation with JSON stdin/stdout protocol.

Port of ``modes/rpc/rpc-mode.ts``.

Listens for JSON commands on stdin, outputs events and responses as JSON on stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from typing import TYPE_CHECKING, Any

from nu_coding_agent.modes.rpc.jsonl import attach_jsonl_line_reader, serialize_json_line

if TYPE_CHECKING:
    from nu_coding_agent.core.agent_session import AgentSession
    from nu_coding_agent.core.agent_session_runtime import AgentSessionRuntime

logger = logging.getLogger(__name__)


def _success(cmd_id: str | None, command: str, data: Any = None) -> dict[str, Any]:
    """Build a success response."""
    resp: dict[str, Any] = {"id": cmd_id, "type": "response", "command": command, "success": True}
    if data is not None:
        resp["data"] = data
    return resp


def _error(cmd_id: str | None, command: str, message: str) -> dict[str, Any]:
    """Build an error response."""
    return {"id": cmd_id, "type": "response", "command": command, "success": False, "error": message}


def _to_dict(obj: Any) -> Any:
    """Convert an object to a JSON-serializable dict."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(obj)
    return obj


async def run_rpc_mode(runtime: AgentSessionRuntime) -> None:
    """Run in RPC mode.

    Listens for JSON commands on stdin, outputs events and responses on stdout.
    Blocks until stdin is closed or shutdown is requested.

    Args:
        runtime: The agent session runtime that manages session lifecycle.
    """
    session: AgentSession = runtime.session
    unsubscribe_fn: object | None = None
    shutdown_requested = False

    # Pending extension UI requests
    pending_extension_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def output(obj: dict[str, Any] | object) -> None:
        """Write a JSON line to stdout."""
        if isinstance(obj, dict):
            line = serialize_json_line(obj)
        else:
            # For Pydantic models or other objects
            try:
                data = obj.model_dump() if hasattr(obj, "model_dump") else obj  # type: ignore[union-attr]
            except Exception:
                data = obj
            line = serialize_json_line(data)
        sys.stdout.write(line)
        sys.stdout.flush()

    async def rebind_session() -> None:
        """Rebind session and event subscription after session change."""
        nonlocal session, unsubscribe_fn

        session = runtime.session

        # Unsubscribe from old session events
        if unsubscribe_fn is not None and callable(unsubscribe_fn):
            unsubscribe_fn()

        # Subscribe to new session events
        def on_event(event: object) -> None:
            if isinstance(event, dict):
                output(event)
            elif hasattr(event, "model_dump"):
                output(event.model_dump())  # type: ignore[union-attr]
            else:
                output({"type": "event", "data": str(event)})

        unsubscribe_fn = session.subscribe(on_event)

        # Bind extensions with RPC UI context
        if hasattr(session, "bind_extensions"):
            await session.bind_extensions(  # type: ignore[attr-defined]
                ui_context=_create_extension_ui_context(output, pending_extension_requests),
            )

    await rebind_session()

    async def handle_command(command: dict[str, Any]) -> dict[str, Any]:
        """Handle a single RPC command and return a response."""
        nonlocal shutdown_requested

        cmd_id = command.get("id")
        cmd_type = command.get("type", "")

        try:
            return await _dispatch_command(
                cmd_type=cmd_type,
                cmd_id=cmd_id,
                command=command,
                session=session,
                runtime=runtime,
                rebind_session=rebind_session,
                output=output,
            )
        except Exception as exc:
            return _error(cmd_id, cmd_type, str(exc))

    async def handle_input_line(line: str) -> None:
        """Handle a single JSON line from stdin."""
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            output(_error(None, "parse", f"Failed to parse command: {exc}"))
            return

        # Handle extension UI responses
        if isinstance(parsed, dict) and parsed.get("type") == "extension_ui_response":
            req_id = parsed.get("id", "")
            future = pending_extension_requests.pop(req_id, None)
            if future is not None and not future.done():
                future.set_result(parsed)
            return

        response = await handle_command(parsed)
        output(response)

    # Set up stdin reader
    detach = await attach_jsonl_line_reader(handle_input_line)

    try:
        # Keep running until stdin closes
        # The attach_jsonl_line_reader will handle EOF
        while not shutdown_requested:
            await asyncio.sleep(0.1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        detach()
        if unsubscribe_fn is not None and callable(unsubscribe_fn):
            unsubscribe_fn()
        await runtime.dispose()


async def _dispatch_command(
    *,
    cmd_type: str,
    cmd_id: str | None,
    command: dict[str, Any],
    session: AgentSession,
    runtime: AgentSessionRuntime,
    rebind_session: Any,
    output: Any,
) -> dict[str, Any]:
    """Dispatch a command to the appropriate handler."""

    # =========================================================================
    # Prompting
    # =========================================================================

    if cmd_type == "prompt":
        message = command.get("message", "")
        # Don't await — events will stream
        asyncio.create_task(_safe_prompt(session, message, command, cmd_id, output))
        return _success(cmd_id, "prompt")

    if cmd_type == "steer":
        message = command.get("message", "")
        from nu_ai import TextContent as TC
        from nu_ai import UserMessage

        ts = int(time.time() * 1000)
        session.agent.steer(UserMessage(role="user", content=[TC(type="text", text=message)], timestamp=ts))
        return _success(cmd_id, "steer")

    if cmd_type == "follow_up":
        message = command.get("message", "")
        from nu_ai import TextContent as TC2
        from nu_ai import UserMessage

        ts = int(time.time() * 1000)
        session.agent.follow_up(UserMessage(role="user", content=[TC2(type="text", text=message)], timestamp=ts))
        return _success(cmd_id, "follow_up")

    if cmd_type == "abort":
        session.agent.abort()
        return _success(cmd_id, "abort")

    if cmd_type == "new_session":
        parent_session = command.get("parentSession")
        kwargs: dict[str, Any] = {}
        if parent_session:
            kwargs["parent_session"] = parent_session
        cancelled = await runtime.new_session(**kwargs)
        if not cancelled:
            await rebind_session()
        return _success(cmd_id, "new_session", {"cancelled": cancelled})

    # =========================================================================
    # State
    # =========================================================================

    if cmd_type == "get_state":
        stats = session.get_stats()
        state = {
            "model": _to_dict(session.model) if session.model else None,
            "thinkingLevel": getattr(session, "thinking_level", "off"),
            "isStreaming": getattr(session, "is_streaming", False),
            "isCompacting": getattr(session, "is_compacting", False),
            "steeringMode": getattr(session, "steering_mode", "all"),
            "followUpMode": getattr(session, "follow_up_mode", "all"),
            "sessionFile": session.session_manager.get_session_file(),
            "sessionId": getattr(session, "session_id", ""),
            "sessionName": getattr(session, "session_name", None),
            "autoCompactionEnabled": getattr(session, "auto_compaction_enabled", True),
            "messageCount": len(session.agent.state.messages),
            "pendingMessageCount": getattr(session, "pending_message_count", 0),
        }
        return _success(cmd_id, "get_state", state)

    # =========================================================================
    # Model
    # =========================================================================

    if cmd_type == "set_model":
        provider = command.get("provider", "")
        model_id = command.get("modelId", "")
        models = session.model_registry.get_available()
        model = next(
            (m for m in models if m.provider == provider and m.id == model_id),
            None,
        )
        if model is None:
            return _error(cmd_id, "set_model", f"Model not found: {provider}/{model_id}")
        session.set_model(model)
        return _success(cmd_id, "set_model", _to_dict(model))

    if cmd_type == "cycle_model":
        if hasattr(session, "cycle_model"):
            result = await session.cycle_model()  # type: ignore[attr-defined]
            return _success(cmd_id, "cycle_model", result)
        return _success(cmd_id, "cycle_model", None)

    if cmd_type == "get_available_models":
        models = session.model_registry.get_available()
        return _success(
            cmd_id,
            "get_available_models",
            {
                "models": [_to_dict(m) for m in models],
            },
        )

    # =========================================================================
    # Thinking
    # =========================================================================

    if cmd_type == "set_thinking_level":
        level = command.get("level", "off")
        if hasattr(session, "set_thinking_level"):
            session.set_thinking_level(level)  # type: ignore[attr-defined]
        return _success(cmd_id, "set_thinking_level")

    if cmd_type == "cycle_thinking_level":
        if hasattr(session, "cycle_thinking_level"):
            level = session.cycle_thinking_level()  # type: ignore[attr-defined]
            return _success(cmd_id, "cycle_thinking_level", {"level": level} if level else None)
        return _success(cmd_id, "cycle_thinking_level", None)

    # =========================================================================
    # Queue modes
    # =========================================================================

    if cmd_type == "set_steering_mode":
        if hasattr(session, "set_steering_mode"):
            session.set_steering_mode(command.get("mode", "all"))  # type: ignore[attr-defined]
        return _success(cmd_id, "set_steering_mode")

    if cmd_type == "set_follow_up_mode":
        if hasattr(session, "set_follow_up_mode"):
            session.set_follow_up_mode(command.get("mode", "all"))  # type: ignore[attr-defined]
        return _success(cmd_id, "set_follow_up_mode")

    # =========================================================================
    # Compaction
    # =========================================================================

    if cmd_type == "compact":
        custom_instructions = command.get("customInstructions")
        result = await session.compact(custom_instructions)
        return _success(cmd_id, "compact", _to_dict(result))

    if cmd_type == "set_auto_compaction":
        if hasattr(session, "set_auto_compaction_enabled"):
            session.set_auto_compaction_enabled(command.get("enabled", True))  # type: ignore[attr-defined]
        return _success(cmd_id, "set_auto_compaction")

    # =========================================================================
    # Retry
    # =========================================================================

    if cmd_type == "set_auto_retry":
        if hasattr(session, "set_auto_retry_enabled"):
            session.set_auto_retry_enabled(command.get("enabled", True))  # type: ignore[attr-defined]
        return _success(cmd_id, "set_auto_retry")

    if cmd_type == "abort_retry":
        if hasattr(session, "abort_retry"):
            session.abort_retry()  # type: ignore[attr-defined]
        return _success(cmd_id, "abort_retry")

    # =========================================================================
    # Bash
    # =========================================================================

    if cmd_type == "bash":
        cmd = command.get("command", "")
        if hasattr(session, "execute_bash"):
            result = await session.execute_bash(cmd)  # type: ignore[attr-defined]
            return _success(cmd_id, "bash", result)
        return _error(cmd_id, "bash", "Bash execution not available")

    if cmd_type == "abort_bash":
        if hasattr(session, "abort_bash"):
            session.abort_bash()  # type: ignore[attr-defined]
        return _success(cmd_id, "abort_bash")

    # =========================================================================
    # Session
    # =========================================================================

    if cmd_type == "get_session_stats":
        stats = session.get_stats()
        return _success(cmd_id, "get_session_stats", _to_dict(stats))

    if cmd_type == "export_html":
        output_path = command.get("outputPath")
        if hasattr(session, "export_to_html"):
            path = await session.export_to_html(output_path)  # type: ignore[attr-defined]
            return _success(cmd_id, "export_html", {"path": path})
        return _error(cmd_id, "export_html", "HTML export not available")

    if cmd_type == "switch_session":
        session_path = command.get("sessionPath", "")
        cancelled = await runtime.switch_session(session_path)
        if not cancelled:
            await rebind_session()
        return _success(cmd_id, "switch_session", {"cancelled": cancelled})

    if cmd_type == "fork":
        entry_id = command.get("entryId", "")
        fork_cancelled, selected_text = await runtime.fork(entry_id)
        if not fork_cancelled:
            await rebind_session()
        return _success(
            cmd_id,
            "fork",
            {
                "text": selected_text or "",
                "cancelled": fork_cancelled,
            },
        )

    if cmd_type == "get_fork_messages":
        if hasattr(session, "get_user_messages_for_forking"):
            messages = session.get_user_messages_for_forking()  # type: ignore[attr-defined]
            return _success(cmd_id, "get_fork_messages", {"messages": messages})
        return _success(cmd_id, "get_fork_messages", {"messages": []})

    if cmd_type == "get_last_assistant_text":
        if hasattr(session, "get_last_assistant_text"):
            text = session.get_last_assistant_text()  # type: ignore[attr-defined]
            return _success(cmd_id, "get_last_assistant_text", {"text": text})
        # Fall back to checking messages
        from nu_ai import AssistantMessage, TextContent

        messages = session.agent.state.messages
        for msg in reversed(messages):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextContent):
                        return _success(cmd_id, "get_last_assistant_text", {"text": block.text})
        return _success(cmd_id, "get_last_assistant_text", {"text": None})

    if cmd_type == "set_session_name":
        name = command.get("name", "").strip()
        if not name:
            return _error(cmd_id, "set_session_name", "Session name cannot be empty")
        if hasattr(session, "set_session_name"):
            session.set_session_name(name)  # type: ignore[attr-defined]
        return _success(cmd_id, "set_session_name")

    # =========================================================================
    # Messages
    # =========================================================================

    if cmd_type == "get_messages":
        messages = session.agent.state.messages
        msg_dicts = [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]
        return _success(cmd_id, "get_messages", {"messages": msg_dicts})

    # =========================================================================
    # Commands (available for invocation via prompt)
    # =========================================================================

    if cmd_type == "get_commands":
        commands: list[dict[str, Any]] = []

        # Extension commands
        runner = session.extension_runner
        if runner is not None and hasattr(runner, "get_registered_commands"):
            for cmd in runner.get_registered_commands():  # type: ignore[attr-defined]
                commands.append(
                    {
                        "name": getattr(cmd, "invocation_name", ""),
                        "description": getattr(cmd, "description", None),
                        "source": "extension",
                        "sourceInfo": getattr(cmd, "source_info", {}),
                    }
                )

        return _success(cmd_id, "get_commands", {"commands": commands})

    # Unknown command
    return _error(cmd_id, cmd_type, f"Unknown command: {cmd_type}")


async def _safe_prompt(
    session: AgentSession,
    message: str,
    command: dict[str, Any],
    cmd_id: str | None,
    output: Any,
) -> None:
    """Safely run a prompt, catching errors."""
    try:
        images = command.get("images")
        await session.prompt(message, images=images)
    except Exception as exc:
        output(_error(cmd_id, "prompt", str(exc)))


def _create_extension_ui_context(
    output: Any,
    pending_requests: dict[str, asyncio.Future[dict[str, Any]]],
) -> dict[str, Any]:
    """Create an extension UI context that uses the RPC protocol.

    Returns a dict with UI method implementations that emit RPC requests
    and wait for RPC responses.
    """

    async def select(title: str, options: list[str], **kwargs: Any) -> str | None:
        req_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        pending_requests[req_id] = future
        output(
            {
                "type": "extension_ui_request",
                "id": req_id,
                "method": "select",
                "title": title,
                "options": options,
            }
        )
        response = await future
        if response.get("cancelled"):
            return None
        return response.get("value")

    async def confirm(title: str, message: str, **kwargs: Any) -> bool:
        req_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        pending_requests[req_id] = future
        output(
            {
                "type": "extension_ui_request",
                "id": req_id,
                "method": "confirm",
                "title": title,
                "message": message,
            }
        )
        response = await future
        if response.get("cancelled"):
            return False
        return response.get("confirmed", False)

    async def input_text(title: str, placeholder: str | None = None, **kwargs: Any) -> str | None:
        req_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        pending_requests[req_id] = future
        output(
            {
                "type": "extension_ui_request",
                "id": req_id,
                "method": "input",
                "title": title,
                "placeholder": placeholder,
            }
        )
        response = await future
        if response.get("cancelled"):
            return None
        return response.get("value")

    def notify(message: str, notify_type: str | None = None) -> None:
        output(
            {
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "notify",
                "message": message,
                "notifyType": notify_type,
            }
        )

    def set_status(key: str, text: str | None) -> None:
        output(
            {
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "setStatus",
                "statusKey": key,
                "statusText": text,
            }
        )

    def set_title(title: str) -> None:
        output(
            {
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "setTitle",
                "title": title,
            }
        )

    def set_editor_text(text: str) -> None:
        output(
            {
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "set_editor_text",
                "text": text,
            }
        )

    return {
        "select": select,
        "confirm": confirm,
        "input": input_text,
        "notify": notify,
        "set_status": set_status,
        "set_title": set_title,
        "set_editor_text": set_editor_text,
        "on_terminal_input": lambda: lambda: None,
        "set_working_message": lambda _msg=None: None,
        "set_hidden_thinking_label": lambda _label=None: None,
        "set_widget": lambda _key, _content, **_kw: None,
        "set_footer": lambda _factory: None,
        "set_header": lambda _factory: None,
        "paste_to_editor": set_editor_text,
        "get_editor_text": lambda: "",
        "set_tools_expanded": lambda _expanded: None,
        "get_tools_expanded": lambda: False,
    }
