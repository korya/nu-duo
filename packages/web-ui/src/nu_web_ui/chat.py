"""WebSocket endpoint for streaming agent events to the frontend.

Protocol
--------
Client → server (on connect, via query param or first message):
  ``session_id`` — optional UUID of an existing session to resume.

Server → client (JSON messages):
  Each message is a serialized ``AgentEvent`` TypedDict, e.g.::

      {"type": "message_start", "message": {...}}
      {"type": "message_update", "message": {...}, "assistant_message_event": {...}}
      {"type": "agent_end", "messages": [...]}

  Additionally the server sends control frames::

      {"type": "connected", "session_id": "<uuid>"}
      {"type": "error", "message": "<text>"}

Client → server (after connection, as JSON):
  ``{"type": "prompt", "content": "user text"}``  — send a new prompt.
  ``{"type": "abort"}``                           — abort current run.
  ``{"type": "ping"}``                            — keep-alive.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import UTC
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect
from nu_agent_core.agent import Agent, AgentOptions
from nu_ai.types import TextContent, UserMessage

from nu_web_ui.types import CostBreakdown, SessionData, SessionInfo, UsageStats

if TYPE_CHECKING:
    from nu_web_ui.storage import ProviderKeysStore, SessionsStore, SettingsStore

logger = logging.getLogger(__name__)


def _make_empty_usage() -> UsageStats:
    return UsageStats(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=CostBreakdown(),
    )


async def chat_websocket(
    websocket: WebSocket,
    sessions_store: SessionsStore,
    settings_store: SettingsStore,
    provider_keys_store: ProviderKeysStore,
    session_id: str | None = None,
) -> None:
    """Handle a WebSocket connection for streaming agent chat.

    Args:
        websocket: The FastAPI WebSocket connection.
        sessions_store: Store for loading/saving session data.
        settings_store: Store for loading application settings (proxy, etc.).
        provider_keys_store: Store for looking up provider API keys.
        session_id: Optional session UUID to resume.  A new session is created
            if ``None`` or not found.
    """
    await websocket.accept()

    # Resolve or create session.
    active_session_id = session_id or str(uuid.uuid4())
    existing_data: SessionData | None = None
    if session_id:
        existing_data = await sessions_store.get_session(session_id)

    now_iso = _now_iso()

    if existing_data is None:
        existing_data = SessionData(
            id=active_session_id,
            title="",
            model=None,
            thinkingLevel="off",
            messages=[],
            createdAt=now_iso,
            lastModified=now_iso,
        )

    # Build the agent.
    await settings_store.get_settings()

    async def _get_api_key(provider: str) -> str | None:
        return await provider_keys_store.get_key(provider)

    agent = Agent(
        AgentOptions(
            get_api_key=_get_api_key,
            initial_state={
                "messages": existing_data.messages or [],
            },
        )
    )

    # Notify client we are connected.
    await _send(websocket, {"type": "connected", "session_id": active_session_id})

    # Subscriber: forward every agent event to the WebSocket.
    async def _on_event(event: Any, _signal: asyncio.Event) -> None:
        try:
            await _send(websocket, _serialise_event(event))
        except Exception:
            pass  # WebSocket might have closed — agent loop will be aborted below.

    unsubscribe = agent.subscribe(_on_event)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except TimeoutError:
                # Send a ping to keep the connection alive.
                await _send(websocket, {"type": "ping"})
                continue

            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, {"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "prompt":
                content_text: str = str(msg.get("content", ""))
                if not content_text.strip():
                    continue
                user_message = UserMessage(
                    content=[TextContent(text=content_text)],
                    timestamp=int(time.time() * 1000),
                )
                # Run the prompt in the background so we can keep receiving.
                asyncio.create_task(
                    _run_prompt_and_save(
                        agent,
                        user_message,
                        active_session_id,
                        sessions_store,
                        websocket,
                    )
                )

            elif msg_type == "abort":
                agent.abort()

            elif msg_type == "ping":
                await _send(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected for session %s", active_session_id)
    except Exception as exc:
        logger.exception("Unexpected error in chat WebSocket: %s", exc)
        with contextlib.suppress(Exception):
            await _send(websocket, {"type": "error", "message": str(exc)})
    finally:
        unsubscribe()
        agent.abort()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_prompt_and_save(
    agent: Agent,
    user_message: Any,
    session_id: str,
    sessions_store: SessionsStore,
    websocket: WebSocket,
) -> None:
    try:
        await agent.prompt(user_message)
    except Exception as exc:
        logger.warning("Agent prompt error: %s", exc)
        with contextlib.suppress(Exception):
            await _send(websocket, {"type": "error", "message": str(exc)})
    finally:
        # Persist session after each prompt completes.
        try:
            await _persist_session(agent, session_id, sessions_store)
        except Exception as exc:
            logger.warning("Failed to persist session %s: %s", session_id, exc)


async def _persist_session(
    agent: Agent,
    session_id: str,
    sessions_store: SessionsStore,
) -> None:
    now_iso = _now_iso()
    messages = list(agent.state.messages)

    # Try to load existing metadata to preserve createdAt / usage etc.
    existing_meta = await sessions_store.get_metadata(session_id)

    created_at = existing_meta.created_at if existing_meta else now_iso

    # Build a simple preview from the first user message text.
    preview = ""
    for m in messages:
        if getattr(m, "role", None) == "user":
            content = getattr(m, "content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        preview = block.text[:2048]
                        break
            elif isinstance(content, str):
                preview = content[:2048]
            if preview:
                break

    # Determine title.
    title = existing_meta.title if existing_meta and existing_meta.title else preview[:80]

    meta = SessionInfo(
        id=session_id,
        title=title,
        createdAt=created_at,
        lastModified=now_iso,
        messageCount=len(messages),
        usage=existing_meta.usage if existing_meta else _make_empty_usage(),
        thinkingLevel=agent.state.thinking_level,
        preview=preview,
    )

    # Serialize messages to plain dicts for storage.
    serialised_messages: list[Any] = []
    for m in messages:
        if hasattr(m, "model_dump"):
            serialised_messages.append(m.model_dump(by_alias=True))
        elif hasattr(m, "__dict__"):
            serialised_messages.append(vars(m))
        else:
            serialised_messages.append(m)

    data = SessionData(
        id=session_id,
        title=title,
        model=None,  # TODO: expose model from Agent state
        thinkingLevel=agent.state.thinking_level,
        messages=serialised_messages,
        createdAt=created_at,
        lastModified=now_iso,
    )

    await sessions_store.save_session(data, meta)


def _serialise_event(event: Any) -> dict[str, Any]:
    """Best-effort serialisation of an AgentEvent TypedDict to a plain dict."""
    if isinstance(event, dict):
        return _deep_serialise(event)
    return {"type": "unknown", "raw": str(event)}


def _deep_serialise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_serialise(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _deep_serialise(obj.model_dump(by_alias=True))
    if hasattr(obj, "__dict__"):
        return _deep_serialise(vars(obj))
    return obj


async def _send(websocket: WebSocket, payload: Any) -> None:
    await websocket.send_text(json.dumps(payload, default=str))


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


__all__ = ["chat_websocket"]
