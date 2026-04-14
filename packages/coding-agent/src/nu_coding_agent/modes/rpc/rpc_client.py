"""Programmatic RPC client for the coding agent.

Port of ``modes/rpc/rpc-client.ts``.

Spawns a coding-agent subprocess in RPC mode and communicates via JSON lines
over stdin/stdout.  Provides an async API for all RPC commands.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Private type aliases used throughout the module.
_EventCallback = Callable[[dict[str, Any]], None]
_Unsubscribe = Callable[[], None]

__all__ = [
    "ModelInfo",
    "RpcClient",
    "RpcClientError",
    "RpcClientOptions",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RpcClientError(Exception):
    """Raised when the RPC subprocess returns an error or times out."""


# ---------------------------------------------------------------------------
# Options / helper types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RpcClientOptions:
    """Options for :class:`RpcClient`."""

    cli_path: str | None = None
    """Path to the ``nu`` CLI binary. Defaults to ``shutil.which("nu")``."""

    cwd: str | None = None
    """Working directory for the subprocess."""

    env: dict[str, str] | None = None
    """Extra environment variables passed to the subprocess."""

    provider: str | None = None
    """LLM provider to use (forwarded as ``--provider``)."""

    model: str | None = None
    """Model id to use (forwarded as ``--model``)."""

    args: list[str] | None = None
    """Extra CLI arguments appended after ``--rpc``."""


@dataclass(slots=True)
class ModelInfo:
    """Minimal model descriptor returned by the RPC server."""

    id: str
    provider: str
    api: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RpcClient:
    """Programmatic client for the coding agent RPC mode.

    Spawns a subprocess running ``nu --rpc`` and communicates via JSON lines.

    Usage::

        client = RpcClient(RpcClientOptions(model="claude-sonnet-4-20250514"))
        await client.start()
        events = await client.prompt_and_wait("Hello!")
        await client.stop()
    """

    def __init__(self, options: RpcClientOptions | None = None) -> None:
        self._options = options or RpcClientOptions()
        self._process: asyncio.subprocess.Process | None = None
        self._next_id: int = 0
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._event_listeners: list[_EventCallback] = []
        self._reader_task: asyncio.Task[None] | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Spawn the subprocess and begin reading stdout."""
        cli = self._options.cli_path or shutil.which("nu")
        if cli is None:
            raise RpcClientError(
                "Cannot find 'nu' CLI.  Pass cli_path in RpcClientOptions or ensure 'nu' is on the PATH."
            )

        argv: list[str] = [cli, "--rpc"]
        if self._options.provider:
            argv += ["--provider", self._options.provider]
        if self._options.model:
            argv += ["--model", self._options.model]
        if self._options.args:
            argv += self._options.args

        self._process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._options.cwd,
            env=self._options.env,
        )

        self._reader_task = asyncio.create_task(self._read_loop(), name="rpc-client-reader")

    async def stop(self) -> None:
        """Terminate the subprocess and clean up."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

        proc = self._process
        if proc is not None:
            if proc.stdin is not None:
                proc.stdin.close()
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (TimeoutError, ProcessLookupError):
                proc.kill()
            self._process = None

        # Reject any pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RpcClientError("Client stopped"))
        self._pending.clear()

    # -- internal transport --------------------------------------------------

    def _alloc_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    async def _send(self, payload: dict[str, Any]) -> None:
        """Write a JSON line to the subprocess stdin."""
        proc = self._process
        if proc is None or proc.stdin is None:
            raise RpcClientError("Subprocess not running")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        proc.stdin.write(line.encode())
        await proc.stdin.drain()

    async def _request(self, cmd_type: str, **kwargs: Any) -> dict[str, Any]:
        """Send a command and wait for the matching response."""
        req_id = self._alloc_id()
        payload: dict[str, Any] = {"id": req_id, "type": cmd_type, **kwargs}

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = future

        await self._send(payload)

        try:
            response = await future
        except asyncio.CancelledError:
            self._pending.pop(req_id, None)
            raise

        if not response.get("success", False):
            raise RpcClientError(response.get("error", "Unknown RPC error"))

        return response

    async def _read_loop(self) -> None:
        """Background task: read JSON lines from stdout and route them."""
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        buffer = ""
        while True:
            try:
                chunk = await proc.stdout.read(8192)
            except (asyncio.CancelledError, ConnectionResetError):
                break
            if not chunk:
                break

            buffer += chunk.decode("utf-8", errors="replace")

            while True:
                idx = buffer.find("\n")
                if idx == -1:
                    break
                raw_line = buffer[:idx].rstrip("\r")
                buffer = buffer[idx + 1 :]
                if not raw_line:
                    continue
                self._dispatch_line(raw_line)

    def _dispatch_line(self, line: str) -> None:
        """Parse a JSON line and route to response future or event listeners."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("RPC: failed to parse line: %s", line[:200])
            return

        if not isinstance(msg, dict):
            return

        # Responses carry an id that matches a pending request.
        msg_id = msg.get("id")
        if msg.get("type") == "response" and msg_id is not None:
            future = self._pending.pop(str(msg_id), None)
            if future is not None and not future.done():
                future.set_result(msg)
            return

        # Everything else is an event — broadcast to listeners.
        for cb in self._event_listeners:
            try:
                cb(msg)
            except Exception:
                logger.exception("RPC: event listener error")

    # -- events --------------------------------------------------------------

    def on_event(self, callback: _EventCallback) -> _Unsubscribe:
        """Subscribe to events.  Returns an unsubscribe function."""
        self._event_listeners.append(callback)

        def _unsub() -> None:
            with contextlib.suppress(ValueError):
                self._event_listeners.remove(callback)

        return _unsub

    # -- prompting -----------------------------------------------------------

    async def prompt(self, text: str, *, images: list[dict[str, Any]] | None = None) -> None:
        """Send a user prompt (non-blocking on server side; events stream back)."""
        kwargs: dict[str, Any] = {"message": text}
        if images is not None:
            kwargs["images"] = images
        await self._request("prompt", **kwargs)

    async def steer(self, text: str) -> None:
        """Inject a steering message while the agent is streaming."""
        await self._request("steer", message=text)

    async def follow_up(self, text: str) -> None:
        """Queue a follow-up message."""
        await self._request("follow_up", message=text)

    async def abort(self) -> None:
        """Abort the current agent turn."""
        await self._request("abort")

    # -- session management --------------------------------------------------

    async def new_session(self, *, parent_session: str | None = None) -> dict[str, Any]:
        """Create a new session, optionally forking from *parent_session*."""
        kwargs: dict[str, Any] = {}
        if parent_session is not None:
            kwargs["parentSession"] = parent_session
        resp = await self._request("new_session", **kwargs)
        return resp.get("data", {})

    async def switch_session(self, session_path: str) -> dict[str, Any]:
        """Switch to an existing session file."""
        resp = await self._request("switch_session", sessionPath=session_path)
        return resp.get("data", {})

    async def fork(self, entry_id: str, text: str) -> dict[str, Any]:
        """Fork the conversation at *entry_id*."""
        resp = await self._request("fork", entryId=entry_id, message=text)
        return resp.get("data", {})

    async def get_state(self) -> dict[str, Any]:
        """Return the current session state."""
        resp = await self._request("get_state")
        return resp.get("data", {})

    # -- model control -------------------------------------------------------

    async def set_model(self, model_id: str, *, provider: str = "") -> dict[str, Any]:
        """Set the active model."""
        resp = await self._request("set_model", modelId=model_id, provider=provider)
        return resp.get("data", {})

    async def cycle_model(self, direction: int = 1) -> dict[str, Any] | None:
        """Cycle through available models."""
        resp = await self._request("cycle_model", direction=direction)
        return resp.get("data")

    async def get_available_models(self) -> list[dict[str, Any]]:
        """Return the list of available models."""
        resp = await self._request("get_available_models")
        return resp.get("data", {}).get("models", [])

    async def set_thinking_level(self, level: str) -> None:
        """Set the thinking budget level (e.g. ``"off"``, ``"low"``, ``"high"``)."""
        await self._request("set_thinking_level", level=level)

    async def cycle_thinking_level(self, direction: int = 1) -> str | None:
        """Cycle through thinking levels.  Returns the new level."""
        resp = await self._request("cycle_thinking_level", direction=direction)
        data = resp.get("data")
        return data.get("level") if isinstance(data, dict) else None

    # -- compaction ----------------------------------------------------------

    async def compact(self, *, custom_instructions: str | None = None) -> dict[str, Any] | None:
        """Trigger message compaction."""
        kwargs: dict[str, Any] = {}
        if custom_instructions is not None:
            kwargs["customInstructions"] = custom_instructions
        resp = await self._request("compact", **kwargs)
        return resp.get("data")

    async def set_auto_compaction(self, enabled: bool) -> None:
        """Enable or disable auto-compaction."""
        await self._request("set_auto_compaction", enabled=enabled)

    # -- bash ----------------------------------------------------------------

    async def bash(self, command: str) -> dict[str, Any] | None:
        """Execute a bash command inside the agent session."""
        resp = await self._request("bash", command=command)
        return resp.get("data")

    async def abort_bash(self) -> None:
        """Abort a running bash command."""
        await self._request("abort_bash")

    # -- introspection -------------------------------------------------------

    async def get_session_stats(self) -> dict[str, Any]:
        """Return session statistics."""
        resp = await self._request("get_session_stats")
        return resp.get("data", {})

    async def export_html(self, *, output_path: str | None = None) -> str:
        """Export the session as HTML.  Returns the file path."""
        kwargs: dict[str, Any] = {}
        if output_path is not None:
            kwargs["outputPath"] = output_path
        resp = await self._request("export_html", **kwargs)
        return resp.get("data", {}).get("path", "")

    async def get_messages(self) -> list[dict[str, Any]]:
        """Return all conversation messages."""
        resp = await self._request("get_messages")
        return resp.get("data", {}).get("messages", [])

    async def get_fork_messages(self) -> list[dict[str, Any]]:
        """Return messages available for forking."""
        resp = await self._request("get_fork_messages")
        return resp.get("data", {}).get("messages", [])

    async def get_last_assistant_text(self) -> str | None:
        """Return the last assistant text block."""
        resp = await self._request("get_last_assistant_text")
        return resp.get("data", {}).get("text")

    async def get_commands(self) -> list[dict[str, Any]]:
        """Return available slash commands."""
        resp = await self._request("get_commands")
        return resp.get("data", {}).get("commands", [])

    # -- queue modes ---------------------------------------------------------

    async def set_steering_mode(self, mode: str) -> None:
        await self._request("set_steering_mode", mode=mode)

    async def set_follow_up_mode(self, mode: str) -> None:
        await self._request("set_follow_up_mode", mode=mode)

    # -- retry ---------------------------------------------------------------

    async def set_auto_retry(self, enabled: bool) -> None:
        await self._request("set_auto_retry", enabled=enabled)

    async def abort_retry(self) -> None:
        await self._request("abort_retry")

    # -- session naming ------------------------------------------------------

    async def set_session_name(self, name: str) -> None:
        await self._request("set_session_name", name=name)

    # -- convenience ---------------------------------------------------------

    async def wait_for_idle(self, *, max_wait: float = 30.0) -> None:
        """Wait until the agent is idle (not streaming).

        Polls ``get_state`` and checks ``isStreaming``.

        Raises:
            asyncio.TimeoutError: If the agent does not become idle in *timeout* seconds.
        """

        async def _poll() -> None:
            while True:
                state = await self.get_state()
                if not state.get("isStreaming", False):
                    return
                await asyncio.sleep(0.25)

        await asyncio.wait_for(_poll(), timeout=max_wait)

    async def prompt_and_wait(
        self,
        text: str,
        *,
        images: list[dict[str, Any]] | None = None,
        max_wait: float = 60.0,
    ) -> list[dict[str, Any]]:
        """Send a prompt and collect all events until the agent is idle.

        Returns the list of events received.
        """
        collected: list[dict[str, Any]] = []
        idle_event = asyncio.Event()

        def _on_event(event: dict[str, Any]) -> None:
            collected.append(event)
            # The server emits an event with isStreaming=False when done.
            # We also look for type="agent_status" with streaming=false,
            # or simply poll via get_state after prompt returns.
            ev_type = event.get("type", "")
            if ev_type in ("agent_turn_end", "turn_end"):
                idle_event.set()

        unsub = self.on_event(_on_event)
        try:
            await self.prompt(text, images=images)

            # Wait for the turn-end event, or fall back to polling.
            try:
                await asyncio.wait_for(idle_event.wait(), timeout=max_wait)
            except TimeoutError:
                # Final check via state — maybe we missed the event.
                state = await self.get_state()
                if state.get("isStreaming", False):
                    raise TimeoutError(f"Agent did not finish within {max_wait}s") from None
        finally:
            unsub()

        return collected

    # -- context manager -----------------------------------------------------

    async def __aenter__(self) -> RpcClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
