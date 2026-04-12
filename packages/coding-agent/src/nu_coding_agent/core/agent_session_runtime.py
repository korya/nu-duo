"""Agent session runtime — port of ``packages/coding-agent/src/core/agent-session-runtime.ts``.

Owns the current :class:`AgentSession` plus its cwd-bound services.
Provides session lifecycle operations: switch, new, fork, import,
and dispose. Extension hooks (``session_before_switch``,
``session_before_fork``) are emitted before destructive operations
so extensions can cancel them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nu_coding_agent.core.session_cwd import assert_session_cwd_exists
from nu_coding_agent.core.session_manager import SessionManager

if TYPE_CHECKING:
    from nu_coding_agent.core.agent_session import AgentSession
    from nu_coding_agent.core.agent_session_services import (
        AgentSessionRuntimeDiagnostic,
        AgentSessionServices,
    )


@staticmethod
def _extract_user_message_text(content: Any) -> str:
    """Extract plain text from a user message's content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            for part in content
            if (isinstance(part, dict) and part.get("type") == "text") or getattr(part, "type", None) == "text"
        )
    return ""


class AgentSessionRuntime:
    """Owns the current AgentSession + its cwd-bound services.

    Session replacement methods tear down the current runtime, then
    create and apply the next one. If creation fails, the error
    propagates to the caller.
    """

    def __init__(
        self,
        session: AgentSession,
        services: AgentSessionServices,
        *,
        diagnostics: list[AgentSessionRuntimeDiagnostic] | None = None,
        model_fallback_message: str | None = None,
    ) -> None:
        self._session = session
        self._services = services
        self._diagnostics = list(diagnostics or [])
        self._model_fallback_message = model_fallback_message

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def services(self) -> AgentSessionServices:
        return self._services

    @property
    def cwd(self) -> str:
        return self._services.cwd

    @property
    def diagnostics(self) -> list[AgentSessionRuntimeDiagnostic]:
        return list(self._diagnostics)

    @property
    def model_fallback_message(self) -> str | None:
        return self._model_fallback_message

    # ------------------------------------------------------------------
    # Extension hooks
    # ------------------------------------------------------------------

    async def _emit_before_switch(self, reason: str, target_session_file: str | None = None) -> bool:
        """Emit ``session_before_switch``. Returns ``True`` if cancelled."""
        runner = self._session.extension_runner
        if runner is None or not runner.has_handlers("session_before_switch"):
            return False
        results = await runner.emit_with_results(
            {"type": "session_before_switch", "reason": reason, "targetSessionFile": target_session_file}
        )
        return any(getattr(r, "cancel", False) or (isinstance(r, dict) and r.get("cancel")) for r in results)

    async def _emit_before_fork(self, entry_id: str) -> bool:
        """Emit ``session_before_fork``. Returns ``True`` if cancelled."""
        runner = self._session.extension_runner
        if runner is None or not runner.has_handlers("session_before_fork"):
            return False
        results = await runner.emit_with_results({"type": "session_before_fork", "entryId": entry_id})
        return any(getattr(r, "cancel", False) or (isinstance(r, dict) and r.get("cancel")) for r in results)

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    async def _teardown_current(self) -> None:
        """Shut down the current session (extension hooks + close)."""
        await self._session.shutdown()

    async def switch_session(self, session_path: str, cwd_override: str | None = None) -> bool:
        """Switch to an existing session file.

        Returns ``True`` if cancelled by an extension, ``False`` on success.
        """
        if await self._emit_before_switch("resume", session_path):
            return True

        sm = SessionManager.open(session_path, cwd_override=cwd_override)
        assert_session_cwd_exists(sm, self.cwd)
        await self._teardown_current()
        # Rebuild — the caller should re-create the AgentSession from the new SM.
        # For now, we update the session manager reference.
        self._session.set_session_manager(sm)
        return False

    async def new_session(self, *, parent_session: str | None = None) -> bool:
        """Create a fresh session.

        Returns ``True`` if cancelled by an extension, ``False`` on success.
        """
        if await self._emit_before_switch("new"):
            return True

        session_dir = self._session.session_manager.get_session_dir()
        sm = SessionManager.create(self.cwd, session_dir)
        if parent_session:
            sm.new_session({"parentSession": parent_session})

        await self._teardown_current()
        self._session.set_session_manager(sm)
        return False

    async def fork(self, entry_id: str) -> tuple[bool, str | None]:
        """Fork the session at the given entry.

        Returns ``(cancelled, selected_text)``.
        """
        if await self._emit_before_fork(entry_id):
            return True, None

        sm = self._session.session_manager
        entry = sm.get_entry(entry_id)
        if entry is None:
            raise ValueError(f"Entry {entry_id} not found")

        message = entry.get("message", {})
        selected_text = _extract_user_message_text(message.get("content", ""))

        parent_id = entry.get("parentId")
        if parent_id is None:
            # Fork from root — create a new session
            session_dir = sm.get_session_dir()
            new_sm = SessionManager.create(self.cwd, session_dir)
            new_sm.new_session({"parentSession": sm.get_session_file()})
            await self._teardown_current()
            self._session.set_session_manager(new_sm)
        else:
            sm.create_branched_session(parent_id)
            # The session manager's state is updated in-place.

        return False, selected_text

    async def dispose(self) -> None:
        """Shut down the runtime, emitting session_shutdown."""
        await self._session.shutdown()


def create_agent_session_runtime(
    session: AgentSession,
    services: AgentSessionServices,
    *,
    diagnostics: list[AgentSessionRuntimeDiagnostic] | None = None,
    model_fallback_message: str | None = None,
) -> AgentSessionRuntime:
    """Build an :class:`AgentSessionRuntime` from a session + services."""
    return AgentSessionRuntime(
        session,
        services,
        diagnostics=diagnostics,
        model_fallback_message=model_fallback_message,
    )


__all__ = [
    "AgentSessionRuntime",
    "create_agent_session_runtime",
]
