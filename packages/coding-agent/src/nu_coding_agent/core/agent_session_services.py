"""Agent session services — port of ``packages/coding-agent/src/core/agent-session-services.ts``.

Provides the :class:`AgentSessionServices` container and factory
functions for building cwd-bound services. The services container
holds everything a session needs — auth, model registry, settings,
resource loader — and is recreated when the cwd changes (e.g. on
session switch or fork).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nu_coding_agent.config import get_agent_dir

if TYPE_CHECKING:
    from nu_coding_agent.core.auth_storage import AuthStorage
    from nu_coding_agent.core.model_registry import ModelRegistry
    from nu_coding_agent.core.settings_manager import SettingsManager


@dataclass(slots=True)
class AgentSessionRuntimeDiagnostic:
    """Non-fatal issue collected during service creation.

    The app layer decides whether to show warnings or abort.
    """

    type: str  # "info" | "warning" | "error"
    message: str


@dataclass(slots=True)
class AgentSessionServices:
    """Cwd-bound runtime services for one effective session.

    Recreated whenever the session cwd changes (e.g. on switch/fork).
    """

    cwd: str
    agent_dir: str
    auth_storage: AuthStorage
    settings_manager: SettingsManager
    model_registry: ModelRegistry
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)


async def create_agent_session_services(
    *,
    cwd: str,
    agent_dir: str | None = None,
    auth_storage: AuthStorage | None = None,
    settings_manager: SettingsManager | None = None,
    model_registry: ModelRegistry | None = None,
) -> AgentSessionServices:
    """Create cwd-bound runtime services.

    Builds or reuses AuthStorage, SettingsManager, and ModelRegistry.
    Returns a services container plus any diagnostics collected.
    """
    from nu_coding_agent.core.auth_storage import AuthStorage as _AuthStorage  # noqa: PLC0415
    from nu_coding_agent.core.model_registry import ModelRegistry as _ModelRegistry  # noqa: PLC0415
    from nu_coding_agent.core.settings_manager import SettingsManager as _SettingsManager  # noqa: PLC0415

    resolved_agent_dir = agent_dir or get_agent_dir()
    resolved_auth = auth_storage or _AuthStorage.create()
    resolved_settings = settings_manager or _SettingsManager.create(cwd, resolved_agent_dir)
    resolved_registry = model_registry or _ModelRegistry.create(resolved_auth)

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []

    return AgentSessionServices(
        cwd=cwd,
        agent_dir=resolved_agent_dir,
        auth_storage=resolved_auth,
        settings_manager=resolved_settings,
        model_registry=resolved_registry,
        diagnostics=diagnostics,
    )


__all__ = [
    "AgentSessionRuntimeDiagnostic",
    "AgentSessionServices",
    "create_agent_session_services",
]
