"""Default constants — direct port of ``packages/coding-agent/src/core/defaults.ts``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nu_agent_core.types import ThinkingLevel


DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium"


__all__ = ["DEFAULT_THINKING_LEVEL"]
