"""Resource diagnostic types — direct port of ``packages/coding-agent/src/core/diagnostics.ts``.

Used by extension/skill/prompt/theme loaders to surface collisions and load
errors back up to the CLI without forcing them to print directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type ResourceType = Literal["extension", "skill", "prompt", "theme"]
type DiagnosticType = Literal["warning", "error", "collision"]


@dataclass(slots=True)
class ResourceCollision:
    """Two resources of the same kind that share a name; the loader picks a winner."""

    resource_type: ResourceType
    name: str
    winner_path: str
    loser_path: str
    winner_source: str | None = None
    loser_source: str | None = None


@dataclass(slots=True)
class ResourceDiagnostic:
    """A warning, error, or collision raised during resource loading."""

    type: DiagnosticType
    message: str
    path: str | None = None
    collision: ResourceCollision | None = None


__all__ = [
    "DiagnosticType",
    "ResourceCollision",
    "ResourceDiagnostic",
    "ResourceType",
]
