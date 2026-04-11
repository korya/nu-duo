"""Source-info value type for resources — direct port of ``packages/coding-agent/src/core/source-info.ts``.

Resources (extensions, skills, prompts, themes) carry a :class:`SourceInfo`
that records where they came from on disk so the loader can render
collisions, manage scopes, and unload packages cleanly.

The upstream :func:`createSourceInfo` consumes a ``PathMetadata`` produced
by ``package-manager.ts`` (~2.2k LoC, not yet ported). To keep this slice
self-contained we declare a structural :class:`PathMetadataLike`
:class:`typing.Protocol` with the four fields the constructor reads. The
real ``PackageManager`` port will satisfy it without needing to import
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

type SourceScope = Literal["user", "project", "temporary"]
type SourceOrigin = Literal["package", "top-level"]


class PathMetadataLike(Protocol):
    """Structural shape of ``package-manager``'s ``PathMetadata``.

    Defined inline so :func:`create_source_info` does not depend on the
    full :mod:`package_manager` port (still pending).
    """

    source: str
    scope: SourceScope
    origin: SourceOrigin
    base_dir: str | None


@dataclass(slots=True)
class SourceInfo:
    """Where a loaded resource came from."""

    path: str
    source: str
    scope: SourceScope
    origin: SourceOrigin
    base_dir: str | None = None


def create_source_info(path: str, metadata: PathMetadataLike) -> SourceInfo:
    """Build a :class:`SourceInfo` from a path-metadata record."""
    return SourceInfo(
        path=path,
        source=metadata.source,
        scope=metadata.scope,
        origin=metadata.origin,
        base_dir=metadata.base_dir,
    )


def create_synthetic_source_info(
    path: str,
    *,
    source: str,
    scope: SourceScope = "temporary",
    origin: SourceOrigin = "top-level",
    base_dir: str | None = None,
) -> SourceInfo:
    """Build a :class:`SourceInfo` for resources that have no on-disk package."""
    return SourceInfo(path=path, source=source, scope=scope, origin=origin, base_dir=base_dir)


__all__ = [
    "PathMetadataLike",
    "SourceInfo",
    "SourceOrigin",
    "SourceScope",
    "create_source_info",
    "create_synthetic_source_info",
]
