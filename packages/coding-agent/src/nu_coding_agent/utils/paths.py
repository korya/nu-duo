"""Path-classification helper — direct port of ``packages/coding-agent/src/utils/paths.ts``.

Used by the resource loader to decide whether a string in a settings file
points at a local directory (a "bare name" or relative path) or at a
non-local source (npm package, git url, github shortcut, http url, ssh).
"""

from __future__ import annotations

_NON_LOCAL_PREFIXES = ("npm:", "git:", "github:", "http:", "https:", "ssh:")


def is_local_path(value: str) -> bool:
    """Return ``True`` for bare/relative paths and ``False`` for npm/git/url sources."""
    trimmed = value.strip()
    return not trimmed.startswith(_NON_LOCAL_PREFIXES)


__all__ = ["is_local_path"]
