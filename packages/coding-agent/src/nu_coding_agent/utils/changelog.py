"""CHANGELOG.md parser — direct port of ``packages/coding-agent/src/utils/changelog.ts``.

Walks ``## [x.y.z] ...`` headings and accumulates the body of each
release. Used by the interactive ``/changelog`` slash command and by the
"new since last run" notice.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_VERSION_HEADING_RE = re.compile(r"##\s+\[?(\d+)\.(\d+)\.(\d+)\]?")


@dataclass(slots=True)
class ChangelogEntry:
    major: int
    minor: int
    patch: int
    content: str


def parse_changelog(changelog_path: str) -> list[ChangelogEntry]:
    """Read ``changelog_path`` and return one :class:`ChangelogEntry` per release."""
    path = Path(changelog_path)
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Warning: Could not parse changelog: {exc}", file=sys.stderr)
        return []

    entries: list[ChangelogEntry] = []
    current_lines: list[str] = []
    current_version: tuple[int, int, int] | None = None

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_version is not None and current_lines:
                major, minor, patch = current_version
                entries.append(
                    ChangelogEntry(
                        major=major,
                        minor=minor,
                        patch=patch,
                        content="\n".join(current_lines).strip(),
                    )
                )
            match = _VERSION_HEADING_RE.match(line)
            if match is not None:
                current_version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                current_lines = [line]
            else:
                current_version = None
                current_lines = []
        elif current_version is not None:
            current_lines.append(line)

    if current_version is not None and current_lines:
        major, minor, patch = current_version
        entries.append(
            ChangelogEntry(
                major=major,
                minor=minor,
                patch=patch,
                content="\n".join(current_lines).strip(),
            )
        )

    return entries


def compare_versions(v1: ChangelogEntry, v2: ChangelogEntry) -> int:
    """Return ``-1``/``0``/``1`` for ``v1`` ``<``/``==``/``>`` ``v2``."""
    if v1.major != v2.major:
        return -1 if v1.major < v2.major else 1
    if v1.minor != v2.minor:
        return -1 if v1.minor < v2.minor else 1
    if v1.patch != v2.patch:
        return -1 if v1.patch < v2.patch else 1
    return 0


def get_new_entries(entries: list[ChangelogEntry], last_version: str) -> list[ChangelogEntry]:
    """Return entries strictly newer than ``last_version`` (e.g. ``"1.2.3"``)."""
    parts = [int(p) if p.isdigit() else 0 for p in last_version.split(".")]
    while len(parts) < 3:
        parts.append(0)
    last = ChangelogEntry(major=parts[0], minor=parts[1], patch=parts[2], content="")
    return [entry for entry in entries if compare_versions(entry, last) > 0]


__all__ = [
    "ChangelogEntry",
    "compare_versions",
    "get_new_entries",
    "parse_changelog",
]
