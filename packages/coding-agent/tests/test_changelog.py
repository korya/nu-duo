"""Tests for ``nu_coding_agent.utils.changelog``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nu_coding_agent.utils.changelog import (
    ChangelogEntry,
    compare_versions,
    get_new_entries,
    parse_changelog,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_changelog_missing_file(tmp_path: Path) -> None:
    assert parse_changelog(str(tmp_path / "missing.md")) == []


def test_parse_changelog_basic(tmp_path: Path) -> None:
    file = tmp_path / "CHANGELOG.md"
    file.write_text(
        "# Changelog\n\n"
        "## [1.2.3] - 2026-01-01\n"
        "- new feature\n"
        "- bug fix\n\n"
        "## [1.2.2] - 2025-12-31\n"
        "- earlier change\n"
    )
    entries = parse_changelog(str(file))
    assert len(entries) == 2
    assert entries[0].major == 1
    assert entries[0].minor == 2
    assert entries[0].patch == 3
    assert "new feature" in entries[0].content
    assert entries[1].patch == 2
    assert "earlier change" in entries[1].content


def test_parse_changelog_unbracketed_version(tmp_path: Path) -> None:
    file = tmp_path / "CHANGELOG.md"
    file.write_text("## 0.5.0\n\n- bare version line\n")
    entries = parse_changelog(str(file))
    assert len(entries) == 1
    assert entries[0].minor == 5


def test_parse_changelog_skips_non_version_headings(tmp_path: Path) -> None:
    file = tmp_path / "CHANGELOG.md"
    file.write_text("## Unreleased\n- pending\n\n## [1.0.0]\n- shipped\n")
    entries = parse_changelog(str(file))
    assert len(entries) == 1
    assert entries[0].major == 1


def test_compare_versions() -> None:
    a = ChangelogEntry(1, 2, 3, "")
    b = ChangelogEntry(1, 2, 4, "")
    c = ChangelogEntry(1, 3, 0, "")
    d = ChangelogEntry(2, 0, 0, "")
    assert compare_versions(a, a) == 0
    assert compare_versions(a, b) == -1
    assert compare_versions(b, a) == 1
    assert compare_versions(a, c) == -1
    assert compare_versions(c, d) == -1
    assert compare_versions(d, c) == 1


def test_get_new_entries() -> None:
    entries = [
        ChangelogEntry(2, 0, 0, "v2"),
        ChangelogEntry(1, 5, 0, "v1.5"),
        ChangelogEntry(1, 2, 3, "v1.2.3"),
        ChangelogEntry(1, 2, 0, "v1.2.0"),
    ]
    new = get_new_entries(entries, "1.2.3")
    assert {(e.major, e.minor, e.patch) for e in new} == {(2, 0, 0), (1, 5, 0)}


def test_get_new_entries_short_version() -> None:
    # "1.0" → patch defaults to 0.
    entries = [ChangelogEntry(1, 0, 1, "")]
    new = get_new_entries(entries, "1.0")
    assert len(new) == 1


def test_get_new_entries_empty_string() -> None:
    entries = [ChangelogEntry(0, 0, 1, "")]
    assert len(get_new_entries(entries, "")) == 1
