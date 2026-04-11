"""Tests for ``nu_coding_agent.core.source_info`` and ``nu_coding_agent.core.slash_commands``."""

from __future__ import annotations

from dataclasses import dataclass

from nu_coding_agent.core.slash_commands import BUILTIN_SLASH_COMMANDS, SlashCommandInfo
from nu_coding_agent.core.source_info import (
    SourceInfo,
    create_source_info,
    create_synthetic_source_info,
)


@dataclass
class _FakePathMetadata:
    source: str
    scope: str
    origin: str
    base_dir: str | None


def test_create_source_info_copies_fields() -> None:
    meta = _FakePathMetadata(source="npm:foo", scope="user", origin="package", base_dir="/pkg")
    info = create_source_info("/pkg/x.json", meta)  # type: ignore[arg-type]
    assert info == SourceInfo(
        path="/pkg/x.json",
        source="npm:foo",
        scope="user",
        origin="package",
        base_dir="/pkg",
    )


def test_create_synthetic_source_info_defaults() -> None:
    info = create_synthetic_source_info("/tmp/x", source="cli-flag")
    assert info.scope == "temporary"
    assert info.origin == "top-level"
    assert info.base_dir is None
    assert info.source == "cli-flag"


def test_create_synthetic_source_info_overrides() -> None:
    info = create_synthetic_source_info(
        "/p",
        source="git:foo",
        scope="project",
        origin="package",
        base_dir="/p/.nu",
    )
    assert info.scope == "project"
    assert info.origin == "package"
    assert info.base_dir == "/p/.nu"


def test_builtin_slash_commands_present() -> None:
    names = {c.name for c in BUILTIN_SLASH_COMMANDS}
    assert "quit" in names
    assert "model" in names
    assert "compact" in names
    assert len(BUILTIN_SLASH_COMMANDS) == 20


def test_slash_command_info_holds_source_info() -> None:
    src = create_synthetic_source_info("/tmp", source="ext:foo")
    info = SlashCommandInfo(name="hello", source="extension", source_info=src, description="hi")
    assert info.name == "hello"
    assert info.source == "extension"
    assert info.source_info.source == "ext:foo"
