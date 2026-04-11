"""Tests for ``nu_coding_agent.core.defaults`` and ``nu_coding_agent.core.diagnostics``."""

from __future__ import annotations

from nu_coding_agent.core.defaults import DEFAULT_THINKING_LEVEL
from nu_coding_agent.core.diagnostics import ResourceCollision, ResourceDiagnostic


def test_default_thinking_level_matches_ts() -> None:
    assert DEFAULT_THINKING_LEVEL == "medium"


def test_resource_collision_round_trip() -> None:
    collision = ResourceCollision(
        resource_type="extension",
        name="my-ext",
        winner_path="/a",
        loser_path="/b",
        winner_source="npm:foo",
        loser_source="local",
    )
    assert collision.resource_type == "extension"
    assert collision.winner_path == "/a"
    assert collision.loser_source == "local"


def test_resource_diagnostic_optional_fields() -> None:
    diag = ResourceDiagnostic(type="warning", message="hi")
    assert diag.path is None
    assert diag.collision is None

    full = ResourceDiagnostic(
        type="collision",
        message="dup",
        path="/x",
        collision=ResourceCollision(
            resource_type="skill",
            name="s",
            winner_path="/w",
            loser_path="/l",
        ),
    )
    assert full.collision is not None
    assert full.collision.name == "s"
