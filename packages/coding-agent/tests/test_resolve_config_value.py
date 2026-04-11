"""Tests for ``nu_coding_agent.core.resolve_config_value``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from nu_coding_agent.core.resolve_config_value import (
    clear_config_value_cache,
    resolve_config_value,
    resolve_config_value_or_throw,
    resolve_config_value_uncached,
    resolve_headers,
    resolve_headers_or_throw,
)


@pytest.fixture(autouse=True)
def _clear_cache():  # pyright: ignore[reportUnusedFunction]
    clear_config_value_cache()
    yield
    clear_config_value_cache()


def test_literal_string_returned_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NU_TEST_LITERAL", raising=False)
    assert resolve_config_value("NU_TEST_LITERAL") == "NU_TEST_LITERAL"


def test_env_var_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_TEST_KEY", "secret")
    assert resolve_config_value("NU_TEST_KEY") == "secret"


def test_shell_command_executed() -> None:
    assert resolve_config_value("!echo hello") == "hello"


def test_shell_command_failure_returns_none() -> None:
    assert resolve_config_value("!exit 1") is None


def test_shell_command_cached(tmp_path: Path) -> None:
    counter = tmp_path / "counter.txt"
    cmd = f'!echo hi && echo x >> "{counter}"'
    resolve_config_value(cmd)
    resolve_config_value(cmd)
    resolve_config_value(cmd)
    assert counter.exists()
    assert counter.read_text().count("x") == 1


def test_uncached_runs_every_time(tmp_path: Path) -> None:
    counter = tmp_path / "counter.txt"
    cmd = f'!echo hi && echo x >> "{counter}"'
    resolve_config_value_uncached(cmd)
    resolve_config_value_uncached(cmd)
    assert counter.read_text().count("x") == 2


def test_or_throw_raises_on_failure() -> None:
    with pytest.raises(ValueError, match="shell command"):
        resolve_config_value_or_throw("!exit 1", "test value")


def test_resolve_headers_skips_none() -> None:
    assert resolve_headers(None) is None
    assert resolve_headers({}) is None


def test_resolve_headers_resolves_each_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_HEADER_TOK", "abc")
    headers = {"Authorization": "NU_HEADER_TOK", "X-Other": "literal"}
    out = resolve_headers(headers)
    assert out == {"Authorization": "abc", "X-Other": "literal"}


def test_resolve_headers_or_throw_propagates_failure() -> None:
    with pytest.raises(ValueError, match='header "Authorization"'):
        resolve_headers_or_throw({"Authorization": "!exit 1"}, "anthropic")
