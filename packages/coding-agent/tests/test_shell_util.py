"""Tests for ``nu_coding_agent.utils.shell``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from nu_coding_agent.utils.shell import (
    get_shell_config,
    get_shell_env,
    reset_shell_config_cache,
    sanitize_binary_output,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_cache():  # pyright: ignore[reportUnusedFunction]
    reset_shell_config_cache()
    yield
    reset_shell_config_cache()


def test_sanitize_keeps_tab_lf_cr() -> None:
    assert sanitize_binary_output("a\tb\nc\rd") == "a\tb\nc\rd"


def test_sanitize_drops_other_control_chars() -> None:
    assert sanitize_binary_output("a\x01b\x1fc") == "abc"


def test_sanitize_drops_format_range() -> None:
    assert sanitize_binary_output("a\ufff9b\ufffac\ufffbd") == "abcd"


def test_sanitize_keeps_normal_text() -> None:
    assert sanitize_binary_output("hello world") == "hello world"


def test_sanitize_keeps_unicode() -> None:
    assert sanitize_binary_output("ελληνικά 日本語 🚀") == "ελληνικά 日本語 🚀"


def test_get_shell_config_uses_settings_loader_when_path_exists(tmp_path: Path) -> None:
    fake_shell = tmp_path / "fake-bash"
    fake_shell.write_text("#!/bin/sh\n")
    cfg = get_shell_config(settings_loader=lambda: str(fake_shell))
    assert cfg.shell == str(fake_shell)
    assert cfg.args == ["-c"]


def test_get_shell_config_raises_when_settings_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-shell"
    with pytest.raises(ValueError, match="Custom shell path not found"):
        get_shell_config(settings_loader=lambda: str(missing))


def test_get_shell_config_caches_result(tmp_path: Path) -> None:
    fake_shell = tmp_path / "x-shell"
    fake_shell.write_text("#!/bin/sh\n")
    calls = 0

    def loader() -> str | None:
        nonlocal calls
        calls += 1
        return str(fake_shell)

    a = get_shell_config(settings_loader=loader)
    b = get_shell_config(settings_loader=loader)
    assert a is b
    assert calls == 1  # cached after first call


def test_get_shell_config_default_resolves(tmp_path: Path) -> None:
    cfg = get_shell_config(settings_loader=lambda: None)
    assert cfg.shell  # something resolved
    assert cfg.args == ["-c"]


def test_get_shell_env_includes_bin_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NU_CODING_AGENT_DIR", str(tmp_path))
    env = get_shell_env()
    path_key = next(k for k in env if k.lower() == "path")
    assert str(tmp_path / "bin") in env[path_key].split(":")


def test_get_shell_env_does_not_duplicate_bin_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NU_CODING_AGENT_DIR", str(tmp_path))
    bin_dir = str(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin")
    env = get_shell_env()
    path_key = next(k for k in env if k.lower() == "path")
    parts = env[path_key].split(":")
    assert parts.count(bin_dir) == 1
