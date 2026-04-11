"""Tests for ``nu_coding_agent.config``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nu_coding_agent import config

if TYPE_CHECKING:
    import pytest


def test_constants() -> None:
    assert config.APP_NAME == "nu"
    assert config.CONFIG_DIR_NAME == ".nu"
    assert config.ENV_AGENT_DIR == "NU_CODING_AGENT_DIR"


def test_get_agent_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NU_CODING_AGENT_DIR", raising=False)
    assert config.get_agent_dir() == str(Path.home() / ".nu" / "agent")


def test_get_agent_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NU_CODING_AGENT_DIR", str(tmp_path))
    assert config.get_agent_dir() == str(tmp_path)


def test_get_agent_dir_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_CODING_AGENT_DIR", "~/custom-agent")
    assert config.get_agent_dir() == str(Path.home() / "custom-agent")


def test_get_agent_dir_just_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_CODING_AGENT_DIR", "~")
    assert config.get_agent_dir() == str(Path.home())


def test_subdir_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NU_CODING_AGENT_DIR", str(tmp_path))
    assert config.get_prompts_dir() == str(tmp_path / "prompts")
    assert config.get_settings_path() == str(tmp_path / "settings.json")
    assert config.get_auth_path() == str(tmp_path / "auth.json")
    assert config.get_models_path() == str(tmp_path / "models.json")
    assert config.get_tools_dir() == str(tmp_path / "tools")
    assert config.get_bin_dir() == str(tmp_path / "bin")
    assert config.get_sessions_dir() == str(tmp_path / "sessions")
    assert config.get_custom_themes_dir() == str(tmp_path / "themes")
    assert config.get_debug_log_path() == str(tmp_path / "nu-debug.log")


def test_get_share_viewer_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NU_SHARE_VIEWER_URL", raising=False)
    assert config.get_share_viewer_url("abc") == "https://pi.dev/session/#abc"


def test_get_share_viewer_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_SHARE_VIEWER_URL", "https://example.com/s/")
    assert config.get_share_viewer_url("xyz") == "https://example.com/s/#xyz"


def test_get_package_dir_returns_existing_path() -> None:
    pkg_dir = Path(config.get_package_dir())
    assert pkg_dir.exists()
    assert (pkg_dir / "__init__.py").exists() or (pkg_dir / "core").exists()


def test_get_package_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NU_PACKAGE_DIR", str(tmp_path))
    assert config.get_package_dir() == str(tmp_path)
