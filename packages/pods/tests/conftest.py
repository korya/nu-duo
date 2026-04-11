"""Shared pytest fixtures for ``nu_pods`` tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from nu_pods import ssh as ssh_mod
from nu_pods.commands import prompt as prompt_mod


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``NU_PODS_CONFIG_DIR`` at a per-test temp directory."""
    monkeypatch.setenv("NU_PODS_CONFIG_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture
def reset_ssh_runners() -> Iterator[None]:
    """Restore SSH runner stubs after each test that overrides them."""
    yield
    ssh_mod.reset_runners()


@pytest.fixture
def reset_prompt_launcher() -> Iterator[None]:
    yield
    prompt_mod.reset_launcher()


@pytest.fixture
def clean_pod_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ``HF_TOKEN`` / ``NU_API_KEY`` / ``PI_API_KEY`` for negative tests."""
    for key in ("HF_TOKEN", "NU_API_KEY", "PI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
