"""Tests for ``nu_pods.config`` persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nu_pods.config import (
    add_pod,
    get_active_pod,
    get_config_dir,
    get_config_path,
    load_config,
    remove_pod,
    save_config,
    set_active_pod,
)
from nu_pods.types import GPU, Config, Pod


def test_config_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NU_PODS_CONFIG_DIR", raising=False)
    assert get_config_dir() == Path.home() / ".nu"
    assert get_config_path() == Path.home() / ".nu" / "pods.json"


def test_config_dir_override(isolated_config: Path) -> None:
    assert get_config_dir() == isolated_config


def test_load_config_missing_returns_empty(isolated_config: Path) -> None:
    config = load_config()
    assert config.pods == {}
    assert config.active is None


def test_load_config_corrupt_returns_empty(isolated_config: Path) -> None:
    (isolated_config / "pods.json").write_text("not json")
    config = load_config()
    assert config.pods == {}
    assert config.active is None


def test_save_and_load_round_trip(isolated_config: Path) -> None:
    pod = Pod(
        ssh="ssh root@h1",
        gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")],
        models_path="/mnt/sfs",
        vllm_version="release",
    )
    save_config(Config(pods={"first": pod}, active="first"))

    raw = json.loads((isolated_config / "pods.json").read_text())
    assert raw["active"] == "first"
    assert raw["pods"]["first"]["modelsPath"] == "/mnt/sfs"

    loaded = load_config()
    assert loaded.active == "first"
    assert loaded.pods["first"].vllm_version == "release"


def test_add_pod_makes_first_pod_active(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    assert load_config().active == "a"

    # Adding another should NOT steal active.
    add_pod("b", Pod(ssh="ssh b"))
    assert load_config().active == "a"


def test_add_pod_preserves_existing_active(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    set_active_pod("a")
    add_pod("b", Pod(ssh="ssh b"))
    assert load_config().active == "a"


def test_remove_pod_clears_active(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    remove_pod("a")
    config = load_config()
    assert "a" not in config.pods
    assert config.active is None


def test_remove_unknown_pod_is_noop(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    remove_pod("ghost")
    assert "a" in load_config().pods


def test_get_active_pod_none_when_unset(isolated_config: Path) -> None:
    assert get_active_pod() is None


def test_get_active_pod_none_when_dangling(isolated_config: Path) -> None:
    save_config(Config(pods={}, active="ghost"))
    assert get_active_pod() is None


def test_get_active_pod_returns_named_pod(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    active = get_active_pod()
    assert active is not None
    assert active.name == "a"
    assert active.pod.ssh == "ssh a"


def test_set_active_pod(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    add_pod("b", Pod(ssh="ssh b"))
    set_active_pod("b")
    assert load_config().active == "b"
