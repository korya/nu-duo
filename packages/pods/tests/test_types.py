"""Tests for ``nu_pods.types`` (de)serialization."""

from __future__ import annotations

import pytest
from nu_pods.types import GPU, Config, Model, Pod, PodsError


def test_gpu_round_trip() -> None:
    gpu = GPU(id=0, name="NVIDIA H200", memory="141GB")
    assert GPU.from_dict(gpu.to_dict()) == gpu


def test_model_round_trip() -> None:
    model = Model(model="Qwen/Qwen3-Coder-30B-A3B-Instruct", port=8001, gpu=[0, 1], pid=12345)
    assert Model.from_dict(model.to_dict()) == model


def test_pod_round_trip_with_optional_fields() -> None:
    pod = Pod(
        ssh="ssh root@host -p 22",
        gpus=[GPU(id=0, name="NVIDIA H100", memory="80GB")],
        models={"primary": Model(model="m", port=8001, gpu=[0], pid=42)},
        models_path="/mnt/sfs",
        vllm_version="release",
    )
    raw = pod.to_dict()
    # camelCase keys preserved on the wire for cross-version compat.
    assert raw["modelsPath"] == "/mnt/sfs"
    assert raw["vllmVersion"] == "release"
    restored = Pod.from_dict(raw)
    assert restored == pod


def test_pod_omits_none_optional_fields() -> None:
    pod = Pod(ssh="ssh root@host")
    raw = pod.to_dict()
    assert "modelsPath" not in raw
    assert "vllmVersion" not in raw
    assert raw["gpus"] == []
    assert raw["models"] == {}


def test_pod_rejects_invalid_vllm_version() -> None:
    with pytest.raises(PodsError, match="vllmVersion"):
        Pod.from_dict({"ssh": "ssh root@host", "vllmVersion": "bogus"})


def test_config_round_trip_with_active() -> None:
    config = Config(
        pods={"a": Pod(ssh="ssh a"), "b": Pod(ssh="ssh b")},
        active="b",
    )
    raw = config.to_dict()
    assert raw["active"] == "b"
    assert set(raw["pods"]) == {"a", "b"}
    restored = Config.from_dict(raw)
    assert restored.active == "b"
    assert set(restored.pods) == {"a", "b"}


def test_config_round_trip_without_active() -> None:
    config = Config(pods={"a": Pod(ssh="ssh a")})
    assert "active" not in config.to_dict()
    assert Config.from_dict(config.to_dict()).active is None


def test_pods_error_carries_exit_code() -> None:
    err = PodsError("boom", exit_code=2)
    assert err.message == "boom"
    assert err.exit_code == 2
    assert str(err) == "boom"
