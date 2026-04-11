"""Tests for ``nu_pods.model_configs`` lookup logic."""

from __future__ import annotations

from nu_pods.model_configs import (
    get_known_models,
    get_model_config,
    get_model_name,
    gpu_type_matches,
    gpu_type_token,
    is_known_model,
)
from nu_pods.types import GPU


def _h200(n: int) -> list[GPU]:
    return [GPU(id=i, name="NVIDIA H200", memory="141GB") for i in range(n)]


def test_gpu_type_token() -> None:
    assert gpu_type_token("NVIDIA H200") == "H200"
    assert gpu_type_token("NVIDIA H100 80GB PCIe") == "H100"
    assert gpu_type_token("") == ""


def test_gpu_type_matches() -> None:
    assert gpu_type_matches([], "H200") is True  # no constraint
    assert gpu_type_matches(["H100", "H200"], "H200") is True
    assert gpu_type_matches(["H200"], "H100") is False
    assert gpu_type_matches(["H200"], "") is False  # empty pod type


def test_known_models_loaded() -> None:
    models = get_known_models()
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in models
    assert is_known_model("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    assert not is_known_model("nonexistent/model")


def test_get_model_name_known_and_unknown() -> None:
    assert get_model_name("Qwen/Qwen3-Coder-30B-A3B-Instruct") == "Qwen3-Coder-30B"
    assert get_model_name("unknown/x") == "unknown/x"


def test_get_model_config_picks_first_compatible() -> None:
    # 1xH200 should resolve to the 1-GPU config.
    config = get_model_config("Qwen/Qwen3-Coder-30B-A3B-Instruct", _h200(1))
    assert config is not None
    assert config.gpu_count == 1


def test_get_model_config_respects_requested_count() -> None:
    config = get_model_config("Qwen/Qwen3-Coder-30B-A3B-Instruct", _h200(4), 2)
    assert config is not None
    assert config.gpu_count == 2
    assert "--tensor-parallel-size" in config.args


def test_get_model_config_unknown_returns_none() -> None:
    assert get_model_config("nonexistent/model", _h200(1)) is None


def test_get_model_config_insufficient_gpus_returns_none() -> None:
    # GLM-4.5 needs at least 8 H200s; a single H200 isn't enough.
    config = get_model_config("zai-org/GLM-4.5", _h200(1))
    assert config is None


def test_get_model_config_gpu_type_mismatch_returns_none() -> None:
    # GPT-OSS-20B is configured for H100/H200 (and B200) — an A100
    # pod has neither match.
    a100 = [GPU(id=0, name="NVIDIA A100", memory="80GB")]
    assert get_model_config("openai/gpt-oss-20b", a100) is None


def test_get_model_config_no_match_for_requested_count() -> None:
    # Qwen3-Coder-30B-FP8 only has a 1-GPU config; requesting 2 → None.
    config = get_model_config("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", _h200(2), 2)
    assert config is None
