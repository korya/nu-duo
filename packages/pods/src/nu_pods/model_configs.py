"""Known-model configuration loader.

Port of ``packages/pods/src/model-configs.ts``. The catalogue itself
lives in ``resources/models.json`` (vendored verbatim from upstream so
the two implementations stay in lockstep).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nu_pods.types import GPU


@dataclass(slots=True)
class ModelConfig:
    gpu_count: int
    gpu_types: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(
            gpu_count=int(data.get("gpuCount", 1)),
            gpu_types=list(data.get("gpuTypes", []) or []),
            args=list(data.get("args", []) or []),
            env=dict(data.get("env", {}) or {}),
            notes=data.get("notes"),
        )


@dataclass(slots=True)
class ModelInfo:
    id: str
    name: str
    configs: list[ModelConfig] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_dict(cls, model_id: str, data: dict[str, Any]) -> ModelInfo:
        return cls(
            id=model_id,
            name=str(data.get("name", model_id)),
            configs=[ModelConfig.from_dict(c) for c in (data.get("configs") or [])],
            notes=data.get("notes"),
        )


@lru_cache(maxsize=1)
def _load_models_json() -> dict[str, ModelInfo]:
    text = resources.files("nu_pods.resources").joinpath("models.json").read_text(encoding="utf-8")
    raw = json.loads(text)
    models = raw.get("models") or {}
    return {model_id: ModelInfo.from_dict(model_id, info) for model_id, info in models.items()}


def get_known_models() -> dict[str, ModelInfo]:
    """Return all known models keyed by model id."""
    return dict(_load_models_json())


def is_known_model(model_id: str) -> bool:
    return model_id in _load_models_json()


def get_model_name(model_id: str) -> str:
    info = _load_models_json().get(model_id)
    return info.name if info is not None else model_id


def gpu_type_token(gpu_name: str) -> str:
    """Extract a short type token from an ``nvidia-smi`` GPU name.

    Example: ``"NVIDIA H200"`` → ``"H200"``. Mirrors the TS extraction
    in ``showKnownModels``: strip ``NVIDIA``, take the first word.
    """
    return gpu_name.replace("NVIDIA", "").strip().split(" ")[0] if gpu_name else ""


def gpu_type_matches(config_types: list[str], pod_type: str) -> bool:
    if not config_types:
        return True
    if not pod_type:
        return False
    return any(t in pod_type or pod_type in t for t in config_types)


def get_model_config(model_id: str, gpus: list[GPU], requested_gpu_count: int | None = None) -> ModelConfig | None:
    """Pick the best :class:`ModelConfig` for ``model_id`` on ``gpus``.

    If ``requested_gpu_count`` is given, only configs with exactly that
    many GPUs are considered. Otherwise we walk the configs in order
    and return the first one whose GPU count and (optional) GPU type
    matches the available pod hardware.
    """
    info = _load_models_json().get(model_id)
    if info is None or not info.configs:
        return None

    pod_type = gpu_type_token(gpus[0].name) if gpus else ""

    for config in info.configs:
        if requested_gpu_count is not None and config.gpu_count != requested_gpu_count:
            continue
        if config.gpu_count > len(gpus):
            continue
        if not gpu_type_matches(config.gpu_types, pod_type):
            continue
        return config

    return None
