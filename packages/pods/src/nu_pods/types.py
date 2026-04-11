"""Core types for ``nu_pods``.

Direct port of ``packages/pods/src/types.ts``. Field names match the
TypeScript originals (camelCase) on the wire so config files written by
either implementation can be read by the other; in-memory the dataclass
fields use ``snake_case`` and ``to_dict``/``from_dict`` translate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VllmVersion = Literal["release", "nightly", "gpt-oss"]


class PodsError(Exception):
    """Raised by command-layer code instead of calling ``sys.exit``.

    The CLI catches it and translates ``exit_code`` into a process exit.
    Tests can assert on ``exit_code`` and ``message`` directly without
    monkey-patching ``sys.exit``.
    """

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


@dataclass(slots=True)
class GPU:
    """Single GPU description (matches TS ``GPU`` interface)."""

    id: int
    name: str
    memory: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "memory": self.memory}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPU:
        return cls(id=int(data["id"]), name=str(data["name"]), memory=str(data["memory"]))


@dataclass(slots=True)
class Model:
    """A running vLLM model on a pod."""

    model: str
    port: int
    gpu: list[int]
    pid: int

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "port": self.port, "gpu": list(self.gpu), "pid": self.pid}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Model:
        return cls(
            model=str(data["model"]),
            port=int(data["port"]),
            gpu=[int(g) for g in data.get("gpu", [])],
            pid=int(data["pid"]),
        )


@dataclass(slots=True)
class Pod:
    """A configured GPU pod."""

    ssh: str
    gpus: list[GPU] = field(default_factory=list)
    models: dict[str, Model] = field(default_factory=dict)
    models_path: str | None = None
    vllm_version: VllmVersion | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ssh": self.ssh,
            "gpus": [g.to_dict() for g in self.gpus],
            "models": {name: m.to_dict() for name, m in self.models.items()},
        }
        if self.models_path is not None:
            out["modelsPath"] = self.models_path
        if self.vllm_version is not None:
            out["vllmVersion"] = self.vllm_version
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pod:
        gpus = [GPU.from_dict(g) for g in data.get("gpus", [])]
        models = {name: Model.from_dict(m) for name, m in (data.get("models") or {}).items()}
        vllm_version = data.get("vllmVersion")
        if vllm_version is not None and vllm_version not in ("release", "nightly", "gpt-oss"):
            raise PodsError(f"Invalid vllmVersion: {vllm_version!r}")
        return cls(
            ssh=str(data["ssh"]),
            gpus=gpus,
            models=models,
            models_path=data.get("modelsPath"),
            vllm_version=vllm_version,
        )


@dataclass(slots=True)
class Config:
    """Top-level pods configuration."""

    pods: dict[str, Pod] = field(default_factory=dict)
    active: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"pods": {name: p.to_dict() for name, p in self.pods.items()}}
        if self.active is not None:
            out["active"] = self.active
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        pods = {name: Pod.from_dict(p) for name, p in (data.get("pods") or {}).items()}
        return cls(pods=pods, active=data.get("active"))


@dataclass(slots=True)
class ActivePod:
    """Result of looking up the active pod — the name plus the pod itself."""

    name: str
    pod: Pod
