"""Pods configuration persistence.

Port of ``packages/pods/src/config.ts``. Stores pods configuration as
JSON in ``$NU_PODS_CONFIG_DIR/pods.json`` (default
``~/.nu/pods.json``). The on-disk shape is byte-compatible with the TS
version (camelCase keys via :meth:`Pod.to_dict`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from nu_pods.types import ActivePod, Config, Pod


def get_config_dir() -> Path:
    """Resolve the config directory (override via ``NU_PODS_CONFIG_DIR``)."""
    override = os.environ.get("NU_PODS_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".nu"


def get_config_path() -> Path:
    return get_config_dir() / "pods.json"


def load_config() -> Config:
    """Load the config file. Returns an empty :class:`Config` if absent."""
    path = get_config_path()
    if not path.exists():
        return Config()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Mirror TS behaviour: on a corrupt file, treat as empty rather
        # than crash — the user can re-run ``setup`` to repair it.
        return Config()
    return Config.from_dict(raw)


def save_config(config: Config) -> None:
    """Persist the config file, creating the parent directory if needed."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def get_active_pod() -> ActivePod | None:
    """Return the currently active pod, or ``None`` if none is set."""
    config = load_config()
    if not config.active:
        return None
    pod = config.pods.get(config.active)
    if pod is None:
        return None
    return ActivePod(name=config.active, pod=pod)


def add_pod(name: str, pod: Pod) -> None:
    """Add or replace a pod. The first pod added becomes active."""
    config = load_config()
    config.pods[name] = pod
    if not config.active:
        config.active = name
    save_config(config)


def remove_pod(name: str) -> None:
    """Remove a pod. If it was active, clear the active pointer."""
    config = load_config()
    if name in config.pods:
        del config.pods[name]
    if config.active == name:
        config.active = None
    save_config(config)


def set_active_pod(name: str) -> None:
    """Set the active pod (no-op safety: caller validates existence)."""
    config = load_config()
    config.active = name
    save_config(config)
