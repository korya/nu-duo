"""Auto-generated model catalog (snapshot).

Loads ``resources/models_generated.json`` at import time and materialises it
into ``MODELS: dict[str, dict[str, Model]]``. The JSON file is a byte-for-byte
translation of ``packages/ai/src/models.generated.ts`` from the upstream TS
monorepo; regenerate it via ``scripts/generate_models.py`` (to be ported).

Do not edit the JSON file manually.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Final

from pi_ai.types import Model


def _load() -> dict[str, dict[str, Model]]:
    raw = json.loads(files("pi_ai.resources").joinpath("models_generated.json").read_text(encoding="utf-8"))
    catalog: dict[str, dict[str, Model]] = {}
    for provider, models in raw.items():
        catalog[provider] = {model_id: Model.model_validate(record) for model_id, record in models.items()}
    return catalog


MODELS: Final[dict[str, dict[str, Model]]] = _load()


__all__ = ["MODELS"]
