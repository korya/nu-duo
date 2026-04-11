"""Command-layer for ``nu-pods``.

Each submodule corresponds to a TS file under
``packages/pods/src/commands/`` upstream.
"""

from nu_pods.commands.models import (
    list_models,
    show_known_models,
    start_model,
    stop_all_models,
    stop_model,
    view_logs,
)
from nu_pods.commands.pods import (
    list_pods,
    remove_pod_command,
    setup_pod,
    switch_active_pod,
)
from nu_pods.commands.prompt import prompt_model

__all__ = [
    "list_models",
    "list_pods",
    "prompt_model",
    "remove_pod_command",
    "setup_pod",
    "show_known_models",
    "start_model",
    "stop_all_models",
    "stop_model",
    "switch_active_pod",
    "view_logs",
]
