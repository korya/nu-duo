"""Bridge from a deployed model to the coding agent.

Port of ``packages/pods/src/commands/prompt.ts``. The TS version
ultimately delegates to ``pi-coding-agent`` (or stubs out with ``throw
new Error("Not implemented")`` in the upstream snapshot we read). The
Python version produces the argv list that would be passed to ``nu``
and exposes a callable hook so the CLI / tests can decide whether to
exec, subprocess, or just inspect.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TextIO

from nu_pods.config import get_active_pod, load_config
from nu_pods.types import ActivePod, Pod, PodsError


@dataclass(slots=True)
class PromptInvocation:
    """Argv-and-env bundle that would launch the coding agent."""

    args: list[str] = field(default_factory=list)
    system_prompt: str = ""
    base_url: str = ""
    model_id: str = ""
    api_key: str = ""


# Pluggable launcher: by default the CLI raises ``PodsError`` because
# the agent-launch path is not yet wired into nu_coding_agent. Tests
# (and the eventual real implementation) override this.
PromptLauncher = Callable[[PromptInvocation], Awaitable[int]]


async def _default_launcher(_: PromptInvocation) -> int:
    raise PodsError("Agent launch is not yet implemented in nu-pods")


@dataclass(slots=True)
class _LauncherState:
    launcher: PromptLauncher


_state = _LauncherState(launcher=_default_launcher)


def set_launcher(launcher: PromptLauncher) -> PromptLauncher:
    """Override the launcher (for tests and the eventual real binding)."""
    prev = _state.launcher
    _state.launcher = launcher
    return prev


def reset_launcher() -> None:
    _state.launcher = _default_launcher


def _resolve_active_pod(pod_override: str | None) -> ActivePod:
    if pod_override:
        config = load_config()
        pod = config.pods.get(pod_override)
        if pod is None:
            raise PodsError(f"Pod '{pod_override}' not found")
        return ActivePod(name=pod_override, pod=pod)
    active = get_active_pod()
    if active is None:
        raise PodsError("No active pod. Use 'nu-pods pods active <name>' to set one.")
    return active


def _extract_host(pod: Pod) -> str:
    for token in pod.ssh.split(" "):
        if "@" in token:
            return token.split("@", 1)[1]
    return "localhost"


def build_invocation(
    model_name: str,
    user_args: list[str],
    *,
    pod_override: str | None = None,
    api_key_override: str | None = None,
    cwd: str | None = None,
) -> PromptInvocation:
    """Build a :class:`PromptInvocation` for the given model name."""
    active = _resolve_active_pod(pod_override)
    model_config = active.pod.models.get(model_name)
    if model_config is None:
        raise PodsError(f"Model '{model_name}' not found on pod '{active.name}'")

    host = _extract_host(active.pod)
    base_url = f"http://{host}:{model_config.port}/v1"
    api_key = api_key_override or os.environ.get("NU_API_KEY") or os.environ.get("PI_API_KEY") or "dummy"
    api = "responses" if "gpt-oss" in model_config.model.lower() else "completions"

    working_dir = cwd if cwd is not None else os.getcwd()
    system_prompt = (
        "You help the user understand and navigate the codebase in the current working directory.\n\n"
        "You can read files, list directories, and execute shell commands via the respective tools.\n\n"
        "Do not output file contents you read via the read_file tool directly, unless asked to.\n\n"
        "Do not output markdown tables as part of your responses.\n\n"
        "Keep your responses concise and relevant to the user's request.\n\n"
        "File paths you output must include line numbers where possible, "
        'e.g. "src/index.ts:10-20" for lines 10 to 20 in src/index.ts.\n\n'
        f"Current working directory: {working_dir}"
    )

    args: list[str] = [
        "--base-url",
        base_url,
        "--model",
        model_config.model,
        "--api-key",
        api_key,
        "--api",
        api,
        "--system-prompt",
        system_prompt,
        *user_args,
    ]

    return PromptInvocation(
        args=args,
        system_prompt=system_prompt,
        base_url=base_url,
        model_id=model_config.model,
        api_key=api_key,
    )


async def prompt_model(
    model_name: str,
    user_args: list[str],
    *,
    pod: str | None = None,
    api_key: str | None = None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Resolve the target model and hand off to the configured launcher."""
    try:
        invocation = build_invocation(model_name, user_args, pod_override=pod, api_key_override=api_key)
    except PodsError as exc:
        stderr.write(exc.message + "\n")
        raise
    return await _state.launcher(invocation)
