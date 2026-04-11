"""``nu-pods`` command-line entry point.

Hand-rolled argv parser matching the project's existing style
(see ``nu_coding_agent.cli``). Mirrors the TS subcommand layout in
``packages/pods/src/cli.ts``:

    nu-pods pods setup <name> <ssh> [--mount ...] [--models-path ...] [--vllm ...]
    nu-pods pods active <name>
    nu-pods pods remove <name>
    nu-pods pods list
    nu-pods start <model> --name <name> [--pod ...] [--memory ...] [--context ...] [--gpus N] [--vllm ARG ...]
    nu-pods stop <name> [--pod ...]
    nu-pods stop --all [--pod ...]
    nu-pods list [--pod ...]
    nu-pods logs <name> [--pod ...]
    nu-pods models
    nu-pods agent <model> [--pod ...] [--api-key ...] [-- ...args]

Excluded from coverage as a thin glue layer (matches the policy
applied to ``nu_coding_agent.cli``).
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Sequence

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
from nu_pods.types import PodsError, VllmVersion

USAGE = """nu-pods - vLLM pod manager (Python port of pi-pods)

Usage:
  nu-pods pods setup <name> <ssh> [--mount CMD] [--models-path PATH] [--vllm VER]
  nu-pods pods active <name>
  nu-pods pods remove <name>
  nu-pods pods list
  nu-pods start <model> --name <name> [--pod NAME] [--memory PCT]
                       [--context SIZE] [--gpus N] [--vllm ARG ...]
  nu-pods stop <name> [--pod NAME]
  nu-pods stop --all [--pod NAME]
  nu-pods list [--pod NAME]
  nu-pods logs <name> [--pod NAME]
  nu-pods models
  nu-pods agent <model> [--pod NAME] [--api-key KEY] [-- ARGS...]
"""


def _take_value(argv: list[str], i: int, flag: str) -> tuple[str, int]:
    if i + 1 >= len(argv):
        raise PodsError(f"{flag} requires a value")
    return argv[i + 1], i + 2


def _parse_vllm_version(value: str) -> VllmVersion:
    if value not in ("release", "nightly", "gpt-oss"):
        raise PodsError(f"Unknown --vllm version: {value}")
    return value  # type: ignore[return-value]


async def dispatch(argv: Sequence[str], stdout: TextIO, stderr: TextIO) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        stdout.write(USAGE)
        return 0

    command = args.pop(0)

    if command == "pods":
        if not args:
            stdout.write(USAGE)
            return 1
        sub = args.pop(0)
        if sub == "list":
            list_pods(stdout=stdout)
            return 0
        if sub == "active":
            if not args:
                raise PodsError("pods active requires a name")
            switch_active_pod(args[0], stdout=stdout, stderr=stderr)
            return 0
        if sub == "remove":
            if not args:
                raise PodsError("pods remove requires a name")
            remove_pod_command(args[0], stdout=stdout, stderr=stderr)
            return 0
        if sub == "setup":
            if len(args) < 2:
                raise PodsError("pods setup requires <name> <ssh>")
            name = args[0]
            ssh = args[1]
            i = 2
            mount: str | None = None
            models_path: str | None = None
            vllm: VllmVersion | None = None
            while i < len(args):
                tok = args[i]
                if tok == "--mount":
                    mount, i = _take_value(args, i, "--mount")
                elif tok == "--models-path":
                    models_path, i = _take_value(args, i, "--models-path")
                elif tok == "--vllm":
                    raw, i = _take_value(args, i, "--vllm")
                    vllm = _parse_vllm_version(raw)
                else:
                    raise PodsError(f"Unknown flag: {tok}")
            await setup_pod(
                name,
                ssh,
                mount=mount,
                models_path=models_path,
                vllm=vllm,
                stdout=stdout,
                stderr=stderr,
            )
            return 0
        raise PodsError(f"Unknown pods subcommand: {sub}")

    if command == "models":
        show_known_models(stdout=stdout)
        return 0

    if command == "list":
        pod = _extract_pod_flag(args)
        await list_models(pod=pod, stdout=stdout, stderr=stderr)
        return 0

    if command == "logs":
        if not args:
            raise PodsError("logs requires <name>")
        name = args.pop(0)
        pod = _extract_pod_flag(args)
        await view_logs(name, pod=pod, stdout=stdout, stderr=stderr)
        return 0

    if command == "stop":
        pod = _extract_pod_flag(args)
        if "--all" in args:
            await stop_all_models(pod=pod, stdout=stdout, stderr=stderr)
            return 0
        if not args:
            raise PodsError("stop requires <name> or --all")
        await stop_model(args[0], pod=pod, stdout=stdout, stderr=stderr)
        return 0

    if command == "start":
        if not args:
            raise PodsError("start requires <model>")
        model_id = args.pop(0)
        name: str | None = None
        pod: str | None = None
        memory: str | None = None
        context: str | None = None
        gpus: int | None = None
        vllm_args: list[str] = []
        i = 0
        while i < len(args):
            tok = args[i]
            if tok == "--name":
                name, i = _take_value(args, i, "--name")
            elif tok == "--pod":
                pod, i = _take_value(args, i, "--pod")
            elif tok == "--memory":
                memory, i = _take_value(args, i, "--memory")
            elif tok == "--context":
                context, i = _take_value(args, i, "--context")
            elif tok == "--gpus":
                raw, i = _take_value(args, i, "--gpus")
                gpus = int(raw)
            elif tok == "--vllm":
                # Everything after --vllm is forwarded to vLLM verbatim.
                vllm_args = list(args[i + 1 :])
                break
            else:
                raise PodsError(f"Unknown flag: {tok}")
        if not name:
            raise PodsError("start requires --name")
        await start_model(
            model_id,
            name,
            pod=pod,
            vllm_args=vllm_args or None,
            memory=memory,
            context=context,
            gpus=gpus,
            stdout=stdout,
            stderr=stderr,
        )
        return 0

    if command == "agent":
        if not args:
            raise PodsError("agent requires <model>")
        model_name = args.pop(0)
        pod: str | None = None
        api_key: str | None = None
        passthrough: list[str] = []
        i = 0
        while i < len(args):
            tok = args[i]
            if tok == "--pod":
                pod, i = _take_value(args, i, "--pod")
            elif tok == "--api-key":
                api_key, i = _take_value(args, i, "--api-key")
            elif tok == "--":
                passthrough = list(args[i + 1 :])
                break
            else:
                passthrough = list(args[i:])
                break
        return await prompt_model(
            model_name,
            passthrough,
            pod=pod,
            api_key=api_key,
            stdout=stdout,
            stderr=stderr,
        )

    raise PodsError(f"Unknown command: {command}")


def _extract_pod_flag(args: list[str]) -> str | None:
    """Pop ``--pod NAME`` from ``args`` in place and return the value."""
    if "--pod" in args:
        idx = args.index("--pod")
        if idx + 1 >= len(args):
            raise PodsError("--pod requires a value")
        value = args[idx + 1]
        del args[idx : idx + 2]
        return value
    return None


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        return asyncio.run(dispatch(argv, sys.stdout, sys.stderr))
    except PodsError as exc:
        sys.stderr.write(f"{exc.message}\n")
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
