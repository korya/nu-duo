"""Pod-management commands.

Port of ``packages/pods/src/commands/pods.ts``. Functions raise
:class:`PodsError` on failure rather than calling ``sys.exit`` so the
CLI layer (and tests) can decide what to do with the exit code.
"""

from __future__ import annotations

import os
from importlib import resources
from typing import TextIO

from nu_pods import ssh as ssh_mod
from nu_pods.config import add_pod, load_config, remove_pod, set_active_pod
from nu_pods.types import GPU, Pod, PodsError, VllmVersion


def _print(out: TextIO, line: str) -> None:
    out.write(line + "\n")


def list_pods(*, stdout: TextIO) -> None:
    """Print all configured pods to ``stdout``."""
    config = load_config()
    pod_names = list(config.pods.keys())

    if not pod_names:
        _print(stdout, "No pods configured. Use 'nu-pods pods setup' to add a pod.")
        return

    _print(stdout, "Configured pods:")
    for name in pod_names:
        pod = config.pods[name]
        marker = "*" if config.active == name else " "
        gpu_count = len(pod.gpus)
        gpu_info = f"{gpu_count}x {pod.gpus[0].name}" if gpu_count > 0 else "no GPUs detected"
        vllm_info = f" (vLLM: {pod.vllm_version})" if pod.vllm_version else ""
        _print(stdout, f"{marker} {name} - {gpu_info}{vllm_info} - {pod.ssh}")
        if pod.models_path:
            _print(stdout, f"    Models: {pod.models_path}")
        if pod.vllm_version == "gpt-oss":
            _print(stdout, "    ⚠️  GPT-OSS build - only for GPT-OSS models")


def parse_gpu_csv(csv: str) -> list[GPU]:
    gpus: list[GPU] = []
    for line in csv.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 1 or not parts[0]:
            continue
        try:
            gpu_id = int(parts[0])
        except ValueError:
            continue
        name = parts[1] if len(parts) > 1 and parts[1] else "Unknown"
        memory = parts[2] if len(parts) > 2 and parts[2] else "Unknown"
        gpus.append(GPU(id=gpu_id, name=name, memory=memory))
    return gpus


async def setup_pod(
    name: str,
    ssh_cmd: str,
    *,
    mount: str | None = None,
    models_path: str | None = None,
    vllm: VllmVersion | None = None,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    """Bootstrap a fresh GPU pod for vLLM deployments."""
    hf_token = os.environ.get("HF_TOKEN")
    nu_api_key = os.environ.get("NU_API_KEY") or os.environ.get("PI_API_KEY")

    if not hf_token:
        _print(stderr, "ERROR: HF_TOKEN environment variable is required")
        _print(stderr, "Get a token from: https://huggingface.co/settings/tokens")
        _print(stderr, "Then run: export HF_TOKEN=your_token_here")
        raise PodsError("HF_TOKEN missing")

    if not nu_api_key:
        _print(stderr, "ERROR: NU_API_KEY environment variable is required")
        _print(stderr, "Set an API key: export NU_API_KEY=your_api_key_here")
        raise PodsError("NU_API_KEY missing")

    # Determine models path: explicit flag wins; otherwise extract the
    # last token from the mount command (mirrors TS behaviour).
    resolved_models_path = models_path
    if not resolved_models_path and mount:
        parts = mount.split(" ")
        resolved_models_path = parts[-1]

    if not resolved_models_path:
        _print(stderr, "ERROR: --models-path is required (or must be extractable from --mount)")
        raise PodsError("models-path missing")

    chosen_vllm: VllmVersion = vllm or "release"

    _print(stdout, f"Setting up pod '{name}'...")
    _print(stdout, f"SSH: {ssh_cmd}")
    _print(stdout, f"Models path: {resolved_models_path}")
    gpt_oss_warn = " (GPT-OSS special build)" if chosen_vllm == "gpt-oss" else ""
    _print(stdout, f"vLLM version: {chosen_vllm}{gpt_oss_warn}")
    if mount:
        _print(stdout, f"Mount command: {mount}")
    _print(stdout, "")

    _print(stdout, "Testing SSH connection...")
    test_result = await ssh_mod.ssh_exec(ssh_cmd, "echo 'SSH OK'")
    if test_result.exit_code != 0:
        _print(stderr, "Failed to connect via SSH")
        _print(stderr, test_result.stderr)
        raise PodsError("SSH connection failed")
    _print(stdout, "✓ SSH connection successful")

    _print(stdout, "Copying setup script...")
    script_path = str(resources.files("nu_pods.resources").joinpath("pod_setup.sh"))
    success = await ssh_mod.scp_file(ssh_cmd, script_path, "/tmp/pod_setup.sh")
    if not success:
        _print(stderr, "Failed to copy setup script")
        raise PodsError("scp failed")
    _print(stdout, "✓ Setup script copied")

    setup_cmd = (
        f"bash /tmp/pod_setup.sh --models-path '{resolved_models_path}'"
        f" --hf-token '{hf_token}' --vllm-api-key '{nu_api_key}'"
    )
    if mount:
        setup_cmd += f" --mount '{mount}'"
    setup_cmd += f" --vllm '{chosen_vllm}'"

    _print(stdout, "")
    _print(stdout, "Running setup (this will take 2-5 minutes)...")
    _print(stdout, "")

    exit_code = await ssh_mod.ssh_exec_stream(ssh_cmd, setup_cmd, force_tty=True)
    if exit_code != 0:
        _print(stderr, "\nSetup failed. Check the output above for errors.")
        raise PodsError("setup script failed")

    _print(stdout, "")
    _print(stdout, "Detecting GPU configuration...")
    gpu_result = await ssh_mod.ssh_exec(
        ssh_cmd,
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
    )

    gpus: list[GPU] = []
    if gpu_result.exit_code == 0 and gpu_result.stdout:
        gpus = parse_gpu_csv(gpu_result.stdout)

    _print(stdout, f"✓ Detected {len(gpus)} GPU(s)")
    for gpu in gpus:
        _print(stdout, f"  GPU {gpu.id}: {gpu.name} ({gpu.memory})")

    pod = Pod(
        ssh=ssh_cmd,
        gpus=gpus,
        models={},
        models_path=resolved_models_path,
        vllm_version=chosen_vllm,
    )
    add_pod(name, pod)
    _print(stdout, "")
    _print(stdout, f"✓ Pod '{name}' setup complete and set as active pod")
    _print(stdout, "")
    _print(stdout, "You can now deploy models with:")
    _print(stdout, "  nu-pods start <model> --name <name>")


def switch_active_pod(name: str, *, stdout: TextIO, stderr: TextIO) -> None:
    config = load_config()
    if name not in config.pods:
        _print(stderr, f"Pod '{name}' not found")
        _print(stderr, "")
        _print(stderr, "Available pods:")
        for pod_name in config.pods:
            _print(stderr, f"  {pod_name}")
        raise PodsError(f"unknown pod {name!r}")

    set_active_pod(name)
    _print(stdout, f"✓ Switched active pod to '{name}'")


def remove_pod_command(name: str, *, stdout: TextIO, stderr: TextIO) -> None:
    config = load_config()
    if name not in config.pods:
        _print(stderr, f"Pod '{name}' not found")
        raise PodsError(f"unknown pod {name!r}")

    remove_pod(name)
    _print(stdout, f"✓ Removed pod '{name}' from configuration")
    _print(stdout, "Note: This only removes the local configuration. The remote pod is not affected.")
