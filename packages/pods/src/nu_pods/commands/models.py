"""Model lifecycle commands.

Port of ``packages/pods/src/commands/models.ts``. Pure helpers
(port/GPU selection, vLLM args building, env building, log line
classification) are factored out so unit tests can drive them
without spinning up an SSH session.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from importlib import resources
from typing import TextIO

from nu_pods import ssh as ssh_mod
from nu_pods.config import get_active_pod, load_config, save_config
from nu_pods.model_configs import (
    ModelConfig,
    get_model_config,
    get_model_name,
    gpu_type_token,
    is_known_model,
)
from nu_pods.types import Model, Pod, PodsError

# ---------------------------------------------------------------------------
# Pure helpers (no SSH, no environment access)
# ---------------------------------------------------------------------------


def get_pod(pod_override: str | None = None) -> tuple[str, Pod]:
    """Resolve the target pod (override → active pod). Raises ``PodsError``."""
    if pod_override:
        config = load_config()
        pod = config.pods.get(pod_override)
        if pod is None:
            raise PodsError(f"Pod '{pod_override}' not found")
        return pod_override, pod

    active = get_active_pod()
    if active is None:
        raise PodsError("No active pod. Use 'nu-pods pods active <name>' to set one.")
    return active.name, active.pod


def next_port(pod: Pod, start: int = 8001) -> int:
    """Pick the next free port on a pod, starting from ``start``."""
    used = {m.port for m in pod.models.values()}
    port = start
    while port in used:
        port += 1
    return port


def select_gpus(pod: Pod, count: int = 1) -> list[int]:
    """Choose ``count`` GPU ids on the pod, preferring least-used.

    Mirrors the TS round-robin: count current usage across all running
    models and pick the GPUs with the fewest references.
    """
    if count == len(pod.gpus):
        return [g.id for g in pod.gpus]

    usage: dict[int, int] = {gpu.id: 0 for gpu in pod.gpus}
    for model in pod.models.values():
        for gpu_id in model.gpu:
            if gpu_id in usage:
                usage[gpu_id] += 1

    sorted_ids = sorted(usage.items(), key=lambda kv: kv[1])
    return [gpu_id for gpu_id, _ in sorted_ids[:count]]


_CONTEXT_SIZES: dict[str, int] = {
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
}


def parse_context_size(value: str) -> int:
    """Translate ``"4k"`` etc. to a token count, or fall back to ``int``."""
    named = _CONTEXT_SIZES.get(value.lower())
    if named is not None:
        return named
    return int(value)


def parse_memory_fraction(value: str) -> float:
    """Translate ``"50%"`` to ``0.5``."""
    return float(value.replace("%", "")) / 100


def apply_memory_and_context(args: list[str], memory: str | None, context: str | None) -> list[str]:
    """Strip+re-apply ``--gpu-memory-utilization`` / ``--max-model-len``."""
    out = list(args)
    if memory:
        out = [a for a in out if "gpu-memory-utilization" not in a]
        out.extend(["--gpu-memory-utilization", str(parse_memory_fraction(memory))])
    if context:
        out = [a for a in out if "max-model-len" not in a]
        out.extend(["--max-model-len", str(parse_context_size(context))])
    return out


def build_env_block(
    *,
    hf_token: str,
    nu_api_key: str,
    gpus: list[int],
    extra_env: dict[str, str] | None = None,
) -> str:
    """Build the ``export FOO='bar'`` block written to the remote shell."""
    pairs: list[str] = [
        f"HF_TOKEN='{hf_token}'",
        f"PI_API_KEY='{nu_api_key}'",
        "HF_HUB_ENABLE_HF_TRANSFER=1",
        "VLLM_NO_USAGE_STATS=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "FORCE_COLOR=1",
        "TERM=xterm-256color",
    ]
    if len(gpus) == 1:
        pairs.append(f"CUDA_VISIBLE_DEVICES={gpus[0]}")
    if extra_env:
        for key, value in extra_env.items():
            pairs.append(f"{key}='{value}'")
    return "\n".join(f"export {p}" for p in pairs)


def render_model_run_script(model_id: str, name: str, port: int, vllm_args: list[str]) -> str:
    """Substitute placeholders in the bundled ``model_run.sh`` template."""
    template = resources.files("nu_pods.resources").joinpath("model_run.sh").read_text(encoding="utf-8")
    return (
        template.replace("{{MODEL_ID}}", model_id)
        .replace("{{NAME}}", name)
        .replace("{{PORT}}", str(port))
        .replace("{{VLLM_ARGS}}", " ".join(vllm_args))
    )


@dataclass(slots=True)
class StartupClassification:
    """Result of inspecting one streamed log line."""

    complete: bool = False
    failed: bool = False
    reason: str = ""


def classify_log_line(line: str) -> StartupClassification:
    """Decide whether a log line signals startup-complete or failure."""
    result = StartupClassification()
    if "Application startup complete" in line:
        result.complete = True
        return result
    if "Model runner exiting with code" in line and "code 0" not in line:
        result.failed = True
        result.reason = "Model runner failed to start"
        return result
    if "Script exited with code" in line and "code 0" not in line:
        result.failed = True
        result.reason = "Script failed to execute"
        return result
    if "torch.OutOfMemoryError" in line or "CUDA out of memory" in line:
        result.failed = True
        result.reason = "Out of GPU memory (OOM)"
        return result
    if "RuntimeError: Engine core initialization failed" in line:
        result.failed = True
        result.reason = "vLLM engine initialization failed"
    return result


def extract_host(ssh_cmd: str) -> str:
    """Pull ``host`` out of an ``"ssh user@host ..."`` command string."""
    for token in ssh_cmd.split(" "):
        if "@" in token:
            return token.split("@", 1)[1]
    return "localhost"


# ---------------------------------------------------------------------------
# Resolution: pick a (gpus, vllm_args, model_config) triple for ``startModel``
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelResolution:
    gpus: list[int]
    vllm_args: list[str]
    config: ModelConfig | None


def resolve_model_deployment(
    pod: Pod,
    model_id: str,
    *,
    custom_vllm_args: list[str] | None,
    requested_gpus: int | None,
) -> ModelResolution:
    """Pick GPUs and vLLM args for a model deployment.

    Direct port of the giant if/else block at the top of ``startModel``
    in TS. Raises :class:`PodsError` on user-facing failures so the
    caller can decide how to print the explanation.
    """
    if custom_vllm_args:
        return ModelResolution(gpus=[], vllm_args=list(custom_vllm_args), config=None)

    if is_known_model(model_id):
        if requested_gpus is not None:
            if requested_gpus > len(pod.gpus):
                raise PodsError(f"Error: Requested {requested_gpus} GPUs but pod only has {len(pod.gpus)}")
            config = get_model_config(model_id, pod.gpus, requested_gpus)
            if config is None:
                raise PodsError(
                    f"Model '{get_model_name(model_id)}' does not have a configuration for {requested_gpus} GPU(s)"
                )
            return ModelResolution(
                gpus=select_gpus(pod, requested_gpus),
                vllm_args=list(config.args),
                config=config,
            )

        # Original behaviour: try the largest config that fits.
        for gpu_count in range(len(pod.gpus), 0, -1):
            config = get_model_config(model_id, pod.gpus, gpu_count)
            if config is not None:
                return ModelResolution(
                    gpus=select_gpus(pod, gpu_count),
                    vllm_args=list(config.args),
                    config=config,
                )
        raise PodsError(f"Model '{get_model_name(model_id)}' not compatible with this pod's GPUs")

    # Unknown model: single GPU default; --gpus is rejected.
    if requested_gpus is not None:
        raise PodsError("Error: --gpus can only be used with predefined models")
    return ModelResolution(gpus=select_gpus(pod, 1), vllm_args=[], config=None)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _print(out: TextIO, line: str = "") -> None:
    out.write(line + "\n")


async def start_model(
    model_id: str,
    name: str,
    *,
    pod: str | None = None,
    vllm_args: list[str] | None = None,
    memory: str | None = None,
    context: str | None = None,
    gpus: int | None = None,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    pod_name, pod_obj = get_pod(pod)

    if not pod_obj.models_path:
        raise PodsError("Pod does not have a models path configured")
    if name in pod_obj.models:
        raise PodsError(f"Model '{name}' already exists on pod '{pod_name}'")

    port = next_port(pod_obj)

    try:
        resolution = resolve_model_deployment(pod_obj, model_id, custom_vllm_args=vllm_args, requested_gpus=gpus)
    except PodsError as exc:
        _print(stderr, exc.message)
        raise

    final_args = resolution.vllm_args if vllm_args else apply_memory_and_context(resolution.vllm_args, memory, context)

    _print(stdout, f"Starting model '{name}' on pod '{pod_name}'...")
    _print(stdout, f"Model: {model_id}")
    _print(stdout, f"Port: {port}")
    gpu_str = ", ".join(str(g) for g in resolution.gpus) if resolution.gpus else "Managed by vLLM"
    _print(stdout, f"GPU(s): {gpu_str}")
    if resolution.config and resolution.config.notes:
        _print(stdout, f"Note: {resolution.config.notes}")
    _print(stdout)

    # Upload customised model_run.sh via a heredoc.
    script_content = render_model_run_script(model_id, name, port, final_args)
    upload_cmd = f"cat > /tmp/model_run_{name}.sh << 'EOF'\n{script_content}\nEOF\nchmod +x /tmp/model_run_{name}.sh"
    await ssh_mod.ssh_exec(pod_obj.ssh, upload_cmd)

    hf_token = os.environ.get("HF_TOKEN", "")
    nu_api_key = os.environ.get("NU_API_KEY") or os.environ.get("PI_API_KEY") or ""
    env_block = build_env_block(
        hf_token=hf_token,
        nu_api_key=nu_api_key,
        gpus=resolution.gpus,
        extra_env=resolution.config.env if resolution.config else None,
    )

    start_cmd = f"""
{env_block}
mkdir -p ~/.vllm_logs
cat > /tmp/model_wrapper_{name}.sh << 'WRAPPER'
#!/bin/bash
script -q -f -c "/tmp/model_run_{name}.sh" ~/.vllm_logs/{name}.log
exit_code=$?
echo "Script exited with code $exit_code" >> ~/.vllm_logs/{name}.log
exit $exit_code
WRAPPER
chmod +x /tmp/model_wrapper_{name}.sh
setsid /tmp/model_wrapper_{name}.sh </dev/null >/dev/null 2>&1 &
echo $!
exit 0
"""

    pid_result = await ssh_mod.ssh_exec(pod_obj.ssh, start_cmd)
    pid_str = pid_result.stdout.strip()
    try:
        pid = int(pid_str)
    except ValueError:
        _print(stderr, "Failed to start model runner")
        raise PodsError("model runner did not return a PID") from None

    config = load_config()
    if pod_name in config.pods:
        config.pods[pod_name].models[name] = Model(model=model_id, port=port, gpu=list(resolution.gpus), pid=pid)
        save_config(config)

    _print(stdout, f"Model runner started with PID: {pid}")
    _print(stdout, "Streaming logs... (waiting for startup)")
    _print(stdout)

    # Brief delay so the remote log file has a chance to materialise
    # before we open the tail. Mirrors the TS ``setTimeout(500)``.
    await asyncio.sleep(0.05)  # tighter in tests; production tail will catch up

    host = extract_host(pod_obj.ssh)
    tail_result = await ssh_mod.ssh_exec(pod_obj.ssh, f"tail -n 200 ~/.vllm_logs/{name}.log")
    interrupted = False
    startup_complete = False
    startup_failed = False
    failure_reason = ""

    if tail_result.exit_code == 0:
        for line in tail_result.stdout.split("\n"):
            if not line:
                continue
            _print(stdout, line)
            classification = classify_log_line(line)
            if classification.complete:
                startup_complete = True
                break
            if classification.failed:
                startup_failed = True
                failure_reason = classification.reason
                break
    else:
        interrupted = True

    if startup_failed:
        _print(stdout, f"\n✗ Model failed to start: {failure_reason}")
        config = load_config()
        if pod_name in config.pods and name in config.pods[pod_name].models:
            del config.pods[pod_name].models[name]
            save_config(config)
        _print(stdout, "\nModel has been removed from configuration.")
        if "OOM" in failure_reason or "memory" in failure_reason:
            _print(stdout, "\nSuggestions:")
            _print(stdout, "  • Try reducing GPU memory utilization: --memory 50%")
            _print(stdout, "  • Use a smaller context window: --context 4k")
            _print(stdout, "  • Use a quantized version of the model (e.g., FP8)")
            _print(stdout, "  • Use more GPUs with tensor parallelism")
            _print(stdout, "  • Try a smaller model variant")
        _print(stdout, f'\nCheck full logs: nu-pods ssh "tail -100 ~/.vllm_logs/{name}.log"')
        raise PodsError(f"model {name!r} failed to start: {failure_reason}")

    if startup_complete:
        _print(stdout, "\n✓ Model started successfully!")
        _print(stdout, "\nConnection Details:")
        _print(stdout, "─" * 50)
        _print(stdout, f"Base URL:    http://{host}:{port}/v1")
        _print(stdout, f"Model:       {model_id}")
        _print(stdout, f"API Key:     {nu_api_key or '(not set)'}")
        _print(stdout, "─" * 50)
        _print(stdout, "\nExport for shell:")
        _print(stdout, f'export OPENAI_BASE_URL="http://{host}:{port}/v1"')
        _print(stdout, f'export OPENAI_API_KEY="{nu_api_key or "your-api-key"}"')
        _print(stdout, f'export OPENAI_MODEL="{model_id}"')
        _print(stdout, "")
        _print(stdout, f"Monitor logs:     nu-pods logs {name}")
        _print(stdout, f"Stop model:       nu-pods stop {name}")
    elif interrupted:
        _print(stdout, "\n\nLog tail unavailable. Model deployment continues in background.")
        _print(stdout, f"Check status: nu-pods logs {name}")
        _print(stdout, f"Stop model: nu-pods stop {name}")
    else:
        _print(stdout, "\n\nLog stream ended. Model may still be running.")
        _print(stdout, f"Check status: nu-pods logs {name}")
        _print(stdout, f"Stop model: nu-pods stop {name}")


async def stop_model(name: str, *, pod: str | None = None, stdout: TextIO, stderr: TextIO) -> None:
    pod_name, pod_obj = get_pod(pod)

    model = pod_obj.models.get(name)
    if model is None:
        _print(stderr, f"Model '{name}' not found on pod '{pod_name}'")
        raise PodsError(f"unknown model {name!r}")

    _print(stdout, f"Stopping model '{name}' on pod '{pod_name}'...")
    kill_cmd = f"pkill -TERM -P {model.pid} 2>/dev/null || true; kill {model.pid} 2>/dev/null || true"
    await ssh_mod.ssh_exec(pod_obj.ssh, kill_cmd)

    config = load_config()
    if pod_name in config.pods and name in config.pods[pod_name].models:
        del config.pods[pod_name].models[name]
        save_config(config)

    _print(stdout, f"✓ Model '{name}' stopped")


async def stop_all_models(*, pod: str | None = None, stdout: TextIO, stderr: TextIO) -> None:
    pod_name, pod_obj = get_pod(pod)

    model_names = list(pod_obj.models.keys())
    if not model_names:
        _print(stdout, f"No models running on pod '{pod_name}'")
        return

    _print(stdout, f"Stopping {len(model_names)} model(s) on pod '{pod_name}'...")
    pids = " ".join(str(m.pid) for m in pod_obj.models.values())
    kill_cmd = f"for PID in {pids}; do pkill -TERM -P $PID 2>/dev/null || true; kill $PID 2>/dev/null || true; done"
    await ssh_mod.ssh_exec(pod_obj.ssh, kill_cmd)

    config = load_config()
    if pod_name in config.pods:
        config.pods[pod_name].models = {}
        save_config(config)

    _print(stdout, f"✓ Stopped all models: {', '.join(model_names)}")


async def list_models(*, pod: str | None = None, stdout: TextIO, stderr: TextIO) -> None:
    pod_name, pod_obj = get_pod(pod)

    model_names = list(pod_obj.models.keys())
    if not model_names:
        _print(stdout, f"No models running on pod '{pod_name}'")
        return

    host = extract_host(pod_obj.ssh)
    _print(stdout, f"Models on pod '{pod_name}':")
    for name in model_names:
        model = pod_obj.models[name]
        if len(model.gpu) > 1:
            gpu_str = f"GPUs {','.join(str(g) for g in model.gpu)}"
        elif len(model.gpu) == 1:
            gpu_str = f"GPU {model.gpu[0]}"
        else:
            gpu_str = "GPU unknown"
        _print(stdout, f"  {name} - Port {model.port} - {gpu_str} - PID {model.pid}")
        _print(stdout, f"    Model: {model.model}")
        _print(stdout, f"    URL: http://{host}:{model.port}/v1")

    _print(stdout)
    _print(stdout, "Verifying processes...")
    any_dead = False
    for name in model_names:
        model = pod_obj.models[name]
        check_cmd = (
            f"if ps -p {model.pid} > /dev/null 2>&1; then "
            f"if curl -s -f http://localhost:{model.port}/health > /dev/null 2>&1; then "
            f"echo running; "
            f"else "
            f"if tail -n 20 ~/.vllm_logs/{name}.log 2>/dev/null | grep -q 'ERROR\\|Failed\\|Cuda error\\|died'; then "
            f"echo crashed; else echo starting; fi; "
            f"fi; "
            f"else echo dead; fi"
        )
        result = await ssh_mod.ssh_exec(pod_obj.ssh, check_cmd)
        status = result.stdout.strip()
        if status == "dead":
            _print(stdout, f"  {name}: Process {model.pid} is not running")
            any_dead = True
        elif status == "crashed":
            _print(stdout, f"  {name}: vLLM crashed (check logs with 'nu-pods logs {name}')")
            any_dead = True
        elif status == "starting":
            _print(stdout, f"  {name}: Still starting up...")

    if any_dead:
        _print(stdout)
        _print(stdout, "Some models are not running. Clean up with:")
        _print(stdout, "  nu-pods stop <name>")
    else:
        _print(stdout, "✓ All processes verified")


async def view_logs(name: str, *, pod: str | None = None, stdout: TextIO, stderr: TextIO) -> None:
    pod_name, pod_obj = get_pod(pod)
    model = pod_obj.models.get(name)
    if model is None:
        _print(stderr, f"Model '{name}' not found on pod '{pod_name}'")
        raise PodsError(f"unknown model {name!r}")

    _print(stdout, f"Streaming logs for '{name}' on pod '{pod_name}'...")
    _print(stdout, "Press Ctrl+C to stop")
    _print(stdout)

    await ssh_mod.ssh_exec_stream(pod_obj.ssh, f"tail -f ~/.vllm_logs/{name}.log")


# ---------------------------------------------------------------------------
# show_known_models
# ---------------------------------------------------------------------------


def _models_json_path() -> str:
    return str(resources.files("nu_pods.resources").joinpath("models.json"))


def show_known_models(*, stdout: TextIO) -> None:
    with open(_models_json_path(), encoding="utf-8") as f:
        models_json = json.load(f)
    models = models_json.get("models", {})

    active = get_active_pod()
    pod_gpu_count = 0
    pod_gpu_type = ""
    if active is not None:
        pod_gpu_count = len(active.pod.gpus)
        if active.pod.gpus:
            pod_gpu_type = gpu_type_token(active.pod.gpus[0].name)
        _print(stdout, f"Known Models for {active.name} ({pod_gpu_count}x {pod_gpu_type or 'GPU'}):")
        _print(stdout)
    else:
        _print(stdout, "Known Models:")
        _print(stdout, "No active pod. Use 'nu-pods pods active <name>' to filter compatible models.")
        _print(stdout)

    _print(stdout, "Usage: nu-pods start <model> --name <name> [options]")
    _print(stdout)

    compatible: dict[str, list[dict[str, str]]] = OrderedDict()
    incompatible: dict[str, list[dict[str, str]]] = OrderedDict()

    for model_id, info in models.items():
        family = info.get("name", "Other").split("-")[0] or "Other"
        is_compatible = False
        compatible_config = ""
        min_gpu = "Unknown"
        notes: str | None = info.get("notes")

        configs = info.get("configs") or []
        if configs:
            sorted_configs = sorted(configs, key=lambda c: c.get("gpuCount", 1))
            min_config = sorted_configs[0]
            min_gpu_count = min_config.get("gpuCount", 1)
            gpu_types = "/".join(min_config.get("gpuTypes", []) or []) or "H100/H200"
            min_gpu = f"{min_gpu_count}x {gpu_types}" if min_gpu_count > 1 else f"1x {gpu_types}"
            notes = min_config.get("notes") or info.get("notes")

            if active is not None and pod_gpu_count > 0:
                for cfg in sorted_configs:
                    cfg_count = cfg.get("gpuCount", 1)
                    cfg_types = cfg.get("gpuTypes") or []
                    if cfg_count > pod_gpu_count:
                        continue
                    if cfg_types and not any(t in pod_gpu_type or pod_gpu_type in t for t in cfg_types):
                        continue
                    is_compatible = True
                    compatible_config = f"1x {pod_gpu_type}" if cfg_count == 1 else f"{cfg_count}x {pod_gpu_type}"
                    notes = cfg.get("notes") or info.get("notes")
                    break

        entry = {"id": model_id, "name": info.get("name", model_id), "notes": notes or ""}
        if active is not None and is_compatible:
            entry["config"] = compatible_config
            compatible.setdefault(family, []).append(entry)
        else:
            entry["min_gpu"] = min_gpu
            incompatible.setdefault(family, []).append(entry)

    if active is not None and compatible:
        _print(stdout, "✓ Compatible Models:")
        _print(stdout)
        for family in sorted(compatible.keys()):
            _print(stdout, f"{family} Models:")
            for entry in sorted(compatible[family], key=lambda e: e["name"]):
                _print(stdout, f"  {entry['id']}")
                _print(stdout, f"    Name: {entry['name']}")
                _print(stdout, f"    Config: {entry['config']}")
                if entry["notes"]:
                    _print(stdout, f"    Note: {entry['notes']}")
                _print(stdout)

    if incompatible:
        if active is not None and compatible:
            _print(stdout, "✗ Incompatible Models (need more/different GPUs):")
            _print(stdout)
        for family in sorted(incompatible.keys()):
            _print(stdout, f"{family} Models:")
            for entry in sorted(incompatible[family], key=lambda e: e["name"]):
                _print(stdout, f"  {entry['id']}")
                _print(stdout, f"    Name: {entry['name']}")
                _print(stdout, f"    Min Hardware: {entry['min_gpu']}")
                if entry["notes"] and active is None:
                    _print(stdout, f"    Note: {entry['notes']}")
                _print(stdout)

    _print(stdout, "For unknown models, defaults to single GPU deployment.")
    _print(stdout, "Use --vllm to pass custom arguments to vLLM.")
