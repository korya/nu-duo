"""Tests for ``nu_pods.commands.models`` (helpers + start/stop/list/show)."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from nu_pods import ssh as ssh_mod
from nu_pods.commands.models import (
    apply_memory_and_context,
    build_env_block,
    classify_log_line,
    extract_host,
    get_pod,
    list_models,
    next_port,
    parse_context_size,
    parse_memory_fraction,
    render_model_run_script,
    resolve_model_deployment,
    select_gpus,
    show_known_models,
    start_model,
    stop_all_models,
    stop_model,
    view_logs,
)
from nu_pods.config import add_pod, load_config
from nu_pods.ssh import SshResult
from nu_pods.types import GPU, Model, Pod, PodsError

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _pod(*, gpus: int = 2, models: dict[str, Model] | None = None) -> Pod:
    return Pod(
        ssh="ssh root@host",
        gpus=[GPU(id=i, name="NVIDIA H200", memory="141GB") for i in range(gpus)],
        models=models or {},
        models_path="/mnt/sfs",
        vllm_version="release",
    )


def test_next_port_default() -> None:
    assert next_port(_pod()) == 8001


def test_next_port_skips_used() -> None:
    pod = _pod(
        models={
            "a": Model(model="m", port=8001, gpu=[0], pid=1),
            "b": Model(model="m", port=8002, gpu=[1], pid=2),
        }
    )
    assert next_port(pod) == 8003


def test_select_gpus_full_count_returns_all() -> None:
    pod = _pod(gpus=3)
    assert select_gpus(pod, 3) == [0, 1, 2]


def test_select_gpus_least_used_first() -> None:
    pod = _pod(
        gpus=4,
        models={
            "a": Model(model="m", port=8001, gpu=[0, 1], pid=1),
            "b": Model(model="m", port=8002, gpu=[0], pid=2),
        },
    )
    chosen = select_gpus(pod, 2)
    # GPUs 2 and 3 have zero usage, so they should win.
    assert set(chosen) == {2, 3}


def test_parse_context_size_named() -> None:
    assert parse_context_size("4k") == 4096
    assert parse_context_size("128K") == 131072
    assert parse_context_size("16384") == 16384


def test_parse_memory_fraction() -> None:
    assert parse_memory_fraction("50%") == pytest.approx(0.5)
    assert parse_memory_fraction("0.9") == pytest.approx(0.009)


def test_apply_memory_and_context_replaces_existing() -> None:
    # NOTE: matches the TS implementation, which strips flag *names*
    # by substring match only — orphaned values would survive. The
    # bundled configs in models.json never pre-set these flags so it
    # never bites in practice; we keep the behaviour byte-faithful.
    args = ["--keep-me", "--gpu-memory-utilization", "--max-model-len"]
    out = apply_memory_and_context(args, "50%", "8k")
    assert "--keep-me" in out
    assert out.count("--gpu-memory-utilization") == 1
    assert "0.5" in out
    assert out.count("--max-model-len") == 1
    assert "8192" in out


def test_apply_memory_and_context_no_overrides() -> None:
    args = ["--enable-auto-tool-choice"]
    assert apply_memory_and_context(args, None, None) == args


def test_build_env_block_single_gpu() -> None:
    block = build_env_block(hf_token="hf", nu_api_key="nu", gpus=[3])
    assert "CUDA_VISIBLE_DEVICES=3" in block
    assert "export HF_TOKEN='hf'" in block
    assert "export PI_API_KEY='nu'" in block


def test_build_env_block_multi_gpu_no_cuda_var() -> None:
    block = build_env_block(hf_token="hf", nu_api_key="nu", gpus=[0, 1])
    assert "CUDA_VISIBLE_DEVICES" not in block


def test_build_env_block_extra_env() -> None:
    block = build_env_block(hf_token="hf", nu_api_key="nu", gpus=[0], extra_env={"VLLM_USE_DEEP_GEMM": "1"})
    assert "VLLM_USE_DEEP_GEMM='1'" in block


def test_render_model_run_script_substitutes_placeholders() -> None:
    script = render_model_run_script("org/model", "alpha", 8001, ["--foo", "bar"])
    assert "{{MODEL_ID}}" not in script
    assert "{{NAME}}" not in script
    assert "{{PORT}}" not in script
    assert "{{VLLM_ARGS}}" not in script
    assert "org/model" in script
    assert 'NAME="alpha"' in script
    assert 'PORT="8001"' in script
    assert "--foo bar" in script


def test_classify_log_line_complete() -> None:
    result = classify_log_line("INFO Application startup complete.")
    assert result.complete is True
    assert result.failed is False


def test_classify_log_line_oom() -> None:
    result = classify_log_line("torch.OutOfMemoryError: CUDA out of memory")
    assert result.failed is True
    assert "OOM" in result.reason


def test_classify_log_line_runner_failed() -> None:
    result = classify_log_line("Model runner exiting with code 1")
    assert result.failed is True


def test_classify_log_line_runner_clean_exit_ignored() -> None:
    result = classify_log_line("Model runner exiting with code 0")
    assert result.failed is False
    assert result.complete is False


def test_classify_log_line_engine_failure() -> None:
    result = classify_log_line("RuntimeError: Engine core initialization failed: foo")
    assert result.failed is True


def test_classify_log_line_script_failure() -> None:
    result = classify_log_line("Script exited with code 137")
    assert result.failed is True


def test_classify_log_line_neutral() -> None:
    result = classify_log_line("just some chatter")
    assert result.complete is False
    assert result.failed is False


def test_extract_host() -> None:
    assert extract_host("ssh root@86.38.238.55") == "86.38.238.55"
    assert extract_host("ssh -p 22 root@h.example") == "h.example"
    assert extract_host("ssh nohostatall") == "localhost"


# ---------------------------------------------------------------------------
# resolve_model_deployment
# ---------------------------------------------------------------------------


def test_resolve_custom_vllm_args_short_circuits() -> None:
    pod = _pod()
    res = resolve_model_deployment(pod, "anything/at/all", custom_vllm_args=["--foo"], requested_gpus=None)
    assert res.gpus == []
    assert res.vllm_args == ["--foo"]
    assert res.config is None


def test_resolve_known_model_default_picks_largest_fit() -> None:
    pod = _pod(gpus=2)
    res = resolve_model_deployment(pod, "Qwen/Qwen3-Coder-30B-A3B-Instruct", custom_vllm_args=None, requested_gpus=None)
    assert res.config is not None
    assert res.config.gpu_count == 2
    assert "--tensor-parallel-size" in res.vllm_args
    assert len(res.gpus) == 2


def test_resolve_known_model_explicit_gpu_count() -> None:
    pod = _pod(gpus=2)
    res = resolve_model_deployment(pod, "Qwen/Qwen3-Coder-30B-A3B-Instruct", custom_vllm_args=None, requested_gpus=1)
    assert res.config is not None
    assert res.config.gpu_count == 1
    assert len(res.gpus) == 1


def test_resolve_known_model_too_many_gpus_raises() -> None:
    pod = _pod(gpus=1)
    with pytest.raises(PodsError, match="only has 1"):
        resolve_model_deployment(pod, "Qwen/Qwen3-Coder-30B-A3B-Instruct", custom_vllm_args=None, requested_gpus=4)


def test_resolve_known_model_no_matching_config_raises() -> None:
    pod = _pod(gpus=4)
    # Qwen3-Coder-30B-FP8 only has a 1-GPU config.
    with pytest.raises(PodsError, match="does not have a configuration"):
        resolve_model_deployment(pod, "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", custom_vllm_args=None, requested_gpus=2)


def test_resolve_known_model_no_compatible_config_raises() -> None:
    a100_pod = Pod(ssh="ssh a", gpus=[GPU(id=0, name="NVIDIA A100", memory="80GB")], models_path="/m")
    with pytest.raises(PodsError, match="not compatible"):
        resolve_model_deployment(a100_pod, "openai/gpt-oss-20b", custom_vllm_args=None, requested_gpus=None)


def test_resolve_unknown_model_default_single_gpu() -> None:
    pod = _pod(gpus=2)
    res = resolve_model_deployment(pod, "ghost/model", custom_vllm_args=None, requested_gpus=None)
    assert res.config is None
    assert len(res.gpus) == 1


def test_resolve_unknown_model_with_gpus_flag_raises() -> None:
    pod = _pod()
    with pytest.raises(PodsError, match="--gpus can only be used"):
        resolve_model_deployment(pod, "ghost/model", custom_vllm_args=None, requested_gpus=2)


# ---------------------------------------------------------------------------
# get_pod
# ---------------------------------------------------------------------------


def test_get_pod_no_active_raises(isolated_config: Path) -> None:
    with pytest.raises(PodsError, match="No active pod"):
        get_pod()


def test_get_pod_override_unknown(isolated_config: Path) -> None:
    with pytest.raises(PodsError, match="not found"):
        get_pod("ghost")


def test_get_pod_override_resolves(isolated_config: Path) -> None:
    add_pod("a", _pod())
    name, pod = get_pod("a")
    assert name == "a"
    assert pod.ssh == "ssh root@host"


def test_get_pod_active_resolves(isolated_config: Path) -> None:
    add_pod("a", _pod())
    name, _ = get_pod()
    assert name == "a"


# ---------------------------------------------------------------------------
# Command flows — heavily mocked SSH
# ---------------------------------------------------------------------------


class _ScriptedSsh:
    def __init__(self) -> None:
        self.responses: dict[str, SshResult] = {}
        self.default = SshResult(stdout="", stderr="", exit_code=0)
        self.calls: list[str] = []

    def queue(self, fragment: str, *, stdout: str = "", exit_code: int = 0) -> None:
        self.responses[fragment] = SshResult(stdout=stdout, stderr="", exit_code=exit_code)

    async def run(self, argv: Sequence[str]) -> SshResult:
        cmd = argv[-1]
        self.calls.append(cmd)
        for fragment, result in self.responses.items():
            if fragment in cmd:
                return result
        return self.default

    async def stream(self, argv: Sequence[str], silent: bool) -> int:
        self.calls.append(argv[-1])
        return 0

    async def scp(self, argv: Sequence[str]) -> bool:
        return True


@pytest.fixture
def scripted_ssh(reset_ssh_runners: None) -> _ScriptedSsh:
    s = _ScriptedSsh()
    ssh_mod.set_runners(run=s.run, stream=s.stream, scp=s.scp)
    return s


async def test_start_model_success(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(gpus=1))

    # First call: heredoc upload (no PID needed). Second call: setsid → returns PID.
    # Third call: tail of logs → contains startup-complete marker.
    scripted_ssh.queue("setsid", stdout="12345\n")
    scripted_ssh.queue("tail -n 200", stdout="INFO Application startup complete.\n")

    out, err = io.StringIO(), io.StringIO()
    await start_model("Qwen/Qwen3-Coder-30B-A3B-Instruct", "alpha", stdout=out, stderr=err)

    text = out.getvalue()
    assert "Model started successfully" in text
    assert "Base URL:    http://host:8001/v1" in text
    pod = load_config().pods["p1"]
    assert "alpha" in pod.models
    assert pod.models["alpha"].pid == 12345


async def test_start_model_no_models_path(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", Pod(ssh="ssh root@host", gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")]))
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="models path"):
        await start_model("Qwen/Qwen3-Coder-30B-A3B-Instruct", "alpha", stdout=out, stderr=err)


async def test_start_model_already_exists(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", _pod(models={"alpha": Model(model="x", port=8001, gpu=[0], pid=1)}))
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="already exists"):
        await start_model("Qwen/Qwen3-Coder-30B-A3B-Instruct", "alpha", stdout=out, stderr=err)


async def test_start_model_failure_cleans_config(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(gpus=1))
    scripted_ssh.queue("setsid", stdout="9999\n")
    scripted_ssh.queue(
        "tail -n 200",
        stdout="loading...\ntorch.OutOfMemoryError: CUDA out of memory.\n",
    )

    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="failed to start"):
        await start_model("Qwen/Qwen3-Coder-30B-A3B-Instruct", "alpha", stdout=out, stderr=err)

    assert "alpha" not in load_config().pods["p1"].models
    text = out.getvalue()
    assert "Out of GPU memory" in text
    assert "Suggestions:" in text


async def test_start_model_runner_did_not_return_pid(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(gpus=1))
    scripted_ssh.queue("setsid", stdout="not-a-number\n")
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError):
        await start_model("Qwen/Qwen3-Coder-30B-A3B-Instruct", "alpha", stdout=out, stderr=err)
    assert "Failed to start" in err.getvalue()


async def test_start_model_resolution_error_writes_stderr(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(gpus=1))
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="--gpus can only"):
        await start_model("ghost/model", "alpha", gpus=2, stdout=out, stderr=err)
    assert "--gpus can only" in err.getvalue()


async def test_start_model_log_stream_unavailable_branch(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(gpus=1))
    scripted_ssh.queue("setsid", stdout="555\n")
    scripted_ssh.queue("tail -n 200", exit_code=1)
    out, err = io.StringIO(), io.StringIO()
    await start_model("Qwen/Qwen3-Coder-30B-A3B-Instruct", "alpha", stdout=out, stderr=err)
    assert "Log tail unavailable" in out.getvalue()


async def test_start_model_log_stream_ended_branch(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(gpus=1))
    scripted_ssh.queue("setsid", stdout="555\n")
    scripted_ssh.queue("tail -n 200", stdout="boring line\nanother boring line\n")
    out, err = io.StringIO(), io.StringIO()
    await start_model("Qwen/Qwen3-Coder-30B-A3B-Instruct", "alpha", stdout=out, stderr=err)
    assert "Log stream ended" in out.getvalue()


async def test_start_model_with_custom_vllm_args(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(gpus=1))
    scripted_ssh.queue("setsid", stdout="42\n")
    scripted_ssh.queue("tail -n 200", stdout="INFO Application startup complete.\n")
    out, err = io.StringIO(), io.StringIO()
    await start_model(
        "ghost/model",
        "alpha",
        vllm_args=["--my-flag", "yes"],
        stdout=out,
        stderr=err,
    )
    # Heredoc upload should mention our custom flag.
    assert any("--my-flag yes" in c for c in scripted_ssh.calls)
    assert "Managed by vLLM" in out.getvalue()


async def test_start_model_with_memory_and_context_overrides(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(gpus=1))
    scripted_ssh.queue("setsid", stdout="42\n")
    scripted_ssh.queue("tail -n 200", stdout="INFO Application startup complete.\n")
    out, err = io.StringIO(), io.StringIO()
    await start_model(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "alpha",
        memory="50%",
        context="8k",
        stdout=out,
        stderr=err,
    )
    upload = next(c for c in scripted_ssh.calls if "model_run_alpha" in c and "{{" not in c)
    assert "--gpu-memory-utilization" in upload
    assert "0.5" in upload
    assert "--max-model-len" in upload
    assert "8192" in upload


async def test_stop_model_unknown(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", _pod())
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError):
        await stop_model("ghost", stdout=out, stderr=err)


async def test_stop_model_success(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", _pod(models={"alpha": Model(model="m", port=8001, gpu=[0], pid=1234)}))
    out, err = io.StringIO(), io.StringIO()
    await stop_model("alpha", stdout=out, stderr=err)
    assert "Model 'alpha' stopped" in out.getvalue()
    assert "alpha" not in load_config().pods["p1"].models
    assert any("kill 1234" in c for c in scripted_ssh.calls)


async def test_stop_all_models_empty(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", _pod())
    out, err = io.StringIO(), io.StringIO()
    await stop_all_models(stdout=out, stderr=err)
    assert "No models running" in out.getvalue()


async def test_stop_all_models_clears_them(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod(
        "p1",
        _pod(
            models={
                "a": Model(model="m", port=8001, gpu=[0], pid=11),
                "b": Model(model="m", port=8002, gpu=[1], pid=22),
            }
        ),
    )
    out, err = io.StringIO(), io.StringIO()
    await stop_all_models(stdout=out, stderr=err)
    assert load_config().pods["p1"].models == {}
    text = out.getvalue()
    assert "Stopped all models" in text


async def test_list_models_empty(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", _pod())
    out, err = io.StringIO(), io.StringIO()
    await list_models(stdout=out, stderr=err)
    assert "No models running" in out.getvalue()


async def test_list_models_running_status(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod(
        "p1",
        _pod(
            models={
                "a": Model(model="m", port=8001, gpu=[0, 1], pid=11),
                "b": Model(model="m", port=8002, gpu=[2], pid=22),
                "c": Model(model="m", port=8003, gpu=[], pid=33),
            }
        ),
    )
    scripted_ssh.queue("ps -p", stdout="running\n")
    out, err = io.StringIO(), io.StringIO()
    await list_models(stdout=out, stderr=err)
    text = out.getvalue()
    assert "GPUs 0,1" in text
    assert "GPU 2" in text
    assert "GPU unknown" in text
    assert "All processes verified" in text


async def test_list_models_dead_status(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", _pod(models={"a": Model(model="m", port=8001, gpu=[0], pid=11)}))
    scripted_ssh.default = SshResult(stdout="dead\n", stderr="", exit_code=0)
    out, err = io.StringIO(), io.StringIO()
    await list_models(stdout=out, stderr=err)
    text = out.getvalue()
    assert "is not running" in text
    assert "nu-pods stop <name>" in text


async def test_list_models_crashed_and_starting(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod(
        "p1",
        _pod(
            models={
                "a": Model(model="m", port=8001, gpu=[0], pid=11),
                "b": Model(model="m", port=8002, gpu=[0], pid=12),
            }
        ),
    )

    # Return different statuses based on which port is being checked.
    async def run(argv: Sequence[str]) -> SshResult:
        cmd = argv[-1]
        if "8001" in cmd:
            return SshResult(stdout="crashed\n", stderr="", exit_code=0)
        if "8002" in cmd:
            return SshResult(stdout="starting\n", stderr="", exit_code=0)
        return SshResult(stdout="", stderr="", exit_code=0)

    ssh_mod.set_runners(run=run)
    out, err = io.StringIO(), io.StringIO()
    await list_models(stdout=out, stderr=err)
    text = out.getvalue()
    assert "vLLM crashed" in text
    assert "Still starting up" in text


async def test_view_logs_unknown(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", _pod())
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError):
        await view_logs("ghost", stdout=out, stderr=err)


async def test_view_logs_success(
    isolated_config: Path,
    scripted_ssh: _ScriptedSsh,
) -> None:
    add_pod("p1", _pod(models={"a": Model(model="m", port=8001, gpu=[0], pid=11)}))
    out, err = io.StringIO(), io.StringIO()
    await view_logs("a", stdout=out, stderr=err)
    assert "Streaming logs for 'a'" in out.getvalue()
    assert any("tail -f ~/.vllm_logs/a.log" in c for c in scripted_ssh.calls)


# ---------------------------------------------------------------------------
# show_known_models
# ---------------------------------------------------------------------------


def test_show_known_models_no_active_pod(isolated_config: Path) -> None:
    out = io.StringIO()
    show_known_models(stdout=out)
    text = out.getvalue()
    assert "No active pod" in text
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in text


def test_show_known_models_with_compatible_active_pod(isolated_config: Path) -> None:
    add_pod("p1", _pod(gpus=2))
    out = io.StringIO()
    show_known_models(stdout=out)
    text = out.getvalue()
    assert "Compatible Models" in text
    assert "2x H200" in text or "1x H200" in text
    assert "Incompatible Models" in text  # GLM-4.5 needs 8 GPUs
