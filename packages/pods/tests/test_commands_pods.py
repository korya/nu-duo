"""Tests for ``nu_pods.commands.pods``."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from nu_pods import ssh as ssh_mod
from nu_pods.commands.pods import (
    list_pods,
    parse_gpu_csv,
    remove_pod_command,
    setup_pod,
    switch_active_pod,
)
from nu_pods.config import add_pod, load_config
from nu_pods.ssh import SshResult
from nu_pods.types import GPU, Pod, PodsError


def test_parse_gpu_csv_full_lines() -> None:
    csv = "0, NVIDIA H200, 141GB\n1, NVIDIA H200, 141GB"
    gpus = parse_gpu_csv(csv)
    assert gpus == [
        GPU(id=0, name="NVIDIA H200", memory="141GB"),
        GPU(id=1, name="NVIDIA H200", memory="141GB"),
    ]


def test_parse_gpu_csv_skips_garbage_and_blanks() -> None:
    csv = "\n0, NVIDIA H100, 80GB\nnot-a-row\n,,"
    gpus = parse_gpu_csv(csv)
    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA H100"


def test_parse_gpu_csv_fills_unknowns() -> None:
    gpus = parse_gpu_csv("0,,")
    assert gpus[0].name == "Unknown"
    assert gpus[0].memory == "Unknown"


def test_list_pods_empty(isolated_config: Path) -> None:
    out = io.StringIO()
    list_pods(stdout=out)
    assert "No pods configured" in out.getvalue()


def test_list_pods_renders_active_marker(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh root@a", gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")]))
    add_pod("b", Pod(ssh="ssh root@b", models_path="/mnt/sfs", vllm_version="gpt-oss"))
    out = io.StringIO()
    list_pods(stdout=out)
    text = out.getvalue()
    assert "* a" in text  # first added is active
    assert "  b" in text
    assert "1x NVIDIA H200" in text
    assert "no GPUs detected" in text
    assert "Models: /mnt/sfs" in text
    assert "GPT-OSS build" in text


def test_switch_active_pod_unknown(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError):
        switch_active_pod("ghost", stdout=out, stderr=err)
    assert "not found" in err.getvalue()


def test_switch_active_pod_success(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    add_pod("b", Pod(ssh="ssh b"))
    out, err = io.StringIO(), io.StringIO()
    switch_active_pod("b", stdout=out, stderr=err)
    assert "Switched active pod to 'b'" in out.getvalue()
    assert load_config().active == "b"


def test_remove_pod_command_unknown(isolated_config: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError):
        remove_pod_command("ghost", stdout=out, stderr=err)
    assert "not found" in err.getvalue()


def test_remove_pod_command_success(isolated_config: Path) -> None:
    add_pod("a", Pod(ssh="ssh a"))
    out, err = io.StringIO(), io.StringIO()
    remove_pod_command("a", stdout=out, stderr=err)
    assert "Removed pod 'a'" in out.getvalue()
    assert "a" not in load_config().pods


# ---------------------------------------------------------------------------
# setup_pod — heavily mocked SSH layer
# ---------------------------------------------------------------------------


class _SshScript:
    """Tiny scripted SSH runner for setup_pod end-to-end tests."""

    def __init__(self, *, gpu_csv: str = "0, NVIDIA H200, 141GB", connection_ok: bool = True) -> None:
        self.gpu_csv = gpu_csv
        self.connection_ok = connection_ok
        self.exec_calls: list[str] = []
        self.stream_calls: list[str] = []
        self.scp_calls: list[tuple[str, ...]] = []
        self.scp_ok = True
        self.stream_exit = 0

    async def run(self, argv: Sequence[str]) -> SshResult:
        cmd = argv[-1]
        self.exec_calls.append(cmd)
        if cmd == "echo 'SSH OK'":
            return SshResult(stdout="SSH OK\n", stderr="", exit_code=0 if self.connection_ok else 255)
        if "nvidia-smi" in cmd:
            return SshResult(stdout=self.gpu_csv, stderr="", exit_code=0)
        return SshResult(stdout="", stderr="", exit_code=0)

    async def stream(self, argv: Sequence[str], silent: bool) -> int:
        self.stream_calls.append(argv[-1])
        return self.stream_exit

    async def scp(self, argv: Sequence[str]) -> bool:
        self.scp_calls.append(tuple(argv))
        return self.scp_ok


@pytest.fixture
def ssh_script(reset_ssh_runners: None) -> _SshScript:
    script = _SshScript()
    ssh_mod.set_runners(run=script.run, stream=script.stream, scp=script.scp)
    return script


async def test_setup_pod_requires_hf_token(
    isolated_config: Path,
    clean_pod_env: None,
    ssh_script: _SshScript,
) -> None:
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="HF_TOKEN"):
        await setup_pod("p1", "ssh root@host", stdout=out, stderr=err)
    assert "HF_TOKEN" in err.getvalue()


async def test_setup_pod_requires_api_key(
    isolated_config: Path,
    clean_pod_env: None,
    ssh_script: _SshScript,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="NU_API_KEY"):
        await setup_pod("p1", "ssh root@host", stdout=out, stderr=err)


async def test_setup_pod_requires_models_path(
    isolated_config: Path,
    clean_pod_env: None,
    ssh_script: _SshScript,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("NU_API_KEY", "nu_xxx")
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="models-path"):
        await setup_pod("p1", "ssh root@host", stdout=out, stderr=err)


async def test_setup_pod_extracts_models_path_from_mount(
    isolated_config: Path,
    clean_pod_env: None,
    ssh_script: _SshScript,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("NU_API_KEY", "nu_xxx")
    out, err = io.StringIO(), io.StringIO()
    await setup_pod(
        "p1",
        "ssh root@host",
        mount="mount -t nfs server:/share /mnt/sfs",
        stdout=out,
        stderr=err,
    )
    pod = load_config().pods["p1"]
    assert pod.models_path == "/mnt/sfs"
    assert pod.vllm_version == "release"
    assert len(pod.gpus) == 1
    assert "✓ Pod 'p1' setup complete" in out.getvalue()


async def test_setup_pod_ssh_failure(
    isolated_config: Path,
    clean_pod_env: None,
    ssh_script: _SshScript,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("NU_API_KEY", "nu_xxx")
    ssh_script.connection_ok = False
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="SSH"):
        await setup_pod("p1", "ssh root@host", models_path="/mnt/sfs", stdout=out, stderr=err)


async def test_setup_pod_scp_failure(
    isolated_config: Path,
    clean_pod_env: None,
    ssh_script: _SshScript,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("NU_API_KEY", "nu_xxx")
    ssh_script.scp_ok = False
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="scp"):
        await setup_pod("p1", "ssh root@host", models_path="/mnt/sfs", stdout=out, stderr=err)


async def test_setup_pod_setup_script_failure(
    isolated_config: Path,
    clean_pod_env: None,
    ssh_script: _SshScript,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("NU_API_KEY", "nu_xxx")
    ssh_script.stream_exit = 1
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError, match="setup script"):
        await setup_pod("p1", "ssh root@host", models_path="/mnt/sfs", stdout=out, stderr=err)


async def test_setup_pod_explicit_gpt_oss_vllm(
    isolated_config: Path,
    clean_pod_env: None,
    ssh_script: _SshScript,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("NU_API_KEY", "nu_xxx")
    out, err = io.StringIO(), io.StringIO()
    await setup_pod(
        "p1",
        "ssh root@host",
        models_path="/mnt/sfs",
        vllm="gpt-oss",
        stdout=out,
        stderr=err,
    )
    assert "GPT-OSS special build" in out.getvalue()
    assert load_config().pods["p1"].vllm_version == "gpt-oss"
    # The setup-script invocation should carry the chosen vllm version.
    assert any("--vllm 'gpt-oss'" in cmd for cmd in ssh_script.stream_calls)
