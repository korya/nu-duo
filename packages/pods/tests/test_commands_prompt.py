"""Tests for ``nu_pods.commands.prompt``."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from nu_pods.commands.prompt import (
    PromptInvocation,
    build_invocation,
    prompt_model,
    reset_launcher,
    set_launcher,
)
from nu_pods.config import add_pod
from nu_pods.types import GPU, Model, Pod, PodsError


def _pod(models: dict[str, Model] | None = None) -> Pod:
    return Pod(
        ssh="ssh root@h.example",
        gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")],
        models=models or {},
        models_path="/mnt/sfs",
    )


def test_build_invocation_unknown_pod_raises(isolated_config: Path) -> None:
    with pytest.raises(PodsError, match="not found"):
        build_invocation("alpha", [], pod_override="ghost")


def test_build_invocation_no_active_pod_raises(isolated_config: Path) -> None:
    with pytest.raises(PodsError, match="No active pod"):
        build_invocation("alpha", [])


def test_build_invocation_unknown_model_raises(isolated_config: Path) -> None:
    add_pod("p1", _pod())
    with pytest.raises(PodsError, match="not found"):
        build_invocation("ghost", [])


def test_build_invocation_completions_api(isolated_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod(
        "p1",
        _pod(
            models={
                "alpha": Model(model="Qwen/Qwen3-Coder-30B-A3B-Instruct", port=8001, gpu=[0], pid=11),
            }
        ),
    )
    inv = build_invocation("alpha", ["--print", "hi"], cwd="/work")
    assert inv.base_url == "http://h.example:8001/v1"
    assert inv.model_id == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert inv.api_key == "nu_x"
    assert "--api" in inv.args
    assert inv.args[inv.args.index("--api") + 1] == "completions"
    assert "--print" in inv.args
    assert "Current working directory: /work" in inv.system_prompt


def test_build_invocation_responses_api_for_gpt_oss(isolated_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NU_API_KEY", raising=False)
    monkeypatch.delenv("PI_API_KEY", raising=False)
    add_pod(
        "p1",
        _pod(
            models={
                "g": Model(model="openai/gpt-oss-20b", port=8002, gpu=[0], pid=12),
            }
        ),
    )
    inv = build_invocation("g", [], api_key_override="explicit-key")
    assert inv.api_key == "explicit-key"
    assert inv.args[inv.args.index("--api") + 1] == "responses"


def test_build_invocation_falls_back_to_pi_api_key(isolated_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NU_API_KEY", raising=False)
    monkeypatch.setenv("PI_API_KEY", "legacy")
    add_pod("p1", _pod(models={"a": Model(model="m", port=8001, gpu=[0], pid=1)}))
    inv = build_invocation("a", [])
    assert inv.api_key == "legacy"


def test_build_invocation_dummy_when_no_keys(isolated_config: Path, clean_pod_env: None) -> None:
    add_pod("p1", _pod(models={"a": Model(model="m", port=8001, gpu=[0], pid=1)}))
    inv = build_invocation("a", [])
    assert inv.api_key == "dummy"


async def test_prompt_model_invokes_launcher(
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_prompt_launcher: None,
) -> None:
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(models={"a": Model(model="m", port=8001, gpu=[0], pid=1)}))

    captured: list[PromptInvocation] = []

    async def fake(invocation: PromptInvocation) -> int:
        captured.append(invocation)
        return 7

    set_launcher(fake)
    out, err = io.StringIO(), io.StringIO()
    rc = await prompt_model("a", ["--print", "hello"], stdout=out, stderr=err)
    assert rc == 7
    assert captured and captured[0].model_id == "m"


async def test_prompt_model_default_launcher_dispatches_to_nu_cli(
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_prompt_launcher: None,
) -> None:
    """The default launcher hands the invocation argv off to nu_coding_agent."""
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod("p1", _pod(models={"a": Model(model="m", port=8001, gpu=[0], pid=1)}))
    reset_launcher()

    # Patch the lazy import target so we can capture the argv without
    # spinning up a real Agent (no LLM endpoint, no SessionManager I/O).
    captured_argv: list[list[str]] = []

    async def fake_async_main(argv: list[str]) -> int:
        captured_argv.append(list(argv))
        return 42

    import nu_coding_agent.cli as nu_cli

    monkeypatch.setattr(nu_cli, "async_main", fake_async_main)

    out, err = io.StringIO(), io.StringIO()
    rc = await prompt_model("a", ["--print", "hi"], stdout=out, stderr=err)
    assert rc == 42

    assert len(captured_argv) == 1
    argv = captured_argv[0]
    assert argv[0] == "nu"
    # The invocation argv should contain the routing flags + the user prompt.
    assert "--base-url" in argv
    assert "--model" in argv
    assert "--api-key" in argv
    assert "--api" in argv
    assert "--system-prompt" in argv
    assert "--print" in argv
    assert "hi" in argv


async def test_prompt_model_unknown_model_writes_stderr(
    isolated_config: Path,
    reset_prompt_launcher: None,
) -> None:
    add_pod("p1", _pod())
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(PodsError):
        await prompt_model("ghost", [], stdout=out, stderr=err)
    assert "not found" in err.getvalue()
