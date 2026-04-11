"""Tests for ``nu_pods.cli`` argv dispatch.

The CLI module is excluded from coverage (thin glue), but a handful
of round-trip tests catch parser regressions and verify that the
documented subcommand grammar still wires through to the command
layer.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from nu_pods import ssh as ssh_mod
from nu_pods.cli import dispatch
from nu_pods.commands import prompt as prompt_mod
from nu_pods.commands.prompt import PromptInvocation
from nu_pods.config import add_pod, load_config
from nu_pods.ssh import SshResult
from nu_pods.types import GPU, Model, Pod


@pytest.fixture
def stub_ssh(reset_ssh_runners: None) -> list[str]:
    calls: list[str] = []

    async def fake_run(argv: Sequence[str]) -> SshResult:
        calls.append(argv[-1])
        return SshResult(stdout="", stderr="", exit_code=0)

    async def fake_stream(argv: Sequence[str], silent: bool) -> int:
        calls.append(argv[-1])
        return 0

    async def fake_scp(argv: Sequence[str]) -> bool:
        return True

    ssh_mod.set_runners(run=fake_run, stream=fake_stream, scp=fake_scp)
    return calls


def _pod() -> Pod:
    return Pod(
        ssh="ssh root@host",
        gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")],
        models_path="/mnt/sfs",
    )


async def test_main_help() -> None:
    out, err = io.StringIO(), io.StringIO()
    rc = await dispatch(["--help"], out, err)
    assert rc == 0
    assert "nu-pods" in out.getvalue()


async def test_main_unknown_subcommand_raises() -> None:
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(Exception, match="Unknown pods subcommand"):
        await dispatch(["pods", "wat"], out, err)


async def test_dispatch_pods_list_empty(isolated_config: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    rc = await dispatch(["pods", "list"], out, err)
    assert rc == 0
    assert "No pods configured" in out.getvalue()


async def test_dispatch_pods_active_and_remove(isolated_config: Path) -> None:
    add_pod("a", _pod())
    add_pod("b", _pod())
    out, err = io.StringIO(), io.StringIO()
    await dispatch(["pods", "active", "b"], out, err)
    assert load_config().active == "b"
    out2, err2 = io.StringIO(), io.StringIO()
    await dispatch(["pods", "remove", "a"], out2, err2)
    assert "a" not in load_config().pods


async def test_dispatch_models_command(isolated_config: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    rc = await dispatch(["models"], out, err)
    assert rc == 0
    assert "Known Models" in out.getvalue()


async def test_dispatch_list_with_pod_flag(isolated_config: Path, stub_ssh: list[str]) -> None:
    add_pod("p1", _pod())
    out, err = io.StringIO(), io.StringIO()
    rc = await dispatch(["list", "--pod", "p1"], out, err)
    assert rc == 0
    assert "No models running" in out.getvalue()


async def test_dispatch_logs_subcommand(isolated_config: Path, stub_ssh: list[str]) -> None:
    add_pod(
        "p1",
        Pod(
            ssh="ssh root@host",
            gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")],
            models={"a": Model(model="m", port=8001, gpu=[0], pid=1)},
            models_path="/mnt/sfs",
        ),
    )
    out, err = io.StringIO(), io.StringIO()
    rc = await dispatch(["logs", "a", "--pod", "p1"], out, err)
    assert rc == 0
    assert any("tail -f" in c for c in stub_ssh)


async def test_dispatch_stop_with_name(isolated_config: Path, stub_ssh: list[str]) -> None:
    add_pod(
        "p1",
        Pod(
            ssh="ssh root@host",
            gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")],
            models={"a": Model(model="m", port=8001, gpu=[0], pid=99)},
            models_path="/mnt/sfs",
        ),
    )
    out, err = io.StringIO(), io.StringIO()
    rc = await dispatch(["stop", "a"], out, err)
    assert rc == 0
    assert "a" not in load_config().pods["p1"].models


async def test_dispatch_stop_all(isolated_config: Path, stub_ssh: list[str]) -> None:
    add_pod(
        "p1",
        Pod(
            ssh="ssh root@host",
            gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")],
            models={"a": Model(model="m", port=8001, gpu=[0], pid=99)},
            models_path="/mnt/sfs",
        ),
    )
    out, err = io.StringIO(), io.StringIO()
    rc = await dispatch(["stop", "--all", "--pod", "p1"], out, err)
    assert rc == 0
    assert load_config().pods["p1"].models == {}


async def test_dispatch_agent_subcommand(
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_prompt_launcher: None,
) -> None:
    monkeypatch.setenv("NU_API_KEY", "nu_x")
    add_pod(
        "p1",
        Pod(
            ssh="ssh root@host",
            gpus=[GPU(id=0, name="NVIDIA H200", memory="141GB")],
            models={"alpha": Model(model="m", port=8001, gpu=[0], pid=1)},
            models_path="/mnt/sfs",
        ),
    )

    captured: list[PromptInvocation] = []

    async def fake(invocation: PromptInvocation) -> int:
        captured.append(invocation)
        return 5

    prompt_mod.set_launcher(fake)
    out, err = io.StringIO(), io.StringIO()
    rc = await dispatch(["agent", "alpha", "--pod", "p1", "--", "--print", "hi"], out, err)
    assert rc == 5
    assert captured and "--print" in captured[0].args
