"""Tests for ``nu_pods.ssh`` argv assembly + runner injection."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from nu_pods import ssh as ssh_mod
from nu_pods.ssh import SshResult, scp_file, scp_host_args, split_ssh, ssh_exec, ssh_exec_stream


@pytest.fixture
def captured_argv(reset_ssh_runners: None) -> list[list[str]]:
    captures: list[list[str]] = []

    async def fake_run(argv: Sequence[str]) -> SshResult:
        captures.append(list(argv))
        return SshResult(stdout="ok\n", stderr="", exit_code=0)

    async def fake_stream(argv: Sequence[str], silent: bool) -> int:
        captures.append([*argv, f"<silent={silent}>"])
        return 0

    async def fake_scp(argv: Sequence[str]) -> bool:
        captures.append(list(argv))
        return True

    ssh_mod.set_runners(run=fake_run, stream=fake_stream, scp=fake_scp)
    return captures


def test_split_ssh_basic() -> None:
    cmd, args = split_ssh("ssh root@host")
    assert cmd == "ssh"
    assert args == ["root@host"]


def test_split_ssh_with_port_and_options() -> None:
    cmd, args = split_ssh("ssh -p 2222 -i /path/to/key root@host")
    assert cmd == "ssh"
    assert args == ["-p", "2222", "-i", "/path/to/key", "root@host"]


def test_split_ssh_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        split_ssh("")


def test_scp_host_args_simple() -> None:
    host, extras = scp_host_args("ssh root@host")
    assert host == "root@host"
    assert extras == []


def test_scp_host_args_with_port() -> None:
    host, extras = scp_host_args("ssh -p 2222 root@host")
    assert host == "root@host"
    assert extras == ["-P", "2222"]


def test_scp_host_args_no_host_raises() -> None:
    with pytest.raises(ValueError, match="host"):
        scp_host_args("ssh -p 22")


async def test_ssh_exec_builds_argv(captured_argv: list[list[str]]) -> None:
    result = await ssh_exec("ssh root@host", "echo hi")
    assert result.exit_code == 0
    assert captured_argv[-1] == ["ssh", "root@host", "echo hi"]


async def test_ssh_exec_with_keepalive(captured_argv: list[list[str]]) -> None:
    await ssh_exec("ssh root@host", "uptime", keep_alive=True)
    argv = captured_argv[-1]
    assert "ServerAliveInterval=30" in argv
    assert "ServerAliveCountMax=120" in argv
    assert argv[-2] == "root@host"
    assert argv[-1] == "uptime"


async def test_ssh_exec_stream_force_tty(captured_argv: list[list[str]]) -> None:
    code = await ssh_exec_stream("ssh root@host", "bash setup.sh", force_tty=True)
    assert code == 0
    argv = captured_argv[-1]
    assert "-tt" in argv
    assert "<silent=False>" in argv


async def test_ssh_exec_stream_silent(captured_argv: list[list[str]]) -> None:
    await ssh_exec_stream("ssh root@host", "noisy", silent=True)
    assert "<silent=True>" in captured_argv[-1]


async def test_scp_file_uses_port_flag(captured_argv: list[list[str]]) -> None:
    ok = await scp_file("ssh -p 2222 root@host", "/local/file", "/remote/file")
    assert ok
    argv = captured_argv[-1]
    assert argv[0] == "scp"
    assert "-P" in argv
    assert "2222" in argv
    assert argv[-1] == "root@host:/remote/file"
    assert argv[-2] == "/local/file"


async def test_scp_file_returns_false_on_unparseable_ssh(reset_ssh_runners: None) -> None:
    assert await scp_file("", "/local", "/remote") is False
