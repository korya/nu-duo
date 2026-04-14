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


# ---------------------------------------------------------------------------
# set_runners partial override
# ---------------------------------------------------------------------------


def test_set_runners_partial_run_only(reset_ssh_runners: None) -> None:
    """Override only `run`, leaving stream and scp as defaults."""

    async def custom_run(argv: Sequence[str]) -> SshResult:
        return SshResult(stdout="custom", stderr="", exit_code=42)

    prev_run, prev_stream, prev_scp = ssh_mod.set_runners(run=custom_run)
    # prev values are the defaults
    assert prev_run is not None
    assert prev_stream is not None
    assert prev_scp is not None
    # Only run was overridden
    assert ssh_mod._state.run is custom_run
    assert ssh_mod._state.stream is prev_stream
    assert ssh_mod._state.scp is prev_scp


def test_set_runners_partial_stream_only(reset_ssh_runners: None) -> None:
    """Override only `stream`."""

    async def custom_stream(argv: Sequence[str], silent: bool) -> int:
        return 99

    prev_run_before = ssh_mod._state.run
    ssh_mod.set_runners(stream=custom_stream)
    assert ssh_mod._state.stream is custom_stream
    assert ssh_mod._state.run is prev_run_before  # unchanged


def test_set_runners_partial_scp_only(reset_ssh_runners: None) -> None:
    """Override only `scp`."""

    async def custom_scp(argv: Sequence[str]) -> bool:
        return False

    prev_run_before = ssh_mod._state.run
    ssh_mod.set_runners(scp=custom_scp)
    assert ssh_mod._state.scp is custom_scp
    assert ssh_mod._state.run is prev_run_before


# ---------------------------------------------------------------------------
# ssh_exec_stream with keep_alive + force_tty combined
# ---------------------------------------------------------------------------


async def test_ssh_exec_stream_keep_alive_and_force_tty(captured_argv: list[list[str]]) -> None:
    code = await ssh_exec_stream(
        "ssh root@host", "cmd", keep_alive=True, force_tty=True,
    )
    assert code == 0
    argv = captured_argv[-1]
    assert "ServerAliveInterval=30" in argv
    assert "-tt" in argv
    # -tt should appear after keepalive opts but before host
    tt_idx = argv.index("-tt")
    host_idx = argv.index("root@host")
    assert tt_idx < host_idx


# ---------------------------------------------------------------------------
# scp_host_args with extra tokens after host (line 145)
# ---------------------------------------------------------------------------


def test_scp_host_args_extra_tokens_after_host() -> None:
    """Tokens after the host that don't contain '@' are dropped (line 145)."""
    host, extras = scp_host_args("ssh root@host -v -X")
    assert host == "root@host"
    # -v and -X are not -p so they're dropped
    assert extras == []


def test_scp_host_args_identity_file_not_passed() -> None:
    """Identity file flag is not forwarded to scp (only -p is)."""
    host, extras = scp_host_args("ssh -i /path/to/key root@host")
    assert host == "root@host"
    # -i is treated as a regular arg and dropped
    assert extras == []


async def test_scp_file_with_identity_and_port(captured_argv: list[list[str]]) -> None:
    """Identity file does not appear in scp argv; port does."""
    ok = await scp_file("ssh -p 2222 -i /key root@host", "/l", "/r")
    assert ok
    argv = captured_argv[-1]
    assert argv[0] == "scp"
    assert "-P" in argv
    assert "2222" in argv
    # Identity file is not forwarded
    assert "-i" not in argv
    assert "/key" not in argv


# ---------------------------------------------------------------------------
# Default runner functions (lines 45-78)
# ---------------------------------------------------------------------------


class TestDefaultRunners:
    """Cover _default_run, _default_stream, _default_scp by calling them
    against a real subprocess (``echo`` / ``true`` / ``false``).
    """

    async def test_default_run_success(self, reset_ssh_runners: None) -> None:
        result = await ssh_mod._default_run(["echo", "hello"])
        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_default_run_failure(self, reset_ssh_runners: None) -> None:
        result = await ssh_mod._default_run(["false"])
        assert result.exit_code != 0

    async def test_default_stream_silent(self, reset_ssh_runners: None) -> None:
        code = await ssh_mod._default_stream(["true"], silent=True)
        assert code == 0

    async def test_default_stream_not_silent(self, reset_ssh_runners: None) -> None:
        code = await ssh_mod._default_stream(["true"], silent=False)
        assert code == 0

    async def test_default_stream_failure(self, reset_ssh_runners: None) -> None:
        code = await ssh_mod._default_stream(["false"], silent=True)
        assert code != 0

    async def test_default_scp_success(self, reset_ssh_runners: None) -> None:
        ok = await ssh_mod._default_scp(["true"])
        assert ok is True

    async def test_default_scp_failure(self, reset_ssh_runners: None) -> None:
        ok = await ssh_mod._default_scp(["false"])
        assert ok is False
