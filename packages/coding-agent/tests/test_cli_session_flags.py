"""Tests for the session-related flags wired into ``nu_coding_agent.cli``.

The CLI module is excluded from coverage as a thin glue layer, but the
session-flag plumbing is the entire point of the recent slice — it's
what unlocks ``--continue`` / ``--session FILE`` against a real
:class:`SessionManager`. These tests pin the parser + SessionManager
selection so a regression here is loud.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from nu_ai.types import Model, ModelCost
from nu_coding_agent.cli import (
    _Args,  # pyright: ignore[reportPrivateUsage]
    _build_auth_storage,  # pyright: ignore[reportPrivateUsage]
    _build_session_manager,  # pyright: ignore[reportPrivateUsage]
    _parse_args,  # pyright: ignore[reportPrivateUsage]
    _resolve_model,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    import pytest


def _ok(argv: list[str]) -> _Args:
    args, exit_code = _parse_args(["nu", *argv])
    assert exit_code is None, f"parser returned exit code {exit_code} for {argv!r}"
    return args


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def test_continue_short_and_long_flag() -> None:
    assert _ok(["-c", "hi"]).continue_session is True
    assert _ok(["--continue", "hi"]).continue_session is True


def test_session_file_flag_consumes_value() -> None:
    args = _ok(["--session", "/tmp/sess.jsonl", "hi"])
    assert args.session_file == "/tmp/sess.jsonl"
    assert args.positional == ["hi"]


def test_session_file_flag_missing_value_errors() -> None:
    _, exit_code = _parse_args(["nu", "--session"])
    assert exit_code == 2


def test_ephemeral_flag() -> None:
    assert _ok(["--ephemeral", "hi"]).ephemeral is True


def test_default_flags_all_off() -> None:
    args = _ok(["hi"])
    assert args.continue_session is False
    assert args.session_file is None
    assert args.ephemeral is False


def test_combined_session_flags_parse_independently() -> None:
    # The flags don't enforce mutual exclusion at the parser layer —
    # _build_session_manager imposes a documented precedence order
    # (ephemeral > session > continue).
    args = _ok(["--ephemeral", "--continue", "--session", "/tmp/x.jsonl", "hi"])
    assert args.ephemeral is True
    assert args.continue_session is True
    assert args.session_file == "/tmp/x.jsonl"


# ---------------------------------------------------------------------------
# _build_session_manager — picks the right SessionManager flavour
# ---------------------------------------------------------------------------


def _faux_args(**overrides: object) -> _Args:
    args = _Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_build_session_manager_default_persists(tmp_path: Path) -> None:
    sm = _build_session_manager(_faux_args(), str(tmp_path))
    assert sm.is_persisted() is True
    assert sm.get_cwd() == str(tmp_path)


def test_build_session_manager_ephemeral_in_memory(tmp_path: Path) -> None:
    sm = _build_session_manager(_faux_args(ephemeral=True), str(tmp_path))
    assert sm.is_persisted() is False
    assert sm.get_session_file() is None


def test_build_session_manager_ephemeral_beats_session_and_continue(tmp_path: Path) -> None:
    sm = _build_session_manager(
        _faux_args(ephemeral=True, continue_session=True, session_file="/anything.jsonl"),
        str(tmp_path),
    )
    assert sm.is_persisted() is False


def test_build_session_manager_session_file_loads_jsonl(tmp_path: Path) -> None:
    fixture = tmp_path / "sess.jsonl"
    fixture.write_text(
        '{"type":"session","version":3,"id":"sess-x","timestamp":"2024-01-01T00:00:00.000Z","cwd":"/orig"}\n'
        '{"type":"message","id":"e1","parentId":null,"timestamp":"2024-01-01T00:00:01.000Z",'
        '"message":{"role":"user","content":"hi","timestamp":1}}\n',
        encoding="utf-8",
    )
    sm = _build_session_manager(
        _faux_args(session_file=str(fixture)),
        str(tmp_path),
    )
    assert sm.is_persisted() is True
    # cwd_override propagates from the CLI argument.
    assert sm.get_cwd() == str(tmp_path)
    # The pre-existing entry is loaded so a follow-up turn can resume it.
    assert any(e.get("id") == "e1" for e in sm.get_entries())


def test_build_session_manager_continue_recent_creates_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--continue`` with no prior session falls back to a fresh one."""
    monkeypatch.setenv("NU_AGENT_DIR", str(tmp_path / "agent"))
    sm = _build_session_manager(_faux_args(continue_session=True), str(tmp_path))
    assert sm.is_persisted() is True
    assert sm.get_entries() == []


def test_build_session_manager_continue_recent_resumes_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NU_AGENT_DIR", str(tmp_path / "agent"))
    cwd = str(tmp_path / "work")
    Path(cwd).mkdir()

    # First run: create a persisted session and append one assistant turn
    # so the file is actually flushed to disk.
    sm1 = _build_session_manager(_faux_args(), cwd)
    sm1.append_message({"role": "user", "content": "hi", "timestamp": 1})
    sm1.append_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "provider": "openai",
            "model": "m",
            "api": "openai-completions",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "stop",
            "timestamp": 2,
        }
    )
    first_file = sm1.get_session_file()
    assert first_file is not None
    assert Path(first_file).exists()

    # Second run with --continue should pick the same file up.
    sm2 = _build_session_manager(_faux_args(continue_session=True), cwd)
    assert sm2.get_session_file() == first_file
    ids = [e.get("id") for e in sm2.get_entries()]
    assert len(ids) == 2  # user + assistant


# ---------------------------------------------------------------------------
# _build_auth_storage — runtime override path
# ---------------------------------------------------------------------------


def _model() -> Model:
    return Model(
        id="m",
        name="m",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=1000,
        max_tokens=100,
    )


def test_build_auth_storage_runtime_override_seeds_provider() -> None:
    storage = _build_auth_storage(_faux_args(api_key="explicit-key"), _model())
    assert storage.has_auth("openai") is True


def test_build_auth_storage_no_api_key_no_seeded_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    storage = _build_auth_storage(_faux_args(), _model())
    # Without --api-key, has_auth depends on env vars (and we just removed
    # the relevant one), so the storage is genuinely empty.
    assert storage.has_auth("openai") is False


# ---------------------------------------------------------------------------
# --base-url / --api parsing + custom model construction (the path used by
# `nu-pods agent` to route at a deployed vLLM endpoint)
# ---------------------------------------------------------------------------


def test_parse_base_url_and_api() -> None:
    args = _ok(
        [
            "--base-url",
            "http://h:8001/v1",
            "--api",
            "openai-completions",
            "--model",
            "Qwen/Qwen3-Coder-30B",
            "--api-key",
            "k",
            "hi",
        ]
    )
    assert args.base_url == "http://h:8001/v1"
    assert args.api == "openai-completions"
    assert args.model == "Qwen/Qwen3-Coder-30B"
    assert args.positional == ["hi"]


def test_base_url_missing_value_errors() -> None:
    _, exit_code = _parse_args(["nu", "--base-url"])
    assert exit_code == 2


def test_api_flag_missing_value_errors() -> None:
    _, exit_code = _parse_args(["nu", "--api"])
    assert exit_code == 2


def test_resolve_model_with_base_url_builds_custom_model() -> None:
    args = _faux_args(
        base_url="http://pod.example:8001/v1",
        api="openai-completions",
        model="Qwen/Qwen3-Coder-30B",
    )
    model = _resolve_model(args)
    assert model is not None
    assert model.base_url == "http://pod.example:8001/v1"
    assert model.api == "openai-completions"
    assert model.id == "Qwen/Qwen3-Coder-30B"
    assert model.provider == "openai"


def test_resolve_model_with_base_url_no_model_returns_none() -> None:
    args = _faux_args(base_url="http://pod.example:8001/v1")
    assert _resolve_model(args) is None


def test_resolve_model_with_base_url_defaults_to_openai_completions_api() -> None:
    args = _faux_args(base_url="http://pod.example:8001/v1", model="m")
    model = _resolve_model(args)
    assert model is not None
    assert model.api == "openai-completions"


# ---------------------------------------------------------------------------
# Session JSONL produced by the CLI is byte-shape compatible with TS.
# This is the round-trip cousin of the in-source tests in
# tests/test_session_manager.py — running it through the CLI helpers
# (build_session_manager → SessionManager) catches any wiring drift.
# ---------------------------------------------------------------------------


def test_cli_built_session_manager_writes_camelcase_jsonl(tmp_path: Path) -> None:
    sm = _build_session_manager(_faux_args(), str(tmp_path / "work"))
    sm.append_message({"role": "user", "content": "hi", "timestamp": 1})
    sm.append_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "ack"}],
            "provider": "openai",
            "model": "m",
            "api": "openai-completions",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "stop",
            "timestamp": 2,
        }
    )
    file = sm.get_session_file()
    assert file is not None
    text = Path(file).read_text(encoding="utf-8")
    assert '"parentId"' in text
    assert '"parent_id"' not in text
    # Header is the first line and JSON-parsable.
    header = json.loads(text.splitlines()[0])
    assert header["type"] == "session"
