"""Tests for ``nu_coding_agent.core.session_cwd``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from nu_coding_agent.core.session_cwd import (
    MissingSessionCwdError,
    SessionCwdIssue,
    assert_session_cwd_exists,
    format_missing_session_cwd_error,
    format_missing_session_cwd_prompt,
    get_missing_session_cwd_issue,
)


@dataclass
class FakeSessionManager:
    cwd: str
    session_file: str | None

    def get_cwd(self) -> str:
        return self.cwd

    def get_session_file(self) -> str | None:
        return self.session_file


def test_no_session_file_returns_none(tmp_path: Path) -> None:
    sm = FakeSessionManager(cwd=str(tmp_path / "missing"), session_file=None)
    assert get_missing_session_cwd_issue(sm, str(tmp_path)) is None


def test_existing_cwd_returns_none(tmp_path: Path) -> None:
    sm = FakeSessionManager(cwd=str(tmp_path), session_file="/sess.jsonl")
    assert get_missing_session_cwd_issue(sm, str(tmp_path)) is None


def test_missing_cwd_returns_issue(tmp_path: Path) -> None:
    missing = tmp_path / "ghost"
    sm = FakeSessionManager(cwd=str(missing), session_file="/sess.jsonl")
    issue = get_missing_session_cwd_issue(sm, str(tmp_path))
    assert issue is not None
    assert issue.session_cwd == str(missing)
    assert issue.session_file == "/sess.jsonl"
    assert issue.fallback_cwd == str(tmp_path)


def test_assert_raises_with_issue(tmp_path: Path) -> None:
    sm = FakeSessionManager(cwd=str(tmp_path / "ghost"), session_file="/x.jsonl")
    with pytest.raises(MissingSessionCwdError) as exc_info:
        assert_session_cwd_exists(sm, str(tmp_path))
    assert exc_info.value.issue.session_file == "/x.jsonl"


def test_assert_no_raise_when_ok(tmp_path: Path) -> None:
    sm = FakeSessionManager(cwd=str(tmp_path), session_file="/x.jsonl")
    assert_session_cwd_exists(sm, str(tmp_path))


def test_format_error_includes_paths() -> None:
    issue = SessionCwdIssue(session_cwd="/gone", fallback_cwd="/here", session_file="/s.jsonl")
    out = format_missing_session_cwd_error(issue)
    assert "/gone" in out
    assert "/here" in out
    assert "/s.jsonl" in out


def test_format_error_no_session_file() -> None:
    issue = SessionCwdIssue(session_cwd="/gone", fallback_cwd="/here")
    out = format_missing_session_cwd_error(issue)
    assert "Session file" not in out


def test_format_prompt_short() -> None:
    issue = SessionCwdIssue(session_cwd="/gone", fallback_cwd="/here")
    out = format_missing_session_cwd_prompt(issue)
    assert "/gone" in out
    assert "/here" in out
