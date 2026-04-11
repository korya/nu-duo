"""Session-cwd validation — direct port of ``packages/coding-agent/src/core/session-cwd.ts``.

When resuming a session, the stored cwd may no longer exist (deleted
worktree, moved directory, …). The helpers here let the CLI detect and
report that condition without having to know about the
:class:`SessionManager` internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class SessionCwdIssue:
    """Describes the mismatch between the stored session cwd and reality."""

    session_cwd: str
    fallback_cwd: str
    session_file: str | None = None


class SessionCwdSource(Protocol):
    """The two methods :func:`get_missing_session_cwd_issue` reads from a session manager."""

    def get_cwd(self) -> str: ...
    def get_session_file(self) -> str | None: ...


def get_missing_session_cwd_issue(
    session_manager: SessionCwdSource,
    fallback_cwd: str,
) -> SessionCwdIssue | None:
    """Return an issue object if the session's cwd is missing on disk, else ``None``."""
    session_file = session_manager.get_session_file()
    if not session_file:
        return None
    session_cwd = session_manager.get_cwd()
    if not session_cwd or Path(session_cwd).exists():
        return None
    return SessionCwdIssue(
        session_cwd=session_cwd,
        fallback_cwd=fallback_cwd,
        session_file=session_file,
    )


def format_missing_session_cwd_error(issue: SessionCwdIssue) -> str:
    """Long-form error suitable for ``print``-mode failure messages."""
    session_file = f"\nSession file: {issue.session_file}" if issue.session_file else ""
    return (
        f"Stored session working directory does not exist: {issue.session_cwd}{session_file}\n"
        f"Current working directory: {issue.fallback_cwd}"
    )


def format_missing_session_cwd_prompt(issue: SessionCwdIssue) -> str:
    """Short prompt body the interactive UI shows when offering a fallback."""
    return f"cwd from session file does not exist\n{issue.session_cwd}\n\ncontinue in current cwd\n{issue.fallback_cwd}"


class MissingSessionCwdError(Exception):
    """Raised by :func:`assert_session_cwd_exists` when the cwd is gone."""

    def __init__(self, issue: SessionCwdIssue) -> None:
        super().__init__(format_missing_session_cwd_error(issue))
        self.issue = issue


def assert_session_cwd_exists(session_manager: SessionCwdSource, fallback_cwd: str) -> None:
    """Raise :class:`MissingSessionCwdError` if the session's cwd is missing."""
    issue = get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue is not None:
        raise MissingSessionCwdError(issue)


__all__ = [
    "MissingSessionCwdError",
    "SessionCwdIssue",
    "SessionCwdSource",
    "assert_session_cwd_exists",
    "format_missing_session_cwd_error",
    "format_missing_session_cwd_prompt",
    "get_missing_session_cwd_issue",
]
