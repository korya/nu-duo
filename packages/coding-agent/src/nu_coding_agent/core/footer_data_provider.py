"""Footer data provider — port of ``packages/coding-agent/src/core/footer-data-provider.ts``.

Provides git branch name and extension status tracking for the UI footer.
Branch detection uses file-based HEAD parsing (with git CLI fallback for
reftable repos), and a file-system watcher for live updates.  Extension
statuses are a simple string map managed by the extension runtime.

File watching strategy
~~~~~~~~~~~~~~~~~~~~~~
The upstream TS version uses ``fs.watch`` / ``fs.watchFile``.  We mirror
that with :mod:`os` + a background :class:`threading.Thread` that polls
HEAD's directory every 250 ms (cheap ``os.scandir`` call).  If the HEAD
file's mtime or size changes we schedule an async branch refresh with a
500 ms debounce — same as upstream's ``WATCH_DEBOUNCE_MS``.

No new dependencies are introduced.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Git path resolution
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _GitPaths:
    """Resolved git metadata paths (mirrors the TS ``GitPaths`` type)."""

    repo_dir: str
    common_git_dir: str
    head_path: str


def _find_git_paths(cwd: str) -> _GitPaths | None:
    """Walk up from *cwd* to find ``.git``, handling worktrees."""
    directory = cwd
    while True:
        git_path = os.path.join(directory, ".git")
        if os.path.exists(git_path):
            try:
                if os.path.isfile(git_path):
                    content = Path(git_path).read_text(encoding="utf-8").strip()
                    if content.startswith("gitdir: "):
                        git_dir = os.path.normpath(
                            os.path.join(directory, content[len("gitdir: ") :].strip()),
                        )
                        head_path = os.path.join(git_dir, "HEAD")
                        if not os.path.exists(head_path):
                            return None
                        commondir_path = os.path.join(git_dir, "commondir")
                        if os.path.exists(commondir_path):
                            rel = Path(commondir_path).read_text(encoding="utf-8").strip()
                            common_git_dir = os.path.normpath(os.path.join(git_dir, rel))
                        else:
                            common_git_dir = git_dir
                        return _GitPaths(
                            repo_dir=directory,
                            common_git_dir=common_git_dir,
                            head_path=head_path,
                        )
                elif os.path.isdir(git_path):
                    head_path = os.path.join(git_path, "HEAD")
                    if not os.path.exists(head_path):
                        return None
                    return _GitPaths(
                        repo_dir=directory,
                        common_git_dir=git_path,
                        head_path=head_path,
                    )
            except OSError:
                return None
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


# ---------------------------------------------------------------------------
# Git branch helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _resolve_branch_with_git_sync(repo_dir: str) -> str | None:
    """Ask ``git symbolic-ref`` for the current branch (synchronous)."""
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        branch = result.stdout.strip() if result.returncode == 0 else ""
        return branch or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _read_branch_from_head(git_paths: _GitPaths) -> str | None:
    """Parse ``HEAD`` to extract the branch name (or ``"detached"``)."""
    try:
        content = Path(git_paths.head_path).read_text(encoding="utf-8").strip()
        if content.startswith("ref: refs/heads/"):
            branch = content[len("ref: refs/heads/") :]
            if branch == ".invalid":
                # Reftable repos write ".invalid" — fall back to git CLI.
                return _resolve_branch_with_git_sync(git_paths.repo_dir) or "detached"
            return branch
        return "detached"
    except OSError:
        return None


# ---------------------------------------------------------------------------
# File watcher thread
# ---------------------------------------------------------------------------


class _HeadWatcher:
    """Polls the git HEAD directory (+ optional reftable dir) for changes.

    When a change is detected, it calls *on_change* (from the watcher
    thread).  The caller is responsible for debouncing.
    """

    _POLL_INTERVAL = 0.25  # seconds

    def __init__(self, git_paths: _GitPaths, on_change: Callable[[], None]) -> None:
        self._on_change = on_change
        self._stop_event = threading.Event()

        # Directories / files to watch.
        self._head_dir = os.path.dirname(git_paths.head_path)
        reftable_dir = os.path.join(git_paths.common_git_dir, "reftable")
        self._reftable_dir: str | None = reftable_dir if os.path.isdir(reftable_dir) else None
        tables_list = os.path.join(reftable_dir, "tables.list") if self._reftable_dir else ""
        self._tables_list_path: str | None = tables_list if tables_list and os.path.exists(tables_list) else None

        # Initial snapshots.
        self._head_snapshot = self._stat_dir(self._head_dir)
        self._reftable_snapshot = self._stat_dir(self._reftable_dir) if self._reftable_dir else None
        self._tables_list_snapshot = self._stat_file(self._tables_list_path) if self._tables_list_path else None

        self._thread = threading.Thread(target=self._run, daemon=True, name="footer-git-watcher")
        self._thread.start()

    # -- snapshot helpers ---------------------------------------------------

    @staticmethod
    def _stat_file(path: str | None) -> tuple[float, float, int] | None:
        if not path:
            return None
        try:
            st = os.stat(path)
            return (st.st_mtime, st.st_ctime, st.st_size)
        except OSError:
            return None

    @staticmethod
    def _stat_dir(path: str | None) -> tuple[float, float, int] | None:
        """Return a cheap fingerprint for the directory's HEAD file."""
        if not path:
            return None
        head = os.path.join(path, "HEAD") if not path.endswith("HEAD") else path
        try:
            st = os.stat(head)
            return (st.st_mtime, st.st_ctime, st.st_size)
        except OSError:
            return None

    # -- poll loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._POLL_INTERVAL)
            if self._stop_event.is_set():
                break
            try:
                changed = False

                # Check HEAD directory.
                snap = self._stat_dir(self._head_dir)
                if snap != self._head_snapshot:
                    self._head_snapshot = snap
                    changed = True

                # Check reftable directory.
                if self._reftable_dir:
                    snap = self._stat_dir(self._reftable_dir)
                    if snap != self._reftable_snapshot:
                        self._reftable_snapshot = snap
                        changed = True

                # Check tables.list file.
                if self._tables_list_path:
                    snap = self._stat_file(self._tables_list_path)
                    if snap != self._tables_list_snapshot:
                        self._tables_list_snapshot = snap
                        changed = True

                if changed:
                    self._on_change()
            except Exception:
                log.debug("git watcher poll error", exc_info=True)

    def stop(self) -> None:
        self._stop_event.set()
        # Don't join — it's a daemon thread and we don't want to block.


# ---------------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------------


class FooterDataProvider:
    """Provides git branch and extension status for the UI footer.

    Port of the TypeScript ``FooterDataProvider`` (339 LOC).
    """

    _WATCH_DEBOUNCE_MS: float = 0.5  # seconds (500 ms)

    def __init__(self, cwd: str | None = None) -> None:
        if cwd is None:
            cwd = os.getcwd()
        self._cwd = cwd
        self._extension_statuses: dict[str, str] = {}
        self._cached_branch: str | None | object = _SENTINEL  # _SENTINEL ≙ "not yet resolved"
        self._git_paths: _GitPaths | None | object = _SENTINEL
        self._watcher: _HeadWatcher | None = None
        self._branch_change_callbacks: set[Callable[[str | None], None]] = set()
        self._available_provider_count: int = 0
        self._disposed = False

        # Debounce state.
        self._refresh_timer: threading.Timer | None = None
        self._refresh_lock = threading.Lock()

        # Eagerly resolve git paths and set up watcher.
        self._git_paths = _find_git_paths(cwd)
        self._setup_watcher()

    # -- public API ---------------------------------------------------------

    def get_git_branch(self) -> str | None:
        """Return current git branch name, ``"detached"``, or ``None``."""
        if self._cached_branch is _SENTINEL:
            self._cached_branch = self._resolve_branch_sync()
        return self._cached_branch  # type: ignore[return-value]

    def get_extension_statuses(self) -> dict[str, str]:
        """Return a *copy* of extension path -> status text mapping."""
        return dict(self._extension_statuses)

    def set_extension_status(self, key: str, status: str) -> None:
        """Set (or update) an extension status entry."""
        self._extension_statuses[key] = status

    def remove_extension_status(self, key: str) -> None:
        """Remove an extension status entry (no-op if missing)."""
        self._extension_statuses.pop(key, None)

    def clear_extension_statuses(self) -> None:
        """Remove all extension statuses."""
        self._extension_statuses.clear()

    def get_available_provider_count(self) -> int:
        """Number of unique providers with available models (for footer display)."""
        return self._available_provider_count

    def set_available_provider_count(self, count: int) -> None:
        self._available_provider_count = count

    def on_branch_change(self, callback: Callable[[str | None], None]) -> Callable[[], None]:
        """Subscribe to branch changes.  Returns an unsubscribe function."""
        self._branch_change_callbacks.add(callback)

        def _unsubscribe() -> None:
            self._branch_change_callbacks.discard(callback)

        return _unsubscribe

    def set_cwd(self, cwd: str) -> None:
        """Change working directory and re-detect git branch."""
        if self._cwd == cwd:
            return
        self._cwd = cwd
        self._cancel_refresh_timer()
        self._teardown_watcher()
        self._cached_branch = _SENTINEL
        self._git_paths = _find_git_paths(cwd)
        self._setup_watcher()
        self._notify_branch_change()

    def dispose(self) -> None:
        """Clean up watchers and timers."""
        self._disposed = True
        self._cancel_refresh_timer()
        self._teardown_watcher()
        self._branch_change_callbacks.clear()

    # -- private helpers ----------------------------------------------------

    def _resolve_branch_sync(self) -> str | None:
        git_paths = self._git_paths
        if not isinstance(git_paths, _GitPaths):
            return None
        return _read_branch_from_head(git_paths)

    def _notify_branch_change(self) -> None:
        branch = self.get_git_branch()
        for cb in list(self._branch_change_callbacks):
            try:
                cb(branch)
            except Exception:
                log.debug("branch change callback error", exc_info=True)

    def _schedule_refresh(self) -> None:
        if self._disposed:
            return
        with self._refresh_lock:
            if self._refresh_timer is not None:
                return  # already scheduled
            self._refresh_timer = threading.Timer(
                self._WATCH_DEBOUNCE_MS,
                self._do_refresh,
            )
            self._refresh_timer.daemon = True
            self._refresh_timer.start()

    def _do_refresh(self) -> None:
        with self._refresh_lock:
            self._refresh_timer = None
        if self._disposed:
            return
        old_branch = self._cached_branch
        self._cached_branch = _SENTINEL  # force re-read
        new_branch = self.get_git_branch()
        if old_branch is not _SENTINEL and old_branch != new_branch:
            self._notify_branch_change()

    def _cancel_refresh_timer(self) -> None:
        with self._refresh_lock:
            if self._refresh_timer is not None:
                self._refresh_timer.cancel()
                self._refresh_timer = None

    def _setup_watcher(self) -> None:
        git_paths = self._git_paths
        if not isinstance(git_paths, _GitPaths):
            return
        try:
            self._watcher = _HeadWatcher(git_paths, on_change=self._schedule_refresh)
        except Exception:
            log.debug("failed to set up git watcher", exc_info=True)

    def _teardown_watcher(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None


# ---------------------------------------------------------------------------
# Read-only protocol (mirrors the TS ``ReadonlyFooterDataProvider`` type)
# ---------------------------------------------------------------------------


class ReadonlyFooterDataProvider:
    """Read-only view for extensions — excludes mutation methods and dispose."""

    def __init__(self, provider: FooterDataProvider) -> None:
        self._provider = provider

    def get_git_branch(self) -> str | None:
        return self._provider.get_git_branch()

    def get_extension_statuses(self) -> dict[str, str]:
        return self._provider.get_extension_statuses()

    def get_available_provider_count(self) -> int:
        return self._provider.get_available_provider_count()

    def on_branch_change(self, callback: Callable[[str | None], None]) -> Callable[[], None]:
        return self._provider.on_branch_change(callback)


__all__ = [
    "FooterDataProvider",
    "ReadonlyFooterDataProvider",
]
