"""Tests for ``nu_coding_agent.core.footer_data_provider``."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nu_coding_agent.core.footer_data_provider import (
    FooterDataProvider,
    ReadonlyFooterDataProvider,
    _GitPaths,
    _HeadWatcher,
    _find_git_paths,
    _read_branch_from_head,
    _resolve_branch_with_git_sync,
)


# ---------------------------------------------------------------------------
# _find_git_paths
# ---------------------------------------------------------------------------


class TestFindGitPaths:
    def test_regular_git_dir(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        result = _find_git_paths(str(tmp_path))
        assert result is not None
        assert result.repo_dir == str(tmp_path)
        assert result.common_git_dir == str(git_dir)
        assert result.head_path == str(git_dir / "HEAD")

    def test_worktree_git_file(self, tmp_path: Path) -> None:
        # Simulate a worktree: .git is a file pointing to a gitdir
        main_git = tmp_path / "main_repo" / ".git"
        main_git.mkdir(parents=True)
        worktree_gitdir = main_git / "worktrees" / "wt1"
        worktree_gitdir.mkdir(parents=True)
        (worktree_gitdir / "HEAD").write_text("ref: refs/heads/feature\n")
        (worktree_gitdir / "commondir").write_text("../..\n")

        wt_dir = tmp_path / "worktree"
        wt_dir.mkdir()
        (wt_dir / ".git").write_text(f"gitdir: {worktree_gitdir}\n")

        result = _find_git_paths(str(wt_dir))
        assert result is not None
        assert result.repo_dir == str(wt_dir)
        assert result.head_path == str(worktree_gitdir / "HEAD")
        assert result.common_git_dir == str(main_git)

    def test_worktree_no_commondir(self, tmp_path: Path) -> None:
        gitdir = tmp_path / "gitdir"
        gitdir.mkdir()
        (gitdir / "HEAD").write_text("ref: refs/heads/main\n")

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").write_text(f"gitdir: {gitdir}\n")

        result = _find_git_paths(str(repo))
        assert result is not None
        assert result.common_git_dir == str(gitdir)

    def test_no_git_dir(self, tmp_path: Path) -> None:
        result = _find_git_paths(str(tmp_path))
        assert result is None

    def test_git_dir_missing_head(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        result = _find_git_paths(str(tmp_path))
        assert result is None

    def test_worktree_missing_head(self, tmp_path: Path) -> None:
        gitdir = tmp_path / "gitdir"
        gitdir.mkdir()
        # No HEAD file in gitdir

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").write_text(f"gitdir: {gitdir}\n")

        result = _find_git_paths(str(repo))
        assert result is None

    def test_walks_up_to_parent(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        subdir = tmp_path / "a" / "b" / "c"
        subdir.mkdir(parents=True)

        result = _find_git_paths(str(subdir))
        assert result is not None
        assert result.repo_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# _read_branch_from_head
# ---------------------------------------------------------------------------


class TestReadBranchFromHead:
    def test_normal_branch(self, tmp_path: Path) -> None:
        head = tmp_path / "HEAD"
        head.write_text("ref: refs/heads/feature-xyz\n")
        gp = _GitPaths(repo_dir=str(tmp_path), common_git_dir=str(tmp_path), head_path=str(head))
        assert _read_branch_from_head(gp) == "feature-xyz"

    def test_detached_head(self, tmp_path: Path) -> None:
        head = tmp_path / "HEAD"
        head.write_text("abc123def456\n")
        gp = _GitPaths(repo_dir=str(tmp_path), common_git_dir=str(tmp_path), head_path=str(head))
        assert _read_branch_from_head(gp) == "detached"

    @patch("nu_coding_agent.core.footer_data_provider._resolve_branch_with_git_sync", return_value="main")
    def test_invalid_reftable(self, mock_git: MagicMock, tmp_path: Path) -> None:
        head = tmp_path / "HEAD"
        head.write_text("ref: refs/heads/.invalid\n")
        gp = _GitPaths(repo_dir=str(tmp_path), common_git_dir=str(tmp_path), head_path=str(head))
        assert _read_branch_from_head(gp) == "main"
        mock_git.assert_called_once_with(str(tmp_path))

    @patch("nu_coding_agent.core.footer_data_provider._resolve_branch_with_git_sync", return_value=None)
    def test_invalid_reftable_fallback_detached(self, mock_git: MagicMock, tmp_path: Path) -> None:
        head = tmp_path / "HEAD"
        head.write_text("ref: refs/heads/.invalid\n")
        gp = _GitPaths(repo_dir=str(tmp_path), common_git_dir=str(tmp_path), head_path=str(head))
        assert _read_branch_from_head(gp) == "detached"

    def test_missing_head_file(self, tmp_path: Path) -> None:
        gp = _GitPaths(
            repo_dir=str(tmp_path),
            common_git_dir=str(tmp_path),
            head_path=str(tmp_path / "nonexistent"),
        )
        assert _read_branch_from_head(gp) is None


# ---------------------------------------------------------------------------
# _resolve_branch_with_git_sync
# ---------------------------------------------------------------------------


class TestResolveBranchWithGitSync:
    @patch("nu_coding_agent.core.footer_data_provider.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="main\n")
        assert _resolve_branch_with_git_sync("/some/repo") == "main"

    @patch("nu_coding_agent.core.footer_data_provider.subprocess.run")
    def test_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert _resolve_branch_with_git_sync("/some/repo") is None

    @patch("nu_coding_agent.core.footer_data_provider.subprocess.run", side_effect=OSError("no git"))
    def test_oserror(self, mock_run: MagicMock) -> None:
        assert _resolve_branch_with_git_sync("/some/repo") is None

    @patch(
        "nu_coding_agent.core.footer_data_provider.subprocess.run",
        side_effect=__import__("subprocess").TimeoutExpired(cmd="git", timeout=5),
    )
    def test_timeout(self, mock_run: MagicMock) -> None:
        assert _resolve_branch_with_git_sync("/some/repo") is None


# ---------------------------------------------------------------------------
# FooterDataProvider
# ---------------------------------------------------------------------------


class TestFooterDataProvider:
    def test_get_git_branch(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/develop\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            assert provider.get_git_branch() == "develop"
            # Second call uses cache
            assert provider.get_git_branch() == "develop"
        finally:
            provider.dispose()

    def test_get_git_branch_no_repo(self, tmp_path: Path) -> None:
        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            assert provider.get_git_branch() is None
        finally:
            provider.dispose()

    def test_extension_statuses(self) -> None:
        provider = FooterDataProvider.__new__(FooterDataProvider)
        provider._extension_statuses = {}
        provider._cached_branch = None
        provider._git_paths = None
        provider._watcher = None
        provider._branch_change_callbacks = set()
        provider._available_provider_count = 0
        provider._disposed = False
        provider._refresh_timer = None
        provider._refresh_lock = __import__("threading").Lock()

        provider.set_extension_status("ext1", "loading")
        provider.set_extension_status("ext2", "ready")
        statuses = provider.get_extension_statuses()
        assert statuses == {"ext1": "loading", "ext2": "ready"}
        # Must be a copy
        statuses["ext3"] = "new"
        assert "ext3" not in provider.get_extension_statuses()

        provider.remove_extension_status("ext1")
        assert "ext1" not in provider.get_extension_statuses()

        provider.clear_extension_statuses()
        assert provider.get_extension_statuses() == {}

    def test_provider_count(self, tmp_path: Path) -> None:
        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            assert provider.get_available_provider_count() == 0
            provider.set_available_provider_count(3)
            assert provider.get_available_provider_count() == 3
        finally:
            provider.dispose()

    def test_branch_change_callback(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            received: list[str | None] = []
            unsub = provider.on_branch_change(lambda b: received.append(b))

            # Trigger set_cwd to same dir (should be no-op)
            provider.set_cwd(str(tmp_path))
            assert received == []

            # Change cwd to a subdir (still under same .git) triggers notification
            # but branch is still "main" since git repo is found by walking up
            new_dir = tmp_path / "sub"
            new_dir.mkdir()
            provider.set_cwd(str(new_dir))
            assert received == ["main"]

            # Unsubscribe
            unsub()
            provider.set_cwd(str(tmp_path))
            assert len(received) == 1  # no new callback
        finally:
            provider.dispose()

    def test_set_cwd(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            assert provider.get_git_branch() == "main"

            other = tmp_path / "other"
            other.mkdir()
            other_git = other / ".git"
            other_git.mkdir()
            (other_git / "HEAD").write_text("ref: refs/heads/feature\n")

            provider.set_cwd(str(other))
            assert provider.get_git_branch() == "feature"
        finally:
            provider.dispose()

    def test_dispose(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        provider.dispose()
        assert provider._disposed is True
        assert provider._watcher is None
        assert len(provider._branch_change_callbacks) == 0


# ---------------------------------------------------------------------------
# ReadonlyFooterDataProvider
# ---------------------------------------------------------------------------


class TestReadonlyFooterDataProvider:
    def test_delegates(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            provider.set_extension_status("ext1", "ok")
            provider.set_available_provider_count(2)

            ro = ReadonlyFooterDataProvider(provider)
            assert ro.get_git_branch() == "main"
            assert ro.get_extension_statuses() == {"ext1": "ok"}
            assert ro.get_available_provider_count() == 2

            received: list[str | None] = []
            unsub = ro.on_branch_change(lambda b: received.append(b))
            unsub()  # just verify it returns a callable
        finally:
            provider.dispose()


# ---------------------------------------------------------------------------
# _HeadWatcher
# ---------------------------------------------------------------------------


class TestHeadWatcher:
    def test_stop(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        gp = _GitPaths(
            repo_dir=str(tmp_path),
            common_git_dir=str(git_dir),
            head_path=str(git_dir / "HEAD"),
        )
        on_change = MagicMock()
        watcher = _HeadWatcher(gp, on_change)
        watcher.stop()
        assert watcher._stop_event.is_set()

    def test_stat_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _HeadWatcher._stat_file(str(f))
        assert result is not None
        assert len(result) == 3

    def test_stat_file_none(self) -> None:
        assert _HeadWatcher._stat_file(None) is None

    def test_stat_file_missing(self) -> None:
        assert _HeadWatcher._stat_file("/nonexistent/file") is None

    def test_stat_dir(self, tmp_path: Path) -> None:
        (tmp_path / "HEAD").write_text("ref: refs/heads/main\n")
        result = _HeadWatcher._stat_dir(str(tmp_path))
        assert result is not None

    def test_stat_dir_none(self) -> None:
        assert _HeadWatcher._stat_dir(None) is None

    def test_stat_dir_missing(self) -> None:
        assert _HeadWatcher._stat_dir("/nonexistent/dir") is None

    def test_stat_dir_path_ending_with_head(self, tmp_path: Path) -> None:
        head_file = tmp_path / "HEAD"
        head_file.write_text("ref: refs/heads/main\n")
        result = _HeadWatcher._stat_dir(str(head_file))
        assert result is not None


class TestFooterScheduleRefresh:
    def test_schedule_refresh_when_disposed(self, tmp_path: Path) -> None:
        provider = FooterDataProvider(cwd=str(tmp_path))
        provider._disposed = True
        provider._schedule_refresh()
        assert provider._refresh_timer is None
        provider.dispose()

    def test_schedule_refresh_dedup(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            # First schedule
            provider._schedule_refresh()
            timer1 = provider._refresh_timer
            assert timer1 is not None
            # Second schedule should be a no-op (already scheduled)
            provider._schedule_refresh()
            assert provider._refresh_timer is timer1
        finally:
            provider.dispose()

    def test_do_refresh_when_disposed(self, tmp_path: Path) -> None:
        provider = FooterDataProvider(cwd=str(tmp_path))
        provider._disposed = True
        # Should not raise
        provider._do_refresh()
        provider.dispose()

    def test_do_refresh_branch_change(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            # Prime the cache
            assert provider.get_git_branch() == "main"

            received: list[str | None] = []
            provider.on_branch_change(lambda b: received.append(b))

            # Change HEAD then do refresh
            (git_dir / "HEAD").write_text("ref: refs/heads/feature\n")
            provider._do_refresh()
            assert received == ["feature"]
        finally:
            provider.dispose()

    def test_do_refresh_same_branch(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            assert provider.get_git_branch() == "main"
            received: list[str | None] = []
            provider.on_branch_change(lambda b: received.append(b))
            provider._do_refresh()
            # Same branch, no callback
            assert received == []
        finally:
            provider.dispose()

    def test_callback_error_swallowed(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            def bad_callback(b: str | None) -> None:
                raise RuntimeError("boom")

            provider.on_branch_change(bad_callback)
            # Should not raise
            provider._notify_branch_change()
        finally:
            provider.dispose()

    def test_cancel_refresh_timer(self, tmp_path: Path) -> None:
        provider = FooterDataProvider(cwd=str(tmp_path))
        try:
            # Cancel when no timer — should be fine
            provider._cancel_refresh_timer()
            assert provider._refresh_timer is None
        finally:
            provider.dispose()


class TestHeadWatcherRun:
    def test_detects_change(self, tmp_path: Path) -> None:
        """Test that the watcher detects a HEAD change and calls on_change."""
        import time

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        head = git_dir / "HEAD"
        head.write_text("ref: refs/heads/main\n")

        gp = _GitPaths(
            repo_dir=str(tmp_path),
            common_git_dir=str(git_dir),
            head_path=str(head),
        )
        on_change = MagicMock()
        watcher = _HeadWatcher(gp, on_change)
        try:
            # Modify HEAD to trigger change detection
            time.sleep(0.1)
            head.write_text("ref: refs/heads/feature\n")
            # Wait for poll
            time.sleep(0.6)
            assert on_change.call_count >= 1
        finally:
            watcher.stop()

    def test_watcher_with_reftable(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        head = git_dir / "HEAD"
        head.write_text("ref: refs/heads/main\n")
        reftable = git_dir / "reftable"
        reftable.mkdir()
        tables = reftable / "tables.list"
        tables.write_text("table1")

        gp = _GitPaths(
            repo_dir=str(tmp_path),
            common_git_dir=str(git_dir),
            head_path=str(head),
        )
        on_change = MagicMock()
        watcher = _HeadWatcher(gp, on_change)
        assert watcher._reftable_dir is not None
        assert watcher._tables_list_path is not None
        watcher.stop()


class TestFindGitPathsEdgeCases:
    def test_git_file_without_gitdir_prefix(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").write_text("something random\n")
        # .git file doesn't start with "gitdir: " — should be caught by the
        # subsequent code that checks for HEAD, but actually the code only
        # enters the `if content.startswith("gitdir: ")` branch.
        # So it falls through to parent walk.
        result = _find_git_paths(str(repo))
        # The parent tmp_path has no .git, so None
        assert result is None
