"""Tests for nu_coding_agent.core.tools.path_utils."""

from __future__ import annotations

from pathlib import Path

from nu_coding_agent.core.tools.path_utils import (
    expand_path,
    resolve_read_path,
    resolve_to_cwd,
)


class TestExpandPath:
    def test_at_prefix_stripped(self) -> None:
        assert expand_path("@foo/bar.txt") == "foo/bar.txt"

    def test_tilde_expands_to_home(self) -> None:
        assert expand_path("~") == str(Path.home())

    def test_tilde_slash_expands_to_home(self) -> None:
        assert expand_path("~/foo") == f"{Path.home()}/foo"

    def test_unicode_space_normalized(self) -> None:
        # U+00A0 NBSP between words.
        assert expand_path("foo\u00a0bar.txt") == "foo bar.txt"

    def test_plain_path_passthrough(self) -> None:
        assert expand_path("/abs/path") == "/abs/path"


class TestResolveToCwd:
    def test_absolute_path_passthrough(self) -> None:
        assert resolve_to_cwd("/etc/hosts", "/tmp") == "/etc/hosts"

    def test_relative_resolved_to_cwd(self, tmp_path: Path) -> None:
        result = resolve_to_cwd("foo.txt", str(tmp_path))
        assert result == str(tmp_path / "foo.txt")

    def test_tilde_expanded(self) -> None:
        result = resolve_to_cwd("~/foo", "/tmp")
        assert result.startswith(str(Path.home()))


class TestResolveReadPath:
    def test_existing_file_returned_directly(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        assert resolve_read_path(str(f), str(tmp_path)) == str(f)

    def test_missing_file_returns_resolved_path(self, tmp_path: Path) -> None:
        # No fallback variant matches → original resolved path returned.
        assert resolve_read_path("nope.txt", str(tmp_path)) == str(tmp_path / "nope.txt")
