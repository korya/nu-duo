"""Tests for ``nu_coding_agent.utils.git``."""

from __future__ import annotations

import pytest
from nu_coding_agent.utils.git import GitSource, parse_git_url


def test_https_url() -> None:
    src = parse_git_url("https://github.com/owner/repo")
    assert isinstance(src, GitSource)
    assert src.host == "github.com"
    assert src.path == "owner/repo"
    assert src.ref is None
    assert src.pinned is False
    assert src.repo == "https://github.com/owner/repo"


def test_https_with_ref_suffix_at() -> None:
    src = parse_git_url("https://github.com/owner/repo@main")
    assert src is not None
    assert src.ref == "main"
    assert src.pinned is True
    assert src.repo == "https://github.com/owner/repo"


def test_https_with_ref_suffix_hash() -> None:
    src = parse_git_url("https://github.com/owner/repo#deadbeef")
    assert src is not None
    assert src.ref == "deadbeef"
    assert src.pinned is True


def test_https_strips_dot_git_suffix() -> None:
    src = parse_git_url("https://github.com/owner/repo.git")
    assert src is not None
    assert src.path == "owner/repo"


def test_scp_like_url() -> None:
    src = parse_git_url("git@github.com:owner/repo")
    assert src is not None
    assert src.host == "github.com"
    assert src.path == "owner/repo"
    assert src.ref is None


def test_scp_like_with_ref() -> None:
    src = parse_git_url("git@github.com:owner/repo@v1.0")
    assert src is not None
    assert src.ref == "v1.0"
    assert src.pinned is True


def test_ssh_protocol() -> None:
    src = parse_git_url("ssh://git@github.com/owner/repo")
    assert src is not None
    assert src.host == "github.com"
    assert src.path == "owner/repo"


def test_github_shorthand() -> None:
    src = parse_git_url("github:owner/repo")
    assert src is not None
    assert src.host == "github.com"
    assert src.path == "owner/repo"
    assert src.repo == "https://github.com/owner/repo"


def test_github_shorthand_with_ref() -> None:
    src = parse_git_url("github:owner/repo#main")
    assert src is not None
    assert src.ref == "main"
    assert src.pinned is True


def test_gitlab_shorthand() -> None:
    src = parse_git_url("gitlab:group/sub/repo")
    assert src is not None
    assert src.host == "gitlab.com"


def test_bitbucket_shorthand() -> None:
    src = parse_git_url("bitbucket:owner/repo")
    assert src is not None
    assert src.host == "bitbucket.org"


def test_bare_name_rejected() -> None:
    """Without an explicit protocol, bare names are not git sources."""
    assert parse_git_url("just-a-package") is None


def test_url_without_protocol_rejected() -> None:
    """``github.com/owner/repo`` requires the ``https://`` prefix or a shorthand."""
    assert parse_git_url("github.com/owner/repo") is None


def test_git_prefix_with_bare_host() -> None:
    """The ``git:`` prefix lets shorthand-ish forms through to the parser."""
    src = parse_git_url("git:github.com/owner/repo")
    assert src is not None
    assert src.host == "github.com"
    assert src.path == "owner/repo"


def test_invalid_url_returns_none() -> None:
    assert parse_git_url("https://") is None


def test_single_path_segment_rejected() -> None:
    """``https://host/onlyone`` isn't a valid owner/repo path."""
    assert parse_git_url("https://github.com/onlyone") is None


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not-a-url",
        "ftp://example.com/foo/bar",  # unsupported protocol
    ],
)
def test_rejects_garbage(url: str) -> None:
    assert parse_git_url(url) is None


def test_scp_like_with_only_owner_rejected() -> None:
    """SCP-like URLs need at least owner/repo."""
    assert parse_git_url("git@github.com:owner") is None


def test_https_with_host_and_no_path_rejected() -> None:
    assert parse_git_url("https://github.com/") is None


def test_git_protocol_url() -> None:
    src = parse_git_url("git://github.com/owner/repo")
    assert src is not None
    assert src.path == "owner/repo"


def test_ref_at_end_of_https_strips_correctly() -> None:
    src = parse_git_url("https://github.com/owner/repo@v1.0.0")
    assert src is not None
    assert src.repo == "https://github.com/owner/repo"
    assert src.ref == "v1.0.0"


def test_localhost_host_accepted() -> None:
    """Bare hostnames are rejected, but ``localhost`` is whitelisted."""
    src = parse_git_url("git:localhost/owner/repo")
    assert src is not None
    assert src.host == "localhost"


def test_git_prefix_with_https_url() -> None:
    src = parse_git_url("git:https://github.com/owner/repo")
    assert src is not None
    assert src.host == "github.com"


def test_url_with_dot_git_suffix_in_path() -> None:
    src = parse_git_url("https://gitea.example.com/owner/repo.git")
    assert src is not None
    assert src.path == "owner/repo"
