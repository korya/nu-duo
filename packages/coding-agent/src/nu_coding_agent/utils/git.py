"""Git URL parser — port of ``packages/coding-agent/src/utils/git.ts``.

The upstream uses the ``hosted-git-info`` npm package to recognise
GitHub / GitLab / Bitbucket shorthands. Python doesn't have a 1:1
equivalent, so this port hand-rolls support for the same input forms
the package_manager actually feeds it:

* ``https://github.com/owner/repo``
* ``http://...`` / ``ssh://...`` / ``git://...`` URLs
* ``git@github.com:owner/repo`` (SCP-like)
* ``github:owner/repo`` shorthand (with optional ``#ref``)
* ``git:`` prefixed forms above
* an optional ``@<ref>`` or ``#<ref>`` suffix on any of the above

Anything that doesn't fit these shapes returns ``None`` so the package
manager treats it as a non-git source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse, urlunparse


@dataclass(slots=True)
class GitSource:
    """Parsed git URL — mirrors the upstream ``GitSource`` interface."""

    type: Literal["git"]
    repo: str
    """Clone URL (no ref suffix)."""
    host: str
    """Git host domain, e.g. ``github.com``."""
    path: str
    """Repository path, e.g. ``user/repo``."""
    ref: str | None = None
    """Branch, tag, or commit if specified."""
    pinned: bool = False
    """``True`` when a ref was specified — package_manager skips auto-update."""


_SCP_LIKE_RE = re.compile(r"^git@([^:]+):(.+)$")
_PROTO_RE = re.compile(r"^(https?|ssh|git)://", re.IGNORECASE)
_KNOWN_SHORTHAND_HOSTS = {
    "github": "github.com",
    "gitlab": "gitlab.com",
    "bitbucket": "bitbucket.org",
    "gist": "gist.github.com",
}


def _split_ref(url: str) -> tuple[str, str | None]:
    """Strip an ``@<ref>`` (or trailing ``#<ref>``) suffix from ``url``.

    Returns ``(repo_without_ref, ref_or_none)``. Mirrors the upstream
    ``splitRef`` helper exactly.
    """
    # Hash form: github.com/owner/repo#main
    if "#" in url:
        repo_part, ref = url.rsplit("#", 1)
        if repo_part and ref:
            return repo_part, ref

    # SCP-like: git@host:owner/repo@ref
    scp_match = _SCP_LIKE_RE.match(url)
    if scp_match:
        path_with_maybe_ref = scp_match.group(2)
        ref_separator = path_with_maybe_ref.find("@")
        if ref_separator < 0:
            return url, None
        repo_path = path_with_maybe_ref[:ref_separator]
        ref = path_with_maybe_ref[ref_separator + 1 :]
        if not repo_path or not ref:
            return url, None
        return f"git@{scp_match.group(1)}:{repo_path}", ref

    # URL form: https://host/owner/repo@ref
    if "://" in url:
        try:
            parsed = urlparse(url)
        except ValueError:
            return url, None
        path_with_maybe_ref = parsed.path.lstrip("/")
        ref_separator = path_with_maybe_ref.find("@")
        if ref_separator < 0:
            return url, None
        repo_path = path_with_maybe_ref[:ref_separator]
        ref = path_with_maybe_ref[ref_separator + 1 :]
        if not repo_path or not ref:
            return url, None
        new_parsed = parsed._replace(path=f"/{repo_path}")
        return urlunparse(new_parsed).rstrip("/"), ref

    # Bare host/owner/repo@ref
    slash_index = url.find("/")
    if slash_index < 0:
        return url, None
    host = url[:slash_index]
    path_with_maybe_ref = url[slash_index + 1 :]
    ref_separator = path_with_maybe_ref.find("@")
    if ref_separator < 0:
        return url, None
    repo_path = path_with_maybe_ref[:ref_separator]
    ref = path_with_maybe_ref[ref_separator + 1 :]
    if not repo_path or not ref:
        return url, None
    return f"{host}/{repo_path}", ref


def _parse_shorthand(url: str) -> GitSource | None:
    """Parse ``github:owner/repo`` style shorthands."""
    for prefix, host in _KNOWN_SHORTHAND_HOSTS.items():
        marker = f"{prefix}:"
        if not url.startswith(marker):
            continue
        rest = url[len(marker) :]
        if "/" not in rest:
            return None
        repo_without_ref, ref = _split_ref(rest)
        if "/" not in repo_without_ref:
            return None
        normalized_path = repo_without_ref.removesuffix(".git")
        return GitSource(
            type="git",
            repo=f"https://{host}/{normalized_path}",
            host=host,
            path=normalized_path,
            ref=ref,
            pinned=bool(ref),
        )
    return None


def _parse_generic(url: str) -> GitSource | None:
    """Fallback for explicit-protocol URLs and SCP-like forms."""
    repo_without_ref, ref = _split_ref(url)
    repo = repo_without_ref
    host = ""
    path = ""

    scp_match = _SCP_LIKE_RE.match(repo_without_ref)
    if scp_match:
        host = scp_match.group(1)
        path = scp_match.group(2)
    elif _PROTO_RE.match(repo_without_ref):
        try:
            parsed = urlparse(repo_without_ref)
        except ValueError:
            return None
        host = parsed.hostname or ""
        path = parsed.path.lstrip("/")
    else:
        slash_index = repo_without_ref.find("/")
        if slash_index < 0:
            return None
        host = repo_without_ref[:slash_index]
        path = repo_without_ref[slash_index + 1 :]
        if "." not in host and host != "localhost":
            return None
        repo = f"https://{repo_without_ref}"

    normalized_path = path.removesuffix(".git").lstrip("/")
    if not host or not normalized_path or len(normalized_path.split("/")) < 2:
        return None

    return GitSource(
        type="git",
        repo=repo,
        host=host,
        path=normalized_path,
        ref=ref,
        pinned=bool(ref),
    )


def parse_git_url(source: str) -> GitSource | None:
    """Parse ``source`` into a :class:`GitSource`, returning ``None`` on miss.

    Resolution order matches the upstream:

    1. ``github:`` / ``gitlab:`` / ``bitbucket:`` / ``gist:`` shorthands
       (no other prefix required).
    2. With a ``git:`` prefix, accept any historical shorthand form.
    3. Without a prefix, only explicit-protocol URLs (``https``,
       ``http``, ``ssh``, ``git``) or ``git@host:path`` SCP-like forms
       are accepted — bare hostnames are rejected to avoid clashing with
       npm package names that happen to contain a slash.
    """
    trimmed = source.strip()

    # Vendor shorthands take priority — they don't need any other prefix.
    shorthand = _parse_shorthand(trimmed)
    if shorthand is not None:
        return shorthand

    has_git_prefix = trimmed.startswith("git:") and not trimmed.startswith("git://")
    url = trimmed[4:].strip() if has_git_prefix else trimmed

    if not has_git_prefix and not _PROTO_RE.match(url) and not url.startswith("git@"):
        return None

    nested_shorthand = _parse_shorthand(url)
    if nested_shorthand is not None:
        return nested_shorthand

    return _parse_generic(url)


__all__ = ["GitSource", "parse_git_url"]
