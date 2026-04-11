"""Path resolution helpers shared by every file-touching tool.

Direct port of ``packages/coding-agent/src/core/tools/path-utils.ts``.
Keeps the macOS-specific filename fallbacks (NFD normalization, narrow
no-break space before AM/PM in screenshot filenames, curly-quote
substitution) so paste-from-Finder paths resolve the same way they do
upstream.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202f"
_AM_PM_PATTERN = re.compile(r" (AM|PM)\.")


def _normalize_unicode_spaces(s: str) -> str:
    return _UNICODE_SPACES.sub(" ", s)


def _try_macos_screenshot_path(file_path: str) -> str:
    """Replace `` AM.``/`` PM.`` with the narrow-no-break-space variant macOS uses."""
    return _AM_PM_PATTERN.sub(rf"{_NARROW_NO_BREAK_SPACE}\1.", file_path)


def _try_nfd_variant(file_path: str) -> str:
    """macOS stores filenames in NFD (decomposed) form; convert user input."""
    return unicodedata.normalize("NFD", file_path)


def _try_curly_quote_variant(file_path: str) -> str:
    """Replace U+0027 apostrophe with U+2019 (right single quotation mark)."""
    return file_path.replace("'", "\u2019")


def _file_exists(file_path: str) -> bool:
    return Path(file_path).exists()


def _normalize_at_prefix(file_path: str) -> str:
    return file_path[1:] if file_path.startswith("@") else file_path


def expand_path(file_path: str) -> str:
    """Expand ``~`` / ``~/...`` and normalize Unicode spaces and ``@`` prefix."""
    normalized = _normalize_unicode_spaces(_normalize_at_prefix(file_path))
    if normalized == "~":
        return str(Path.home())
    if normalized.startswith("~/"):
        return str(Path.home()) + normalized[1:]
    return normalized


def resolve_to_cwd(file_path: str, cwd: str) -> str:
    """Resolve ``file_path`` relative to ``cwd`` (absolute paths pass through)."""
    expanded = expand_path(file_path)
    if os.path.isabs(expanded):
        return expanded
    return str(Path(cwd) / expanded)


def resolve_read_path(file_path: str, cwd: str) -> str:
    """Resolve a path for reading, falling back through macOS filename variants."""
    resolved = resolve_to_cwd(file_path, cwd)
    if _file_exists(resolved):
        return resolved

    am_pm_variant = _try_macos_screenshot_path(resolved)
    if am_pm_variant != resolved and _file_exists(am_pm_variant):
        return am_pm_variant

    nfd_variant = _try_nfd_variant(resolved)
    if nfd_variant != resolved and _file_exists(nfd_variant):
        return nfd_variant

    curly_variant = _try_curly_quote_variant(resolved)
    if curly_variant != resolved and _file_exists(curly_variant):
        return curly_variant

    nfd_curly_variant = _try_curly_quote_variant(nfd_variant)
    if nfd_curly_variant != resolved and _file_exists(nfd_curly_variant):
        return nfd_curly_variant

    return resolved


__all__ = ["expand_path", "resolve_read_path", "resolve_to_cwd"]
