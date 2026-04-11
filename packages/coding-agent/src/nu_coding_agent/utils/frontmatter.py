"""YAML frontmatter parser — direct port of ``packages/coding-agent/src/utils/frontmatter.ts``.

The original uses the ``yaml`` npm package; we use :mod:`yaml` (PyYAML).
The slicing offsets and CRLF normalisation match the upstream byte-for-byte
so prompt files written by either implementation parse identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(slots=True)
class ParsedFrontmatter:
    frontmatter: dict[str, Any]
    body: str


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _extract_frontmatter(content: str) -> tuple[str | None, str]:
    normalized = _normalize_newlines(content)
    if not normalized.startswith("---"):
        return None, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return None, normalized
    return normalized[4:end_index], normalized[end_index + 4 :].strip()


def parse_frontmatter(content: str) -> ParsedFrontmatter:
    """Split a markdown document into its YAML frontmatter and body."""
    yaml_string, body = _extract_frontmatter(content)
    if yaml_string is None:
        return ParsedFrontmatter(frontmatter={}, body=body)
    parsed = yaml.safe_load(yaml_string)
    return ParsedFrontmatter(frontmatter=parsed if isinstance(parsed, dict) else {}, body=body)


def strip_frontmatter(content: str) -> str:
    """Return the body of ``content`` with any leading frontmatter stripped."""
    return parse_frontmatter(content).body


__all__ = ["ParsedFrontmatter", "parse_frontmatter", "strip_frontmatter"]
