"""Tests for ``nu_coding_agent.utils.frontmatter``."""

from __future__ import annotations

from nu_coding_agent.utils.frontmatter import parse_frontmatter, strip_frontmatter


def test_no_frontmatter_returns_empty_dict() -> None:
    parsed = parse_frontmatter("hello world")
    assert parsed.frontmatter == {}
    assert parsed.body == "hello world"


def test_basic_frontmatter() -> None:
    content = "---\ndescription: hi\nname: foo\n---\n\nbody text"
    parsed = parse_frontmatter(content)
    assert parsed.frontmatter == {"description": "hi", "name": "foo"}
    assert parsed.body == "body text"


def test_unterminated_frontmatter_treated_as_body() -> None:
    parsed = parse_frontmatter("---\nfoo: bar\n\nno end marker")
    assert parsed.frontmatter == {}
    assert parsed.body == "---\nfoo: bar\n\nno end marker"


def test_crlf_normalised() -> None:
    parsed = parse_frontmatter("---\r\ndescription: x\r\n---\r\nbody")
    assert parsed.frontmatter == {"description": "x"}
    assert parsed.body == "body"


def test_strip_frontmatter() -> None:
    assert strip_frontmatter("---\ndescription: x\n---\n\nbody") == "body"
    assert strip_frontmatter("just body") == "just body"


def test_empty_yaml_returns_dict() -> None:
    parsed = parse_frontmatter("---\n---\nbody")
    assert parsed.frontmatter == {}
    assert parsed.body == "body"
