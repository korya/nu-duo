"""Tests for ``nu_coding_agent.utils.paths``."""

from __future__ import annotations

import pytest
from nu_coding_agent.utils.paths import is_local_path


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("./local", True),
        ("relative", True),
        ("/absolute/path", True),
        ("~/expand", True),
        ("  spaced  ", True),
        ("npm:foo", False),
        ("git:foo", False),
        ("github:owner/repo", False),
        ("http://example.com", False),
        ("https://example.com", False),
        ("ssh://host", False),
    ],
)
def test_is_local_path(value: str, expected: bool) -> None:
    assert is_local_path(value) is expected
