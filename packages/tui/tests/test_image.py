"""Tests for ``nu_tui.components.Image``."""

from __future__ import annotations

import os

from nu_tui.components.image import Image


def test_image_empty_data_renders_alt_text() -> None:
    img = Image(alt_text="[no image]")
    lines = img.render(40)
    assert lines == ["[no image]"]


def test_image_no_data_no_alt_renders_empty() -> None:
    img = Image(alt_text="")
    assert img.render(40) == []


def test_image_with_data_in_unsupported_terminal(monkeypatch: object) -> None:
    """Without Kitty/iTerm2, falls back to alt text."""
    # Clear terminal detection env vars
    for var in ("TERM", "TERM_PROGRAM"):
        os.environ.pop(var, None)

    img = Image(image_data=b"fake png data", alt_text="[image]")
    lines = img.render(40)
    assert lines == ["[image]"]


def test_set_image_replaces_data() -> None:
    img = Image()
    img.set_image(b"new data")
    assert img._image_data == b"new data"  # pyright: ignore[reportPrivateUsage]
