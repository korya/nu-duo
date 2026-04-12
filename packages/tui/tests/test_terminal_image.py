"""Tests for ``nu_tui.terminal_image``."""

from __future__ import annotations

from nu_tui.terminal_image import (
    is_image_line,
    render_iterm2_image,
    render_kitty_image,
)


def test_render_kitty_image_produces_apc_sequence() -> None:
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    lines = render_kitty_image(data, columns=40, rows=10)
    assert len(lines) >= 1
    assert lines[0].startswith("\x1b_G")
    assert "a=T" in lines[0]
    assert "f=100" in lines[0]


def test_render_kitty_image_chunks_large_data() -> None:
    data = b"\x00" * 10000
    lines = render_kitty_image(data)
    assert len(lines) >= 2  # should chunk at 4096 bytes
    assert "m=1" in lines[0]  # more chunks follow
    assert "m=0" in lines[-1]  # last chunk


def test_render_iterm2_image_produces_osc_sequence() -> None:
    data = b"fake image data"
    line = render_iterm2_image(data, name="test.png")
    assert "\x1b]1337;File=" in line
    assert "inline=1" in line


def test_is_image_line_kitty() -> None:
    assert is_image_line("\x1b_Ga=T,f=100;data\x1b\\") is True
    assert is_image_line("normal text") is False


def test_is_image_line_iterm2() -> None:
    assert is_image_line("\x1b]1337;File=inline=1:data\x07") is True
