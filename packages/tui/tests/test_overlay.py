"""Tests for ``nu_tui.overlay``."""

from __future__ import annotations

from nu_tui.overlay import OverlayHandle, OverlayMargin, OverlayOptions


def test_overlay_handle_lifecycle() -> None:
    handle = OverlayHandle()
    assert handle.is_hidden() is False
    assert handle.is_focused() is False

    handle.focus()
    assert handle.is_focused() is True

    handle.unfocus()
    assert handle.is_focused() is False

    handle.set_hidden(True)
    assert handle.is_hidden() is True

    handle.set_hidden(False)
    assert handle.is_hidden() is False


def test_overlay_handle_permanent_hide() -> None:
    handle = OverlayHandle()
    handle.hide()
    assert handle.is_hidden() is True
    # Can't un-hide after permanent removal
    handle.set_hidden(False)
    assert handle.is_hidden() is True


def test_overlay_options_defaults() -> None:
    opts = OverlayOptions()
    assert opts.anchor == "center"
    assert opts.offset_x == 0
    assert opts.non_capturing is False


def test_overlay_margin() -> None:
    m = OverlayMargin(top=1, right=2, bottom=3, left=4)
    assert m.top == 1
    assert m.right == 2
