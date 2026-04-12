"""Tests for ``nu_tui.components.Loader`` and ``CancellableLoader``."""

from __future__ import annotations

from nu_tui.components.loader import CancellableLoader, Loader

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_renders_spinner_and_message() -> None:
    loader = Loader(message="Working...")
    lines = loader.render(30)
    assert len(lines) == 2  # empty line + spinner line
    assert lines[0] == ""
    assert "Working..." in lines[1]
    # First frame should be ⠋
    assert "⠋" in lines[1]


def test_loader_set_message_changes_text() -> None:
    loader = Loader(message="first")
    assert "first" in loader.render(30)[1]
    loader.set_message("second")
    assert "second" in loader.render(30)[1]


def test_loader_stop_does_not_crash() -> None:
    loader = Loader(message="x")
    loader.stop()
    loader.stop()  # idempotent


def test_loader_start_without_event_loop_is_safe() -> None:
    """start() outside an event loop just skips the animation task."""
    loader = Loader(message="x")
    loader.start()  # no running loop → no-op
    loader.stop()


def test_loader_custom_color_fns() -> None:
    loader = Loader(
        spinner_color_fn=lambda s: f"[{s}]",
        message_color_fn=lambda s: f"({s})",
        message="test",
    )
    lines = loader.render(30)
    assert "[" in lines[1]
    assert "(" in lines[1]


def test_loader_request_render_callback() -> None:
    called: list[bool] = []
    Loader(message="x", request_render=lambda: called.append(True))
    # The constructor calls _update_display which fires the callback.
    assert len(called) >= 1


# ---------------------------------------------------------------------------
# CancellableLoader
# ---------------------------------------------------------------------------


def test_cancellable_loader_starts_uncancelled() -> None:
    cl = CancellableLoader(message="Loading")
    assert cl.cancelled is False


def test_cancellable_loader_escape_cancels() -> None:
    cl = CancellableLoader(message="Loading")
    aborted: list[bool] = []
    cl.on_abort = lambda: aborted.append(True)
    cl.handle_input("escape")
    assert cl.cancelled is True
    assert aborted == [True]


def test_cancellable_loader_dispose_stops() -> None:
    cl = CancellableLoader(message="Loading")
    cl.dispose()
    # Should not crash on subsequent calls.
    cl.dispose()


def test_cancellable_loader_renders_like_loader() -> None:
    cl = CancellableLoader(message="Thinking...")
    lines = cl.render(30)
    assert len(lines) == 2
    assert "Thinking..." in lines[1]
