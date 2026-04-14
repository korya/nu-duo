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


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_loader_frame_cycling() -> None:
    """Manual frame cycling changes the displayed spinner character (lines 84-87)."""
    loader = Loader(message="test")
    # Frame 0 = ⠋
    assert "⠋" in loader.render(30)[1]
    # Manually advance the frame
    loader._current_frame = 1
    loader._update_display()
    assert "⠙" in loader.render(30)[1]
    loader._current_frame = 2
    loader._update_display()
    assert "⠹" in loader.render(30)[1]


def test_loader_start_already_started() -> None:
    """start() when already started or stopped is a no-op (line 58)."""
    loader = Loader(message="test")
    loader._stopped = True
    loader.start()  # no-op because _stopped
    assert loader._task is None


def test_loader_stop_clears_task() -> None:
    """stop() cancels the task and sets stopped flag (lines 70-71)."""
    loader = Loader(message="test")
    loader.stop()
    assert loader._stopped is True
    assert loader._task is None
    # Idempotent
    loader.stop()
    assert loader._stopped is True


async def test_loader_spin_async() -> None:
    """start() in an async context creates the animation task (line 61)."""
    import asyncio

    loader = Loader(message="spinning")
    loader.start()
    assert loader._task is not None
    # Let it run for a couple of ticks
    await asyncio.sleep(0.2)
    assert loader._current_frame > 0
    loader.stop()


def test_cancellable_loader_event_property() -> None:
    """cancelled_event property returns the asyncio.Event (line 109)."""
    import asyncio

    cl = CancellableLoader(message="test")
    assert isinstance(cl.cancelled_event, asyncio.Event)
    assert not cl.cancelled_event.is_set()


def test_cancellable_loader_escape_without_callback() -> None:
    """Escape cancels even without on_abort callback (lines 115-117)."""
    cl = CancellableLoader(message="test")
    cl.on_abort = None
    cl.handle_input("escape")
    assert cl.cancelled is True


def test_cancellable_loader_non_cancel_key_ignored() -> None:
    """Non-escape keys are ignored by CancellableLoader."""
    cl = CancellableLoader(message="test")
    cl.handle_input("a")
    assert cl.cancelled is False
