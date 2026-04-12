"""``Loader`` + ``CancellableLoader`` — ports of upstream loader/cancellable-loader.

Animated spinner that updates its display at a fixed interval.
The Python port uses ``asyncio`` timers instead of ``setInterval``
since the rendering lifecycle is async.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from nu_tui.components.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable

_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_INTERVAL_SECONDS = 0.08


class Loader(Text):
    """Animated spinner with a message.

    Starts spinning immediately on construction. Call :meth:`stop` to
    halt the animation. The spinner is rendered as a single line with
    an empty line above it (matching upstream ``Loader.render``).

    ``spinner_color_fn`` and ``message_color_fn`` are callables that
    wrap their argument in ANSI escape sequences. Pass ``lambda s: s``
    for no styling.
    """

    def __init__(
        self,
        spinner_color_fn: Callable[[str], str] = lambda s: s,
        message_color_fn: Callable[[str], str] = lambda s: s,
        message: str = "Loading...",
        *,
        request_render: Callable[[], None] | None = None,
    ) -> None:
        super().__init__("")
        self._spinner_color_fn = spinner_color_fn
        self._message_color_fn = message_color_fn
        self._message = message
        self._request_render = request_render
        self._current_frame = 0
        self._task: asyncio.Task[None] | None = None
        self._stopped = False
        self._update_display()

    def render(self, width: int) -> list[str]:
        return ["", *super().render(width)]

    def start(self) -> None:
        """Start the spinner animation loop. Called automatically by __init__."""
        if self._task is not None or self._stopped:
            return
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._spin())
        except RuntimeError:
            # No running loop (e.g. tests not running under asyncio) — skip.
            pass

    def stop(self) -> None:
        """Halt the animation."""
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def set_message(self, message: str) -> None:
        self._message = message
        self._update_display()

    def _update_display(self) -> None:
        frame = _FRAMES[self._current_frame % len(_FRAMES)]
        self.set_text(f"{self._spinner_color_fn(frame)} {self._message_color_fn(self._message)}")
        if self._request_render is not None:
            self._request_render()

    async def _spin(self) -> None:
        while not self._stopped:
            await asyncio.sleep(_INTERVAL_SECONDS)
            self._current_frame = (self._current_frame + 1) % len(_FRAMES)
            self._update_display()


class CancellableLoader(Loader):
    """A :class:`Loader` that can be cancelled with Escape.

    Exposes an :class:`asyncio.Event` (``cancelled_event``) that is
    set when the user presses Escape. Consumers ``await`` it or check
    ``loader.cancelled`` to react.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cancelled_event = asyncio.Event()
        self.on_abort: Callable[[], None] | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled_event.is_set()

    @property
    def cancelled_event(self) -> asyncio.Event:
        return self._cancelled_event

    def handle_input(self, data: str) -> None:
        from nu_tui.keybindings import get_keybindings  # noqa: PLC0415

        kb = get_keybindings()
        if kb.matches(data, "tui.select.cancel"):
            self._cancelled_event.set()
            if self.on_abort:
                self.on_abort()

    def dispose(self) -> None:
        """Stop the spinner and mark as cancelled."""
        self.stop()


__all__ = ["CancellableLoader", "Loader"]
