"""Stdin buffer — port of ``packages/tui/src/stdin-buffer.ts``.

Provides a buffered stdin reader for raw terminal input. In the
Python port, Textual handles stdin reading, so this module is
primarily provided for **structural completeness** and for use
cases where code runs outside Textual (e.g. a raw-terminal mode
for testing or a non-Textual REPL).

The upstream is 386 LoC with escape sequence buffering, timeout
handling, and bracketed paste support. This port provides the
core data types and a simplified synchronous reader.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InputEvent:
    """A decoded input event from the terminal."""

    raw: str
    """The raw bytes received from stdin."""
    key: str = ""
    """Decoded key name (e.g. ``"up"``, ``"ctrl+c"``, ``"a"``)."""
    is_paste: bool = False
    """True if this event came from a bracketed paste."""


class StdinBuffer:
    """Buffered stdin reader with escape-sequence awareness.

    In the Textual-backed interactive mode, this class is not used —
    Textual handles stdin directly. This exists for:

    * Structural parity with the upstream pi-tui API.
    * Non-Textual code paths (tests, raw terminal utilities).
    * Extension code that wants to read raw stdin.

    The :meth:`read_event` method is a simplified version of the
    upstream's full CSI parser + Kitty protocol decoder. It handles
    the most common escape sequences and falls back to returning the
    raw data as the key name.
    """

    _ESCAPE_TIMEOUT_MS = 50

    def __init__(self) -> None:
        self._buffer = ""
        self._paste_mode = False
        self._paste_buffer = ""

    def feed(self, data: str) -> list[InputEvent]:
        """Feed raw terminal data and return decoded events.

        This is the push-based API: the caller reads stdin and pushes
        chunks into the buffer. The buffer decodes them into events.
        """
        events: list[InputEvent] = []

        for char in data:
            self._buffer += char

            # Bracketed paste detection
            if self._buffer.endswith("\x1b[200~"):
                self._paste_mode = True
                self._paste_buffer = ""
                self._buffer = ""
                continue

            if self._paste_mode:
                self._paste_buffer += char
                if self._paste_buffer.endswith("\x1b[201~"):
                    content = self._paste_buffer[: -len("\x1b[201~")]
                    events.append(InputEvent(raw=content, key=content, is_paste=True))
                    self._paste_mode = False
                    self._paste_buffer = ""
                    self._buffer = ""
                continue

        # Try to decode the buffer as a key sequence
        if self._buffer and not self._paste_mode:
            decoded = self._decode_buffer()
            if decoded is not None:
                events.append(decoded)
                self._buffer = ""

        return events

    def _decode_buffer(self) -> InputEvent | None:
        """Try to decode the current buffer into an InputEvent."""
        buf = self._buffer

        # Simple single characters
        if len(buf) == 1:
            code = ord(buf)
            if code == 27:
                return None  # Might be start of escape sequence
            if code in {13, 10}:
                return InputEvent(raw=buf, key="enter")
            if code == 9:
                return InputEvent(raw=buf, key="tab")
            if code == 127:
                return InputEvent(raw=buf, key="backspace")
            if 1 <= code <= 26:
                return InputEvent(raw=buf, key=f"ctrl+{chr(code + 96)}")
            return InputEvent(raw=buf, key=buf)

        # Escape sequences
        if buf == "\x1b":
            return None  # Incomplete — might be ESC or start of CSI

        # CSI sequences
        if buf.startswith("\x1b["):
            return self._decode_csi(buf)

        # Alt + character
        if buf.startswith("\x1b") and len(buf) == 2:
            return InputEvent(raw=buf, key=f"alt+{buf[1]}")

        # Unknown — return raw
        return InputEvent(raw=buf, key=buf)

    def _decode_csi(self, buf: str) -> InputEvent | None:
        """Decode a CSI (ESC [) sequence."""
        # Arrow keys
        _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}
        if len(buf) == 3 and buf[2] in _ARROWS:
            return InputEvent(raw=buf, key=_ARROWS[buf[2]])

        # Function keys and special keys with ~ suffix
        _TILDE = {
            "2": "insert",
            "3": "delete",
            "5": "pageUp",
            "6": "pageDown",
            "15": "f5",
            "17": "f6",
            "18": "f7",
            "19": "f8",
            "20": "f9",
            "21": "f10",
            "23": "f11",
            "24": "f12",
        }
        if buf.endswith("~"):
            num = buf[2:-1].split(";", maxsplit=1)[0]
            if num in _TILDE:
                return InputEvent(raw=buf, key=_TILDE[num])

        # Kitty CSI-u: ESC [ codepoint u
        if buf.endswith("u"):
            body = buf[2:-1]
            parts = body.split(";")
            try:
                codepoint = int(parts[0])
            except ValueError:
                return InputEvent(raw=buf, key=buf)

            # Decode modifiers if present
            modifiers = int(parts[1]) - 1 if len(parts) > 1 else 0
            key = chr(codepoint) if 32 <= codepoint < 127 else buf
            prefix = ""
            if modifiers & 4:
                prefix += "ctrl+"
            if modifiers & 1:
                prefix += "shift+"
            if modifiers & 2:
                prefix += "alt+"
            return InputEvent(raw=buf, key=f"{prefix}{key}")

        # Modified arrow keys: ESC [ 1 ; modifier A/B/C/D
        if len(buf) >= 5 and buf[2] == "1" and buf[3] == ";" and buf[-1] in "ABCDHF":
            arrow = _ARROWS.get(buf[-1], buf[-1])
            try:
                mod = int(buf[4:-1])
            except ValueError:
                return InputEvent(raw=buf, key=buf)
            prefix = ""
            mod -= 1
            if mod & 4:
                prefix += "ctrl+"
            if mod & 1:
                prefix += "shift+"
            if mod & 2:
                prefix += "alt+"
            return InputEvent(raw=buf, key=f"{prefix}{arrow}")

        # Incomplete or unknown
        if buf[-1].isalpha() or buf[-1] == "~":
            return InputEvent(raw=buf, key=buf)
        return None  # Might be incomplete

    def flush(self) -> list[InputEvent]:
        """Force-flush any buffered data as events."""
        events: list[InputEvent] = []
        if self._buffer:
            if self._buffer == "\x1b":
                events.append(InputEvent(raw=self._buffer, key="escape"))
            else:
                events.append(InputEvent(raw=self._buffer, key=self._buffer))
            self._buffer = ""
        return events


__all__ = ["InputEvent", "StdinBuffer"]
