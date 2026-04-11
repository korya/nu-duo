"""Unpaired-surrogate scrubbing.

Port of ``packages/ai/src/utils/sanitize-unicode.ts``.

Python ``str`` objects carry unicode code points, not UTF-16 units, so
emoji like 😀 are a single code point and never involve surrogates. However,
strings can still contain *isolated* surrogate code points (``\\ud800``
through ``\\udfff``) — these arrive from flawed upstream data and crash many
API providers during JSON serialization. This function removes them while
preserving any validly paired surrogate sequences.
"""

from __future__ import annotations


def sanitize_surrogates(text: str) -> str:
    """Return ``text`` with unpaired surrogate code points removed.

    Paired surrogates (a high surrogate ``0xD800-0xDBFF`` immediately
    followed by a low surrogate ``0xDC00-0xDFFF``) are preserved so callers
    that legitimately hand us UTF-16 half-pairs keep their data intact.
    """
    result: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        code = ord(ch)
        if 0xD800 <= code <= 0xDBFF:
            # High surrogate — peek ahead for a paired low surrogate.
            if i + 1 < length:
                nxt = ord(text[i + 1])
                if 0xDC00 <= nxt <= 0xDFFF:
                    result.append(ch)
                    result.append(text[i + 1])
                    i += 2
                    continue
            # Unpaired high surrogate → drop.
            i += 1
            continue
        if 0xDC00 <= code <= 0xDFFF:
            # Unpaired low surrogate → drop. (Paired ones were consumed above.)
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


__all__ = ["sanitize_surrogates"]
