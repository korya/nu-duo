"""Fast deterministic string hash.

Port of ``packages/ai/src/utils/hash.ts`` — a non-cryptographic 64-bit hash
used only to shorten arbitrary strings into fixed-length identifiers. The
implementation mirrors the JavaScript one bit-for-bit so hashes computed in
Python match hashes computed by the upstream TS version.
"""

from __future__ import annotations

_MASK32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Mirror of ``Math.imul`` — 32-bit integer multiplication (two's complement)."""
    return (a * b) & _MASK32


def short_hash(value: str) -> str:
    """Return a compact base-36 hash of ``value``.

    Deterministic and portable across languages. Not cryptographically
    secure — intended solely for shortening long identifiers.
    """
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    for ch in value:
        code = ord(ch) & _MASK32
        h1 = _imul(h1 ^ code, 2654435761)
        h2 = _imul(h2 ^ code, 1597334677)
    h1 = _imul(h1 ^ (h1 >> 16), 2246822507) ^ _imul(h2 ^ (h2 >> 13), 3266489909)
    h1 &= _MASK32
    h2 = _imul(h2 ^ (h2 >> 16), 2246822507) ^ _imul(h1 ^ (h1 >> 13), 3266489909)
    h2 &= _MASK32
    return _to_base36(h2) + _to_base36(h1)


_BASE36_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        value, rem = divmod(value, 36)
        digits.append(_BASE36_DIGITS[rem])
    return "".join(reversed(digits))


__all__ = ["short_hash"]
