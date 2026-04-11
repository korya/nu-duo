"""Key identifiers and matching.

Smaller-than-upstream port of ``packages/tui/src/keys.ts``. The TS
version contains 1356 lines mostly dedicated to parsing raw stdin
escape sequences from various terminal protocols (legacy + Kitty).
Textual already handles all of that for us, so the Python port
exposes only the surface that consumers actually need:

* :data:`KeyId` — string type alias used by :mod:`pi_tui.keybindings`.
* :class:`Key` — builder helper with constants for special / symbol /
  function keys and modifier composition methods.
* :func:`normalize_key_id` — canonicalizes a key id (lowercase,
  modifier order ``ctrl+shift+alt+key``).
* :func:`matches_key` — case-insensitive, modifier-order-insensitive
  comparison.

When the components layer lands the actual stdin parser will live
behind a Textual ``on_key`` bridge that converts :class:`textual.events.Key`
into a normalized :data:`KeyId`.
"""

from __future__ import annotations

type KeyId = str
"""Identifier for a single keystroke (e.g. ``"a"``, ``"ctrl+c"``, ``"alt+enter"``)."""


# Canonical modifier order on the wire — every normalized id sorts modifiers
# in this order so ``ctrl+shift+a`` and ``shift+ctrl+a`` compare equal.
_MODIFIER_ORDER: tuple[str, ...] = ("ctrl", "shift", "alt")
_MODIFIER_SET = frozenset(_MODIFIER_ORDER)


class _KeyHelper:
    """Builder for key identifiers — port of the upstream ``Key`` namespace."""

    # Special keys ----------------------------------------------------
    escape = "escape"
    esc = "esc"
    enter = "enter"
    return_ = "return"
    tab = "tab"
    space = "space"
    backspace = "backspace"
    delete = "delete"
    insert = "insert"
    clear = "clear"
    home = "home"
    end = "end"
    page_up = "pageUp"
    page_down = "pageDown"
    up = "up"
    down = "down"
    left = "left"
    right = "right"

    f1 = "f1"
    f2 = "f2"
    f3 = "f3"
    f4 = "f4"
    f5 = "f5"
    f6 = "f6"
    f7 = "f7"
    f8 = "f8"
    f9 = "f9"
    f10 = "f10"
    f11 = "f11"
    f12 = "f12"

    # Symbol keys -----------------------------------------------------
    backtick = "`"
    hyphen = "-"
    equals = "="
    leftbracket = "["
    rightbracket = "]"
    backslash = "\\"
    semicolon = ";"
    quote = "'"
    comma = ","
    period = "."
    slash = "/"
    exclamation = "!"
    at = "@"
    hash = "#"
    dollar = "$"
    percent = "%"
    caret = "^"
    ampersand = "&"
    asterisk = "*"
    leftparen = "("
    rightparen = ")"
    underscore = "_"
    plus = "+"
    pipe = "|"
    tilde = "~"
    leftbrace = "{"
    rightbrace = "}"
    colon = ":"
    lessthan = "<"
    greaterthan = ">"
    question = "?"

    # Modifier helpers ------------------------------------------------
    @staticmethod
    def ctrl(key: str) -> KeyId:
        return f"ctrl+{key}"

    @staticmethod
    def shift(key: str) -> KeyId:
        return f"shift+{key}"

    @staticmethod
    def alt(key: str) -> KeyId:
        return f"alt+{key}"

    @staticmethod
    def ctrl_shift(key: str) -> KeyId:
        return f"ctrl+shift+{key}"

    @staticmethod
    def shift_ctrl(key: str) -> KeyId:
        return f"shift+ctrl+{key}"

    @staticmethod
    def ctrl_alt(key: str) -> KeyId:
        return f"ctrl+alt+{key}"

    @staticmethod
    def alt_ctrl(key: str) -> KeyId:
        return f"alt+ctrl+{key}"

    @staticmethod
    def shift_alt(key: str) -> KeyId:
        return f"shift+alt+{key}"

    @staticmethod
    def alt_shift(key: str) -> KeyId:
        return f"alt+shift+{key}"

    @staticmethod
    def ctrl_shift_alt(key: str) -> KeyId:
        return f"ctrl+shift+alt+{key}"


Key = _KeyHelper
"""Module-level :class:`_KeyHelper` instance, mirroring the TS ``Key`` const."""


def normalize_key_id(key: KeyId) -> KeyId:
    """Canonicalize ``key`` to ``ctrl+shift+alt+<base>`` order, lowercase.

    Strips whitespace, lowercases the modifiers (the base key is also
    lowercased — pi_tui's KeyId space is case-insensitive), and sorts
    modifiers into the canonical :data:`_MODIFIER_ORDER`.
    """
    cleaned = key.replace(" ", "")
    if not cleaned:
        return cleaned
    parts = cleaned.split("+")
    # The base key is the *last* component; preceding components are
    # modifiers. We do not lowercase symbol keys aggressively because
    # `+` itself can be the base.
    if len(parts) == 1:
        return parts[0].lower()

    base = parts[-1].lower()
    raw_modifiers = [m.lower() for m in parts[:-1]]
    seen: set[str] = set()
    sorted_modifiers: list[str] = []
    for mod in _MODIFIER_ORDER:
        if mod in raw_modifiers and mod not in seen:
            sorted_modifiers.append(mod)
            seen.add(mod)
    # Append any unknown modifiers at the end so we don't silently drop them.
    for mod in raw_modifiers:
        if mod not in seen and mod not in _MODIFIER_SET:
            sorted_modifiers.append(mod)
            seen.add(mod)
    return "+".join([*sorted_modifiers, base])


def matches_key(data: str, key_id: KeyId) -> bool:
    """Return ``True`` iff the input ``data`` matches ``key_id``.

    Both arguments are normalized via :func:`normalize_key_id` before
    comparison so modifier order and casing don't affect the match.
    """
    return normalize_key_id(data) == normalize_key_id(key_id)


__all__ = [
    "Key",
    "KeyId",
    "matches_key",
    "normalize_key_id",
]
