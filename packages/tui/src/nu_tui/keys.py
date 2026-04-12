"""Key identifiers and matching.

Smaller-than-upstream port of ``packages/tui/src/keys.ts``. The TS
version contains 1356 lines mostly dedicated to parsing raw stdin
escape sequences from various terminal protocols (legacy + Kitty).
Textual already handles all of that for us, so the Python port
exposes only the surface that consumers actually need:

* :data:`KeyId` — string type alias used by :mod:`nu_tui.keybindings`.
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

import re as _re

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
    lowercased — nu_tui's KeyId space is case-insensitive), and sorts
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


# =============================================================================
# Legacy terminal escape-sequence tables
# =============================================================================

#: Maps key names to the list of escape sequences that terminals emit for
#: them in legacy (non-Kitty) mode.  Mirrors ``LEGACY_KEY_SEQUENCES`` in the
#: upstream ``keys.ts``.
LEGACY_KEY_SEQUENCES: dict[str, list[str]] = {
    "up": ["\x1b[A", "\x1bOA"],
    "down": ["\x1b[B", "\x1bOB"],
    "right": ["\x1b[C", "\x1bOC"],
    "left": ["\x1b[D", "\x1bOD"],
    "home": ["\x1b[H", "\x1bOH", "\x1b[1~", "\x1b[7~"],
    "end": ["\x1b[F", "\x1bOF", "\x1b[4~", "\x1b[8~"],
    "insert": ["\x1b[2~"],
    "delete": ["\x1b[3~"],
    "pageUp": ["\x1b[5~", "\x1b[[5~"],
    "pageDown": ["\x1b[6~", "\x1b[[6~"],
    "clear": ["\x1b[E", "\x1bOE"],
    "f1": ["\x1bOP", "\x1b[11~", "\x1b[[A"],
    "f2": ["\x1bOQ", "\x1b[12~", "\x1b[[B"],
    "f3": ["\x1bOR", "\x1b[13~", "\x1b[[C"],
    "f4": ["\x1bOS", "\x1b[14~", "\x1b[[D"],
    "f5": ["\x1b[15~", "\x1b[[E"],
    "f6": ["\x1b[17~"],
    "f7": ["\x1b[18~"],
    "f8": ["\x1b[19~"],
    "f9": ["\x1b[20~"],
    "f10": ["\x1b[21~"],
    "f11": ["\x1b[23~"],
    "f12": ["\x1b[24~"],
}

#: Direct escape-sequence → key-id lookup built from the upstream
#: ``LEGACY_SEQUENCE_KEY_IDS`` constant.
_LEGACY_SEQUENCE_KEY_IDS: dict[str, str] = {
    "\x1bOA": "up",
    "\x1bOB": "down",
    "\x1bOC": "right",
    "\x1bOD": "left",
    "\x1bOH": "home",
    "\x1bOF": "end",
    "\x1b[E": "clear",
    "\x1bOE": "clear",
    "\x1bOe": "ctrl+clear",
    "\x1b[e": "shift+clear",
    "\x1b[2~": "insert",
    "\x1b[2$": "shift+insert",
    "\x1b[2^": "ctrl+insert",
    "\x1b[3$": "shift+delete",
    "\x1b[3^": "ctrl+delete",
    "\x1b[[5~": "pageUp",
    "\x1b[[6~": "pageDown",
    "\x1b[a": "shift+up",
    "\x1b[b": "shift+down",
    "\x1b[c": "shift+right",
    "\x1b[d": "shift+left",
    "\x1bOa": "ctrl+up",
    "\x1bOb": "ctrl+down",
    "\x1bOc": "ctrl+right",
    "\x1bOd": "ctrl+left",
    "\x1b[5$": "shift+pageUp",
    "\x1b[6$": "shift+pageDown",
    "\x1b[7$": "shift+home",
    "\x1b[8$": "shift+end",
    "\x1b[5^": "ctrl+pageUp",
    "\x1b[6^": "ctrl+pageDown",
    "\x1b[7^": "ctrl+home",
    "\x1b[8^": "ctrl+end",
    "\x1bOP": "f1",
    "\x1bOQ": "f2",
    "\x1bOR": "f3",
    "\x1bOS": "f4",
    "\x1b[11~": "f1",
    "\x1b[12~": "f2",
    "\x1b[13~": "f3",
    "\x1b[14~": "f4",
    "\x1b[[A": "f1",
    "\x1b[[B": "f2",
    "\x1b[[C": "f3",
    "\x1b[[D": "f4",
    "\x1b[[E": "f5",
    "\x1b[15~": "f5",
    "\x1b[17~": "f6",
    "\x1b[18~": "f7",
    "\x1b[19~": "f8",
    "\x1b[20~": "f9",
    "\x1b[21~": "f10",
    "\x1b[23~": "f11",
    "\x1b[24~": "f12",
    "\x1bb": "alt+left",
    "\x1bf": "alt+right",
    "\x1bp": "alt+up",
    "\x1bn": "alt+down",
}

# =============================================================================
# CSI-u / Kitty helpers (lightweight — no Kitty state machine needed here)
# =============================================================================

_CSI_U_RE = _re.compile(r"^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$")
_ARROW_MOD_RE = _re.compile(r"^\x1b\[1;(\d+)(?::(\d+))?([ABCD])$")
_FUNC_MOD_RE = _re.compile(r"^\x1b\[(\d+)(?:;(\d+))?(?::(\d+))?~$")
_MOD_OTHER_RE = _re.compile(r"^\x1b\[27;(\d+);(\d+)~$")

_ARROW_CODES: dict[str, str] = {"A": "up", "B": "down", "C": "right", "D": "left"}
_FUNC_NUMS: dict[int, str] = {
    2: "insert",
    3: "delete",
    5: "pageUp",
    6: "pageDown",
    7: "home",
    8: "end",
}
_MOD_SHIFT = 1
_MOD_ALT = 2
_MOD_CTRL = 4
_LOCK_MASK = 64 + 128

_CODEPOINT_NAMES: dict[int, str] = {
    27: "escape",
    9: "tab",
    13: "enter",
    32: "space",
    127: "backspace",
    57414: "enter",  # kpEnter
}


def _modifier_prefix(mod: int) -> str:
    """Return a ``+``-terminated modifier prefix string for bitmask *mod*."""
    effective = mod & ~_LOCK_MASK
    parts: list[str] = []
    if effective & _MOD_SHIFT:
        parts.append("shift")
    if effective & _MOD_CTRL:
        parts.append("ctrl")
    if effective & _MOD_ALT:
        parts.append("alt")
    return ("+".join(parts) + "+") if parts else ""


def _parse_csi_u(data: str) -> str | None:
    """Attempt to decode a CSI-u (Kitty) sequence → key-id string."""
    m = _CSI_U_RE.match(data)
    if not m:
        return None
    codepoint = int(m.group(1))
    shifted = int(m.group(2)) if m.group(2) else None
    mod_val = int(m.group(4)) if m.group(4) else 1
    modifier = mod_val - 1

    prefix = _modifier_prefix(modifier)
    effective = shifted if (shifted is not None and modifier & _MOD_SHIFT) else codepoint

    name = _CODEPOINT_NAMES.get(effective)
    if name:
        return prefix + name
    # Arrow pseudo-codepoints
    arrow_map = {-1: "up", -2: "down", -3: "right", -4: "left"}
    if effective in arrow_map:
        return prefix + arrow_map[effective]
    # Printable ASCII
    if 32 <= effective <= 126:
        return prefix + chr(effective)
    return None


def _parse_arrow_mod(data: str) -> str | None:
    m = _ARROW_MOD_RE.match(data)
    if not m:
        return None
    mod_val = int(m.group(1))
    modifier = mod_val - 1
    key = _ARROW_CODES[m.group(3)]
    return _modifier_prefix(modifier) + key


def _parse_func_mod(data: str) -> str | None:
    m = _FUNC_MOD_RE.match(data)
    if not m:
        return None
    num = int(m.group(1))
    mod_val = int(m.group(2)) if m.group(2) else 1
    modifier = mod_val - 1
    key = _FUNC_NUMS.get(num)
    if key is None:
        return None
    return _modifier_prefix(modifier) + key


def _parse_mod_other_keys(data: str) -> str | None:
    m = _MOD_OTHER_RE.match(data)
    if not m:
        return None
    mod_val = int(m.group(1))
    codepoint = int(m.group(2))
    modifier = mod_val - 1
    prefix = _modifier_prefix(modifier)
    name = _CODEPOINT_NAMES.get(codepoint)
    if name:
        return prefix + name
    if 32 <= codepoint <= 126:
        return prefix + chr(codepoint)
    return None


def decode_key(data: str) -> str | None:
    """Map raw terminal escape sequences to a normalized :data:`KeyId`.

    Handles:

    * Kitty CSI-u sequences (``ESC [ <cp> ; <mod> u``).
    * Arrow / functional key sequences with modifiers.
    * xterm modifyOtherKeys (``ESC [ 27 ; mod ; cp ~``).
    * The :data:`_LEGACY_SEQUENCE_KEY_IDS` lookup table.
    * Common bare escape codes.

    Returns ``None`` for unrecognized input so callers can fall back to
    passing the raw character through (e.g. printable ASCII).
    """
    if not data:
        return None

    # CSI-u / Kitty
    if result := _parse_csi_u(data):
        return result
    # Arrow with modifier
    if result := _parse_arrow_mod(data):
        return result
    # Functional with modifier
    if result := _parse_func_mod(data):
        return result
    # xterm modifyOtherKeys
    if result := _parse_mod_other_keys(data):
        return result

    # Direct lookup table
    if data in _LEGACY_SEQUENCE_KEY_IDS:
        return _LEGACY_SEQUENCE_KEY_IDS[data]

    # Bare escape codes
    if data == "\x1b":
        return "escape"
    if data == "\x1c":
        return "ctrl+\\"
    if data == "\x1d":
        return "ctrl+]"
    if data == "\x1f":
        return "ctrl+-"
    if data == "\x1b\x1b":
        return "ctrl+alt+["
    if data == "\x1b\x1c":
        return "ctrl+alt+\\"
    if data == "\x1b\x1d":
        return "ctrl+alt+]"
    if data == "\x1b\x1f":
        return "ctrl+alt+-"
    if data == "\t":
        return "tab"
    if data in ("\r", "\n", "\x1bOM"):
        return "enter"
    if data == "\x00":
        return "ctrl+space"
    if data == " ":
        return "space"
    if data == "\x7f":
        return "backspace"
    if data == "\x08":
        return "backspace"
    if data == "\x1b[Z":
        return "shift+tab"
    if data in ("\x1b\x7f", "\x1b\b"):
        return "alt+backspace"
    if data == "\x1b[A":
        return "up"
    if data == "\x1b[B":
        return "down"
    if data == "\x1b[C":
        return "right"
    if data == "\x1b[D":
        return "left"
    if data in ("\x1b[H", "\x1bOH"):
        return "home"
    if data in ("\x1b[F", "\x1bOF"):
        return "end"
    if data == "\x1b[3~":
        return "delete"
    if data == "\x1b[5~":
        return "pageUp"
    if data == "\x1b[6~":
        return "pageDown"

    # ESC + printable (alt+key or ctrl+alt+key)
    if len(data) == 2 and data[0] == "\x1b":
        code = ord(data[1])
        if 1 <= code <= 26:
            return f"ctrl+alt+{chr(code + 96)}"
        if (97 <= code <= 122) or (48 <= code <= 57):
            return f"alt+{chr(code)}"

    # Raw single character
    if len(data) == 1:
        code = ord(data)
        if 1 <= code <= 26:
            return f"ctrl+{chr(code + 96)}"
        if 32 <= code <= 126:
            return data

    return None


# =============================================================================
# Display label helper
# =============================================================================

_KEY_DISPLAY_NAMES: dict[str, str] = {
    "escape": "Esc",
    "esc": "Esc",
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "space": "Space",
    "backspace": "Backspace",
    "delete": "Delete",
    "insert": "Insert",
    "clear": "Clear",
    "home": "Home",
    "end": "End",
    # camelCase originals
    "pageUp": "Page Up",
    "pageDown": "Page Down",
    # normalize_key_id lowercases everything, so add lowercase aliases too
    "pageup": "Page Up",
    "pagedown": "Page Down",
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
}

_MODIFIER_DISPLAY: dict[str, str] = {
    "ctrl": "Ctrl",
    "shift": "Shift",
    "alt": "Alt",
}


def get_key_label(key_id: KeyId) -> str:
    """Return a human-readable display label for *key_id*.

    Examples::

        get_key_label("ctrl+c")       -> "Ctrl+C"
        get_key_label("shift+enter")  -> "Shift+Enter"
        get_key_label("escape")       -> "Esc"
        get_key_label("a")            -> "A"

    Modifiers are printed in canonical order (Ctrl → Shift → Alt) and the
    base key is looked up in :data:`_KEY_DISPLAY_NAMES`; unknown keys are
    upper-cased for display.
    """
    normalized = normalize_key_id(key_id)
    parts = normalized.split("+")
    base = parts[-1]
    modifiers = parts[:-1]

    base_label = _KEY_DISPLAY_NAMES.get(base, base.upper() if len(base) == 1 else base)
    modifier_labels = [_MODIFIER_DISPLAY.get(m, m.capitalize()) for m in modifiers]

    if modifier_labels:
        return "+".join([*modifier_labels, base_label])
    return base_label


__all__ = [
    "LEGACY_KEY_SEQUENCES",
    "Key",
    "KeyId",
    "decode_key",
    "get_key_label",
    "matches_key",
    "normalize_key_id",
]
