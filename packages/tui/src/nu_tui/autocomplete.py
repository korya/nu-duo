"""Autocomplete system — port of ``packages/tui/src/autocomplete.ts``.

Provides the :class:`AutocompleteProvider` protocol and concrete
implementations for file-path completion and slash-command completion.
The :class:`CombinedAutocompleteProvider` merges multiple providers
into a single suggestion stream.

The upstream is 773 LoC; this port covers the core protocol and the
file-path provider (~350 LoC). Slash-command completion depends on
the command registry which lives in ``nu_coding_agent`` and will be
wired in a follow-up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class AutocompleteSuggestion:
    """A single completion suggestion."""

    value: str
    label: str
    description: str = ""
    insert_text: str | None = None
    kind: str = "text"  # "text" | "file" | "directory" | "command"


@dataclass(slots=True)
class AutocompleteSuggestions:
    """Result from an autocomplete provider."""

    items: list[AutocompleteSuggestion] = field(default_factory=list)
    prefix: str = ""
    replace_start: int = 0
    replace_end: int = 0


class AutocompleteProvider(Protocol):
    """Protocol for autocomplete providers."""

    async def get_suggestions(
        self,
        text: str,
        cursor: int,
        *,
        force: bool = False,
    ) -> AutocompleteSuggestions | None: ...


# ---------------------------------------------------------------------------
# File path autocomplete
# ---------------------------------------------------------------------------


def _is_path_trigger(text: str, cursor: int) -> tuple[str, int] | None:
    """Check if the text at cursor position looks like a path being typed.

    Returns ``(prefix, start_pos)`` or ``None``.
    """
    # Walk backwards from cursor to find the start of the path
    start = cursor
    while start > 0 and text[start - 1] not in (" ", "\t", "\n", '"', "'", "(", ")"):
        start -= 1
    prefix = text[start:cursor]
    if not prefix:
        return None
    # Must look like a path (contains / or . or ~)
    if "/" in prefix or prefix.startswith((".", "~")):
        return prefix, start
    return None


class FileAutocompleteProvider:
    """Autocomplete provider for file paths.

    Completes file and directory names relative to ``cwd``. Supports
    ``~/`` expansion and both relative and absolute paths.
    """

    def __init__(self, cwd: str) -> None:
        self._cwd = cwd

    async def get_suggestions(
        self,
        text: str,
        cursor: int,
        *,
        force: bool = False,
    ) -> AutocompleteSuggestions | None:
        trigger = _is_path_trigger(text, cursor)
        if trigger is None:
            return None

        prefix, start_pos = trigger
        return self._complete_path(prefix, start_pos, cursor)

    def _complete_path(self, prefix: str, start_pos: int, end_pos: int) -> AutocompleteSuggestions:
        # Expand ~ to home directory
        expanded = os.path.expanduser(prefix)

        # Split into directory and partial filename
        if expanded.endswith("/"):
            search_dir = expanded
            partial = ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            partial = os.path.basename(expanded)

        # Resolve relative to cwd
        if not os.path.isabs(search_dir):
            search_dir = os.path.join(self._cwd, search_dir)

        items: list[AutocompleteSuggestion] = []
        try:
            entries = sorted(os.listdir(search_dir))
        except OSError:
            return AutocompleteSuggestions(prefix=prefix, replace_start=start_pos, replace_end=end_pos)

        for name in entries:
            if name.startswith(".") and not partial.startswith("."):
                continue
            if partial and not name.lower().startswith(partial.lower()):
                continue

            full_path = os.path.join(search_dir, name)
            is_dir = os.path.isdir(full_path)

            # Build the insert text from the original prefix
            if prefix.endswith("/"):
                insert = f"{prefix}{name}"
            else:
                dir_part = os.path.dirname(prefix)
                insert = f"{dir_part}/{name}" if dir_part else name

            if is_dir:
                insert += "/"

            items.append(
                AutocompleteSuggestion(
                    value=name,
                    label=f"{name}/" if is_dir else name,
                    description="directory" if is_dir else "file",
                    insert_text=insert,
                    kind="directory" if is_dir else "file",
                )
            )

        return AutocompleteSuggestions(
            items=items,
            prefix=prefix,
            replace_start=start_pos,
            replace_end=end_pos,
        )


# ---------------------------------------------------------------------------
# Combined provider
# ---------------------------------------------------------------------------


class CombinedAutocompleteProvider:
    """Merges suggestions from multiple providers.

    Queries each provider in order; returns the first non-None result.
    """

    def __init__(self, providers: list[Any] | None = None) -> None:
        self._providers: list[Any] = list(providers or [])

    def add_provider(self, provider: Any) -> None:
        self._providers.append(provider)

    async def get_suggestions(
        self,
        text: str,
        cursor: int,
        *,
        force: bool = False,
    ) -> AutocompleteSuggestions | None:
        for provider in self._providers:
            result = await provider.get_suggestions(text, cursor, force=force)
            if result is not None and result.items:
                return result
        return None


__all__ = [
    "AutocompleteProvider",
    "AutocompleteSuggestion",
    "AutocompleteSuggestions",
    "CombinedAutocompleteProvider",
    "FileAutocompleteProvider",
]
