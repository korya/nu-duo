"""Fuzzy matching utilities.

Direct port of ``packages/tui/src/fuzzy.ts``. Matches if all query
characters appear in order (not necessarily consecutive). Lower score
means a better match — :func:`fuzzy_filter` sorts ascending.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class FuzzyMatch:
    matches: bool
    score: float


_WORD_BOUNDARY = re.compile(r"[\s\-_./:]")
_ALPHA_NUMERIC = re.compile(r"^(?P<letters>[a-z]+)(?P<digits>[0-9]+)$")
_NUMERIC_ALPHA = re.compile(r"^(?P<digits>[0-9]+)(?P<letters>[a-z]+)$")


def _match_query(normalized_query: str, text_lower: str) -> FuzzyMatch:
    if len(normalized_query) == 0:
        return FuzzyMatch(matches=True, score=0)
    if len(normalized_query) > len(text_lower):
        return FuzzyMatch(matches=False, score=0)

    query_index = 0
    score: float = 0
    last_match_index = -1
    consecutive_matches = 0

    for i, ch in enumerate(text_lower):
        if query_index >= len(normalized_query):
            break
        if ch == normalized_query[query_index]:
            is_word_boundary = i == 0 or _WORD_BOUNDARY.match(text_lower[i - 1]) is not None

            if last_match_index == i - 1:
                consecutive_matches += 1
                score -= consecutive_matches * 5
            else:
                consecutive_matches = 0
                if last_match_index >= 0:
                    score += (i - last_match_index - 1) * 2

            if is_word_boundary:
                score -= 10

            score += i * 0.1

            last_match_index = i
            query_index += 1

    if query_index < len(normalized_query):
        return FuzzyMatch(matches=False, score=0)

    return FuzzyMatch(matches=True, score=score)


def fuzzy_match(query: str, text: str) -> FuzzyMatch:
    """Score how well ``query`` matches ``text``.

    Returns :class:`FuzzyMatch` with ``matches`` and ``score``. Lower
    scores are better. Falls back to a swapped alphanumeric query
    (``"abc12"`` ↔ ``"12abc"``) if the primary query fails.
    """
    query_lower = query.lower()
    text_lower = text.lower()

    primary = _match_query(query_lower, text_lower)
    if primary.matches:
        return primary

    alpha_numeric = _ALPHA_NUMERIC.match(query_lower)
    numeric_alpha = _NUMERIC_ALPHA.match(query_lower)
    if alpha_numeric is not None:
        swapped_query = f"{alpha_numeric.group('digits')}{alpha_numeric.group('letters')}"
    elif numeric_alpha is not None:
        swapped_query = f"{numeric_alpha.group('letters')}{numeric_alpha.group('digits')}"
    else:
        return primary

    swapped = _match_query(swapped_query, text_lower)
    if not swapped.matches:
        return primary
    return FuzzyMatch(matches=True, score=swapped.score + 5)


def fuzzy_filter[T](items: list[T], query: str, get_text: Callable[[T], str]) -> list[T]:
    """Filter and sort items by fuzzy match quality (best first).

    Supports space-separated tokens — every token must match the item's
    text for it to be included. Items are returned sorted by total score
    (lower is better).
    """
    if not query.strip():
        return list(items)
    tokens = [t for t in query.strip().split() if t]
    if not tokens:
        return list(items)

    results: list[tuple[T, float]] = []
    for item in items:
        text = get_text(item)
        total_score: float = 0
        all_match = True
        for token in tokens:
            match = fuzzy_match(token, text)
            if match.matches:
                total_score += match.score
            else:
                all_match = False
                break
        if all_match:
            results.append((item, total_score))

    results.sort(key=lambda r: r[1])
    return [item for item, _ in results]


__all__ = ["FuzzyMatch", "fuzzy_filter", "fuzzy_match"]
