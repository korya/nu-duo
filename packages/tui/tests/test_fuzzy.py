"""Tests for nu_tui.fuzzy.

Port of ``packages/tui/src/fuzzy.ts``. Verifies the scoring contract:

* Lower score = better match.
* Consecutive matches reward, gaps penalize.
* Word-boundary matches receive a bonus.
* Empty query matches everything (score 0).
* alpha-numeric / numeric-alpha swap fallback when the primary query fails.
* :func:`fuzzy_filter` keeps original order on equal scores and removes
  non-matching items.
"""

from __future__ import annotations

from nu_tui.fuzzy import fuzzy_filter, fuzzy_match


class TestFuzzyMatch:
    def test_empty_query_matches(self) -> None:
        result = fuzzy_match("", "anything")
        assert result.matches is True
        assert result.score == 0

    def test_query_longer_than_text_does_not_match(self) -> None:
        result = fuzzy_match("longerquery", "short")
        assert result.matches is False

    def test_exact_substring_match(self) -> None:
        result = fuzzy_match("abc", "abc")
        assert result.matches is True
        # Three consecutive matches with word-boundary bonus on the first.
        assert result.score < 0

    def test_non_matching_returns_no_match(self) -> None:
        result = fuzzy_match("xyz", "abc")
        assert result.matches is False

    def test_case_insensitive_match(self) -> None:
        result = fuzzy_match("ABC", "abcdef")
        assert result.matches is True

    def test_consecutive_matches_score_better_than_gappy(self) -> None:
        consecutive = fuzzy_match("foo", "foobar")
        gappy = fuzzy_match("foo", "f_o_o_bar")
        assert consecutive.matches is True
        assert gappy.matches is True
        assert consecutive.score < gappy.score

    def test_word_boundary_matches_score_better(self) -> None:
        boundary = fuzzy_match("py", "the python")  # 'p' after space
        middle = fuzzy_match("py", "happy day")  # 'p' mid-word
        assert boundary.score < middle.score

    def test_swapped_alphanumeric_fallback_matches(self) -> None:
        # "12abc" doesn't appear directly but swap-fallback tries "abc12".
        result = fuzzy_match("12abc", "abc12")
        assert result.matches is True


class TestFuzzyFilter:
    def test_empty_query_returns_input_unchanged(self) -> None:
        items = ["foo", "bar", "baz"]
        assert fuzzy_filter(items, "", lambda s: s) == items

    def test_filters_non_matching(self) -> None:
        items = ["foo.py", "bar.txt", "baz.py"]
        result = fuzzy_filter(items, "py", lambda s: s)
        assert "bar.txt" not in result
        assert "foo.py" in result
        assert "baz.py" in result

    def test_sorts_by_score_best_first(self) -> None:
        items = ["fubar", "foobar"]  # "foobar" has consecutive 'foo'
        result = fuzzy_filter(items, "foo", lambda s: s)
        assert result[0] == "foobar"

    def test_multi_token_query_requires_all_match(self) -> None:
        items = ["foo bar baz", "foo qux", "bar baz"]
        result = fuzzy_filter(items, "foo bar", lambda s: s)
        assert "foo bar baz" in result
        assert "foo qux" not in result
        assert "bar baz" not in result

    def test_filter_with_getter(self) -> None:
        items = [{"name": "alpha"}, {"name": "beta"}]
        result = fuzzy_filter(items, "al", lambda d: d["name"])
        assert result == [{"name": "alpha"}]

    def test_whitespace_only_query_returns_input(self) -> None:
        items = ["a", "b"]
        assert fuzzy_filter(items, "   ", lambda s: s) == items
