"""Tests for ``nu_tui.autocomplete``."""

from __future__ import annotations

from pathlib import Path

from nu_tui.autocomplete import (
    AutocompleteSuggestion,
    AutocompleteSuggestions,
    CombinedAutocompleteProvider,
    FileAutocompleteProvider,
)


async def test_file_autocomplete_lists_directory(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("")
    (tmp_path / "world.txt").write_text("")
    (tmp_path / "subdir").mkdir()

    provider = FileAutocompleteProvider(str(tmp_path))
    result = await provider.get_suggestions(f"{tmp_path}/", len(str(tmp_path)) + 1)
    assert result is not None
    names = [s.value for s in result.items]
    assert "hello.py" in names
    assert "world.txt" in names
    assert "subdir" in names


async def test_file_autocomplete_filters_by_prefix(tmp_path: Path) -> None:
    (tmp_path / "abc.py").write_text("")
    (tmp_path / "xyz.py").write_text("")

    provider = FileAutocompleteProvider(str(tmp_path))
    text = f"{tmp_path}/ab"
    result = await provider.get_suggestions(text, len(text))
    assert result is not None
    assert len(result.items) == 1
    assert result.items[0].value == "abc.py"


async def test_file_autocomplete_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "mydir").mkdir()
    provider = FileAutocompleteProvider(str(tmp_path))
    text = f"{tmp_path}/"
    result = await provider.get_suggestions(text, len(text))
    assert result is not None
    dir_item = next(s for s in result.items if s.value == "mydir")
    assert dir_item.kind == "directory"
    assert dir_item.label.endswith("/")


async def test_file_autocomplete_relative_path(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("")
    provider = FileAutocompleteProvider(str(tmp_path))
    result = await provider.get_suggestions("./", 2)
    assert result is not None
    assert any(s.value == "file.txt" for s in result.items)


async def test_file_autocomplete_no_trigger_returns_none(tmp_path: Path) -> None:
    provider = FileAutocompleteProvider(str(tmp_path))
    result = await provider.get_suggestions("hello world", 11)
    assert result is None


async def test_combined_provider_first_match_wins(tmp_path: Path) -> None:
    (tmp_path / "test.py").write_text("")

    class _FixedProvider:
        async def get_suggestions(
            self, text: str, cursor: int, *, force: bool = False
        ) -> AutocompleteSuggestions | None:
            return AutocompleteSuggestions(items=[AutocompleteSuggestion(value="fixed", label="fixed")])

    combined = CombinedAutocompleteProvider([_FixedProvider()])
    result = await combined.get_suggestions("anything", 8)
    assert result is not None
    assert result.items[0].value == "fixed"


async def test_combined_provider_falls_through_on_none() -> None:
    class _EmptyProvider:
        async def get_suggestions(self, text: str, cursor: int, *, force: bool = False) -> None:
            return None

    class _FallbackProvider:
        async def get_suggestions(
            self, text: str, cursor: int, *, force: bool = False
        ) -> AutocompleteSuggestions | None:
            return AutocompleteSuggestions(items=[AutocompleteSuggestion(value="fallback", label="fallback")])

    combined = CombinedAutocompleteProvider([_EmptyProvider(), _FallbackProvider()])
    result = await combined.get_suggestions("x", 1)
    assert result is not None
    assert result.items[0].value == "fallback"


async def test_combined_provider_empty_returns_none() -> None:
    combined = CombinedAutocompleteProvider([])
    result = await combined.get_suggestions("x", 1)
    assert result is None
