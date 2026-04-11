"""Tests for ``nu_coding_agent.core.prompt_templates``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nu_coding_agent.core.prompt_templates import (
    LoadPromptTemplatesOptions,
    expand_prompt_template,
    load_prompt_templates,
    parse_command_args,
    substitute_args,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# parse_command_args
# ---------------------------------------------------------------------------


def test_parse_command_args_simple() -> None:
    assert parse_command_args("foo bar baz") == ["foo", "bar", "baz"]


def test_parse_command_args_double_quoted() -> None:
    assert parse_command_args('"hello world" bar') == ["hello world", "bar"]


def test_parse_command_args_single_quoted() -> None:
    assert parse_command_args("'hi there'") == ["hi there"]


def test_parse_command_args_empty() -> None:
    assert parse_command_args("") == []


def test_parse_command_args_tabs_and_spaces() -> None:
    assert parse_command_args("a\tb  c") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# substitute_args
# ---------------------------------------------------------------------------


def test_substitute_positional() -> None:
    assert substitute_args("$1 and $2", ["foo", "bar"]) == "foo and bar"


def test_substitute_missing_positional_is_empty() -> None:
    assert substitute_args("$1-$2-$3", ["a"]) == "a--"


def test_substitute_arguments_keyword() -> None:
    assert substitute_args("$ARGUMENTS", ["a", "b", "c"]) == "a b c"


def test_substitute_dollar_at() -> None:
    assert substitute_args("Got $@", ["x", "y"]) == "Got x y"


def test_substitute_slice_from_n() -> None:
    assert substitute_args("${@:2}", ["one", "two", "three"]) == "two three"


def test_substitute_slice_from_n_with_length() -> None:
    assert substitute_args("${@:2:1}", ["one", "two", "three"]) == "two"


def test_positional_substituted_before_wildcard() -> None:
    # If $1 = "$ARGUMENTS", the wildcard must NOT be re-expanded.
    assert substitute_args("$1", ["$ARGUMENTS"]) == "$ARGUMENTS"


# ---------------------------------------------------------------------------
# load_prompt_templates / expand_prompt_template
# ---------------------------------------------------------------------------


def _make_prompt(dir_path: Path, name: str, *, body: str = "Hi $1!", description: str | None = "Greet") -> None:
    if description is None:
        dir_path.joinpath(f"{name}.md").write_text(body)
    else:
        dir_path.joinpath(f"{name}.md").write_text(f"---\ndescription: {description}\n---\n{body}")


def test_load_templates_from_explicit_paths(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    _make_prompt(prompts, "greet", body="Hello $1", description="Say hi")

    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(prompts)],
            include_defaults=False,
        )
    )
    assert len(templates) == 1
    template = templates[0]
    assert template.name == "greet"
    assert template.description == "Say hi"
    assert template.content == "Hello $1"


def test_description_from_first_body_line(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "x.md").write_text("First line is description\nbody body body\n")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(prompts)],
            include_defaults=False,
        )
    )
    assert templates[0].description == "First line is description"


def test_description_truncated_at_60_chars(tmp_path: Path) -> None:
    prompts = tmp_path / "p"
    prompts.mkdir()
    long_line = "a" * 80
    (prompts / "x.md").write_text(long_line)
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(prompts)],
            include_defaults=False,
        )
    )
    assert templates[0].description == "a" * 60 + "..."


def test_load_templates_from_single_file(tmp_path: Path) -> None:
    file = tmp_path / "single.md"
    file.write_text("---\ndescription: solo\n---\nhello")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(file)],
            include_defaults=False,
        )
    )
    assert len(templates) == 1
    assert templates[0].name == "single"


def test_load_templates_skips_missing_paths(tmp_path: Path) -> None:
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(tmp_path / "missing")],
            include_defaults=False,
        )
    )
    assert templates == []


def test_load_defaults_uses_project_dir(tmp_path: Path) -> None:
    project_prompts = tmp_path / ".nu" / "prompts"
    project_prompts.mkdir(parents=True)
    _make_prompt(project_prompts, "p1")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "no-agent"),  # nonexistent → no global templates
            include_defaults=True,
        )
    )
    names = {t.name for t in templates}
    assert "p1" in names


def test_expand_prompt_template_match(tmp_path: Path) -> None:
    prompts = tmp_path / "p"
    prompts.mkdir()
    _make_prompt(prompts, "greet", body="Hi $1!")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(prompts)],
            include_defaults=False,
        )
    )
    assert expand_prompt_template("/greet world", templates) == "Hi world!"


def test_expand_prompt_template_no_match() -> None:
    assert expand_prompt_template("/unknown", []) == "/unknown"


def test_expand_prompt_template_non_slash_passes_through() -> None:
    assert expand_prompt_template("just text", []) == "just text"


def test_load_template_skips_non_md(tmp_path: Path) -> None:
    prompts = tmp_path / "p"
    prompts.mkdir()
    (prompts / "ignored.txt").write_text("not markdown")
    (prompts / "real.md").write_text("hi")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(prompts)],
            include_defaults=False,
        )
    )
    names = {t.name for t in templates}
    assert names == {"real"}


def test_load_template_handles_directory_in_prompt_paths(tmp_path: Path) -> None:
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "x.md").write_text("hi")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(inner)],
            include_defaults=False,
        )
    )
    assert len(templates) == 1


def test_load_template_skips_non_md_explicit_path(tmp_path: Path) -> None:
    file = tmp_path / "x.txt"
    file.write_text("hi")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(file)],
            include_defaults=False,
        )
    )
    assert templates == []


def test_description_skips_blank_first_lines(tmp_path: Path) -> None:
    prompts = tmp_path / "p"
    prompts.mkdir()
    (prompts / "x.md").write_text("\n\n  \nactual content here\n")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(prompts)],
            include_defaults=False,
        )
    )
    assert templates[0].description == "actual content here"


def test_description_with_non_string_frontmatter(tmp_path: Path) -> None:
    prompts = tmp_path / "p"
    prompts.mkdir()
    (prompts / "x.md").write_text("---\ndescription: 123\n---\nfallback line\n")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=[str(prompts)],
            include_defaults=False,
        )
    )
    # Non-string frontmatter description falls through to first body line.
    assert templates[0].description == "fallback line"


def test_load_with_explicit_tilde_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "templates"
    home.mkdir()
    (home / "h.md").write_text("hello")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=["~/templates"],
            include_defaults=False,
        )
    )
    assert {t.name for t in templates} == {"h"}


def test_load_with_bare_tilde_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "h.md").write_text("hi")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=["~"],
            include_defaults=False,
        )
    )
    assert {t.name for t in templates} == {"h"}


def test_load_with_tilde_no_slash_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "x.md").write_text("hi")
    templates = load_prompt_templates(
        LoadPromptTemplatesOptions(
            cwd=str(tmp_path),
            prompt_paths=["~sub"],
            include_defaults=False,
        )
    )
    assert {t.name for t in templates} == {"x"}


def test_default_options() -> None:
    # Calling with no options should not crash; returns a list (possibly empty).
    templates = load_prompt_templates()
    assert isinstance(templates, list)
