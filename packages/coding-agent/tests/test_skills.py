"""Tests for ``nu_coding_agent.core.skills``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nu_coding_agent.core.skills import (
    LoadSkillsFromDirOptions,
    LoadSkillsOptions,
    Skill,
    format_skills_for_prompt,
    load_skills,
    load_skills_from_dir,
)
from nu_coding_agent.core.source_info import SourceInfo

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(parent: Path, name: str, *, description: str = "Demo skill") -> Path:
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\nbody for {name}")
    return skill_file


# ---------------------------------------------------------------------------
# load_skills_from_dir
# ---------------------------------------------------------------------------


def test_load_skills_finds_skill_md(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert len(result.skills) == 1
    assert result.skills[0].name == "alpha"
    assert result.diagnostics == []


def test_load_skills_recurses_into_subdirs(tmp_path: Path) -> None:
    _write_skill(tmp_path / "nested", "bravo")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert any(s.name == "bravo" for s in result.skills)


def test_load_skills_treats_skill_md_as_root(tmp_path: Path) -> None:
    """When a directory contains SKILL.md, sub-md files must be ignored."""
    skill_dir = tmp_path / "rooty"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: rooty\ndescription: top level\n---\n\nbody")
    nested = skill_dir / "subskills"
    nested.mkdir()
    (nested / "should-not-load.md").write_text("---\nname: should-not-load\ndescription: hidden\n---\n\nbody")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    names = {s.name for s in result.skills}
    assert names == {"rooty"}


def test_load_skills_md_root_files(tmp_path: Path) -> None:
    """When the root dir has no SKILL.md, plain ``.md`` direct children are loaded.

    The validator requires the file basename to match the parent directory
    name, so we name the file after ``tmp_path``'s last segment.
    """
    parent_name = tmp_path.name
    (tmp_path / f"{parent_name}.md").write_text(f"---\nname: {parent_name}\ndescription: flat description\n---\n")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert any(s.name == parent_name for s in result.skills)


def test_missing_directory_returns_empty(tmp_path: Path) -> None:
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path / "no-such"), source="user"))
    assert result.skills == []
    assert result.diagnostics == []


def test_missing_description_skipped_with_diagnostic(tmp_path: Path) -> None:
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "SKILL.md").write_text("---\nname: bad\n---\n\nno description here")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert result.skills == []
    assert any("description is required" in d.message for d in result.diagnostics)


def test_invalid_name_emits_warning(tmp_path: Path) -> None:
    weird = tmp_path / "Weird_Name"
    weird.mkdir()
    (weird / "SKILL.md").write_text("---\nname: Weird_Name\ndescription: ok\n---\n")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert any("invalid characters" in d.message for d in result.diagnostics)


def test_long_description_emits_warning(tmp_path: Path) -> None:
    huge = tmp_path / "huge"
    huge.mkdir()
    desc = "x" * 2000
    (huge / "SKILL.md").write_text(f"---\nname: huge\ndescription: {desc}\n---\n")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert any("exceeds" in d.message and "description" in d.message for d in result.diagnostics)


def test_disable_model_invocation_flag(tmp_path: Path) -> None:
    skill_dir = tmp_path / "stealth"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: stealth\ndescription: hidden\ndisable-model-invocation: true\n---\n"
    )
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert len(result.skills) == 1
    assert result.skills[0].disable_model_invocation is True


def test_dotfiles_skipped(tmp_path: Path) -> None:
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "SKILL.md").write_text("---\nname: hidden\ndescription: x\n---\n")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert result.skills == []


def test_node_modules_skipped(tmp_path: Path) -> None:
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "SKILL.md").write_text("---\nname: pkg\ndescription: x\n---\n")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    assert result.skills == []


def test_gitignore_filters_files(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored-skill/\n")
    _write_skill(tmp_path, "ignored-skill")
    _write_skill(tmp_path, "kept-skill")
    result = load_skills_from_dir(LoadSkillsFromDirOptions(dir=str(tmp_path), source="user"))
    names = {s.name for s in result.skills}
    assert names == {"kept-skill"}


# ---------------------------------------------------------------------------
# load_skills (top-level orchestrator)
# ---------------------------------------------------------------------------


def test_load_skills_with_explicit_paths(tmp_path: Path) -> None:
    skills_dir = tmp_path / "user-skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "explicit")
    result = load_skills(
        LoadSkillsOptions(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "no-agent"),
            skill_paths=[str(skills_dir)],
            include_defaults=False,
        )
    )
    assert any(s.name == "explicit" for s in result.skills)


def test_load_skills_handles_missing_path(tmp_path: Path) -> None:
    result = load_skills(
        LoadSkillsOptions(
            cwd=str(tmp_path),
            skill_paths=[str(tmp_path / "missing")],
            include_defaults=False,
        )
    )
    assert any("does not exist" in d.message for d in result.diagnostics)


def test_load_skills_explicit_md_file(tmp_path: Path) -> None:
    md = tmp_path / "single-skill.md"
    md.write_text("---\nname: single-skill\ndescription: solo\n---\n")
    result = load_skills(
        LoadSkillsOptions(
            cwd=str(tmp_path),
            skill_paths=[str(md)],
            include_defaults=False,
        )
    )
    assert any(s.name == "single-skill" for s in result.skills)


def test_load_skills_explicit_non_md_path(tmp_path: Path) -> None:
    txt = tmp_path / "not-a-skill.txt"
    txt.write_text("hi")
    result = load_skills(
        LoadSkillsOptions(
            cwd=str(tmp_path),
            skill_paths=[str(txt)],
            include_defaults=False,
        )
    )
    assert any("not a markdown file" in d.message for d in result.diagnostics)


def test_collision_reported(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_skill(a, "shared", description="from a")
    _write_skill(b, "shared", description="from b")
    result = load_skills(
        LoadSkillsOptions(
            cwd=str(tmp_path),
            skill_paths=[str(a), str(b)],
            include_defaults=False,
        )
    )
    assert any(d.type == "collision" for d in result.diagnostics)
    # Winner is whichever was loaded first.
    names = {s.name for s in result.skills}
    assert names == {"shared"}


def test_default_load_uses_user_and_project_dirs(tmp_path: Path) -> None:
    project_skills = tmp_path / ".nu" / "skills"
    project_skills.mkdir(parents=True)
    _write_skill(project_skills, "project-skill")
    result = load_skills(
        LoadSkillsOptions(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "no-agent"),
            include_defaults=True,
        )
    )
    assert any(s.name == "project-skill" for s in result.skills)


def test_load_skills_tilde_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    user = tmp_path / "skills"
    user.mkdir()
    _write_skill(user, "tilde-loaded")
    result = load_skills(
        LoadSkillsOptions(
            cwd=str(tmp_path),
            skill_paths=["~/skills"],
            include_defaults=False,
        )
    )
    assert any(s.name == "tilde-loaded" for s in result.skills)


# ---------------------------------------------------------------------------
# format_skills_for_prompt
# ---------------------------------------------------------------------------


def _make_skill(
    name: str,
    description: str = "demo",
    *,
    disable: bool = False,
    file_path: str = "/tmp/skill.md",
) -> Skill:
    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir="/tmp",
        source_info=SourceInfo(path=file_path, source="local", scope="user", origin="top-level"),
        disable_model_invocation=disable,
    )


def test_format_skills_empty_list_returns_empty_string() -> None:
    assert format_skills_for_prompt([]) == ""


def test_format_skills_includes_visible_only() -> None:
    out = format_skills_for_prompt(
        [
            _make_skill("alpha"),
            _make_skill("hidden", disable=True),
        ]
    )
    assert "alpha" in out
    assert "hidden" not in out


def test_format_skills_xml_escaping() -> None:
    out = format_skills_for_prompt([_make_skill("a&b", description="<go> & 'quoted'")])
    assert "&amp;" in out
    assert "&lt;" in out
    assert "&apos;" in out


def test_format_skills_all_hidden_returns_empty_string() -> None:
    assert format_skills_for_prompt([_make_skill("alpha", disable=True)]) == ""
