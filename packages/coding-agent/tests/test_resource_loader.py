"""Tests for ``nu_coding_agent.core.resource_loader``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nu_coding_agent.core.resource_loader import (
    ContextFile,
    ResourceLoader,
    ResourceLoaderOptions,
    ThemeInfo,
    _expand_tilde,
    _load_context_file_from_dir,
    _load_theme_from_file,
    _load_themes_from_dir,
    _merge_paths,
    _resolve_path,
    load_project_context_files,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestExpandTilde:
    def test_just_tilde(self) -> None:
        assert _expand_tilde("~") == str(Path.home())

    def test_tilde_slash(self) -> None:
        assert _expand_tilde("~/foo") == str(Path.home() / "foo")

    def test_tilde_no_slash(self) -> None:
        # "~bar" -> home / "bar"
        assert _expand_tilde("~bar") == str(Path.home() / "bar")

    def test_absolute(self) -> None:
        assert _expand_tilde("/absolute/path") == "/absolute/path"

    def test_strips_whitespace(self) -> None:
        assert _expand_tilde("  /foo  ") == "/foo"


class TestResolvePath:
    def test_absolute(self) -> None:
        assert _resolve_path("/abs/path", "/cwd") == "/abs/path"

    def test_relative(self) -> None:
        result = _resolve_path("rel/path", "/cwd")
        assert result == str(Path("/cwd/rel/path").resolve())


class TestMergePaths:
    def test_deduplicates(self) -> None:
        result = _merge_paths(["/a", "/b"], ["/b", "/c"], "/cwd")
        assert result == ["/a", "/b", "/c"]


# ---------------------------------------------------------------------------
# Context file loading
# ---------------------------------------------------------------------------


class TestLoadContextFileFromDir:
    def test_agents_md(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("# Agent info")
        result = _load_context_file_from_dir(str(tmp_path))
        assert result is not None
        assert result.content == "# Agent info"

    def test_claude_md(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Claude info")
        result = _load_context_file_from_dir(str(tmp_path))
        assert result is not None
        assert result.content == "# Claude info"

    def test_agents_md_preferred(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("agents")
        (tmp_path / "CLAUDE.md").write_text("claude")
        result = _load_context_file_from_dir(str(tmp_path))
        assert result is not None
        assert result.content == "agents"

    def test_no_file(self, tmp_path: Path) -> None:
        result = _load_context_file_from_dir(str(tmp_path))
        assert result is None


class TestLoadProjectContextFiles:
    def test_walks_up(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "AGENTS.md").write_text("global")

        project = tmp_path / "project"
        project.mkdir()
        (project / "AGENTS.md").write_text("project")

        subdir = project / "sub"
        subdir.mkdir()

        files = load_project_context_files(cwd=str(subdir), agent_dir=str(agent_dir))
        # global first, then ancestors from root down to cwd
        assert len(files) >= 2
        assert files[0].content == "global"
        # project AGENTS.md should be in there
        contents = [f.content for f in files]
        assert "project" in contents

    def test_empty(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        files = load_project_context_files(cwd=str(cwd), agent_dir=str(agent_dir))
        assert files == []


# ---------------------------------------------------------------------------
# Theme loading
# ---------------------------------------------------------------------------


class TestLoadThemeFromFile:
    def test_valid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "dark.json"
        f.write_text(json.dumps({"name": "Dark Mode", "colors": {}}))
        theme, diag = _load_theme_from_file(str(f), "user")
        assert theme is not None
        assert theme.name == "Dark Mode"
        assert diag is None

    def test_no_name_in_json(self, tmp_path: Path) -> None:
        f = tmp_path / "my-theme.json"
        f.write_text(json.dumps({"colors": {}}))
        theme, diag = _load_theme_from_file(str(f), "user")
        assert theme is not None
        assert theme.name == "my-theme"  # falls back to stem

    def test_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json{{{")
        theme, diag = _load_theme_from_file(str(f), "user")
        assert theme is None
        assert diag is not None
        assert diag.type == "warning"


class TestLoadThemesFromDir:
    def test_loads_all_json(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text(json.dumps({"name": "A"}))
        (tmp_path / "b.json").write_text(json.dumps({"name": "B"}))
        (tmp_path / "not_theme.txt").write_text("skip")

        themes, diags = _load_themes_from_dir(str(tmp_path), "user")
        assert len(themes) == 2
        assert diags == []

    def test_nonexistent_dir(self) -> None:
        themes, diags = _load_themes_from_dir("/nonexistent/dir", "user")
        assert themes == []
        assert diags == []


# ---------------------------------------------------------------------------
# ResourceLoader
# ---------------------------------------------------------------------------


class TestResourceLoader:
    def test_defaults(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        assert loader.get_extension_paths() == []
        assert loader.get_skills() == ([], [])
        assert loader.get_prompts() == ([], [])
        assert loader.get_themes() == ([], [])
        assert loader.get_agents_files() == []
        assert loader.get_system_prompt() is None
        assert loader.get_append_system_prompt() == []

    def test_get_extensions(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        ext_dir = cwd / ".nu" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "my_ext.py").write_text("# extension")
        (ext_dir / "readme.txt").write_text("not ext")  # should be ignored

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        paths = loader.get_extension_paths()
        assert len(paths) == 1
        assert paths[0].endswith("my_ext.py")

    def test_no_extensions_flag(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        ext_dir = cwd / ".nu" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "my_ext.py").write_text("# ext")

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir), no_extensions=True))
        loader.reload()
        assert loader.get_extension_paths() == []

    def test_get_themes(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        theme_dir = agent_dir / "themes"
        theme_dir.mkdir()
        (theme_dir / "cool.json").write_text(json.dumps({"name": "Cool"}))

        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        themes, diags = loader.get_themes()
        assert len(themes) == 1
        assert themes[0].name == "Cool"

    def test_no_themes_flag(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir), no_themes=True))
        loader.reload()
        themes, _ = loader.get_themes()
        assert themes == []

    def test_theme_collision(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        agent_themes = agent_dir / "themes"
        agent_themes.mkdir()
        (agent_themes / "mytheme.json").write_text(json.dumps({"name": "T"}))

        cwd = tmp_path / "project"
        cwd.mkdir()
        project_themes = cwd / ".nu" / "themes"
        project_themes.mkdir(parents=True)
        (project_themes / "mytheme.json").write_text(json.dumps({"name": "T"}))

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        themes, diags = loader.get_themes()
        # Project wins (loaded first), user is a collision
        assert len(themes) == 1
        collision_diags = [d for d in diags if d.type == "collision"]
        assert len(collision_diags) == 1

    def test_get_agents_files(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "AGENTS.md").write_text("global agent")

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AGENTS.md").write_text("project agent")

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        files = loader.get_agents_files()
        assert len(files) == 2
        assert files[0].content == "global agent"

    def test_system_prompt_from_file(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        nu_dir = cwd / ".nu"
        nu_dir.mkdir()
        (nu_dir / "SYSTEM.md").write_text("You are a helpful assistant.")

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        assert loader.get_system_prompt() == "You are a helpful assistant."

    def test_system_prompt_from_option(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            system_prompt="Custom prompt",
        ))
        loader.reload()
        assert loader.get_system_prompt() == "Custom prompt"

    def test_append_system_prompt(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        nu_dir = cwd / ".nu"
        nu_dir.mkdir()
        (nu_dir / "APPEND_SYSTEM.md").write_text("Extra instructions")

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        assert loader.get_append_system_prompt() == ["Extra instructions"]

    def test_additional_theme_paths_file(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        extra_theme = tmp_path / "extra.json"
        extra_theme.write_text(json.dumps({"name": "Extra"}))

        loader = ResourceLoader(ResourceLoaderOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            additional_theme_paths=[str(extra_theme)],
        ))
        loader.reload()
        themes, _ = loader.get_themes()
        assert any(t.name == "Extra" for t in themes)

    def test_additional_theme_paths_nonexistent(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            additional_theme_paths=["/nonexistent/theme.json"],
        ))
        loader.reload()
        _, diags = loader.get_themes()
        assert any("does not exist" in d.message for d in diags)

    def test_get_diagnostics(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        diags = loader.get_diagnostics()
        assert isinstance(diags, list)

    def test_additional_theme_paths_dir(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        theme_dir = tmp_path / "extra_themes"
        theme_dir.mkdir()
        (theme_dir / "dark.json").write_text(json.dumps({"name": "Dark"}))

        loader = ResourceLoader(ResourceLoaderOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            additional_theme_paths=[str(theme_dir)],
        ))
        loader.reload()
        themes, _ = loader.get_themes()
        assert any(t.name == "Dark" for t in themes)

    def test_additional_theme_paths_not_json(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        txt_file = tmp_path / "not_theme.txt"
        txt_file.write_text("not a theme")

        loader = ResourceLoader(ResourceLoaderOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            additional_theme_paths=[str(txt_file)],
        ))
        loader.reload()
        _, diags = loader.get_themes()
        assert any("not a JSON file" in d.message for d in diags)

    def test_no_skills_flag(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(
            cwd=str(cwd), agent_dir=str(agent_dir), no_skills=True
        ))
        loader.reload()
        skills, diags = loader.get_skills()
        assert skills == []
        assert diags == []

    def test_no_prompts_flag(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(
            cwd=str(cwd), agent_dir=str(agent_dir), no_prompts=True
        ))
        loader.reload()
        prompts, diags = loader.get_prompts()
        assert prompts == []
        assert diags == []

    def test_user_level_extensions(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        ext_dir = agent_dir / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "my_ext.py").write_text("# ext")

        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        paths = loader.get_extension_paths()
        assert any("my_ext.py" in p for p in paths)

    def test_additional_extension_paths(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        extra_ext = tmp_path / "my_ext.py"
        extra_ext.write_text("# ext")

        loader = ResourceLoader(ResourceLoaderOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            additional_extension_paths=[str(extra_ext)],
        ))
        loader.reload()
        paths = loader.get_extension_paths()
        assert str(extra_ext) in paths
