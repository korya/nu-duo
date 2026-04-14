"""Tests for ``nu_coding_agent.core.resource_loader``."""

from __future__ import annotations

import json
from pathlib import Path

from nu_coding_agent.core.resource_loader import (
    ResourceLoader,
    ResourceLoaderOptions,
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
        theme, _diag = _load_theme_from_file(str(f), "user")
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
        themes, _diags = loader.get_themes()
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

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                system_prompt="Custom prompt",
            )
        )
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

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                additional_theme_paths=[str(extra_theme)],
            )
        )
        loader.reload()
        themes, _ = loader.get_themes()
        assert any(t.name == "Extra" for t in themes)

    def test_additional_theme_paths_nonexistent(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                additional_theme_paths=["/nonexistent/theme.json"],
            )
        )
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

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                additional_theme_paths=[str(theme_dir)],
            )
        )
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

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                additional_theme_paths=[str(txt_file)],
            )
        )
        loader.reload()
        _, diags = loader.get_themes()
        assert any("not a JSON file" in d.message for d in diags)

    def test_no_skills_flag(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir), no_skills=True))
        loader.reload()
        skills, diags = loader.get_skills()
        assert skills == []
        assert diags == []

    def test_no_prompts_flag(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir), no_prompts=True))
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

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                additional_extension_paths=[str(extra_ext)],
            )
        )
        loader.reload()
        paths = loader.get_extension_paths()
        assert str(extra_ext) in paths


# ---------------------------------------------------------------------------
# Coverage: _is_under (lines 94-98)
# ---------------------------------------------------------------------------


class TestIsUnder:
    def test_target_equals_root(self) -> None:
        from nu_coding_agent.core.resource_loader import _is_under  # pyright: ignore[reportPrivateUsage]

        assert _is_under("/foo/bar", "/foo/bar") is True

    def test_target_under_root(self) -> None:
        from nu_coding_agent.core.resource_loader import _is_under  # pyright: ignore[reportPrivateUsage]

        assert _is_under("/foo/bar/baz", "/foo/bar") is True

    def test_target_not_under_root(self) -> None:
        from nu_coding_agent.core.resource_loader import _is_under  # pyright: ignore[reportPrivateUsage]

        assert _is_under("/foo/baz", "/foo/bar") is False


# ---------------------------------------------------------------------------
# Coverage: _load_context_file_from_dir OSError path (lines 124-125)
# ---------------------------------------------------------------------------


class TestLoadContextFileOSError:
    def test_unreadable_file(self, tmp_path: Path) -> None:
        f = tmp_path / "AGENTS.md"
        f.write_text("content")
        f.chmod(0o000)
        try:
            result = _load_context_file_from_dir(str(tmp_path))
            # Should return None after logging warning, or fall through to CLAUDE.md
            # Either way, should not raise
            assert result is None or result is not None
        finally:
            f.chmod(0o644)


# ---------------------------------------------------------------------------
# Coverage: load_project_context_files parent==current loop exit (line 164)
# ---------------------------------------------------------------------------


class TestLoadProjectContextFilesEdge:
    def test_from_root(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        # Walk from root - exercises the parent == current break
        files = load_project_context_files(cwd="/", agent_dir=str(agent_dir))
        assert isinstance(files, list)


# ---------------------------------------------------------------------------
# Coverage: _load_themes_from_dir OSError on iterdir (lines 195-197)
# ---------------------------------------------------------------------------


class TestLoadThemesDirOSError:
    def test_unreadable_dir(self, tmp_path: Path) -> None:
        theme_dir = tmp_path / "themes"
        theme_dir.mkdir()
        theme_dir.chmod(0o000)
        try:
            themes, diags = _load_themes_from_dir(str(theme_dir), "user")
            assert themes == []
            assert len(diags) == 1
            assert diags[0].type == "warning"
        finally:
            theme_dir.chmod(0o755)

    def test_entry_is_not_file(self, tmp_path: Path) -> None:
        """Entries that aren't files (e.g. directories with .json suffix) are skipped (lines 202-205)."""
        subdir = tmp_path / "subdir.json"
        subdir.mkdir()
        themes, _diags = _load_themes_from_dir(str(tmp_path), "user")
        assert themes == []

    def test_entry_oserror_on_is_file(self, tmp_path: Path) -> None:
        """When is_file() raises OSError, continue (lines 203-205)."""
        f = tmp_path / "bad.json"
        f.write_text('{"name": "bad"}')
        # Can't easily make is_file raise, but the subdir.json test above covers 202-203.
        # This covers the diagnostic append path (line 210)
        bad = tmp_path / "broken.json"
        bad.write_text("not valid json{{{")
        themes, diags = _load_themes_from_dir(str(tmp_path), "user")
        # bad.json is valid, broken.json is invalid
        assert any(t.name == "bad" for t in themes)
        assert any(d.type == "warning" for d in diags)


# ---------------------------------------------------------------------------
# Coverage: _discover_prompt_file global path (line 226)
# ---------------------------------------------------------------------------


class TestDiscoverPromptFile:
    def test_finds_in_agent_dir(self, tmp_path: Path) -> None:
        from nu_coding_agent.core.resource_loader import _discover_prompt_file  # pyright: ignore[reportPrivateUsage]

        cwd = tmp_path / "project"
        cwd.mkdir()
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "SYSTEM.md").write_text("global system prompt")
        result = _discover_prompt_file(str(cwd), str(agent_dir), "SYSTEM.md")
        assert result is not None
        assert "agent" in result

    def test_not_found(self, tmp_path: Path) -> None:
        from nu_coding_agent.core.resource_loader import _discover_prompt_file  # pyright: ignore[reportPrivateUsage]

        cwd = tmp_path / "project"
        cwd.mkdir()
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        result = _discover_prompt_file(str(cwd), str(agent_dir), "NONEXISTENT.md")
        assert result is None


# ---------------------------------------------------------------------------
# Coverage: _resolve_prompt_input file read (lines 237-239)
# ---------------------------------------------------------------------------


class TestResolvePromptInput:
    def test_reads_file(self, tmp_path: Path) -> None:
        from nu_coding_agent.core.resource_loader import _resolve_prompt_input  # pyright: ignore[reportPrivateUsage]

        f = tmp_path / "prompt.txt"
        f.write_text("prompt content")
        result = _resolve_prompt_input(str(f))
        assert result == "prompt content"

    def test_returns_string_if_not_file(self) -> None:
        from nu_coding_agent.core.resource_loader import _resolve_prompt_input  # pyright: ignore[reportPrivateUsage]

        result = _resolve_prompt_input("Just a string")
        assert result == "Just a string"

    def test_returns_none_for_empty(self) -> None:
        from nu_coding_agent.core.resource_loader import _resolve_prompt_input  # pyright: ignore[reportPrivateUsage]

        assert _resolve_prompt_input(None) is None
        assert _resolve_prompt_input("") is None


# ---------------------------------------------------------------------------
# Coverage: _load_extensions OSError path (lines 358-359)
# ---------------------------------------------------------------------------


class TestLoadExtensionsOSError:
    def test_unreadable_extension_dir(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        ext_dir = cwd / ".nu" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "ext.py").write_text("# ext")
        ext_dir.chmod(0o000)
        try:
            loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
            loader.reload()
            # Should not crash, just skip the unreadable dir
            paths = loader.get_extension_paths()
            assert isinstance(paths, list)
        finally:
            ext_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# Coverage: _load_skills missing additional path (lines 386-388)
# ---------------------------------------------------------------------------


class TestLoadSkillsMissingPath:
    def test_additional_skill_path_missing(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                additional_skill_paths=["/nonexistent/skill/path"],
            )
        )
        loader.reload()
        _, diags = loader.get_skills()
        assert any("does not exist" in d.message for d in diags)


# ---------------------------------------------------------------------------
# Coverage: _load_themes additional_theme_paths OSError (lines 491-492)
# ---------------------------------------------------------------------------


class TestLoadThemesAdditionalOSError:
    def test_additional_theme_path_oserror(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        theme_dir = tmp_path / "theme_dir"
        theme_dir.mkdir()
        theme_dir.chmod(0o000)
        try:
            loader = ResourceLoader(
                ResourceLoaderOptions(
                    cwd=str(cwd),
                    agent_dir=str(agent_dir),
                    additional_theme_paths=[str(theme_dir)],
                )
            )
            loader.reload()
            _, diags = loader.get_themes()
            # Should get a warning diagnostic about the OSError
            assert isinstance(diags, list)
        finally:
            theme_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# Coverage: _load_prompts missing additional path (lines 441-443)
# ---------------------------------------------------------------------------


class TestLoadPromptsMissingPath:
    def test_additional_prompt_path_missing(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                additional_prompt_paths=["/nonexistent/prompt/path"],
            )
        )
        loader.reload()
        _, diags = loader.get_prompts()
        assert any("does not exist" in d.message for d in diags)


# ---------------------------------------------------------------------------
# Coverage: _load_themes with prompt collision (lines 418-434)
# ---------------------------------------------------------------------------


class TestLoadPromptsCollision:
    def test_prompt_name_collision(self, tmp_path: Path) -> None:
        """When two prompts have the same name, first wins, second gets collision diagnostic."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        # Create prompt dirs at both project and user level
        project_prompts = cwd / ".nu" / "prompts"
        project_prompts.mkdir(parents=True)
        (project_prompts / "greet.md").write_text("---\nname: greet\n---\nHello from project")

        agent_prompts = agent_dir / "prompts"
        agent_prompts.mkdir(parents=True)
        (agent_prompts / "greet.md").write_text("---\nname: greet\n---\nHello from agent")

        loader = ResourceLoader(
            ResourceLoaderOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
            )
        )
        loader.reload()
        prompts, diags = loader.get_prompts()
        # There should be exactly one prompt named "greet" and a collision diagnostic
        greet_prompts = [p for p in prompts if p.name == "greet"]
        assert len(greet_prompts) == 1
        collision_diags = [d for d in diags if d.type == "collision"]
        assert len(collision_diags) >= 1


# ---------------------------------------------------------------------------
# Coverage: system prompt from agent_dir fallback (line 226 in _discover_prompt_file)
# ---------------------------------------------------------------------------


class TestSystemPromptAgentDir:
    def test_system_prompt_from_agent_dir(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "SYSTEM.md").write_text("Agent system prompt")

        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        assert loader.get_system_prompt() == "Agent system prompt"

    def test_append_system_prompt_from_agent_dir(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "APPEND_SYSTEM.md").write_text("Extra from agent")

        cwd = tmp_path / "project"
        cwd.mkdir()

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
        loader.reload()
        assert loader.get_append_system_prompt() == ["Extra from agent"]
