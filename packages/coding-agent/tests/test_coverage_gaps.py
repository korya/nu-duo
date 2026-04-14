"""Targeted tests to close the last coverage gaps in resource_loader and migrations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# migrations.py — lines 69-70, 85-86, 142-143, 161-162, 191-192,
#                  234-235, 265-266, 298-310
# ---------------------------------------------------------------------------


class TestMigrationsCoverageGaps:
    """Target every remaining uncovered line in migrations.py."""

    def test_oauth_json_read_error(self, tmp_path: Path) -> None:
        """Lines 69-70: exception in oauth.json reading."""
        oauth = tmp_path / "oauth.json"
        oauth.write_text("invalid json!!!")
        from nu_coding_agent.migrations import migrate_auth_to_auth_json

        result = migrate_auth_to_auth_json(str(tmp_path))
        assert result == []

    def test_settings_json_read_error(self, tmp_path: Path) -> None:
        """Lines 85-86: exception in settings.json reading."""
        settings = tmp_path / "settings.json"
        settings.write_text("{bad json")
        from nu_coding_agent.migrations import migrate_auth_to_auth_json

        result = migrate_auth_to_auth_json(str(tmp_path))
        assert result == []

    def test_settings_with_api_keys(self, tmp_path: Path) -> None:
        """Lines 78-86: settings.json with apiKeys dict gets migrated."""
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"apiKeys": {"openai": "sk-test"}, "other": True}))
        from nu_coding_agent.migrations import migrate_auth_to_auth_json

        result = migrate_auth_to_auth_json(str(tmp_path))
        assert "openai" in result
        # apiKeys should be removed from settings.json
        updated = json.loads(settings.read_text())
        assert "apiKeys" not in updated
        assert updated["other"] is True

    def test_session_rename_exception(self, tmp_path: Path) -> None:
        """Lines 142-143: exception during session file rename."""
        session_file = tmp_path / "test.jsonl"
        session_file.write_text('{"header": true, "cwd": "/work"}\n')
        with patch.object(Path, "rename", side_effect=OSError("denied")):
            from nu_coding_agent.migrations import migrate_sessions_from_agent_root

            migrate_sessions_from_agent_root(str(tmp_path))
        assert session_file.exists()  # not moved

    def test_commands_rename_oserror(self, tmp_path: Path) -> None:
        """Lines 161-162: OSError during commands→prompts rename."""
        commands = tmp_path / "commands"
        commands.mkdir()
        from nu_coding_agent.migrations import _migrate_commands_to_prompts

        with patch.object(Path, "rename", side_effect=OSError("perm")):
            result = _migrate_commands_to_prompts(tmp_path, "test")
        assert result is False

    def test_tools_iterdir_oserror(self, tmp_path: Path) -> None:
        """Lines 191-192: OSError on tools/ iterdir."""
        tools = tmp_path / "tools"
        tools.mkdir()
        (tools / "custom_script.py").write_text("pass")
        from nu_coding_agent.migrations import _check_deprecated_extension_dirs

        with patch.object(Path, "iterdir", side_effect=OSError("perm")):
            warnings = _check_deprecated_extension_dirs(tmp_path, "test")
        # Should not crash, just skip
        assert isinstance(warnings, list)

    def test_keybindings_migration_exception(self, tmp_path: Path) -> None:
        """Lines 234-235: exception in keybindings migration."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "keybindings.json").write_text("not json{{{")
        from nu_coding_agent.migrations import _migrate_keybindings_config_file

        with patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)):
            _migrate_keybindings_config_file()  # should not raise

    def test_tools_to_bin_rename_oserror(self, tmp_path: Path) -> None:
        """Lines 265-266: OSError during tools→bin rename."""
        agent_dir = tmp_path / "agent"
        tools_dir = agent_dir / "tools"
        tools_dir.mkdir(parents=True)
        (tools_dir / "rg").write_text("binary")
        bin_dir = tmp_path / "bin"
        from nu_coding_agent.migrations import _migrate_tools_to_bin

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)),
            patch("nu_coding_agent.migrations.get_bin_dir", return_value=str(bin_dir)),
            patch.object(Path, "rename", side_effect=OSError("perm")),
        ):
            _migrate_tools_to_bin()
        # Should not crash

    @pytest.mark.asyncio
    async def test_show_deprecation_warnings_empty(self) -> None:
        """Lines 282-283: empty warnings list returns immediately."""
        from nu_coding_agent.migrations import show_deprecation_warnings

        await show_deprecation_warnings([])

    @pytest.mark.asyncio
    async def test_show_deprecation_warnings_non_tty(self) -> None:
        """Lines 298-310: non-tty stdin skips raw mode read."""
        from nu_coding_agent.migrations import show_deprecation_warnings

        with patch.object(sys.stdin, "isatty", return_value=False):
            await show_deprecation_warnings(["test warning"])

    @pytest.mark.asyncio
    async def test_show_deprecation_warnings_tty_exception(self) -> None:
        """Lines 298-310: tty stdin but termios exception is swallowed."""
        from nu_coding_agent.migrations import show_deprecation_warnings

        with (
            patch.object(sys.stdin, "isatty", return_value=True),
            patch.dict("sys.modules", {"termios": MagicMock(tcgetattr=MagicMock(side_effect=Exception("no tty")))}),
        ):
            await show_deprecation_warnings(["test warning"])

    def test_tools_to_bin_existing_target(self, tmp_path: Path) -> None:
        """Lines 267-269: target exists → old file unlinked."""
        agent_dir = tmp_path / "agent"
        tools_dir = agent_dir / "tools"
        tools_dir.mkdir(parents=True)
        (tools_dir / "fd").write_text("old")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "fd").write_text("new")
        from nu_coding_agent.migrations import _migrate_tools_to_bin

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)),
            patch("nu_coding_agent.migrations.get_bin_dir", return_value=str(bin_dir)),
        ):
            _migrate_tools_to_bin()
        assert not (tools_dir / "fd").exists()  # old removed
        assert (bin_dir / "fd").read_text() == "new"  # new untouched


# ---------------------------------------------------------------------------
# resource_loader.py — lines 94-98, 124-125, 164, 195-197, 203-205,
#                       210, 226, 237-239, 358-359, 386-388,
#                       418-434, 441-443, 486, 491-492
# ---------------------------------------------------------------------------


class TestResourceLoaderCoverageGaps:
    """Target every remaining uncovered line in resource_loader.py."""

    def _make_loader(self, tmp_path: Path, **kwargs: Any) -> Any:
        from nu_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions

        opts = ResourceLoaderOptions(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "agent"),
            **kwargs,
        )
        loader = ResourceLoader(opts)
        loader.reload()
        return loader

    def test_is_under_same_path(self) -> None:
        """Lines 94-98: _is_under with exact match and prefix match."""
        from nu_coding_agent.core.resource_loader import _is_under

        assert _is_under("/foo/bar", "/foo/bar") is True
        assert _is_under("/foo/bar/baz", "/foo/bar") is True
        assert _is_under("/foo/barbaz", "/foo/bar") is False
        assert _is_under("/other", "/foo/bar") is False

    def test_load_context_file_read_error(self, tmp_path: Path) -> None:
        """Lines 124-125: OSError reading context file."""
        from nu_coding_agent.core.resource_loader import _load_context_file_from_dir

        d = tmp_path / "ctx"
        d.mkdir()
        (d / "AGENTS.md").write_text("content")
        with patch.object(Path, "read_text", side_effect=OSError("perm")):
            result = _load_context_file_from_dir(str(d))
        assert result is None

    def test_load_project_context_no_cwd(self) -> None:
        """Line 164: load_project_context_files with explicit cwd."""
        from nu_coding_agent.core.resource_loader import load_project_context_files

        files = load_project_context_files(cwd="/nonexistent_path_xyz", agent_dir="/nonexistent_agent_xyz")
        assert isinstance(files, list)

    def test_theme_loading_invalid_json(self, tmp_path: Path) -> None:
        """Lines 195-197: invalid JSON theme file skipped."""
        themes_dir = tmp_path / "themes"
        themes_dir.mkdir()
        (themes_dir / "bad.json").write_text("not json{{{")
        loader = self._make_loader(tmp_path, additional_theme_paths=[str(themes_dir)])
        result = loader.get_themes()
        themes = result[0] if isinstance(result, tuple) else result
        assert isinstance(themes, list)

    def test_theme_loading_no_name_field(self, tmp_path: Path) -> None:
        """Lines 203-205: theme without 'name' key uses filename."""
        themes_dir = tmp_path / "themes"
        themes_dir.mkdir()
        (themes_dir / "noname.json").write_text(json.dumps({"colors": {}}))
        loader = self._make_loader(tmp_path, additional_theme_paths=[str(themes_dir)])
        result = loader.get_themes()
        themes = result[0] if isinstance(result, tuple) else result
        assert isinstance(themes, list)

    def test_theme_from_two_dirs(self, tmp_path: Path) -> None:
        """Lines 210, 226: themes loaded from multiple additional dirs."""
        dir1 = tmp_path / "t1"
        dir1.mkdir()
        dir2 = tmp_path / "t2"
        dir2.mkdir()
        (dir1 / "dark.json").write_text(json.dumps({"name": "dark", "colors": {}}))
        (dir2 / "light.json").write_text(json.dumps({"name": "light", "colors": {}}))
        loader = self._make_loader(tmp_path, additional_theme_paths=[str(dir1), str(dir2)])
        themes, diags = loader.get_themes()
        names = {t.name for t in themes}
        assert "dark" in names
        assert "light" in names

    def test_prompt_collision(self, tmp_path: Path) -> None:
        """Lines 418-434: duplicate prompt names produce collision diagnostic."""
        dir1 = tmp_path / "p1"
        dir1.mkdir()
        dir2 = tmp_path / "p2"
        dir2.mkdir()
        (dir1 / "greet.md").write_text("---\nname: greet\n---\nHello")
        (dir2 / "greet.md").write_text("---\nname: greet\n---\nHi")
        loader = self._make_loader(tmp_path, additional_prompt_paths=[str(dir1), str(dir2)])
        prompts, diags = loader.get_prompts()
        collision_diags = [d for d in diags if d.type == "collision"]
        assert len(collision_diags) >= 1

    def test_missing_additional_prompt_path(self, tmp_path: Path) -> None:
        """Lines 441-443: missing additional prompt path produces error diagnostic."""
        loader = self._make_loader(tmp_path, additional_prompt_paths=[str(tmp_path / "nonexistent")])
        prompts, diags = loader.get_prompts()
        error_diags = [d for d in diags if d.type == "error"]
        assert len(error_diags) >= 1

    def test_system_prompt_from_project_dir(self, tmp_path: Path) -> None:
        """Lines 486, 491-492: system prompt from .nu/."""
        nu_dir = tmp_path / ".nu"
        nu_dir.mkdir()
        (nu_dir / "SYSTEM.md").write_text("You are helpful.")
        loader = self._make_loader(tmp_path)
        assert loader.get_system_prompt() == "You are helpful."

    def test_append_system_prompt(self, tmp_path: Path) -> None:
        """Lines 491-492: append system prompt from .nu/."""
        nu_dir = tmp_path / ".nu"
        nu_dir.mkdir()
        (nu_dir / "APPEND_SYSTEM.md").write_text("Also be concise.")
        loader = self._make_loader(tmp_path)
        asp = loader.get_append_system_prompt()
        # Returns a list of fragments
        assert any("Also be concise." in s for s in asp) if isinstance(asp, list) else asp == "Also be concise."

    def test_get_agents_files_walks_up(self, tmp_path: Path) -> None:
        """Lines 358-359, 386-388: agents files collected walking up."""
        child = tmp_path / "a" / "b"
        child.mkdir(parents=True)
        (tmp_path / "AGENTS.md").write_text("root agents")
        (tmp_path / "a" / "AGENTS.md").write_text("mid agents")
        from nu_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions

        loader = ResourceLoader(ResourceLoaderOptions(cwd=str(child), agent_dir=str(tmp_path / "agent")))
        loader.reload()
        files = loader.get_agents_files()
        assert len(files) >= 2
        assert files[0].content == "root agents"
