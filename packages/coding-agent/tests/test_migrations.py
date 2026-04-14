"""Tests for ``nu_coding_agent.migrations``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nu_coding_agent.migrations import (
    MigrationResult,
    _check_deprecated_extension_dirs,
    _migrate_commands_to_prompts,
    migrate_auth_to_auth_json,
    migrate_sessions_from_agent_root,
    run_migrations,
)


# ---------------------------------------------------------------------------
# migrate_auth_to_auth_json
# ---------------------------------------------------------------------------


class TestMigrateAuthToAuthJson:
    def test_skips_if_auth_exists(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text("{}")
        result = migrate_auth_to_auth_json(str(tmp_path))
        assert result == []

    def test_migrates_oauth(self, tmp_path: Path) -> None:
        oauth_data = {"github": {"access_token": "tok123"}}
        (tmp_path / "oauth.json").write_text(json.dumps(oauth_data))

        result = migrate_auth_to_auth_json(str(tmp_path))
        assert "github" in result

        auth = json.loads((tmp_path / "auth.json").read_text())
        assert auth["github"]["type"] == "oauth"
        assert auth["github"]["access_token"] == "tok123"

        # oauth.json should be renamed
        assert not (tmp_path / "oauth.json").exists()
        assert (tmp_path / "oauth.json.migrated").exists()

    def test_migrates_api_keys_from_settings(self, tmp_path: Path) -> None:
        settings = {"apiKeys": {"openai": "sk-xxx"}, "other": "value"}
        (tmp_path / "settings.json").write_text(json.dumps(settings))

        result = migrate_auth_to_auth_json(str(tmp_path))
        assert "openai" in result

        auth = json.loads((tmp_path / "auth.json").read_text())
        assert auth["openai"]["type"] == "api_key"
        assert auth["openai"]["key"] == "sk-xxx"

        # apiKeys should be removed from settings
        updated_settings = json.loads((tmp_path / "settings.json").read_text())
        assert "apiKeys" not in updated_settings
        assert updated_settings["other"] == "value"

    def test_migrates_both(self, tmp_path: Path) -> None:
        (tmp_path / "oauth.json").write_text(json.dumps({"github": {"token": "gh_tok"}}))
        (tmp_path / "settings.json").write_text(json.dumps({"apiKeys": {"openai": "sk-key"}}))

        result = migrate_auth_to_auth_json(str(tmp_path))
        assert "github" in result
        assert "openai" in result

        auth = json.loads((tmp_path / "auth.json").read_text())
        assert len(auth) == 2

    def test_oauth_takes_precedence(self, tmp_path: Path) -> None:
        (tmp_path / "oauth.json").write_text(json.dumps({"openai": {"token": "oauth_tok"}}))
        (tmp_path / "settings.json").write_text(json.dumps({"apiKeys": {"openai": "sk-key"}}))

        migrate_auth_to_auth_json(str(tmp_path))
        auth = json.loads((tmp_path / "auth.json").read_text())
        # oauth entry wins (it's processed first)
        assert auth["openai"]["type"] == "oauth"

    def test_nothing_to_migrate(self, tmp_path: Path) -> None:
        result = migrate_auth_to_auth_json(str(tmp_path))
        assert result == []
        assert not (tmp_path / "auth.json").exists()

    def test_auth_json_permissions(self, tmp_path: Path) -> None:
        (tmp_path / "oauth.json").write_text(json.dumps({"gh": {}}))
        migrate_auth_to_auth_json(str(tmp_path))
        mode = (tmp_path / "auth.json").stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# migrate_sessions_from_agent_root
# ---------------------------------------------------------------------------


class TestMigrateSessionsFromAgentRoot:
    def test_moves_jsonl_files(self, tmp_path: Path) -> None:
        header = json.dumps({"type": "session", "cwd": "/home/user/project"})
        (tmp_path / "session1.jsonl").write_text(f"{header}\n{{\"msg\": \"hi\"}}\n")

        migrate_sessions_from_agent_root(str(tmp_path))

        # File should be moved into sessions/<encoded>/
        assert not (tmp_path / "session1.jsonl").exists()
        sessions_dir = tmp_path / "sessions"
        assert sessions_dir.exists()
        # Find the moved file
        moved_files = list(sessions_dir.rglob("session1.jsonl"))
        assert len(moved_files) == 1

    def test_skips_non_session_jsonl(self, tmp_path: Path) -> None:
        (tmp_path / "other.jsonl").write_text(json.dumps({"type": "log", "data": "x"}) + "\n")
        migrate_sessions_from_agent_root(str(tmp_path))
        assert (tmp_path / "other.jsonl").exists()

    def test_skips_empty_jsonl(self, tmp_path: Path) -> None:
        (tmp_path / "empty.jsonl").write_text("\n")
        migrate_sessions_from_agent_root(str(tmp_path))
        assert (tmp_path / "empty.jsonl").exists()

    def test_no_jsonl_files(self, tmp_path: Path) -> None:
        # Should not raise
        migrate_sessions_from_agent_root(str(tmp_path))

    def test_nonexistent_dir(self) -> None:
        # Should not raise
        migrate_sessions_from_agent_root("/nonexistent/dir")

    def test_skips_if_destination_exists(self, tmp_path: Path) -> None:
        header = json.dumps({"type": "session", "cwd": "/home/user/proj"})
        (tmp_path / "s.jsonl").write_text(f"{header}\n")

        # Pre-create the destination with correct encoding:
        # strip leading /, replace /\\: with -, wrap with --
        safe_path = "--home-user-proj--"
        dest_dir = tmp_path / "sessions" / safe_path
        dest_dir.mkdir(parents=True)
        (dest_dir / "s.jsonl").write_text("existing")

        migrate_sessions_from_agent_root(str(tmp_path))
        # Original should still be there (not moved because dest exists)
        assert (tmp_path / "s.jsonl").exists()


# ---------------------------------------------------------------------------
# _migrate_commands_to_prompts
# ---------------------------------------------------------------------------


class TestMigrateCommandsToPrompts:
    def test_renames_commands_to_prompts(self, tmp_path: Path) -> None:
        (tmp_path / "commands").mkdir()
        (tmp_path / "commands" / "greet.md").write_text("hi")

        result = _migrate_commands_to_prompts(tmp_path, "Test")
        assert result is True
        assert (tmp_path / "prompts").exists()
        assert not (tmp_path / "commands").exists()

    def test_no_op_if_prompts_exists(self, tmp_path: Path) -> None:
        (tmp_path / "commands").mkdir()
        (tmp_path / "prompts").mkdir()

        result = _migrate_commands_to_prompts(tmp_path, "Test")
        assert result is False

    def test_no_op_if_no_commands(self, tmp_path: Path) -> None:
        result = _migrate_commands_to_prompts(tmp_path, "Test")
        assert result is False


# ---------------------------------------------------------------------------
# _check_deprecated_extension_dirs
# ---------------------------------------------------------------------------


class TestCheckDeprecatedExtensionDirs:
    def test_warns_on_hooks(self, tmp_path: Path) -> None:
        (tmp_path / "hooks").mkdir()
        warnings = _check_deprecated_extension_dirs(tmp_path, "Test")
        assert any("hooks" in w.lower() for w in warnings)

    def test_warns_on_tools_with_custom(self, tmp_path: Path) -> None:
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "my_tool.py").write_text("# custom")

        warnings = _check_deprecated_extension_dirs(tmp_path, "Test")
        assert any("tools" in w.lower() for w in warnings)

    def test_no_warn_on_tools_with_only_managed(self, tmp_path: Path) -> None:
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "fd").write_text("")
        (tools_dir / "rg").write_text("")

        warnings = _check_deprecated_extension_dirs(tmp_path, "Test")
        # No warning about tools since only managed binaries
        tools_warnings = [w for w in warnings if "custom tools" in w.lower()]
        assert tools_warnings == []

    def test_no_deprecated_dirs(self, tmp_path: Path) -> None:
        warnings = _check_deprecated_extension_dirs(tmp_path, "Test")
        assert warnings == []


# ---------------------------------------------------------------------------
# run_migrations
# ---------------------------------------------------------------------------


class TestMigrateToolsToBin:
    def test_moves_binaries(self, tmp_path: Path) -> None:
        from nu_coding_agent.migrations import _migrate_tools_to_bin

        agent_dir = tmp_path / "agent"
        tools_dir = agent_dir / "tools"
        tools_dir.mkdir(parents=True)
        (tools_dir / "fd").write_text("binary")
        (tools_dir / "rg").write_text("binary")

        bin_dir = tmp_path / "bin"

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)),
            patch("nu_coding_agent.migrations.get_bin_dir", return_value=str(bin_dir)),
        ):
            _migrate_tools_to_bin()

        assert (bin_dir / "fd").exists()
        assert (bin_dir / "rg").exists()
        assert not (tools_dir / "fd").exists()
        assert not (tools_dir / "rg").exists()

    def test_no_tools_dir(self, tmp_path: Path) -> None:
        from nu_coding_agent.migrations import _migrate_tools_to_bin

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)),
            patch("nu_coding_agent.migrations.get_bin_dir", return_value=str(tmp_path / "bin")),
        ):
            _migrate_tools_to_bin()  # should not raise

    def test_removes_old_if_new_exists(self, tmp_path: Path) -> None:
        from nu_coding_agent.migrations import _migrate_tools_to_bin

        agent_dir = tmp_path / "agent"
        tools_dir = agent_dir / "tools"
        tools_dir.mkdir(parents=True)
        (tools_dir / "fd").write_text("old")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "fd").write_text("new")

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)),
            patch("nu_coding_agent.migrations.get_bin_dir", return_value=str(bin_dir)),
        ):
            _migrate_tools_to_bin()

        assert (bin_dir / "fd").read_text() == "new"
        assert not (tools_dir / "fd").exists()


class TestMigrateExtensionSystem:
    def test_commands_to_prompts(self, tmp_path: Path) -> None:
        from nu_coding_agent.migrations import _migrate_extension_system

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / ".nu" / "commands").mkdir(parents=True)

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)),
        ):
            warnings = _migrate_extension_system(str(cwd))
            assert isinstance(warnings, list)

        assert (cwd / ".nu" / "prompts").exists()


class TestMigrateKeybindingsConfigFile:
    def test_no_file(self, tmp_path: Path) -> None:
        from nu_coding_agent.migrations import _migrate_keybindings_config_file

        with patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(tmp_path)):
            _migrate_keybindings_config_file()  # should not raise

    def test_not_a_dict(self, tmp_path: Path) -> None:
        from nu_coding_agent.migrations import _migrate_keybindings_config_file

        (tmp_path / "keybindings.json").write_text('"just a string"')
        with patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(tmp_path)):
            _migrate_keybindings_config_file()  # should not raise

    def test_migrates(self, tmp_path: Path) -> None:
        from nu_coding_agent.migrations import _migrate_keybindings_config_file

        (tmp_path / "keybindings.json").write_text(json.dumps({"bindings": {}}))

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(tmp_path)),
            patch(
                "nu_coding_agent.core.keybindings.migrate_keybindings_config",
                return_value=({"migrated": True}, True),
            ),
        ):
            _migrate_keybindings_config_file()
            content = json.loads((tmp_path / "keybindings.json").read_text())
            assert content == {"migrated": True}

    def test_no_migration_needed(self, tmp_path: Path) -> None:
        from nu_coding_agent.migrations import _migrate_keybindings_config_file

        (tmp_path / "keybindings.json").write_text(json.dumps({"bindings": {}}))

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(tmp_path)),
            patch(
                "nu_coding_agent.core.keybindings.migrate_keybindings_config",
                return_value=({"bindings": {}}, False),
            ),
        ):
            _migrate_keybindings_config_file()


class TestShowDeprecationWarnings:
    @pytest.mark.asyncio
    async def test_no_warnings(self) -> None:
        from nu_coding_agent.migrations import show_deprecation_warnings

        # Should not raise
        await show_deprecation_warnings([])

    @pytest.mark.asyncio
    async def test_with_warnings_non_tty(self) -> None:
        from nu_coding_agent.migrations import show_deprecation_warnings

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            await show_deprecation_warnings(["test warning"])


class TestRunMigrations:
    def test_returns_result(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)),
            patch("nu_coding_agent.migrations.get_bin_dir", return_value=str(tmp_path / "bin")),
        ):
            result = run_migrations(cwd=str(tmp_path))
            assert isinstance(result, MigrationResult)
            assert isinstance(result.migrated_auth_providers, list)
            assert isinstance(result.deprecation_warnings, list)

    def test_with_oauth_migration(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "oauth.json").write_text(json.dumps({"github": {"tok": "x"}}))

        with (
            patch("nu_coding_agent.migrations.get_agent_dir", return_value=str(agent_dir)),
            patch("nu_coding_agent.migrations.get_bin_dir", return_value=str(tmp_path / "bin")),
        ):
            result = run_migrations(cwd=str(tmp_path))
            assert "github" in result.migrated_auth_providers
