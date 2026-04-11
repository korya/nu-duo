"""Tests for ``nu_coding_agent.core.settings_manager``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nu_coding_agent.core.settings_manager import (
    FileSettingsStorage,
    InMemorySettingsStorage,
    SettingsManager,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    sm = SettingsManager.in_memory()
    assert sm.get_steering_mode() == "one-at-a-time"
    assert sm.get_follow_up_mode() == "one-at-a-time"
    assert sm.get_transport() == "sse"
    assert sm.get_compaction_enabled() is True
    assert sm.get_compaction_reserve_tokens() == 16384
    assert sm.get_compaction_keep_recent_tokens() == 20000
    assert sm.get_retry_enabled() is True
    assert sm.get_enable_skill_commands() is True
    assert sm.get_double_escape_action() == "tree"
    assert sm.get_tree_filter_mode() == "default"
    assert sm.get_editor_padding_x() == 0
    assert sm.get_autocomplete_max_visible() == 5
    assert sm.get_code_block_indent() == "  "


def test_seeded_in_memory() -> None:
    sm = SettingsManager.in_memory({"defaultModel": "claude-opus-4", "theme": "dark"})
    assert sm.get_default_model() == "claude-opus-4"
    assert sm.get_theme() == "dark"


# ---------------------------------------------------------------------------
# Top-level setters
# ---------------------------------------------------------------------------


def test_set_default_model_persists_to_storage() -> None:
    storage = InMemorySettingsStorage()
    sm = SettingsManager.from_storage(storage)
    sm.set_default_model_and_provider("openai", "gpt-4o")
    assert sm.get_default_model() == "gpt-4o"
    assert sm.get_default_provider() == "openai"
    sm2 = SettingsManager.from_storage(storage)
    assert sm2.get_default_model() == "gpt-4o"
    assert sm2.get_default_provider() == "openai"


def test_set_steering_mode() -> None:
    sm = SettingsManager.in_memory()
    sm.set_steering_mode("all")
    assert sm.get_steering_mode() == "all"


def test_set_default_thinking_level() -> None:
    sm = SettingsManager.in_memory()
    sm.set_default_thinking_level("high")
    assert sm.get_default_thinking_level() == "high"


def test_set_transport() -> None:
    sm = SettingsManager.in_memory()
    sm.set_transport("websocket")
    assert sm.get_transport() == "websocket"


def test_set_theme() -> None:
    sm = SettingsManager.in_memory()
    sm.set_theme("solarized")
    assert sm.get_theme() == "solarized"


def test_set_last_changelog_version() -> None:
    sm = SettingsManager.in_memory()
    sm.set_last_changelog_version("1.2.3")
    assert sm.get_last_changelog_version() == "1.2.3"


# ---------------------------------------------------------------------------
# Nested objects (compaction / retry / terminal / images)
# ---------------------------------------------------------------------------


def test_compaction_enabled_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_compaction_enabled(False)
    assert sm.get_compaction_enabled() is False
    summary = sm.get_compaction_settings()
    assert summary == {"enabled": False, "reserveTokens": 16384, "keepRecentTokens": 20000}


def test_retry_enabled_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_retry_enabled(False)
    assert sm.get_retry_enabled() is False


def test_show_images_default_and_setter() -> None:
    sm = SettingsManager.in_memory()
    assert sm.get_show_images() is True
    sm.set_show_images(False)
    assert sm.get_show_images() is False


def test_image_auto_resize_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_image_auto_resize(False)
    assert sm.get_image_auto_resize() is False


def test_clear_on_shrink_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_CLEAR_ON_SHRINK", "1")
    sm = SettingsManager.in_memory()
    assert sm.get_clear_on_shrink() is True
    sm.set_clear_on_shrink(False)
    assert sm.get_clear_on_shrink() is False


def test_show_hardware_cursor_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_HARDWARE_CURSOR", "1")
    sm = SettingsManager.in_memory()
    assert sm.get_show_hardware_cursor() is True
    sm.set_show_hardware_cursor(False)
    assert sm.get_show_hardware_cursor() is False


# ---------------------------------------------------------------------------
# Path arrays (extensions / skills / prompts / themes / packages)
# ---------------------------------------------------------------------------


def test_extension_paths_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_extension_paths(["/a", "/b"])
    assert sm.get_extension_paths() == ["/a", "/b"]


def test_skill_paths_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_skill_paths(["~/skills"])
    assert sm.get_skill_paths() == ["~/skills"]


def test_packages_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_packages(["npm:foo", {"source": "git:bar", "extensions": ["x"]}])
    pkgs = sm.get_packages()
    assert pkgs[0] == "npm:foo"
    assert pkgs[1]["source"] == "git:bar"


def test_project_skill_paths_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_project_skill_paths(["./project-skills"])
    assert sm.get_skill_paths() == ["./project-skills"]


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------


def test_editor_padding_clamped() -> None:
    sm = SettingsManager.in_memory()
    sm.set_editor_padding_x(99)
    assert sm.get_editor_padding_x() == 3
    sm.set_editor_padding_x(-5)
    assert sm.get_editor_padding_x() == 0


def test_autocomplete_max_visible_clamped() -> None:
    sm = SettingsManager.in_memory()
    sm.set_autocomplete_max_visible(50)
    assert sm.get_autocomplete_max_visible() == 20
    sm.set_autocomplete_max_visible(1)
    assert sm.get_autocomplete_max_visible() == 3


def test_tree_filter_mode_validation() -> None:
    sm = SettingsManager.in_memory()
    sm.set_tree_filter_mode("user-only")
    assert sm.get_tree_filter_mode() == "user-only"
    sm.set_tree_filter_mode("invalid")
    assert sm.get_tree_filter_mode() == "default"


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_migrates_queue_mode_to_steering_mode() -> None:
    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _current: json.dumps({"queueMode": "all"}))
    sm = SettingsManager.from_storage(storage)
    assert sm.get_steering_mode() == "all"
    assert "queueMode" not in sm.get_global_settings()


def test_migrates_websockets_to_transport() -> None:
    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _current: json.dumps({"websockets": True}))
    sm = SettingsManager.from_storage(storage)
    assert sm.get_transport() == "websocket"


def test_migrates_legacy_skills_object() -> None:
    storage = InMemorySettingsStorage()
    storage.with_lock(
        "global",
        lambda _current: json.dumps({"skills": {"enableSkillCommands": False, "customDirectories": ["/x"]}}),
    )
    sm = SettingsManager.from_storage(storage)
    assert sm.get_enable_skill_commands() is False
    assert sm.get_skill_paths() == ["/x"]


def test_migrates_legacy_skills_object_without_dirs() -> None:
    storage = InMemorySettingsStorage()
    storage.with_lock(
        "global",
        lambda _current: json.dumps({"skills": {"enableSkillCommands": True}}),
    )
    sm = SettingsManager.from_storage(storage)
    assert sm.get_enable_skill_commands() is True
    assert sm.get_skill_paths() == []


# ---------------------------------------------------------------------------
# Project overrides
# ---------------------------------------------------------------------------


def test_project_settings_override_global() -> None:
    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _: json.dumps({"theme": "dark", "defaultModel": "g"}))
    storage.with_lock("project", lambda _: json.dumps({"theme": "light"}))
    sm = SettingsManager.from_storage(storage)
    assert sm.get_theme() == "light"
    assert sm.get_default_model() == "g"


def test_project_compaction_merges_into_global() -> None:
    storage = InMemorySettingsStorage()
    storage.with_lock(
        "global",
        lambda _: json.dumps({"compaction": {"enabled": True, "reserveTokens": 1000}}),
    )
    storage.with_lock(
        "project",
        lambda _: json.dumps({"compaction": {"reserveTokens": 9999}}),
    )
    sm = SettingsManager.from_storage(storage)
    assert sm.get_compaction_enabled() is True
    assert sm.get_compaction_reserve_tokens() == 9999


# ---------------------------------------------------------------------------
# File backend round-trip
# ---------------------------------------------------------------------------


def test_file_backend_round_trip(tmp_path: Path) -> None:
    storage = FileSettingsStorage(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"))
    sm = SettingsManager.from_storage(storage)
    sm.set_default_model("claude-opus-4")
    storage2 = FileSettingsStorage(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"))
    sm2 = SettingsManager.from_storage(storage2)
    assert sm2.get_default_model() == "claude-opus-4"


def test_file_backend_creates_directories(tmp_path: Path) -> None:
    agent_dir = tmp_path / "deep" / "agent"
    storage = FileSettingsStorage(cwd=str(tmp_path), agent_dir=str(agent_dir))
    sm = SettingsManager.from_storage(storage)
    sm.set_default_model("model-x")
    assert (agent_dir / "settings.json").exists()


def test_load_error_recorded() -> None:
    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _: "not valid json {{")
    sm = SettingsManager.from_storage(storage)
    drained = sm.drain_errors()
    assert len(drained) == 1
    assert drained[0].scope == "global"


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


async def test_reload_picks_up_external_changes() -> None:
    storage = InMemorySettingsStorage()
    sm = SettingsManager.from_storage(storage)
    storage.with_lock("global", lambda _: json.dumps({"defaultModel": "from-outside"}))
    await sm.reload()
    assert sm.get_default_model() == "from-outside"


# ---------------------------------------------------------------------------
# Apply overrides
# ---------------------------------------------------------------------------


def test_apply_overrides() -> None:
    sm = SettingsManager.in_memory({"defaultModel": "first"})
    sm.apply_overrides({"defaultModel": "override", "theme": "ocean"})
    assert sm.get_default_model() == "override"
    assert sm.get_theme() == "ocean"


# ---------------------------------------------------------------------------
# All remaining getters/setters — sweep test for parity coverage
# ---------------------------------------------------------------------------


def test_set_default_provider() -> None:
    sm = SettingsManager.in_memory()
    sm.set_default_provider("anthropic")
    assert sm.get_default_provider() == "anthropic"


def test_set_follow_up_mode() -> None:
    sm = SettingsManager.in_memory()
    sm.set_follow_up_mode("all")
    assert sm.get_follow_up_mode() == "all"


def test_branch_summary_settings_default_and_skip() -> None:
    sm = SettingsManager.in_memory()
    summary = sm.get_branch_summary_settings()
    assert summary == {"reserveTokens": 16384, "skipPrompt": False}
    assert sm.get_branch_summary_skip_prompt() is False


def test_retry_settings_default() -> None:
    sm = SettingsManager.in_memory()
    settings = sm.get_retry_settings()
    assert settings["enabled"] is True
    assert settings["maxRetries"] == 3
    assert settings["baseDelayMs"] == 2000
    assert settings["maxDelayMs"] == 60000


def test_hide_thinking_block_setter() -> None:
    sm = SettingsManager.in_memory()
    assert sm.get_hide_thinking_block() is False
    sm.set_hide_thinking_block(True)
    assert sm.get_hide_thinking_block() is True


def test_shell_path_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_shell_path("/bin/zsh")
    assert sm.get_shell_path() == "/bin/zsh"
    sm.set_shell_path(None)
    assert sm.get_shell_path() is None


def test_quiet_startup_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_quiet_startup(True)
    assert sm.get_quiet_startup() is True


def test_shell_command_prefix_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_shell_command_prefix("shopt -s expand_aliases")
    assert sm.get_shell_command_prefix() == "shopt -s expand_aliases"


def test_npm_command_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_npm_command(["mise", "exec", "node@20", "--", "npm"])
    assert sm.get_npm_command() == ["mise", "exec", "node@20", "--", "npm"]
    sm.set_npm_command(None)
    assert sm.get_npm_command() is None


def test_collapse_changelog_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_collapse_changelog(True)
    assert sm.get_collapse_changelog() is True


def test_prompt_template_paths_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_prompt_template_paths(["/p"])
    assert sm.get_prompt_template_paths() == ["/p"]
    sm.set_project_prompt_template_paths(["/proj-p"])
    assert sm.get_prompt_template_paths() == ["/proj-p"]


def test_theme_paths_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_theme_paths(["/t"])
    assert sm.get_theme_paths() == ["/t"]
    sm.set_project_theme_paths(["/proj-t"])
    assert sm.get_theme_paths() == ["/proj-t"]


def test_project_extension_paths_override_global() -> None:
    sm = SettingsManager.in_memory()
    sm.set_extension_paths(["/global"])
    sm.set_project_extension_paths(["/project"])
    assert sm.get_extension_paths() == ["/project"]


def test_project_packages_override_global() -> None:
    sm = SettingsManager.in_memory()
    sm.set_packages(["npm:foo"])
    sm.set_project_packages(["npm:bar"])
    assert sm.get_packages() == ["npm:bar"]


def test_enable_skill_commands_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_enable_skill_commands(False)
    assert sm.get_enable_skill_commands() is False


def test_thinking_budgets_passthrough() -> None:
    sm = SettingsManager.in_memory({"thinkingBudgets": {"low": 100, "high": 5000}})
    assert sm.get_thinking_budgets() == {"low": 100, "high": 5000}


def test_block_images_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_block_images(True)
    assert sm.get_block_images() is True


def test_enabled_models_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_enabled_models(["openai:*", "anthropic:claude-*"])
    assert sm.get_enabled_models() == ["openai:*", "anthropic:claude-*"]
    sm.set_enabled_models(None)
    assert sm.get_enabled_models() is None


def test_double_escape_action_setter() -> None:
    sm = SettingsManager.in_memory()
    sm.set_double_escape_action("fork")
    assert sm.get_double_escape_action() == "fork"


def test_session_dir_passthrough() -> None:
    sm = SettingsManager.in_memory({"sessionDir": "/custom/sessions"})
    assert sm.get_session_dir() == "/custom/sessions"


def test_get_global_and_project_settings_returns_copies() -> None:
    sm = SettingsManager.in_memory({"theme": "dark"})
    snap = sm.get_global_settings()
    snap["theme"] = "tampered"
    assert sm.get_theme() == "dark"


def test_clear_on_shrink_explicit_setting_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_CLEAR_ON_SHRINK", "1")
    sm = SettingsManager.in_memory({"terminal": {"clearOnShrink": False}})
    assert sm.get_clear_on_shrink() is False


def test_show_hardware_cursor_explicit_setting_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_HARDWARE_CURSOR", "1")
    sm = SettingsManager.in_memory({"showHardwareCursor": False})
    assert sm.get_show_hardware_cursor() is False


async def test_flush_drains_queue() -> None:
    sm = SettingsManager.in_memory()
    sm.set_default_model("x")
    await sm.flush()  # should be a no-op since writes already drained
    assert sm.get_default_model() == "x"
