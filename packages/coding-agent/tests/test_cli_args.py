"""Tests for ``nu_coding_agent.cli_args``."""

from __future__ import annotations

from nu_coding_agent.cli_args import (
    Args,
    is_valid_thinking_level,
    parse_args,
    render_help,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_parse_empty_args() -> None:
    args = parse_args([])
    assert isinstance(args, Args)
    assert args.help is False
    assert args.messages == []
    assert args.file_args == []
    assert args.unknown_flags == {}


# ---------------------------------------------------------------------------
# Boolean / one-off flags
# ---------------------------------------------------------------------------


def test_help_flag_short_and_long() -> None:
    assert parse_args(["--help"]).help is True
    assert parse_args(["-h"]).help is True


def test_version_flag_short_and_long() -> None:
    assert parse_args(["--version"]).version is True
    assert parse_args(["-v"]).version is True


def test_continue_resume_flags() -> None:
    assert parse_args(["--continue"]).continue_session is True
    assert parse_args(["-c"]).continue_session is True
    assert parse_args(["--resume"]).resume is True
    assert parse_args(["-r"]).resume is True


def test_print_flag() -> None:
    assert parse_args(["--print"]).print is True
    assert parse_args(["-p"]).print is True


def test_no_session_no_tools_flags() -> None:
    args = parse_args(["--no-session", "--no-tools"])
    assert args.no_session is True
    assert args.no_tools is True


def test_offline_verbose_flags() -> None:
    args = parse_args(["--offline", "--verbose"])
    assert args.offline is True
    assert args.verbose is True


def test_no_extensions_skills_prompts_themes() -> None:
    args = parse_args(["--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes"])
    assert args.no_extensions is True
    assert args.no_skills is True
    assert args.no_prompt_templates is True
    assert args.no_themes is True


def test_short_aliases_for_no_flags() -> None:
    args = parse_args(["-ne", "-ns", "-np"])
    assert args.no_extensions is True
    assert args.no_skills is True
    assert args.no_prompt_templates is True


# ---------------------------------------------------------------------------
# Value flags
# ---------------------------------------------------------------------------


def test_provider_and_model() -> None:
    args = parse_args(["--provider", "openai", "--model", "gpt-4o"])
    assert args.provider == "openai"
    assert args.model == "gpt-4o"


def test_api_key_and_system_prompt() -> None:
    args = parse_args(
        [
            "--api-key",
            "sk-foo",
            "--system-prompt",
            "you are nice",
            "--append-system-prompt",
            "and brief",
        ]
    )
    assert args.api_key == "sk-foo"
    assert args.system_prompt == "you are nice"
    assert args.append_system_prompt == "and brief"


def test_session_path_and_fork() -> None:
    args = parse_args(
        [
            "--session",
            "/tmp/s.jsonl",
            "--fork",
            "abc1234",
            "--session-dir",
            "/tmp/sessions",
        ]
    )
    assert args.session == "/tmp/s.jsonl"
    assert args.fork == "abc1234"
    assert args.session_dir == "/tmp/sessions"


def test_mode_flag_valid_and_invalid() -> None:
    assert parse_args(["--mode", "text"]).mode == "text"
    assert parse_args(["--mode", "json"]).mode == "json"
    assert parse_args(["--mode", "rpc"]).mode == "rpc"
    # Invalid mode is silently dropped (matches upstream).
    assert parse_args(["--mode", "bogus"]).mode is None


def test_thinking_valid_and_invalid() -> None:
    args = parse_args(["--thinking", "high"])
    assert args.thinking == "high"
    assert args.diagnostics == []
    invalid = parse_args(["--thinking", "bogus"])
    assert invalid.thinking is None
    assert any("Invalid thinking" in d.message for d in invalid.diagnostics)


def test_export_flag() -> None:
    assert parse_args(["--export", "/tmp/out.html"]).export == "/tmp/out.html"


# ---------------------------------------------------------------------------
# Multi-value flags
# ---------------------------------------------------------------------------


def test_extension_repeats() -> None:
    args = parse_args(["--extension", "/a", "-e", "/b"])
    assert args.extensions == ["/a", "/b"]


def test_skill_repeats() -> None:
    args = parse_args(["--skill", "/a", "--skill", "/b"])
    assert args.skills == ["/a", "/b"]


def test_prompt_template_and_theme_repeats() -> None:
    args = parse_args(["--prompt-template", "/a", "--theme", "/t", "--theme", "/t2"])
    assert args.prompt_templates == ["/a"]
    assert args.themes == ["/t", "/t2"]


def test_models_csv() -> None:
    args = parse_args(["--models", "anthropic/*, openai/gpt-4o, *sonnet*"])
    assert args.models == ["anthropic/*", "openai/gpt-4o", "*sonnet*"]


def test_tools_csv_filters_unknown() -> None:
    args = parse_args(["--tools", "read,bash,bogus,grep"])
    assert args.tools == ["read", "bash", "grep"]
    assert any("bogus" in d.message for d in args.diagnostics)


# ---------------------------------------------------------------------------
# list-models special handling
# ---------------------------------------------------------------------------


def test_list_models_no_arg() -> None:
    args = parse_args(["--list-models"])
    assert args.list_models is True


def test_list_models_with_search() -> None:
    args = parse_args(["--list-models", "claude"])
    assert args.list_models == "claude"


def test_list_models_followed_by_flag_treats_as_bare() -> None:
    args = parse_args(["--list-models", "--print"])
    assert args.list_models is True
    assert args.print is True


def test_list_models_followed_by_file_arg_treats_as_bare() -> None:
    args = parse_args(["--list-models", "@file.md"])
    assert args.list_models is True
    assert args.file_args == ["file.md"]


# ---------------------------------------------------------------------------
# File args + messages
# ---------------------------------------------------------------------------


def test_file_args() -> None:
    args = parse_args(["@prompt.md", "@image.png"])
    assert args.file_args == ["prompt.md", "image.png"]


def test_positional_messages() -> None:
    args = parse_args(["hello", "world"])
    assert args.messages == ["hello", "world"]


def test_mixed_files_messages_options() -> None:
    args = parse_args(["--print", "@a.md", "first message", "--model", "claude", "second"])
    assert args.print is True
    assert args.file_args == ["a.md"]
    assert args.model == "claude"
    assert args.messages == ["first message", "second"]


# ---------------------------------------------------------------------------
# Unknown flags & errors
# ---------------------------------------------------------------------------


def test_unknown_long_flag_with_value() -> None:
    args = parse_args(["--plan", "deep"])
    assert args.unknown_flags == {"plan": "deep"}


def test_unknown_long_flag_boolean() -> None:
    args = parse_args(["--plan"])
    assert args.unknown_flags == {"plan": True}


def test_unknown_long_flag_with_equals() -> None:
    args = parse_args(["--plan=brief"])
    assert args.unknown_flags == {"plan": "brief"}


def test_unknown_short_flag_emits_error() -> None:
    args = parse_args(["-x"])
    assert any(d.type == "error" and "-x" in d.message for d in args.diagnostics)


def test_unknown_long_flag_followed_by_at_arg_treats_as_bool() -> None:
    args = parse_args(["--plan", "@file.md"])
    assert args.unknown_flags == {"plan": True}
    assert args.file_args == ["file.md"]


# ---------------------------------------------------------------------------
# is_valid_thinking_level
# ---------------------------------------------------------------------------


def test_is_valid_thinking_level() -> None:
    for level in ("off", "minimal", "low", "medium", "high", "xhigh"):
        assert is_valid_thinking_level(level) is True
    assert is_valid_thinking_level("nope") is False


# ---------------------------------------------------------------------------
# render_help
# ---------------------------------------------------------------------------


def test_render_help_includes_app_name_and_options() -> None:
    text = render_help()
    assert "nu" in text
    assert "--provider" in text
    assert "--thinking" in text
    assert "--list-models" in text
    assert "--help" in text
