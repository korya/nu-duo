"""CLI argument parser — direct port of ``packages/coding-agent/src/cli/args.ts``.

Implements the same flag set, file-arg (``@path``) handling, multi-value
arguments (``--extension``, ``--skill``, ``--prompt-template``,
``--theme``), and the unknown-flag bucket that the upstream uses to
forward extension-defined flags. The diagnostic list mirrors the
upstream's behaviour: warnings for unknown ``--tools`` entries and
invalid ``--thinking`` levels, errors for unknown short options.

Lives at ``nu_coding_agent.cli_args`` (rather than ``nu_coding_agent.cli.args``)
because there's already a top-level ``cli.py`` module shipping the
``nu`` console-script entry point. Once the upstream's full CLI is
ported, the entry point and this parser will move into a proper
``nu_coding_agent.cli`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nu_coding_agent.config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR
from nu_coding_agent.core.tools import ALL_TOOL_NAMES

type Mode = Literal["text", "json", "rpc"]
type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


_VALID_THINKING_LEVELS: frozenset[str] = frozenset({"off", "minimal", "low", "medium", "high", "xhigh"})


@dataclass(slots=True)
class Diagnostic:
    type: Literal["warning", "error"]
    message: str


@dataclass(slots=True)
class Args:
    """Parsed CLI arguments — mirrors the upstream ``Args`` interface."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    thinking: ThinkingLevel | None = None
    continue_session: bool = False
    resume: bool = False
    help: bool = False
    version: bool = False
    mode: Mode | None = None
    no_session: bool = False
    session: str | None = None
    fork: str | None = None
    session_dir: str | None = None
    models: list[str] | None = None
    tools: list[str] | None = None
    no_tools: bool = False
    extensions: list[str] | None = None
    no_extensions: bool = False
    print: bool = False
    export: str | None = None
    no_skills: bool = False
    skills: list[str] | None = None
    prompt_templates: list[str] | None = None
    no_prompt_templates: bool = False
    themes: list[str] | None = None
    no_themes: bool = False
    list_models: str | bool | None = None
    offline: bool = False
    verbose: bool = False
    messages: list[str] = field(default_factory=list)
    file_args: list[str] = field(default_factory=list)
    unknown_flags: dict[str, bool | str] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)


def is_valid_thinking_level(level: str) -> bool:
    """Return ``True`` for the six legal thinking-level strings."""
    return level in _VALID_THINKING_LEVELS


def parse_args(argv: list[str]) -> Args:
    """Parse ``argv`` (without the program name) into an :class:`Args` record.

    Mirrors the upstream's ``parseArgs`` switch verbatim — same flag
    spellings, same short aliases, same diagnostics. Unknown ``--`` flags
    land in :attr:`Args.unknown_flags` so the eventual extension layer
    can forward them. Unknown short flags become diagnostic errors.
    """
    result = Args()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--help", "-h"):
            result.help = True
        elif arg in ("--version", "-v"):
            result.version = True
        elif arg == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
            i += 1
            if mode in ("text", "json", "rpc"):
                result.mode = mode  # type: ignore[assignment]
        elif arg in ("--continue", "-c"):
            result.continue_session = True
        elif arg in ("--resume", "-r"):
            result.resume = True
        elif arg == "--provider" and i + 1 < len(argv):
            i += 1
            result.provider = argv[i]
        elif arg == "--model" and i + 1 < len(argv):
            i += 1
            result.model = argv[i]
        elif arg == "--api-key" and i + 1 < len(argv):
            i += 1
            result.api_key = argv[i]
        elif arg == "--system-prompt" and i + 1 < len(argv):
            i += 1
            result.system_prompt = argv[i]
        elif arg == "--append-system-prompt" and i + 1 < len(argv):
            i += 1
            result.append_system_prompt = argv[i]
        elif arg == "--no-session":
            result.no_session = True
        elif arg == "--session" and i + 1 < len(argv):
            i += 1
            result.session = argv[i]
        elif arg == "--fork" and i + 1 < len(argv):
            i += 1
            result.fork = argv[i]
        elif arg == "--session-dir" and i + 1 < len(argv):
            i += 1
            result.session_dir = argv[i]
        elif arg == "--models" and i + 1 < len(argv):
            i += 1
            result.models = [s.strip() for s in argv[i].split(",")]
        elif arg == "--no-tools":
            result.no_tools = True
        elif arg == "--tools" and i + 1 < len(argv):
            i += 1
            tool_names = [s.strip() for s in argv[i].split(",")]
            valid_tools: list[str] = []
            for name in tool_names:
                if name in ALL_TOOL_NAMES:
                    valid_tools.append(name)
                else:
                    result.diagnostics.append(
                        Diagnostic(
                            type="warning",
                            message=(f'Unknown tool "{name}". Valid tools: ' + ", ".join(ALL_TOOL_NAMES)),
                        )
                    )
            result.tools = valid_tools
        elif arg == "--thinking" and i + 1 < len(argv):
            i += 1
            level = argv[i]
            if is_valid_thinking_level(level):
                result.thinking = level  # type: ignore[assignment]
            else:
                result.diagnostics.append(
                    Diagnostic(
                        type="warning",
                        message=(
                            f'Invalid thinking level "{level}". Valid values: '
                            + ", ".join(sorted(_VALID_THINKING_LEVELS))
                        ),
                    )
                )
        elif arg in ("--print", "-p"):
            result.print = True
        elif arg == "--export" and i + 1 < len(argv):
            i += 1
            result.export = argv[i]
        elif arg in ("--extension", "-e") and i + 1 < len(argv):
            i += 1
            result.extensions = result.extensions or []
            result.extensions.append(argv[i])
        elif arg in ("--no-extensions", "-ne"):
            result.no_extensions = True
        elif arg == "--skill" and i + 1 < len(argv):
            i += 1
            result.skills = result.skills or []
            result.skills.append(argv[i])
        elif arg == "--prompt-template" and i + 1 < len(argv):
            i += 1
            result.prompt_templates = result.prompt_templates or []
            result.prompt_templates.append(argv[i])
        elif arg == "--theme" and i + 1 < len(argv):
            i += 1
            result.themes = result.themes or []
            result.themes.append(argv[i])
        elif arg in ("--no-skills", "-ns"):
            result.no_skills = True
        elif arg in ("--no-prompt-templates", "-np"):
            result.no_prompt_templates = True
        elif arg == "--no-themes":
            result.no_themes = True
        elif arg == "--list-models":
            if i + 1 < len(argv) and not argv[i + 1].startswith("-") and not argv[i + 1].startswith("@"):
                i += 1
                result.list_models = argv[i]
            else:
                result.list_models = True
        elif arg == "--verbose":
            result.verbose = True
        elif arg == "--offline":
            result.offline = True
        elif arg.startswith("@"):
            result.file_args.append(arg[1:])
        elif arg.startswith("--"):
            eq_index = arg.find("=")
            if eq_index != -1:
                result.unknown_flags[arg[2:eq_index]] = arg[eq_index + 1 :]
            else:
                flag_name = arg[2:]
                next_arg = argv[i + 1] if i + 1 < len(argv) else None
                if next_arg is not None and not next_arg.startswith("-") and not next_arg.startswith("@"):
                    result.unknown_flags[flag_name] = next_arg
                    i += 1
                else:
                    result.unknown_flags[flag_name] = True
        elif arg.startswith("-") and not arg.startswith("--"):
            result.diagnostics.append(Diagnostic(type="error", message=f"Unknown option: {arg}"))
        else:
            result.messages.append(arg)
        i += 1
    return result


def render_help() -> str:
    """Return the help text printed by ``nu --help`` (without ANSI styling)."""
    return f"""{APP_NAME} - AI coding assistant with read, bash, edit, write tools

Usage:
  {APP_NAME} [options] [@files...] [messages...]

Options:
  --provider <name>              Provider name
  --model <pattern>              Model pattern or ID (supports "provider/id" and optional ":<thinking>")
  --api-key <key>                API key (defaults to env vars)
  --system-prompt <text>         System prompt (default: coding assistant prompt)
  --append-system-prompt <text>  Append text or file contents to the system prompt
  --mode <mode>                  Output mode: text (default), json, or rpc
  --print, -p                    Non-interactive mode: process prompt and exit
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path>               Use specific session file
  --fork <path>                  Fork specific session file or partial UUID into a new session
  --session-dir <dir>            Directory for session storage and lookup
  --no-session                   Don't save session (ephemeral)
  --models <patterns>            Comma-separated model patterns for Ctrl+P cycling
  --no-tools                     Disable all built-in tools
  --tools <tools>                Comma-separated list of tools to enable
                                 Available: {", ".join(ALL_TOOL_NAMES)}
  --thinking <level>             Set thinking level: off, minimal, low, medium, high, xhigh
  --extension, -e <path>         Load an extension file (can be used multiple times)
  --no-extensions, -ne           Disable extension discovery (explicit -e paths still work)
  --skill <path>                 Load a skill file or directory (can be used multiple times)
  --no-skills, -ns               Disable skills discovery and loading
  --prompt-template <path>       Load a prompt template file or directory (can be used multiple times)
  --no-prompt-templates, -np     Disable prompt template discovery and loading
  --theme <path>                 Load a theme file or directory (can be used multiple times)
  --no-themes                    Disable theme discovery and loading
  --export <file>                Export session file to HTML and exit
  --list-models [search]         List available models (with optional fuzzy search)
  --verbose                      Force verbose startup
  --offline                      Disable startup network operations
  --help, -h                     Show this help
  --version, -v                  Show version number

Environment Variables:
  ANTHROPIC_API_KEY                - Anthropic Claude API key
  OPENAI_API_KEY                   - OpenAI GPT API key
  GEMINI_API_KEY                   - Google Gemini API key
  {ENV_AGENT_DIR:<32} - Session storage directory (default: ~/{CONFIG_DIR_NAME}/agent)
"""


__all__ = [
    "Args",
    "Diagnostic",
    "Mode",
    "ThinkingLevel",
    "is_valid_thinking_level",
    "parse_args",
    "render_help",
]
