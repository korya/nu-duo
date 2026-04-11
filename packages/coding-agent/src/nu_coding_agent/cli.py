"""``nu`` command-line entry point.

Minimal port of ``packages/coding-agent/src/cli/{args,cli}.ts``. The
upstream version supports interactive, print, json, and rpc modes plus
session management, extensions, skills, and themes. This Python port
currently supports:

* ``nu --print "..."`` — single-shot print mode (no session persistence
  yet — that lands when nu_coding_agent.core.session_manager is ported).
* ``--openai`` (default) / ``--anthropic`` provider shortcuts.
* ``--model <id>`` to override the default model id.
* ``--api-key <key>`` to override env-based credentials.
* ``--system-prompt <text>`` to set the system prompt.
* ``--cwd <path>`` to root the seven tools at a different directory.
* ``--no-tools`` to run without any tools (pure chat).

Keys are loaded from a ``.env`` file via ``python-dotenv`` if one is
present in the current directory or any parent. Override the model id
via ``NU_OPENAI_MODEL`` / ``NU_ANTHROPIC_MODEL``.

The interactive mode and the JSON / RPC modes will land alongside
the session manager in a follow-up phase.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nu_agent_core.agent import Agent, AgentOptions
from nu_ai import (
    AssistantMessage,
    Model,
    ModelCost,
    TextContent,
    get_env_api_key,
    get_model,
)

from nu_coding_agent.core.tools import create_all_tools

if TYPE_CHECKING:
    from nu_agent_core.types import AgentEvent, AgentTool


_DEFAULT_OPENAI_MODEL_ID = os.environ.get("NU_OPENAI_MODEL", "gpt-4o-mini")
_DEFAULT_ANTHROPIC_MODEL_ID = os.environ.get("NU_ANTHROPIC_MODEL", "claude-sonnet-4-5")
_VERSION = "0.0.0"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Args:
    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    cwd: str | None = None
    no_tools: bool = False
    print_mode: bool = False
    show_help: bool = False
    show_version: bool = False
    quiet: bool = False
    positional: list[str] = field(default_factory=list)


def _parse_args(argv: list[str]) -> tuple[_Args, int | None]:
    """Parse ``argv[1:]`` into a structured :class:`_Args`.

    Returns ``(args, exit_code)`` — when ``exit_code`` is not ``None``
    the caller should exit immediately (used for ``--help``/``--version``
    and unknown-flag errors).
    """
    args = _Args()
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in {"-h", "--help"}:
            args.show_help = True
            return args, None
        if arg in {"-v", "--version"}:
            args.show_version = True
            return args, None
        if arg in {"-p", "--print"}:
            args.print_mode = True
        elif arg in {"-q", "--quiet"}:
            args.quiet = True
        elif arg == "--openai":
            args.provider = "openai"
        elif arg == "--anthropic":
            args.provider = "anthropic"
        elif arg == "--no-tools":
            args.no_tools = True
        elif arg == "--model":
            i += 1
            if i >= len(argv):
                _stderr("--model requires a value")
                return args, 2
            args.model = argv[i]
        elif arg == "--api-key":
            i += 1
            if i >= len(argv):
                _stderr("--api-key requires a value")
                return args, 2
            args.api_key = argv[i]
        elif arg == "--system-prompt":
            i += 1
            if i >= len(argv):
                _stderr("--system-prompt requires a value")
                return args, 2
            args.system_prompt = argv[i]
        elif arg == "--cwd":
            i += 1
            if i >= len(argv):
                _stderr("--cwd requires a value")
                return args, 2
            args.cwd = argv[i]
        elif arg.startswith("-"):
            _stderr(f"Unknown flag: {arg}")
            return args, 2
        else:
            args.positional.append(arg)
        i += 1
    return args, None


def _print_help() -> None:
    print(
        """\
nu — minimal coding agent CLI (Python port of pi-coding-agent)

USAGE
  nu [OPTIONS] [PROMPT...]

  When a prompt is provided, runs in print mode by default — sends the
  prompt, streams the response, and exits.

OPTIONS
  -p, --print              Single-shot print mode (default when a prompt
                           is given).
  -q, --quiet              Suppress streaming + tool-call output; only
                           print the final assistant text.

  --openai                 Use OpenAI (default).  Reads OPENAI_API_KEY.
  --anthropic              Use Anthropic.  Reads ANTHROPIC_API_KEY (or
                           ANTHROPIC_OAUTH_TOKEN).

  --model ID               Override the default model id.  Defaults:
                             OpenAI    : gpt-4o-mini
                             Anthropic : claude-sonnet-4-5
                           Override defaults via NU_OPENAI_MODEL /
                           NU_ANTHROPIC_MODEL env vars.

  --api-key KEY            Override env-based credentials.
  --system-prompt TEXT     Set the system prompt.
  --cwd PATH               Root the seven tools at PATH (default: cwd).
  --no-tools               Run without any tools (pure chat).

  -h, --help               Show this help and exit.
  -v, --version            Print version and exit.

EXAMPLES
  nu --print "what is 2 + 2"
  nu -p "summarise README.md in three bullets" --cwd .
  nu --anthropic --print "explain the agent loop" --no-tools
"""
    )


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Provider / model resolution
# ---------------------------------------------------------------------------


def _build_openai_model(model_id: str) -> Model:
    """Build an OpenAI :class:`Model` targeting the Chat Completions API.

    The catalog tags every official OpenAI model as ``openai-responses``
    (the new Responses API), but only the ``openai-completions`` provider
    is currently ported. ``/v1/chat/completions`` still accepts current
    models, so we construct a ``Model`` literal that routes through the
    supported provider with cost metadata copied from the catalog where
    available.
    """
    catalog = get_model("openai", model_id)
    if catalog is not None:
        cost = catalog.cost
        context_window = catalog.context_window
        max_tokens = catalog.max_tokens
        inputs = catalog.input
    else:
        cost = ModelCost(input=0, output=0, cache_read=0, cache_write=0)
        context_window = 128_000
        max_tokens = 16_384
        inputs = ["text", "image"]
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=inputs,  # type: ignore[arg-type]
        cost=cost,
        context_window=context_window,
        max_tokens=max_tokens,
    )


def _build_anthropic_model(model_id: str) -> Model | None:
    return get_model("anthropic", model_id)


def _resolve_model(args: _Args) -> Model | None:
    """Pick the model based on ``--openai``/``--anthropic`` and ``--model``."""
    if args.provider == "openai":
        return _build_openai_model(args.model or _DEFAULT_OPENAI_MODEL_ID)
    return _build_anthropic_model(args.model or _DEFAULT_ANTHROPIC_MODEL_ID)


def _check_credentials(args: _Args) -> str | None:
    """Return an error message string if credentials are missing."""
    if args.api_key:
        return None
    if args.provider == "openai":
        if get_env_api_key("openai") is None:
            return "Missing OPENAI_API_KEY — set it in your environment or in a top-level .env file before running nu."
    elif args.provider == "anthropic" and get_env_api_key("anthropic") is None:
        return (
            "Missing ANTHROPIC_API_KEY (or ANTHROPIC_OAUTH_TOKEN) — set it in "
            "your environment or in a top-level .env file before running nu."
        )
    return None


# ---------------------------------------------------------------------------
# Streaming output renderer (matches the sample app's style)
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… ({len(text) - limit} more chars)"


class _PrintState:
    def __init__(self) -> None:
        self.in_text = False
        self.in_thinking = False
        self.final_text_chunks: list[str] = []


def _handle_assistant_event(
    inner: Any,
    state: _PrintState,
    *,
    quiet: bool,
) -> None:
    inner_type = inner.type
    if inner_type == "text_delta":
        state.final_text_chunks.append(inner.delta)
        if quiet:
            return
        if state.in_thinking:
            print()
            state.in_thinking = False
        if not state.in_text:
            print("\033[2m[text]\033[0m ", end="", flush=True)
            state.in_text = True
        print(inner.delta, end="", flush=True)
    elif inner_type == "text_end":
        if state.in_text and not quiet:
            print(flush=True)
            state.in_text = False
    elif inner_type == "thinking_delta":
        if quiet:
            return
        if state.in_text:
            print()
            state.in_text = False
        if not state.in_thinking:
            print("\033[2m[thinking]\033[0m ", end="", flush=True)
            state.in_thinking = True
        print(f"\033[2m{inner.delta}\033[0m", end="", flush=True)
    elif inner_type == "thinking_end":
        if state.in_thinking and not quiet:
            print(flush=True)
            state.in_thinking = False
    elif inner_type == "toolcall_end":
        if quiet:
            return
        if state.in_text or state.in_thinking:
            print()
            state.in_text = False
            state.in_thinking = False
        tc = inner.tool_call
        args_preview = _truncate(json.dumps(tc.arguments, ensure_ascii=False))
        print(f"\033[2m[tool]\033[0m {tc.name}({args_preview})", flush=True)


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Walk upward from the cwd looking for a ``.env`` file and load it."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415 — optional dep, lazy import
    except ImportError:
        return
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        env_path = candidate / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_cwd_sync(arg: str | None) -> str:
    return str(Path(arg).resolve()) if arg else str(Path.cwd().resolve())


async def _run_print_mode(args: _Args, model: Model) -> int:
    cwd = await asyncio.to_thread(_resolve_cwd_sync, args.cwd)
    tools: list[AgentTool[Any, Any]] = [] if args.no_tools else create_all_tools(cwd)

    initial_state: dict[str, Any] = {"model": model, "tools": tools}
    if args.system_prompt is not None:
        initial_state["system_prompt"] = args.system_prompt

    agent = Agent(AgentOptions(initial_state=initial_state))

    if not args.quiet:
        print(f"\033[2m[provider]\033[0m {args.provider}")
        print(f"\033[2m[model]   \033[0m {model.id}")
        print(f"\033[2m[cwd]     \033[0m {cwd}")
        prompt = " ".join(args.positional)
        print(f"\033[2m[task]    \033[0m {prompt}\n", flush=True)

    state = _PrintState()

    async def listener(event: AgentEvent, _signal: Any) -> None:
        if event["type"] == "message_update":
            _handle_assistant_event(event["assistant_message_event"], state, quiet=args.quiet)
            return
        if event["type"] == "tool_execution_end" and not args.quiet:
            text_chunks: list[str] = [
                block.text
                for block in (getattr(event["result"], "content", None) or [])
                if isinstance(block, TextContent)
            ]
            preview = _truncate("\n".join(text_chunks))
            status = "error" if event["is_error"] else "ok"
            print(f"\033[2m[result {status}]\033[0m {preview}", flush=True)
            return
        if event["type"] == "agent_end" and not args.quiet:
            print(flush=True)

    agent.subscribe(listener)

    prompt = " ".join(args.positional)
    try:
        await agent.prompt(prompt)
    except KeyboardInterrupt:
        agent.abort()
        await agent.wait_for_idle()
        _stderr("[aborted]")
        return 130

    final = agent.state.messages[-1] if agent.state.messages else None
    if isinstance(final, AssistantMessage) and final.stop_reason == "error":
        _stderr(f"[error] {final.error_message or 'unknown error'}")
        return 1

    # In quiet mode the streaming printer never echoed text — print the
    # accumulated final response now.
    if args.quiet and state.final_text_chunks:
        sys.stdout.write("".join(state.final_text_chunks))
        sys.stdout.write("\n")
        sys.stdout.flush()

    return 0


async def _async_main(argv: list[str]) -> int:
    _load_dotenv()
    args, exit_code = _parse_args(argv)
    if exit_code is not None:
        return exit_code
    if args.show_help:
        _print_help()
        return 0
    if args.show_version:
        print(f"nu {_VERSION}")
        return 0

    if not args.positional:
        _print_help()
        return 0

    error = _check_credentials(args)
    if error is not None:
        _stderr(error)
        return 2

    model = _resolve_model(args)
    if model is None:
        target = args.model or (_DEFAULT_OPENAI_MODEL_ID if args.provider == "openai" else _DEFAULT_ANTHROPIC_MODEL_ID)
        _stderr(f"Unknown {args.provider} model id: {target}")
        return 2

    return await _run_print_mode(args, model)


def main() -> None:
    """Synchronous entry point used by the ``nu`` console script."""
    sys.exit(asyncio.run(_async_main(list(sys.argv))))


if __name__ == "__main__":
    main()
