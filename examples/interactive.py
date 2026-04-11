"""Interactive REPL sample exercising the four core tools against a real provider.

Multi-turn variant of ``one-shot.py``. Provider switching via
``--openai`` / ``--anthropic`` flags. For minimal "just run a single
provider" examples see ``interactive-openai.py`` /
``interactive-anthropic.py``.

Run with::

    just run-example-interactive                          # OpenAI (default)
    uv run python examples/interactive.py --anthropic
    uv run python examples/interactive.py --openai /path/to/repo

CLI shape::

    interactive.py [--openai|--anthropic] [REPO]

Defaults:

* Provider — ``--openai`` (``gpt-4o-mini``).
* Repo     — the workspace root.

Type messages at the ``[N] you >`` prompt. Type ``/quit``, ``/exit``,
``/q``, ``Ctrl-D``, or ``Ctrl-C`` to exit. ``Ctrl-C`` mid-turn aborts
the current request without exiting.

API keys are loaded from a top-level ``.env`` file via :mod:`dotenv`.
Recognized variables:

* ``OPENAI_API_KEY`` for ``--openai``
* ``ANTHROPIC_API_KEY`` (or ``ANTHROPIC_OAUTH_TOKEN``) for ``--anthropic``

Override the model with ``NU_SAMPLE_OPENAI_MODEL`` /
``NU_SAMPLE_ANTHROPIC_MODEL``.

The sample subscribes to the agent's event stream and prints text and
thinking deltas live so you can see the model working — without that the
script would appear to hang for several seconds. Tool calls are summarised
as ``[tool_name(args)]`` lines and tool results as truncated previews.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from nu_agent_core.agent import Agent, AgentOptions
from nu_ai import (
    AssistantMessage,
    Model,
    ModelCost,
    TextContent,
    get_env_api_key,
    get_model,
)
from nu_coding_agent.core.tools.bash import create_bash_tool
from nu_coding_agent.core.tools.edit import create_edit_tool
from nu_coding_agent.core.tools.read import create_read_tool
from nu_coding_agent.core.tools.write import create_write_tool

if TYPE_CHECKING:
    from nu_agent_core.types import AgentEvent

_DEFAULT_OPENAI_MODEL_ID = os.environ.get("NU_SAMPLE_OPENAI_MODEL", "gpt-4o-mini")
_DEFAULT_ANTHROPIC_MODEL_ID = os.environ.get("NU_SAMPLE_ANTHROPIC_MODEL", "claude-haiku-4-5")

_EXIT_COMMANDS = {"/quit", "/exit", "/q"}


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def _build_openai_model(model_id: str) -> Model:
    """Build an OpenAI ``Model`` targeting the Chat Completions endpoint."""
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


# ---------------------------------------------------------------------------
# Streaming output renderer (same as one-shot.py)
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… ({len(text) - limit} more chars)"


class _PrintState:
    """Track which streamed block we're currently rendering."""

    def __init__(self) -> None:
        self.in_text = False
        self.in_thinking = False

    def reset(self) -> None:
        self.in_text = False
        self.in_thinking = False


def _handle_assistant_event(inner: Any, state: _PrintState) -> None:
    """Render one streamed assistant event to stdout."""
    inner_type = inner.type
    if inner_type == "text_start":
        if state.in_thinking:
            print()
            state.in_thinking = False
        if not state.in_text:
            print("\033[2m[text]\033[0m ", end="", flush=True)
            state.in_text = True
    elif inner_type == "text_delta":
        print(inner.delta, end="", flush=True)
    elif inner_type == "text_end":
        if state.in_text:
            print(flush=True)
            state.in_text = False
    elif inner_type == "thinking_start":
        if state.in_text:
            print()
            state.in_text = False
        if not state.in_thinking:
            print("\033[2m[thinking]\033[0m ", end="", flush=True)
            state.in_thinking = True
    elif inner_type == "thinking_delta":
        print(f"\033[2m{inner.delta}\033[0m", end="", flush=True)
    elif inner_type == "thinking_end":
        if state.in_thinking:
            print(flush=True)
            state.in_thinking = False
    elif inner_type == "toolcall_end":
        if state.in_text or state.in_thinking:
            print()
            state.in_text = False
            state.in_thinking = False
        tc = inner.tool_call
        args_preview = _truncate(json.dumps(tc.arguments, ensure_ascii=False))
        print(f"\033[2m[tool]\033[0m {tc.name}({args_preview})", flush=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class _ParsedArgs:
    def __init__(self, *, provider: str, repo_arg: str | None) -> None:
        self.provider = provider
        self.repo_arg = repo_arg


def _parse_args(argv: list[str]) -> _ParsedArgs | int:
    """Parse the provider flag plus an optional positional ``[REPO]``."""
    provider = "openai"
    positional: list[str] = []
    for arg in argv[1:]:
        if arg in {"-h", "--help"}:
            print(__doc__)
            return 0
        if arg == "--openai":
            provider = "openai"
        elif arg == "--anthropic":
            provider = "anthropic"
        elif arg.startswith("--"):
            print(f"Unknown flag: {arg}", file=sys.stderr)
            return 2
        else:
            positional.append(arg)

    repo_arg = positional[0] if positional else None
    return _ParsedArgs(provider=provider, repo_arg=repo_arg)


def _resolve_repo(arg: str | None) -> Path:
    if arg is not None:
        return Path(arg).resolve()
    return Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


# ---------------------------------------------------------------------------
# Main REPL loop
# ---------------------------------------------------------------------------


async def _main() -> int:
    _load_dotenv()
    parsed = _parse_args(sys.argv)
    if isinstance(parsed, int):
        return parsed

    if parsed.provider == "openai":
        if get_env_api_key("openai") is None:
            print(
                "Missing OPENAI_API_KEY — set it in your environment or in a top-level "
                ".env file before running this sample.",
                file=sys.stderr,
            )
            return 2
        model: Model | None = _build_openai_model(_DEFAULT_OPENAI_MODEL_ID)
    else:
        if get_env_api_key("anthropic") is None:
            print(
                "Missing ANTHROPIC_API_KEY (or ANTHROPIC_OAUTH_TOKEN) — set it in your "
                "environment or in a top-level .env file before running this sample.",
                file=sys.stderr,
            )
            return 2
        model = _build_anthropic_model(_DEFAULT_ANTHROPIC_MODEL_ID)
        if model is None:
            print(f"Unknown Anthropic model id: {_DEFAULT_ANTHROPIC_MODEL_ID}", file=sys.stderr)
            return 2

    repo = await asyncio.to_thread(_resolve_repo, parsed.repo_arg)
    if not await asyncio.to_thread(repo.exists):
        print(f"Repo path does not exist: {repo}", file=sys.stderr)
        return 2

    print(f"\033[2m[provider]\033[0m {parsed.provider}")
    print(f"\033[2m[model]   \033[0m {model.id}")
    print(f"\033[2m[cwd]     \033[0m {repo}")
    print(
        "\033[2m[hint]    \033[0m type a message; /quit, /exit, Ctrl-D, or Ctrl-C to exit\n",
        flush=True,
    )

    agent = Agent(
        AgentOptions(
            initial_state={
                "model": model,
                "tools": [
                    create_read_tool(str(repo)),
                    create_write_tool(str(repo)),
                    create_edit_tool(str(repo)),
                    create_bash_tool(str(repo)),
                ],
            },
        )
    )

    state = _PrintState()

    async def listener(event: AgentEvent, _signal: Any) -> None:
        if event["type"] == "message_update":
            _handle_assistant_event(event["assistant_message_event"], state)
            return
        if event["type"] == "tool_execution_end":
            text_chunks: list[str] = [
                block.text
                for block in (getattr(event["result"], "content", None) or [])
                if isinstance(block, TextContent)
            ]
            preview = _truncate("\n".join(text_chunks))
            status = "error" if event["is_error"] else "ok"
            print(f"\033[2m[result {status}]\033[0m {preview}", flush=True)
            return
        if event["type"] == "agent_end":
            print(flush=True)

    agent.subscribe(listener)

    turn = 1
    while True:
        try:
            prompt = await asyncio.to_thread(input, f"\033[1m[{turn}] you >\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt in _EXIT_COMMANDS:
            return 0

        state.reset()
        try:
            await agent.prompt(prompt)
        except KeyboardInterrupt:
            agent.abort()
            await agent.wait_for_idle()
            print("\n\033[2m[aborted]\033[0m", flush=True)
            turn += 1
            continue

        final = agent.state.messages[-1] if agent.state.messages else None
        if isinstance(final, AssistantMessage) and final.stop_reason == "error":
            print(
                f"\n\033[31m[error]\033[0m {final.error_message or 'unknown error'}",
                file=sys.stderr,
            )
        turn += 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
