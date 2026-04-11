"""Minimal interactive REPL example: chat with Anthropic ``claude-haiku-4-5``.

Run with::

    uv run python examples/interactive-anthropic.py

Type a message and hit Enter to send. The agent keeps the conversation
history across turns so you can have a back-and-forth. Type ``/quit``,
``/exit``, ``Ctrl-D``, or ``Ctrl-C`` to exit.

Set ``ANTHROPIC_API_KEY`` (or ``ANTHROPIC_OAUTH_TOKEN``) in a top-level
``.env`` file (or your shell) before running.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from nu_agent_core.agent import Agent, AgentOptions
from nu_ai import AssistantMessage, get_env_api_key, get_model

MODEL_ID = "claude-haiku-4-5"
EXIT_COMMANDS = {"/quit", "/exit", "/q"}


def _load_dotenv() -> None:
    """Load ``ANTHROPIC_API_KEY`` from a workspace-root ``.env`` if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


async def main() -> int:
    _load_dotenv()

    if get_env_api_key("anthropic") is None:
        print(
            "Missing ANTHROPIC_API_KEY (or ANTHROPIC_OAUTH_TOKEN) — set it in your "
            "environment or in a top-level .env file.",
            file=sys.stderr,
        )
        return 2

    model = get_model("anthropic", MODEL_ID)
    if model is None:
        print(f"Unknown Anthropic model id: {MODEL_ID}", file=sys.stderr)
        return 2

    agent = Agent(AgentOptions(initial_state={"model": model}))

    print(f"[model] {MODEL_ID}")
    print("Type a message and hit Enter. /quit, /exit, Ctrl-D, or Ctrl-C to exit.\n", flush=True)

    turn = 1
    while True:
        try:
            prompt = await asyncio.to_thread(input, f"[{turn}] you > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt in EXIT_COMMANDS:
            return 0

        try:
            await agent.prompt(prompt)
        except KeyboardInterrupt:
            agent.abort()
            await agent.wait_for_idle()
            print("\n[aborted]")
            continue

        final = agent.state.messages[-1] if agent.state.messages else None
        if isinstance(final, AssistantMessage):
            if final.stop_reason == "error":
                print(f"[error] {final.error_message or 'unknown error'}", file=sys.stderr)
            else:
                for block in final.content:
                    text = getattr(block, "text", None)
                    if text:
                        print(f"[agent] {text}")
        print()
        turn += 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
