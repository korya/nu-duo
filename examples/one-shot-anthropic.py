"""Minimal one-shot example: a single prompt against Anthropic ``claude-haiku-4-5``.

Run with::

    just run-one-shot-anthropic                            # uses the default prompt
    just run-one-shot-anthropic "your prompt here"

Uses ``claude-haiku-4-5`` — Anthropic's cheapest current-generation
model ($1/M input, $5/M output as of writing). The model lookup goes
through :func:`nu_ai.get_model` which returns a fully-populated
:class:`Model` from the bundled catalog.

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
DEFAULT_PROMPT = "What is the capital of Iceland?"


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

    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT
    agent = Agent(AgentOptions(initial_state={"model": model}))

    print(f"[model] {MODEL_ID}\n[task]  {prompt}\n", flush=True)
    await agent.prompt(prompt)

    final = agent.state.messages[-1] if agent.state.messages else None
    if isinstance(final, AssistantMessage):
        if final.stop_reason == "error":
            print(f"[error] {final.error_message or 'unknown error'}", file=sys.stderr)
            return 1
        for block in final.content:
            text = getattr(block, "text", None)
            if text:
                print(text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
