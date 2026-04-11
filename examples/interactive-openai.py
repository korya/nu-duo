"""Minimal interactive REPL example: chat with OpenAI ``gpt-4o-mini``.

Run with::

    just run-example-interactive

Type a message and hit Enter to send. The agent keeps the conversation
history across turns so you can have a back-and-forth. Type ``/quit``,
``/exit``, ``Ctrl-D``, or ``Ctrl-C`` to exit.

Set ``OPENAI_API_KEY`` in a top-level ``.env`` file (or your shell)
before running.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from nu_agent_core.agent import Agent, AgentOptions
from nu_ai import AssistantMessage, Model, ModelCost, get_env_api_key, get_model

MODEL_ID = "gpt-4o-mini"
EXIT_COMMANDS = {"/quit", "/exit", "/q"}


def build_openai_model(model_id: str) -> Model:
    """Build a chat-completions :class:`Model` from the catalog (with sane fallbacks)."""
    catalog = get_model("openai", model_id)
    cost = catalog.cost if catalog else ModelCost(input=0, output=0, cache_read=0, cache_write=0)
    inputs = catalog.input if catalog else ["text"]
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=inputs,  # type: ignore[arg-type]
        cost=cost,
        context_window=catalog.context_window if catalog else 128_000,
        max_tokens=catalog.max_tokens if catalog else 16_384,
    )


def _load_dotenv() -> None:
    """Load ``OPENAI_API_KEY`` from a workspace-root ``.env`` if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


async def main() -> int:
    _load_dotenv()

    if get_env_api_key("openai") is None:
        print(
            "Missing OPENAI_API_KEY — set it in your environment or in a top-level .env file.",
            file=sys.stderr,
        )
        return 2

    model = build_openai_model(MODEL_ID)
    agent = Agent(AgentOptions(initial_state={"model": model}))

    print(f"[model] {MODEL_ID}")
    print("Type a message and hit Enter. /quit, /exit, Ctrl-D, or Ctrl-C to exit.\n", flush=True)

    turn = 1
    while True:
        try:
            prompt = await asyncio.to_thread(input, f"[{turn}] you > ")
        except (EOFError, KeyboardInterrupt):
            print()  # newline so the shell prompt isn't glued to the [you >]
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
