"""Google Generative AI example: one-shot prompt against Gemini.

Demonstrates using a non-OpenAI/Anthropic provider. The Google
provider uses the ``google-genai`` SDK.

Run with::

    uv run python examples/google-provider.py
    uv run python examples/google-provider.py "explain quantum computing"

Set ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) before running.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from nu_agent_core.agent import Agent, AgentOptions
from nu_ai import AssistantMessage, TextContent, get_env_api_key, get_model

MODEL_ID = "gemini-2.0-flash"
DEFAULT_PROMPT = "What are three interesting facts about the Mariana Trench?"


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


async def main() -> int:
    _load_dotenv()

    if get_env_api_key("google") is None:
        print("Missing GOOGLE_API_KEY or GEMINI_API_KEY", file=sys.stderr)
        return 2

    model = get_model("google", MODEL_ID)
    if model is None:
        print(f"Model {MODEL_ID} not found in catalog", file=sys.stderr)
        return 2

    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT

    agent = Agent(AgentOptions(initial_state={"model": model}))
    print(f"[model] {model.provider}/{model.id}")
    print(f"[task]  {prompt}\n")

    await agent.prompt(prompt)

    final = agent.state.messages[-1] if agent.state.messages else None
    if isinstance(final, AssistantMessage):
        if final.stop_reason == "error":
            print(f"[error] {final.error_message or 'unknown'}", file=sys.stderr)
            return 1
        for block in final.content:
            if isinstance(block, TextContent):
                print(block.text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
