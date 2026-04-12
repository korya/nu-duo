"""SDK example: create an agent session programmatically.

Demonstrates the high-level ``create_agent_session`` factory from
``nu_coding_agent.core.sdk`` — the recommended way to build a session
for embedding in other applications.

Run with::

    uv run python examples/sdk-session.py
    uv run python examples/sdk-session.py "your prompt here"

Set ``OPENAI_API_KEY`` in a top-level ``.env`` file before running.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv


async def main() -> int:
    # Load .env from workspace root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    from nu_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session

    prompt = " ".join(sys.argv[1:]) or "List the Python files in the current directory."

    # One call builds everything: auth, registry, session manager, tools, agent.
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(Path.cwd()),
            provider="openai",
        )
    )

    session = result.session
    if result.model_fallback_message:
        print(f"⚠ {result.model_fallback_message}", file=sys.stderr)

    model = session.model
    print(f"[model] {model.provider}/{model.id}" if model else "[model] none")
    print(f"[task]  {prompt}\n")

    try:
        await session.prompt(prompt)
    finally:
        await session.shutdown()

    # Print the final assistant text
    from nu_ai.types import AssistantMessage, TextContent

    messages = session.agent.state.messages
    if messages:
        last = messages[-1]
        if isinstance(last, AssistantMessage):
            for block in last.content:
                if isinstance(block, TextContent):
                    print(block.text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
