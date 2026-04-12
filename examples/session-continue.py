"""Session continuation: resume the most recent session.

Demonstrates ``--continue`` semantics programmatically. The first run
creates a persisted session; the second run picks it up automatically.

Run with::

    uv run python examples/session-continue.py "what is 2+2"
    uv run python examples/session-continue.py "and what about 3+3?"

The second invocation resumes the session from the first, so the
model sees the full conversation history.

Set ``OPENAI_API_KEY`` before running.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from nu_agent_core.agent import Agent, AgentOptions
from nu_ai import AssistantMessage, Model, ModelCost, TextContent, get_env_api_key, get_model
from nu_coding_agent.core.agent_session import AgentSession
from nu_coding_agent.core.auth_storage import AuthStorage
from nu_coding_agent.core.model_registry import ModelRegistry
from nu_coding_agent.core.session_manager import SessionManager


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _build_model() -> Model:
    catalog = get_model("openai", "gpt-4o-mini")
    cost = catalog.cost if catalog else ModelCost(input=0, output=0, cache_read=0, cache_write=0)
    return Model(
        id="gpt-4o-mini",
        name="gpt-4o-mini",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text"],
        cost=cost,
        context_window=128_000,
        max_tokens=16_384,
    )


async def main() -> int:
    _load_dotenv()
    if get_env_api_key("openai") is None:
        print("Missing OPENAI_API_KEY", file=sys.stderr)
        return 2

    prompt = " ".join(sys.argv[1:]) or "Hello!"
    model = _build_model()
    cwd = str(Path.cwd())

    # continue_recent: resumes the most recent session, or creates a new one
    sm = SessionManager.continue_recent(cwd)

    auth = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(auth)
    registry._models.append(model)  # type: ignore[attr-defined]

    agent = Agent(AgentOptions(initial_state={"model": model, "system_prompt": "", "tools": []}))
    session = AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=auth,
        cwd=cwd,
    )

    session_file = sm.get_session_file()
    entry_count = len(sm.get_entries())
    print(f"[session] {session_file or '(new)'} ({entry_count} entries)")
    print(f"[prompt]  {prompt}\n")

    try:
        await session.prompt(prompt)
    finally:
        await session.shutdown()

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
