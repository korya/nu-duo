"""Extension example: a simple extension that reacts to agent events.

Demonstrates the extension API: load an in-process factory, attach it
to an AgentSession, and observe lifecycle events (agent_start,
message_end, session_shutdown) from a real prompt.

Run with::

    uv run python examples/extension-hello.py

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
from nu_coding_agent.core.extensions import (
    ExtensionAPI,
    ExtensionContext,
    ExtensionRunner,
    load_extensions_from_factories,
)
from nu_coding_agent.core.model_registry import ModelRegistry
from nu_coding_agent.core.session_manager import SessionManager


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


# ---------------------------------------------------------------------------
# The extension factory — this is what an extension package exports.
# ---------------------------------------------------------------------------


def my_extension(api: ExtensionAPI) -> None:
    """A simple extension that prints lifecycle events."""

    def on_agent_start(event: object, ctx: ExtensionContext) -> None:
        print("  [extension] 🚀 Agent started!")

    def on_message_end(event: object, ctx: ExtensionContext) -> None:
        msg = getattr(event, "message", None)
        role = getattr(msg, "role", None) if msg else None
        if role == "assistant":
            print("  [extension] 💬 Assistant finished speaking")

    def on_shutdown(event: object, ctx: ExtensionContext) -> None:
        print("  [extension] 👋 Session shutting down")

    api.on("agent_start", on_agent_start)
    api.on("message_end", on_message_end)
    api.on("session_shutdown", on_shutdown)


async def main() -> int:
    _load_dotenv()
    if get_env_api_key("openai") is None:
        print("Missing OPENAI_API_KEY", file=sys.stderr)
        return 2

    # Load the extension
    result = await load_extensions_from_factories(
        [
            ("<inline:hello-extension>", my_extension),
        ]
    )
    runner = ExtensionRunner.create(
        extensions=result.extensions,
        runtime=result.runtime,
        cwd=str(Path.cwd()),
    )
    print(f"Loaded {len(result.extensions)} extension(s)\n")

    # Build a model + session
    catalog = get_model("openai", "gpt-4o-mini")
    cost = catalog.cost if catalog else ModelCost(input=0, output=0, cache_read=0, cache_write=0)
    model = Model(
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

    auth = AuthStorage.in_memory()
    registry = ModelRegistry.in_memory(auth)
    registry._models.append(model)  # type: ignore[attr-defined]

    agent = Agent(AgentOptions(initial_state={"model": model, "system_prompt": "", "tools": []}))
    sm = SessionManager.in_memory()
    session = AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=auth,
        cwd=str(Path.cwd()),
        extension_runner=runner,
    )

    print("Prompting: 'Say hello in exactly 5 words.'\n")
    try:
        await session.prompt("Say hello in exactly 5 words.")
    finally:
        await session.shutdown()

    # Print result
    messages = session.agent.state.messages
    if messages:
        last = messages[-1]
        if isinstance(last, AssistantMessage):
            for block in last.content:
                if isinstance(block, TextContent):
                    print(f"\n  Assistant: {block.text}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
