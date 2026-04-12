"""Custom tool example: register a tool via the extension API.

Demonstrates how extensions can ship LLM-callable tools. The example
registers a ``weather`` tool that the LLM can call, and the tool
returns fake weather data.

Run with::

    uv run python examples/custom-tool.py

Set ``OPENAI_API_KEY`` before running.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from nu_agent_core.agent import Agent, AgentOptions
from nu_agent_core.types import AgentTool, AgentToolResult
from nu_ai import AssistantMessage, Model, ModelCost, TextContent, get_env_api_key, get_model
from nu_coding_agent.core.agent_session import AgentSession
from nu_coding_agent.core.auth_storage import AuthStorage
from nu_coding_agent.core.extensions import (
    ExtensionAPI,
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
# Extension that registers a custom "weather" tool
# ---------------------------------------------------------------------------


def weather_extension(api: ExtensionAPI) -> None:
    """Extension that provides a fake weather lookup tool."""

    async def execute(
        tool_call_id: str,
        params: Any,
        signal: Any,
        on_update: Any,
    ) -> AgentToolResult[Any]:
        city = params.get("city", "Unknown")
        # Fake weather data
        data = {
            "city": city,
            "temperature": "22°C",
            "conditions": "Partly cloudy",
            "humidity": "45%",
        }
        return AgentToolResult(
            content=[TextContent(text=json.dumps(data, indent=2))],
            details=None,
        )

    tool = AgentTool(
        name="weather",
        description="Get the current weather for a city. Returns temperature, conditions, and humidity.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
        label="Weather",
        execute=execute,
        prompt_snippet="weather(city) — look up current weather for a city",
    )

    api.register_tool(tool)


async def main() -> int:
    _load_dotenv()
    if get_env_api_key("openai") is None:
        print("Missing OPENAI_API_KEY", file=sys.stderr)
        return 2

    # Load extension with the weather tool
    result = await load_extensions_from_factories(
        [
            ("<inline:weather>", weather_extension),
        ]
    )
    runner = ExtensionRunner.create(
        extensions=result.extensions,
        runtime=result.runtime,
        cwd=str(Path.cwd()),
    )

    # Build model + session
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

    agent = Agent(
        AgentOptions(
            initial_state={
                "model": model,
                "system_prompt": "You have a weather tool. Use it when asked about weather.",
                "tools": [],
            }
        )
    )
    sm = SessionManager.in_memory()
    session = AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=auth,
        cwd=str(Path.cwd()),
        extension_runner=runner,
    )

    # Apply extension tools so the agent can use them
    applied = session.apply_extension_tools()
    print(f"[tools] {applied} extension tool(s) registered")
    print(f"[tools] {[t.name for t in agent.state.tools]}")
    print()

    prompt = "What's the weather like in Tokyo?"
    print(f"[prompt] {prompt}\n")

    # Subscribe to see tool calls
    def on_event(event: Any) -> None:
        if event["type"] == "tool_execution_end":
            name = event.get("tool_name", "?")
            print(f"  [tool called] {name}")

    session.subscribe(on_event)

    try:
        await session.prompt(prompt)
    finally:
        await session.shutdown()

    # Print result
    messages = session.agent.state.messages
    if messages:
        last = messages[-1]
        if isinstance(last, AssistantMessage):
            for block in last.content:
                if isinstance(block, TextContent):
                    print(f"\n{block.text}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
