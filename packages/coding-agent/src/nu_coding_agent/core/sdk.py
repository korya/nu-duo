"""SDK — programmatic entry point for creating agent sessions.

Port of ``packages/coding-agent/src/core/sdk.ts``. Provides a
single high-level factory :func:`create_agent_session` that builds
everything (auth, registry, session manager, extensions, agent,
session) from a minimal set of options. This is the recommended
entry point for embedding ``nu_coding_agent`` in other applications.

Usage::

    from nu_coding_agent.core.sdk import create_agent_session

    result = await create_agent_session(cwd="/my/project")
    session = result.session
    await session.prompt("what files are in this directory?")
    await session.shutdown()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nu_agent_core.agent import Agent, AgentOptions

from nu_coding_agent.config import get_agent_dir
from nu_coding_agent.core.agent_session import AgentSession
from nu_coding_agent.core.auth_storage import AuthStorage
from nu_coding_agent.core.defaults import DEFAULT_THINKING_LEVEL
from nu_coding_agent.core.messages import convert_to_llm
from nu_coding_agent.core.model_registry import ModelRegistry
from nu_coding_agent.core.model_resolver import find_initial_model
from nu_coding_agent.core.session_manager import SessionManager, get_default_session_dir
from nu_coding_agent.core.settings_manager import SettingsManager
from nu_coding_agent.core.system_prompt import BuildSystemPromptOptions, build_system_prompt
from nu_coding_agent.core.tools import create_all_tools

if TYPE_CHECKING:
    from nu_coding_agent.core.extensions import ExtensionRunner


@dataclass(slots=True)
class CreateAgentSessionOptions:
    """Options for :func:`create_agent_session`."""

    cwd: str | None = None
    agent_dir: str | None = None
    auth_storage: AuthStorage | None = None
    model_registry: ModelRegistry | None = None
    model: Any = None  # Model | None
    thinking_level: str | None = None  # ThinkingLevel | None
    tools: list[Any] | None = None
    session_manager: SessionManager | None = None
    settings_manager: SettingsManager | None = None
    system_prompt: str | None = None
    no_tools: bool = False
    api_key: str | None = None
    provider: str | None = None


@dataclass(slots=True)
class CreateAgentSessionResult:
    """Result from :func:`create_agent_session`."""

    session: AgentSession
    extensions_result: Any = None  # LoadExtensionsResult | None
    model_fallback_message: str | None = None


async def create_agent_session(
    options: CreateAgentSessionOptions | None = None,
) -> CreateAgentSessionResult:
    """Create an AgentSession from the supplied options.

    This is the high-level "one-call" factory. It builds or reuses
    AuthStorage, ModelRegistry, SessionManager, and SettingsManager,
    discovers extensions, resolves the initial model, creates the
    Agent, and wires everything into an AgentSession.
    """
    opts = options or CreateAgentSessionOptions()
    cwd = opts.cwd or os.getcwd()
    agent_dir = opts.agent_dir or get_agent_dir()

    # Auth
    auth_storage = opts.auth_storage or AuthStorage.create()
    if opts.api_key and opts.provider:
        auth_storage.set_runtime_api_key(opts.provider, opts.api_key)

    # Registry
    model_registry = opts.model_registry or ModelRegistry.create(auth_storage)

    # Settings
    settings_manager = opts.settings_manager or SettingsManager.create(cwd, agent_dir)

    # Session
    session_manager = opts.session_manager or SessionManager.create(cwd, get_default_session_dir(cwd, agent_dir))

    # Extensions
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        ExtensionRunner,
        discover_and_load_extensions,
    )

    ext_result = await discover_and_load_extensions()
    runner: ExtensionRunner | None = None
    if ext_result.extensions:
        runner = ExtensionRunner.create(
            extensions=ext_result.extensions,
            runtime=ext_result.runtime,
            cwd=cwd,
        )

    # Model resolution
    model = opts.model
    model_fallback_message: str | None = None

    # Try to restore from existing session
    existing = session_manager.build_session_context()
    has_existing = len(existing.messages) > 0

    if model is None and has_existing and existing.model is not None:
        restored = model_registry.find(
            existing.model.get("provider", ""),
            existing.model.get("modelId", ""),
        )
        if restored is not None and model_registry.has_configured_auth(restored):
            model = restored
        else:
            provider = existing.model.get("provider", "?")
            model_id = existing.model.get("modelId", "?")
            model_fallback_message = f"Could not restore model {provider}/{model_id}"

    if model is None:
        result = find_initial_model(
            cli_provider=opts.provider,
            cli_model=None,
            scoped_models=[],
            is_continuing=has_existing,
            default_provider=settings_manager.get_default_provider(),
            default_model_id=settings_manager.get_default_model(),
            default_thinking_level=None,
            model_registry=model_registry,
        )
        model = getattr(result, "model", None)
        if model is None and model_fallback_message:
            model_fallback_message += ". No models available."

    # Thinking level
    thinking_level = opts.thinking_level or DEFAULT_THINKING_LEVEL
    if model is None or not getattr(model, "reasoning", False):
        thinking_level = "off"

    # Tools
    tools: list[Any] = []
    if not opts.no_tools:
        tools = opts.tools or list(create_all_tools(cwd))

    # System prompt
    system_prompt = build_system_prompt(
        BuildSystemPromptOptions(
            custom_prompt=opts.system_prompt,
            tools=tools,
            cwd=cwd,
        )
    )

    # Agent
    agent = Agent(
        AgentOptions(
            initial_state={
                "model": model,
                "tools": tools,
                "system_prompt": system_prompt,
                "thinking_level": thinking_level,
            },
            convert_to_llm=_make_convert_to_llm(),
            stream_fn=_make_stream_fn(model_registry),
        )
    )

    # Restore messages from existing session
    if has_existing:
        agent.state.messages = existing.messages

    # Session
    session = AgentSession.create(
        agent=agent,
        session_manager=session_manager,
        model_registry=model_registry,
        auth_storage=auth_storage,
        cwd=cwd,
        extension_runner=runner,
    )

    # Apply extension tools
    session.apply_extension_tools()

    return CreateAgentSessionResult(
        session=session,
        extensions_result=ext_result,
        model_fallback_message=model_fallback_message,
    )


def _make_convert_to_llm():
    """Build the convertToLlm function for the Agent."""

    async def converter(messages: list[Any]) -> list[Any]:
        return convert_to_llm(messages)

    return converter


def _make_stream_fn(model_registry: ModelRegistry):
    """Build the streamFn for the Agent using the model registry for auth."""
    from nu_ai import stream_simple  # noqa: PLC0415

    def stream_fn(model: Any, context: Any, options: Any | None = None):
        # The stream function needs to be synchronous for the Agent constructor
        # but the actual streaming is async. The Agent's agent loop handles this.
        return stream_simple(model, context, options)

    return stream_fn


__all__ = [
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    "create_agent_session",
]
