"""Tests for ``nu_coding_agent.modes.print_mode``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nu_agent_core.agent import Agent, AgentOptions
from nu_ai.api_registry import get_api_provider
from nu_ai.providers.faux import (
    faux_assistant_message,
    register_faux_provider,
)
from nu_ai.types import AssistantMessage, Message, ToolResultMessage, UserMessage
from nu_coding_agent.modes.print_mode import (
    PrintModeOptions,
    run_print_mode,
)

if TYPE_CHECKING:
    import pytest
    from nu_ai.utils.event_stream import AssistantMessageEventStream


def _make_stream_fn(api: str):
    provider = get_api_provider(api)
    assert provider is not None

    def stream_fn(model: Any, context: Any, options: Any | None = None) -> AssistantMessageEventStream:
        return provider.stream_simple(model, context, options)

    return stream_fn


async def _convert_to_llm(messages: list[Any]) -> list[Message]:
    return [m for m in messages if isinstance(m, UserMessage | AssistantMessage | ToolResultMessage)]


def _build_agent(registration: Any) -> Agent:
    options = AgentOptions(
        initial_state={
            "model": registration.get_model(),
            "system_prompt": "",
            "tools": [],
        },
        convert_to_llm=_convert_to_llm,
        stream_fn=_make_stream_fn(registration.api),
    )
    return Agent(options)


async def test_print_mode_text_writes_assistant_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("hello world")])
        agent = _build_agent(registration)

        async def send(text: str, images: Any) -> None:
            await agent.prompt(text)
            await agent.wait_for_idle()

        rc = await run_print_mode(
            agent,
            send,
            PrintModeOptions(mode="text", initial_message="hi"),
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "hello world" in captured.out
    finally:
        registration.unregister()


async def test_print_mode_text_no_messages_returns_zero() -> None:
    registration = register_faux_provider()
    try:
        agent = _build_agent(registration)

        async def send(text: str, images: Any) -> None:
            return None

        rc = await run_print_mode(agent, send, PrintModeOptions(mode="text"))
        assert rc == 0
    finally:
        registration.unregister()


async def test_print_mode_text_error_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("partial", stop_reason="error", error_message="boom")])
        agent = _build_agent(registration)

        async def send(text: str, images: Any) -> None:
            await agent.prompt(text)
            await agent.wait_for_idle()

        rc = await run_print_mode(agent, send, PrintModeOptions(mode="text", initial_message="x"))
        captured = capsys.readouterr()
        assert rc == 1
        assert "boom" in captured.err
    finally:
        registration.unregister()


async def test_print_mode_json_writes_event_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("ack")])
        agent = _build_agent(registration)

        async def send(text: str, images: Any) -> None:
            await agent.prompt(text)
            await agent.wait_for_idle()

        rc = await run_print_mode(
            agent,
            send,
            PrintModeOptions(mode="json", initial_message="hi"),
        )
        captured = capsys.readouterr()
        assert rc == 0
        # JSON mode emits at least one line per event.
        assert captured.out
        assert "\n" in captured.out
    finally:
        registration.unregister()


async def test_print_mode_handles_send_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registration = register_faux_provider()
    try:
        agent = _build_agent(registration)

        async def boom(text: str, images: Any) -> None:
            raise RuntimeError("kaboom")

        rc = await run_print_mode(
            agent,
            boom,
            PrintModeOptions(mode="text", initial_message="hi"),
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "kaboom" in captured.err
    finally:
        registration.unregister()


async def test_print_mode_runs_extra_messages(capsys: pytest.CaptureFixture[str]) -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses(
            [
                faux_assistant_message("first reply"),
                faux_assistant_message("second reply"),
            ]
        )
        agent = _build_agent(registration)

        async def send(text: str, images: Any) -> None:
            await agent.prompt(text)
            await agent.wait_for_idle()

        rc = await run_print_mode(
            agent,
            send,
            PrintModeOptions(mode="text", initial_message="one", messages=["two"]),
        )
        captured = capsys.readouterr()
        assert rc == 0
        # Only the final assistant text appears in text mode.
        assert "second reply" in captured.out
    finally:
        registration.unregister()
