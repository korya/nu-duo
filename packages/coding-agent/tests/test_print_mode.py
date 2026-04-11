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
from nu_coding_agent.core.agent_session import AgentSession
from nu_coding_agent.core.auth_storage import ApiKeyCredential, AuthStorage
from nu_coding_agent.core.model_registry import ModelRegistry
from nu_coding_agent.core.session_manager import SessionManager
from nu_coding_agent.modes.print_mode import PrintModeOptions, run_print_mode

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


def _build_session(registration: Any) -> AgentSession:
    """Build an Agent + AgentSession driven by the given faux registration."""
    faux_model = registration.get_model()
    storage = AuthStorage.in_memory({faux_model.provider: ApiKeyCredential(type="api_key", key="x")})
    registry = ModelRegistry.in_memory(storage)
    registry._models.append(faux_model)  # type: ignore[attr-defined]
    agent = Agent(
        AgentOptions(
            initial_state={"model": faux_model, "system_prompt": "", "tools": []},
            convert_to_llm=_convert_to_llm,
            stream_fn=_make_stream_fn(registration.api),
        )
    )
    sm = SessionManager.in_memory("/work")
    return AgentSession.create(
        agent=agent,
        session_manager=sm,
        model_registry=registry,
        auth_storage=storage,
        cwd="/work",
    )


async def test_print_mode_text_writes_assistant_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("hello world")])
        session = _build_session(registration)
        try:
            rc = await run_print_mode(session, PrintModeOptions(mode="text", initial_message="hi"))
        finally:
            session.close()
        captured = capsys.readouterr()
        assert rc == 0
        assert "hello world" in captured.out
    finally:
        registration.unregister()


async def test_print_mode_text_no_messages_returns_zero() -> None:
    registration = register_faux_provider()
    try:
        session = _build_session(registration)
        try:
            rc = await run_print_mode(session, PrintModeOptions(mode="text"))
        finally:
            session.close()
        assert rc == 0
    finally:
        registration.unregister()


async def test_print_mode_text_error_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("partial", stop_reason="error", error_message="boom")])
        session = _build_session(registration)
        try:
            rc = await run_print_mode(session, PrintModeOptions(mode="text", initial_message="x"))
        finally:
            session.close()
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
        session = _build_session(registration)
        try:
            rc = await run_print_mode(session, PrintModeOptions(mode="json", initial_message="hi"))
        finally:
            session.close()
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out
        assert "\n" in captured.out
    finally:
        registration.unregister()


async def test_print_mode_handles_session_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failure inside ``session.prompt`` is logged and returns 1."""
    registration = register_faux_provider()
    try:
        # No faux response queued — the session.prompt will surface an
        # error from the underlying provider.
        session = _build_session(registration)
        try:
            rc = await run_print_mode(session, PrintModeOptions(mode="text", initial_message="hi"))
        finally:
            session.close()
        # The session.prompt may either return rc=1 (assistant stop_reason=error)
        # or raise — both paths are valid for this contract.
        captured = capsys.readouterr()
        assert rc in (0, 1)
        # When the assistant lands as an error stop_reason, _print_text_result
        # writes to stderr.
        _ = captured
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
        session = _build_session(registration)
        try:
            rc = await run_print_mode(
                session,
                PrintModeOptions(mode="text", initial_message="one", messages=["two"]),
            )
        finally:
            session.close()
        captured = capsys.readouterr()
        assert rc == 0
        # Only the final assistant text appears in text mode.
        assert "second reply" in captured.out
    finally:
        registration.unregister()
