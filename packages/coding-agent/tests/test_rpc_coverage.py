"""Additional tests for RPC mode + RPC client + JSONL to raise coverage to >= 90%.

Covers:
- _dispatch_command for all major command paths
- _to_dict helper
- run_rpc_mode with mocked stdin/stdout
- RpcClient lifecycle (start, stop, _dispatch_line, on_event, request/response)
- JSONL edge cases (async on_line, ConnectionResetError)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nu_coding_agent.modes.rpc.jsonl import attach_jsonl_line_reader, serialize_json_line
from nu_coding_agent.modes.rpc.rpc_client import (
    ModelInfo,
    RpcClient,
    RpcClientError,
    RpcClientOptions,
)
from nu_coding_agent.modes.rpc.rpc_mode import (
    _create_extension_ui_context,
    _dispatch_command,
    _error,
    _safe_prompt,
    _success,
    _to_dict,
)


# ===========================================================================
# _to_dict helper
# ===========================================================================


class TestToDict:
    def test_none_returns_none(self) -> None:
        assert _to_dict(None) is None

    def test_pydantic_model_dump(self) -> None:
        class FakeModel:
            def model_dump(self) -> dict[str, Any]:
                return {"key": "value"}

        assert _to_dict(FakeModel()) == {"key": "value"}

    def test_dataclass_asdict(self) -> None:
        @dataclass
        class Pt:
            x: int
            y: int

        result = _to_dict(Pt(1, 2))
        assert result == {"x": 1, "y": 2}

    def test_plain_object_passthrough(self) -> None:
        assert _to_dict("hello") == "hello"
        assert _to_dict(42) == 42


# ===========================================================================
# _dispatch_command — all major branches
# ===========================================================================


def _make_session() -> MagicMock:
    """Build a mock AgentSession that satisfies all _dispatch_command paths."""
    session = MagicMock()
    model_mock = MagicMock()
    model_mock.provider = "anthropic"
    model_mock.id = "claude-opus-4-6"
    model_mock.model_dump.return_value = {"provider": "anthropic", "id": "claude-opus-4-6"}
    session.model = model_mock

    # agent.state.messages
    session.agent.state.messages = []

    # get_stats
    session.get_stats.return_value = {"tokens": 100}

    # session_manager
    session.session_manager.get_session_file.return_value = "/tmp/session.jsonl"

    # model_registry
    model_obj = MagicMock()
    model_obj.provider = "anthropic"
    model_obj.id = "claude-opus-4-6"
    model_obj.model_dump.return_value = {"provider": "anthropic", "id": "claude-opus-4-6"}
    session.model_registry.get_available.return_value = [model_obj]

    # extension_runner
    session.extension_runner = None

    return session


def _make_runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.session = _make_session()
    runtime.new_session = AsyncMock(return_value=False)
    runtime.switch_session = AsyncMock(return_value=False)
    runtime.fork = AsyncMock(return_value=(False, "forked text"))
    runtime.dispose = AsyncMock()
    return runtime


class TestDispatchCommand:
    async def test_prompt_command(self) -> None:
        session = _make_session()
        session.prompt = AsyncMock()
        resp = await _dispatch_command(
            cmd_type="prompt",
            cmd_id="1",
            command={"type": "prompt", "message": "hello"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert resp["command"] == "prompt"

    async def test_steer_command(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="steer",
            cmd_id="2",
            command={"type": "steer", "message": "focus"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        session.agent.steer.assert_called_once()

    async def test_follow_up_command(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="follow_up",
            cmd_id="3",
            command={"type": "follow_up", "message": "more"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        session.agent.follow_up.assert_called_once()

    async def test_abort_command(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="abort",
            cmd_id="4",
            command={"type": "abort"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        session.agent.abort.assert_called_once()

    async def test_new_session_command(self) -> None:
        runtime = _make_runtime()
        rebind = AsyncMock()
        resp = await _dispatch_command(
            cmd_type="new_session",
            cmd_id="5",
            command={"type": "new_session"},
            session=runtime.session,
            runtime=runtime,
            rebind_session=rebind,
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert resp["data"]["cancelled"] is False
        rebind.assert_awaited_once()

    async def test_new_session_with_parent(self) -> None:
        runtime = _make_runtime()
        resp = await _dispatch_command(
            cmd_type="new_session",
            cmd_id="5b",
            command={"type": "new_session", "parentSession": "parent-123"},
            session=runtime.session,
            runtime=runtime,
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        runtime.new_session.assert_awaited_once_with(parent_session="parent-123")

    async def test_get_state_command(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="get_state",
            cmd_id="6",
            command={"type": "get_state"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert "sessionFile" in resp["data"]

    async def test_set_model_found(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="set_model",
            cmd_id="7",
            command={"type": "set_model", "provider": "anthropic", "modelId": "claude-opus-4-6"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        session.set_model.assert_called_once()

    async def test_set_model_not_found(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="set_model",
            cmd_id="7b",
            command={"type": "set_model", "provider": "bogus", "modelId": "bogus"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is False
        assert "not found" in resp["error"]

    async def test_cycle_model_command(self) -> None:
        session = _make_session()
        session.cycle_model = AsyncMock(return_value={"model": "new"})
        resp = await _dispatch_command(
            cmd_type="cycle_model",
            cmd_id="8",
            command={"type": "cycle_model"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_cycle_model_no_method(self) -> None:
        session = _make_session()
        # Remove cycle_model attribute
        del session.cycle_model
        resp = await _dispatch_command(
            cmd_type="cycle_model",
            cmd_id="8b",
            command={"type": "cycle_model"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        # _success with None data doesn't include "data" key
        assert "data" not in resp

    async def test_get_available_models(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="get_available_models",
            cmd_id="9",
            command={"type": "get_available_models"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert "models" in resp["data"]

    async def test_set_thinking_level(self) -> None:
        session = _make_session()
        session.set_thinking_level = MagicMock()
        resp = await _dispatch_command(
            cmd_type="set_thinking_level",
            cmd_id="10",
            command={"type": "set_thinking_level", "level": "high"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_cycle_thinking_level(self) -> None:
        session = _make_session()
        session.cycle_thinking_level = MagicMock(return_value="medium")
        resp = await _dispatch_command(
            cmd_type="cycle_thinking_level",
            cmd_id="11",
            command={"type": "cycle_thinking_level"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_cycle_thinking_level_no_method(self) -> None:
        session = _make_session()
        del session.cycle_thinking_level
        resp = await _dispatch_command(
            cmd_type="cycle_thinking_level",
            cmd_id="11b",
            command={"type": "cycle_thinking_level"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_set_steering_mode(self) -> None:
        session = _make_session()
        session.set_steering_mode = MagicMock()
        resp = await _dispatch_command(
            cmd_type="set_steering_mode",
            cmd_id="12",
            command={"type": "set_steering_mode", "mode": "none"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_set_follow_up_mode(self) -> None:
        session = _make_session()
        session.set_follow_up_mode = MagicMock()
        resp = await _dispatch_command(
            cmd_type="set_follow_up_mode",
            cmd_id="13",
            command={"type": "set_follow_up_mode", "mode": "all"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_compact_command(self) -> None:
        session = _make_session()
        session.compact = AsyncMock(return_value={"summary": "done"})
        resp = await _dispatch_command(
            cmd_type="compact",
            cmd_id="14",
            command={"type": "compact", "customInstructions": "be brief"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_set_auto_compaction(self) -> None:
        session = _make_session()
        session.set_auto_compaction_enabled = MagicMock()
        resp = await _dispatch_command(
            cmd_type="set_auto_compaction",
            cmd_id="15",
            command={"type": "set_auto_compaction", "enabled": False},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_set_auto_retry(self) -> None:
        session = _make_session()
        session.set_auto_retry_enabled = MagicMock()
        resp = await _dispatch_command(
            cmd_type="set_auto_retry",
            cmd_id="16",
            command={"type": "set_auto_retry", "enabled": True},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_abort_retry(self) -> None:
        session = _make_session()
        session.abort_retry = MagicMock()
        resp = await _dispatch_command(
            cmd_type="abort_retry",
            cmd_id="17",
            command={"type": "abort_retry"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_bash_command(self) -> None:
        session = _make_session()
        session.execute_bash = AsyncMock(return_value={"output": "hi"})
        resp = await _dispatch_command(
            cmd_type="bash",
            cmd_id="18",
            command={"type": "bash", "command": "echo hi"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_bash_not_available(self) -> None:
        session = _make_session()
        del session.execute_bash
        resp = await _dispatch_command(
            cmd_type="bash",
            cmd_id="18b",
            command={"type": "bash", "command": "echo hi"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is False

    async def test_abort_bash(self) -> None:
        session = _make_session()
        session.abort_bash = MagicMock()
        resp = await _dispatch_command(
            cmd_type="abort_bash",
            cmd_id="19",
            command={"type": "abort_bash"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_get_session_stats(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="get_session_stats",
            cmd_id="20",
            command={"type": "get_session_stats"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_export_html(self) -> None:
        session = _make_session()
        session.export_to_html = AsyncMock(return_value="/tmp/out.html")
        resp = await _dispatch_command(
            cmd_type="export_html",
            cmd_id="21",
            command={"type": "export_html", "outputPath": "/tmp/out.html"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_export_html_not_available(self) -> None:
        session = _make_session()
        del session.export_to_html
        resp = await _dispatch_command(
            cmd_type="export_html",
            cmd_id="21b",
            command={"type": "export_html"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is False

    async def test_switch_session(self) -> None:
        runtime = _make_runtime()
        rebind = AsyncMock()
        resp = await _dispatch_command(
            cmd_type="switch_session",
            cmd_id="22",
            command={"type": "switch_session", "sessionPath": "/tmp/s.jsonl"},
            session=runtime.session,
            runtime=runtime,
            rebind_session=rebind,
            output=lambda x: None,
        )
        assert resp["success"] is True
        rebind.assert_awaited_once()

    async def test_fork(self) -> None:
        runtime = _make_runtime()
        rebind = AsyncMock()
        resp = await _dispatch_command(
            cmd_type="fork",
            cmd_id="23",
            command={"type": "fork", "entryId": "e1"},
            session=runtime.session,
            runtime=runtime,
            rebind_session=rebind,
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert resp["data"]["text"] == "forked text"

    async def test_get_fork_messages(self) -> None:
        session = _make_session()
        session.get_user_messages_for_forking = MagicMock(return_value=[{"id": "m1"}])
        resp = await _dispatch_command(
            cmd_type="get_fork_messages",
            cmd_id="24",
            command={"type": "get_fork_messages"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert resp["data"]["messages"] == [{"id": "m1"}]

    async def test_get_fork_messages_no_method(self) -> None:
        session = _make_session()
        del session.get_user_messages_for_forking
        resp = await _dispatch_command(
            cmd_type="get_fork_messages",
            cmd_id="24b",
            command={"type": "get_fork_messages"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert resp["data"]["messages"] == []

    async def test_get_last_assistant_text_with_method(self) -> None:
        session = _make_session()
        session.get_last_assistant_text = MagicMock(return_value="hi there")
        resp = await _dispatch_command(
            cmd_type="get_last_assistant_text",
            cmd_id="25",
            command={"type": "get_last_assistant_text"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert resp["data"]["text"] == "hi there"

    async def test_get_last_assistant_text_fallback(self) -> None:
        session = _make_session()
        del session.get_last_assistant_text
        # Set up messages with AssistantMessage-like structure
        from nu_ai import AssistantMessage, TextContent
        from nu_ai.types import Cost, Usage

        _zero_cost = Cost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0)
        _zero_usage = Usage(input=0, output=0, cacheRead=0, cacheWrite=0, totalTokens=0, cost=_zero_cost)
        session.agent.state.messages = [
            AssistantMessage(
                content=[TextContent(type="text", text="last text")],
                api="messages",
                provider="anthropic",
                model="test",
                usage=_zero_usage,
                stopReason="stop",
                timestamp=1000,
            )
        ]
        resp = await _dispatch_command(
            cmd_type="get_last_assistant_text",
            cmd_id="25b",
            command={"type": "get_last_assistant_text"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert resp["data"]["text"] == "last text"

    async def test_get_last_assistant_text_no_messages(self) -> None:
        session = _make_session()
        del session.get_last_assistant_text
        session.agent.state.messages = []
        resp = await _dispatch_command(
            cmd_type="get_last_assistant_text",
            cmd_id="25c",
            command={"type": "get_last_assistant_text"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert resp["data"]["text"] is None

    async def test_set_session_name(self) -> None:
        session = _make_session()
        session.set_session_name = MagicMock()
        resp = await _dispatch_command(
            cmd_type="set_session_name",
            cmd_id="26",
            command={"type": "set_session_name", "name": "My Chat"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True

    async def test_set_session_name_empty(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="set_session_name",
            cmd_id="26b",
            command={"type": "set_session_name", "name": "  "},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is False

    async def test_get_messages(self) -> None:
        session = _make_session()
        from nu_ai import TextContent, UserMessage

        session.agent.state.messages = [
            UserMessage(content=[TextContent(type="text", text="hi")], timestamp=1000)
        ]
        resp = await _dispatch_command(
            cmd_type="get_messages",
            cmd_id="27",
            command={"type": "get_messages"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert len(resp["data"]["messages"]) == 1

    async def test_get_commands(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="get_commands",
            cmd_id="28",
            command={"type": "get_commands"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert "commands" in resp["data"]

    async def test_get_commands_with_extension_runner(self) -> None:
        session = _make_session()
        cmd = MagicMock()
        cmd.invocation_name = "test-cmd"
        cmd.description = "A test"
        cmd.source_info = {}
        runner = MagicMock()
        runner.get_registered_commands.return_value = [cmd]
        session.extension_runner = runner
        resp = await _dispatch_command(
            cmd_type="get_commands",
            cmd_id="28b",
            command={"type": "get_commands"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is True
        assert any(c["name"] == "test-cmd" for c in resp["data"]["commands"])

    async def test_unknown_command(self) -> None:
        session = _make_session()
        resp = await _dispatch_command(
            cmd_type="bogus_command",
            cmd_id="99",
            command={"type": "bogus_command"},
            session=session,
            runtime=MagicMock(),
            rebind_session=AsyncMock(),
            output=lambda x: None,
        )
        assert resp["success"] is False
        assert "Unknown command" in resp["error"]


# ===========================================================================
# _safe_prompt
# ===========================================================================


class TestSafePrompt:
    async def test_safe_prompt_exception_outputs_error(self) -> None:
        session = MagicMock()
        session.prompt = AsyncMock(side_effect=RuntimeError("oops"))
        outputs: list[dict[str, Any]] = []
        await _safe_prompt(session, "hi", {"type": "prompt"}, "req-1", outputs.append)
        assert len(outputs) == 1
        assert outputs[0]["success"] is False

    async def test_safe_prompt_with_images(self) -> None:
        session = MagicMock()
        session.prompt = AsyncMock()
        await _safe_prompt(session, "hi", {"type": "prompt", "images": [{"url": "http://x"}]}, "req-1", lambda x: None)
        session.prompt.assert_awaited_once_with("hi", images=[{"url": "http://x"}])


# ===========================================================================
# run_rpc_mode (integration-ish, mocked I/O)
# ===========================================================================


class TestRunRpcMode:
    async def test_run_rpc_mode_processes_command_and_shuts_down(self) -> None:
        from nu_coding_agent.modes.rpc.rpc_mode import run_rpc_mode

        runtime = _make_runtime()
        runtime.session.subscribe = MagicMock(return_value=lambda: None)
        del runtime.session.bind_extensions  # no bind_extensions attribute

        # Use a simple "abort" command that produces a simple JSON response
        command = json.dumps({"id": "1", "type": "abort"}) + "\n"

        captured_output: list[str] = []
        fake_stdout = MagicMock()
        fake_stdout.write = lambda s: captured_output.append(s)
        fake_stdout.flush = MagicMock()

        with (
            patch("nu_coding_agent.modes.rpc.rpc_mode.attach_jsonl_line_reader") as mock_attach,
            patch("nu_coding_agent.modes.rpc.rpc_mode.sys.stdout", fake_stdout),
        ):
            async def fake_attach_fn(on_line, **kwargs):
                for raw_line in command.strip().split("\n"):
                    await on_line(raw_line)
                return lambda: None

            mock_attach.side_effect = fake_attach_fn

            task = asyncio.create_task(run_rpc_mode(runtime))
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(captured_output) > 0
        parsed = json.loads(captured_output[0])
        assert parsed["success"] is True

    async def test_run_rpc_mode_handles_parse_error(self) -> None:
        from nu_coding_agent.modes.rpc.rpc_mode import run_rpc_mode

        runtime = _make_runtime()
        runtime.session.subscribe = MagicMock(return_value=lambda: None)
        del runtime.session.bind_extensions

        captured_output: list[str] = []
        fake_stdout = MagicMock()
        fake_stdout.write = lambda s: captured_output.append(s)
        fake_stdout.flush = MagicMock()

        with (
            patch("nu_coding_agent.modes.rpc.rpc_mode.attach_jsonl_line_reader") as mock_attach,
            patch("nu_coding_agent.modes.rpc.rpc_mode.sys.stdout", fake_stdout),
        ):
            async def fake_attach_fn(on_line, **kwargs):
                await on_line("not valid json")
                return lambda: None

            mock_attach.side_effect = fake_attach_fn

            task = asyncio.create_task(run_rpc_mode(runtime))
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(captured_output) > 0
        parsed = json.loads(captured_output[0])
        assert parsed["success"] is False
        assert "parse" in parsed["command"]

    async def test_run_rpc_mode_handles_extension_ui_response(self) -> None:
        from nu_coding_agent.modes.rpc.rpc_mode import run_rpc_mode

        runtime = _make_runtime()
        runtime.session.subscribe = MagicMock(return_value=lambda: None)
        del runtime.session.bind_extensions

        captured_output: list[str] = []
        fake_stdout = MagicMock()
        fake_stdout.write = lambda s: captured_output.append(s)
        fake_stdout.flush = MagicMock()

        with (
            patch("nu_coding_agent.modes.rpc.rpc_mode.attach_jsonl_line_reader") as mock_attach,
            patch("nu_coding_agent.modes.rpc.rpc_mode.sys.stdout", fake_stdout),
        ):
            async def fake_attach_fn(on_line, **kwargs):
                # Send an extension UI response - should be handled without output
                await on_line(json.dumps({"type": "extension_ui_response", "id": "xxx"}))
                return lambda: None

            mock_attach.side_effect = fake_attach_fn

            task = asyncio.create_task(run_rpc_mode(runtime))
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Extension UI response should not produce output (no matching pending request)
        assert len(captured_output) == 0

    async def test_run_rpc_mode_exception_in_dispatch(self) -> None:
        from nu_coding_agent.modes.rpc.rpc_mode import run_rpc_mode

        runtime = _make_runtime()
        runtime.session.subscribe = MagicMock(return_value=lambda: None)
        del runtime.session.bind_extensions
        # Make agent.abort raise to trigger exception path in handle_command
        runtime.session.agent.abort.side_effect = RuntimeError("test error")

        captured_output: list[str] = []
        fake_stdout = MagicMock()
        fake_stdout.write = lambda s: captured_output.append(s)
        fake_stdout.flush = MagicMock()

        with (
            patch("nu_coding_agent.modes.rpc.rpc_mode.attach_jsonl_line_reader") as mock_attach,
            patch("nu_coding_agent.modes.rpc.rpc_mode.sys.stdout", fake_stdout),
        ):
            async def fake_attach_fn(on_line, **kwargs):
                await on_line(json.dumps({"id": "1", "type": "abort"}))
                return lambda: None

            mock_attach.side_effect = fake_attach_fn

            task = asyncio.create_task(run_rpc_mode(runtime))
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(captured_output) > 0
        parsed = json.loads(captured_output[0])
        assert parsed["success"] is False
        assert "test error" in parsed["error"]


# ===========================================================================
# RpcClient tests
# ===========================================================================


class TestRpcClientOptions:
    def test_defaults(self) -> None:
        opts = RpcClientOptions()
        assert opts.cli_path is None
        assert opts.cwd is None

    def test_model_info(self) -> None:
        mi = ModelInfo(id="gpt-4", provider="openai", api="chat")
        assert mi.id == "gpt-4"


class TestRpcClientLifecycle:
    async def test_start_raises_when_no_cli_found(self) -> None:
        client = RpcClient(RpcClientOptions(cli_path=None))
        with patch("shutil.which", return_value=None):
            with pytest.raises(RpcClientError, match="Cannot find"):
                await client.start()

    async def test_stop_without_start_is_safe(self) -> None:
        client = RpcClient()
        await client.stop()  # should not raise

    async def test_send_raises_when_not_started(self) -> None:
        client = RpcClient()
        with pytest.raises(RpcClientError, match="not running"):
            await client._send({"type": "test"})

    async def test_context_manager(self) -> None:
        """Test __aenter__ and __aexit__."""
        client = RpcClient(RpcClientOptions(cli_path="/bin/echo"))
        # Mock start/stop to avoid actual subprocess
        client.start = AsyncMock()
        client.stop = AsyncMock()
        async with client as c:
            assert c is client
        client.start.assert_awaited_once()
        client.stop.assert_awaited_once()


class TestRpcClientDispatchLine:
    def test_dispatch_response_resolves_future(self) -> None:
        client = RpcClient()
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        client._pending["42"] = future
        client._dispatch_line('{"type":"response","id":"42","success":true,"data":{}}')
        assert future.done()
        assert future.result()["success"] is True
        loop.close()

    def test_dispatch_invalid_json_does_not_crash(self) -> None:
        client = RpcClient()
        client._dispatch_line("not json at all")  # should not raise

    def test_dispatch_non_dict_is_ignored(self) -> None:
        client = RpcClient()
        client._dispatch_line('"just a string"')  # should not raise

    def test_dispatch_event_broadcasts_to_listeners(self) -> None:
        client = RpcClient()
        events: list[dict[str, Any]] = []
        client.on_event(events.append)
        client._dispatch_line('{"type":"agent_turn_end"}')
        assert len(events) == 1
        assert events[0]["type"] == "agent_turn_end"

    def test_dispatch_event_listener_error_swallowed(self) -> None:
        client = RpcClient()

        def bad_listener(event: dict[str, Any]) -> None:
            raise RuntimeError("listener boom")

        client.on_event(bad_listener)
        client._dispatch_line('{"type":"event"}')  # should not raise


class TestRpcClientOnEvent:
    def test_subscribe_and_unsubscribe(self) -> None:
        client = RpcClient()
        events: list[dict[str, Any]] = []
        unsub = client.on_event(events.append)
        client._dispatch_line('{"type":"e1"}')
        assert len(events) == 1
        unsub()
        client._dispatch_line('{"type":"e2"}')
        assert len(events) == 1  # no new events


class TestRpcClientAllocId:
    def test_alloc_id_increments(self) -> None:
        client = RpcClient()
        assert client._alloc_id() == "1"
        assert client._alloc_id() == "2"


class TestRpcClientRequest:
    async def test_request_resolves_on_success(self) -> None:
        client = RpcClient()
        # Mock _send to not actually write, and simulate response
        client._process = MagicMock()
        client._process.stdin = MagicMock()
        client._process.stdin.write = MagicMock()
        client._process.stdin.drain = AsyncMock()

        async def do_request():
            return await client._request("get_state")

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.05)

        # Resolve the pending future
        req_id = "1"
        future = client._pending.get(req_id)
        assert future is not None
        future.set_result({"type": "response", "id": req_id, "success": True, "data": {"x": 1}})

        result = await task
        assert result["success"] is True

    async def test_request_raises_on_error_response(self) -> None:
        client = RpcClient()
        client._process = MagicMock()
        client._process.stdin = MagicMock()
        client._process.stdin.write = MagicMock()
        client._process.stdin.drain = AsyncMock()

        async def do_request():
            return await client._request("bad_cmd")

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.05)

        req_id = "1"
        future = client._pending.get(req_id)
        future.set_result({"type": "response", "id": req_id, "success": False, "error": "nope"})

        with pytest.raises(RpcClientError, match="nope"):
            await task


class TestRpcClientAPIMethods:
    """Test that each API method calls _request with the right type."""

    async def _setup_client(self) -> RpcClient:
        client = RpcClient()
        # Make _request return a canned response
        client._request = AsyncMock(return_value={"success": True, "data": {"models": [], "messages": [], "commands": [], "text": "hi", "path": "/tmp/x.html"}})
        return client

    async def test_prompt(self) -> None:
        c = await self._setup_client()
        await c.prompt("hi")
        c._request.assert_awaited_with("prompt", message="hi")

    async def test_prompt_with_images(self) -> None:
        c = await self._setup_client()
        await c.prompt("hi", images=[{"url": "x"}])
        c._request.assert_awaited_with("prompt", message="hi", images=[{"url": "x"}])

    async def test_steer(self) -> None:
        c = await self._setup_client()
        await c.steer("focus")
        c._request.assert_awaited_with("steer", message="focus")

    async def test_follow_up(self) -> None:
        c = await self._setup_client()
        await c.follow_up("more")
        c._request.assert_awaited_with("follow_up", message="more")

    async def test_abort(self) -> None:
        c = await self._setup_client()
        await c.abort()
        c._request.assert_awaited_with("abort")

    async def test_new_session(self) -> None:
        c = await self._setup_client()
        await c.new_session(parent_session="p")
        c._request.assert_awaited_with("new_session", parentSession="p")

    async def test_switch_session(self) -> None:
        c = await self._setup_client()
        await c.switch_session("/tmp/s.jsonl")
        c._request.assert_awaited_with("switch_session", sessionPath="/tmp/s.jsonl")

    async def test_fork(self) -> None:
        c = await self._setup_client()
        await c.fork("e1", "text")
        c._request.assert_awaited_with("fork", entryId="e1", message="text")

    async def test_get_state(self) -> None:
        c = await self._setup_client()
        result = await c.get_state()
        assert isinstance(result, dict)

    async def test_set_model(self) -> None:
        c = await self._setup_client()
        await c.set_model("gpt-4", provider="openai")

    async def test_cycle_model(self) -> None:
        c = await self._setup_client()
        await c.cycle_model()

    async def test_get_available_models(self) -> None:
        c = await self._setup_client()
        result = await c.get_available_models()
        assert isinstance(result, list)

    async def test_set_thinking_level(self) -> None:
        c = await self._setup_client()
        await c.set_thinking_level("high")

    async def test_cycle_thinking_level(self) -> None:
        c = await self._setup_client()
        c._request = AsyncMock(return_value={"success": True, "data": {"level": "high"}})
        result = await c.cycle_thinking_level()
        assert result == "high"

    async def test_cycle_thinking_level_no_data(self) -> None:
        c = await self._setup_client()
        c._request = AsyncMock(return_value={"success": True, "data": None})
        result = await c.cycle_thinking_level()
        assert result is None

    async def test_compact(self) -> None:
        c = await self._setup_client()
        await c.compact(custom_instructions="be brief")

    async def test_set_auto_compaction(self) -> None:
        c = await self._setup_client()
        await c.set_auto_compaction(True)

    async def test_bash(self) -> None:
        c = await self._setup_client()
        await c.bash("echo hi")

    async def test_abort_bash(self) -> None:
        c = await self._setup_client()
        await c.abort_bash()

    async def test_get_session_stats(self) -> None:
        c = await self._setup_client()
        result = await c.get_session_stats()
        assert isinstance(result, dict)

    async def test_export_html(self) -> None:
        c = await self._setup_client()
        result = await c.export_html(output_path="/tmp/x.html")
        assert isinstance(result, str)

    async def test_get_messages(self) -> None:
        c = await self._setup_client()
        result = await c.get_messages()
        assert isinstance(result, list)

    async def test_get_fork_messages(self) -> None:
        c = await self._setup_client()
        result = await c.get_fork_messages()
        assert isinstance(result, list)

    async def test_get_last_assistant_text(self) -> None:
        c = await self._setup_client()
        result = await c.get_last_assistant_text()
        assert result == "hi"

    async def test_get_commands(self) -> None:
        c = await self._setup_client()
        result = await c.get_commands()
        assert isinstance(result, list)

    async def test_set_steering_mode(self) -> None:
        c = await self._setup_client()
        await c.set_steering_mode("all")

    async def test_set_follow_up_mode(self) -> None:
        c = await self._setup_client()
        await c.set_follow_up_mode("all")

    async def test_set_auto_retry(self) -> None:
        c = await self._setup_client()
        await c.set_auto_retry(True)

    async def test_abort_retry(self) -> None:
        c = await self._setup_client()
        await c.abort_retry()

    async def test_set_session_name(self) -> None:
        c = await self._setup_client()
        await c.set_session_name("My Chat")


class TestRpcClientReadLoop:
    async def test_read_loop_processes_lines(self) -> None:
        client = RpcClient()
        events: list[dict[str, Any]] = []
        client.on_event(events.append)

        # Create a fake process with stdout
        proc = MagicMock()
        stdout = asyncio.StreamReader()
        proc.stdout = stdout
        client._process = proc

        # Feed data
        line = json.dumps({"type": "test_event"}) + "\n"
        stdout.feed_data(line.encode())
        stdout.feed_eof()

        # Run read loop
        await client._read_loop()
        assert len(events) == 1
        assert events[0]["type"] == "test_event"


class TestRpcClientStop:
    async def test_stop_terminates_process_and_rejects_pending(self) -> None:
        client = RpcClient()

        # Set up a fake process
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.close = MagicMock()
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        client._process = proc

        # Create pending future
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        client._pending["1"] = future

        # Create a fake reader task
        client._reader_task = asyncio.create_task(asyncio.sleep(10))

        await client.stop()

        assert client._process is None
        assert client._pending == {}
        assert future.done()
        with pytest.raises(RpcClientError, match="stopped"):
            future.result()


class TestRpcClientWaitForIdle:
    async def test_wait_for_idle_returns_when_not_streaming(self) -> None:
        client = RpcClient()
        client.get_state = AsyncMock(return_value={"isStreaming": False})
        await client.wait_for_idle(max_wait=1.0)

    async def test_wait_for_idle_times_out(self) -> None:
        client = RpcClient()
        client.get_state = AsyncMock(return_value={"isStreaming": True})
        with pytest.raises(TimeoutError):
            await client.wait_for_idle(max_wait=0.2)


class TestRpcClientPromptAndWait:
    async def test_prompt_and_wait_collects_events(self) -> None:
        client = RpcClient()
        client._request = AsyncMock(return_value={"success": True, "data": {}})

        # Simulate turn_end event
        async def fake_prompt(text, *, images=None):
            # Broadcast a turn_end event
            for cb in client._event_listeners:
                cb({"type": "agent_turn_end"})

        client.prompt = fake_prompt
        events = await client.prompt_and_wait("hello", max_wait=2.0)
        assert any(e["type"] == "agent_turn_end" for e in events)

    async def test_prompt_and_wait_timeout_checks_state(self) -> None:
        client = RpcClient()

        async def fake_prompt(text, *, images=None):
            pass  # no turn_end event

        client.prompt = fake_prompt
        client.get_state = AsyncMock(return_value={"isStreaming": False})
        events = await client.prompt_and_wait("hello", max_wait=0.2)
        assert isinstance(events, list)

    async def test_prompt_and_wait_timeout_still_streaming(self) -> None:
        client = RpcClient()

        async def fake_prompt(text, *, images=None):
            pass

        client.prompt = fake_prompt
        client.get_state = AsyncMock(return_value={"isStreaming": True})
        with pytest.raises(TimeoutError, match="did not finish"):
            await client.prompt_and_wait("hello", max_wait=0.2)


# ===========================================================================
# JSONL edge cases
# ===========================================================================


class TestJsonlEdgeCases:
    async def test_async_on_line_callback(self) -> None:
        """on_line returns a coroutine — should be awaited."""
        reader = asyncio.StreamReader()
        lines: list[str] = []

        async def async_handler(line: str) -> None:
            lines.append(line)

        detach = await attach_jsonl_line_reader(async_handler, stream=reader)
        reader.feed_data(b'{"ok":true}\n')
        reader.feed_eof()
        await asyncio.sleep(0.1)
        detach()
        assert len(lines) == 1

    async def test_eof_with_remaining_buffer_async_callback(self) -> None:
        """EOF flushes buffer, and the on_line callback is async."""
        reader = asyncio.StreamReader()
        lines: list[str] = []

        async def async_handler(line: str) -> None:
            lines.append(line)

        detach = await attach_jsonl_line_reader(async_handler, stream=reader)
        reader.feed_data(b'{"final":true}')  # no trailing newline
        reader.feed_eof()
        await asyncio.sleep(0.1)
        detach()
        assert len(lines) == 1

    async def test_crlf_at_eof(self) -> None:
        reader = asyncio.StreamReader()
        lines: list[str] = []
        detach = await attach_jsonl_line_reader(lines.append, stream=reader)
        reader.feed_data(b'{"a":1}\r')  # trailing \r but no \n at EOF
        reader.feed_eof()
        await asyncio.sleep(0.1)
        detach()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"a": 1}

    async def test_connection_reset_handled(self) -> None:
        """ConnectionResetError during read should not crash."""
        reader = asyncio.StreamReader()
        lines: list[str] = []
        detach = await attach_jsonl_line_reader(lines.append, stream=reader)

        # Simulate ConnectionResetError by setting exception on reader
        reader.set_exception(ConnectionResetError("reset"))
        await asyncio.sleep(0.1)
        detach()
        # Should not have crashed

    async def test_chunked_data(self) -> None:
        """Data arriving in small chunks should still be reassembled."""
        reader = asyncio.StreamReader()
        lines: list[str] = []
        detach = await attach_jsonl_line_reader(lines.append, stream=reader)
        reader.feed_data(b'{"typ')
        reader.feed_data(b'e":"t')
        reader.feed_data(b'est"}\n')
        reader.feed_eof()
        await asyncio.sleep(0.1)
        detach()
        assert len(lines) == 1
        assert json.loads(lines[0])["type"] == "test"


# ===========================================================================
# Extension UI context — additional coverage
# ===========================================================================


class TestExtensionUIContextExtra:
    async def test_input_text_cancelled(self) -> None:
        outputs: list[dict[str, Any]] = []
        pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        ctx = _create_extension_ui_context(outputs.append, pending)

        task = asyncio.create_task(ctx["input"]("Enter name", "placeholder"))
        await asyncio.sleep(0.05)
        req_id = outputs[0]["id"]
        pending[req_id].set_result({"cancelled": True})
        result = await task
        assert result is None

    async def test_input_text_with_value(self) -> None:
        outputs: list[dict[str, Any]] = []
        pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        ctx = _create_extension_ui_context(outputs.append, pending)

        task = asyncio.create_task(ctx["input"]("Enter name"))
        await asyncio.sleep(0.05)
        req_id = outputs[0]["id"]
        pending[req_id].set_result({"value": "Claude"})
        result = await task
        assert result == "Claude"

    async def test_confirm_with_confirmed(self) -> None:
        outputs: list[dict[str, Any]] = []
        pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        ctx = _create_extension_ui_context(outputs.append, pending)

        task = asyncio.create_task(ctx["confirm"]("Sure?", "Really?"))
        await asyncio.sleep(0.05)
        req_id = outputs[0]["id"]
        pending[req_id].set_result({"confirmed": True})
        result = await task
        assert result is True

    async def test_select_cancelled(self) -> None:
        outputs: list[dict[str, Any]] = []
        pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        ctx = _create_extension_ui_context(outputs.append, pending)

        task = asyncio.create_task(ctx["select"]("Pick", ["a", "b"]))
        await asyncio.sleep(0.05)
        req_id = outputs[0]["id"]
        pending[req_id].set_result({"cancelled": True})
        result = await task
        assert result is None

    def test_set_editor_text(self) -> None:
        outputs: list[dict[str, Any]] = []
        ctx = _create_extension_ui_context(outputs.append, {})
        ctx["set_editor_text"]("Hello world")
        assert outputs[0]["method"] == "set_editor_text"
        assert outputs[0]["text"] == "Hello world"

    def test_noop_lambdas(self) -> None:
        ctx = _create_extension_ui_context(lambda _: None, {})
        # These should not raise
        ctx["set_working_message"]("working...")
        ctx["set_hidden_thinking_label"]("thinking")
        ctx["set_widget"]("k", "content")
        ctx["set_footer"](lambda: None)
        ctx["set_header"](lambda: None)
        ctx["set_tools_expanded"](True)
        unsub = ctx["on_terminal_input"]()
        unsub()  # the returned lambda should be callable
        ctx["paste_to_editor"]("pasted")
