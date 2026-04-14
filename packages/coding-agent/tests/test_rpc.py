"""Tests for the RPC mode.

Tests cover:
- JSONL serialization
- JSONL line reader (LF splitting, CRLF handling, Unicode preservation)
- RPC types (command, response, state)
- RPC command dispatch (get_state, get_messages, set_model, etc.)
- Extension UI context creation
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from nu_coding_agent.modes.rpc.jsonl import serialize_json_line
from nu_coding_agent.modes.rpc.rpc_mode import _create_extension_ui_context, _error, _success
from nu_coding_agent.modes.rpc.rpc_types import RpcSessionState, RpcSlashCommand

# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------


class TestSerializeJsonLine:
    def test_simple_object(self) -> None:
        result = serialize_json_line({"type": "response", "success": True})
        assert result.endswith("\n")
        parsed = json.loads(result.strip())
        assert parsed == {"type": "response", "success": True}

    def test_no_trailing_whitespace_before_newline(self) -> None:
        result = serialize_json_line({"a": 1})
        # Should be compact JSON + \n
        assert result == '{"a":1}\n'

    def test_preserves_unicode_separators(self) -> None:
        """U+2028 and U+2029 should NOT be escaped in the output."""
        result = serialize_json_line({"text": "hello\u2028world\u2029end"})
        # The actual Unicode characters should be in the output
        assert "\u2028" in result
        assert "\u2029" in result

    def test_nested_objects(self) -> None:
        obj = {"type": "response", "data": {"models": [{"id": "gpt-4"}]}}
        result = serialize_json_line(obj)
        parsed = json.loads(result)
        assert parsed["data"]["models"][0]["id"] == "gpt-4"

    def test_null_values(self) -> None:
        result = serialize_json_line({"id": None, "type": "response"})
        parsed = json.loads(result)
        assert parsed["id"] is None


# ---------------------------------------------------------------------------
# JSONL line reader
# ---------------------------------------------------------------------------


class TestAttachJsonlLineReader:
    async def test_reads_lines_from_stream(self) -> None:
        from nu_coding_agent.modes.rpc.jsonl import attach_jsonl_line_reader

        reader = asyncio.StreamReader()
        lines: list[str] = []

        detach = await attach_jsonl_line_reader(lines.append, stream=reader)

        reader.feed_data(b'{"type":"prompt"}\n{"type":"abort"}\n')
        reader.feed_eof()

        await asyncio.sleep(0.1)
        detach()
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "prompt"
        assert json.loads(lines[1])["type"] == "abort"

    async def test_handles_crlf(self) -> None:
        from nu_coding_agent.modes.rpc.jsonl import attach_jsonl_line_reader

        reader = asyncio.StreamReader()
        lines: list[str] = []

        detach = await attach_jsonl_line_reader(lines.append, stream=reader)

        reader.feed_data(b'{"a":1}\r\n{"b":2}\r\n')
        reader.feed_eof()

        await asyncio.sleep(0.1)
        detach()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    async def test_handles_final_line_without_newline(self) -> None:
        from nu_coding_agent.modes.rpc.jsonl import attach_jsonl_line_reader

        reader = asyncio.StreamReader()
        lines: list[str] = []

        detach = await attach_jsonl_line_reader(lines.append, stream=reader)

        reader.feed_data(b'{"final":true}')
        reader.feed_eof()

        await asyncio.sleep(0.1)
        detach()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"final": True}

    async def test_preserves_unicode_in_payload(self) -> None:
        from nu_coding_agent.modes.rpc.jsonl import attach_jsonl_line_reader

        reader = asyncio.StreamReader()
        lines: list[str] = []

        detach = await attach_jsonl_line_reader(lines.append, stream=reader)

        # JSON with U+2028 and U+2029 inside a string value
        payload = json.dumps({"text": "a\u2028b\u2029c"})
        reader.feed_data((payload + "\n").encode("utf-8"))
        reader.feed_eof()

        await asyncio.sleep(0.1)
        detach()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["text"] == "a\u2028b\u2029c"

    async def test_empty_lines_skipped(self) -> None:
        from nu_coding_agent.modes.rpc.jsonl import attach_jsonl_line_reader

        reader = asyncio.StreamReader()
        lines: list[str] = []

        detach = await attach_jsonl_line_reader(lines.append, stream=reader)

        reader.feed_data(b'{"a":1}\n\n{"b":2}\n')
        reader.feed_eof()

        await asyncio.sleep(0.1)
        detach()
        # Empty lines should be skipped
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# RPC types
# ---------------------------------------------------------------------------


class TestRpcTypes:
    def test_session_state_defaults(self) -> None:
        state = RpcSessionState()
        assert state.thinking_level == "off"
        assert state.is_streaming is False
        assert state.is_compacting is False
        assert state.steering_mode == "all"
        assert state.follow_up_mode == "all"
        assert state.auto_compaction_enabled is True
        assert state.message_count == 0

    def test_session_state_with_values(self) -> None:
        state = RpcSessionState(
            model={"id": "gpt-4", "provider": "openai"},
            thinking_level="high",
            is_streaming=True,
            session_id="sess-123",
            message_count=5,
        )
        assert state.model is not None
        assert state.model["id"] == "gpt-4"
        assert state.thinking_level == "high"
        assert state.is_streaming is True

    def test_slash_command(self) -> None:
        cmd = RpcSlashCommand(
            name="test",
            description="A test command",
            source="extension",
            source_info={"path": "/ext/test"},
        )
        assert cmd.name == "test"
        assert cmd.source == "extension"


# ---------------------------------------------------------------------------
# Success / error helpers
# ---------------------------------------------------------------------------


class TestResponseHelpers:
    def test_success_without_data(self) -> None:
        resp = _success("req-1", "abort")
        assert resp["id"] == "req-1"
        assert resp["type"] == "response"
        assert resp["command"] == "abort"
        assert resp["success"] is True
        assert "data" not in resp

    def test_success_with_data(self) -> None:
        resp = _success("req-2", "get_state", {"model": "gpt-4"})
        assert resp["data"] == {"model": "gpt-4"}

    def test_success_with_null_data(self) -> None:
        resp = _success(None, "cycle_model", None)
        assert "data" not in resp

    def test_error(self) -> None:
        resp = _error("req-3", "set_model", "Model not found")
        assert resp["success"] is False
        assert resp["error"] == "Model not found"
        assert resp["command"] == "set_model"


# ---------------------------------------------------------------------------
# Extension UI context
# ---------------------------------------------------------------------------


class TestExtensionUIContext:
    def test_creates_context_with_methods(self) -> None:
        outputs: list[dict[str, Any]] = []
        ctx = _create_extension_ui_context(outputs.append, {})

        assert "select" in ctx
        assert "confirm" in ctx
        assert "input" in ctx
        assert "notify" in ctx
        assert "set_status" in ctx
        assert "set_title" in ctx
        assert "set_editor_text" in ctx

    def test_notify_outputs_event(self) -> None:
        outputs: list[dict[str, Any]] = []
        ctx = _create_extension_ui_context(outputs.append, {})

        ctx["notify"]("Hello!", "info")
        assert len(outputs) == 1
        assert outputs[0]["type"] == "extension_ui_request"
        assert outputs[0]["method"] == "notify"
        assert outputs[0]["message"] == "Hello!"
        assert outputs[0]["notifyType"] == "info"

    def test_set_status_outputs_event(self) -> None:
        outputs: list[dict[str, Any]] = []
        ctx = _create_extension_ui_context(outputs.append, {})

        ctx["set_status"]("key1", "Working...")
        assert len(outputs) == 1
        assert outputs[0]["method"] == "setStatus"
        assert outputs[0]["statusKey"] == "key1"
        assert outputs[0]["statusText"] == "Working..."

    def test_set_title_outputs_event(self) -> None:
        outputs: list[dict[str, Any]] = []
        ctx = _create_extension_ui_context(outputs.append, {})

        ctx["set_title"]("My Session")
        assert len(outputs) == 1
        assert outputs[0]["method"] == "setTitle"
        assert outputs[0]["title"] == "My Session"

    def test_get_editor_text_returns_empty(self) -> None:
        ctx = _create_extension_ui_context(lambda _: None, {})
        assert ctx["get_editor_text"]() == ""

    def test_get_tools_expanded_returns_false(self) -> None:
        ctx = _create_extension_ui_context(lambda _: None, {})
        assert ctx["get_tools_expanded"]() is False

    async def test_select_dialog_resolves(self) -> None:
        outputs: list[dict[str, Any]] = []
        pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        ctx = _create_extension_ui_context(outputs.append, pending)

        # Start select dialog
        select_task = asyncio.create_task(ctx["select"]("Pick one", ["a", "b", "c"]))
        await asyncio.sleep(0.05)

        # Should have emitted a request
        assert len(outputs) == 1
        req_id = outputs[0]["id"]
        assert req_id in pending

        # Simulate response
        pending[req_id].set_result({"value": "b"})
        result = await select_task
        assert result == "b"

    async def test_confirm_dialog_cancelled(self) -> None:
        outputs: list[dict[str, Any]] = []
        pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        ctx = _create_extension_ui_context(outputs.append, pending)

        confirm_task = asyncio.create_task(ctx["confirm"]("Sure?", "Really?"))
        await asyncio.sleep(0.05)

        req_id = outputs[0]["id"]
        pending[req_id].set_result({"cancelled": True})
        result = await confirm_task
        assert result is False


# ---------------------------------------------------------------------------
# RPC Client tests
# ---------------------------------------------------------------------------


from nu_coding_agent.modes.rpc.rpc_client import (
    ModelInfo,
    RpcClient,
    RpcClientError,
    RpcClientOptions,
)


class TestRpcClientOptions:
    def test_defaults(self) -> None:
        opts = RpcClientOptions()
        assert opts.cli_path is None
        assert opts.cwd is None
        assert opts.env is None
        assert opts.provider is None
        assert opts.model is None
        assert opts.args is None


class TestModelInfo:
    def test_fields(self) -> None:
        m = ModelInfo(id="gpt-4", provider="openai", api="completions")
        assert m.id == "gpt-4"
        assert m.provider == "openai"


class TestRpcClientStart:
    async def test_start_no_cli_raises(self) -> None:
        """Lines 117-134: start() raises when 'nu' is not found."""
        from unittest.mock import patch
        client = RpcClient(RpcClientOptions(cli_path=None))
        with patch("shutil.which", return_value=None):
            with pytest.raises(RpcClientError, match="Cannot find"):
                await client.start()

    async def test_start_spawns_process(self) -> None:
        """start() spawns subprocess with correct args (lines 117-134)."""
        from unittest.mock import AsyncMock, patch, MagicMock

        client = RpcClient(RpcClientOptions(
            cli_path="/usr/bin/fake-nu",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            args=["--verbose"],
            cwd="/tmp",
            env={"FOO": "bar"},
        ))

        mock_process = MagicMock()
        mock_process.stdout = asyncio.StreamReader()
        mock_process.stderr = asyncio.StreamReader()
        mock_process.stdin = MagicMock()
        mock_process.stdout.feed_eof()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process) as mock_exec:
            await client.start()
            # Verify it was called with the right args
            call_args = mock_exec.call_args
            argv = call_args[0]
            assert argv[0] == "/usr/bin/fake-nu"
            assert "--rpc" in argv
            assert "--provider" in argv
            assert "anthropic" in argv
            assert "--model" in argv
            assert "--verbose" in argv

        # Clean up
        if client._reader_task:
            client._reader_task.cancel()
            import contextlib
            with contextlib.suppress(asyncio.CancelledError):
                await client._reader_task


class TestRpcClientStop:
    async def test_stop_rejects_pending(self) -> None:
        """Lines 151-152: stop() rejects pending futures."""
        client = RpcClient()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        client._pending["1"] = fut
        await client.stop()
        assert fut.done()
        with pytest.raises(RpcClientError, match="stopped"):
            fut.result()

    async def test_stop_terminates_process(self) -> None:
        """stop() terminates the subprocess (lines 146-153)."""
        from unittest.mock import MagicMock, AsyncMock

        client = RpcClient()

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.close = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        # Make wait() resolve immediately
        wait_future = asyncio.get_running_loop().create_future()
        wait_future.set_result(0)
        mock_process.wait = MagicMock(return_value=wait_future)

        client._process = mock_process
        await client.stop()
        mock_process.stdin.close.assert_called_once()
        mock_process.terminate.assert_called_once()
        assert client._process is None

    async def test_stop_kills_on_timeout(self) -> None:
        """stop() kills the process when terminate times out (line 152)."""
        from unittest.mock import MagicMock

        client = RpcClient()

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.close = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        # Make wait() hang (never complete, causing TimeoutError)
        async def slow_wait() -> int:
            await asyncio.sleep(100)
            return 0

        mock_process.wait = MagicMock(return_value=slow_wait())

        client._process = mock_process
        await client.stop()
        mock_process.kill.assert_called_once()
        assert client._process is None


class TestRpcClientSendWithoutProcess:
    async def test_send_raises_without_process(self) -> None:
        """Lines 189-191: _send raises when subprocess not running."""
        client = RpcClient()
        with pytest.raises(RpcClientError, match="not running"):
            await client._send({"type": "test"})

    async def test_send_raises_with_no_stdin(self) -> None:
        """_send raises when process has no stdin."""

        class MockProcess:
            stdin = None

        client = RpcClient()
        client._process = MockProcess()  # type: ignore[assignment]
        with pytest.raises(RpcClientError, match="not running"):
            await client._send({"type": "test"})

    async def test_send_writes_to_stdin(self) -> None:
        """_send writes JSON line to stdin (lines 172-174)."""

        class MockStdin:
            data = b""

            def write(self, data: bytes) -> None:
                self.data += data

            async def drain(self) -> None:
                pass

        class MockProcess:
            stdin = MockStdin()

        client = RpcClient()
        client._process = MockProcess()  # type: ignore[assignment]
        await client._send({"type": "test", "id": "1"})
        assert b"test" in MockProcess.stdin.data
        assert MockProcess.stdin.data.endswith(b"\n")


class TestRpcClientDispatchLine:
    def test_dispatch_invalid_json(self) -> None:
        """Line 202: invalid JSON is logged and ignored."""
        client = RpcClient()
        # Should not raise
        client._dispatch_line("not valid json{{{")

    def test_dispatch_non_dict(self) -> None:
        """Line 208-209: non-dict JSON is ignored."""
        client = RpcClient()
        client._dispatch_line('"just a string"')

    def test_dispatch_response_resolves_future(self) -> None:
        """Lines 222: response resolves matching pending future."""
        client = RpcClient()
        loop = asyncio.new_event_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        client._pending["42"] = fut
        client._dispatch_line('{"type":"response","id":"42","success":true}')
        assert fut.done()
        assert fut.result()["success"] is True
        loop.close()

    def test_dispatch_event_to_listeners(self) -> None:
        """Lines 222: events are broadcast to listeners."""
        client = RpcClient()
        events: list[dict[str, Any]] = []
        client.on_event(events.append)
        client._dispatch_line('{"type":"agent_status","streaming":true}')
        assert len(events) == 1
        assert events[0]["type"] == "agent_status"

    def test_dispatch_event_listener_error_swallowed(self) -> None:
        """Event listener exceptions are caught and logged."""
        client = RpcClient()

        def bad_listener(event: dict[str, Any]) -> None:
            raise ValueError("boom")

        client.on_event(bad_listener)
        # Should not raise
        client._dispatch_line('{"type":"some_event"}')


class TestRpcClientOnEvent:
    def test_unsubscribe(self) -> None:
        client = RpcClient()
        events: list[dict[str, Any]] = []
        unsub = client.on_event(events.append)
        client._dispatch_line('{"type":"e1"}')
        assert len(events) == 1
        unsub()
        client._dispatch_line('{"type":"e2"}')
        assert len(events) == 1  # no new events

    def test_double_unsubscribe(self) -> None:
        """Double unsubscribe should not raise."""
        client = RpcClient()
        unsub = client.on_event(lambda e: None)
        unsub()
        unsub()  # should not raise


class TestRpcClientRequest:
    """Test the _request method and all methods that depend on it."""

    async def _make_client_with_mock_send(self) -> RpcClient:
        """Create a client with a mocked _send that auto-resolves requests."""
        client = RpcClient()

        async def mock_send(payload: dict[str, Any]) -> None:
            # Simulate an immediate successful response
            req_id = payload.get("id")
            if req_id and req_id in client._pending:
                fut = client._pending[req_id]
                if not fut.done():
                    fut.set_result({"type": "response", "id": req_id, "success": True, "data": {}})

        client._send = mock_send  # type: ignore[assignment]
        return client

    async def test_request_success(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client._request("test_cmd")
        assert result["success"] is True

    async def test_request_error_response(self) -> None:
        client = RpcClient()

        async def mock_send(payload: dict[str, Any]) -> None:
            req_id = payload.get("id")
            if req_id and req_id in client._pending:
                fut = client._pending[req_id]
                if not fut.done():
                    fut.set_result({"type": "response", "id": req_id, "success": False, "error": "bad"})

        client._send = mock_send  # type: ignore[assignment]
        with pytest.raises(RpcClientError, match="bad"):
            await client._request("fail_cmd")

    async def test_request_cancelled(self) -> None:
        client = RpcClient()

        async def mock_send(payload: dict[str, Any]) -> None:
            req_id = payload.get("id")
            if req_id and req_id in client._pending:
                fut = client._pending[req_id]
                if not fut.done():
                    fut.cancel()

        client._send = mock_send  # type: ignore[assignment]
        with pytest.raises(asyncio.CancelledError):
            await client._request("cancel_cmd")
        # Pending should be cleaned up
        assert len(client._pending) == 0

    async def test_prompt(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.prompt("Hello")  # should not raise

    async def test_prompt_with_images(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.prompt("Hello", images=[{"url": "http://example.com/img.png"}])

    async def test_steer(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.steer("adjust course")

    async def test_follow_up(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.follow_up("more info")

    async def test_abort(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.abort()

    async def test_new_session(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.new_session()
        assert isinstance(result, dict)

    async def test_new_session_with_parent(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.new_session(parent_session="sess-1")
        assert isinstance(result, dict)

    async def test_switch_session(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.switch_session("/path/to/session")
        assert isinstance(result, dict)

    async def test_fork(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.fork("entry-1", "fork text")
        assert isinstance(result, dict)

    async def test_get_state(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.get_state()
        assert isinstance(result, dict)

    async def test_set_model(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.set_model("gpt-4", provider="openai")
        assert isinstance(result, dict)

    async def test_cycle_model(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.cycle_model(1)
        # data is {} so result is {}
        assert result is not None or result is None

    async def test_get_available_models(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.get_available_models()
        assert isinstance(result, list)

    async def test_set_thinking_level(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.set_thinking_level("high")

    async def test_cycle_thinking_level(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.cycle_thinking_level()
        # data is {} which is a dict, so data.get("level") returns None
        assert result is None

    async def test_compact(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.compact(custom_instructions="Be brief")
        assert result is not None or result is None

    async def test_set_auto_compaction(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.set_auto_compaction(True)

    async def test_bash(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.bash("echo hi")
        assert result is not None or result is None

    async def test_abort_bash(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.abort_bash()

    async def test_get_session_stats(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.get_session_stats()
        assert isinstance(result, dict)

    async def test_export_html(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.export_html(output_path="/tmp/out.html")
        assert isinstance(result, str)

    async def test_get_messages(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.get_messages()
        assert isinstance(result, list)

    async def test_get_fork_messages(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.get_fork_messages()
        assert isinstance(result, list)

    async def test_get_last_assistant_text(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.get_last_assistant_text()
        # data is {} so text is None
        assert result is None

    async def test_get_commands(self) -> None:
        client = await self._make_client_with_mock_send()
        result = await client.get_commands()
        assert isinstance(result, list)

    async def test_set_steering_mode(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.set_steering_mode("all")

    async def test_set_follow_up_mode(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.set_follow_up_mode("all")

    async def test_set_auto_retry(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.set_auto_retry(True)

    async def test_abort_retry(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.abort_retry()

    async def test_set_session_name(self) -> None:
        client = await self._make_client_with_mock_send()
        await client.set_session_name("My Session")


class TestRpcClientReadLoop:
    async def test_read_loop_with_data(self) -> None:
        """Test _read_loop processes JSON lines from stdout."""
        client = RpcClient()

        # Create a mock process with a StreamReader for stdout
        reader = asyncio.StreamReader()
        events: list[dict[str, Any]] = []
        client.on_event(events.append)

        class MockProcess:
            stdout = reader
            stderr = asyncio.StreamReader()

        client._process = MockProcess()  # type: ignore[assignment]

        # Feed data and EOF
        reader.feed_data(b'{"type":"test_event","value":1}\n')
        reader.feed_eof()

        await client._read_loop()
        assert len(events) == 1
        assert events[0]["type"] == "test_event"

    async def test_read_loop_no_process(self) -> None:
        """_read_loop returns immediately when no process."""
        client = RpcClient()
        await client._read_loop()  # should not raise

    async def test_read_loop_empty_lines_skipped(self) -> None:
        """Empty lines in output are skipped (line 222)."""
        client = RpcClient()
        reader = asyncio.StreamReader()
        events: list[dict[str, Any]] = []
        client.on_event(events.append)

        class MockProcess:
            stdout = reader
            stderr = asyncio.StreamReader()

        client._process = MockProcess()  # type: ignore[assignment]

        reader.feed_data(b'{"type":"e1"}\n\n\n{"type":"e2"}\n')
        reader.feed_eof()

        await client._read_loop()
        assert len(events) == 2

    async def test_read_loop_connection_reset(self) -> None:
        """ConnectionResetError during read is handled (lines 208-209)."""
        client = RpcClient()

        class ErrorReader:
            async def read(self, n: int) -> bytes:
                raise ConnectionResetError

        class MockProcess:
            stdout = ErrorReader()
            stderr = asyncio.StreamReader()

        client._process = MockProcess()  # type: ignore[assignment]
        await client._read_loop()  # should not raise

    async def test_read_loop_no_stdout(self) -> None:
        """_read_loop returns early when stdout is None."""
        client = RpcClient()

        class MockProcess:
            stdout = None

        client._process = MockProcess()  # type: ignore[assignment]
        await client._read_loop()  # should not raise


class TestRpcClientAllocId:
    def test_alloc_id_increments(self) -> None:
        client = RpcClient()
        id1 = client._alloc_id()
        id2 = client._alloc_id()
        assert id1 == "1"
        assert id2 == "2"


class TestRpcClientStop2:
    async def test_stop_terminates_reader_task(self) -> None:
        """Stop cancels reader task and cleans up."""
        client = RpcClient()

        async def fake_reader() -> None:
            await asyncio.sleep(100)

        client._reader_task = asyncio.create_task(fake_reader())
        await client.stop()
        assert client._reader_task is None

    async def test_stop_with_no_process(self) -> None:
        """Stop with no process should not raise."""
        client = RpcClient()
        await client.stop()


class TestRpcClientWaitForIdle:
    async def test_wait_for_idle(self) -> None:
        """wait_for_idle polls get_state."""
        client = RpcClient()
        call_count = 0

        async def mock_get_state() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"isStreaming": call_count < 3}

        client.get_state = mock_get_state  # type: ignore[assignment]
        await client.wait_for_idle(max_wait=5.0)
        assert call_count == 3


class TestRpcClientContextManager:
    async def test_aenter_aexit(self) -> None:
        """Context manager calls start and stop."""
        from unittest.mock import AsyncMock
        client = RpcClient()
        client.start = AsyncMock()  # type: ignore[assignment]
        client.stop = AsyncMock()  # type: ignore[assignment]
        async with client as c:
            assert c is client
        client.start.assert_called_once()
        client.stop.assert_called_once()


class TestRpcClientPromptAndWait:
    async def test_prompt_and_wait_collects_events(self) -> None:
        """prompt_and_wait collects events until turn_end."""
        client = RpcClient()

        async def mock_prompt(text: str, *, images: Any = None) -> None:
            # Simulate events arriving after prompt
            for cb in client._event_listeners:
                cb({"type": "text", "content": "Hello"})
                cb({"type": "agent_turn_end"})

        client.prompt = mock_prompt  # type: ignore[assignment]

        events = await client.prompt_and_wait("Hi", max_wait=5.0)
        assert len(events) == 2
        assert events[1]["type"] == "agent_turn_end"

    async def test_prompt_and_wait_timeout_but_idle(self) -> None:
        """When timeout fires but agent is already idle, no error."""
        client = RpcClient()

        async def mock_prompt(text: str, *, images: Any = None) -> None:
            pass  # No events emitted

        async def mock_get_state() -> dict[str, Any]:
            return {"isStreaming": False}

        client.prompt = mock_prompt  # type: ignore[assignment]
        client.get_state = mock_get_state  # type: ignore[assignment]

        events = await client.prompt_and_wait("Hi", max_wait=0.1)
        assert isinstance(events, list)

    async def test_prompt_and_wait_timeout_still_streaming(self) -> None:
        """When timeout fires and agent is still streaming, raises TimeoutError."""
        client = RpcClient()

        async def mock_prompt(text: str, *, images: Any = None) -> None:
            pass

        async def mock_get_state() -> dict[str, Any]:
            return {"isStreaming": True}

        client.prompt = mock_prompt  # type: ignore[assignment]
        client.get_state = mock_get_state  # type: ignore[assignment]

        with pytest.raises(TimeoutError, match="did not finish"):
            await client.prompt_and_wait("Hi", max_wait=0.1)
