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
