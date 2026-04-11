"""Tests for nu_ai.types.

Ported from the implicit TS type contracts in packages/ai/src/types.ts. The
upstream repo has no dedicated types test file (TS catches these at compile
time), so we assert behaviour the rest of nu_ai relies on:

1. Discriminated-union parsing via the ``type``/``role`` tag.
2. Round-trip serialization in **camelCase** (session JSONL must stay
   byte-compatible with the upstream TS format).
3. Python attributes are snake_case while wire format stays camelCase.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from nu_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Content,
    Cost,
    ImageContent,
    Message,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pydantic import TypeAdapter, ValidationError

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


class TestTextContent:
    def test_round_trip(self) -> None:
        content = TextContent(text="hello")
        dumped = content.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {"type": "text", "text": "hello"}
        assert TextContent.model_validate(dumped) == content

    def test_text_signature_serialized_as_camel_case(self) -> None:
        content = TextContent(text="hi", text_signature="sig-1")
        dumped = content.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {"type": "text", "text": "hi", "textSignature": "sig-1"}

    def test_accepts_both_alias_and_field_name(self) -> None:
        # Python code can construct with snake_case; JSON input arrives as camelCase.
        from_json = TextContent.model_validate({"type": "text", "text": "x", "textSignature": "s"})
        from_py = TextContent(text="x", text_signature="s")
        assert from_json == from_py


class TestThinkingContent:
    def test_round_trip_plain(self) -> None:
        t = ThinkingContent(thinking="...")
        assert t.model_dump(by_alias=True, exclude_none=True) == {"type": "thinking", "thinking": "..."}

    def test_redacted_and_signature(self) -> None:
        t = ThinkingContent(thinking="", thinking_signature="opaque", redacted=True)
        assert t.model_dump(by_alias=True, exclude_none=True) == {
            "type": "thinking",
            "thinking": "",
            "thinkingSignature": "opaque",
            "redacted": True,
        }


class TestImageContent:
    def test_round_trip(self) -> None:
        img = ImageContent(data="aGVsbG8=", mime_type="image/png")
        dumped = img.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}
        assert ImageContent.model_validate(dumped) == img


class TestToolCall:
    def test_round_trip(self) -> None:
        tc = ToolCall(id="call_1", name="bash", arguments={"cmd": "ls"})
        dumped = tc.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {"type": "toolCall", "id": "call_1", "name": "bash", "arguments": {"cmd": "ls"}}

    def test_thought_signature_camel_case(self) -> None:
        tc = ToolCall(
            id="c",
            name="n",
            arguments={},
            thought_signature="google-opaque",
        )
        assert tc.model_dump(by_alias=True, exclude_none=True)["thoughtSignature"] == "google-opaque"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestUserMessage:
    def test_string_content(self) -> None:
        msg = UserMessage(content="hello", timestamp=1_700_000_000_000)
        dumped = msg.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {"role": "user", "content": "hello", "timestamp": 1_700_000_000_000}

    def test_list_content(self) -> None:
        msg = UserMessage(
            content=[TextContent(text="hi"), ImageContent(data="d", mime_type="image/png")],
            timestamp=1,
        )
        dumped = msg.model_dump(by_alias=True, exclude_none=True)
        assert dumped["content"] == [
            {"type": "text", "text": "hi"},
            {"type": "image", "data": "d", "mimeType": "image/png"},
        ]


class TestAssistantMessage:
    def _fresh(self) -> AssistantMessage:
        return AssistantMessage(
            content=[TextContent(text="hi")],
            api="anthropic-messages",
            provider="anthropic",
            model="claude-opus-4",
            usage=Usage(
                input=10,
                output=5,
                cache_read=0,
                cache_write=0,
                total_tokens=15,
                cost=Cost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0, total=0.0),
            ),
            stop_reason="stop",
            timestamp=1,
        )

    def test_round_trip_camel_case_fields(self) -> None:
        msg = self._fresh()
        dumped = msg.model_dump(by_alias=True, exclude_none=True)
        assert dumped["role"] == "assistant"
        assert dumped["stopReason"] == "stop"
        assert dumped["usage"]["totalTokens"] == 15
        assert dumped["usage"]["cacheRead"] == 0

    def test_stop_reason_error_requires_error_message(self) -> None:
        msg = self._fresh()
        msg.stop_reason = "error"
        msg.error_message = "boom"
        dumped = msg.model_dump(by_alias=True, exclude_none=True)
        assert dumped["stopReason"] == "error"
        assert dumped["errorMessage"] == "boom"

    def test_response_id_camel_case(self) -> None:
        msg = self._fresh()
        msg.response_id = "msg_abc"
        assert msg.model_dump(by_alias=True, exclude_none=True)["responseId"] == "msg_abc"


class TestToolResultMessage:
    def test_round_trip(self) -> None:
        tr = ToolResultMessage(
            tool_call_id="call_1",
            tool_name="bash",
            content=[TextContent(text="ok")],
            is_error=False,
            timestamp=1,
        )
        dumped = tr.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {
            "role": "toolResult",
            "toolCallId": "call_1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
            "timestamp": 1,
        }

    def test_details_are_opaque(self) -> None:
        tr = ToolResultMessage(
            tool_call_id="c",
            tool_name="t",
            content=[TextContent(text="")],
            is_error=False,
            timestamp=0,
            details={"exit_code": 0, "stdout": "hi"},
        )
        dumped = tr.model_dump(by_alias=True, exclude_none=True)
        assert dumped["details"] == {"exit_code": 0, "stdout": "hi"}


# ---------------------------------------------------------------------------
# Discriminated unions — the Message and Content adapters dispatch on tags.
# ---------------------------------------------------------------------------


class TestMessageDiscriminatedUnion:
    adapter: TypeAdapter[Message] = TypeAdapter(Message)

    def test_user_role_parses_to_user_message(self) -> None:
        msg = self.adapter.validate_python({"role": "user", "content": "x", "timestamp": 1})
        assert isinstance(msg, UserMessage)

    def test_assistant_role_parses_to_assistant_message(self) -> None:
        msg = self.adapter.validate_python(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "x"}],
                "api": "anthropic-messages",
                "provider": "anthropic",
                "model": "claude",
                "usage": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 0,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                },
                "stopReason": "stop",
                "timestamp": 1,
            }
        )
        assert isinstance(msg, AssistantMessage)

    def test_tool_result_role_parses_to_tool_result_message(self) -> None:
        msg = self.adapter.validate_python(
            {
                "role": "toolResult",
                "toolCallId": "c",
                "toolName": "t",
                "content": [{"type": "text", "text": ""}],
                "isError": False,
                "timestamp": 1,
            }
        )
        assert isinstance(msg, ToolResultMessage)

    def test_unknown_role_raises(self) -> None:
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"role": "system", "content": "x", "timestamp": 1})


class TestContentDiscriminatedUnion:
    adapter: TypeAdapter[Content] = TypeAdapter(Content)

    @pytest.mark.parametrize(
        ("payload", "cls"),
        [
            ({"type": "text", "text": "x"}, TextContent),
            ({"type": "thinking", "thinking": "x"}, ThinkingContent),
            ({"type": "image", "data": "d", "mimeType": "image/png"}, ImageContent),
            ({"type": "toolCall", "id": "c", "name": "n", "arguments": {}}, ToolCall),
        ],
    )
    def test_dispatch_by_type_tag(self, payload: dict[str, Any], cls: type[object]) -> None:
        result = self.adapter.validate_python(payload)
        assert isinstance(result, cls)


# ---------------------------------------------------------------------------
# StreamOptions + Tool
# ---------------------------------------------------------------------------


class TestStreamOptions:
    def test_defaults_are_none(self) -> None:
        opts = StreamOptions()
        dumped = opts.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {}

    def test_max_tokens_camel_case(self) -> None:
        opts = StreamOptions(max_tokens=1024, cache_retention="short")
        dumped = opts.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {"maxTokens": 1024, "cacheRetention": "short"}

    def test_unknown_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            StreamOptions.model_validate({"maxTokens": 1, "notAField": True})


class TestTool:
    def test_tool_stores_raw_json_schema(self) -> None:
        tool = Tool(
            name="bash",
            description="run a command",
            parameters={"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
        )
        dumped = tool.model_dump(by_alias=True, exclude_none=True)
        assert dumped["parameters"]["required"] == ["cmd"]


# ---------------------------------------------------------------------------
# AssistantMessageEvent — discriminated union with nested AssistantMessage.
# ---------------------------------------------------------------------------


class TestAssistantMessageEvent:
    adapter: TypeAdapter[AssistantMessageEvent] = TypeAdapter(AssistantMessageEvent)

    @staticmethod
    def _partial() -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": [],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "claude",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "stop",
            "timestamp": 1,
        }

    def test_start_event(self) -> None:
        evt = self.adapter.validate_python({"type": "start", "partial": self._partial()})
        assert evt.type == "start"

    def test_text_delta_event(self) -> None:
        evt = self.adapter.validate_python(
            {"type": "text_delta", "contentIndex": 0, "delta": "hi", "partial": self._partial()}
        )
        assert evt.type == "text_delta"
        assert evt.content_index == 0
        assert evt.delta == "hi"

    def test_done_event_carries_final_message(self) -> None:
        final = self._partial()
        final["content"] = [{"type": "text", "text": "done"}]
        evt = self.adapter.validate_python({"type": "done", "reason": "stop", "message": final})
        assert evt.type == "done"
        assert evt.reason == "stop"
        assert isinstance(evt.message, AssistantMessage)

    def test_error_event_carries_error_message(self) -> None:
        final = self._partial()
        final["stopReason"] = "error"
        final["errorMessage"] = "boom"
        evt = self.adapter.validate_python({"type": "error", "reason": "error", "error": final})
        assert evt.type == "error"
        assert evt.error.error_message == "boom"


# ---------------------------------------------------------------------------
# JSON round-trip — the headline test: everything must survive JSON bytes.
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    adapter: TypeAdapter[Message] = TypeAdapter(Message)

    def test_full_assistant_message_round_trip_via_json(self) -> None:
        original = AssistantMessage(
            content=[
                ThinkingContent(thinking="let me think"),
                TextContent(text="hello"),
                ToolCall(id="c1", name="bash", arguments={"cmd": "ls"}),
            ],
            api="anthropic-messages",
            provider="anthropic",
            model="claude-opus-4",
            response_id="msg_123",
            usage=Usage(
                input=10,
                output=20,
                cache_read=1,
                cache_write=2,
                total_tokens=33,
                cost=Cost(input=0.01, output=0.02, cache_read=0.0, cache_write=0.0, total=0.03),
            ),
            stop_reason="toolUse",
            timestamp=1_700_000_000_000,
        )
        wire = original.model_dump_json(by_alias=True, exclude_none=True)
        parsed = self.adapter.validate_json(wire)
        assert parsed == original

    def test_keys_on_wire_are_camel_case(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="c",
            tool_name="t",
            content=[TextContent(text="ok")],
            is_error=False,
            timestamp=1,
        )
        wire = json.loads(msg.model_dump_json(by_alias=True, exclude_none=True))
        assert set(wire.keys()) == {"role", "toolCallId", "toolName", "content", "isError", "timestamp"}
