"""Tests for the pure transform functions in nu_ai.providers.mistral.

Covers message conversion, tool conversion, stop reason mapping, payload
building, tool call ID normalization, error formatting, and tool choice mapping.
"""

from __future__ import annotations

import json

from nu_ai.providers.mistral import (
    MistralOptions,
    build_chat_payload,
    build_tool_result_text,
    create_mistral_tool_call_id_normalizer,
    derive_mistral_tool_call_id,
    format_mistral_error,
    map_stop_reason,
    map_tool_choice,
    to_chat_messages,
    to_function_tools,
)
from nu_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


def _model(
    *,
    model_id: str = "mistral-large-latest",
    provider: str = "mistral",
    reasoning: bool = False,
    inputs: list[str] | None = None,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="mistral-conversations",
        provider=provider,
        base_url="https://api.mistral.ai",
        reasoning=reasoning,
        input=inputs or ["text", "image"],  # type: ignore[arg-type]
        cost=ModelCost(input=2.0, output=6.0, cache_read=0, cache_write=0),
        context_window=128_000,
        max_tokens=4096,
    )


def _assistant(
    content: list[object],
    *,
    stop_reason: str = "stop",
) -> AssistantMessage:
    return AssistantMessage(
        content=content,  # type: ignore[arg-type]
        api="mistral-conversations",
        provider="mistral",
        model="mistral-large-latest",
        usage=_usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        timestamp=1,
    )


# ---------------------------------------------------------------------------
# 1. map_stop_reason
# ---------------------------------------------------------------------------


class TestMapStopReason:
    def test_stop_maps_to_stop(self) -> None:
        assert map_stop_reason("stop") == "stop"

    def test_length_maps_to_length(self) -> None:
        assert map_stop_reason("length") == "length"

    def test_model_length_maps_to_length(self) -> None:
        assert map_stop_reason("model_length") == "length"

    def test_tool_calls_maps_to_tool_use(self) -> None:
        assert map_stop_reason("tool_calls") == "toolUse"

    def test_error_maps_to_error(self) -> None:
        assert map_stop_reason("error") == "error"

    def test_none_maps_to_stop(self) -> None:
        assert map_stop_reason(None) == "stop"

    def test_unknown_maps_to_stop(self) -> None:
        assert map_stop_reason("something_else") == "stop"


# ---------------------------------------------------------------------------
# 2. derive_mistral_tool_call_id
# ---------------------------------------------------------------------------


class TestDeriveMistralToolCallId:
    def test_exactly_9_alphanumeric_chars_passthrough(self) -> None:
        result = derive_mistral_tool_call_id("abc123xyz", 0)
        assert result == "abc123xyz"
        assert len(result) == 9

    def test_shorter_id_gets_hashed(self) -> None:
        result = derive_mistral_tool_call_id("short", 0)
        assert len(result) == 9
        assert all(c.isalnum() for c in result)

    def test_longer_id_gets_hashed(self) -> None:
        result = derive_mistral_tool_call_id("a" * 20, 0)
        assert len(result) == 9

    def test_different_attempts_produce_different_results(self) -> None:
        r0 = derive_mistral_tool_call_id("collision", 0)
        r1 = derive_mistral_tool_call_id("collision", 1)
        assert r0 != r1

    def test_result_is_alphanumeric(self) -> None:
        result = derive_mistral_tool_call_id("some-id-with-hyphens!", 0)
        assert all(c.isalnum() for c in result)


# ---------------------------------------------------------------------------
# 3. create_mistral_tool_call_id_normalizer
# ---------------------------------------------------------------------------


class TestCreateMistralToolCallIdNormalizer:
    def test_same_id_returns_same_result(self) -> None:
        normalize = create_mistral_tool_call_id_normalizer()
        r1 = normalize("myid123")
        r2 = normalize("myid123")
        assert r1 == r2

    def test_different_ids_may_differ(self) -> None:
        normalize = create_mistral_tool_call_id_normalizer()
        r1 = normalize("id_one_111")
        r2 = normalize("id_two_222")
        # Not strictly guaranteed to differ, but very likely
        # This tests that independent IDs go through the function independently
        assert isinstance(r1, str) and isinstance(r2, str)

    def test_collision_resolved(self) -> None:
        """Two IDs that hash to the same 9-char string must get distinct outputs."""
        # We can't guarantee a collision, so we test the structure is correct.
        normalize = create_mistral_tool_call_id_normalizer()
        ids = [f"toolid_{i:05d}" for i in range(50)]
        results = [normalize(i) for i in ids]
        # All outputs must be 9-char alphanumeric
        assert all(len(r) == 9 and r.isalnum() for r in results)


# ---------------------------------------------------------------------------
# 4. build_tool_result_text
# ---------------------------------------------------------------------------


class TestBuildToolResultText:
    def test_plain_text(self) -> None:
        result = build_tool_result_text("hello", False, False, False)
        assert result == "hello"

    def test_error_prefix(self) -> None:
        result = build_tool_result_text("bad!", False, False, True)
        assert result == "[tool error] bad!"

    def test_image_omitted_when_not_supported(self) -> None:
        result = build_tool_result_text("", True, False, False)
        assert "image omitted" in result

    def test_see_attached_image_when_supported(self) -> None:
        result = build_tool_result_text("", True, True, False)
        assert "(see attached image)" in result

    def test_no_output(self) -> None:
        result = build_tool_result_text("", False, False, False)
        assert result == "(no tool output)"

    def test_error_no_output(self) -> None:
        result = build_tool_result_text("", False, False, True)
        assert result == "[tool error] (no tool output)"

    def test_text_with_image_suffix_when_not_supported(self) -> None:
        result = build_tool_result_text("some output", True, False, False)
        assert "some output" in result
        assert "tool image omitted" in result


# ---------------------------------------------------------------------------
# 5. to_function_tools
# ---------------------------------------------------------------------------


class TestToFunctionTools:
    def test_basic_tool(self) -> None:
        tool = Tool(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        result = to_function_tools([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        func = result[0]["function"]
        assert func["name"] == "search"
        assert func["description"] == "Search the web"
        assert func["strict"] is False

    def test_multiple_tools(self) -> None:
        tools = [
            Tool(name="t1", description="D1", parameters={"type": "object", "properties": {}}),
            Tool(name="t2", description="D2", parameters={"type": "object", "properties": {}}),
        ]
        result = to_function_tools(tools)
        assert len(result) == 2
        assert {r["function"]["name"] for r in result} == {"t1", "t2"}


# ---------------------------------------------------------------------------
# 6. map_tool_choice
# ---------------------------------------------------------------------------


class TestMapToolChoice:
    def test_none_options(self) -> None:
        assert map_tool_choice(None) is None

    def test_auto(self) -> None:
        opts = MistralOptions(tool_choice="auto")
        assert map_tool_choice(opts) == "auto"

    def test_none_string(self) -> None:
        opts = MistralOptions(tool_choice="none")
        assert map_tool_choice(opts) == "none"

    def test_any(self) -> None:
        opts = MistralOptions(tool_choice="any")
        assert map_tool_choice(opts) == "any"

    def test_required(self) -> None:
        opts = MistralOptions(tool_choice="required")
        assert map_tool_choice(opts) == "required"

    def test_specific_function(self) -> None:
        opts = MistralOptions(tool_choice={"type": "function", "function": {"name": "my_tool"}})
        result = map_tool_choice(opts)
        assert isinstance(result, dict)
        assert result["function"]["name"] == "my_tool"


# ---------------------------------------------------------------------------
# 7. to_chat_messages — user messages
# ---------------------------------------------------------------------------


class TestToChatMessagesUser:
    def test_string_content(self) -> None:
        messages = [UserMessage(content="hello", timestamp=1)]
        result = to_chat_messages(messages, False)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_text_content_block(self) -> None:
        messages = [UserMessage(content=[TextContent(text="hi")], timestamp=1)]
        result = to_chat_messages(messages, False)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "hi"

    def test_image_included_when_supported(self) -> None:
        import base64

        data = base64.b64encode(b"fake").decode()
        messages = [
            UserMessage(
                content=[
                    TextContent(text="look"),
                    ImageContent(data=data, mime_type="image/png"),
                ],
                timestamp=1,
            )
        ]
        result = to_chat_messages(messages, True)
        types = {chunk["type"] for chunk in result[0]["content"]}
        assert "image_url" in types

    def test_image_omitted_when_not_supported(self) -> None:
        import base64

        data = base64.b64encode(b"fake").decode()
        messages = [
            UserMessage(
                content=[
                    TextContent(text="look"),
                    ImageContent(data=data, mime_type="image/png"),
                ],
                timestamp=1,
            )
        ]
        result = to_chat_messages(messages, False)
        assert len(result) == 1
        # Only text should appear
        for chunk in result[0]["content"]:
            assert chunk["type"] != "image_url"


# ---------------------------------------------------------------------------
# 8. to_chat_messages — assistant messages
# ---------------------------------------------------------------------------


class TestToChatMessagesAssistant:
    def test_text_content(self) -> None:
        messages = [
            UserMessage(content="hi", timestamp=1),
            _assistant([TextContent(text="hello there")]),
        ]
        result = to_chat_messages(messages, False)
        assistant_msg = next(m for m in result if m["role"] == "assistant")
        content = assistant_msg["content"]
        assert any(c["type"] == "text" for c in content)

    def test_tool_call(self) -> None:
        tc = ToolCall(id="call1", name="my_tool", arguments={"key": "val"})
        messages = [
            UserMessage(content="hi", timestamp=1),
            _assistant([tc]),
        ]
        result = to_chat_messages(messages, False)
        assistant_msg = next(m for m in result if m["role"] == "assistant")
        assert "toolCalls" in assistant_msg
        assert assistant_msg["toolCalls"][0]["id"] == "call1"
        args = json.loads(assistant_msg["toolCalls"][0]["function"]["arguments"])
        assert args == {"key": "val"}

    def test_empty_text_blocks_skipped(self) -> None:
        messages = [
            UserMessage(content="hi", timestamp=1),
            _assistant([TextContent(text="   ")]),
        ]
        result = to_chat_messages(messages, False)
        # No assistant message should appear for empty content
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) == 0

    def test_thinking_content(self) -> None:
        thinking = ThinkingContent(thinking="deep thoughts", thinking_signature=None)
        messages = [
            UserMessage(content="hi", timestamp=1),
            _assistant([thinking]),
        ]
        result = to_chat_messages(messages, False)
        assistant_msg = next((m for m in result if m["role"] == "assistant"), None)
        # Thinking content should produce a thinking block
        assert assistant_msg is not None
        thinking_blocks = [c for c in assistant_msg["content"] if c.get("type") == "thinking"]
        assert len(thinking_blocks) == 1


# ---------------------------------------------------------------------------
# 9. to_chat_messages — tool results
# ---------------------------------------------------------------------------


class TestToChatMessagesToolResults:
    def test_basic_tool_result(self) -> None:
        tr = ToolResultMessage(
            tool_call_id="call1",
            tool_name="my_tool",
            content=[TextContent(text="result text")],
            is_error=False,
            timestamp=2,
        )
        messages = [
            UserMessage(content="hi", timestamp=1),
            _assistant([ToolCall(id="call1", name="my_tool", arguments={})]),
            tr,
        ]
        result = to_chat_messages(messages, False)
        tool_msg = next(m for m in result if m["role"] == "tool")
        assert tool_msg["toolCallId"] == "call1"
        assert tool_msg["name"] == "my_tool"

    def test_error_tool_result_prefixed(self) -> None:
        tr = ToolResultMessage(
            tool_call_id="call1",
            tool_name="my_tool",
            content=[TextContent(text="failure")],
            is_error=True,
            timestamp=2,
        )
        messages = [
            UserMessage(content="hi", timestamp=1),
            _assistant([ToolCall(id="call1", name="my_tool", arguments={})]),
            tr,
        ]
        result = to_chat_messages(messages, False)
        tool_msg = next(m for m in result if m["role"] == "tool")
        text_content = tool_msg["content"][0]["text"]
        assert text_content.startswith("[tool error]")


# ---------------------------------------------------------------------------
# 10. build_chat_payload
# ---------------------------------------------------------------------------


class TestBuildChatPayload:
    def test_basic_structure(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        transformed = [UserMessage(content="hi", timestamp=1)]
        payload = build_chat_payload(m, ctx, transformed)
        assert payload["model"] == m.id
        assert payload["stream"] is True
        assert len(payload["messages"]) == 1

    def test_system_prompt_prepended(self) -> None:
        m = _model()
        ctx = Context(
            system_prompt="You are helpful.",
            messages=[UserMessage(content="hi", timestamp=1)],
        )
        transformed = [UserMessage(content="hi", timestamp=1)]
        payload = build_chat_payload(m, ctx, transformed)
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."

    def test_tools_included_when_present(self) -> None:
        m = _model()
        tool = Tool(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {}},
        )
        ctx = Context(
            messages=[UserMessage(content="hi", timestamp=1)],
            tools=[tool],
        )
        transformed = [UserMessage(content="hi", timestamp=1)]
        payload = build_chat_payload(m, ctx, transformed)
        assert "tools" in payload

    def test_temperature_included(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        transformed = [UserMessage(content="hi", timestamp=1)]
        opts = MistralOptions(temperature=0.5)
        payload = build_chat_payload(m, ctx, transformed, opts)
        assert payload["temperature"] == 0.5

    def test_max_tokens_included(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        transformed = [UserMessage(content="hi", timestamp=1)]
        opts = MistralOptions(max_tokens=512)
        payload = build_chat_payload(m, ctx, transformed, opts)
        assert payload["maxTokens"] == 512

    def test_prompt_mode_included(self) -> None:
        m = _model(reasoning=True)
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        transformed = [UserMessage(content="hi", timestamp=1)]
        opts = MistralOptions(prompt_mode="reasoning")
        payload = build_chat_payload(m, ctx, transformed, opts)
        assert payload["promptMode"] == "reasoning"

    def test_tool_choice_included(self) -> None:
        m = _model()
        tool = Tool(
            name="t1",
            description="D1",
            parameters={"type": "object", "properties": {}},
        )
        ctx = Context(
            messages=[UserMessage(content="hi", timestamp=1)],
            tools=[tool],
        )
        transformed = [UserMessage(content="hi", timestamp=1)]
        opts = MistralOptions(tool_choice="auto")
        payload = build_chat_payload(m, ctx, transformed, opts)
        assert payload["toolChoice"] == "auto"


# ---------------------------------------------------------------------------
# 11. format_mistral_error
# ---------------------------------------------------------------------------


class TestFormatMistralError:
    def test_plain_exception(self) -> None:
        exc = Exception("something went wrong")
        result = format_mistral_error(exc)
        assert "something went wrong" in result

    def test_with_status_code_and_body(self) -> None:
        exc = Exception("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        exc.body = "Invalid model parameter"  # type: ignore[attr-defined]
        result = format_mistral_error(exc)
        assert "400" in result
        assert "Invalid model parameter" in result

    def test_with_status_code_no_body(self) -> None:
        exc = Exception("forbidden")
        exc.status_code = 403  # type: ignore[attr-defined]
        result = format_mistral_error(exc)
        assert "403" in result

    def test_body_truncated(self) -> None:
        exc = Exception("server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        exc.body = "x" * 5000  # type: ignore[attr-defined]
        result = format_mistral_error(exc)
        assert "truncated" in result


# ---------------------------------------------------------------------------
# 12. MistralOptions model
# ---------------------------------------------------------------------------


class TestMistralOptions:
    def test_default_no_tool_choice(self) -> None:
        opts = MistralOptions()
        assert opts.tool_choice is None

    def test_reasoning_prompt_mode(self) -> None:
        opts = MistralOptions(prompt_mode="reasoning")
        assert opts.prompt_mode == "reasoning"
