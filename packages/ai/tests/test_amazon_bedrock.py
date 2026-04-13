"""Tests for the pure transform functions in nu_ai.providers.amazon_bedrock.

Covers message conversion, tool conversion, stop reason mapping, params
building, system prompt construction, cache point injection, image handling,
thinking/reasoning support, and additional model request fields.
"""

from __future__ import annotations

import pytest
from nu_ai.providers.amazon_bedrock import (
    BedrockOptions,
    build_additional_model_request_fields,
    build_params,
    build_system_prompt,
    convert_messages,
    convert_tool_config,
    create_image_block,
    map_stop_reason,
    map_thinking_level_to_effort,
    normalize_tool_call_id,
    resolve_cache_retention,
    supports_adaptive_thinking,
    supports_prompt_caching,
    supports_thinking_signature,
)
from nu_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    ThinkingBudgets,
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
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    provider: str = "amazon-bedrock",
    reasoning: bool = False,
    max_tokens: int = 8096,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="bedrock-converse-stream",
        provider=provider,
        base_url="",
        reasoning=reasoning,
        input=["text", "image"],  # type: ignore[arg-type]
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=200_000,
        max_tokens=max_tokens,
    )


def _assistant(
    content: list[object],
    *,
    stop_reason: str = "stop",
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
) -> AssistantMessage:
    return AssistantMessage(
        content=content,  # type: ignore[arg-type]
        api="bedrock-converse-stream",
        provider="amazon-bedrock",
        model=model_id,
        usage=_usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        timestamp=1,
    )


# ---------------------------------------------------------------------------
# 1. normalize_tool_call_id
# ---------------------------------------------------------------------------


class TestNormalizeToolCallId:
    def test_already_valid(self) -> None:
        assert normalize_tool_call_id("abc123") == "abc123"

    def test_replaces_illegal_chars(self) -> None:
        result = normalize_tool_call_id("tool.call/id!")
        assert all(c.isalnum() or c in "_-" for c in result)

    def test_truncates_to_64_chars(self) -> None:
        long_id = "a" * 100
        assert len(normalize_tool_call_id(long_id)) == 64

    def test_preserves_hyphens_and_underscores(self) -> None:
        result = normalize_tool_call_id("tool-call_id")
        assert result == "tool-call_id"


# ---------------------------------------------------------------------------
# 2. map_stop_reason
# ---------------------------------------------------------------------------


class TestMapStopReason:
    def test_end_turn_maps_to_stop(self) -> None:
        assert map_stop_reason("end_turn") == "stop"

    def test_stop_sequence_maps_to_stop(self) -> None:
        assert map_stop_reason("stop_sequence") == "stop"

    def test_max_tokens_maps_to_length(self) -> None:
        assert map_stop_reason("max_tokens") == "length"

    def test_context_window_exceeded_maps_to_length(self) -> None:
        assert map_stop_reason("model_context_window_exceeded") == "length"

    def test_tool_use_maps_to_tool_use(self) -> None:
        assert map_stop_reason("tool_use") == "toolUse"

    def test_unknown_maps_to_error(self) -> None:
        assert map_stop_reason("something_weird") == "error"

    def test_none_maps_to_error(self) -> None:
        assert map_stop_reason(None) == "error"


# ---------------------------------------------------------------------------
# 3. supports_* helpers
# ---------------------------------------------------------------------------


class TestSupportHelpers:
    def test_adaptive_thinking_opus46(self) -> None:
        assert supports_adaptive_thinking("anthropic.claude-opus-4-6") is True

    def test_adaptive_thinking_sonnet46(self) -> None:
        assert supports_adaptive_thinking("anthropic.claude-sonnet-4.6") is True

    def test_adaptive_thinking_sonnet35(self) -> None:
        assert supports_adaptive_thinking("anthropic.claude-3-5-sonnet") is False

    def test_prompt_caching_claude4(self) -> None:
        m = _model(model_id="anthropic.claude-opus-4-0")
        assert supports_prompt_caching(m) is True

    def test_prompt_caching_claude37_sonnet(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        assert supports_prompt_caching(m) is True

    def test_prompt_caching_claude35_haiku(self) -> None:
        m = _model(model_id="anthropic.claude-3-5-haiku-20241022-v1:0")
        assert supports_prompt_caching(m) is True

    def test_no_prompt_caching_nova(self) -> None:
        m = _model(model_id="amazon.nova-lite-v1:0")
        assert supports_prompt_caching(m) is False

    def test_thinking_signature_anthropic_claude(self) -> None:
        m = _model(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert supports_thinking_signature(m) is True

    def test_no_thinking_signature_nova(self) -> None:
        m = _model(model_id="amazon.nova-pro-v1:0")
        assert supports_thinking_signature(m) is False


# ---------------------------------------------------------------------------
# 4. resolve_cache_retention
# ---------------------------------------------------------------------------


class TestResolveCacheRetention:
    def test_explicit_short(self) -> None:
        assert resolve_cache_retention("short") == "short"

    def test_explicit_long(self) -> None:
        assert resolve_cache_retention("long") == "long"

    def test_explicit_none(self) -> None:
        assert resolve_cache_retention("none") == "none"

    def test_defaults_to_short(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PI_CACHE_RETENTION", raising=False)
        assert resolve_cache_retention(None) == "short"

    def test_env_overrides_to_long(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PI_CACHE_RETENTION", "long")
        assert resolve_cache_retention(None) == "long"


# ---------------------------------------------------------------------------
# 5. convert_messages — basic cases
# ---------------------------------------------------------------------------


class TestConvertMessagesBasic:
    def test_string_user_message(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hello", timestamp=1)])
        result = convert_messages(ctx, m, "none")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"text": "hello"}]

    def test_user_message_with_text_content(self) -> None:
        m = _model()
        ctx = Context(
            messages=[
                UserMessage(
                    content=[TextContent(text="hello world")],
                    timestamp=1,
                )
            ]
        )
        result = convert_messages(ctx, m, "none")
        assert result[0]["content"] == [{"text": "hello world"}]

    def test_assistant_message_with_text(self) -> None:
        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant([TextContent(text="hello there")]),
            ]
        )
        result = convert_messages(ctx, m, "none")
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == [{"text": "hello there"}]

    def test_skips_empty_assistant_message(self) -> None:
        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant([]),
            ]
        )
        result = convert_messages(ctx, m, "none")
        # Empty assistant message should be skipped
        assert len(result) == 1

    def test_skips_empty_text_blocks_in_assistant(self) -> None:
        m = _model()
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant([TextContent(text="   ")]),
            ]
        )
        result = convert_messages(ctx, m, "none")
        assert len(result) == 1  # empty assistant message skipped

    def test_tool_call_in_assistant(self) -> None:
        m = _model()
        tc = ToolCall(id="call1", name="my_tool", arguments={"key": "val"})
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant([tc]),
            ]
        )
        result = convert_messages(ctx, m, "none")
        assert result[1]["role"] == "assistant"
        tool_use = result[1]["content"][0]["toolUse"]
        assert tool_use["toolUseId"] == "call1"
        assert tool_use["name"] == "my_tool"
        assert tool_use["input"] == {"key": "val"}


# ---------------------------------------------------------------------------
# 6. convert_messages — tool results
# ---------------------------------------------------------------------------


class TestConvertMessagesToolResults:
    def test_single_tool_result(self) -> None:
        m = _model()
        tc = ToolCall(id="call1", name="my_tool", arguments={})
        tr = ToolResultMessage(
            tool_call_id="call1",
            tool_name="my_tool",
            content=[TextContent(text="result text")],
            is_error=False,
            timestamp=2,
        )
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1), _assistant([tc]), tr])
        result = convert_messages(ctx, m, "none")
        # User, assistant, user-with-tool-result
        tool_result_msg = result[2]
        assert tool_result_msg["role"] == "user"
        tr_block = tool_result_msg["content"][0]["toolResult"]
        assert tr_block["toolUseId"] == "call1"
        assert tr_block["status"] == "success"

    def test_error_tool_result(self) -> None:
        m = _model()
        tc = ToolCall(id="call1", name="my_tool", arguments={})
        tr = ToolResultMessage(
            tool_call_id="call1",
            tool_name="my_tool",
            content=[TextContent(text="error!")],
            is_error=True,
            timestamp=2,
        )
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1), _assistant([tc]), tr])
        result = convert_messages(ctx, m, "none")
        tr_block = result[2]["content"][0]["toolResult"]
        assert tr_block["status"] == "error"

    def test_consecutive_tool_results_merged(self) -> None:
        m = _model()
        tc1 = ToolCall(id="c1", name="t1", arguments={})
        tc2 = ToolCall(id="c2", name="t2", arguments={})
        tr1 = ToolResultMessage(
            tool_call_id="c1",
            tool_name="t1",
            content=[TextContent(text="r1")],
            is_error=False,
            timestamp=2,
        )
        tr2 = ToolResultMessage(
            tool_call_id="c2",
            tool_name="t2",
            content=[TextContent(text="r2")],
            is_error=False,
            timestamp=3,
        )
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1), _assistant([tc1, tc2]), tr1, tr2])
        result = convert_messages(ctx, m, "none")
        # Consecutive tool results should be in ONE user message
        user_tool_msg = result[2]
        assert user_tool_msg["role"] == "user"
        assert len(user_tool_msg["content"]) == 2

    def test_image_in_tool_result(self) -> None:
        m = _model()
        tc = ToolCall(id="c1", name="t1", arguments={})
        import base64

        fake_data = base64.b64encode(b"\x89PNG\r\n").decode()
        tr = ToolResultMessage(
            tool_call_id="c1",
            tool_name="t1",
            content=[ImageContent(data=fake_data, mime_type="image/png")],
            is_error=False,
            timestamp=2,
        )
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1), _assistant([tc]), tr])
        result = convert_messages(ctx, m, "none")
        tool_result_content = result[2]["content"][0]["toolResult"]["content"]
        assert any("image" in block for block in tool_result_content)


# ---------------------------------------------------------------------------
# 7. convert_messages — thinking blocks
# ---------------------------------------------------------------------------


class TestConvertMessagesThinking:
    def test_thinking_with_signature_for_claude(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        thinking = ThinkingContent(
            thinking="deep thoughts",
            thinking_signature="sig123",
        )
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant([thinking], model_id=m.id),
            ]
        )
        result = convert_messages(ctx, m, "none")
        reasoning_block = result[1]["content"][0]["reasoningContent"]
        assert reasoning_block["reasoningText"]["text"] == "deep thoughts"
        assert reasoning_block["reasoningText"]["signature"] == "sig123"

    def test_thinking_without_signature_falls_back_to_text(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        thinking = ThinkingContent(thinking="deep thoughts", thinking_signature=None)
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant([thinking], model_id=m.id),
            ]
        )
        result = convert_messages(ctx, m, "none")
        # No valid signature → fall back to plain text block
        assert "text" in result[1]["content"][0]
        assert result[1]["content"][0]["text"] == "deep thoughts"

    def test_thinking_non_claude_omits_signature(self) -> None:
        m = _model(model_id="amazon.nova-pro-v1:0")
        thinking = ThinkingContent(thinking="thoughts", thinking_signature="sig")
        ctx = Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                _assistant([thinking], model_id=m.id),
            ]
        )
        result = convert_messages(ctx, m, "none")
        rc = result[1]["content"][0]["reasoningContent"]["reasoningText"]
        assert "signature" not in rc


# ---------------------------------------------------------------------------
# 8. build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_none_when_no_system_prompt(self) -> None:
        m = _model()
        assert build_system_prompt(None, m, "none") is None

    def test_plain_text_block(self) -> None:
        m = _model(model_id="amazon.nova-lite-v1:0")
        result = build_system_prompt("You are helpful.", m, "none")
        assert result == [{"text": "You are helpful."}]

    def test_cache_point_added_for_claude(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        result = build_system_prompt("Be helpful.", m, "short")
        assert result is not None
        assert len(result) == 2
        assert "cachePoint" in result[1]

    def test_no_cache_point_when_retention_none(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        result = build_system_prompt("Be helpful.", m, "none")
        assert result is not None
        assert len(result) == 1  # no cache point

    def test_long_cache_point_has_ttl(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        result = build_system_prompt("Be helpful.", m, "long")
        assert result is not None
        cache_block = result[1]["cachePoint"]
        assert "ttl" in cache_block


# ---------------------------------------------------------------------------
# 9. convert_tool_config
# ---------------------------------------------------------------------------


class TestConvertToolConfig:
    def _tool(self, name: str = "my_tool") -> Tool:
        return Tool(
            name=name,
            description=f"Does {name}",
            parameters={"type": "object", "properties": {}},
        )

    def test_none_when_no_tools(self) -> None:
        assert convert_tool_config(None) is None
        assert convert_tool_config([]) is None

    def test_none_when_tool_choice_is_none_str(self) -> None:
        opts = BedrockOptions(tool_choice="none")
        assert convert_tool_config([self._tool()], opts) is None

    def test_basic_tool_list(self) -> None:
        result = convert_tool_config([self._tool()])
        assert result is not None
        assert "tools" in result
        spec = result["tools"][0]["toolSpec"]
        assert spec["name"] == "my_tool"

    def test_auto_tool_choice(self) -> None:
        opts = BedrockOptions(tool_choice="auto")
        result = convert_tool_config([self._tool()], opts)
        assert result is not None
        assert result["toolChoice"] == {"auto": {}}

    def test_any_tool_choice(self) -> None:
        opts = BedrockOptions(tool_choice="any")
        result = convert_tool_config([self._tool()], opts)
        assert result is not None
        assert result["toolChoice"] == {"any": {}}

    def test_specific_tool_choice(self) -> None:
        opts = BedrockOptions(tool_choice={"type": "tool", "name": "my_tool"})
        result = convert_tool_config([self._tool()], opts)
        assert result is not None
        assert result["toolChoice"] == {"tool": {"name": "my_tool"}}

    def test_multiple_tools(self) -> None:
        tools = [self._tool("t1"), self._tool("t2")]
        result = convert_tool_config(tools)
        assert result is not None
        assert len(result["tools"]) == 2


# ---------------------------------------------------------------------------
# 10. map_thinking_level_to_effort
# ---------------------------------------------------------------------------


class TestMapThinkingLevelToEffort:
    def test_minimal(self) -> None:
        assert map_thinking_level_to_effort("minimal", "anything") == "low"

    def test_low(self) -> None:
        assert map_thinking_level_to_effort("low", "anything") == "low"

    def test_medium(self) -> None:
        assert map_thinking_level_to_effort("medium", "anything") == "medium"

    def test_high(self) -> None:
        assert map_thinking_level_to_effort("high", "anything") == "high"

    def test_xhigh_non_opus46(self) -> None:
        assert map_thinking_level_to_effort("xhigh", "anthropic.claude-3-7-sonnet") == "high"

    def test_xhigh_opus46_returns_max(self) -> None:
        assert map_thinking_level_to_effort("xhigh", "anthropic.claude-opus-4-6") == "max"


# ---------------------------------------------------------------------------
# 11. build_additional_model_request_fields
# ---------------------------------------------------------------------------


class TestBuildAdditionalModelRequestFields:
    def test_none_when_no_reasoning(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0", reasoning=True)
        opts = BedrockOptions(reasoning=None)
        assert build_additional_model_request_fields(m, opts) is None

    def test_none_when_model_not_reasoning(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0", reasoning=False)
        opts = BedrockOptions(reasoning="high")
        assert build_additional_model_request_fields(m, opts) is None

    def test_none_for_non_claude_model(self) -> None:
        m = _model(model_id="amazon.nova-pro-v1:0", reasoning=True)
        opts = BedrockOptions(reasoning="high")
        assert build_additional_model_request_fields(m, opts) is None

    def test_adaptive_thinking_for_opus46(self) -> None:
        m = _model(model_id="anthropic.claude-opus-4-6", reasoning=True)
        opts = BedrockOptions(reasoning="high")
        result = build_additional_model_request_fields(m, opts)
        assert result is not None
        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"]["effort"] == "high"

    def test_budget_based_for_older_claude(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0", reasoning=True)
        opts = BedrockOptions(reasoning="medium")
        result = build_additional_model_request_fields(m, opts)
        assert result is not None
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 8192

    def test_custom_budget_override(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0", reasoning=True)
        opts = BedrockOptions(
            reasoning="high",
            thinking_budgets=ThinkingBudgets(high=20000),
        )
        result = build_additional_model_request_fields(m, opts)
        assert result is not None
        assert result["thinking"]["budget_tokens"] == 20000

    def test_interleaved_thinking_included_by_default(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0", reasoning=True)
        opts = BedrockOptions(reasoning="high")
        result = build_additional_model_request_fields(m, opts)
        assert result is not None
        assert "anthropic_beta" in result

    def test_no_interleaved_thinking_when_disabled(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0", reasoning=True)
        opts = BedrockOptions(reasoning="high", interleaved_thinking=False)
        result = build_additional_model_request_fields(m, opts)
        assert result is not None
        assert "anthropic_beta" not in result


# ---------------------------------------------------------------------------
# 12. build_params
# ---------------------------------------------------------------------------


class TestBuildParams:
    def test_basic_structure(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        params = build_params(m, ctx)
        assert params["modelId"] == m.id
        assert "messages" in params
        assert len(params["messages"]) == 1

    def test_system_prompt_included(self) -> None:
        m = _model(model_id="amazon.nova-lite-v1:0")
        ctx = Context(
            system_prompt="You are helpful.",
            messages=[UserMessage(content="hi", timestamp=1)],
        )
        params = build_params(m, ctx)
        assert "system" in params
        assert params["system"][0]["text"] == "You are helpful."

    def test_no_system_prompt_field_when_absent(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        params = build_params(m, ctx)
        assert "system" not in params

    def test_inference_config_max_tokens(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        opts = BedrockOptions(max_tokens=1024)
        params = build_params(m, ctx, opts)
        assert params["inferenceConfig"]["maxTokens"] == 1024

    def test_inference_config_temperature(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        opts = BedrockOptions(temperature=0.7)
        params = build_params(m, ctx, opts)
        assert params["inferenceConfig"]["temperature"] == 0.7

    def test_tool_config_included(self) -> None:
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
        params = build_params(m, ctx)
        assert "toolConfig" in params

    def test_request_metadata_included(self) -> None:
        m = _model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        opts = BedrockOptions(request_metadata={"project": "demo"})
        params = build_params(m, ctx, opts)
        assert params["requestMetadata"] == {"project": "demo"}


# ---------------------------------------------------------------------------
# 13. create_image_block
# ---------------------------------------------------------------------------


class TestCreateImageBlock:
    def test_png_image(self) -> None:
        import base64

        data = base64.b64encode(b"fake-png-bytes").decode()
        block = create_image_block("image/png", data)
        assert block["format"] == "png"
        assert isinstance(block["source"]["bytes"], bytes)

    def test_jpeg_image(self) -> None:
        import base64

        data = base64.b64encode(b"fake-jpeg").decode()
        block = create_image_block("image/jpeg", data)
        assert block["format"] == "jpeg"

    def test_jpg_alias(self) -> None:
        import base64

        data = base64.b64encode(b"fake-jpeg").decode()
        block = create_image_block("image/jpg", data)
        assert block["format"] == "jpeg"

    def test_gif_image(self) -> None:
        import base64

        data = base64.b64encode(b"fake-gif").decode()
        block = create_image_block("image/gif", data)
        assert block["format"] == "gif"

    def test_webp_image(self) -> None:
        import base64

        data = base64.b64encode(b"fake-webp").decode()
        block = create_image_block("image/webp", data)
        assert block["format"] == "webp"

    def test_unknown_mime_raises(self) -> None:
        import base64

        data = base64.b64encode(b"bytes").decode()
        with pytest.raises(ValueError, match="Unknown image type"):
            create_image_block("image/bmp", data)


# ---------------------------------------------------------------------------
# 14. cache point injected on last user message
# ---------------------------------------------------------------------------


class TestCachePointOnLastUserMessage:
    def test_cache_point_added_to_last_user_msg(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        result = convert_messages(ctx, m, "short")
        last_content = result[-1]["content"]
        assert any("cachePoint" in block for block in last_content)

    def test_no_cache_point_when_retention_none(self) -> None:
        m = _model(model_id="anthropic.claude-3-7-sonnet-20250219-v1:0")
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        result = convert_messages(ctx, m, "none")
        last_content = result[-1]["content"]
        assert not any("cachePoint" in block for block in last_content)
