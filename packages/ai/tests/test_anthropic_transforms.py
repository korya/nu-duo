"""Tests for the pure transform functions in nu_ai.providers.anthropic.

Covers every branch of the logic ported from
``packages/ai/src/providers/anthropic.ts`` that does not require the
``anthropic`` SDK — message conversion, tool conversion, params building,
thinking mode selection, OAuth detection, Claude Code tool-name
normalization, and stop-reason mapping.

Also ports the documented expectations from
``packages/ai/test/anthropic-thinking-disable.test.ts`` (the payload-capture
portion that works without network access).
"""

from __future__ import annotations

import pytest
from nu_ai.providers.anthropic import (
    build_params,
    convert_content_blocks,
    convert_messages,
    convert_tools,
    from_claude_code_name,
    get_cache_control,
    is_oauth_token,
    map_stop_reason,
    map_thinking_level_to_effort,
    merge_headers,
    normalize_tool_call_id,
    resolve_cache_retention,
    supports_adaptive_thinking,
    to_claude_code_name,
)
from nu_ai.types import (
    AnthropicOptions,
    AssistantMessage,
    Context,
    Cost,
    ImageContent,
    Message,
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
    model_id: str = "claude-sonnet-4-5",
    provider: str = "anthropic",
    base_url: str = "https://api.anthropic.com",
    reasoning: bool = True,
    inputs: list[str] | None = None,
    max_tokens: int = 64_000,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="anthropic-messages",
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        input=inputs or ["text", "image"],  # type: ignore[arg-type]
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=200_000,
        max_tokens=max_tokens,
    )


def _assistant(
    content: list[object],
    *,
    api: str = "anthropic-messages",
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5",
    stop_reason: str = "stop",
) -> AssistantMessage:
    return AssistantMessage(
        content=content,  # type: ignore[arg-type]
        api=api,
        provider=provider,
        model=model,
        usage=_usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        timestamp=1,
    )


# ---------------------------------------------------------------------------
# resolve_cache_retention / get_cache_control
# ---------------------------------------------------------------------------


class TestResolveCacheRetention:
    def test_default_is_short(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PI_CACHE_RETENTION", raising=False)
        assert resolve_cache_retention(None) == "short"

    def test_explicit_override(self) -> None:
        assert resolve_cache_retention("long") == "long"
        assert resolve_cache_retention("none") == "none"

    def test_env_var_long(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PI_CACHE_RETENTION", "long")
        assert resolve_cache_retention(None) == "long"

    def test_env_var_other_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PI_CACHE_RETENTION", "crazy")
        assert resolve_cache_retention(None) == "short"


class TestGetCacheControl:
    def test_none_returns_no_control(self) -> None:
        result = get_cache_control("https://api.anthropic.com", "none")
        assert result["retention"] == "none"
        assert result["cache_control"] is None

    def test_short_returns_ephemeral_without_ttl(self) -> None:
        result = get_cache_control("https://api.anthropic.com", "short")
        assert result["cache_control"] == {"type": "ephemeral"}

    def test_long_on_anthropic_uses_1h_ttl(self) -> None:
        result = get_cache_control("https://api.anthropic.com", "long")
        assert result["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_long_on_non_anthropic_has_no_ttl(self) -> None:
        result = get_cache_control("https://api.individual.githubcopilot.com", "long")
        assert result["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# is_oauth_token
# ---------------------------------------------------------------------------


class TestIsOauthToken:
    def test_oauth_prefix_detected(self) -> None:
        assert is_oauth_token("sk-ant-oat01-abcdef") is True

    def test_api_key_not_oauth(self) -> None:
        assert is_oauth_token("sk-ant-api03-abcdef") is False

    def test_empty_string(self) -> None:
        assert is_oauth_token("") is False


# ---------------------------------------------------------------------------
# Claude Code tool name mapping
# ---------------------------------------------------------------------------


class TestClaudeCodeToolNames:
    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("read", "Read"),
            ("READ", "Read"),
            ("write", "Write"),
            ("edit", "Edit"),
            ("bash", "Bash"),
            ("grep", "Grep"),
            ("glob", "Glob"),
            ("todowrite", "TodoWrite"),
            ("webfetch", "WebFetch"),
            ("websearch", "WebSearch"),
        ],
    )
    def test_to_claude_code_name(self, input_name: str, expected: str) -> None:
        assert to_claude_code_name(input_name) == expected

    def test_to_claude_code_passthrough_for_unknown(self) -> None:
        assert to_claude_code_name("my_custom_tool") == "my_custom_tool"

    def test_from_claude_code_name_with_tools_restores_original(self) -> None:
        tools = [Tool(name="todowrite", description="", parameters={"type": "object"})]
        assert from_claude_code_name("TodoWrite", tools) == "todowrite"

    def test_from_claude_code_name_without_tools_returns_as_is(self) -> None:
        assert from_claude_code_name("TodoWrite", None) == "TodoWrite"

    def test_from_claude_code_name_not_found_returns_as_is(self) -> None:
        tools = [Tool(name="other", description="", parameters={"type": "object"})]
        assert from_claude_code_name("TodoWrite", tools) == "TodoWrite"


# ---------------------------------------------------------------------------
# merge_headers
# ---------------------------------------------------------------------------


class TestMergeHeaders:
    def test_none_sources_ignored(self) -> None:
        assert merge_headers(None, None) == {}

    def test_later_overrides_earlier(self) -> None:
        result = merge_headers({"a": "1"}, {"a": "2", "b": "3"})
        assert result == {"a": "2", "b": "3"}

    def test_mixed_none_and_dicts(self) -> None:
        result = merge_headers({"x": "y"}, None, {"z": "w"})
        assert result == {"x": "y", "z": "w"}


# ---------------------------------------------------------------------------
# supports_adaptive_thinking / map_thinking_level_to_effort
# ---------------------------------------------------------------------------


class TestSupportsAdaptiveThinking:
    @pytest.mark.parametrize(
        "model_id",
        [
            "claude-opus-4-6",
            "claude-opus-4.6",
            "claude-sonnet-4-6",
            "claude-sonnet-4.6",
            "claude-opus-4-6-20251120",
        ],
    )
    def test_supported(self, model_id: str) -> None:
        assert supports_adaptive_thinking(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        ["claude-opus-4", "claude-sonnet-4-5", "claude-3-5-sonnet", "claude-opus-3"],
    )
    def test_not_supported(self, model_id: str) -> None:
        assert supports_adaptive_thinking(model_id) is False


class TestMapThinkingLevelToEffort:
    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            (None, "high"),
        ],
    )
    def test_non_xhigh(self, level: str | None, expected: str) -> None:
        assert map_thinking_level_to_effort(level, "claude-sonnet-4-5") == expected  # type: ignore[arg-type]

    def test_xhigh_on_opus_46_becomes_max(self) -> None:
        assert map_thinking_level_to_effort("xhigh", "claude-opus-4-6") == "max"

    def test_xhigh_on_sonnet_46_becomes_high(self) -> None:
        assert map_thinking_level_to_effort("xhigh", "claude-sonnet-4-6") == "high"

    def test_xhigh_on_older_becomes_high(self) -> None:
        assert map_thinking_level_to_effort("xhigh", "claude-sonnet-4-5") == "high"


# ---------------------------------------------------------------------------
# normalize_tool_call_id / map_stop_reason
# ---------------------------------------------------------------------------


class TestNormalizeToolCallId:
    def test_keeps_valid_chars(self) -> None:
        assert normalize_tool_call_id("tool_call-123") == "tool_call-123"

    def test_replaces_illegal_with_underscore(self) -> None:
        assert normalize_tool_call_id("tool|call:foo$bar") == "tool_call_foo_bar"

    def test_truncates_to_64(self) -> None:
        long = "a" * 100
        assert len(normalize_tool_call_id(long)) == 64


class TestMapStopReason:
    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "toolUse"),
            ("refusal", "error"),
            ("pause_turn", "stop"),
            ("stop_sequence", "stop"),
            ("sensitive", "error"),
        ],
    )
    def test_known_reasons(self, reason: str, expected: str) -> None:
        assert map_stop_reason(reason) == expected

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unhandled stop reason"):
            map_stop_reason("brand_new_reason")


# ---------------------------------------------------------------------------
# convert_content_blocks
# ---------------------------------------------------------------------------


class TestConvertContentBlocks:
    def test_text_only_returns_string(self) -> None:
        result = convert_content_blocks([TextContent(text="hello")])
        assert result == "hello"

    def test_multiple_text_blocks_joined_with_newline(self) -> None:
        result = convert_content_blocks([TextContent(text="a"), TextContent(text="b")])
        assert result == "a\nb"

    def test_with_image_returns_block_list(self) -> None:
        result = convert_content_blocks(
            [
                TextContent(text="look at this"),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ]
        )
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "look at this"}
        assert result[1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
        }

    def test_image_only_gets_placeholder_text(self) -> None:
        result = convert_content_blocks([ImageContent(data="d", mime_type="image/jpeg")])
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "(see attached image)"}


# ---------------------------------------------------------------------------
# convert_tools
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_basic_tool(self) -> None:
        tools = [
            Tool(
                name="bash",
                description="run",
                parameters={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            )
        ]
        converted = convert_tools(tools, is_oauth_token=False)
        assert converted == [
            {
                "name": "bash",
                "description": "run",
                "input_schema": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            }
        ]

    def test_oauth_renames_matching_tools(self) -> None:
        tools = [Tool(name="read", description="", parameters={"type": "object"})]
        converted = convert_tools(tools, is_oauth_token=True)
        assert converted[0]["name"] == "Read"

    def test_empty_list(self) -> None:
        assert convert_tools([], is_oauth_token=False) == []

    def test_missing_properties_defaults_to_empty(self) -> None:
        tools = [Tool(name="t", description="", parameters={"type": "object"})]
        converted = convert_tools(tools, is_oauth_token=False)
        assert converted[0]["input_schema"]["properties"] == {}
        assert converted[0]["input_schema"]["required"] == []


# ---------------------------------------------------------------------------
# convert_messages
# ---------------------------------------------------------------------------


class TestConvertMessages:
    def test_user_string_content(self) -> None:
        msgs: list[Message] = [UserMessage(content="hi", timestamp=1)]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        assert result == [{"role": "user", "content": "hi"}]

    def test_user_empty_string_skipped(self) -> None:
        msgs: list[Message] = [UserMessage(content="   ", timestamp=1)]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        assert result == []

    def test_user_with_text_and_image(self) -> None:
        msgs: list[Message] = [
            UserMessage(
                content=[
                    TextContent(text="look"),
                    ImageContent(data="d", mime_type="image/png"),
                ],
                timestamp=1,
            ),
        ]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

    def test_user_image_filtered_for_text_only_model(self) -> None:
        msgs: list[Message] = [
            UserMessage(
                content=[
                    TextContent(text="hi"),
                    ImageContent(data="d", mime_type="image/png"),
                ],
                timestamp=1,
            ),
        ]
        text_only = _model(inputs=["text"])
        result = convert_messages(msgs, text_only, is_oauth_token=False)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert all(b["type"] != "image" for b in content)

    def test_assistant_text_blocks(self) -> None:
        msgs: list[Message] = [_assistant(content=[TextContent(text="hi")])]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        assert result[0]["role"] == "assistant"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0] == {"type": "text", "text": "hi"}

    def test_assistant_empty_text_skipped(self) -> None:
        msgs: list[Message] = [_assistant(content=[TextContent(text="   ")])]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        assert result == []  # no content → message dropped

    def test_assistant_thinking_with_signature(self) -> None:
        msgs: list[Message] = [
            _assistant(
                content=[
                    ThinkingContent(thinking="hmm", thinking_signature="sig"),
                    TextContent(text="answer"),
                ],
            ),
        ]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "thinking", "thinking": "hmm", "signature": "sig"}
        assert content[1] == {"type": "text", "text": "answer"}

    def test_assistant_thinking_without_signature_becomes_text(self) -> None:
        msgs: list[Message] = [
            _assistant(
                content=[
                    ThinkingContent(thinking="hmm"),
                    TextContent(text="answer"),
                ],
            ),
        ]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "hmm"}
        assert content[1] == {"type": "text", "text": "answer"}

    def test_assistant_redacted_thinking(self) -> None:
        msgs: list[Message] = [
            _assistant(
                content=[
                    ThinkingContent(thinking="", thinking_signature="enc", redacted=True),
                    TextContent(text="answer"),
                ],
            ),
        ]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "redacted_thinking", "data": "enc"}

    def test_assistant_tool_call(self) -> None:
        msgs: list[Message] = [
            _assistant(
                content=[ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})],
                stop_reason="toolUse",
            ),
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="bash",
                content=[TextContent(text="ok")],
                is_error=False,
                timestamp=1,
            ),
        ]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        assert len(result) == 2
        assert result[0]["content"][0] == {
            "type": "tool_use",
            "id": "c1",
            "name": "bash",
            "input": {"cmd": "ls"},
        }
        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["type"] == "tool_result"
        assert result[1]["content"][0]["tool_use_id"] == "c1"

    def test_oauth_renames_assistant_tool_calls(self) -> None:
        msgs: list[Message] = [
            _assistant(
                content=[ToolCall(id="c1", name="read", arguments={})],
                stop_reason="toolUse",
            ),
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="read",
                content=[TextContent(text="ok")],
                is_error=False,
                timestamp=1,
            ),
        ]
        result = convert_messages(msgs, _model(), is_oauth_token=True)
        assert result[0]["content"][0]["name"] == "Read"

    def test_consecutive_tool_results_merged(self) -> None:
        msgs: list[Message] = [
            _assistant(
                content=[
                    ToolCall(id="a", name="bash", arguments={}),
                    ToolCall(id="b", name="bash", arguments={}),
                ],
                stop_reason="toolUse",
            ),
            ToolResultMessage(
                tool_call_id="a",
                tool_name="bash",
                content=[TextContent(text="1")],
                is_error=False,
                timestamp=1,
            ),
            ToolResultMessage(
                tool_call_id="b",
                tool_name="bash",
                content=[TextContent(text="2")],
                is_error=False,
                timestamp=1,
            ),
        ]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        # Two consecutive tool results collapse into a single user message
        # carrying both tool_result blocks.
        assert len(result) == 2
        content = result[1]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["tool_use_id"] == "a"
        assert content[1]["tool_use_id"] == "b"

    def test_tool_result_normalized_id(self) -> None:
        # transform_messages re-maps tool call ids before convert_messages
        # runs, so an id with forbidden characters should come out clean.
        msgs: list[Message] = [
            _assistant(
                content=[ToolCall(id="c1|weird$", name="bash", arguments={})],
                stop_reason="toolUse",
                api="openai-completions",  # force cross-model normalization
                provider="openai",
                model="gpt-4o",
            ),
            ToolResultMessage(
                tool_call_id="c1|weird$",
                tool_name="bash",
                content=[TextContent(text="ok")],
                is_error=False,
                timestamp=1,
            ),
        ]
        result = convert_messages(msgs, _model(), is_oauth_token=False)
        tc_block = result[0]["content"][0]
        tr_block = result[1]["content"][0]
        assert tc_block["id"] == "c1_weird_"
        assert tr_block["tool_use_id"] == "c1_weird_"


# ---------------------------------------------------------------------------
# build_params
# ---------------------------------------------------------------------------


def _context(*messages: Message, tools: list[Tool] | None = None, system: str | None = None) -> Context:
    return Context(system_prompt=system, messages=list(messages), tools=tools)


class TestBuildParams:
    def test_basic_params(self) -> None:
        model = _model(max_tokens=3_000)
        ctx = _context(UserMessage(content="hi", timestamp=1))
        params = build_params(model, ctx, is_oauth_token=False)
        assert params["model"] == "claude-sonnet-4-5"
        assert params["stream"] is True
        # Default max_tokens is model.max_tokens / 3 (integer division).
        assert params["max_tokens"] == 1_000

    def test_custom_max_tokens(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(max_tokens=500),
        )
        assert params["max_tokens"] == 500

    def test_system_prompt_non_oauth(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1), system="you are helpful"),
            is_oauth_token=False,
        )
        assert params["system"] == [
            {"type": "text", "text": "you are helpful", "cache_control": {"type": "ephemeral"}},
        ]

    def test_system_prompt_oauth_prepends_claude_code_identity(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1), system="extra instructions"),
            is_oauth_token=True,
        )
        system = params["system"]
        assert isinstance(system, list)
        assert len(system) == 2
        assert "Claude Code" in system[0]["text"]
        assert system[1]["text"] == "extra instructions"

    def test_oauth_always_has_claude_code_identity_even_without_system(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=True,
        )
        system = params["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert "Claude Code" in system[0]["text"]

    def test_temperature_only_set_when_thinking_disabled(self) -> None:
        opts_thinking = AnthropicOptions(temperature=0.5, thinking_enabled=True)
        opts_no_thinking = AnthropicOptions(temperature=0.5, thinking_enabled=False)
        ctx = _context(UserMessage(content="hi", timestamp=1))
        params_thinking = build_params(_model(), ctx, is_oauth_token=False, options=opts_thinking)
        params_no_thinking = build_params(_model(), ctx, is_oauth_token=False, options=opts_no_thinking)
        assert "temperature" not in params_thinking
        assert params_no_thinking["temperature"] == 0.5

    def test_thinking_disabled_for_reasoning_model(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=False),
        )
        assert params["thinking"] == {"type": "disabled"}
        assert "output_config" not in params

    def test_thinking_disabled_for_adaptive_model(self) -> None:
        params = build_params(
            _model(model_id="claude-opus-4-6"),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=False),
        )
        assert params["thinking"] == {"type": "disabled"}
        assert "output_config" not in params

    def test_thinking_adaptive_with_effort(self) -> None:
        params = build_params(
            _model(model_id="claude-opus-4-6"),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=True, effort="max"),
        )
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "max"}

    def test_thinking_budget_based_for_older_model(self) -> None:
        params = build_params(
            _model(model_id="claude-sonnet-4-5"),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=True, thinking_budget_tokens=4096),
        )
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_no_thinking_for_non_reasoning_model(self) -> None:
        params = build_params(
            _model(reasoning=False),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=True),
        )
        assert "thinking" not in params

    def test_metadata_user_id_forwarded(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(metadata={"user_id": "user-123", "extra": "ignored"}),
        )
        assert params["metadata"] == {"user_id": "user-123"}

    def test_metadata_non_string_user_id_ignored(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(metadata={"user_id": 42}),
        )
        assert "metadata" not in params

    def test_tool_choice_string(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(tool_choice="any"),
        )
        assert params["tool_choice"] == {"type": "any"}

    def test_tool_choice_object(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(tool_choice={"type": "tool", "name": "bash"}),
        )
        assert params["tool_choice"] == {"type": "tool", "name": "bash"}

    def test_tools_converted(self) -> None:
        tools = [Tool(name="bash", description="run", parameters={"type": "object", "properties": {}})]
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1), tools=tools),
            is_oauth_token=False,
        )
        assert len(params["tools"]) == 1
        assert params["tools"][0]["name"] == "bash"

    def test_cache_control_added_to_last_user_message(self) -> None:
        params = build_params(
            _model(),
            _context(UserMessage(content="hi", timestamp=1)),
            is_oauth_token=False,
        )
        last = params["messages"][-1]
        # String content is converted to a text block with cache_control.
        assert isinstance(last["content"], list)
        assert last["content"][-1].get("cache_control") == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Port of anthropic-thinking-disable.test.ts (payload-capture portion)
# ---------------------------------------------------------------------------


class TestThinkingDisablePayload:
    """Payload-level port of ``anthropic-thinking-disable.test.ts``.

    The upstream test invokes ``streamSimple`` with a captured ``onPayload``
    hook. Our equivalent hits :func:`build_params` directly — the same
    payload the hook would see, with no live endpoint required.
    """

    def test_budget_based_model_thinking_disabled(self) -> None:
        params = build_params(
            _model(model_id="claude-sonnet-4-5"),
            _context(UserMessage(content="Hello", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=False),
        )
        assert params["thinking"] == {"type": "disabled"}
        assert "output_config" not in params

    def test_adaptive_model_thinking_disabled(self) -> None:
        params = build_params(
            _model(model_id="claude-opus-4-6"),
            _context(UserMessage(content="Hello", timestamp=1)),
            is_oauth_token=False,
            options=AnthropicOptions(thinking_enabled=False),
        )
        assert params["thinking"] == {"type": "disabled"}
        assert "output_config" not in params
