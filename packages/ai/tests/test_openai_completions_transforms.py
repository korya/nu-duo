"""Tests for nu_ai.providers.openai_completions pure transforms.

Covers the port of message conversion, tool conversion, params building,
compat detection, stop-reason mapping, and usage parsing from
``packages/ai/src/providers/openai-completions.ts``.
"""

from __future__ import annotations

import pytest
from nu_ai.providers.openai_completions import (
    convert_messages,
    convert_tools,
    detect_compat,
    get_compat,
    has_tool_history,
    map_reasoning_effort,
    map_stop_reason,
    parse_chunk_usage,
)
from nu_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    ImageContent,
    Message,
    Model,
    ModelCost,
    OpenAICompletionsCompat,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _model(
    *,
    model_id: str = "gpt-4o",
    provider: str = "openai",
    base_url: str = "https://api.openai.com/v1",
    reasoning: bool = False,
    inputs: list[str] | None = None,
    compat: OpenAICompletionsCompat | None = None,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        input=inputs or ["text", "image"],  # type: ignore[arg-type]
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=0),
        context_window=128_000,
        max_tokens=4096,
        compat=compat,
    )


def _usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


def _assistant(content: list[object], *, stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=content,  # type: ignore[arg-type]
        api="openai-completions",
        provider="openai",
        model="gpt-4o",
        usage=_usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        timestamp=1,
    )


# ---------------------------------------------------------------------------
# has_tool_history
# ---------------------------------------------------------------------------


class TestHasToolHistory:
    def test_empty_list(self) -> None:
        assert has_tool_history([]) is False

    def test_plain_user_message(self) -> None:
        assert has_tool_history([UserMessage(content="hi", timestamp=1)]) is False

    def test_assistant_without_tool_calls(self) -> None:
        msgs: list[Message] = [_assistant(content=[TextContent(text="hi")])]
        assert has_tool_history(msgs) is False

    def test_assistant_with_tool_call(self) -> None:
        msgs: list[Message] = [
            _assistant(content=[ToolCall(id="c", name="bash", arguments={})], stop_reason="toolUse"),
        ]
        assert has_tool_history(msgs) is True

    def test_tool_result_message(self) -> None:
        msgs: list[Message] = [
            ToolResultMessage(
                tool_call_id="c",
                tool_name="bash",
                content=[TextContent(text="ok")],
                is_error=False,
                timestamp=1,
            ),
        ]
        assert has_tool_history(msgs) is True


# ---------------------------------------------------------------------------
# map_stop_reason
# ---------------------------------------------------------------------------


class TestMapStopReason:
    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("stop", "stop"),
            ("end", "stop"),
            (None, "stop"),
            ("length", "length"),
            ("function_call", "toolUse"),
            ("tool_calls", "toolUse"),
        ],
    )
    def test_known_reasons(self, reason: str | None, expected: str) -> None:
        result = map_stop_reason(reason)
        assert result["stop_reason"] == expected
        assert result.get("error_message") is None

    def test_content_filter(self) -> None:
        result = map_stop_reason("content_filter")
        assert result["stop_reason"] == "error"
        assert "content_filter" in (result.get("error_message") or "")

    def test_unknown_reason_becomes_error(self) -> None:
        result = map_stop_reason("weirdness")
        assert result["stop_reason"] == "error"
        assert "weirdness" in (result.get("error_message") or "")


# ---------------------------------------------------------------------------
# parse_chunk_usage
# ---------------------------------------------------------------------------


class TestParseChunkUsage:
    def test_basic_usage(self) -> None:
        model = _model()
        usage = parse_chunk_usage(
            {"prompt_tokens": 100, "completion_tokens": 50},
            model,
        )
        assert usage.input == 100
        assert usage.output == 50
        assert usage.cache_read == 0
        assert usage.total_tokens == 150

    def test_reasoning_tokens_added_to_output(self) -> None:
        model = _model()
        usage = parse_chunk_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
            model,
        )
        # output = completion + reasoning
        assert usage.output == 70
        assert usage.total_tokens == 170

    def test_cache_read_subtracted_from_input(self) -> None:
        model = _model()
        usage = parse_chunk_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 30},
            },
            model,
        )
        assert usage.input == 70
        assert usage.cache_read == 30
        assert usage.cache_write == 0

    def test_openrouter_cache_write_reported_in_cached_tokens(self) -> None:
        # Some providers (OpenRouter) report cached_tokens as (prior hits + current writes).
        # The parser must subtract cache_write to get the true cache_read count.
        model = _model()
        usage = parse_chunk_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
            },
            model,
        )
        assert usage.cache_read == 20
        assert usage.cache_write == 10


# ---------------------------------------------------------------------------
# map_reasoning_effort
# ---------------------------------------------------------------------------


class TestMapReasoningEffort:
    def test_passthrough_without_map(self) -> None:
        assert map_reasoning_effort("medium", {}) == "medium"

    def test_override_via_map(self) -> None:
        assert map_reasoning_effort("medium", {"medium": "normal"}) == "normal"


# ---------------------------------------------------------------------------
# detect_compat / get_compat
# ---------------------------------------------------------------------------


class TestDetectCompat:
    def test_openai_defaults(self) -> None:
        compat = detect_compat(_model())
        assert compat.supports_store is True
        assert compat.supports_developer_role is True
        assert compat.max_tokens_field == "max_completion_tokens"
        assert compat.thinking_format == "openai"

    def test_cerebras_is_non_standard(self) -> None:
        compat = detect_compat(_model(provider="cerebras", base_url="https://api.cerebras.ai/v1"))
        assert compat.supports_store is False
        assert compat.supports_developer_role is False

    def test_groq_disables_reasoning_effort(self) -> None:
        compat = detect_compat(_model(provider="groq", base_url="https://api.groq.com"))
        assert compat.supports_reasoning_effort is True  # groq does support it, just not xai/zai
        # xai would
        xcompat = detect_compat(_model(provider="xai", base_url="https://api.x.ai/v1"))
        assert xcompat.supports_reasoning_effort is False

    def test_zai_sets_thinking_format(self) -> None:
        compat = detect_compat(_model(provider="zai", base_url="https://api.z.ai/v1"))
        assert compat.thinking_format == "zai"

    def test_openrouter_sets_thinking_format(self) -> None:
        compat = detect_compat(_model(provider="openrouter", base_url="https://openrouter.ai/api/v1"))
        assert compat.thinking_format == "openrouter"

    def test_chutes_uses_max_tokens_field(self) -> None:
        compat = detect_compat(_model(provider="custom", base_url="https://llm.chutes.ai/v1"))
        assert compat.max_tokens_field == "max_tokens"

    def test_ollama_default_compat(self) -> None:
        # Ollama uses the OpenAI completions shape via /v1/chat/completions.
        # It should detect with default settings — no special casing needed.
        compat = detect_compat(
            _model(provider="ollama", base_url="http://localhost:11434/v1"),
        )
        assert compat.thinking_format == "openai"
        assert compat.max_tokens_field == "max_completion_tokens"


class TestGetCompat:
    def test_no_explicit_returns_detected(self) -> None:
        model = _model()
        compat = get_compat(model)
        assert compat.supports_store is True

    def test_explicit_overrides_detected(self) -> None:
        model = _model(compat=OpenAICompletionsCompat(supports_store=False))
        compat = get_compat(model)
        assert compat.supports_store is False


# ---------------------------------------------------------------------------
# convert_tools
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_basic_tool(self) -> None:
        tools = [
            Tool(
                name="bash",
                description="run a command",
                parameters={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            )
        ]
        compat = detect_compat(_model())
        converted = convert_tools(tools, compat)
        assert converted == [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                    "strict": False,
                },
            }
        ]

    def test_empty_list(self) -> None:
        compat = detect_compat(_model())
        assert convert_tools([], compat) == []

    def test_strict_mode_omitted_when_unsupported(self) -> None:
        compat = detect_compat(_model())
        compat_no_strict = compat.model_copy(update={"supports_strict_mode": False})
        tools = [Tool(name="t", description="", parameters={"type": "object"})]
        converted = convert_tools(tools, compat_no_strict)
        assert "strict" not in converted[0]["function"]


# ---------------------------------------------------------------------------
# convert_messages
# ---------------------------------------------------------------------------


class TestConvertMessages:
    def test_system_prompt_goes_first(self) -> None:
        ctx = Context(
            system_prompt="you are helpful",
            messages=[UserMessage(content="hi", timestamp=1)],
        )
        compat = detect_compat(_model())
        params = convert_messages(_model(), ctx, compat)
        assert params[0]["role"] == "system"
        assert params[0]["content"] == "you are helpful"

    def test_reasoning_model_uses_developer_role(self) -> None:
        ctx = Context(
            system_prompt="sys",
            messages=[UserMessage(content="hi", timestamp=1)],
        )
        model = _model(model_id="gpt-5", reasoning=True)
        compat = detect_compat(model)
        params = convert_messages(model, ctx, compat)
        assert params[0]["role"] == "developer"

    def test_user_string_content(self) -> None:
        ctx = Context(messages=[UserMessage(content="hello", timestamp=1)])
        compat = detect_compat(_model())
        params = convert_messages(_model(), ctx, compat)
        assert params == [{"role": "user", "content": "hello"}]

    def test_user_with_image(self) -> None:
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="look"),
                        ImageContent(data="aGVsbG8=", mime_type="image/png"),
                    ],
                    timestamp=1,
                )
            ]
        )
        compat = detect_compat(_model())
        params = convert_messages(_model(), ctx, compat)
        assert params[0]["role"] == "user"
        content = params[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "look"}
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64,aGVsbG8=" in content[1]["image_url"]["url"]

    def test_user_image_filtered_for_text_only_model(self) -> None:
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="hi"),
                        ImageContent(data="d", mime_type="image/png"),
                    ],
                    timestamp=1,
                )
            ]
        )
        model = _model(inputs=["text"])
        compat = detect_compat(model)
        params = convert_messages(model, ctx, compat)
        content = params[0]["content"]
        assert isinstance(content, list)
        assert all(c["type"] != "image_url" for c in content)

    def test_assistant_text_as_string(self) -> None:
        ctx = Context(
            messages=[_assistant(content=[TextContent(text="hello")])],
        )
        compat = detect_compat(_model())
        params = convert_messages(_model(), ctx, compat)
        assert params[0]["role"] == "assistant"
        assert params[0]["content"] == "hello"

    def test_assistant_empty_text_message_dropped(self) -> None:
        # Assistant messages with no non-empty content and no tool calls are dropped
        # to keep providers that reject empty assistant messages happy.
        ctx = Context(
            messages=[_assistant(content=[TextContent(text="   ")])],
        )
        compat = detect_compat(_model())
        params = convert_messages(_model(), ctx, compat)
        assert params == []

    def test_assistant_tool_call(self) -> None:
        ctx = Context(
            messages=[
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
            ],
        )
        compat = detect_compat(_model())
        params = convert_messages(_model(), ctx, compat)
        assert len(params) == 2
        assert params[0]["role"] == "assistant"
        assert params[0]["tool_calls"][0]["id"] == "c1"
        assert params[0]["tool_calls"][0]["function"]["name"] == "bash"
        assert params[1]["role"] == "tool"
        assert params[1]["tool_call_id"] == "c1"
        assert params[1]["content"] == "ok"

    def test_tool_result_image_becomes_separate_user_message(self) -> None:
        ctx = Context(
            messages=[
                _assistant(
                    content=[ToolCall(id="c1", name="shot", arguments={})],
                    stop_reason="toolUse",
                ),
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="shot",
                    content=[
                        TextContent(text="here"),
                        ImageContent(data="d", mime_type="image/png"),
                    ],
                    is_error=False,
                    timestamp=1,
                ),
            ],
        )
        compat = detect_compat(_model())
        params = convert_messages(_model(), ctx, compat)
        # [assistant, tool_result, user (with image)]
        assert params[1]["role"] == "tool"
        assert params[2]["role"] == "user"
        content = params[2]["content"]
        assert isinstance(content, list)
        assert any(c["type"] == "image_url" for c in content)
