"""Tests for nu_ai.providers.google pure transforms.

Covers the port of ``packages/ai/src/providers/google.ts`` and
``google-shared.ts``: message conversion, tool conversion, thinking-level
mapping, signature handling, stop-reason mapping, and params building.
"""

from __future__ import annotations

import pytest
from nu_ai.providers.google import (
    convert_messages,
    convert_tools,
    get_google_budget,
    get_thinking_level,
    is_gemini_3_flash_model,
    is_gemini_3_pro_model,
    is_gemma_4_model,
    is_thinking_part,
    map_stop_reason,
    map_tool_choice,
    requires_tool_call_id,
    retain_thought_signature,
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
# Fixtures
# ---------------------------------------------------------------------------


def _model(
    *,
    model_id: str = "gemini-2.5-flash",
    provider: str = "google",
    reasoning: bool = True,
    inputs: list[str] | None = None,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="google-generative-ai",
        provider=provider,
        base_url="",
        reasoning=reasoning,
        input=inputs or ["text", "image"],  # type: ignore[arg-type]
        cost=ModelCost(input=0.3, output=2.5, cache_read=0.075, cache_write=0),
        context_window=1_048_576,
        max_tokens=8192,
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


def _assistant(
    content: list[object],
    *,
    model: str = "gemini-2.5-flash",
    provider: str = "google",
) -> AssistantMessage:
    return AssistantMessage(
        content=content,  # type: ignore[arg-type]
        api="google-generative-ai",
        provider=provider,
        model=model,
        usage=_usage(),
        stop_reason="stop",
        timestamp=1,
    )


# ---------------------------------------------------------------------------
# is_thinking_part / retain_thought_signature
# ---------------------------------------------------------------------------


class TestIsThinkingPart:
    def test_thought_true(self) -> None:
        assert is_thinking_part({"thought": True}) is True

    def test_thought_false(self) -> None:
        assert is_thinking_part({"thought": False}) is False

    def test_signature_without_thought_is_not_thinking(self) -> None:
        # Upstream protocol: thoughtSignature can appear on any part type and
        # does NOT imply thinking content.
        assert is_thinking_part({"thoughtSignature": "sig"}) is False


class TestRetainThoughtSignature:
    def test_incoming_overrides(self) -> None:
        assert retain_thought_signature("old", "new") == "new"

    def test_empty_incoming_keeps_existing(self) -> None:
        assert retain_thought_signature("old", "") == "old"

    def test_none_incoming_keeps_existing(self) -> None:
        assert retain_thought_signature("old", None) == "old"

    def test_no_existing_and_no_incoming(self) -> None:
        assert retain_thought_signature(None, None) is None


# ---------------------------------------------------------------------------
# requires_tool_call_id
# ---------------------------------------------------------------------------


class TestRequiresToolCallId:
    @pytest.mark.parametrize("model_id", ["claude-sonnet-4-5", "gpt-oss-120b"])
    def test_claude_and_gpt_oss_require(self, model_id: str) -> None:
        assert requires_tool_call_id(model_id) is True

    def test_gemini_does_not_require(self) -> None:
        assert requires_tool_call_id("gemini-2.5-pro") is False


# ---------------------------------------------------------------------------
# map_stop_reason / map_tool_choice
# ---------------------------------------------------------------------------


class TestMapStopReason:
    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "error"),
            ("RECITATION", "error"),
            ("BLOCKLIST", "error"),
            ("PROHIBITED_CONTENT", "error"),
            ("SPII", "error"),
            ("OTHER", "error"),
        ],
    )
    def test_known_reasons(self, reason: str, expected: str) -> None:
        assert map_stop_reason(reason) == expected


class TestMapToolChoice:
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [("auto", "AUTO"), ("none", "NONE"), ("any", "ANY"), ("weird", "AUTO")],
    )
    def test_maps(self, choice: str, expected: str) -> None:
        assert map_tool_choice(choice) == expected


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------


class TestModelFamilyDetection:
    @pytest.mark.parametrize(
        "model_id",
        ["gemini-3-pro", "gemini-3.0-pro", "gemini-3.1-pro"],
    )
    def test_gemini_3_pro(self, model_id: str) -> None:
        assert is_gemini_3_pro_model(_model(model_id=model_id)) is True

    @pytest.mark.parametrize("model_id", ["gemini-3-flash", "gemini-3.1-flash"])
    def test_gemini_3_flash(self, model_id: str) -> None:
        assert is_gemini_3_flash_model(_model(model_id=model_id)) is True

    def test_gemini_2_5_not_gemini_3(self) -> None:
        assert is_gemini_3_pro_model(_model(model_id="gemini-2.5-pro")) is False
        assert is_gemini_3_flash_model(_model(model_id="gemini-2.5-flash")) is False

    @pytest.mark.parametrize("model_id", ["gemma-4", "gemma4", "gemma-4-2b"])
    def test_gemma_4(self, model_id: str) -> None:
        assert is_gemma_4_model(_model(model_id=model_id)) is True


# ---------------------------------------------------------------------------
# get_thinking_level / get_google_budget
# ---------------------------------------------------------------------------


class TestGetThinkingLevel:
    @pytest.mark.parametrize(
        ("effort", "expected"),
        [("minimal", "MINIMAL"), ("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH")],
    )
    def test_default_mapping(self, effort: str, expected: str) -> None:
        model = _model(model_id="gemini-2.5-pro")
        assert get_thinking_level(effort, model) == expected  # type: ignore[arg-type]

    def test_gemini_3_pro_collapses_to_low_high(self) -> None:
        model = _model(model_id="gemini-3-pro")
        assert get_thinking_level("minimal", model) == "LOW"
        assert get_thinking_level("low", model) == "LOW"
        assert get_thinking_level("medium", model) == "HIGH"
        assert get_thinking_level("high", model) == "HIGH"

    def test_gemma_4_mapping(self) -> None:
        model = _model(model_id="gemma-4-2b")
        assert get_thinking_level("minimal", model) == "MINIMAL"
        assert get_thinking_level("low", model) == "MINIMAL"
        assert get_thinking_level("medium", model) == "HIGH"
        assert get_thinking_level("high", model) == "HIGH"


class TestGetGoogleBudget:
    def test_custom_budget_overrides(self) -> None:
        model = _model(model_id="gemini-2.5-pro")
        budgets = ThinkingBudgets(high=9999)
        assert get_google_budget(model, "high", budgets) == 9999

    def test_gemini_2_5_pro_default_budgets(self) -> None:
        model = _model(model_id="gemini-2.5-pro")
        assert get_google_budget(model, "low") == 2048
        assert get_google_budget(model, "high") == 32768

    def test_gemini_2_5_flash_lite_budgets(self) -> None:
        model = _model(model_id="gemini-2.5-flash-lite")
        assert get_google_budget(model, "minimal") == 512
        assert get_google_budget(model, "high") == 24576

    def test_gemini_2_5_flash_budgets(self) -> None:
        model = _model(model_id="gemini-2.5-flash")
        assert get_google_budget(model, "minimal") == 128
        assert get_google_budget(model, "high") == 24576

    def test_unknown_model_returns_dynamic_sentinel(self) -> None:
        model = _model(model_id="unknown-model")
        assert get_google_budget(model, "high") == -1


# ---------------------------------------------------------------------------
# convert_tools
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_empty_list_returns_none(self) -> None:
        assert convert_tools([]) is None

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
        result = convert_tools(tools)
        assert result is not None
        assert len(result) == 1
        decls = result[0]["function_declarations"]
        assert decls[0]["name"] == "bash"
        assert decls[0]["description"] == "run"
        assert "parameters_json_schema" in decls[0]

    def test_use_parameters_field(self) -> None:
        # For Cloud Code Assist with Claude, legacy ``parameters`` field.
        tools = [Tool(name="t", description="", parameters={"type": "object"})]
        result = convert_tools(tools, use_parameters=True)
        assert result is not None
        assert "parameters" in result[0]["function_declarations"][0]
        assert "parameters_json_schema" not in result[0]["function_declarations"][0]


# ---------------------------------------------------------------------------
# convert_messages
# ---------------------------------------------------------------------------


class TestConvertMessages:
    def test_user_string_content(self) -> None:
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1)])
        contents = convert_messages(_model(), ctx)
        assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]

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
        contents = convert_messages(_model(), ctx)
        parts = contents[0]["parts"]
        assert parts[0] == {"text": "look"}
        assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": "aGVsbG8="}}

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
        contents = convert_messages(model, ctx)
        parts = contents[0]["parts"]
        assert all("inline_data" not in p for p in parts)

    def test_assistant_text(self) -> None:
        ctx = Context(messages=[_assistant(content=[TextContent(text="hi")])])
        contents = convert_messages(_model(), ctx)
        assert contents[0]["role"] == "model"
        assert contents[0]["parts"][0] == {"text": "hi"}

    def test_assistant_thinking_same_model(self) -> None:
        ctx = Context(
            messages=[
                _assistant(
                    content=[
                        ThinkingContent(thinking="reasoning", thinking_signature="YWJjZA=="),
                        TextContent(text="answer"),
                    ],
                )
            ]
        )
        contents = convert_messages(_model(), ctx)
        parts = contents[0]["parts"]
        thinking_part = next(p for p in parts if p.get("thought"))
        assert thinking_part["thought"] is True
        assert thinking_part["text"] == "reasoning"
        assert thinking_part.get("thought_signature") == "YWJjZA=="

    def test_assistant_thinking_cross_model_becomes_plain_text(self) -> None:
        ctx = Context(
            messages=[
                _assistant(
                    content=[ThinkingContent(thinking="reasoning")],
                    model="other-model",
                    provider="other",
                )
            ]
        )
        contents = convert_messages(_model(), ctx)
        parts = contents[0]["parts"]
        assert all("thought" not in p for p in parts)
        assert any(p.get("text") == "reasoning" for p in parts)

    def test_assistant_tool_call(self) -> None:
        ctx = Context(
            messages=[
                _assistant(
                    content=[ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})],
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
        contents = convert_messages(_model(), ctx)
        # [assistant (function call), user (function response)]
        assert contents[0]["role"] == "model"
        assert contents[0]["parts"][0]["function_call"]["name"] == "bash"
        assert contents[1]["role"] == "user"
        fr = contents[1]["parts"][0]["function_response"]
        assert fr["name"] == "bash"
        assert fr["response"]["output"] == "ok"

    def test_tool_result_error_uses_error_key(self) -> None:
        ctx = Context(
            messages=[
                _assistant(
                    content=[ToolCall(id="c1", name="bash", arguments={})],
                ),
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="bash",
                    content=[TextContent(text="failed")],
                    is_error=True,
                    timestamp=1,
                ),
            ],
        )
        contents = convert_messages(_model(), ctx)
        fr = contents[1]["parts"][0]["function_response"]
        assert fr["response"] == {"error": "failed"}

    def test_claude_model_includes_tool_call_id(self) -> None:
        model = _model(model_id="claude-sonnet-4-5")
        ctx = Context(
            messages=[
                _assistant(
                    content=[ToolCall(id="call_1", name="bash", arguments={})],
                    model="claude-sonnet-4-5",
                    provider="google",
                ),
            ],
        )
        contents = convert_messages(model, ctx)
        fc = contents[0]["parts"][0]["function_call"]
        assert fc["id"] == "call_1"

    def test_consecutive_tool_results_merged_into_single_user_turn(self) -> None:
        ctx = Context(
            messages=[
                _assistant(
                    content=[
                        ToolCall(id="a", name="bash", arguments={}),
                        ToolCall(id="b", name="bash", arguments={}),
                    ],
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
            ],
        )
        contents = convert_messages(_model(), ctx)
        # Two tool results collapse into one user turn with two parts.
        user_turns = [c for c in contents if c["role"] == "user"]
        assert len(user_turns) == 1
        assert len(user_turns[0]["parts"]) == 2
