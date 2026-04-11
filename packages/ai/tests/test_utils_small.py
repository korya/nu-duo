"""Tests for the small pi_ai utility modules.

Covers ports of:

* ``packages/ai/src/utils/json-parse.ts``      → pi_ai.utils.json_parse
* ``packages/ai/src/utils/sanitize-unicode.ts`` → pi_ai.utils.sanitize_unicode
* ``packages/ai/src/utils/hash.ts``             → pi_ai.utils.hash
* ``packages/ai/src/utils/overflow.ts``         → pi_ai.utils.overflow
* ``packages/ai/src/utils/validation.ts``       → pi_ai.utils.validation

Upstream has no dedicated test files for these modules, so the tests cover
the documented contracts and known-good behaviour from usage elsewhere.
"""

from __future__ import annotations

import pytest
from pi_ai.types import (
    AssistantMessage,
    Cost,
    TextContent,
    Tool,
    ToolCall,
    Usage,
)
from pi_ai.utils.hash import short_hash
from pi_ai.utils.json_parse import parse_streaming_json
from pi_ai.utils.overflow import get_overflow_patterns, is_context_overflow
from pi_ai.utils.sanitize_unicode import sanitize_surrogates
from pi_ai.utils.validation import validate_tool_arguments, validate_tool_call

# ---------------------------------------------------------------------------
# json_parse
# ---------------------------------------------------------------------------


class TestParseStreamingJson:
    def test_complete_object_parses(self) -> None:
        assert parse_streaming_json('{"a": 1}') == {"a": 1}

    def test_complete_array_parses(self) -> None:
        assert parse_streaming_json("[1, 2, 3]") == [1, 2, 3]

    def test_empty_string_returns_empty_dict(self) -> None:
        assert parse_streaming_json("") == {}

    def test_whitespace_only_returns_empty_dict(self) -> None:
        assert parse_streaming_json("   \n  ") == {}

    def test_none_returns_empty_dict(self) -> None:
        assert parse_streaming_json(None) == {}

    def test_partial_object_tail_trimmed(self) -> None:
        # Partial JSON should parse up to the last complete key/value.
        result = parse_streaming_json('{"cmd": "ls", "flags":')
        assert isinstance(result, dict)
        assert result["cmd"] == "ls"

    def test_partial_object_with_string_still_truncated(self) -> None:
        result = parse_streaming_json('{"a": "hello wor')
        # Implementation is permitted to return {"a": "hello wor"} or {}.
        assert isinstance(result, dict)

    def test_invalid_json_returns_empty_dict(self) -> None:
        assert parse_streaming_json("not json at all") == {}


# ---------------------------------------------------------------------------
# sanitize_unicode
# ---------------------------------------------------------------------------


class TestSanitizeSurrogates:
    def test_plain_ascii_unchanged(self) -> None:
        assert sanitize_surrogates("hello") == "hello"

    def test_valid_emoji_unchanged(self) -> None:
        # Emoji outside BMP is represented as a single code point in Python,
        # not as a surrogate pair — it must pass through untouched.
        assert sanitize_surrogates("Hello 🙈 World") == "Hello 🙈 World"

    def test_unpaired_high_surrogate_removed(self) -> None:
        unpaired = "\ud83d"  # high surrogate with no following low
        assert sanitize_surrogates(f"Text {unpaired} here") == "Text  here"

    def test_unpaired_low_surrogate_removed(self) -> None:
        unpaired = "\udc00"  # low surrogate with no preceding high
        assert sanitize_surrogates(f"a{unpaired}b") == "ab"

    def test_paired_surrogates_preserved(self) -> None:
        # \ud83d\ude00 is the UTF-16 encoding of 😀 — in Python source it's a
        # pair of isolated code points, but sanitize_surrogates must treat them
        # as a valid pair and leave them.
        paired = "\ud83d\ude00"
        assert sanitize_surrogates(f"a{paired}b") == f"a{paired}b"

    def test_mixed_pair_and_orphan(self) -> None:
        paired = "\ud83d\ude00"  # 😀
        orphan = "\ud83d"
        result = sanitize_surrogates(f"{paired}{orphan}end")
        assert result == f"{paired}end"


# ---------------------------------------------------------------------------
# hash
# ---------------------------------------------------------------------------


class TestShortHash:
    def test_is_deterministic(self) -> None:
        assert short_hash("hello") == short_hash("hello")

    def test_different_inputs_different_hashes(self) -> None:
        # Not a hard cryptographic guarantee, just a smoke test.
        assert short_hash("abc") != short_hash("abd")

    def test_empty_string(self) -> None:
        # Should not crash on empty input and should return a string.
        result = short_hash("")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unicode_safe(self) -> None:
        assert isinstance(short_hash("héllo 🙈"), str)

    def test_output_is_base36(self) -> None:
        result = short_hash("abcdefghij")
        assert all(c in "0123456789abcdefghijklmnopqrstuvwxyz" for c in result)


# ---------------------------------------------------------------------------
# overflow
# ---------------------------------------------------------------------------


def _err(error_message: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude",
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="error",
        error_message=error_message,
        timestamp=1,
    )


def _ok(input_tokens: int = 0, cache_read: int = 0) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude",
        usage=Usage(
            input=input_tokens,
            output=0,
            cache_read=cache_read,
            cache_write=0,
            total_tokens=input_tokens + cache_read,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="stop",
        timestamp=1,
    )


class TestOverflowDetection:
    @pytest.mark.parametrize(
        "error_message",
        [
            "prompt is too long: 213462 tokens > 200000 maximum",
            '413 {"error":{"type":"request_too_large","message":"Request exceeds the maximum size"}}',
            "Your input exceeds the context window of this model",
            "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
            "This model's maximum prompt length is 131072 but the request contains 537812 tokens",
            "Please reduce the length of the messages or completion",
            "This endpoint's maximum context length is 32000 tokens. However, you requested about 50000 tokens",
            "the request exceeds the available context size, try increasing it",
            "tokens to keep from the initial prompt is greater than the context length",
            "prompt token count of 50000 exceeds the limit of 32000",
            "invalid params, context window exceeds limit",
            "Your request exceeded model token limit: 100000 (requested: 150000)",
            "Prompt contains 40000 tokens ... too large for model with 32000 maximum context length",
            "prompt too long; exceeded max context length by 5000 tokens",
            "413 status code (no body)",
        ],
    )
    def test_known_overflow_messages_detected(self, error_message: str) -> None:
        assert is_context_overflow(_err(error_message)) is True

    @pytest.mark.parametrize(
        "error_message",
        [
            "Throttling error: Too many tokens, please wait",
            "Service unavailable: Too many tokens",
            "rate limit exceeded",
            "Too many requests",
        ],
    )
    def test_non_overflow_messages_excluded(self, error_message: str) -> None:
        assert is_context_overflow(_err(error_message)) is False

    def test_successful_message_without_context_window_is_not_overflow(self) -> None:
        assert is_context_overflow(_ok(input_tokens=999_999_999)) is False

    def test_silent_overflow_detected_when_context_window_provided(self) -> None:
        msg = _ok(input_tokens=300_000, cache_read=50_000)
        assert is_context_overflow(msg, context_window=200_000) is True

    def test_silent_overflow_with_input_plus_cache_under_limit(self) -> None:
        msg = _ok(input_tokens=100_000, cache_read=50_000)
        assert is_context_overflow(msg, context_window=200_000) is False

    def test_get_overflow_patterns_returns_copy(self) -> None:
        patterns = get_overflow_patterns()
        assert len(patterns) > 0
        # Mutating the returned list must not affect internal state.
        original = len(get_overflow_patterns())
        patterns.clear()
        assert len(get_overflow_patterns()) == original


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _tool(schema: dict[str, object]) -> Tool:
    return Tool(name="bash", description="run", parameters=schema)


class TestValidateToolCall:
    def test_unknown_tool_raises(self) -> None:
        tools: list[Tool] = [_tool({"type": "object", "properties": {}})]
        tc = ToolCall(id="c", name="missing", arguments={})
        with pytest.raises(ValueError, match='Tool "missing" not found'):
            validate_tool_call(tools, tc)

    def test_valid_arguments_returned(self) -> None:
        tools = [
            _tool(
                {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                }
            )
        ]
        tc = ToolCall(id="c", name="bash", arguments={"cmd": "ls"})
        result = validate_tool_call(tools, tc)
        assert result == {"cmd": "ls"}

    def test_missing_required_raises_formatted(self) -> None:
        tool = _tool(
            {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            }
        )
        tc = ToolCall(id="c", name="bash", arguments={})
        with pytest.raises(ValueError, match='Validation failed for tool "bash"'):
            validate_tool_arguments(tool, tc)

    def test_type_coercion_converts_numeric_string(self) -> None:
        tool = _tool(
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }
        )
        tc = ToolCall(id="c", name="bash", arguments={"count": "42"})
        result = validate_tool_arguments(tool, tc)
        assert result == {"count": 42}

    def test_wrong_type_with_no_coercion_raises(self) -> None:
        tool = _tool(
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }
        )
        tc = ToolCall(id="c", name="bash", arguments={"count": "not a number"})
        with pytest.raises(ValueError, match="count"):
            validate_tool_arguments(tool, tc)
