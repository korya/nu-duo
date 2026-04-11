"""Tests for nu_ai.providers.transform_messages.

Ports ``packages/ai/test/transform-messages-copilot-openai-to-anthropic.test.ts``
verbatim (two cases) and adds coverage for every branch of the two-pass
algorithm: cross-model thinking conversion, redacted thinking, thought
signature stripping, tool call id normalization, orphaned tool calls, user
message interrupts, and errored/aborted assistant skipping.
"""

from __future__ import annotations

import re

import pytest
from nu_ai.providers.transform_messages import transform_messages
from nu_ai.types import (
    AssistantMessage,
    Cost,
    ImageContent,
    Message,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
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


def _copilot_claude_model() -> Model:
    return Model(
        id="claude-sonnet-4",
        name="Claude Sonnet 4",
        api="anthropic-messages",
        provider="github-copilot",
        base_url="https://api.individual.githubcopilot.com",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=16000,
    )


def _anthropic_normalize_tool_call_id(
    id_: str,
    _model: Model,
    _source: AssistantMessage,
) -> str:
    """Match the regex normalization used by anthropic.ts."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", id_)[:64]


def _assistant(
    content: list[object],
    *,
    api: str = "openai-completions",
    provider: str = "github-copilot",
    model: str = "gpt-4o",
    stop_reason: str = "stop",
    error_message: str | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=content,  # type: ignore[arg-type]
        api=api,
        provider=provider,
        model=model,
        usage=_usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        error_message=error_message,
        timestamp=1,
    )


def _user(text: str = "hello") -> UserMessage:
    return UserMessage(content=text, timestamp=1)


def _tool_result(tool_call_id: str, text: str = "ok") -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="bash",
        content=[TextContent(text=text)],
        is_error=False,
        timestamp=1,
    )


# ---------------------------------------------------------------------------
# Upstream ports — transform-messages-copilot-openai-to-anthropic.test.ts
# ---------------------------------------------------------------------------


class TestCopilotOpenAIToAnthropicMigration:
    """Port of ``describe("OpenAI to Anthropic session migration for Copilot Claude")``."""

    def test_converts_thinking_blocks_to_plain_text_when_source_model_differs(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _user("hello"),
            _assistant(
                content=[
                    ThinkingContent(
                        thinking="Let me think about this...",
                        thinking_signature="reasoning_content",
                    ),
                    TextContent(text="Hi there!"),
                ],
            ),
        ]

        result = transform_messages(messages, model, _anthropic_normalize_tool_call_id)

        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)

        text_blocks = [b for b in assistant_msg.content if isinstance(b, TextContent)]
        thinking_blocks = [b for b in assistant_msg.content if isinstance(b, ThinkingContent)]
        assert len(thinking_blocks) == 0
        assert len(text_blocks) >= 2

    def test_removes_thought_signature_from_tool_calls_when_migrating_between_models(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _user("run a command"),
            _assistant(
                content=[
                    ToolCall(
                        id="call_123",
                        name="bash",
                        arguments={"command": "ls"},
                        thought_signature='{"type":"reasoning.encrypted","id":"call_123","data":"encrypted"}',
                    ),
                ],
                api="openai-responses",
                provider="github-copilot",
                model="gpt-5",
                stop_reason="toolUse",
            ),
            _tool_result("call_123", "output"),
        ]

        result = transform_messages(messages, model, _anthropic_normalize_tool_call_id)
        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        tool_call = next(b for b in assistant_msg.content if isinstance(b, ToolCall))
        assert tool_call.thought_signature is None


# ---------------------------------------------------------------------------
# Same-model pass-through
# ---------------------------------------------------------------------------


class TestSameModelPassThrough:
    def test_user_messages_unchanged(self) -> None:
        model = _copilot_claude_model()
        user = _user("x")
        result = transform_messages([user], model)
        assert result == [user]

    def test_same_model_preserves_thinking(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _user(),
            _assistant(
                content=[
                    ThinkingContent(thinking="inner"),
                    TextContent(text="reply"),
                ],
                api="anthropic-messages",
                provider="github-copilot",
                model="claude-sonnet-4",
            ),
        ]
        result = transform_messages(messages, model)
        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        assert any(isinstance(b, ThinkingContent) for b in assistant_msg.content)

    def test_same_model_keeps_thinking_with_signature_even_when_empty(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[
                    ThinkingContent(thinking="", thinking_signature="sig"),
                    TextContent(text="hi"),
                ],
                api="anthropic-messages",
                provider="github-copilot",
                model="claude-sonnet-4",
            ),
        ]
        result = transform_messages(messages, model)
        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        thinking = next(b for b in assistant_msg.content if isinstance(b, ThinkingContent))
        assert thinking.thinking_signature == "sig"


# ---------------------------------------------------------------------------
# Cross-model thinking handling
# ---------------------------------------------------------------------------


class TestCrossModelThinking:
    def test_empty_thinking_is_dropped(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[
                    ThinkingContent(thinking="   "),
                    TextContent(text="reply"),
                ],
            ),
        ]
        result = transform_messages(messages, model)
        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        # Only the text block remains.
        assert len(assistant_msg.content) == 1
        assert isinstance(assistant_msg.content[0], TextContent)

    def test_redacted_thinking_dropped_for_cross_model(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[
                    ThinkingContent(thinking="", thinking_signature="enc", redacted=True),
                    TextContent(text="reply"),
                ],
            ),
        ]
        result = transform_messages(messages, model)
        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        assert all(not isinstance(b, ThinkingContent) for b in assistant_msg.content)

    def test_redacted_thinking_kept_for_same_model(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[ThinkingContent(thinking="", thinking_signature="enc", redacted=True)],
                api="anthropic-messages",
                provider="github-copilot",
                model="claude-sonnet-4",
            ),
        ]
        result = transform_messages(messages, model)
        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        assert any(isinstance(b, ThinkingContent) for b in assistant_msg.content)


# ---------------------------------------------------------------------------
# Tool call ID normalization
# ---------------------------------------------------------------------------


class TestToolCallIdNormalization:
    def test_normalized_id_is_propagated_to_tool_result(self) -> None:
        model = _copilot_claude_model()
        raw_id = "tool_call|open:ai:weird$id"
        messages: list[Message] = [
            _assistant(
                content=[ToolCall(id=raw_id, name="bash", arguments={})],
                stop_reason="toolUse",
            ),
            _tool_result(raw_id),
        ]
        result = transform_messages(messages, model, _anthropic_normalize_tool_call_id)

        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        tool_call = next(b for b in assistant_msg.content if isinstance(b, ToolCall))
        normalized = _anthropic_normalize_tool_call_id(raw_id, model, assistant_msg)
        assert tool_call.id == normalized

        tool_result = next(m for m in result if m.role == "toolResult")
        assert isinstance(tool_result, ToolResultMessage)
        assert tool_result.tool_call_id == normalized

    def test_same_model_does_not_normalize_tool_call_ids(self) -> None:
        model = _copilot_claude_model()
        raw_id = "tool_call|weird"
        messages: list[Message] = [
            _assistant(
                content=[ToolCall(id=raw_id, name="bash", arguments={})],
                api="anthropic-messages",
                provider="github-copilot",
                model="claude-sonnet-4",
                stop_reason="toolUse",
            ),
        ]
        result = transform_messages(messages, model, _anthropic_normalize_tool_call_id)
        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        tool_call = next(b for b in assistant_msg.content if isinstance(b, ToolCall))
        assert tool_call.id == raw_id


# ---------------------------------------------------------------------------
# Orphaned tool calls
# ---------------------------------------------------------------------------


class TestOrphanedToolCalls:
    def test_orphaned_tool_call_before_user_inserts_synthetic_result(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[ToolCall(id="call_1", name="bash", arguments={})],
                stop_reason="toolUse",
            ),
            _user("follow up"),
        ]
        result = transform_messages(messages, model)
        assert [m.role for m in result] == ["assistant", "toolResult", "user"]
        synthetic = result[1]
        assert isinstance(synthetic, ToolResultMessage)
        assert synthetic.is_error is True
        assert synthetic.tool_call_id == "call_1"
        assert synthetic.content[0].type == "text"
        assert isinstance(synthetic.content[0], TextContent)
        assert synthetic.content[0].text == "No result provided"

    def test_orphaned_tool_call_before_next_assistant_inserts_synthetic_result(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[ToolCall(id="call_1", name="bash", arguments={})],
                stop_reason="toolUse",
            ),
            _assistant(content=[TextContent(text="continuing")]),
        ]
        result = transform_messages(messages, model)
        assert [m.role for m in result] == ["assistant", "toolResult", "assistant"]

    def test_tool_call_with_matching_result_no_synthetic_inserted(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[ToolCall(id="call_1", name="bash", arguments={})],
                stop_reason="toolUse",
            ),
            _tool_result("call_1"),
            _user("next"),
        ]
        result = transform_messages(messages, model)
        assert [m.role for m in result] == ["assistant", "toolResult", "user"]
        assert len([m for m in result if m.role == "toolResult"]) == 1


# ---------------------------------------------------------------------------
# Errored / aborted assistant messages
# ---------------------------------------------------------------------------


class TestErroredAssistantSkipping:
    def test_errored_assistant_is_skipped(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _user(),
            _assistant(content=[TextContent(text="partial")], stop_reason="error", error_message="boom"),
            _assistant(content=[TextContent(text="recovered")]),
        ]
        result = transform_messages(messages, model)
        assert [m.role for m in result] == ["user", "assistant"]
        final = next(m for m in result if m.role == "assistant")
        assert isinstance(final, AssistantMessage)
        assert isinstance(final.content[0], TextContent)
        assert final.content[0].text == "recovered"

    def test_aborted_assistant_is_skipped(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(content=[TextContent(text="partial")], stop_reason="aborted"),
            _assistant(content=[TextContent(text="ok")]),
        ]
        result = transform_messages(messages, model)
        assert len([m for m in result if m.role == "assistant"]) == 1


# ---------------------------------------------------------------------------
# Cross-model text and image blocks
# ---------------------------------------------------------------------------


class TestCrossModelContentPreservation:
    def test_cross_model_text_block_converted_to_fresh_text(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [_assistant(content=[TextContent(text="hi", text_signature="sig")])]
        result = transform_messages(messages, model)
        assistant_msg = next(m for m in result if m.role == "assistant")
        assert isinstance(assistant_msg, AssistantMessage)
        text = assistant_msg.content[0]
        assert isinstance(text, TextContent)
        assert text.text == "hi"
        # Cross-model drops the signature (fresh TextContent).
        assert text.text_signature is None

    def test_tool_result_with_image_passes_through(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            ToolResultMessage(
                tool_call_id="c",
                tool_name="t",
                content=[ImageContent(data="d", mime_type="image/png")],
                is_error=False,
                timestamp=1,
            ),
        ]
        result = transform_messages(messages, model)
        assert len(result) == 1
        assert isinstance(result[0], ToolResultMessage)


# ---------------------------------------------------------------------------
# Interaction: orphaned tool calls across multiple assistants
# ---------------------------------------------------------------------------


class TestMultipleAssistants:
    def test_two_assistants_each_with_orphaned_tool_calls(self) -> None:
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[ToolCall(id="a", name="bash", arguments={})],
                stop_reason="toolUse",
            ),
            _assistant(
                content=[ToolCall(id="b", name="bash", arguments={})],
                stop_reason="toolUse",
            ),
            _user("done"),
        ]
        result = transform_messages(messages, model)
        # a → synthetic, then second assistant, b → synthetic, then user.
        assert [m.role for m in result] == [
            "assistant",
            "toolResult",
            "assistant",
            "toolResult",
            "user",
        ]
        synth_ids = [m.tool_call_id for m in result if isinstance(m, ToolResultMessage)]
        assert synth_ids == ["a", "b"]


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_message_list(self) -> None:
        model = _copilot_claude_model()
        assert transform_messages([], model) == []

    @pytest.mark.parametrize("stop_reason", ["error", "aborted"])
    def test_errored_assistant_removes_orphan_tracking(self, stop_reason: str) -> None:
        # An errored assistant message must not leak orphan tool calls to
        # the next message: the whole turn is discarded.
        model = _copilot_claude_model()
        messages: list[Message] = [
            _assistant(
                content=[ToolCall(id="c", name="bash", arguments={})],
                stop_reason=stop_reason,
            ),
            _user("next"),
        ]
        result = transform_messages(messages, model)
        assert [m.role for m in result] == ["user"]
