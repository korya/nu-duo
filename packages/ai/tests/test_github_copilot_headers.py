"""Tests for nu_ai.providers.github_copilot_headers.

Port of the documented contract in
``packages/ai/src/providers/github-copilot-headers.ts``.
"""

from __future__ import annotations

from nu_ai.providers.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
    infer_copilot_initiator,
)
from nu_ai.types import (
    AssistantMessage,
    Cost,
    ImageContent,
    Message,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _assistant(text: str = "ok") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="github-copilot",
        model="m",
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="stop",
        timestamp=1,
    )


class TestInferCopilotInitiator:
    def test_empty_messages_returns_user(self) -> None:
        assert infer_copilot_initiator([]) == "user"

    def test_last_user_returns_user(self) -> None:
        msgs: list[Message] = [UserMessage(content="hi", timestamp=1)]
        assert infer_copilot_initiator(msgs) == "user"

    def test_last_assistant_returns_agent(self) -> None:
        msgs: list[Message] = [
            UserMessage(content="hi", timestamp=1),
            _assistant(),
        ]
        assert infer_copilot_initiator(msgs) == "agent"

    def test_last_tool_result_returns_agent(self) -> None:
        msgs: list[Message] = [
            ToolResultMessage(
                tool_call_id="c",
                tool_name="t",
                content=[TextContent(text="ok")],
                is_error=False,
                timestamp=1,
            ),
        ]
        assert infer_copilot_initiator(msgs) == "agent"


class TestHasCopilotVisionInput:
    def test_no_messages(self) -> None:
        assert has_copilot_vision_input([]) is False

    def test_user_string_content_no_images(self) -> None:
        msgs: list[Message] = [UserMessage(content="plain text", timestamp=1)]
        assert has_copilot_vision_input(msgs) is False

    def test_user_list_content_with_image(self) -> None:
        msgs: list[Message] = [
            UserMessage(
                content=[TextContent(text="look"), ImageContent(data="d", mime_type="image/png")],
                timestamp=1,
            ),
        ]
        assert has_copilot_vision_input(msgs) is True

    def test_user_list_content_only_text(self) -> None:
        msgs: list[Message] = [UserMessage(content=[TextContent(text="hi")], timestamp=1)]
        assert has_copilot_vision_input(msgs) is False

    def test_tool_result_with_image(self) -> None:
        msgs: list[Message] = [
            ToolResultMessage(
                tool_call_id="c",
                tool_name="t",
                content=[ImageContent(data="d", mime_type="image/png")],
                is_error=False,
                timestamp=1,
            ),
        ]
        assert has_copilot_vision_input(msgs) is True

    def test_assistant_with_images_ignored(self) -> None:
        # Assistant messages don't count — Copilot vision header only applies
        # to user-supplied images.
        msgs: list[Message] = [_assistant()]
        assert has_copilot_vision_input(msgs) is False


class TestBuildCopilotDynamicHeaders:
    def test_user_no_images(self) -> None:
        msgs: list[Message] = [UserMessage(content="hi", timestamp=1)]
        headers = build_copilot_dynamic_headers(messages=msgs, has_images=False)
        assert headers == {
            "X-Initiator": "user",
            "Openai-Intent": "conversation-edits",
        }

    def test_agent_with_images(self) -> None:
        msgs: list[Message] = [_assistant()]
        headers = build_copilot_dynamic_headers(messages=msgs, has_images=True)
        assert headers == {
            "X-Initiator": "agent",
            "Openai-Intent": "conversation-edits",
            "Copilot-Vision-Request": "true",
        }
