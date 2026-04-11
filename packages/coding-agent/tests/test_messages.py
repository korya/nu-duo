"""Tests for ``nu_coding_agent.core.messages``."""

from __future__ import annotations

from dataclasses import dataclass

from nu_ai.types import ImageContent, TextContent, UserMessage
from nu_coding_agent.core.messages import (
    BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    bash_execution_to_text,
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)


def _bash(
    *,
    output: str = "stdout",
    exit_code: int | None = 0,
    cancelled: bool = False,
    truncated: bool = False,
    full_output_path: str | None = None,
    exclude: bool = False,
) -> BashExecutionMessage:
    return BashExecutionMessage(
        role="bashExecution",
        command="ls",
        output=output,
        exit_code=exit_code,
        cancelled=cancelled,
        truncated=truncated,
        full_output_path=full_output_path,
        timestamp=1000,
        exclude_from_context=exclude,
    )


def test_bash_execution_to_text_simple() -> None:
    text = bash_execution_to_text(_bash(output="hello"))
    assert "Ran `ls`" in text
    assert "```\nhello\n```" in text


def test_bash_execution_to_text_no_output() -> None:
    text = bash_execution_to_text(_bash(output=""))
    assert "(no output)" in text


def test_bash_execution_to_text_cancelled() -> None:
    text = bash_execution_to_text(_bash(cancelled=True))
    assert "(command cancelled)" in text


def test_bash_execution_to_text_nonzero_exit() -> None:
    text = bash_execution_to_text(_bash(exit_code=2))
    assert "exited with code 2" in text


def test_bash_execution_to_text_truncated() -> None:
    text = bash_execution_to_text(_bash(truncated=True, full_output_path="/tmp/out"))
    assert "/tmp/out" in text


def test_create_branch_summary_message_parses_iso() -> None:
    msg = create_branch_summary_message("abc", "msg-1", "2024-01-01T00:00:00Z")
    assert msg.role == "branchSummary"
    assert msg.summary == "abc"
    assert msg.from_id == "msg-1"
    assert msg.timestamp == 1704067200000


def test_create_compaction_summary_message() -> None:
    msg = create_compaction_summary_message("hi", 1234, "2024-01-01T00:00:00Z")
    assert msg.tokens_before == 1234
    assert msg.timestamp == 1704067200000


def test_create_custom_message() -> None:
    msg = create_custom_message("note", "hello", display=True, details=None, timestamp="2024-01-01T00:00:00Z")
    assert msg.role == "custom"
    assert msg.custom_type == "note"
    assert msg.display is True


def test_convert_to_llm_passthrough_user_assistant() -> None:
    user = UserMessage(content="hi", timestamp=1)
    out = convert_to_llm([user])
    assert out == [user]


def test_convert_to_llm_skips_excluded_bash() -> None:
    msg = _bash(exclude=True)
    assert convert_to_llm([msg]) == []


def test_convert_to_llm_bash_to_user() -> None:
    msg = _bash(output="data", exit_code=0)
    out = convert_to_llm([msg])
    assert len(out) == 1
    assert isinstance(out[0], UserMessage)
    assert isinstance(out[0].content, list)
    assert isinstance(out[0].content[0], TextContent)
    assert "Ran `ls`" in out[0].content[0].text


def test_convert_to_llm_custom_string_content() -> None:
    msg = CustomMessage(role="custom", custom_type="x", content="hi", display=True, timestamp=1)
    out = convert_to_llm([msg])
    assert isinstance(out[0], UserMessage)
    assert isinstance(out[0].content, list)
    assert out[0].content[0].text == "hi"  # type: ignore[union-attr]


def test_convert_to_llm_custom_list_content() -> None:
    items: list[TextContent | ImageContent] = [TextContent(text="a"), TextContent(text="b")]
    msg = CustomMessage(role="custom", custom_type="x", content=items, display=True, timestamp=2)
    out = convert_to_llm([msg])
    assert isinstance(out[0].content, list)
    assert len(out[0].content) == 2


def test_convert_to_llm_branch_summary_wraps_text() -> None:
    msg = BranchSummaryMessage(role="branchSummary", summary="S", from_id="m", timestamp=3)
    out = convert_to_llm([msg])
    assert isinstance(out[0], UserMessage)
    assert isinstance(out[0].content, list)
    text = out[0].content[0].text  # type: ignore[union-attr]
    assert text.startswith(BRANCH_SUMMARY_PREFIX)
    assert text.endswith(BRANCH_SUMMARY_SUFFIX)
    assert "S" in text


@dataclass
class _BogusMessage:
    role: str = "totally_made_up"


def test_convert_to_llm_drops_unknown_role() -> None:
    out = convert_to_llm([_BogusMessage()])
    assert out == []


def test_convert_to_llm_drops_message_without_role() -> None:
    out = convert_to_llm([object()])
    assert out == []


def test_convert_to_llm_compaction_summary_wraps_text() -> None:
    msg = CompactionSummaryMessage(
        role="compactionSummary",
        summary="ZZZ",
        tokens_before=100,
        timestamp=4,
    )
    out = convert_to_llm([msg])
    assert isinstance(out[0].content, list)
    text = out[0].content[0].text  # type: ignore[union-attr]
    assert text.startswith(COMPACTION_SUMMARY_PREFIX)
    assert text.endswith(COMPACTION_SUMMARY_SUFFIX)
    assert "ZZZ" in text
