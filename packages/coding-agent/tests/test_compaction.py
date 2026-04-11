"""Tests for ``nu_coding_agent.core.compaction``."""

from __future__ import annotations

from typing import Any

from nu_coding_agent.core.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    calculate_context_tokens,
    compute_file_lists,
    create_file_ops,
    estimate_context_tokens,
    estimate_tokens,
    extract_file_ops_from_message,
    find_cut_point,
    find_turn_start_index,
    format_file_operations,
    get_last_assistant_usage,
    prepare_compaction,
    serialize_conversation,
    should_compact,
)

# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def _assistant_msg(*tool_calls: tuple[str, str]) -> dict[str, Any]:
    """Build an assistant message dict with the given tool calls (name, path)."""
    content: list[dict[str, Any]] = []
    for name, path in tool_calls:
        content.append({"type": "toolCall", "name": name, "arguments": {"path": path}})
    return {"role": "assistant", "content": content}


def test_extract_file_ops_basic() -> None:
    file_ops = create_file_ops()
    extract_file_ops_from_message(
        _assistant_msg(("read", "/a"), ("write", "/b"), ("edit", "/c")),
        file_ops,
    )
    assert file_ops.read == {"/a"}
    assert file_ops.written == {"/b"}
    assert file_ops.edited == {"/c"}


def test_extract_file_ops_skips_non_assistant() -> None:
    file_ops = create_file_ops()
    extract_file_ops_from_message({"role": "user", "content": "hi"}, file_ops)
    assert not file_ops.read


def test_extract_file_ops_skips_non_tool_blocks() -> None:
    file_ops = create_file_ops()
    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}, {"type": "toolCall", "name": "read", "arguments": {}}],
    }
    extract_file_ops_from_message(msg, file_ops)
    assert not file_ops.read  # arguments dict has no "path"


def test_compute_file_lists_dedupes_modified() -> None:
    file_ops = create_file_ops()
    file_ops.read.add("/a")
    file_ops.read.add("/b")
    file_ops.written.add("/b")
    file_ops.edited.add("/c")
    read, modified = compute_file_lists(file_ops)
    assert read == ["/a"]  # /b moved to modified
    assert modified == ["/b", "/c"]


def test_format_file_operations_with_both() -> None:
    out = format_file_operations(["/a"], ["/b"])
    assert "<read-files>" in out
    assert "<modified-files>" in out


def test_format_file_operations_empty() -> None:
    assert format_file_operations([], []) == ""


# ---------------------------------------------------------------------------
# serialize_conversation
# ---------------------------------------------------------------------------


def test_serialize_user_message() -> None:
    msgs = [{"role": "user", "content": "hello"}]
    out = serialize_conversation(msgs)  # type: ignore[arg-type]
    assert "[User]: hello" in out


def test_serialize_assistant_with_tool_call() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking aloud"},
                {"type": "toolCall", "name": "read", "arguments": {"path": "/x"}},
            ],
        }
    ]
    out = serialize_conversation(msgs)  # type: ignore[arg-type]
    assert "[Assistant]: thinking aloud" in out
    assert "[Assistant tool calls]: read" in out


def test_serialize_tool_result_truncated() -> None:
    huge = "x" * 5000
    msgs = [{"role": "toolResult", "content": [{"type": "text", "text": huge}]}]
    out = serialize_conversation(msgs)  # type: ignore[arg-type]
    assert "more characters truncated" in out


def test_serialize_assistant_with_thinking() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "ponder"}],
        }
    ]
    out = serialize_conversation(msgs)  # type: ignore[arg-type]
    assert "[Assistant thinking]: ponder" in out


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_user_string_content() -> None:
    assert estimate_tokens({"role": "user", "content": "abcd"}) == 1  # 4/4 = 1


def test_estimate_tokens_user_list_content() -> None:
    msg = {"role": "user", "content": [{"type": "text", "text": "abcdefgh"}]}
    assert estimate_tokens(msg) == 2  # 8/4


def test_estimate_tokens_assistant() -> None:
    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": "abcdefgh"}],
    }
    assert estimate_tokens(msg) == 2


def test_estimate_tokens_assistant_with_tool_call() -> None:
    msg = {
        "role": "assistant",
        "content": [
            {"type": "toolCall", "name": "read", "arguments": {"path": "/x"}},
        ],
    }
    assert estimate_tokens(msg) > 0


def test_estimate_tokens_tool_result_with_image() -> None:
    msg = {
        "role": "toolResult",
        "content": [{"type": "image"}],
    }
    assert estimate_tokens(msg) == 1200  # 4800/4


def test_estimate_tokens_unknown_role_returns_zero() -> None:
    assert estimate_tokens({"role": "unknown"}) == 0


def test_estimate_tokens_branch_summary() -> None:
    assert estimate_tokens({"role": "branchSummary", "summary": "abcdefgh"}) == 2


# ---------------------------------------------------------------------------
# calculate_context_tokens / get_last_assistant_usage
# ---------------------------------------------------------------------------


class _Usage:
    def __init__(self, total: int, parts: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
        self.total_tokens = total
        self.input, self.output, self.cache_read, self.cache_write = parts


def test_calculate_context_tokens_uses_total() -> None:
    assert calculate_context_tokens(_Usage(total=42)) == 42  # type: ignore[arg-type]


def test_calculate_context_tokens_falls_back_to_components() -> None:
    assert calculate_context_tokens(_Usage(total=0, parts=(1, 2, 3, 4))) == 10  # type: ignore[arg-type]


def test_get_last_assistant_usage_skips_aborted() -> None:
    entries: list[dict[str, Any]] = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "stopReason": "aborted",
                "usage": _Usage(total=999),
            },
        }
    ]
    assert get_last_assistant_usage(entries) is None


def test_get_last_assistant_usage_returns_latest() -> None:
    usage = _Usage(total=42)
    entries: list[dict[str, Any]] = [
        {
            "type": "message",
            "message": {"role": "assistant", "stopReason": "stop", "usage": usage},
        }
    ]
    assert get_last_assistant_usage(entries) is usage  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# estimate_context_tokens / should_compact
# ---------------------------------------------------------------------------


def test_estimate_context_tokens_no_usage_falls_back_to_chars() -> None:
    msgs = [{"role": "user", "content": "abcdefgh"}, {"role": "user", "content": "ijklmnop"}]
    estimate = estimate_context_tokens(msgs)
    assert estimate.usage_tokens == 0
    assert estimate.tokens > 0
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_uses_last_usage() -> None:
    msgs: list[Any] = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "stopReason": "stop",
            "content": [],
            "usage": _Usage(total=100),
        },
        {"role": "user", "content": "trailing"},
    ]
    estimate = estimate_context_tokens(msgs)
    assert estimate.usage_tokens == 100
    assert estimate.trailing_tokens > 0
    assert estimate.last_usage_index == 1


def test_should_compact_disabled() -> None:
    settings = CompactionSettings(enabled=False, reserve_tokens=1000, keep_recent_tokens=500)
    assert should_compact(99999, 100000, settings) is False


def test_should_compact_under_budget() -> None:
    settings = DEFAULT_COMPACTION_SETTINGS
    assert should_compact(50000, 200000, settings) is False


def test_should_compact_over_budget() -> None:
    settings = DEFAULT_COMPACTION_SETTINGS
    assert should_compact(200000, 200000, settings) is True


# ---------------------------------------------------------------------------
# find_cut_point / find_turn_start_index
# ---------------------------------------------------------------------------


def _entry(
    entry_type: str,
    *,
    role: str | None = None,
    content: str = "x",
    entry_id: str = "id",
    parent_id: str | None = None,
) -> dict[str, Any]:
    if entry_type == "message":
        return {
            "type": "message",
            "id": entry_id,
            "parentId": parent_id,
            "message": {"role": role, "content": content},
            "timestamp": "t",
        }
    return {"type": entry_type, "id": entry_id, "parentId": parent_id, "timestamp": "t"}


def test_find_cut_point_no_messages() -> None:
    entries = [_entry("session_info", entry_id="a")]
    result = find_cut_point(entries, 0, 1, 1000)
    assert result.first_kept_entry_index == 0
    assert result.is_split_turn is False


def test_find_cut_point_keeps_recent() -> None:
    entries: list[dict[str, Any]] = []
    for i in range(20):
        entries.append(_entry("message", role="user", content="x" * 200, entry_id=f"u{i}"))
        entries.append(_entry("message", role="assistant", content="x" * 200, entry_id=f"a{i}"))
    result = find_cut_point(entries, 0, len(entries), 200)
    # Should keep the more recent messages
    assert result.first_kept_entry_index > 0


def test_find_turn_start_index_finds_user_message() -> None:
    entries = [
        _entry("message", role="user", entry_id="u1"),
        _entry("message", role="assistant", entry_id="a1"),
        _entry("message", role="assistant", entry_id="a2"),
    ]
    assert find_turn_start_index(entries, 2, 0) == 0


def test_find_turn_start_index_returns_minus_one_when_no_user() -> None:
    entries = [
        _entry("message", role="assistant", entry_id="a1"),
        _entry("message", role="assistant", entry_id="a2"),
    ]
    assert find_turn_start_index(entries, 1, 0) == -1


# ---------------------------------------------------------------------------
# prepare_compaction
# ---------------------------------------------------------------------------


def test_prepare_compaction_skips_if_last_is_compaction() -> None:
    entries = [
        _entry("message", role="user", entry_id="u1"),
        _entry("compaction", entry_id="c1"),
    ]
    assert prepare_compaction(entries, DEFAULT_COMPACTION_SETTINGS) is None


def test_prepare_compaction_returns_preparation() -> None:
    entries: list[dict[str, Any]] = []
    parent_id: str | None = None
    for i in range(20):
        u_id = f"u{i}"
        entries.append(_entry("message", role="user", content="x" * 500, entry_id=u_id, parent_id=parent_id))
        a_id = f"a{i}"
        entries.append(_entry("message", role="assistant", content="x" * 500, entry_id=a_id, parent_id=u_id))
        parent_id = a_id
    prep = prepare_compaction(entries, CompactionSettings(enabled=True, reserve_tokens=1000, keep_recent_tokens=300))
    assert prep is not None
    assert prep.first_kept_entry_id
    assert len(prep.messages_to_summarize) > 0


def test_prepare_compaction_with_previous_compaction() -> None:
    entries: list[dict[str, Any]] = []
    parent_id: str | None = None
    for i in range(5):
        u_id = f"u{i}"
        entries.append(_entry("message", role="user", content="x" * 500, entry_id=u_id, parent_id=parent_id))
        a_id = f"a{i}"
        entries.append(_entry("message", role="assistant", content="x" * 500, entry_id=a_id, parent_id=u_id))
        parent_id = a_id
    # Insert a compaction in the middle.
    entries[4] = {
        "type": "compaction",
        "id": "comp1",
        "parentId": entries[4]["parentId"],
        "summary": "earlier summary",
        "firstKeptEntryId": "u2",
        "tokensBefore": 1000,
        "timestamp": "t",
    }
    prep = prepare_compaction(entries, CompactionSettings(enabled=True, reserve_tokens=1000, keep_recent_tokens=200))
    assert prep is not None
    assert prep.previous_summary == "earlier summary"


# ---------------------------------------------------------------------------
# generate_summary + compact (faux provider end-to-end)
# ---------------------------------------------------------------------------


async def test_generate_summary_with_faux_provider() -> None:
    from nu_ai.providers.faux import (  # noqa: PLC0415
        faux_assistant_message,
        register_faux_provider,
    )
    from nu_coding_agent.core.compaction import generate_summary  # noqa: PLC0415

    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("STRUCTURED SUMMARY OUTPUT")])
        summary = await generate_summary(
            current_messages=[{"role": "user", "content": "do a thing"}],
            model=registration.get_model(),
            reserve_tokens=1000,
            api_key="ignored",
        )
        assert "STRUCTURED SUMMARY OUTPUT" in summary
    finally:
        registration.unregister()


async def test_generate_summary_uses_update_prompt_with_previous_summary() -> None:
    from nu_ai.providers.faux import (  # noqa: PLC0415
        faux_assistant_message,
        register_faux_provider,
    )
    from nu_coding_agent.core.compaction import generate_summary  # noqa: PLC0415

    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("UPDATED SUMMARY")])
        summary = await generate_summary(
            current_messages=[{"role": "user", "content": "more work"}],
            model=registration.get_model(),
            reserve_tokens=1000,
            api_key="ignored",
            previous_summary="earlier text",
            custom_instructions="focus on bugs",
        )
        assert "UPDATED SUMMARY" in summary
    finally:
        registration.unregister()


async def test_compact_runs_against_faux_provider() -> None:
    from nu_ai.providers.faux import (  # noqa: PLC0415
        faux_assistant_message,
        register_faux_provider,
    )
    from nu_coding_agent.core.compaction import (  # noqa: PLC0415
        CompactionPreparation,
        compact,
        create_file_ops,
    )

    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("checkpoint summary")])
        prep = CompactionPreparation(
            first_kept_entry_id="kept",
            messages_to_summarize=[{"role": "user", "content": "old"}],
            turn_prefix_messages=[],
            is_split_turn=False,
            tokens_before=1234,
            file_ops=create_file_ops(),
            settings=DEFAULT_COMPACTION_SETTINGS,
        )
        result = await compact(prep, registration.get_model(), "ignored")
        assert "checkpoint summary" in result.summary
        assert result.first_kept_entry_id == "kept"
        assert result.tokens_before == 1234
    finally:
        registration.unregister()


async def test_compact_split_turn_uses_two_summaries() -> None:
    from nu_ai.providers.faux import (  # noqa: PLC0415
        faux_assistant_message,
        register_faux_provider,
    )
    from nu_coding_agent.core.compaction import (  # noqa: PLC0415
        CompactionPreparation,
        compact,
        create_file_ops,
    )

    registration = register_faux_provider()
    try:
        registration.set_responses(
            [
                faux_assistant_message("HISTORY"),
                faux_assistant_message("TURN PREFIX"),
            ]
        )
        prep = CompactionPreparation(
            first_kept_entry_id="kept",
            messages_to_summarize=[{"role": "user", "content": "old"}],
            turn_prefix_messages=[{"role": "user", "content": "turn prefix"}],
            is_split_turn=True,
            tokens_before=10,
            file_ops=create_file_ops(),
            settings=DEFAULT_COMPACTION_SETTINGS,
        )
        result = await compact(prep, registration.get_model(), "ignored")
        assert "HISTORY" in result.summary
        assert "TURN PREFIX" in result.summary
        assert "Turn Context (split turn)" in result.summary
    finally:
        registration.unregister()


def test_prepare_compaction_with_compaction_details() -> None:
    """Compaction detail blob from a previous compaction should seed file_ops."""
    entries: list[dict[str, Any]] = [
        _entry("message", role="user", entry_id="u1"),
        {
            "type": "compaction",
            "id": "c1",
            "parentId": "u1",
            "timestamp": "t",
            "summary": "earlier",
            "firstKeptEntryId": "u1",
            "tokensBefore": 100,
            "details": {"readFiles": ["/a"], "modifiedFiles": ["/b"]},
        },
        _entry("message", role="user", entry_id="u2"),
        _entry("message", role="assistant", content="x" * 5000, entry_id="a2"),
    ]
    prep = prepare_compaction(entries, CompactionSettings(enabled=True, reserve_tokens=1000, keep_recent_tokens=10))
    assert prep is not None
    assert "/a" in prep.file_ops.read
    assert "/b" in prep.file_ops.edited


def test_get_message_from_entry_branch_summary() -> None:
    from nu_coding_agent.core.compaction.compaction import (  # noqa: PLC0415
        _get_message_from_entry,  # pyright: ignore[reportPrivateUsage]
        _get_message_from_entry_for_compaction,  # pyright: ignore[reportPrivateUsage]
    )

    branch = {
        "type": "branch_summary",
        "summary": "abc",
        "fromId": "x",
        "timestamp": "2026-01-01T00:00:00.000Z",
    }
    msg = _get_message_from_entry(branch)
    assert msg is not None
    assert _get_message_from_entry_for_compaction({"type": "compaction"}) is None


def test_get_message_from_entry_custom_message() -> None:
    from nu_coding_agent.core.compaction.compaction import (  # noqa: PLC0415
        _get_message_from_entry,  # pyright: ignore[reportPrivateUsage]
    )

    custom = {
        "type": "custom_message",
        "customType": "x",
        "content": "hi",
        "display": True,
        "timestamp": "2026-01-01T00:00:00.000Z",
    }
    assert _get_message_from_entry(custom) is not None
