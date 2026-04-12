"""Tests for ``nu_coding_agent.core.compaction``."""

from __future__ import annotations

from pathlib import Path
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
    from nu_ai.providers.faux import (
        faux_assistant_message,
        register_faux_provider,
    )
    from nu_coding_agent.core.compaction import generate_summary

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
    from nu_ai.providers.faux import (
        faux_assistant_message,
        register_faux_provider,
    )
    from nu_coding_agent.core.compaction import generate_summary

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
    from nu_ai.providers.faux import (
        faux_assistant_message,
        register_faux_provider,
    )
    from nu_coding_agent.core.compaction import (
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
    from nu_ai.providers.faux import (
        faux_assistant_message,
        register_faux_provider,
    )
    from nu_coding_agent.core.compaction import (
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
    from nu_coding_agent.core.compaction.compaction import (
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
    from nu_coding_agent.core.compaction.compaction import (
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


# ---------------------------------------------------------------------------
# Reasoning level propagation — direct port of upstream
# compaction-summary-reasoning.test.ts. Verifies the bug fix that landed
# alongside the audit: generate_summary used to drop the reasoning="high"
# hint for reasoning-capable models.
# ---------------------------------------------------------------------------


def _model(*, reasoning: bool):
    from nu_ai.types import Model, ModelCost

    return Model(
        id="m",
        name="m",
        api="openai-completions",
        provider="openai",
        base_url="https://example",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=200_000,
        max_tokens=8_192,
    )


async def test_generate_summary_sets_reasoning_high_for_reasoning_models(monkeypatch: Any) -> None:
    from nu_ai.types import AssistantMessage, Cost, TextContent, Usage
    from nu_coding_agent.core.compaction import compaction as compaction_mod

    captured: list[Any] = []

    async def fake_complete_simple(model: Any, context: Any, options: Any):
        captured.append(options)
        return AssistantMessage(
            content=[TextContent(text="## Goal\nTest")],
            api="openai-completions",
            provider="openai",
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

    monkeypatch.setattr(compaction_mod, "complete_simple", fake_complete_simple)
    await compaction_mod.generate_summary(
        current_messages=[{"role": "user", "content": "Summarize this"}],
        model=_model(reasoning=True),
        reserve_tokens=2000,
        api_key="k",
    )
    assert len(captured) == 1
    assert captured[0].reasoning == "high"


async def test_generate_summary_omits_reasoning_for_non_reasoning_models(monkeypatch: Any) -> None:
    from nu_ai.types import AssistantMessage, Cost, TextContent, Usage
    from nu_coding_agent.core.compaction import compaction as compaction_mod

    captured: list[Any] = []

    async def fake_complete_simple(model: Any, context: Any, options: Any):
        captured.append(options)
        return AssistantMessage(
            content=[TextContent(text="## Goal\nTest")],
            api="openai-completions",
            provider="openai",
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

    monkeypatch.setattr(compaction_mod, "complete_simple", fake_complete_simple)
    await compaction_mod.generate_summary(
        current_messages=[{"role": "user", "content": "Summarize this"}],
        model=_model(reasoning=False),
        reserve_tokens=2000,
        api_key="k",
    )
    assert len(captured) == 1
    assert captured[0].reasoning is None


async def test_turn_prefix_summary_sets_reasoning_high_for_reasoning_models(monkeypatch: Any) -> None:
    """The split-turn helper must propagate the reasoning hint too."""
    from nu_ai.types import AssistantMessage, Cost, TextContent, Usage
    from nu_coding_agent.core.compaction import compaction as compaction_mod

    captured: list[Any] = []

    async def fake_complete_simple(model: Any, context: Any, options: Any):
        captured.append(options)
        return AssistantMessage(
            content=[TextContent(text="prefix summary")],
            api="openai-completions",
            provider="openai",
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

    monkeypatch.setattr(compaction_mod, "complete_simple", fake_complete_simple)
    # _generate_turn_prefix_summary is private but exercised via ``compact``
    # whenever ``preparation.is_split_turn`` is True with prefix messages.
    prep = compaction_mod.CompactionPreparation(
        first_kept_entry_id="kept",
        messages_to_summarize=[{"role": "user", "content": "old"}],
        turn_prefix_messages=[{"role": "user", "content": "prefix"}],
        is_split_turn=True,
        tokens_before=10,
        file_ops=create_file_ops(),
        settings=DEFAULT_COMPACTION_SETTINGS,
    )
    await compaction_mod.compact(prep, _model(reasoning=True), "k")
    # Two complete_simple calls: history summary + turn prefix summary.
    assert len(captured) == 2
    assert all(opts.reasoning == "high" for opts in captured)


# ---------------------------------------------------------------------------
# build_session_context — multi-compaction scenarios. Direct ports of the
# `buildSessionContext` describe block in upstream compaction.test.ts.
# ---------------------------------------------------------------------------


def _msg_entry(role: str, text: str, *, entry_id: str, parent_id: str | None) -> dict[str, Any]:
    """Compact builder used by the multi-compaction tests below."""
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "t",
        "message": {"role": role, "content": text},
    }


def _compaction_entry(*, summary: str, first_kept_entry_id: str, entry_id: str, parent_id: str) -> dict[str, Any]:
    return {
        "type": "compaction",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2024-01-01T00:00:00.000Z",
        "summary": summary,
        "firstKeptEntryId": first_kept_entry_id,
        "tokensBefore": 0,
    }


def test_build_session_context_no_compaction_returns_all_messages() -> None:
    from nu_coding_agent.core.session_manager import build_session_context

    entries = [
        _msg_entry("user", "1", entry_id="u1", parent_id=None),
        _msg_entry("assistant", "a", entry_id="a1", parent_id="u1"),
        _msg_entry("user", "2", entry_id="u2", parent_id="a1"),
        _msg_entry("assistant", "b", entry_id="a2", parent_id="u2"),
    ]
    ctx = build_session_context(entries)
    assert len(ctx.messages) == 4


def test_build_session_context_single_compaction() -> None:
    from nu_coding_agent.core.session_manager import build_session_context

    entries = [
        _msg_entry("user", "1", entry_id="u1", parent_id=None),
        _msg_entry("assistant", "a", entry_id="a1", parent_id="u1"),
        _msg_entry("user", "2", entry_id="u2", parent_id="a1"),
        _msg_entry("assistant", "b", entry_id="a2", parent_id="u2"),
        _compaction_entry(summary="Summary of 1,a,2,b", first_kept_entry_id="u2", entry_id="c1", parent_id="a2"),
        _msg_entry("user", "3", entry_id="u3", parent_id="c1"),
        _msg_entry("assistant", "c", entry_id="a3", parent_id="u3"),
    ]
    ctx = build_session_context(entries)
    # compaction summary + kept (u2, a2) + after (u3, a3) = 5
    assert len(ctx.messages) == 5
    first = ctx.messages[0]
    summary = getattr(first, "summary", None) or (first.get("summary") if isinstance(first, dict) else None)
    assert summary is not None
    assert "Summary of 1,a,2,b" in summary


def test_build_session_context_multiple_compactions_only_latest_matters() -> None:
    from nu_coding_agent.core.session_manager import build_session_context

    entries = [
        _msg_entry("user", "1", entry_id="u1", parent_id=None),
        _msg_entry("assistant", "a", entry_id="a1", parent_id="u1"),
        _compaction_entry(summary="First summary", first_kept_entry_id="u1", entry_id="c1", parent_id="a1"),
        _msg_entry("user", "2", entry_id="u2", parent_id="c1"),
        _msg_entry("assistant", "b", entry_id="b", parent_id="u2"),
        _msg_entry("user", "3", entry_id="u3", parent_id="b"),
        _msg_entry("assistant", "c", entry_id="c", parent_id="u3"),
        _compaction_entry(summary="Second summary", first_kept_entry_id="u3", entry_id="c2", parent_id="c"),
        _msg_entry("user", "4", entry_id="u4", parent_id="c2"),
        _msg_entry("assistant", "d", entry_id="d", parent_id="u4"),
    ]
    ctx = build_session_context(entries)
    # Second summary + kept from u3 (u3, c) + after (u4, d) = 5
    assert len(ctx.messages) == 5
    first = ctx.messages[0]
    summary = getattr(first, "summary", None) or (first.get("summary") if isinstance(first, dict) else None)
    assert summary is not None
    assert "Second summary" in summary
    assert "First summary" not in summary


def test_build_session_context_first_kept_at_first_entry_keeps_all() -> None:
    from nu_coding_agent.core.session_manager import build_session_context

    entries = [
        _msg_entry("user", "1", entry_id="u1", parent_id=None),
        _msg_entry("assistant", "a", entry_id="a1", parent_id="u1"),
        _compaction_entry(summary="First summary", first_kept_entry_id="u1", entry_id="c1", parent_id="a1"),
        _msg_entry("user", "2", entry_id="u2", parent_id="c1"),
        _msg_entry("assistant", "b", entry_id="b", parent_id="u2"),
    ]
    ctx = build_session_context(entries)
    # summary + (u1, a1, u2, b) = 5
    assert len(ctx.messages) == 5


# ---------------------------------------------------------------------------
# Cut point — split-turn detection
# ---------------------------------------------------------------------------


def test_find_cut_point_indicates_split_turn() -> None:
    """Cutting in the middle of a turn must flag is_split_turn + turn_start_index."""
    entries: list[dict[str, Any]] = [
        # Turn 1
        _entry("message", role="user", content="Turn 1", entry_id="u1"),
        _entry("message", role="assistant", content="A1", entry_id="a1", parent_id="u1"),
        # Turn 2 — multiple assistant messages, the last few will fall in the
        # "keep recent" window and force a cut at an assistant entry.
        _entry("message", role="user", content="Turn 2", entry_id="u2", parent_id="a1"),
        _entry("message", role="assistant", content="A2-1" * 200, entry_id="a2_1", parent_id="u2"),
        _entry("message", role="assistant", content="A2-2" * 200, entry_id="a2_2", parent_id="a2_1"),
        _entry("message", role="assistant", content="A2-3" * 200, entry_id="a2_3", parent_id="a2_2"),
    ]
    result = find_cut_point(entries, 0, len(entries), 300)
    cut_entry = entries[result.first_kept_entry_index]
    if cut_entry["message"]["role"] == "assistant":
        assert result.is_split_turn is True
        assert result.turn_start_index == 2  # Turn 2 starts at index 2 (u2)


# ---------------------------------------------------------------------------
# Large session fixture round-trip — direct port of the "Large session
# fixture" describe block. The fixture itself is vendored verbatim from
# upstream so it round-trips identically through both implementations.
# ---------------------------------------------------------------------------


_LARGE_SESSION_FIXTURE = Path(__file__).parent / "fixtures" / "large-session.jsonl"


def _load_large_session_entries() -> list[dict[str, Any]]:
    from nu_coding_agent.core.session_manager import (
        load_entries_from_file,
        migrate_session_entries,
    )

    entries = load_entries_from_file(str(_LARGE_SESSION_FIXTURE))
    migrate_session_entries(entries)
    return entries


def test_large_session_fixture_parses() -> None:
    entries = _load_large_session_entries()
    assert len(entries) > 100
    message_count = sum(1 for e in entries if e.get("type") == "message")
    assert message_count > 100


def test_large_session_fixture_find_cut_point_lands_on_message() -> None:
    entries = _load_large_session_entries()
    result = find_cut_point(entries, 0, len(entries), DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens)
    cut_entry = entries[result.first_kept_entry_index]
    assert cut_entry.get("type") == "message"
    role = cut_entry.get("message", {}).get("role")
    assert role in ("user", "assistant")


def test_large_session_fixture_build_session_context_round_trip() -> None:
    from nu_coding_agent.core.session_manager import build_session_context

    entries = _load_large_session_entries()
    ctx = build_session_context(entries)
    assert len(ctx.messages) > 100
    assert ctx.model is not None
