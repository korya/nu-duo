"""Tests for ``nu_coding_agent.core.compaction.branch_summarization``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nu_coding_agent.core.compaction import (
    GenerateBranchSummaryOptions,
    collect_entries_for_branch_summary,
    generate_branch_summary,
    prepare_branch_entries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg_entry(entry_id: str, parent_id: str | None, role: str, content: str = "x") -> dict[str, Any]:
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"role": role, "content": content},
    }


def _branch_summary_entry(
    entry_id: str, parent_id: str | None, *, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "branch_summary",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "summary": "branch text",
        "fromId": parent_id or "root",
    }
    if details is not None:
        entry["details"] = details
    return entry


@dataclass
class _FakeSession:
    """Minimal session source: a flat tree built from a parent map."""

    entries: dict[str, dict[str, Any]]

    def get_branch(self, from_id: str | None = None) -> list[dict[str, Any]]:
        path: list[dict[str, Any]] = []
        current = self.entries.get(from_id) if from_id else None
        while current is not None:
            path.insert(0, current)
            parent_id = current.get("parentId")
            current = self.entries.get(parent_id) if parent_id else None
        return path

    def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        return self.entries.get(entry_id)


# ---------------------------------------------------------------------------
# collect_entries_for_branch_summary
# ---------------------------------------------------------------------------


def test_collect_entries_returns_empty_when_no_old_leaf() -> None:
    session = _FakeSession(entries={})
    result = collect_entries_for_branch_summary(session, None, "target")
    assert result.entries == []
    assert result.common_ancestor_id is None


def test_collect_entries_finds_common_ancestor() -> None:
    # Tree:    root
    #         /    \
    #       a       b
    #       |       |
    #       a2      b2
    entries = {
        "root": _msg_entry("root", None, "user"),
        "a": _msg_entry("a", "root", "assistant"),
        "a2": _msg_entry("a2", "a", "user"),
        "b": _msg_entry("b", "root", "assistant"),
        "b2": _msg_entry("b2", "b", "user"),
    }
    session = _FakeSession(entries=entries)
    result = collect_entries_for_branch_summary(session, "a2", "b2")
    # Walks back from a2 to the common ancestor (root), collecting a2 → a.
    assert {e["id"] for e in result.entries} == {"a", "a2"}
    assert result.common_ancestor_id == "root"


def test_collect_entries_no_common_ancestor() -> None:
    entries = {
        "old": _msg_entry("old", None, "user"),
        "new": _msg_entry("new", None, "user"),
    }
    session = _FakeSession(entries=entries)
    result = collect_entries_for_branch_summary(session, "old", "new")
    # No shared ancestor — collects everything from old up to root.
    assert {e["id"] for e in result.entries} == {"old"}
    assert result.common_ancestor_id is None


# ---------------------------------------------------------------------------
# prepare_branch_entries
# ---------------------------------------------------------------------------


def test_prepare_branch_entries_no_budget_keeps_all() -> None:
    entries = [
        _msg_entry("u1", None, "user", content="hello"),
        _msg_entry("a1", "u1", "assistant", content="hi"),
    ]
    prep = prepare_branch_entries(entries, token_budget=0)
    assert len(prep.messages) == 2
    assert prep.total_tokens > 0


def test_prepare_branch_entries_budget_drops_old() -> None:
    entries = [_msg_entry(f"u{i}", None, "user", content="x" * 1000) for i in range(20)]
    prep = prepare_branch_entries(entries, token_budget=100)
    # Should drop most messages once over budget.
    assert len(prep.messages) < 20


def test_prepare_branch_entries_collects_file_ops_from_branch_summary() -> None:
    entries = [
        _branch_summary_entry(
            "bs1",
            None,
            details={"readFiles": ["/a"], "modifiedFiles": ["/b"]},
        ),
        _msg_entry("u1", "bs1", "user"),
    ]
    prep = prepare_branch_entries(entries, token_budget=0)
    assert "/a" in prep.file_ops.read
    assert "/b" in prep.file_ops.edited


def test_prepare_branch_entries_skips_hooked_summaries() -> None:
    entries = [
        {
            "type": "branch_summary",
            "id": "bs1",
            "parentId": None,
            "timestamp": "2026-01-01T00:00:00.000Z",
            "summary": "x",
            "fromId": "root",
            "fromHook": True,
            "details": {"readFiles": ["/skipped"]},
        }
    ]
    prep = prepare_branch_entries(entries, token_budget=0)
    assert "/skipped" not in prep.file_ops.read


# ---------------------------------------------------------------------------
# generate_branch_summary (faux provider)
# ---------------------------------------------------------------------------


async def test_generate_branch_summary_with_faux_provider() -> None:
    from nu_ai.providers.faux import (  # noqa: PLC0415
        faux_assistant_message,
        register_faux_provider,
    )

    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("BRANCH SUMMARY OUTPUT")])
        entries = [
            _msg_entry("u1", None, "user", content="hello"),
            _msg_entry("a1", "u1", "assistant", content="hi"),
        ]
        result = await generate_branch_summary(
            entries,
            GenerateBranchSummaryOptions(
                model=registration.get_model(),
                api_key="ignored",
            ),
        )
        assert result.summary is not None
        assert "BRANCH SUMMARY OUTPUT" in result.summary
        assert "explored a different conversation branch" in result.summary
    finally:
        registration.unregister()


async def test_generate_branch_summary_empty_entries() -> None:
    from nu_ai.providers.faux import register_faux_provider  # noqa: PLC0415

    registration = register_faux_provider()
    try:
        result = await generate_branch_summary(
            [],
            GenerateBranchSummaryOptions(
                model=registration.get_model(),
                api_key="ignored",
            ),
        )
        assert result.summary == "No content to summarize"
    finally:
        registration.unregister()


async def test_generate_branch_summary_with_replace_instructions() -> None:
    from nu_ai.providers.faux import (  # noqa: PLC0415
        faux_assistant_message,
        register_faux_provider,
    )

    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("CUSTOM PROMPT REPLY")])
        entries = [_msg_entry("u1", None, "user", content="hi")]
        result = await generate_branch_summary(
            entries,
            GenerateBranchSummaryOptions(
                model=registration.get_model(),
                api_key="ignored",
                custom_instructions="Just say hello",
                replace_instructions=True,
            ),
        )
        assert result.summary is not None
        assert "CUSTOM PROMPT REPLY" in result.summary
    finally:
        registration.unregister()


async def test_generate_branch_summary_with_appended_instructions() -> None:
    from nu_ai.providers.faux import (  # noqa: PLC0415
        faux_assistant_message,
        register_faux_provider,
    )

    registration = register_faux_provider()
    try:
        registration.set_responses([faux_assistant_message("APPENDED REPLY")])
        entries = [_msg_entry("u1", None, "user", content="hi")]
        result = await generate_branch_summary(
            entries,
            GenerateBranchSummaryOptions(
                model=registration.get_model(),
                api_key="ignored",
                custom_instructions="extra focus",
            ),
        )
        assert result.summary is not None
        assert "APPENDED REPLY" in result.summary
    finally:
        registration.unregister()
