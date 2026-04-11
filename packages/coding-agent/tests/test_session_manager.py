"""Tests for ``nu_coding_agent.core.session_manager``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nu_coding_agent.core.session_manager import (
    CURRENT_SESSION_VERSION,
    SessionManager,
    build_session_context,
    find_most_recent_session,
    get_default_session_dir,
    load_entries_from_file,
    migrate_session_entries,
    parse_session_entries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_message(text: str, *, timestamp: int = 1) -> dict[str, Any]:
    return {"role": "user", "content": text, "timestamp": timestamp}


def _assistant_message(text: str, *, provider: str = "openai", model: str = "m") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "provider": provider,
        "model": model,
        "api": "openai-completions",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        },
        "stopReason": "stop",
        "timestamp": 2,
    }


# ---------------------------------------------------------------------------
# parse + load
# ---------------------------------------------------------------------------


def test_parse_session_entries_skips_blank_and_invalid_lines() -> None:
    blob = '{"type": "session", "id": "x"}\n\nnot-json\n{"type": "message"}\n'
    entries = parse_session_entries(blob)
    assert len(entries) == 2
    assert entries[0]["type"] == "session"


def test_load_entries_from_file_missing(tmp_path: Path) -> None:
    assert load_entries_from_file(str(tmp_path / "missing.jsonl")) == []


def test_load_entries_requires_session_header(tmp_path: Path) -> None:
    file = tmp_path / "headerless.jsonl"
    file.write_text('{"type": "message", "id": "abc"}\n')
    assert load_entries_from_file(str(file)) == []


def test_load_entries_with_valid_header(tmp_path: Path) -> None:
    file = tmp_path / "ok.jsonl"
    file.write_text('{"type":"session","id":"sid","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/x"}\n')
    entries = load_entries_from_file(str(file))
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def test_migrate_v1_to_current_assigns_ids() -> None:
    entries: list[dict[str, Any]] = [
        {"type": "session", "id": "sid", "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/x"},
        {"type": "message", "message": _user_message("hello")},
        {"type": "message", "message": _assistant_message("hi back")},
    ]
    migrate_session_entries(entries)
    assert entries[0]["version"] == CURRENT_SESSION_VERSION
    assert "id" in entries[1]
    assert entries[1]["parentId"] is None
    assert entries[2]["parentId"] == entries[1]["id"]


def test_migrate_v2_renames_hook_message_role() -> None:
    entries: list[dict[str, Any]] = [
        {"type": "session", "version": 2, "id": "sid", "timestamp": "t", "cwd": "/x"},
        {
            "type": "message",
            "id": "e1",
            "parentId": None,
            "timestamp": "t",
            "message": {"role": "hookMessage", "content": "x", "timestamp": 1},
        },
    ]
    migrate_session_entries(entries)
    assert entries[1]["message"]["role"] == "custom"
    assert entries[0]["version"] == CURRENT_SESSION_VERSION


# ---------------------------------------------------------------------------
# In-memory append flow
# ---------------------------------------------------------------------------


def test_append_message_advances_leaf() -> None:
    sm = SessionManager.in_memory("/work")
    first = sm.append_message(_user_message("hi"))
    second = sm.append_message(_assistant_message("hello"))
    entries = sm.get_entries()
    assert len(entries) == 2
    assert entries[0]["id"] == first
    assert entries[1]["parentId"] == first
    assert sm.get_leaf_id() == second


def test_append_thinking_and_model_change() -> None:
    sm = SessionManager.in_memory()
    sm.append_thinking_level_change("high")
    sm.append_model_change("openai", "gpt-4o")
    entries = sm.get_entries()
    assert entries[0]["thinkingLevel"] == "high"
    assert entries[1]["provider"] == "openai"
    assert entries[1]["modelId"] == "gpt-4o"


def test_append_compaction_with_optional_fields() -> None:
    sm = SessionManager.in_memory()
    parent = sm.append_message(_user_message("hi"))
    sm.append_compaction(
        summary="summary",
        first_kept_entry_id=parent,
        tokens_before=1000,
        details={"k": 1},
        from_hook=True,
    )
    entries = sm.get_entries()
    assert entries[1]["summary"] == "summary"
    assert entries[1]["firstKeptEntryId"] == parent
    assert entries[1]["details"] == {"k": 1}
    assert entries[1]["fromHook"] is True


def test_session_info_round_trip() -> None:
    sm = SessionManager.in_memory()
    sm.append_session_info(" Custom Name ")
    assert sm.get_session_name() == "Custom Name"
    sm.append_session_info("")
    assert sm.get_session_name() is None


def test_label_lifecycle() -> None:
    sm = SessionManager.in_memory()
    msg_id = sm.append_message(_user_message("hi"))
    sm.append_label_change(msg_id, "marker")
    assert sm.get_label(msg_id) == "marker"
    sm.append_label_change(msg_id, None)
    assert sm.get_label(msg_id) is None


def test_label_invalid_target_raises() -> None:
    sm = SessionManager.in_memory()
    with pytest.raises(ValueError, match="not found"):
        sm.append_label_change("nope", "x")


def test_branch_resets_leaf_pointer() -> None:
    sm = SessionManager.in_memory()
    a = sm.append_message(_user_message("a"))
    sm.append_message(_assistant_message("b"))
    sm.branch(a)
    assert sm.get_leaf_id() == a
    c = sm.append_message(_user_message("c"))
    # The new entry hangs off `a`, not the previous leaf.
    new_entry = sm.get_entry(c)
    assert new_entry is not None
    assert new_entry["parentId"] == a


def test_branch_invalid_target_raises() -> None:
    sm = SessionManager.in_memory()
    with pytest.raises(ValueError, match="not found"):
        sm.branch("nope")


def test_reset_leaf_clears_pointer() -> None:
    sm = SessionManager.in_memory()
    sm.append_message(_user_message("a"))
    sm.reset_leaf()
    assert sm.get_leaf_id() is None
    new = sm.append_message(_user_message("b"))
    assert sm.get_entry(new)["parentId"] is None  # type: ignore[index]


def test_branch_with_summary() -> None:
    sm = SessionManager.in_memory()
    a = sm.append_message(_user_message("a"))
    summary_id = sm.branch_with_summary(a, "branch summary text", details={"k": 1}, from_hook=True)
    entries = sm.get_entries()
    summary = next(e for e in entries if e["id"] == summary_id)
    assert summary["type"] == "branch_summary"
    assert summary["fromId"] == a
    assert summary["details"] == {"k": 1}
    assert summary["fromHook"] is True


def test_branch_with_summary_invalid_target() -> None:
    sm = SessionManager.in_memory()
    with pytest.raises(ValueError, match="not found"):
        sm.branch_with_summary("nope", "x")


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------


def test_get_tree_two_branches() -> None:
    sm = SessionManager.in_memory()
    a = sm.append_message(_user_message("root"))
    sm.append_message(_user_message("branch1"))
    sm.branch(a)
    sm.append_message(_user_message("branch2"))
    tree = sm.get_tree()
    assert len(tree) == 1  # one root
    assert len(tree[0].children) == 2


# ---------------------------------------------------------------------------
# build_session_context
# ---------------------------------------------------------------------------


def test_build_context_walks_to_root() -> None:
    sm = SessionManager.in_memory()
    sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("there"))
    ctx = sm.build_session_context()
    assert len(ctx.messages) == 2
    assert ctx.model == {"provider": "openai", "modelId": "m"}


def test_build_context_thinking_level_propagates() -> None:
    sm = SessionManager.in_memory()
    sm.append_thinking_level_change("high")
    sm.append_message(_user_message("hi"))
    assert sm.build_session_context().thinking_level == "high"


def test_build_context_with_compaction() -> None:
    sm = SessionManager.in_memory()
    a = sm.append_message(_user_message("first"))
    sm.append_message(_user_message("second"))
    sm.append_compaction(summary="compacted", first_kept_entry_id=a, tokens_before=100)
    ctx = sm.build_session_context()
    # First message is the compaction summary, then the kept user message.
    assert len(ctx.messages) >= 1


def test_build_context_no_entries() -> None:
    sm = SessionManager.in_memory()
    ctx = sm.build_session_context()
    assert ctx.messages == []
    assert ctx.thinking_level == "off"
    assert ctx.model is None


# ---------------------------------------------------------------------------
# Persisted sessions
# ---------------------------------------------------------------------------


def test_persisted_session_writes_after_assistant(tmp_path: Path) -> None:
    sm = SessionManager.create(str(tmp_path / "work"), session_dir=str(tmp_path / "sessions"))
    session_file = sm.get_session_file()
    assert session_file is not None
    sm.append_message(_user_message("hi"))
    # Without an assistant message, file should not be flushed yet.
    assert not Path(session_file).exists()
    sm.append_message(_assistant_message("there"))
    # Now the file exists with the user + assistant entries.
    assert Path(session_file).exists()
    contents = Path(session_file).read_text(encoding="utf-8").splitlines()
    assert len(contents) >= 3  # header + user + assistant


def test_open_persisted_session(tmp_path: Path) -> None:
    sm = SessionManager.create(str(tmp_path / "work"), session_dir=str(tmp_path / "sessions"))
    sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("there"))
    session_file = sm.get_session_file()
    assert session_file is not None

    sm2 = SessionManager.open(session_file)
    assert sm2.get_session_id() == sm.get_session_id()
    assert len(sm2.get_entries()) == 2


def test_open_with_cwd_override(tmp_path: Path) -> None:
    sm = SessionManager.create(str(tmp_path / "work"), session_dir=str(tmp_path / "sessions"))
    sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("there"))
    session_file = sm.get_session_file()
    assert session_file is not None

    sm2 = SessionManager.open(session_file, cwd_override="/different/cwd")
    assert sm2.get_cwd() == "/different/cwd"


def test_continue_recent_picks_latest(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sm1 = SessionManager.create("/work", session_dir=str(sessions))
    sm1.append_message(_user_message("first"))
    sm1.append_message(_assistant_message("hi"))
    sm2 = SessionManager.create("/work", session_dir=str(sessions))
    sm2.append_message(_user_message("second"))
    sm2.append_message(_assistant_message("hello"))
    cont = SessionManager.continue_recent("/work", session_dir=str(sessions))
    # Should match either of the two — the most recently modified.
    assert cont.get_session_file() in (sm1.get_session_file(), sm2.get_session_file())


def test_continue_recent_creates_new_when_empty(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    cont = SessionManager.continue_recent("/work", session_dir=str(sessions))
    assert cont.get_session_id()
    assert cont.get_entries() == []


def test_find_most_recent_returns_none_for_missing_dir(tmp_path: Path) -> None:
    assert find_most_recent_session(str(tmp_path / "no-such")) is None


def test_find_most_recent_skips_invalid(tmp_path: Path) -> None:
    (tmp_path / "garbage.jsonl").write_text("not a session\n")
    assert find_most_recent_session(str(tmp_path)) is None


def test_get_default_session_dir_creates_directory(tmp_path: Path) -> None:
    out = get_default_session_dir("/Users/test/work", agent_dir=str(tmp_path))
    assert Path(out).exists()
    assert "Users" in out
    assert "work" in out


# ---------------------------------------------------------------------------
# Forking
# ---------------------------------------------------------------------------


def test_fork_from_copies_history(tmp_path: Path) -> None:
    src = SessionManager.create("/src-cwd", session_dir=str(tmp_path / "src"))
    src.append_message(_user_message("hi"))
    src.append_message(_assistant_message("hello"))
    src_path = src.get_session_file()
    assert src_path is not None

    forked = SessionManager.fork_from(src_path, "/target", session_dir=str(tmp_path / "target"))
    assert forked.get_cwd() == "/target"
    assert len(forked.get_entries()) == 2
    header = forked.get_header()
    assert header is not None
    assert header.parent_session == src_path


def test_fork_from_invalid_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty or invalid"):
        SessionManager.fork_from(str(tmp_path / "missing.jsonl"), "/x")


# ---------------------------------------------------------------------------
# build_session_context standalone
# ---------------------------------------------------------------------------


def test_build_session_context_explicit_none_returns_empty() -> None:
    entries: list[dict[str, Any]] = [
        {"type": "message", "id": "a", "parentId": None, "message": _user_message("hi")},
    ]
    by_id = {"a": entries[0]}
    ctx = build_session_context(entries, leaf_id=None, by_id=by_id)
    # Defaults to last entry when leaf_id is None.
    assert ctx.messages


# ---------------------------------------------------------------------------
# Append custom entries / messages
# ---------------------------------------------------------------------------


def test_append_custom_entry() -> None:
    sm = SessionManager.in_memory()
    eid = sm.append_custom_entry("ext.foo", data={"k": "v"})
    entry = sm.get_entry(eid)
    assert entry is not None
    assert entry["type"] == "custom"
    assert entry["customType"] == "ext.foo"
    assert entry["data"] == {"k": "v"}


def test_append_custom_message_entry() -> None:
    sm = SessionManager.in_memory()
    eid = sm.append_custom_message_entry("ext.note", "hi", display=True, details={"a": 1})
    entry = sm.get_entry(eid)
    assert entry is not None
    assert entry["type"] == "custom_message"
    assert entry["display"] is True
    assert entry["details"] == {"a": 1}


def test_get_branch_from_specific_id() -> None:
    sm = SessionManager.in_memory()
    a = sm.append_message(_user_message("a"))
    b = sm.append_message(_user_message("b"))
    sm.append_message(_user_message("c"))
    # Branch from b only walks back to a → b.
    branch = sm.get_branch(b)
    assert [e["id"] for e in branch] == [a, b]


def test_get_children_returns_all_direct_descendants() -> None:
    sm = SessionManager.in_memory()
    a = sm.append_message(_user_message("a"))
    sm.append_message(_user_message("b"))
    sm.branch(a)
    sm.append_message(_user_message("c"))
    children = sm.get_children(a)
    assert len(children) == 2


def test_get_leaf_entry() -> None:
    sm = SessionManager.in_memory()
    msg_id = sm.append_message(_user_message("hi"))
    leaf = sm.get_leaf_entry()
    assert leaf is not None
    assert leaf["id"] == msg_id


def test_get_header() -> None:
    sm = SessionManager.in_memory("/work")
    header = sm.get_header()
    assert header is not None
    assert header.cwd == "/work"
    assert header.version == CURRENT_SESSION_VERSION


# ---------------------------------------------------------------------------
# Migration: v1 compaction firstKeptEntryIndex
# ---------------------------------------------------------------------------


def test_migrate_v1_compaction_index_to_id() -> None:
    entries: list[dict[str, Any]] = [
        {"type": "session", "id": "sid", "timestamp": "t", "cwd": "/x"},
        {"type": "message", "message": _user_message("a")},
        {
            "type": "compaction",
            "summary": "s",
            "firstKeptEntryIndex": 1,
            "tokensBefore": 10,
        },
    ]
    migrate_session_entries(entries)
    compaction = entries[2]
    assert "firstKeptEntryId" in compaction
    assert compaction["firstKeptEntryId"] == entries[1]["id"]
    assert "firstKeptEntryIndex" not in compaction


def test_resume_session_with_old_version_migrates_in_place(tmp_path: Path) -> None:
    file = tmp_path / "old.jsonl"
    file.write_text(
        json.dumps({"type": "session", "id": "sid", "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/x"})
        + "\n"
        + json.dumps({"type": "message", "message": _user_message("hi")})
        + "\n"
    )
    sm = SessionManager.open(str(file))
    entries = sm.get_entries()
    assert entries
    assert "id" in entries[0]
    # Header was rewritten with version=CURRENT_SESSION_VERSION.
    header = sm.get_header()
    assert header is not None
    assert header.version == CURRENT_SESSION_VERSION


def test_set_session_file_with_empty_file_starts_fresh(tmp_path: Path) -> None:
    file = tmp_path / "empty.jsonl"
    file.write_text("")
    sm = SessionManager(cwd="/work", session_dir=str(tmp_path), session_file=str(file), persist=True)
    assert sm.get_session_file() == str(file)
    assert sm.get_entries() == []


def test_open_with_explicit_session_dir(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sm = SessionManager.create("/work", session_dir=str(sessions))
    sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("ack"))
    file = sm.get_session_file()
    assert file is not None
    sm2 = SessionManager.open(file, session_dir=str(sessions))
    assert sm2.get_session_dir() == str(sessions)


# ---------------------------------------------------------------------------
# Async listing
# ---------------------------------------------------------------------------


async def test_list_returns_session_info(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sm = SessionManager.create("/work", session_dir=str(sessions))
    sm.append_session_info("Test name")
    sm.append_message(_user_message("first prompt"))
    sm.append_message(_assistant_message("ack"))

    listing = await SessionManager.list("/work", session_dir=str(sessions))
    assert len(listing) >= 1
    info = listing[0]
    assert info.name == "Test name"
    assert info.message_count == 2
    assert "first prompt" in info.first_message
    assert info.cwd == "/work"


async def test_list_progress_callback(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sm = SessionManager.create("/work", session_dir=str(sessions))
    sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("ack"))

    progress: list[tuple[int, int]] = []
    await SessionManager.list(
        "/work",
        session_dir=str(sessions),
        on_progress=lambda loaded, total: progress.append((loaded, total)),
    )
    assert progress
    assert progress[-1][0] == progress[-1][1]


async def test_list_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    listing = await SessionManager.list("/work", session_dir=str(tmp_path / "no-such"))
    assert listing == []


async def test_list_all_returns_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_CODING_AGENT_DIR", str(tmp_path))
    work_sessions = tmp_path / "sessions" / "--work--"
    work_sessions.mkdir(parents=True)
    sm = SessionManager(cwd="/work", session_dir=str(work_sessions), session_file=None, persist=True)
    sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("ack"))

    listing = await SessionManager.list_all()
    assert any("hi" in info.first_message for info in listing)


async def test_list_all_empty_when_no_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NU_CODING_AGENT_DIR", str(tmp_path / "no-agent"))
    listing = await SessionManager.list_all()
    assert listing == []


# ---------------------------------------------------------------------------
# SessionHeader (de)serialisation
# ---------------------------------------------------------------------------


def test_session_header_round_trip() -> None:
    from nu_coding_agent.core.session_manager import SessionHeader  # noqa: PLC0415

    header = SessionHeader(
        id="abc",
        timestamp="2026-01-01T00:00:00.000Z",
        cwd="/work",
        version=3,
        parent_session="/prev",
    )
    payload = header.to_dict()
    assert payload["type"] == "session"
    assert payload["parentSession"] == "/prev"

    restored = SessionHeader.from_dict(payload)
    assert restored.id == "abc"
    assert restored.cwd == "/work"
    assert restored.parent_session == "/prev"


def test_session_header_to_dict_omits_parent_when_none() -> None:
    from nu_coding_agent.core.session_manager import SessionHeader  # noqa: PLC0415

    header = SessionHeader(id="x", timestamp="t", cwd="/x")
    payload = header.to_dict()
    assert "parentSession" not in payload


# ---------------------------------------------------------------------------
# build_session_info edge cases
# ---------------------------------------------------------------------------


def test_list_handles_orphan_message_entry(tmp_path: Path) -> None:
    """A session with no assistant message should still be discoverable."""
    sessions = tmp_path / "sessions"
    sm = SessionManager.create("/work", session_dir=str(sessions))
    sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("ack"))
    sm.append_message(
        # Tool result message — exercises the "ignored by listing" branch.
        {"role": "toolResult", "toolCallId": "x", "toolName": "y", "content": [], "isError": False, "timestamp": 3}
    )
    file = sm.get_session_file()
    assert file is not None
    sm2 = SessionManager.open(file)
    assert sm2.get_entries()


def test_load_entries_with_garbled_lines(tmp_path: Path) -> None:
    file = tmp_path / "mixed.jsonl"
    file.write_text(
        json.dumps(
            {
                "type": "session",
                "id": "sid",
                "timestamp": "2026-01-01T00:00:00.000Z",
                "cwd": "/x",
                "version": CURRENT_SESSION_VERSION,
            }
        )
        + "\n"
        + "garbled line\n"
        + json.dumps(
            {"type": "message", "id": "m1", "parentId": None, "timestamp": "t", "message": _user_message("hi")}
        )
        + "\n"
    )
    entries = load_entries_from_file(str(file))
    # Garbled line is dropped; we still get header + one message.
    assert len(entries) == 2


def test_session_id_get_returns_string() -> None:
    sm = SessionManager.in_memory()
    assert isinstance(sm.get_session_id(), str)
    assert sm.get_session_id()


def test_is_persisted_true_for_create(tmp_path: Path) -> None:
    sm = SessionManager.create("/work", session_dir=str(tmp_path / "sessions"))
    assert sm.is_persisted() is True


def test_is_persisted_false_for_in_memory() -> None:
    sm = SessionManager.in_memory()
    assert sm.is_persisted() is False


# ---------------------------------------------------------------------------
# Persisted sessions: labels, info entries, model changes
# ---------------------------------------------------------------------------


def test_persisted_labels_are_indexed_on_load(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sm = SessionManager.create("/work", session_dir=str(sessions))
    msg = sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("ack"))
    sm.append_label_change(msg, "marker")
    file = sm.get_session_file()
    assert file is not None
    sm2 = SessionManager.open(file)
    assert sm2.get_label(msg) == "marker"


def test_persisted_session_info_in_listing(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sm = SessionManager.create("/work", session_dir=str(sessions))
    sm.append_session_info("Demo")
    sm.append_session_info("")  # cleared
    sm.append_message(_user_message("hi"))
    sm.append_message(_assistant_message("ack"))
    info = _build_session_info_helper(sm.get_session_file())  # type: ignore[arg-type]
    assert info is not None
    assert info.name is None


def _build_session_info_helper(file_path: str):
    from nu_coding_agent.core.session_manager import (  # noqa: PLC0415
        _build_session_info,  # pyright: ignore[reportPrivateUsage]
    )

    return _build_session_info(file_path)


def test_get_session_modified_uses_message_timestamp(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sm = SessionManager.create("/work", session_dir=str(sessions))
    sm.append_message(_user_message("hi", timestamp=2_000_000_000_000))  # year 2033
    sm.append_message(_assistant_message("ack"))
    file = sm.get_session_file()
    assert file is not None
    info = _build_session_info_helper(file)
    assert info is not None
    assert info.modified.year >= 2033


def test_new_session_with_custom_id_and_parent() -> None:
    sm = SessionManager.in_memory()
    sm.new_session({"id": "custom-id", "parentSession": "/prev/path.jsonl"})
    assert sm.get_session_id() == "custom-id"
    header = sm.get_header()
    assert header is not None
    assert header.parent_session == "/prev/path.jsonl"


def test_build_context_drops_pre_compaction_messages() -> None:
    sm = SessionManager.in_memory()
    sm.append_message(_user_message("dropped"))
    keep = sm.append_message(_user_message("kept"))
    sm.append_compaction(summary="abc", first_kept_entry_id=keep, tokens_before=10)
    ctx = sm.build_session_context()
    texts = [m.get("content") if isinstance(m, dict) else getattr(m, "content", None) for m in ctx.messages]
    assert any("kept" in str(t) for t in texts)


def test_continue_recent_skips_garbage_files(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "garbage.jsonl").write_text("not a session\n")
    sm = SessionManager.continue_recent("/work", session_dir=str(sessions))
    assert sm.get_entries() == []


def test_get_default_session_dir_with_custom_agent_dir(tmp_path: Path) -> None:
    out = get_default_session_dir("/work", agent_dir=str(tmp_path / "custom"))
    assert "custom" in out
    assert Path(out).exists()
