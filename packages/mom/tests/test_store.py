"""Tests for :class:`nu_mom.store.ChannelStore`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nu_mom.store import Attachment, ChannelStore, LoggedMessage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> ChannelStore:
    """Return a fresh ChannelStore backed by a temp directory."""
    return ChannelStore(working_dir=str(tmp_path), bot_token="xoxb-fake-token")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_working_dir_created(tmp_path: Path) -> None:
    """ChannelStore creates the working directory on construction."""
    new_dir = tmp_path / "mom_data"
    assert not new_dir.exists()
    ChannelStore(working_dir=str(new_dir), bot_token="tok")
    assert new_dir.is_dir()


def test_get_channel_dir_creates_subdirectory(store: ChannelStore, tmp_path: Path) -> None:
    """get_channel_dir returns and creates the per-channel sub-directory."""
    channel_dir = store.get_channel_dir("C12345")
    assert Path(channel_dir).is_dir()
    assert Path(channel_dir).name == "C12345"


@pytest.mark.asyncio
async def test_log_user_message_appends_jsonl(store: ChannelStore, tmp_path: Path) -> None:
    """log_user_message appends a valid JSON line to log.jsonl."""
    channel_id = "C99999"
    msg = LoggedMessage(
        ts="1700000000.000001",
        user="U111",
        text="hello world",
        is_bot=False,
        user_name="alice",
    )

    written = await store.log_user_message(channel_id, msg)
    assert written is True

    log_path = tmp_path / channel_id / "log.jsonl"
    assert log_path.exists()
    line = json.loads(log_path.read_text().strip())
    assert line["user"] == "U111"
    assert line["text"] == "hello world"
    assert line["isBot"] is False
    assert line["userName"] == "alice"


@pytest.mark.asyncio
async def test_log_user_message_deduplication(store: ChannelStore) -> None:
    """Logging the same ts twice returns False on the second attempt."""
    channel_id = "C00001"
    msg = LoggedMessage(
        ts="1700000001.000001",
        user="U222",
        text="duplicate",
        is_bot=False,
    )

    first = await store.log_user_message(channel_id, msg)
    second = await store.log_user_message(channel_id, msg)
    assert first is True
    assert second is False  # duplicate suppressed


@pytest.mark.asyncio
async def test_log_bot_response(store: ChannelStore, tmp_path: Path) -> None:
    """log_bot_response writes an isBot=true entry."""
    channel_id = "D_DM01"
    await store.log_bot_response(channel_id, "Here's your answer", "1700000002.000001")

    log_path = tmp_path / channel_id / "log.jsonl"
    line = json.loads(log_path.read_text().strip())
    assert line["user"] == "bot"
    assert line["isBot"] is True
    assert line["text"] == "Here's your answer"


def test_get_last_timestamp_empty(store: ChannelStore) -> None:
    """get_last_timestamp returns None for a channel with no log."""
    assert store.get_last_timestamp("C_NONEXISTENT") is None


@pytest.mark.asyncio
async def test_get_last_timestamp_after_logging(store: ChannelStore, tmp_path: Path) -> None:
    """get_last_timestamp returns the ts of the most recently appended message."""
    channel_id = "C77777"

    for ts_val in ("1700000010.000001", "1700000020.000002", "1700000030.000003"):
        await store.log_user_message(
            channel_id,
            LoggedMessage(ts=ts_val, user="U1", text="msg", is_bot=False),
        )

    # get_last_timestamp reads the *last line*, so it should be the last one appended
    last_ts = store.get_last_timestamp(channel_id)
    assert last_ts == "1700000030.000003"


def test_generate_local_filename(store: ChannelStore) -> None:
    """generate_local_filename produces a sanitised filename with ts prefix."""
    name = store.generate_local_filename("my file (1).png", "1700000000.123456")
    # Should start with the millisecond timestamp
    assert name.startswith("1700000000123")
    # Should contain sanitised filename characters only
    assert " " not in name
    assert "(" not in name
    assert name.endswith(".png")


def test_process_attachments_returns_list(store: ChannelStore) -> None:
    """process_attachments returns Attachment objects for each valid file."""
    files = [
        {"name": "photo.jpg", "url_private": "https://files.slack.com/fake/photo.jpg"},
        {"name": None, "url_private": "https://files.slack.com/fake/noname"},  # no name — skipped
        {"url_private": "https://files.slack.com/fake/no_name_key"},  # missing name key — skipped
    ]
    attachments = store.process_attachments("C12345", files, "1700000000.000001")
    # Only the first file has a valid name
    assert len(attachments) == 1
    assert isinstance(attachments[0], Attachment)
    assert attachments[0].original == "photo.jpg"
    assert "C12345/attachments/" in attachments[0].local


@pytest.mark.asyncio
async def test_multiple_messages_multiple_lines(store: ChannelStore, tmp_path: Path) -> None:
    """Multiple messages produce multiple JSONL lines."""
    channel_id = "C44444"
    for i in range(3):
        await store.log_user_message(
            channel_id,
            LoggedMessage(ts=f"170000000{i}.000001", user="U1", text=f"msg {i}", is_bot=False),
        )

    log_path = tmp_path / channel_id / "log.jsonl"
    lines = [line for line in log_path.read_text().strip().split("\n") if line]
    assert len(lines) == 3
    texts = [json.loads(line)["text"] for line in lines]
    assert texts == ["msg 0", "msg 1", "msg 2"]
