"""Tests for ``nu_coding_agent.initial_message``."""

from __future__ import annotations

from nu_coding_agent.initial_message import build_initial_message


class TestBuildInitialMessage:
    def test_all_none_returns_none(self) -> None:
        assert build_initial_message() is None

    def test_cli_message_only(self) -> None:
        result = build_initial_message(cli_message="hello")
        assert result is not None
        assert result.text == "hello"
        assert result.images == []

    def test_stdin_only(self) -> None:
        result = build_initial_message(stdin_text="piped input")
        assert result is not None
        assert result.text == "piped input"

    def test_file_text_only(self) -> None:
        result = build_initial_message(file_text='<file name="x.py">\ncode\n</file>\n')
        assert result is not None
        assert '<file name="x.py">' in result.text

    def test_all_sources_combined(self) -> None:
        result = build_initial_message(
            stdin_text="stdin part",
            file_text="file part",
            cli_message="cli part",
        )
        assert result is not None
        assert "stdin part" in result.text
        assert "file part" in result.text
        assert "cli part" in result.text

    def test_order_is_stdin_file_cli(self) -> None:
        result = build_initial_message(
            stdin_text="A",
            file_text="B",
            cli_message="C",
        )
        assert result is not None
        # Parts joined with "" — order: stdin, file, cli
        assert result.text == "ABC"

    def test_images_passed_through(self) -> None:
        imgs = [{"type": "image", "data": "abc", "mime_type": "image/png"}]
        result = build_initial_message(cli_message="hi", images=imgs)
        assert result is not None
        assert len(result.images) == 1
        assert result.images[0]["data"] == "abc"

    def test_images_only_no_text(self) -> None:
        imgs = [{"type": "image", "data": "abc", "mime_type": "image/png"}]
        result = build_initial_message(images=imgs)
        assert result is not None
        assert result.text is None
        assert len(result.images) == 1

    def test_empty_images_list(self) -> None:
        result = build_initial_message(cli_message="hi", images=[])
        assert result is not None
        assert result.images == []

    def test_empty_string_sources(self) -> None:
        # stdin_text="" is still "not None", so it gets appended
        result = build_initial_message(stdin_text="")
        assert result is not None
        assert result.text == ""

    def test_file_text_empty_string_skipped(self) -> None:
        # file_text="" is falsy -> skipped
        result = build_initial_message(file_text="", cli_message="msg")
        assert result is not None
        assert result.text == "msg"

    def test_cli_message_empty_string_skipped(self) -> None:
        result = build_initial_message(cli_message="", stdin_text="x")
        assert result is not None
        assert result.text == "x"
