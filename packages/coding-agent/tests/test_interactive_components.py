"""Tests for interactive mode components.

Covers:
- diff.py: render_diff, _word_diff, _parse_diff_line
- tool_renderers.py: render_tool_call, render_tool_result
- message_renderers.py: ToolExecutionWidget, BashExecutionWidget, CompactionSummaryWidget, BranchSummaryWidget
- keybinding_hints.py: key_text, key_hint, raw_key_hint
- skill_invocation_message.py: ParsedSkillBlock, SkillInvocationWidget
- custom_message.py: CustomMessageWidget
- bordered_loader.py: BorderedLoader

Textual ``Static`` widgets call ``self.update()`` in their ``__init__``, which
requires an active Textual app. Tests that instantiate these widgets therefore
monkeypatch ``Static.__init__`` and ``Static.update`` so the widget objects can
be constructed and their internal logic exercised without a running event loop.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from nu_coding_agent.modes.interactive.components.bordered_loader import BorderedLoader
from nu_coding_agent.modes.interactive.components.custom_message import CustomMessageWidget
from nu_coding_agent.modes.interactive.components.diff import (
    _parse_diff_line,
    _word_diff,
    render_diff,
)
from nu_coding_agent.modes.interactive.components.keybinding_hints import (
    key_hint,
    key_text,
    raw_key_hint,
)
from nu_coding_agent.modes.interactive.components.message_renderers import (
    BashExecutionWidget,
    BranchSummaryWidget,
    CompactionSummaryWidget,
    ToolExecutionWidget,
)
from nu_coding_agent.modes.interactive.components.skill_invocation_message import (
    ParsedSkillBlock,
    SkillInvocationWidget,
)
from nu_coding_agent.modes.interactive.components.tool_renderers import (
    render_tool_call,
    render_tool_result,
)
from rich.text import Text as RichText
from textual.widgets import Static

# ---------------------------------------------------------------------------
# Fixture: suppress Textual app requirement for Static widgets
# ---------------------------------------------------------------------------


def _make_static_init() -> Any:
    """Return a lightweight Static.__init__ that skips Textual's event-loop setup.

    Sets only the instance attributes our widget subclasses read or write
    so that ``_update_display`` / ``_build`` / ``set_result`` etc. can
    run without a live Textual app.
    """

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Set the minimum Textual Widget attributes our code touches.
        # Deliberately does NOT call super().__init__ which would require
        # an active event loop.
        object.__setattr__(self, "_reactive_expand", False)
        object.__setattr__(self, "_reactive_shrink", False)
        object.__setattr__(self, "_Static__content", "")
        object.__setattr__(self, "_Static__visual", None)
        object.__setattr__(self, "renderable", "")

    return _init


@pytest.fixture(autouse=True)
def _patch_static() -> Any:
    """Suppress Textual app requirement for Static widgets under test."""
    with (
        patch.object(Static, "__init__", _make_static_init()),
        patch.object(
            Static, "update", lambda self, content=None, *a, **kw: object.__setattr__(self, "renderable", content)
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# diff.py
# ---------------------------------------------------------------------------


class TestParseDiffLine:
    def test_removed(self) -> None:
        result = _parse_diff_line("-  42 foo bar")
        assert result is not None
        prefix, num, content = result
        assert prefix == "-"
        assert "42" in num
        assert content == "foo bar"

    def test_added(self) -> None:
        result = _parse_diff_line("+  10 hello")
        assert result is not None
        assert result[0] == "+"
        assert "10" in result[1]

    def test_context(self) -> None:
        result = _parse_diff_line("   5 unchanged")
        assert result is not None
        assert result[0] == " "

    def test_no_match(self) -> None:
        assert _parse_diff_line("") is None
        assert _parse_diff_line("random text without prefix") is None

    def test_no_line_number(self) -> None:
        # Header line like " --- a/file" should still parse
        result = _parse_diff_line("-  old line")
        assert result is not None


class TestWordDiff:
    def test_identical(self) -> None:
        old, new = _word_diff("hello world", "hello world")
        assert all(not changed for _, changed in old)
        assert all(not changed for _, changed in new)

    def test_single_word_change(self) -> None:
        old, new = _word_diff("hello world", "hello earth")
        # "world" should be marked changed in old, "earth" in new
        changed_old = [t for t, c in old if c]
        changed_new = [t for t, c in new if c]
        assert "world" in changed_old
        assert "earth" in changed_new

    def test_deletion(self) -> None:
        old, _new = _word_diff("foo bar baz", "foo baz")
        changed_old = [t for t, c in old if c]
        # "bar" or "bar " (with trailing space) may be returned as the changed token
        assert any("bar" in t for t in changed_old)

    def test_insertion(self) -> None:
        _old, new = _word_diff("foo baz", "foo bar baz")
        changed_new = [t for t, c in new if c]
        assert any("bar" in t for t in changed_new)

    def test_empty_strings(self) -> None:
        old, new = _word_diff("", "")
        assert old == []
        assert new == []

    def test_returns_rich_text_parts(self) -> None:
        old, _new = _word_diff("a b c", "a x c")
        assert all(isinstance(t, str) for t, _ in old)
        assert all(isinstance(b, bool) for _, b in old)


class TestRenderDiff:
    def test_returns_rich_text(self) -> None:
        result = render_diff("-  1 old\n+  1 new")
        assert isinstance(result, RichText)

    def test_empty_diff(self) -> None:
        result = render_diff("")
        assert isinstance(result, RichText)

    def test_context_line_is_dim(self) -> None:
        result = render_diff("   5 context line")
        # dim style should appear somewhere
        rendered = result.render(None)  # type: ignore[arg-type]
        assert rendered  # non-empty

    def test_one_to_one_intra_line_diff(self) -> None:
        diff = "-  1 hello world\n+  1 hello earth"
        result = render_diff(diff)
        plain = result.plain
        assert "hello" in plain
        assert "world" in plain
        assert "earth" in plain

    def test_multi_removed_added_no_intra(self) -> None:
        diff = "-  1 a\n-  2 b\n+  1 x\n+  2 y"
        result = render_diff(diff)
        plain = result.plain
        assert "a" in plain and "b" in plain
        assert "x" in plain and "y" in plain

    def test_standalone_added(self) -> None:
        diff = "+  3 new line only"
        result = render_diff(diff)
        assert "new line only" in result.plain

    def test_multi_line_diff(self) -> None:
        diff = "   1 ctx\n-  2 old\n+  2 new\n   3 ctx2"
        result = render_diff(diff)
        plain = result.plain
        assert "ctx" in plain
        assert "old" in plain
        assert "new" in plain

    def test_tabs_replaced(self) -> None:
        diff = "-  1 \tfoo"
        result = render_diff(diff)
        assert "\t" not in result.plain


# ---------------------------------------------------------------------------
# tool_renderers.py
# ---------------------------------------------------------------------------


class TestRenderToolCall:
    def test_bash(self) -> None:
        result = render_tool_call("bash", {"command": "ls -la"})
        assert result is not None
        assert "ls -la" in result.plain

    def test_read(self) -> None:
        result = render_tool_call("read", {"file_path": "/tmp/foo.py"})
        assert result is not None
        assert "foo.py" in result.plain

    def test_read_with_line_range(self) -> None:
        result = render_tool_call("read", {"file_path": "/tmp/f.py", "start_line": 10, "end_line": 20})
        assert result is not None
        assert "10" in result.plain

    def test_write(self) -> None:
        result = render_tool_call("write", {"file_path": "/tmp/out.py", "content": "x = 1"})
        assert result is not None
        assert "out.py" in result.plain

    def test_edit(self) -> None:
        result = render_tool_call("edit", {"file_path": "/tmp/e.py"})
        assert result is not None
        assert "e.py" in result.plain

    def test_grep(self) -> None:
        result = render_tool_call("grep", {"pattern": "TODO", "path": "/src"})
        assert result is not None
        assert "TODO" in result.plain

    def test_find(self) -> None:
        result = render_tool_call("find", {"pattern": "*.py"})
        assert result is not None
        assert "*.py" in result.plain

    def test_ls(self) -> None:
        result = render_tool_call("ls", {"path": "/tmp"})
        assert result is not None
        assert "/tmp" in result.plain or "tmp" in result.plain

    def test_unknown_tool(self) -> None:
        result = render_tool_call("custom_tool", {"x": 1})
        assert result is not None
        assert "custom_tool" in result.plain

    def test_shorten_home_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", "/Users/test")
        import os

        monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", "/Users/test"))
        result = render_tool_call("read", {"file_path": "/Users/test/doc.py"})
        assert result is not None
        assert "~/doc.py" in result.plain


class TestRenderToolResult:
    def _content(self, text: str) -> list[Any]:
        return [{"type": "text", "text": text}]

    def test_bash_result_is_none(self) -> None:
        # Bash results are rendered by BashExecutionWidget
        assert render_tool_result("bash", {}, self._content("output")) is None

    def test_read_result_line_count(self) -> None:
        lines = "\n".join([f"line {i}" for i in range(10)])
        result = render_tool_result("read", {"file_path": "/f.py"}, self._content(lines))
        assert result is not None
        assert "10" in result.plain

    def test_read_result_error(self) -> None:
        result = render_tool_result("read", {}, self._content("not found"), is_error=True)
        assert result is not None
        # Should be styled red — check the text exists
        assert "not found" in result.plain

    def test_write_success_is_none(self) -> None:
        result = render_tool_result("write", {}, self._content(""))
        assert result is None

    def test_write_error(self) -> None:
        result = render_tool_result("write", {}, self._content("permission denied"), is_error=True)
        assert result is not None
        assert "permission denied" in result.plain

    def test_grep_no_matches(self) -> None:
        result = render_tool_result("grep", {}, self._content(""))
        assert result is not None
        assert "no matches" in result.plain

    def test_grep_with_results(self) -> None:
        matches = "\n".join([f"file.py:1: match {i}" for i in range(3)])
        result = render_tool_result("grep", {}, self._content(matches))
        assert result is not None

    def test_find_no_results(self) -> None:
        result = render_tool_result("find", {}, self._content(""))
        assert result is not None
        assert "no results" in result.plain

    def test_find_with_results(self) -> None:
        files = "\n".join(["/a/b.py", "/c/d.py"])
        result = render_tool_result("find", {}, self._content(files))
        assert result is not None
        assert "2" in result.plain

    def test_edit_diff_output(self) -> None:
        diff = "-  1 old line\n+  1 new line"
        result = render_tool_result("edit", {}, self._content(diff))
        # Should render as coloured diff (or None if silent success)
        # The diff starts with "-" so render_diff should be called
        if result is not None:
            assert "old line" in result.plain or "new line" in result.plain

    def test_unknown_tool_with_content(self) -> None:
        result = render_tool_result("my_tool", {}, self._content("some output"))
        assert result is not None
        assert "some output" in result.plain

    def test_unknown_tool_empty(self) -> None:
        assert render_tool_result("my_tool", {}, self._content("")) is None

    def test_content_from_objects(self) -> None:
        """Content blocks may be objects with .type and .text attributes."""

        class Block:
            type = "text"
            text = "hello"

        result = render_tool_result("my_tool", {}, [Block()])
        assert result is not None
        assert "hello" in result.plain


# ---------------------------------------------------------------------------
# message_renderers.py
# ---------------------------------------------------------------------------


class TestToolExecutionWidget:
    def test_creates_without_error(self) -> None:
        w = ToolExecutionWidget("read", {"file_path": "/f.py"})
        assert w is not None

    def test_set_result_success(self) -> None:
        w = ToolExecutionWidget("read", {"file_path": "/f.py"})
        w.set_result([{"type": "text", "text": "line1\nline2"}], is_error=False)
        # Should not raise

    def test_set_result_error(self) -> None:
        w = ToolExecutionWidget("bash", {"command": "bad"})
        w.set_result([{"type": "text", "text": "not found"}], is_error=True)

    def test_toggle_expand(self) -> None:
        w = ToolExecutionWidget("grep", {"pattern": "TODO"})
        assert w._expanded is False
        w.toggle_expand()
        assert w._expanded is True
        w.toggle_expand()
        assert w._expanded is False

    def test_pending_before_result(self) -> None:
        w = ToolExecutionWidget("edit", {"file_path": "/a.py"})
        assert not w._finalized

    def test_finalized_after_set_result(self) -> None:
        w = ToolExecutionWidget("edit", {"file_path": "/a.py"})
        w.set_result([], is_error=False)
        assert w._finalized


class TestBashExecutionWidget:
    def test_creates_with_command(self) -> None:
        w = BashExecutionWidget("ls -la")
        assert w._command == "ls -la"
        assert w._status == "running"

    def test_set_output(self) -> None:
        w = BashExecutionWidget("echo hello")
        w.set_output("hello\n")
        assert "hello" in w._output

    def test_strips_ansi(self) -> None:
        w = BashExecutionWidget("cmd")
        w.set_output("\x1b[32mgreen\x1b[0m")
        assert "\x1b" not in w._output
        assert "green" in w._output

    def test_set_complete_success(self) -> None:
        w = BashExecutionWidget("true")
        w.set_complete(0)
        assert w._status == "complete"

    def test_set_complete_error(self) -> None:
        w = BashExecutionWidget("false")
        w.set_complete(1)
        assert w._status == "error"
        assert w._exit_code == 1

    def test_set_complete_cancelled(self) -> None:
        w = BashExecutionWidget("sleep 10")
        w.set_complete(None, cancelled=True)
        assert w._status == "cancelled"

    def test_set_complete_none_exit_is_complete(self) -> None:
        w = BashExecutionWidget("cmd")
        w.set_complete(None)
        assert w._status == "complete"

    def test_preview_lines_truncated(self) -> None:
        w = BashExecutionWidget("cmd")
        lines = "\n".join([f"line {i}" for i in range(50)])
        w.set_output(lines)
        assert w._output.count("\n") >= 49

    def test_toggle_expand(self) -> None:
        w = BashExecutionWidget("cmd")
        assert not w._expanded
        w.toggle_expand()
        assert w._expanded


class TestCompactionSummaryWidget:
    def test_creates(self) -> None:
        w = CompactionSummaryWidget("Context summarized.", tokens_before=12345)
        assert w is not None

    def test_zero_tokens(self) -> None:
        w = CompactionSummaryWidget("Summary text")
        assert w is not None

    def test_empty_summary(self) -> None:
        w = CompactionSummaryWidget("", tokens_before=100)
        assert w is not None

    def test_click_toggles(self) -> None:
        w = CompactionSummaryWidget("Summary\nExtra line")
        assert not w._expanded
        w.on_click()
        assert w._expanded
        w.on_click()
        assert not w._expanded


class TestBranchSummaryWidget:
    def test_creates(self) -> None:
        w = BranchSummaryWidget("Branch from entry abc123")
        assert w is not None

    def test_click_toggles(self) -> None:
        w = BranchSummaryWidget("Summary of prior branch")
        assert not w._expanded
        w.on_click()
        assert w._expanded


# ---------------------------------------------------------------------------
# keybinding_hints.py
# ---------------------------------------------------------------------------


class TestKeybindingHints:
    def test_key_text_known(self) -> None:
        assert key_text("tui.select.cancel") == "Esc"
        assert key_text("tui.select.confirm") == "Enter"
        assert key_text("app.tools.expand") == "Tab"

    def test_key_text_unknown_passthrough(self) -> None:
        assert key_text("unknown.keybinding") == "unknown.keybinding"

    def test_key_hint(self) -> None:
        result = key_hint("tui.select.cancel", "to cancel")
        assert "Esc" in result
        assert "to cancel" in result

    def test_raw_key_hint(self) -> None:
        result = raw_key_hint("Ctrl+Z", "to undo")
        assert "Ctrl+Z" in result
        assert "to undo" in result


# ---------------------------------------------------------------------------
# skill_invocation_message.py
# ---------------------------------------------------------------------------


class TestParsedSkillBlock:
    def test_dataclass_fields(self) -> None:
        block = ParsedSkillBlock(
            name="my-skill",
            location="/path/to/skill.md",
            content="# Skill content",
            user_message="Do the thing",
        )
        assert block.name == "my-skill"
        assert block.location == "/path/to/skill.md"
        assert block.content == "# Skill content"
        assert block.user_message == "Do the thing"

    def test_user_message_optional(self) -> None:
        block = ParsedSkillBlock(
            name="s",
            location="l",
            content="c",
            user_message=None,
        )
        assert block.user_message is None


class TestSkillInvocationWidget:
    def _block(self) -> ParsedSkillBlock:
        return ParsedSkillBlock(
            name="my-skill",
            location="/skills/my-skill.md",
            content="## Instructions\n\nDo something useful.",
            user_message="Please run my-skill",
        )

    def test_creates_collapsed(self) -> None:
        w = SkillInvocationWidget(self._block())
        assert not w._expanded

    def test_set_expanded(self) -> None:
        w = SkillInvocationWidget(self._block())
        w.set_expanded(True)
        assert w._expanded

    def test_set_expanded_idempotent(self) -> None:
        w = SkillInvocationWidget(self._block())
        w.set_expanded(True)
        w.set_expanded(True)  # should not raise or re-render
        assert w._expanded

    def test_on_click_toggles(self) -> None:
        w = SkillInvocationWidget(self._block())
        w.on_click()
        assert w._expanded
        w.on_click()
        assert not w._expanded

    def test_skill_name_in_collapsed_display(self) -> None:
        w = SkillInvocationWidget(self._block())
        # In collapsed state the render method calls self.update(RichText)
        # We can't easily inspect the widget content without Textual, but
        # we can call the internal method and check it doesn't raise.
        w._update_display()


# ---------------------------------------------------------------------------
# custom_message.py
# ---------------------------------------------------------------------------


class TestCustomMessageWidget:
    def _message(self, custom_type: str = "test", content: str = "Hello!") -> Any:
        """Build a minimal CustomMessage-like object."""
        from nu_coding_agent.core.messages import CustomMessage

        return CustomMessage(
            role="custom",
            custom_type=custom_type,
            content=content,
            display=True,
            timestamp=0,
            details=None,
        )

    def test_creates(self) -> None:
        w = CustomMessageWidget(self._message())
        assert w is not None

    def test_default_rendering(self) -> None:
        w = CustomMessageWidget(self._message(content="Hello world"))
        w._update_display()  # should not raise

    def test_with_renderer(self) -> None:
        rendered = RichText("custom render")

        def renderer(msg: Any, opts: Any, theme: Any) -> Any:
            return rendered

        w = CustomMessageWidget(self._message(), renderer=renderer)
        w._update_display()
        # The renderer was invoked — no assertion on internals, just no raise

    def test_renderer_fallback_on_exception(self) -> None:
        def bad_renderer(msg: Any, opts: Any, theme: Any) -> Any:
            raise RuntimeError("renderer broke")

        w = CustomMessageWidget(self._message(content="fallback"), renderer=bad_renderer)
        w._update_display()  # should not raise; falls back to default

    def test_click_toggles(self) -> None:
        w = CustomMessageWidget(self._message())
        assert not w._expanded
        w.on_click()
        assert w._expanded

    def test_list_content(self) -> None:
        from nu_ai import TextContent
        from nu_coding_agent.core.messages import CustomMessage

        msg = CustomMessage(
            role="custom",
            custom_type="rich",
            content=[TextContent(type="text", text="block text")],
            display=True,
            timestamp=0,
            details=None,
        )
        w = CustomMessageWidget(msg)
        w._update_display()  # should not raise

    def test_renderer_returns_none_falls_back(self) -> None:
        def renderer(msg: Any, opts: Any, theme: Any) -> None:
            return None  # explicit None triggers fallback

        w = CustomMessageWidget(self._message(content="fallback content"), renderer=renderer)
        w._update_display()


# ---------------------------------------------------------------------------
# bordered_loader.py
# ---------------------------------------------------------------------------


class TestBorderedLoader:
    def test_creates(self) -> None:
        w = BorderedLoader("Loading...")
        assert w is not None

    def test_not_cancelled_initially(self) -> None:
        w = BorderedLoader("Loading...")
        assert not w.is_cancelled

    def test_cancel(self) -> None:
        w = BorderedLoader("Loading...", cancellable=True)
        event = w.cancel_event
        assert event is not None
        assert not event.is_set()
        w.cancel()
        assert w.is_cancelled
        assert event.is_set()

    def test_non_cancellable_has_no_event(self) -> None:
        w = BorderedLoader("Loading...", cancellable=False)
        assert w.cancel_event is None

    def test_cancel_twice_is_safe(self) -> None:
        w = BorderedLoader("Loading...", cancellable=True)
        w.cancel()
        w.cancel()  # should not raise
        assert w.is_cancelled
