"""Tests for nu_coding_agent.core.system_prompt."""

from __future__ import annotations

from typing import Any

from nu_agent_core.types import AgentTool, AgentToolResult
from nu_coding_agent.core.system_prompt import (
    BuildSystemPromptOptions,
    ContextFile,
    build_system_prompt,
)
from nu_coding_agent.core.tools import create_all_tools


def _stub_tool(
    name: str,
    *,
    snippet: str | None = None,
    guidelines: list[str] | None = None,
) -> AgentTool[dict[str, Any], None]:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult[None]:
        return AgentToolResult(content=[], details=None)

    return AgentTool[dict[str, Any], None](
        name=name,
        description="",
        parameters={"type": "object"},
        label=name,
        execute=execute,
        prompt_snippet=snippet,
        prompt_guidelines=guidelines,
    )


# ---------------------------------------------------------------------------
# Default prompt body
# ---------------------------------------------------------------------------


class TestDefaultPrompt:
    def test_identity_blurb(self) -> None:
        prompt = build_system_prompt()
        assert "expert coding assistant" in prompt
        assert "nu" in prompt

    def test_no_tools_renders_none(self) -> None:
        prompt = build_system_prompt()
        assert "(none)" in prompt

    def test_visible_tools_listed(self) -> None:
        opts = BuildSystemPromptOptions(
            tools=[
                _stub_tool("read", snippet="Read file contents"),
                _stub_tool("write", snippet="Create or overwrite files"),
            ],
        )
        prompt = build_system_prompt(opts)
        assert "- read: Read file contents" in prompt
        assert "- write: Create or overwrite files" in prompt

    def test_tools_without_snippet_are_filtered(self) -> None:
        opts = BuildSystemPromptOptions(
            tools=[
                _stub_tool("read", snippet="Read file contents"),
                _stub_tool("internal", snippet=None),
            ],
        )
        prompt = build_system_prompt(opts)
        assert "- read:" in prompt
        assert "- internal" not in prompt


# ---------------------------------------------------------------------------
# Guidelines composition
# ---------------------------------------------------------------------------


class TestGuidelines:
    def test_default_guidelines_always_present(self) -> None:
        prompt = build_system_prompt()
        assert "Be concise in your responses" in prompt
        assert "Show file paths clearly when working with files" in prompt

    def test_bash_only_advice(self) -> None:
        opts = BuildSystemPromptOptions(tools=[_stub_tool("bash", snippet="Execute commands")])
        prompt = build_system_prompt(opts)
        assert "Use bash for file operations" in prompt

    def test_bash_with_grep_advice(self) -> None:
        opts = BuildSystemPromptOptions(
            tools=[
                _stub_tool("bash", snippet="Execute commands"),
                _stub_tool("grep", snippet="Search file contents"),
            ]
        )
        prompt = build_system_prompt(opts)
        assert "Prefer grep/find/ls tools over bash" in prompt
        # And the bash-only advice should NOT appear when grep/find/ls exist.
        assert "Use bash for file operations like ls, rg, find" not in prompt

    def test_per_tool_guidelines_included(self) -> None:
        opts = BuildSystemPromptOptions(
            tools=[
                _stub_tool(
                    "read",
                    snippet="Read",
                    guidelines=["Use read instead of cat or sed."],
                )
            ]
        )
        prompt = build_system_prompt(opts)
        assert "- Use read instead of cat or sed." in prompt

    def test_guidelines_deduplicated(self) -> None:
        opts = BuildSystemPromptOptions(
            tools=[
                _stub_tool("a", snippet="A", guidelines=["dup"]),
                _stub_tool("b", snippet="B", guidelines=["dup", "unique"]),
            ]
        )
        prompt = build_system_prompt(opts)
        # "dup" should appear exactly once.
        assert prompt.count("- dup") == 1
        assert "- unique" in prompt


# ---------------------------------------------------------------------------
# Custom prompt path
# ---------------------------------------------------------------------------


class TestCustomPrompt:
    def test_custom_prompt_replaces_body(self) -> None:
        opts = BuildSystemPromptOptions(custom_prompt="My custom system prompt.")
        prompt = build_system_prompt(opts)
        assert prompt.startswith("My custom system prompt.")
        # Identity / tools / guidelines block should be gone.
        assert "expert coding assistant" not in prompt

    def test_custom_prompt_keeps_footer(self) -> None:
        opts = BuildSystemPromptOptions(custom_prompt="X", cwd="/some/dir")
        prompt = build_system_prompt(opts)
        assert "Current working directory: /some/dir" in prompt
        assert "Current date:" in prompt

    def test_custom_prompt_keeps_append_section(self) -> None:
        opts = BuildSystemPromptOptions(
            custom_prompt="X",
            append_system_prompt="extra rules",
        )
        prompt = build_system_prompt(opts)
        assert "extra rules" in prompt


# ---------------------------------------------------------------------------
# Append + context files + footer
# ---------------------------------------------------------------------------


class TestAppendAndContext:
    def test_append_section(self) -> None:
        opts = BuildSystemPromptOptions(append_system_prompt="ADDITIONAL TEXT")
        prompt = build_system_prompt(opts)
        assert "ADDITIONAL TEXT" in prompt

    def test_context_files_rendered(self) -> None:
        opts = BuildSystemPromptOptions(
            context_files=[
                ContextFile(path="AGENTS.md", content="# Agents\nfollow these rules"),
                ContextFile(path="STYLE.md", content="# Style\nbe terse"),
            ],
        )
        prompt = build_system_prompt(opts)
        assert "# Project Context" in prompt
        assert "## AGENTS.md" in prompt
        assert "follow these rules" in prompt
        assert "## STYLE.md" in prompt
        assert "be terse" in prompt

    def test_footer_contains_cwd_and_date(self) -> None:
        opts = BuildSystemPromptOptions(cwd="/tmp/test")
        prompt = build_system_prompt(opts)
        assert "Current working directory: /tmp/test" in prompt
        assert "Current date:" in prompt

    def test_footer_uses_path_cwd_when_unset(self) -> None:
        prompt = build_system_prompt()
        assert "Current working directory:" in prompt


# ---------------------------------------------------------------------------
# Integration with the seven core tools
# ---------------------------------------------------------------------------


class TestIntegrationWithCoreTools:
    def test_all_seven_tools_appear_in_default_prompt(self) -> None:
        tools = create_all_tools("/tmp")
        opts = BuildSystemPromptOptions(tools=tools)
        prompt = build_system_prompt(opts)
        for name in ("read", "write", "edit", "bash", "ls", "find", "grep"):
            assert f"- {name}:" in prompt, f"missing {name}"

    def test_grep_present_triggers_prefer_grep_advice(self) -> None:
        tools = create_all_tools("/tmp")
        prompt = build_system_prompt(BuildSystemPromptOptions(tools=tools))
        assert "Prefer grep/find/ls tools over bash" in prompt
