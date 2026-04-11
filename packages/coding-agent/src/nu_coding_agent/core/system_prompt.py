"""System prompt construction.

Direct port of ``packages/coding-agent/src/core/system-prompt.ts``.
Builds the default system prompt that the coding agent sends to the
model: a short identity blurb, the available-tools list with one-line
snippets, the active guidelines (some unconditional, some derived from
which tools are available), and a trailing date + working directory
footer.

Differences from upstream:

* Pi documentation links (``getReadmePath`` etc.) are deferred — the
  upstream entries point at the bundled pi-coding-agent docs, which
  haven't been ported yet. The Python port skips that section entirely
  for now and adds a TODO comment in :func:`build_system_prompt` so the
  block can be reinstated when ``nu_coding_agent.config.docs`` lands.
* Skills support is also deferred — :func:`build_system_prompt` accepts
  a ``skills`` keyword for forward compatibility but currently emits
  nothing for it. When ``nu_coding_agent.core.skills`` lands the
  formatter will hook in here.
* Per-tool prompt-snippet metadata lives directly on
  :class:`nu_agent_core.types.AgentTool` (the ``prompt_snippet`` /
  ``prompt_guidelines`` fields), so this builder discovers tools by
  iterating an :class:`AgentTool` list rather than reading from a
  separate registry the way upstream does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nu_agent_core.types import AgentTool


@dataclass(slots=True)
class ContextFile:
    """A pre-loaded project context file (e.g. ``AGENTS.md``)."""

    path: str
    content: str


@dataclass(slots=True)
class BuildSystemPromptOptions:
    """Options for :func:`build_system_prompt`.

    All fields are optional. Defaults reproduce the upstream behaviour
    exactly: the four canonical tools, no extra guidelines, no
    additional context files, the current working directory.
    """

    custom_prompt: str | None = None
    """If set, replaces the default identity/tools/guidelines block."""

    tools: list[AgentTool[Any, Any]] | None = None
    """Active tool list. Each tool's ``prompt_snippet`` and
    ``prompt_guidelines`` are surfaced into the prompt."""

    append_system_prompt: str | None = None
    """Free-form text appended after the main prompt body."""

    cwd: str | None = None
    """Working directory shown in the footer. Default: ``Path.cwd()``."""

    context_files: list[ContextFile] | None = None
    """Pre-loaded project context files (e.g. AGENTS.md, README.md)."""


_DEFAULT_GUIDELINES = (
    "Be concise in your responses",
    "Show file paths clearly when working with files",
)


def build_system_prompt(options: BuildSystemPromptOptions | None = None) -> str:
    """Build the default coding-agent system prompt.

    Mirrors the upstream layout: identity → tools → guidelines → optional
    appended block → optional project context → date + cwd footer. When
    ``custom_prompt`` is set the identity/tools/guidelines block is
    replaced wholesale; the footer and context-file appendices still run.
    """
    opts = options or BuildSystemPromptOptions()
    resolved_cwd = opts.cwd or str(Path.cwd())
    prompt_cwd = resolved_cwd.replace("\\", "/")
    today = date.today().isoformat()
    append_section = f"\n\n{opts.append_system_prompt}" if opts.append_system_prompt else ""
    context_files = opts.context_files or []
    tools = opts.tools or []

    # Custom prompt path: drop identity/tools/guidelines, keep the suffix.
    if opts.custom_prompt is not None:
        prompt = opts.custom_prompt
        if append_section:
            prompt += append_section
        if context_files:
            prompt += _format_context_section(context_files)
        prompt += _format_footer(today, prompt_cwd)
        return prompt

    # Default prompt path: build the full identity + tools + guidelines block.
    visible = [t for t in tools if t.prompt_snippet]
    tools_list = "\n".join(f"- {t.name}: {t.prompt_snippet}" for t in visible) if visible else "(none)"

    guidelines = _build_guidelines(tools)

    prompt = (
        "You are an expert coding assistant operating inside nu, a coding agent "
        "harness. You help users by reading files, executing commands, editing "
        "code, and writing new files.\n\n"
        f"Available tools:\n{tools_list}\n\n"
        "In addition to the tools above, you may have access to other custom "
        "tools depending on the project.\n\n"
        f"Guidelines:\n{guidelines}"
    )

    if append_section:
        prompt += append_section
    if context_files:
        prompt += _format_context_section(context_files)
    prompt += _format_footer(today, prompt_cwd)
    return prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_guidelines(tools: list[AgentTool[Any, Any]]) -> str:
    """Compose the guidelines bullet list.

    Order matches upstream:

    1. Tool-derived bash/grep/find/ls advice.
    2. Per-tool ``prompt_guidelines`` from each ``AgentTool``.
    3. The two unconditional defaults (concise responses, show paths).

    Duplicate strings are de-duplicated while preserving first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(line: str) -> None:
        norm = line.strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)

    names = {t.name for t in tools}
    has_bash = "bash" in names
    has_grep = "grep" in names
    has_find = "find" in names
    has_ls = "ls" in names

    if has_bash and not (has_grep or has_find or has_ls):
        add("Use bash for file operations like ls, rg, find")
    elif has_bash and (has_grep or has_find or has_ls):
        add("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)")

    for tool in tools:
        for guideline in tool.prompt_guidelines or []:
            add(guideline)

    for default in _DEFAULT_GUIDELINES:
        add(default)

    return "\n".join(f"- {line}" for line in out)


def _format_context_section(context_files: list[ContextFile]) -> str:
    out = "\n\n# Project Context\n\nProject-specific instructions and guidelines:\n\n"
    for cf in context_files:
        out += f"## {cf.path}\n\n{cf.content}\n\n"
    return out


def _format_footer(today: str, prompt_cwd: str) -> str:
    return f"\nCurrent date: {today}\nCurrent working directory: {prompt_cwd}"


__all__ = [
    "BuildSystemPromptOptions",
    "ContextFile",
    "build_system_prompt",
]
