"""Prompt template loader & expander — direct port of ``packages/coding-agent/src/core/prompt-templates.ts``.

A prompt template is a markdown file with optional YAML frontmatter
(``description`` field). At runtime ``/<name> args...`` in the chat
input is replaced with the template body, with bash-style placeholders
(``$1``, ``$@``, ``$ARGUMENTS``, ``${@:N}``, ``${@:N:L}``) substituted
from the supplied args.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nu_coding_agent.config import CONFIG_DIR_NAME, get_prompts_dir
from nu_coding_agent.core.source_info import SourceInfo, create_synthetic_source_info
from nu_coding_agent.utils.frontmatter import parse_frontmatter

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PromptTemplate:
    """A markdown prompt template loaded from disk."""

    name: str
    description: str
    content: str
    source_info: SourceInfo
    file_path: str


@dataclass(slots=True)
class LoadPromptTemplatesOptions:
    """Knobs for :func:`load_prompt_templates` (matches the TS interface)."""

    cwd: str | None = None
    agent_dir: str | None = None
    prompt_paths: list[str] | None = None
    include_defaults: bool = True


# ---------------------------------------------------------------------------
# Argument parsing & substitution
# ---------------------------------------------------------------------------


def parse_command_args(args_string: str) -> list[str]:
    """Split ``args_string`` into bash-style argv (respecting single/double quotes)."""
    args: list[str] = []
    current = ""
    in_quote: str | None = None

    for char in args_string:
        if in_quote is not None:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += char

    if current:
        args.append(current)
    return args


_POSITIONAL_RE = re.compile(r"\$(\d+)")
_SLICE_RE = re.compile(r"\$\{@:(\d+)(?::(\d+))?\}")


def substitute_args(content: str, args: list[str]) -> str:
    """Apply ``$1``, ``$@``, ``$ARGUMENTS``, and ``${@:N[:L]}`` substitutions to ``content``.

    Substitution happens on the *template* string only; user-supplied
    args are not recursively expanded so a value containing ``$1``
    cannot trigger another replacement (matches upstream behaviour).
    """

    def _positional(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return args[index] if 0 <= index < len(args) else ""

    result = _POSITIONAL_RE.sub(_positional, content)

    def _slice(match: re.Match[str]) -> str:
        start = max(0, int(match.group(1)) - 1)
        length_str = match.group(2)
        if length_str is not None:
            length = int(length_str)
            return " ".join(args[start : start + length])
        return " ".join(args[start:])

    result = _SLICE_RE.sub(_slice, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result


# ---------------------------------------------------------------------------
# Disk loading
# ---------------------------------------------------------------------------


def _load_template_from_file(file_path: str, source_info: SourceInfo) -> PromptTemplate | None:
    try:
        raw_content = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return None
    parsed = parse_frontmatter(raw_content)
    name = Path(file_path).stem
    description_raw = parsed.frontmatter.get("description", "")
    description = description_raw if isinstance(description_raw, str) else ""
    if not description:
        for line in parsed.body.split("\n"):
            stripped = line.strip()
            if stripped:
                description = stripped[:60]
                if len(stripped) > 60:
                    description += "..."
                break
    return PromptTemplate(
        name=name,
        description=description,
        content=parsed.body,
        source_info=source_info,
        file_path=file_path,
    )


def _load_templates_from_dir(
    directory: str,
    get_source_info: Callable[[str], SourceInfo],
) -> list[PromptTemplate]:
    templates: list[PromptTemplate] = []
    dir_path = Path(directory)
    if not dir_path.exists():
        return templates
    try:
        entries = list(dir_path.iterdir())
    except OSError:
        return templates
    for entry in entries:
        try:
            is_file = entry.is_file()
        except OSError:
            continue
        if not is_file or entry.suffix != ".md":
            continue
        template = _load_template_from_file(str(entry), get_source_info(str(entry)))
        if template is not None:
            templates.append(template)
    return templates


def _normalize_path(input_path: str) -> str:
    trimmed = input_path.strip()
    if trimmed == "~":
        return str(Path.home())
    if trimmed.startswith("~/"):
        return str(Path.home() / trimmed[2:])
    if trimmed.startswith("~"):
        return str(Path.home() / trimmed[1:])
    return trimmed


def _resolve_prompt_path(p: str, cwd: str) -> str:
    normalized = _normalize_path(p)
    path = Path(normalized)
    return str(path) if path.is_absolute() else str((Path(cwd) / normalized).resolve())


def _is_under_path(target: str, root: str) -> bool:
    normalized_root = str(Path(root).resolve())
    if target == normalized_root:
        return True
    sep = "/" if not normalized_root.endswith("/") else ""
    return target.startswith(f"{normalized_root}{sep}")


def load_prompt_templates(options: LoadPromptTemplatesOptions | None = None) -> list[PromptTemplate]:
    """Discover prompt templates from the global, project, and explicit paths.

    Resolution order matches the TS upstream:

    1. ``<agent_dir>/prompts/`` — user-level templates.
    2. ``<cwd>/<CONFIG_DIR_NAME>/prompts/`` — project-level templates.
    3. Each entry in ``options.prompt_paths`` (file or directory).
    """
    opts = options or LoadPromptTemplatesOptions()
    resolved_cwd = opts.cwd or str(Path.cwd())
    resolved_agent_dir = opts.agent_dir or get_prompts_dir()
    prompt_paths = opts.prompt_paths or []
    include_defaults = opts.include_defaults

    global_prompts_dir = str(Path(opts.agent_dir) / "prompts") if opts.agent_dir else resolved_agent_dir
    project_prompts_dir = str((Path(resolved_cwd) / CONFIG_DIR_NAME / "prompts").resolve())

    def get_source_info(resolved_path: str) -> SourceInfo:
        if _is_under_path(resolved_path, global_prompts_dir):
            return create_synthetic_source_info(
                resolved_path,
                source="local",
                scope="user",
                base_dir=global_prompts_dir,
            )
        if _is_under_path(resolved_path, project_prompts_dir):
            return create_synthetic_source_info(
                resolved_path,
                source="local",
                scope="project",
                base_dir=project_prompts_dir,
            )
        path = Path(resolved_path)
        base_dir = resolved_path if path.is_dir() else str(path.parent)
        return create_synthetic_source_info(resolved_path, source="local", base_dir=base_dir)

    templates: list[PromptTemplate] = []
    if include_defaults:
        templates.extend(_load_templates_from_dir(global_prompts_dir, get_source_info))
        templates.extend(_load_templates_from_dir(project_prompts_dir, get_source_info))

    for raw_path in prompt_paths:
        resolved_path = _resolve_prompt_path(raw_path, resolved_cwd)
        path = Path(resolved_path)
        if not path.exists():
            continue
        try:
            if path.is_dir():
                templates.extend(_load_templates_from_dir(resolved_path, get_source_info))
            elif path.is_file() and resolved_path.endswith(".md"):
                template = _load_template_from_file(resolved_path, get_source_info(resolved_path))
                if template is not None:
                    templates.append(template)
        except OSError:
            continue

    return templates


def expand_prompt_template(text: str, templates: list[PromptTemplate]) -> str:
    """If ``text`` starts with ``/<name>``, replace it with the matching template."""
    if not text.startswith("/"):
        return text
    space_index = text.find(" ")
    template_name = text[1:] if space_index == -1 else text[1:space_index]
    args_string = "" if space_index == -1 else text[space_index + 1 :]
    template = next((t for t in templates if t.name == template_name), None)
    if template is None:
        return text
    args = parse_command_args(args_string)
    return substitute_args(template.content, args)


__all__ = [
    "LoadPromptTemplatesOptions",
    "PromptTemplate",
    "expand_prompt_template",
    "load_prompt_templates",
    "parse_command_args",
    "substitute_args",
]
