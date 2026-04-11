"""Agent skills loader — direct port of ``packages/coding-agent/src/core/skills.ts``.

A "skill" is a markdown file (typically named ``SKILL.md``) with YAML
frontmatter advertising a name + description. Skills live in three
locations, in priority order:

1. ``<agent_dir>/skills/`` (user-level)
2. ``<cwd>/<CONFIG_DIR_NAME>/skills/`` (project-level)
3. Explicit paths passed via ``--skills`` (or settings ``skills`` array)

The discovery rules mirror the upstream's nested gitignore-aware walker
exactly: if a directory contains a ``SKILL.md`` it's treated as a skill
root and not recursed into; otherwise direct ``.md`` children are
loaded and subdirectories are walked. Symlinks are resolved so the same
file imported from two paths only loads once. Name collisions are
reported as :class:`ResourceDiagnostic` entries (winner = first seen).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathspec import GitIgnoreSpec, PathSpec

from nu_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir
from nu_coding_agent.core.diagnostics import ResourceCollision, ResourceDiagnostic
from nu_coding_agent.core.source_info import SourceInfo, create_synthetic_source_info
from nu_coding_agent.utils.frontmatter import parse_frontmatter

_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")
_NAME_RE = re.compile(r"^[a-z0-9-]+$")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source_info: SourceInfo
    disable_model_invocation: bool = False


@dataclass(slots=True)
class LoadSkillsResult:
    skills: list[Skill]
    diagnostics: list[ResourceDiagnostic]


@dataclass(slots=True)
class LoadSkillsFromDirOptions:
    dir: str
    source: str


@dataclass(slots=True)
class LoadSkillsOptions:
    cwd: str | None = None
    agent_dir: str | None = None
    skill_paths: list[str] | None = None
    include_defaults: bool = True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > _MAX_NAME_LENGTH:
        errors.append(f"name exceeds {_MAX_NAME_LENGTH} characters ({len(name)})")
    if not _NAME_RE.match(name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    errors: list[str] = []
    if not description or not description.strip():
        errors.append("description is required")
    elif len(description) > _MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {_MAX_DESCRIPTION_LENGTH} characters ({len(description)})")
    return errors


# ---------------------------------------------------------------------------
# Source-info helper
# ---------------------------------------------------------------------------


def _create_skill_source_info(file_path: str, base_dir: str, source: str) -> SourceInfo:
    if source == "user":
        return create_synthetic_source_info(file_path, source="local", scope="user", base_dir=base_dir)
    if source == "project":
        return create_synthetic_source_info(file_path, source="local", scope="project", base_dir=base_dir)
    if source == "path":
        return create_synthetic_source_info(file_path, source="local", base_dir=base_dir)
    return create_synthetic_source_info(file_path, source=source, base_dir=base_dir)


# ---------------------------------------------------------------------------
# Ignore-file machinery (mirrors the upstream ``ignore`` package usage)
# ---------------------------------------------------------------------------


def _to_posix(p: str) -> str:
    return p.replace(os.sep, "/")


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None
    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


def _add_ignore_rules(spec: PathSpec, dir_path: str, root_dir: str) -> PathSpec:
    rel_dir = os.path.relpath(dir_path, root_dir)
    prefix = "" if rel_dir in ("", ".") else f"{_to_posix(rel_dir)}/"
    new_patterns: list[str] = []
    for filename in _IGNORE_FILE_NAMES:
        ignore_path = Path(dir_path) / filename
        if not ignore_path.exists():
            continue
        try:
            content = ignore_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in content.splitlines():
            prefixed = _prefix_ignore_pattern(line, prefix)
            if prefixed is not None:
                new_patterns.append(prefixed)
    if not new_patterns:
        return spec
    additional = GitIgnoreSpec.from_lines(new_patterns)
    return PathSpec(list(spec.patterns) + list(additional.patterns))


# ---------------------------------------------------------------------------
# Disk loading
# ---------------------------------------------------------------------------


def _load_skill_from_file(file_path: str, source: str) -> tuple[Skill | None, list[ResourceDiagnostic]]:
    diagnostics: list[ResourceDiagnostic] = []
    try:
        raw_content = Path(file_path).read_text(encoding="utf-8")
    except OSError as exc:
        diagnostics.append(ResourceDiagnostic(type="warning", message=str(exc), path=file_path))
        return None, diagnostics

    parsed = parse_frontmatter(raw_content)
    frontmatter: dict[str, Any] = parsed.frontmatter
    skill_dir = str(Path(file_path).parent)
    parent_dir_name = Path(skill_dir).name

    description_raw = frontmatter.get("description")
    description = description_raw if isinstance(description_raw, str) else None
    for error in _validate_description(description):
        diagnostics.append(ResourceDiagnostic(type="warning", message=error, path=file_path))

    name_raw = frontmatter.get("name")
    name = name_raw if isinstance(name_raw, str) and name_raw else parent_dir_name
    for error in _validate_name(name, parent_dir_name):
        diagnostics.append(ResourceDiagnostic(type="warning", message=error, path=file_path))

    if not description or not description.strip():
        return None, diagnostics

    skill = Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir=skill_dir,
        source_info=_create_skill_source_info(file_path, skill_dir, source),
        disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
    )
    return skill, diagnostics


def _load_skills_from_dir_internal(
    dir_path: str,
    source: str,
    include_root_files: bool,
    ignore_spec: PathSpec | None = None,
    root_dir: str | None = None,
) -> LoadSkillsResult:
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []

    path = Path(dir_path)
    if not path.exists():
        return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

    root = root_dir or dir_path
    spec = ignore_spec if ignore_spec is not None else PathSpec([])
    spec = _add_ignore_rules(spec, dir_path, root)

    try:
        entries = list(path.iterdir())
    except OSError:
        return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

    # First pass: SKILL.md → treat as skill root, no further recursion.
    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        try:
            is_file = entry.is_file()
        except OSError:
            continue
        rel_path = _to_posix(os.path.relpath(str(entry), root))
        if not is_file or spec.match_file(rel_path):
            continue
        skill, diags = _load_skill_from_file(str(entry), source)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(diags)
        return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

    # Second pass: descend into subdirs and load .md children.
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.name == "node_modules":
            continue
        full_path = str(entry)
        try:
            is_dir = entry.is_dir()
            is_file = entry.is_file()
        except OSError:
            continue
        rel_path = _to_posix(os.path.relpath(full_path, root))
        ignore_path = f"{rel_path}/" if is_dir else rel_path
        if spec.match_file(ignore_path):
            continue
        if is_dir:
            sub = _load_skills_from_dir_internal(
                full_path, source, include_root_files=False, ignore_spec=spec, root_dir=root
            )
            skills.extend(sub.skills)
            diagnostics.extend(sub.diagnostics)
            continue
        if not is_file or not include_root_files or not entry.name.endswith(".md"):
            continue
        skill, diags = _load_skill_from_file(full_path, source)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(diags)

    return LoadSkillsResult(skills=skills, diagnostics=diagnostics)


def load_skills_from_dir(options: LoadSkillsFromDirOptions) -> LoadSkillsResult:
    """Walk ``options.dir`` for skill files using the upstream's discovery rules."""
    return _load_skills_from_dir_internal(options.dir, options.source, include_root_files=True)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def _normalize_path(input_path: str) -> str:
    trimmed = input_path.strip()
    if trimmed == "~":
        return str(Path.home())
    if trimmed.startswith("~/"):
        return str(Path.home() / trimmed[2:])
    if trimmed.startswith("~"):
        return str(Path.home() / trimmed[1:])
    return trimmed


def _resolve_skill_path(p: str, cwd: str) -> str:
    normalized = _normalize_path(p)
    path = Path(normalized)
    return str(path) if path.is_absolute() else str((Path(cwd) / normalized).resolve())


def _is_under_path(target: str, root: str) -> bool:
    normalized_root = str(Path(root).resolve())
    if target == normalized_root:
        return True
    sep = "" if normalized_root.endswith(os.sep) else os.sep
    return target.startswith(f"{normalized_root}{sep}")


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def load_skills(options: LoadSkillsOptions | None = None) -> LoadSkillsResult:
    """Load skills from defaults + explicit paths, deduping symlinks and reporting collisions."""
    opts = options or LoadSkillsOptions()
    cwd = opts.cwd or str(Path.cwd())
    agent_dir = opts.agent_dir or get_agent_dir()
    skill_paths = opts.skill_paths or []
    include_defaults = opts.include_defaults

    skill_map: dict[str, Skill] = {}
    real_path_set: set[str] = set()
    all_diagnostics: list[ResourceDiagnostic] = []
    collision_diagnostics: list[ResourceDiagnostic] = []

    def _add(result: LoadSkillsResult) -> None:
        all_diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            try:
                real_path = str(Path(skill.file_path).resolve())
            except OSError:
                real_path = skill.file_path
            if real_path in real_path_set:
                continue
            existing = skill_map.get(skill.name)
            if existing is not None:
                collision_diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.file_path,
                        collision=ResourceCollision(
                            resource_type="skill",
                            name=skill.name,
                            winner_path=existing.file_path,
                            loser_path=skill.file_path,
                        ),
                    )
                )
            else:
                skill_map[skill.name] = skill
                real_path_set.add(real_path)

    user_skills_dir = str(Path(agent_dir) / "skills")
    project_skills_dir = str((Path(cwd) / CONFIG_DIR_NAME / "skills").resolve())

    if include_defaults:
        _add(_load_skills_from_dir_internal(user_skills_dir, "user", include_root_files=True))
        _add(_load_skills_from_dir_internal(project_skills_dir, "project", include_root_files=True))

    def _get_source(resolved_path: str) -> str:
        if not include_defaults:
            if _is_under_path(resolved_path, user_skills_dir):
                return "user"
            if _is_under_path(resolved_path, project_skills_dir):
                return "project"
        return "path"

    for raw_path in skill_paths:
        resolved_path = _resolve_skill_path(raw_path, cwd)
        path = Path(resolved_path)
        if not path.exists():
            all_diagnostics.append(
                ResourceDiagnostic(type="warning", message="skill path does not exist", path=resolved_path)
            )
            continue
        source = _get_source(resolved_path)
        try:
            if path.is_dir():
                _add(_load_skills_from_dir_internal(resolved_path, source, include_root_files=True))
            elif path.is_file() and resolved_path.endswith(".md"):
                skill, diags = _load_skill_from_file(resolved_path, source)
                if skill is not None:
                    _add(LoadSkillsResult(skills=[skill], diagnostics=diags))
                else:
                    all_diagnostics.extend(diags)
            else:
                all_diagnostics.append(
                    ResourceDiagnostic(
                        type="warning",
                        message="skill path is not a markdown file",
                        path=resolved_path,
                    )
                )
        except OSError as exc:
            all_diagnostics.append(ResourceDiagnostic(type="warning", message=str(exc), path=resolved_path))

    return LoadSkillsResult(
        skills=list(skill_map.values()),
        diagnostics=[*all_diagnostics, *collision_diagnostics],
    )


# ---------------------------------------------------------------------------
# System-prompt formatting
# ---------------------------------------------------------------------------


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """Render skills as the XML block embedded into the system prompt.

    Skills with ``disable_model_invocation=True`` are filtered out — they
    can only be triggered explicitly via ``/skill:name``.
    """
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""
    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


__all__ = [
    "LoadSkillsFromDirOptions",
    "LoadSkillsOptions",
    "LoadSkillsResult",
    "Skill",
    "format_skills_for_prompt",
    "load_skills",
    "load_skills_from_dir",
]
