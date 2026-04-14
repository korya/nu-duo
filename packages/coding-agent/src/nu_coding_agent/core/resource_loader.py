"""Unified resource loader — Python port of ``packages/coding-agent/src/core/resource-loader.ts`` (908 LOC).

Loads and manages extensions, skills, prompts, themes, AGENTS.md files,
and custom system prompts from three sources in precedence order:

1. **Project-level** (``{cwd}/.nu/``)
2. **User-level** (``~/.nu/agent/``)
3. **Package-installed** (entry points / additional paths)

Name collisions prefer project > user > package; losers are recorded as
:class:`ResourceDiagnostic` entries so the CLI can display them.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nu_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir
from nu_coding_agent.core.diagnostics import ResourceCollision, ResourceDiagnostic

logger = logging.getLogger(__name__)

_CONTEXT_FILE_NAMES = ("AGENTS.md", "CLAUDE.md")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ContextFile:
    """An AGENTS.md / CLAUDE.md discovered by walking up the directory tree."""

    path: str
    content: str


@dataclass(slots=True)
class ThemeInfo:
    """Lightweight descriptor for a discovered theme JSON file."""

    name: str
    file_path: str
    source: str  # "project", "user", or "package"


@dataclass(slots=True)
class ResourceLoaderOptions:
    """Knobs accepted by :class:`ResourceLoader`."""

    cwd: str | None = None
    agent_dir: str | None = None
    additional_skill_paths: list[str] = field(default_factory=list)
    additional_prompt_paths: list[str] = field(default_factory=list)
    additional_theme_paths: list[str] = field(default_factory=list)
    additional_extension_paths: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    no_skills: bool = False
    no_prompts: bool = False
    no_themes: bool = False
    no_extensions: bool = False


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _expand_tilde(p: str) -> str:
    trimmed = p.strip()
    if trimmed == "~":
        return str(Path.home())
    if trimmed.startswith("~/"):
        return str(Path.home() / trimmed[2:])
    if trimmed.startswith("~"):
        return str(Path.home() / trimmed[1:])
    return trimmed


def _resolve_path(p: str, cwd: str) -> str:
    expanded = _expand_tilde(p)
    path = Path(expanded)
    return str(path) if path.is_absolute() else str((Path(cwd) / expanded).resolve())


def _is_under(target: str, root: str) -> bool:  # pyright: ignore[reportUnusedFunction]
    nr = str(Path(root).resolve())
    if target == nr:
        return True
    sep = "" if nr.endswith(os.sep) else os.sep
    return target.startswith(f"{nr}{sep}")


def _merge_paths(primary: list[str], additional: list[str], cwd: str) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for p in [*primary, *additional]:
        resolved = _resolve_path(p, cwd)
        if resolved not in seen:
            seen.add(resolved)
            merged.append(resolved)
    return merged


# ---------------------------------------------------------------------------
# Context file (AGENTS.md) discovery
# ---------------------------------------------------------------------------


def _load_context_file_from_dir(directory: str) -> ContextFile | None:
    for name in _CONTEXT_FILE_NAMES:
        file_path = os.path.join(directory, name)
        if os.path.isfile(file_path):
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                return ContextFile(path=file_path, content=content)
            except OSError as exc:
                logger.warning("Could not read %s: %s", file_path, exc)
    return None


def load_project_context_files(
    cwd: str | None = None,
    agent_dir: str | None = None,
) -> list[ContextFile]:
    """Walk from *cwd* up to ``/``, collecting AGENTS.md / CLAUDE.md files.

    The user-level agent dir is checked first, then ancestor directories
    from root down to *cwd* (root-first ordering matches the TS upstream).
    """
    resolved_cwd = cwd or str(Path.cwd())
    resolved_agent_dir = agent_dir or get_agent_dir()

    files: list[ContextFile] = []
    seen: set[str] = set()

    # 1) User-level agent dir
    global_ctx = _load_context_file_from_dir(resolved_agent_dir)
    if global_ctx is not None:
        files.append(global_ctx)
        seen.add(global_ctx.path)

    # 2) Walk up from cwd to root, then reverse so root comes first
    ancestors: list[ContextFile] = []
    current = resolved_cwd
    root = os.path.abspath(os.sep)

    while True:
        ctx = _load_context_file_from_dir(current)
        if ctx is not None and ctx.path not in seen:
            ancestors.append(ctx)
            seen.add(ctx.path)
        if current == root:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    ancestors.reverse()
    files.extend(ancestors)
    return files


# ---------------------------------------------------------------------------
# Theme loading helpers
# ---------------------------------------------------------------------------


def _load_theme_from_file(file_path: str, source: str) -> tuple[ThemeInfo | None, ResourceDiagnostic | None]:
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, ResourceDiagnostic(type="warning", message=str(exc), path=file_path)
    name = data.get("name") if isinstance(data, dict) else None
    if not name:
        name = Path(file_path).stem
    return ThemeInfo(name=name, file_path=file_path, source=source), None


def _load_themes_from_dir(directory: str, source: str) -> tuple[list[ThemeInfo], list[ResourceDiagnostic]]:
    themes: list[ThemeInfo] = []
    diagnostics: list[ResourceDiagnostic] = []
    if not os.path.isdir(directory):
        return themes, diagnostics
    try:
        entries = list(Path(directory).iterdir())
    except OSError as exc:
        diagnostics.append(ResourceDiagnostic(type="warning", message=str(exc), path=directory))
        return themes, diagnostics
    for entry in entries:
        if entry.suffix != ".json":
            continue
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        theme, diag = _load_theme_from_file(str(entry), source)
        if theme is not None:
            themes.append(theme)
        if diag is not None:
            diagnostics.append(diag)
    return themes, diagnostics


# ---------------------------------------------------------------------------
# System / append system prompt discovery
# ---------------------------------------------------------------------------


def _discover_prompt_file(cwd: str, agent_dir: str, filename: str) -> str | None:
    """Look for *filename* in project dir first, then user-level agent dir."""
    project_path = os.path.join(cwd, CONFIG_DIR_NAME, filename)
    if os.path.isfile(project_path):
        return project_path
    global_path = os.path.join(agent_dir, filename)
    if os.path.isfile(global_path):
        return global_path
    return None


def _resolve_prompt_input(source: str | None) -> str | None:
    """If *source* is a file path, read and return its content; otherwise return as-is."""
    if not source:
        return None
    if os.path.isfile(source):
        try:
            return Path(source).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read prompt file %s: %s", source, exc)
            return source
    return source


# ---------------------------------------------------------------------------
# ResourceLoader
# ---------------------------------------------------------------------------


class ResourceLoader:
    """Loads and manages extensions, skills, prompts, and themes from configured sources.

    Mirrors the TS ``DefaultResourceLoader``. Call :meth:`reload` after
    construction to perform the initial disk scan; the ``get_*`` accessors
    then return the cached results.
    """

    def __init__(self, options: ResourceLoaderOptions | None = None) -> None:
        opts = options or ResourceLoaderOptions()
        self._cwd = opts.cwd or str(Path.cwd())
        self._agent_dir = opts.agent_dir or get_agent_dir()
        self._additional_skill_paths = list(opts.additional_skill_paths)
        self._additional_prompt_paths = list(opts.additional_prompt_paths)
        self._additional_theme_paths = list(opts.additional_theme_paths)
        self._additional_extension_paths = list(opts.additional_extension_paths)
        self._system_prompt_source = opts.system_prompt
        self._append_system_prompt_source = opts.append_system_prompt
        self._no_skills = opts.no_skills
        self._no_prompts = opts.no_prompts
        self._no_themes = opts.no_themes
        self._no_extensions = opts.no_extensions

        # Cached results (populated by reload)
        self._skills: list[Any] = []
        self._skill_diagnostics: list[ResourceDiagnostic] = []
        self._prompts: list[Any] = []
        self._prompt_diagnostics: list[ResourceDiagnostic] = []
        self._themes: list[ThemeInfo] = []
        self._theme_diagnostics: list[ResourceDiagnostic] = []
        self._extension_paths: list[str] = []
        self._agents_files: list[ContextFile] = []
        self._system_prompt: str | None = None
        self._append_system_prompt: list[str] = []

    # -- Directory shortcuts ------------------------------------------------

    @property
    def _project_dir(self) -> str:
        return os.path.join(self._cwd, CONFIG_DIR_NAME)

    def _project_sub(self, name: str) -> str:
        return os.path.join(self._project_dir, name)

    def _agent_sub(self, name: str) -> str:
        return os.path.join(self._agent_dir, name)

    # -- Public accessors ---------------------------------------------------

    def get_extension_paths(self) -> list[str]:
        """Return list of discovered extension file paths."""
        return list(self._extension_paths)

    def get_skills(self) -> tuple[list[Any], list[ResourceDiagnostic]]:
        """Return (skills, diagnostics)."""
        return list(self._skills), list(self._skill_diagnostics)

    def get_prompts(self) -> tuple[list[Any], list[ResourceDiagnostic]]:
        """Return (prompts, diagnostics)."""
        return list(self._prompts), list(self._prompt_diagnostics)

    def get_themes(self) -> tuple[list[ThemeInfo], list[ResourceDiagnostic]]:
        """Return (themes, diagnostics)."""
        return list(self._themes), list(self._theme_diagnostics)

    def get_agents_files(self) -> list[ContextFile]:
        """Return discovered AGENTS.md / CLAUDE.md files (root-first)."""
        return list(self._agents_files)

    def get_system_prompt(self) -> str | None:
        """Return custom system prompt content, or ``None``."""
        return self._system_prompt

    def get_append_system_prompt(self) -> list[str]:
        """Return append-system-prompt fragments."""
        return list(self._append_system_prompt)

    def get_diagnostics(self) -> list[ResourceDiagnostic]:
        """Aggregate diagnostics from all resource types."""
        return [
            *self._skill_diagnostics,
            *self._prompt_diagnostics,
            *self._theme_diagnostics,
        ]

    # -- Reload -------------------------------------------------------------

    def reload(self) -> None:
        """(Re-)scan all configured sources and cache the results."""
        self._load_extensions()
        self._load_skills()
        self._load_prompts()
        self._load_themes()
        self._load_agents_files()
        self._load_system_prompts()

    # -- Private loaders ----------------------------------------------------

    def _load_extensions(self) -> None:
        """Discover extension .py files from project, user, and additional paths."""
        if self._no_extensions:
            self._extension_paths = []
            return
        paths: list[str] = []
        for directory in (self._project_sub("extensions"), self._agent_sub("extensions")):
            if os.path.isdir(directory):
                try:
                    for entry in Path(directory).iterdir():
                        if entry.suffix == ".py" and entry.is_file():
                            paths.append(str(entry))
                except OSError:
                    pass
        paths = _merge_paths(paths, self._additional_extension_paths, self._cwd)
        self._extension_paths = paths

    def _load_skills(self) -> None:
        """Load skills via the existing :func:`skills.load_skills` machinery."""
        if self._no_skills and not self._additional_skill_paths:
            self._skills = []
            self._skill_diagnostics = []
            return

        from nu_coding_agent.core.skills import LoadSkillsOptions, load_skills  # noqa: PLC0415

        skill_paths = _merge_paths([], self._additional_skill_paths, self._cwd)
        result = load_skills(
            LoadSkillsOptions(
                cwd=self._cwd,
                agent_dir=self._agent_dir,
                skill_paths=skill_paths or None,
                include_defaults=not self._no_skills,
            )
        )
        self._skills = result.skills
        self._skill_diagnostics = result.diagnostics

        # Report missing additional paths
        for p in self._additional_skill_paths:
            resolved = _resolve_path(p, self._cwd)
            if not os.path.exists(resolved) and not any(d.path == resolved for d in self._skill_diagnostics):
                self._skill_diagnostics.append(
                    ResourceDiagnostic(type="error", message="Skill path does not exist", path=resolved)
                )

    def _load_prompts(self) -> None:
        """Load prompt templates via the existing :func:`prompt_templates.load_prompt_templates`."""
        if self._no_prompts and not self._additional_prompt_paths:
            self._prompts = []
            self._prompt_diagnostics = []
            return

        from nu_coding_agent.core.prompt_templates import (  # noqa: PLC0415
            LoadPromptTemplatesOptions,
            load_prompt_templates,
        )

        prompt_paths = _merge_paths([], self._additional_prompt_paths, self._cwd)
        prompts = load_prompt_templates(
            LoadPromptTemplatesOptions(
                cwd=self._cwd,
                agent_dir=self._agent_dir,
                prompt_paths=prompt_paths or None,
                include_defaults=not self._no_prompts,
            )
        )

        # Dedupe by name (first wins = higher-precedence source)
        seen: dict[str, Any] = {}
        diagnostics: list[ResourceDiagnostic] = []
        for prompt in prompts:
            existing = seen.get(prompt.name)
            if existing is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "/{prompt.name}" collision',
                        path=prompt.file_path,
                        collision=ResourceCollision(
                            resource_type="prompt",
                            name=prompt.name,
                            winner_path=existing.file_path,
                            loser_path=prompt.file_path,
                        ),
                    )
                )
            else:
                seen[prompt.name] = prompt

        self._prompts = list(seen.values())
        self._prompt_diagnostics = diagnostics

        # Report missing additional paths
        for p in self._additional_prompt_paths:
            resolved = _resolve_path(p, self._cwd)
            if not os.path.exists(resolved) and not any(d.path == resolved for d in self._prompt_diagnostics):
                self._prompt_diagnostics.append(
                    ResourceDiagnostic(type="error", message="Prompt template path does not exist", path=resolved)
                )

    def _load_themes(self) -> None:
        """Load theme JSON files from project, user, and additional paths."""
        if self._no_themes and not self._additional_theme_paths:
            self._themes = []
            self._theme_diagnostics = []
            return

        all_themes: list[ThemeInfo] = []
        all_diagnostics: list[ResourceDiagnostic] = []

        if not self._no_themes:
            # Project-level themes (highest precedence)
            themes, diags = _load_themes_from_dir(self._project_sub("themes"), "project")
            all_themes.extend(themes)
            all_diagnostics.extend(diags)

            # User-level themes
            themes, diags = _load_themes_from_dir(self._agent_sub("themes"), "user")
            all_themes.extend(themes)
            all_diagnostics.extend(diags)

        # Additional paths
        for raw_path in self._additional_theme_paths:
            resolved = _resolve_path(raw_path, self._cwd)
            if not os.path.exists(resolved):
                all_diagnostics.append(
                    ResourceDiagnostic(type="warning", message="theme path does not exist", path=resolved)
                )
                continue
            try:
                if os.path.isdir(resolved):
                    themes, diags = _load_themes_from_dir(resolved, "package")
                    all_themes.extend(themes)
                    all_diagnostics.extend(diags)
                elif os.path.isfile(resolved) and resolved.endswith(".json"):
                    theme, diag = _load_theme_from_file(resolved, "package")
                    if theme is not None:
                        all_themes.append(theme)
                    if diag is not None:
                        all_diagnostics.append(diag)
                else:
                    all_diagnostics.append(
                        ResourceDiagnostic(type="warning", message="theme path is not a JSON file", path=resolved)
                    )
            except OSError as exc:
                all_diagnostics.append(ResourceDiagnostic(type="warning", message=str(exc), path=resolved))

        # Dedupe by name (first wins)
        seen: dict[str, ThemeInfo] = {}
        for theme in all_themes:
            existing = seen.get(theme.name)
            if existing is not None:
                all_diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "{theme.name}" collision',
                        path=theme.file_path,
                        collision=ResourceCollision(
                            resource_type="theme",
                            name=theme.name,
                            winner_path=existing.file_path,
                            loser_path=theme.file_path,
                        ),
                    )
                )
            else:
                seen[theme.name] = theme

        self._themes = list(seen.values())
        self._theme_diagnostics = all_diagnostics

    def _load_agents_files(self) -> None:
        self._agents_files = load_project_context_files(cwd=self._cwd, agent_dir=self._agent_dir)

    def _load_system_prompts(self) -> None:
        # System prompt
        source = self._system_prompt_source or _discover_prompt_file(self._cwd, self._agent_dir, "SYSTEM.md")
        self._system_prompt = _resolve_prompt_input(source)

        # Append system prompt
        append_source = self._append_system_prompt_source or _discover_prompt_file(
            self._cwd, self._agent_dir, "APPEND_SYSTEM.md"
        )
        resolved_append = _resolve_prompt_input(append_source)
        self._append_system_prompt = [resolved_append] if resolved_append else []


__all__ = [
    "ContextFile",
    "ResourceLoader",
    "ResourceLoaderOptions",
    "ThemeInfo",
    "load_project_context_files",
]
