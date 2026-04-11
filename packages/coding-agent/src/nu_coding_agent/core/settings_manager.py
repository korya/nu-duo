"""Settings storage & manager — direct port of ``packages/coding-agent/src/core/settings-manager.ts``.

The Python port mirrors the upstream's two-scope design 1:1:

* **Global** settings live in ``<agent_dir>/settings.json`` and contain
  the user's defaults (theme, model, compaction tuning, …).
* **Project** settings live in ``<cwd>/<CONFIG_DIR_NAME>/settings.json``
  and override the global ones field-by-field. Nested objects merge
  recursively, top-level scalars and arrays do a wholesale replace.

Both files are accessed under :mod:`filelock` so concurrent ``nu``
instances can't clobber each other's writes. Modifications made via the
``set_*`` methods are tracked in dedicated "modified" sets so that
:meth:`SettingsManager.flush` only writes the fields the current process
actually touched — fields it didn't touch survive even if another
process raced in and changed them on disk.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from filelock import FileLock, Timeout

from nu_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Settings shape — keys mirror ``Settings`` in the upstream. Stored on disk
# as plain JSON; we keep them in a ``dict[str, Any]`` so the wire format
# stays byte-compatible with the TypeScript implementation.
# ---------------------------------------------------------------------------


type Settings = dict[str, Any]
type SettingsScope = Literal["global", "project"]


@dataclass(slots=True)
class SettingsError:
    scope: SettingsScope
    error: Exception


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------


class SettingsStorage(Protocol):
    """Pluggable on-disk surface used by :class:`SettingsManager`."""

    def with_lock(
        self,
        scope: SettingsScope,
        fn: Callable[[str | None], str | None],
    ) -> None: ...


class FileSettingsStorage:
    """Real on-disk backend with :mod:`filelock`-based exclusion."""

    def __init__(self, cwd: str | None = None, agent_dir: str | None = None) -> None:
        cwd_resolved = cwd or str(Path.cwd())
        agent_dir_resolved = agent_dir or get_agent_dir()
        self._global_path = str(Path(agent_dir_resolved) / "settings.json")
        self._project_path = str(Path(cwd_resolved) / CONFIG_DIR_NAME / "settings.json")

    @property
    def global_path(self) -> str:
        return self._global_path

    @property
    def project_path(self) -> str:
        return self._project_path

    def _path_for(self, scope: SettingsScope) -> str:
        return self._global_path if scope == "global" else self._project_path

    def with_lock(
        self,
        scope: SettingsScope,
        fn: Callable[[str | None], str | None],
    ) -> None:
        path = self._path_for(scope)
        path_obj = Path(path)
        file_exists = path_obj.exists()

        lock = FileLock(f"{path}.lock") if file_exists else None
        try:
            if lock is not None:
                lock.acquire(timeout=10)
            current = path_obj.read_text(encoding="utf-8") if file_exists else None
            next_value = fn(current)
            if next_value is not None:
                if not path_obj.parent.exists():
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                if lock is None:
                    lock = FileLock(f"{path}.lock")
                    lock.acquire(timeout=10)
                path_obj.write_text(next_value, encoding="utf-8")
        except Timeout as exc:
            raise RuntimeError(f"Failed to acquire settings lock: {path}") from exc
        finally:
            if lock is not None and lock.is_locked:
                lock.release()


class InMemorySettingsStorage:
    """Test backend that holds the settings blobs in memory."""

    def __init__(self) -> None:
        self._global: str | None = None
        self._project: str | None = None

    def with_lock(
        self,
        scope: SettingsScope,
        fn: Callable[[str | None], str | None],
    ) -> None:
        current = self._global if scope == "global" else self._project
        next_value = fn(current)
        if next_value is None:
            return
        if scope == "global":
            self._global = next_value
        else:
            self._project = next_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge_settings(base: Settings, overrides: Settings) -> Settings:
    """Recursive merge — overrides win for primitives/arrays, dicts merge."""
    result: Settings = dict(base)
    for key, override_value in overrides.items():
        if override_value is None:
            continue
        base_value = base.get(key)
        if isinstance(override_value, dict) and not isinstance(override_value, list) and isinstance(base_value, dict):
            result[key] = {**base_value, **override_value}
        else:
            result[key] = override_value
    return result


def _migrate_settings(settings: dict[str, Any]) -> Settings:
    """Apply the upstream schema migrations to a freshly loaded blob."""
    # queueMode → steeringMode
    if "queueMode" in settings and "steeringMode" not in settings:
        settings["steeringMode"] = settings["queueMode"]
        del settings["queueMode"]

    # legacy websockets boolean → transport enum
    if "transport" not in settings and isinstance(settings.get("websockets"), bool):
        settings["transport"] = "websocket" if settings["websockets"] else "sse"
        del settings["websockets"]

    # old skills object format → array
    skills = settings.get("skills")
    if isinstance(skills, dict):
        enable = skills.get("enableSkillCommands")
        custom_dirs = skills.get("customDirectories")
        if enable is not None and "enableSkillCommands" not in settings:
            settings["enableSkillCommands"] = enable
        if isinstance(custom_dirs, list) and custom_dirs:
            settings["skills"] = custom_dirs
        else:
            del settings["skills"]

    return settings


# ---------------------------------------------------------------------------
# SettingsManager
# ---------------------------------------------------------------------------


class SettingsManager:
    """Two-scope (global/project) settings manager with deferred writes."""

    def __init__(
        self,
        storage: SettingsStorage,
        initial_global: Settings,
        initial_project: Settings,
        global_load_error: Exception | None = None,
        project_load_error: Exception | None = None,
        initial_errors: list[SettingsError] | None = None,
    ) -> None:
        self._storage = storage
        self._global_settings = initial_global
        self._project_settings = initial_project
        self._settings = _deep_merge_settings(self._global_settings, self._project_settings)
        self._global_load_error = global_load_error
        self._project_load_error = project_load_error
        self._errors: list[SettingsError] = list(initial_errors or [])
        self._modified_fields: set[str] = set()
        self._modified_nested_fields: dict[str, set[str]] = {}
        self._modified_project_fields: set[str] = set()
        self._modified_project_nested_fields: dict[str, set[str]] = {}
        self._write_queue: asyncio.Future[None] | None = None
        self._pending_writes: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, cwd: str | None = None, agent_dir: str | None = None) -> SettingsManager:
        return cls.from_storage(FileSettingsStorage(cwd, agent_dir))

    @classmethod
    def from_storage(cls, storage: SettingsStorage) -> SettingsManager:
        global_load = cls._try_load_from_storage(storage, "global")
        project_load = cls._try_load_from_storage(storage, "project")
        initial_errors: list[SettingsError] = []
        if global_load[1] is not None:
            initial_errors.append(SettingsError(scope="global", error=global_load[1]))
        if project_load[1] is not None:
            initial_errors.append(SettingsError(scope="project", error=project_load[1]))
        return cls(
            storage=storage,
            initial_global=global_load[0],
            initial_project=project_load[0],
            global_load_error=global_load[1],
            project_load_error=project_load[1],
            initial_errors=initial_errors,
        )

    @classmethod
    def in_memory(cls, settings: Settings | None = None) -> SettingsManager:
        return cls(
            storage=InMemorySettingsStorage(),
            initial_global=dict(settings or {}),
            initial_project={},
        )

    @staticmethod
    def _load_from_storage(storage: SettingsStorage, scope: SettingsScope) -> Settings:
        captured: dict[str, str | None] = {"value": None}

        def _capture(current: str | None) -> None:
            captured["value"] = current

        storage.with_lock(scope, _capture)
        content = captured["value"]
        if not content:
            return {}
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {}
        return _migrate_settings(parsed)

    @staticmethod
    def _try_load_from_storage(storage: SettingsStorage, scope: SettingsScope) -> tuple[Settings, Exception | None]:
        try:
            return SettingsManager._load_from_storage(storage, scope), None
        except Exception as exc:
            return {}, exc

    # ------------------------------------------------------------------
    # Snapshot / merge / reload
    # ------------------------------------------------------------------

    def get_global_settings(self) -> Settings:
        return copy.deepcopy(self._global_settings)

    def get_project_settings(self) -> Settings:
        return copy.deepcopy(self._project_settings)

    async def reload(self) -> None:
        await self.flush()
        global_load = self._try_load_from_storage(self._storage, "global")
        if global_load[1] is None:
            self._global_settings = global_load[0]
            self._global_load_error = None
        else:
            self._global_load_error = global_load[1]
            self._record_error("global", global_load[1])

        self._modified_fields.clear()
        self._modified_nested_fields.clear()
        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()

        project_load = self._try_load_from_storage(self._storage, "project")
        if project_load[1] is None:
            self._project_settings = project_load[0]
            self._project_load_error = None
        else:
            self._project_load_error = project_load[1]
            self._record_error("project", project_load[1])

        self._settings = _deep_merge_settings(self._global_settings, self._project_settings)

    def apply_overrides(self, overrides: Settings) -> None:
        self._settings = _deep_merge_settings(self._settings, overrides)

    # ------------------------------------------------------------------
    # Modified-field tracking + persistence
    # ------------------------------------------------------------------

    def _mark_modified(self, field: str, nested_key: str | None = None) -> None:
        self._modified_fields.add(field)
        if nested_key:
            self._modified_nested_fields.setdefault(field, set()).add(nested_key)

    def _mark_project_modified(self, field: str, nested_key: str | None = None) -> None:
        self._modified_project_fields.add(field)
        if nested_key:
            self._modified_project_nested_fields.setdefault(field, set()).add(nested_key)

    def _record_error(self, scope: SettingsScope, error: Exception) -> None:
        self._errors.append(SettingsError(scope=scope, error=error))

    def _clear_modified_scope(self, scope: SettingsScope) -> None:
        if scope == "global":
            self._modified_fields.clear()
            self._modified_nested_fields.clear()
            return
        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()

    def _enqueue_write(self, scope: SettingsScope, task: Callable[[], None]) -> None:
        def _wrapped() -> None:
            try:
                task()
                self._clear_modified_scope(scope)
            except Exception as exc:
                self._record_error(scope, exc)

        self._pending_writes.append(_wrapped)
        # Run synchronously when no event loop is active so that
        # ``manager.set_default_model("x")`` immediately persists in the
        # common test/CLI case. ``flush`` is still available for the
        # async REPL caller.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._drain_pending_writes()

    def _drain_pending_writes(self) -> None:
        while self._pending_writes:
            task = self._pending_writes.pop(0)
            task()

    @staticmethod
    def _clone_modified_nested_fields(source: dict[str, set[str]]) -> dict[str, set[str]]:
        return {key: set(value) for key, value in source.items()}

    def _persist_scoped_settings(
        self,
        scope: SettingsScope,
        snapshot_settings: Settings,
        modified_fields: set[str],
        modified_nested_fields: dict[str, set[str]],
    ) -> None:
        def _update(current: str | None) -> str:
            if current:
                parsed = json.loads(current)
                current_file_settings = _migrate_settings(parsed) if isinstance(parsed, dict) else {}
            else:
                current_file_settings = {}
            merged: Settings = dict(current_file_settings)
            for field in modified_fields:
                value = snapshot_settings.get(field)
                if field in modified_nested_fields and isinstance(value, dict):
                    nested_modified = modified_nested_fields[field]
                    base_nested = current_file_settings.get(field) or {}
                    base_nested = base_nested if isinstance(base_nested, dict) else {}
                    merged_nested = dict(base_nested)
                    for nested_key in nested_modified:
                        merged_nested[nested_key] = value.get(nested_key)
                    merged[field] = merged_nested
                else:
                    merged[field] = value
            return json.dumps(merged, indent=2)

        self._storage.with_lock(scope, _update)

    def _save(self) -> None:
        self._settings = _deep_merge_settings(self._global_settings, self._project_settings)
        if self._global_load_error is not None:
            return
        snapshot = copy.deepcopy(self._global_settings)
        modified_fields = set(self._modified_fields)
        modified_nested = self._clone_modified_nested_fields(self._modified_nested_fields)

        def _task() -> None:
            self._persist_scoped_settings("global", snapshot, modified_fields, modified_nested)

        self._enqueue_write("global", _task)

    def _save_project_settings(self, settings: Settings) -> None:
        self._project_settings = copy.deepcopy(settings)
        self._settings = _deep_merge_settings(self._global_settings, self._project_settings)
        if self._project_load_error is not None:
            return
        snapshot = copy.deepcopy(self._project_settings)
        modified_fields = set(self._modified_project_fields)
        modified_nested = self._clone_modified_nested_fields(self._modified_project_nested_fields)

        def _task() -> None:
            self._persist_scoped_settings("project", snapshot, modified_fields, modified_nested)

        self._enqueue_write("project", _task)

    async def flush(self) -> None:
        # Drain any pending tasks. They run synchronously per write but
        # we expose an awaitable for symmetry with the upstream interface.
        self._drain_pending_writes()

    def drain_errors(self) -> list[SettingsError]:
        drained = list(self._errors)
        self._errors.clear()
        return drained

    # ------------------------------------------------------------------
    # Per-field getters / setters — order matches upstream verbatim.
    # ------------------------------------------------------------------

    def get_last_changelog_version(self) -> str | None:
        return self._settings.get("lastChangelogVersion")

    def set_last_changelog_version(self, version: str) -> None:
        self._global_settings["lastChangelogVersion"] = version
        self._mark_modified("lastChangelogVersion")
        self._save()

    def get_session_dir(self) -> str | None:
        return self._settings.get("sessionDir")

    def get_default_provider(self) -> str | None:
        return self._settings.get("defaultProvider")

    def get_default_model(self) -> str | None:
        return self._settings.get("defaultModel")

    def set_default_provider(self, provider: str) -> None:
        self._global_settings["defaultProvider"] = provider
        self._mark_modified("defaultProvider")
        self._save()

    def set_default_model(self, model_id: str) -> None:
        self._global_settings["defaultModel"] = model_id
        self._mark_modified("defaultModel")
        self._save()

    def set_default_model_and_provider(self, provider: str, model_id: str) -> None:
        self._global_settings["defaultProvider"] = provider
        self._global_settings["defaultModel"] = model_id
        self._mark_modified("defaultProvider")
        self._mark_modified("defaultModel")
        self._save()

    def get_steering_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._settings.get("steeringMode") or "one-at-a-time"

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._global_settings["steeringMode"] = mode
        self._mark_modified("steeringMode")
        self._save()

    def get_follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._settings.get("followUpMode") or "one-at-a-time"

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._global_settings["followUpMode"] = mode
        self._mark_modified("followUpMode")
        self._save()

    def get_theme(self) -> str | None:
        return self._settings.get("theme")

    def set_theme(self, theme: str) -> None:
        self._global_settings["theme"] = theme
        self._mark_modified("theme")
        self._save()

    def get_default_thinking_level(self) -> str | None:
        return self._settings.get("defaultThinkingLevel")

    def set_default_thinking_level(self, level: str) -> None:
        self._global_settings["defaultThinkingLevel"] = level
        self._mark_modified("defaultThinkingLevel")
        self._save()

    def get_transport(self) -> str:
        return self._settings.get("transport") or "sse"

    def set_transport(self, transport: str) -> None:
        self._global_settings["transport"] = transport
        self._mark_modified("transport")
        self._save()

    # ---- compaction ----

    def get_compaction_enabled(self) -> bool:
        compaction = self._settings.get("compaction") or {}
        return compaction.get("enabled", True)

    def set_compaction_enabled(self, enabled: bool) -> None:
        compaction = self._global_settings.setdefault("compaction", {})
        compaction["enabled"] = enabled
        self._mark_modified("compaction", "enabled")
        self._save()

    def get_compaction_reserve_tokens(self) -> int:
        compaction = self._settings.get("compaction") or {}
        return compaction.get("reserveTokens", 16384)

    def get_compaction_keep_recent_tokens(self) -> int:
        compaction = self._settings.get("compaction") or {}
        return compaction.get("keepRecentTokens", 20000)

    def get_compaction_settings(self) -> dict[str, Any]:
        return {
            "enabled": self.get_compaction_enabled(),
            "reserveTokens": self.get_compaction_reserve_tokens(),
            "keepRecentTokens": self.get_compaction_keep_recent_tokens(),
        }

    # ---- branch summary ----

    def get_branch_summary_settings(self) -> dict[str, Any]:
        branch = self._settings.get("branchSummary") or {}
        return {
            "reserveTokens": branch.get("reserveTokens", 16384),
            "skipPrompt": branch.get("skipPrompt", False),
        }

    def get_branch_summary_skip_prompt(self) -> bool:
        return self.get_branch_summary_settings()["skipPrompt"]

    # ---- retry ----

    def get_retry_enabled(self) -> bool:
        retry = self._settings.get("retry") or {}
        return retry.get("enabled", True)

    def set_retry_enabled(self, enabled: bool) -> None:
        retry = self._global_settings.setdefault("retry", {})
        retry["enabled"] = enabled
        self._mark_modified("retry", "enabled")
        self._save()

    def get_retry_settings(self) -> dict[str, Any]:
        retry = self._settings.get("retry") or {}
        return {
            "enabled": self.get_retry_enabled(),
            "maxRetries": retry.get("maxRetries", 3),
            "baseDelayMs": retry.get("baseDelayMs", 2000),
            "maxDelayMs": retry.get("maxDelayMs", 60000),
        }

    # ---- thinking + shell + misc ----

    def get_hide_thinking_block(self) -> bool:
        return self._settings.get("hideThinkingBlock", False)

    def set_hide_thinking_block(self, hide: bool) -> None:
        self._global_settings["hideThinkingBlock"] = hide
        self._mark_modified("hideThinkingBlock")
        self._save()

    def get_shell_path(self) -> str | None:
        return self._settings.get("shellPath")

    def set_shell_path(self, path: str | None) -> None:
        self._global_settings["shellPath"] = path
        self._mark_modified("shellPath")
        self._save()

    def get_quiet_startup(self) -> bool:
        return self._settings.get("quietStartup", False)

    def set_quiet_startup(self, quiet: bool) -> None:
        self._global_settings["quietStartup"] = quiet
        self._mark_modified("quietStartup")
        self._save()

    def get_shell_command_prefix(self) -> str | None:
        return self._settings.get("shellCommandPrefix")

    def set_shell_command_prefix(self, prefix: str | None) -> None:
        self._global_settings["shellCommandPrefix"] = prefix
        self._mark_modified("shellCommandPrefix")
        self._save()

    def get_npm_command(self) -> list[str] | None:
        cmd = self._settings.get("npmCommand")
        return list(cmd) if cmd else None

    def set_npm_command(self, command: list[str] | None) -> None:
        self._global_settings["npmCommand"] = list(command) if command else None
        self._mark_modified("npmCommand")
        self._save()

    def get_collapse_changelog(self) -> bool:
        return self._settings.get("collapseChangelog", False)

    def set_collapse_changelog(self, collapse: bool) -> None:
        self._global_settings["collapseChangelog"] = collapse
        self._mark_modified("collapseChangelog")
        self._save()

    # ---- packages / extensions / skills / prompts / themes (global + project) ----

    def get_packages(self) -> list[Any]:
        return list(self._settings.get("packages") or [])

    def set_packages(self, packages: list[Any]) -> None:
        self._global_settings["packages"] = packages
        self._mark_modified("packages")
        self._save()

    def set_project_packages(self, packages: list[Any]) -> None:
        project = copy.deepcopy(self._project_settings)
        project["packages"] = packages
        self._mark_project_modified("packages")
        self._save_project_settings(project)

    def get_extension_paths(self) -> list[str]:
        return list(self._settings.get("extensions") or [])

    def set_extension_paths(self, paths: list[str]) -> None:
        self._global_settings["extensions"] = paths
        self._mark_modified("extensions")
        self._save()

    def set_project_extension_paths(self, paths: list[str]) -> None:
        project = copy.deepcopy(self._project_settings)
        project["extensions"] = paths
        self._mark_project_modified("extensions")
        self._save_project_settings(project)

    def get_skill_paths(self) -> list[str]:
        return list(self._settings.get("skills") or [])

    def set_skill_paths(self, paths: list[str]) -> None:
        self._global_settings["skills"] = paths
        self._mark_modified("skills")
        self._save()

    def set_project_skill_paths(self, paths: list[str]) -> None:
        project = copy.deepcopy(self._project_settings)
        project["skills"] = paths
        self._mark_project_modified("skills")
        self._save_project_settings(project)

    def get_prompt_template_paths(self) -> list[str]:
        return list(self._settings.get("prompts") or [])

    def set_prompt_template_paths(self, paths: list[str]) -> None:
        self._global_settings["prompts"] = paths
        self._mark_modified("prompts")
        self._save()

    def set_project_prompt_template_paths(self, paths: list[str]) -> None:
        project = copy.deepcopy(self._project_settings)
        project["prompts"] = paths
        self._mark_project_modified("prompts")
        self._save_project_settings(project)

    def get_theme_paths(self) -> list[str]:
        return list(self._settings.get("themes") or [])

    def set_theme_paths(self, paths: list[str]) -> None:
        self._global_settings["themes"] = paths
        self._mark_modified("themes")
        self._save()

    def set_project_theme_paths(self, paths: list[str]) -> None:
        project = copy.deepcopy(self._project_settings)
        project["themes"] = paths
        self._mark_project_modified("themes")
        self._save_project_settings(project)

    def get_enable_skill_commands(self) -> bool:
        return self._settings.get("enableSkillCommands", True)

    def set_enable_skill_commands(self, enabled: bool) -> None:
        self._global_settings["enableSkillCommands"] = enabled
        self._mark_modified("enableSkillCommands")
        self._save()

    def get_thinking_budgets(self) -> dict[str, Any] | None:
        return self._settings.get("thinkingBudgets")

    # ---- terminal / images ----

    def get_show_images(self) -> bool:
        terminal = self._settings.get("terminal") or {}
        return terminal.get("showImages", True)

    def set_show_images(self, show: bool) -> None:
        terminal = self._global_settings.setdefault("terminal", {})
        terminal["showImages"] = show
        self._mark_modified("terminal", "showImages")
        self._save()

    def get_clear_on_shrink(self) -> bool:
        terminal = self._settings.get("terminal") or {}
        if "clearOnShrink" in terminal and terminal["clearOnShrink"] is not None:
            return terminal["clearOnShrink"]
        return os.environ.get("NU_CLEAR_ON_SHRINK") == "1"

    def set_clear_on_shrink(self, enabled: bool) -> None:
        terminal = self._global_settings.setdefault("terminal", {})
        terminal["clearOnShrink"] = enabled
        self._mark_modified("terminal", "clearOnShrink")
        self._save()

    def get_image_auto_resize(self) -> bool:
        images = self._settings.get("images") or {}
        return images.get("autoResize", True)

    def set_image_auto_resize(self, enabled: bool) -> None:
        images = self._global_settings.setdefault("images", {})
        images["autoResize"] = enabled
        self._mark_modified("images", "autoResize")
        self._save()

    def get_block_images(self) -> bool:
        images = self._settings.get("images") or {}
        return images.get("blockImages", False)

    def set_block_images(self, blocked: bool) -> None:
        images = self._global_settings.setdefault("images", {})
        images["blockImages"] = blocked
        self._mark_modified("images", "blockImages")
        self._save()

    # ---- enabled models, double-escape, tree filter, hardware cursor ----

    def get_enabled_models(self) -> list[str] | None:
        return self._settings.get("enabledModels")

    def set_enabled_models(self, patterns: list[str] | None) -> None:
        self._global_settings["enabledModels"] = patterns
        self._mark_modified("enabledModels")
        self._save()

    def get_double_escape_action(self) -> Literal["fork", "tree", "none"]:
        return self._settings.get("doubleEscapeAction") or "tree"

    def set_double_escape_action(self, action: Literal["fork", "tree", "none"]) -> None:
        self._global_settings["doubleEscapeAction"] = action
        self._mark_modified("doubleEscapeAction")
        self._save()

    def get_tree_filter_mode(self) -> str:
        mode = self._settings.get("treeFilterMode")
        valid = {"default", "no-tools", "user-only", "labeled-only", "all"}
        return mode if mode in valid else "default"

    def set_tree_filter_mode(self, mode: str) -> None:
        self._global_settings["treeFilterMode"] = mode
        self._mark_modified("treeFilterMode")
        self._save()

    def get_show_hardware_cursor(self) -> bool:
        if "showHardwareCursor" in self._settings:
            return bool(self._settings["showHardwareCursor"])
        return os.environ.get("NU_HARDWARE_CURSOR") == "1"

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        self._global_settings["showHardwareCursor"] = enabled
        self._mark_modified("showHardwareCursor")
        self._save()

    def get_editor_padding_x(self) -> int:
        return self._settings.get("editorPaddingX", 0)

    def set_editor_padding_x(self, padding: int) -> None:
        clamped = max(0, min(3, int(padding)))
        self._global_settings["editorPaddingX"] = clamped
        self._mark_modified("editorPaddingX")
        self._save()

    def get_autocomplete_max_visible(self) -> int:
        return self._settings.get("autocompleteMaxVisible", 5)

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        clamped = max(3, min(20, int(max_visible)))
        self._global_settings["autocompleteMaxVisible"] = clamped
        self._mark_modified("autocompleteMaxVisible")
        self._save()

    def get_code_block_indent(self) -> str:
        markdown = self._settings.get("markdown") or {}
        return markdown.get("codeBlockIndent", "  ")


__all__ = [
    "FileSettingsStorage",
    "InMemorySettingsStorage",
    "Settings",
    "SettingsError",
    "SettingsManager",
    "SettingsScope",
    "SettingsStorage",
]
