"""One-time migrations that run on startup — port of ``migrations.ts`` (315 LOC).

Each migration is idempotent: it checks whether the work has already been done
before touching the filesystem, and swallows errors so that a single broken
migration never blocks the agent from starting.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from nu_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir, get_bin_dir

_MIGRATION_GUIDE_URL = (
    "https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/CHANGELOG.md#extensions-migration"
)
_EXTENSIONS_DOC_URL = "https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md"


@dataclass(slots=True)
class MigrationResult:
    """Outcome of :func:`run_migrations`."""

    migrated_auth_providers: list[str] = field(default_factory=list)
    deprecation_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Auth migration: oauth.json + settings.json apiKeys → auth.json
# ---------------------------------------------------------------------------


def migrate_auth_to_auth_json(agent_dir: str | None = None) -> list[str]:
    """Migrate legacy ``oauth.json`` and ``settings.json`` apiKeys into ``auth.json``.

    Returns a list of provider names that were migrated.  If ``auth.json``
    already exists the migration is skipped entirely.
    """
    ad = Path(agent_dir) if agent_dir else Path(get_agent_dir())
    auth_path = ad / "auth.json"
    oauth_path = ad / "oauth.json"
    settings_path = ad / "settings.json"

    if auth_path.exists():
        return []

    migrated: dict[str, object] = {}
    providers: list[str] = []

    # -- oauth.json ----------------------------------------------------------
    if oauth_path.exists():
        try:
            oauth = json.loads(oauth_path.read_text("utf-8"))
            for provider, cred in oauth.items():
                entry: dict[str, object] = {"type": "oauth"}
                if isinstance(cred, dict):
                    entry.update(cred)
                migrated[provider] = entry
                providers.append(provider)
            oauth_path.rename(oauth_path.with_suffix(".json.migrated"))
        except Exception:
            pass

    # -- settings.json apiKeys -----------------------------------------------
    if settings_path.exists():
        try:
            content = settings_path.read_text("utf-8")
            settings = json.loads(content)
            api_keys = settings.get("apiKeys")
            if isinstance(api_keys, dict):
                for provider, key in api_keys.items():
                    if provider not in migrated and isinstance(key, str):
                        migrated[provider] = {"type": "api_key", "key": key}
                        providers.append(provider)
                del settings["apiKeys"]
                settings_path.write_text(json.dumps(settings, indent=2), "utf-8")
        except Exception:
            pass

    if migrated:
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text(json.dumps(migrated, indent=2), "utf-8")
        # Restrict to owner-only (0o600) — best-effort on platforms that support it.
        with contextlib.suppress(OSError):
            auth_path.chmod(0o600)

    return providers


# ---------------------------------------------------------------------------
# Sessions migration: stray .jsonl files → sessions/<cwd_hash>/
# ---------------------------------------------------------------------------


def migrate_sessions_from_agent_root(agent_dir: str | None = None) -> None:
    """Move ``.jsonl`` files from the agent-dir root into the correct session subdirectory.

    A bug in v0.30.0 saved sessions directly into ``~/.nu/agent/`` instead of
    ``~/.nu/agent/sessions/<encoded-cwd>/``.  This migration reads each file's
    header to discover the original *cwd* and moves the file accordingly.
    """
    ad = Path(agent_dir) if agent_dir else Path(get_agent_dir())

    try:
        files = [f for f in ad.iterdir() if f.is_file() and f.suffix == ".jsonl"]
    except OSError:
        return

    if not files:
        return

    for file in files:
        try:
            first_line = file.read_text("utf-8").split("\n", 1)[0]
            if not first_line.strip():
                continue

            header = json.loads(first_line)
            if header.get("type") != "session" or not header.get("cwd"):
                continue

            cwd: str = header["cwd"]
            # Same encoding as session-manager.ts
            safe_path = "--" + re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", cwd)) + "--"
            correct_dir = ad / "sessions" / safe_path

            correct_dir.mkdir(parents=True, exist_ok=True)
            new_path = correct_dir / file.name

            if new_path.exists():
                continue

            file.rename(new_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Extension system: commands/ → prompts/, deprecated hooks/ & tools/
# ---------------------------------------------------------------------------


def _migrate_commands_to_prompts(base_dir: Path, label: str) -> bool:
    """Rename ``commands/`` to ``prompts/`` under *base_dir* (regular dirs or symlinks)."""
    commands_dir = base_dir / "commands"
    prompts_dir = base_dir / "prompts"

    if commands_dir.exists() and not prompts_dir.exists():
        try:
            commands_dir.rename(prompts_dir)
            print(f"Migrated {label} commands/ \u2192 prompts/", file=sys.stderr)
            return True
        except OSError as exc:
            print(
                f"Warning: Could not migrate {label} commands/ to prompts/: {exc}",
                file=sys.stderr,
            )
    return False


def _check_deprecated_extension_dirs(base_dir: Path, label: str) -> list[str]:
    """Return warnings for any deprecated ``hooks/`` or ``tools/`` directories."""
    warnings: list[str] = []
    hooks_dir = base_dir / "hooks"
    tools_dir = base_dir / "tools"

    if hooks_dir.exists():
        warnings.append(f"{label} hooks/ directory found. Hooks have been renamed to extensions.")

    if tools_dir.exists():
        try:
            _MANAGED_BINARIES = {"fd", "rg", "fd.exe", "rg.exe"}
            custom = [
                e.name
                for e in tools_dir.iterdir()
                if e.name.lower() not in _MANAGED_BINARIES and not e.name.startswith(".")
            ]
            if custom:
                warnings.append(
                    f"{label} tools/ directory contains custom tools. Custom tools have been merged into extensions."
                )
        except OSError:
            pass

    return warnings


def _migrate_extension_system(cwd: str) -> list[str]:
    """Run extension-system migrations and return deprecation warnings."""
    agent_dir = Path(get_agent_dir())
    project_dir = Path(cwd) / CONFIG_DIR_NAME

    _migrate_commands_to_prompts(agent_dir, "Global")
    _migrate_commands_to_prompts(project_dir, "Project")

    warnings = [
        *_check_deprecated_extension_dirs(agent_dir, "Global"),
        *_check_deprecated_extension_dirs(project_dir, "Project"),
    ]
    return warnings


# ---------------------------------------------------------------------------
# Keybindings config migration
# ---------------------------------------------------------------------------


def _migrate_keybindings_config_file() -> None:
    """Migrate old keybindings.json format if present."""
    config_path = Path(get_agent_dir()) / "keybindings.json"
    if not config_path.exists():
        return

    try:
        raw = json.loads(config_path.read_text("utf-8"))
        if not isinstance(raw, dict):
            return

        from nu_coding_agent.core.keybindings import migrate_keybindings_config  # noqa: PLC0415

        config, migrated = migrate_keybindings_config(raw)
        if not migrated:
            return
        config_path.write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Binary management: tools/ → bin/
# ---------------------------------------------------------------------------


def _migrate_tools_to_bin() -> None:
    """Move managed binaries (``fd``, ``rg``) from ``tools/`` to ``bin/``."""
    agent_dir = Path(get_agent_dir())
    tools_dir = agent_dir / "tools"
    bin_dir = Path(get_bin_dir())

    if not tools_dir.exists():
        return

    binaries = ("fd", "rg", "fd.exe", "rg.exe")
    moved_any = False

    for name in binaries:
        old_path = tools_dir / name
        new_path = bin_dir / name

        if old_path.exists():
            bin_dir.mkdir(parents=True, exist_ok=True)
            if not new_path.exists():
                try:
                    old_path.rename(new_path)
                    moved_any = True
                except OSError:
                    pass
            else:
                with contextlib.suppress(OSError):
                    old_path.unlink()

    if moved_any:
        print("Migrated managed binaries tools/ \u2192 bin/", file=sys.stderr)


# ---------------------------------------------------------------------------
# Deprecation warnings UI
# ---------------------------------------------------------------------------


async def show_deprecation_warnings(warnings: list[str]) -> None:
    """Print deprecation warnings to *stderr* and wait for a keypress."""
    if not warnings:
        return

    for warning in warnings:
        print(f"\033[33mWarning: {warning}\033[0m", file=sys.stderr)

    print(
        "\033[33m\nMove your extensions to the extensions/ directory.\033[0m",
        file=sys.stderr,
    )
    print(f"\033[33mMigration guide: {_MIGRATION_GUIDE_URL}\033[0m", file=sys.stderr)
    print(f"\033[33mDocumentation: {_EXTENSIONS_DOC_URL}\033[0m", file=sys.stderr)
    print("\033[2m\nPress any key to continue...\033[0m", file=sys.stderr)

    # Best-effort raw-mode read; falls through immediately if stdin is not a tty.
    if sys.stdin.isatty():
        try:
            import termios  # noqa: PLC0415
            import tty  # noqa: PLC0415

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_migrations(cwd: str | None = None) -> MigrationResult:
    """Run all pending migrations.  Called once on startup.

    Returns a :class:`MigrationResult` with the list of migrated auth
    providers and any deprecation warnings that should be shown to the user.
    """
    effective_cwd = cwd or os.getcwd()

    migrated_providers = migrate_auth_to_auth_json()
    migrate_sessions_from_agent_root()
    _migrate_tools_to_bin()
    _migrate_keybindings_config_file()
    deprecation_warnings = _migrate_extension_system(effective_cwd)

    return MigrationResult(
        migrated_auth_providers=migrated_providers,
        deprecation_warnings=deprecation_warnings,
    )


__all__ = [
    "MigrationResult",
    "migrate_auth_to_auth_json",
    "migrate_sessions_from_agent_root",
    "run_migrations",
    "show_deprecation_warnings",
]
