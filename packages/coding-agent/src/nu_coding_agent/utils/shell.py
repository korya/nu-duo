"""Shell config & process helpers — direct port of ``packages/coding-agent/src/utils/shell.ts``.

* :func:`get_shell_config` resolves the shell to invoke for ``-c`` style
  command execution. The TS version reads ``SettingsManager`` for a
  ``shellPath`` override; until ``settings_manager`` is ported the
  resolver falls back to ``$SHELL`` → ``/bin/bash`` → ``bash`` on
  ``PATH`` → ``sh``. The ``settings_loader`` argument lets callers wire
  up the real settings lookup once it lands.
* :func:`get_shell_env` augments :data:`os.environ` with the agent's
  managed bin directory so spawned shells can find ``fd`` / ``rg`` /
  ``ripgrep`` shipped alongside the package.
* :func:`sanitize_binary_output` filters control / format / surrogate
  characters that would otherwise crash terminal width calculations.
* :func:`kill_process_tree` matches the upstream's cross-platform
  behaviour: ``SIGKILL`` on Unix process groups, ``taskkill /F /T`` on
  Windows.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal as _signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nu_coding_agent.config import get_bin_dir

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class ShellConfig:
    """Result of :func:`get_shell_config` — a shell binary plus its argv flags."""

    shell: str
    args: list[str]


type _SettingsLoader = Callable[[], str | None]
"""Optional callback that returns the user's ``shellPath`` override or ``None``."""


_cached_shell_config: ShellConfig | None = None


def _find_bash_on_path() -> str | None:
    """Return the absolute path to ``bash`` on ``PATH`` (or ``None``)."""
    if sys.platform == "win32":  # pragma: no cover — exercised on Windows only
        try:
            result = subprocess.run(
                ["where", "bash.exe"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode == 0 and result.stdout:
            first_match = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            if first_match and Path(first_match).exists():
                return first_match
        return None

    try:
        result = subprocess.run(
            ["which", "bash"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):  # pragma: no cover — defensive
        return None
    if result.returncode == 0 and result.stdout:
        first = result.stdout.strip().splitlines()
        if first:
            return first[0]
    return None  # pragma: no cover — would mean ``which`` exited zero with empty stdout


def get_shell_config(settings_loader: _SettingsLoader | None = None) -> ShellConfig:
    """Return the cached :class:`ShellConfig` for the active platform.

    ``settings_loader`` is the indirection that lets callers wire up
    :class:`nu_coding_agent.core.settings_manager.SettingsManager` once it's
    ported, without making :mod:`nu_coding_agent.utils.shell` depend on it
    (avoids the circular import the upstream module had).
    """
    global _cached_shell_config  # noqa: PLW0603 — process-lifetime cache, mirrors upstream
    if _cached_shell_config is not None:
        return _cached_shell_config

    custom_shell_path = settings_loader() if settings_loader is not None else None
    if custom_shell_path:
        if Path(custom_shell_path).exists():
            _cached_shell_config = ShellConfig(shell=custom_shell_path, args=["-c"])
            return _cached_shell_config
        from nu_coding_agent.config import get_settings_path  # noqa: PLC0415 — lazy import

        raise ValueError(
            f"Custom shell path not found: {custom_shell_path}\nPlease update shellPath in {get_settings_path()}"
        )

    if sys.platform == "win32":  # pragma: no cover — Windows-only branch
        candidates: list[str] = []
        program_files = os.environ.get("ProgramFiles")  # noqa: SIM112 — preserve Windows env-var casing
        if program_files:
            candidates.append(rf"{program_files}\Git\bin\bash.exe")
        program_files_x86 = os.environ.get("ProgramFiles(x86)")  # noqa: SIM112
        if program_files_x86:
            candidates.append(rf"{program_files_x86}\Git\bin\bash.exe")
        for path in candidates:
            if Path(path).exists():
                _cached_shell_config = ShellConfig(shell=path, args=["-c"])
                return _cached_shell_config
        bash_on_path = _find_bash_on_path()
        if bash_on_path:
            _cached_shell_config = ShellConfig(shell=bash_on_path, args=["-c"])
            return _cached_shell_config
        from nu_coding_agent.config import get_settings_path  # noqa: PLC0415 — lazy import

        raise ValueError(
            "No bash shell found. Options:\n"
            "  1. Install Git for Windows: https://git-scm.com/download/win\n"
            "  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n"
            f"  3. Set shellPath in {get_settings_path()}\n\n"
            "Searched Git Bash in:\n" + "\n".join(f"  {p}" for p in candidates),
        )

    if Path("/bin/bash").exists():
        _cached_shell_config = ShellConfig(shell="/bin/bash", args=["-c"])
        return _cached_shell_config
    bash_on_path = _find_bash_on_path() or shutil.which("bash")
    if bash_on_path:
        _cached_shell_config = ShellConfig(shell=bash_on_path, args=["-c"])
        return _cached_shell_config
    _cached_shell_config = ShellConfig(shell="sh", args=["-c"])  # pragma: no cover — last-resort fallback
    return _cached_shell_config


def reset_shell_config_cache() -> None:
    """Drop the cached :class:`ShellConfig`. Exposed for tests."""
    global _cached_shell_config  # noqa: PLW0603
    _cached_shell_config = None


def get_shell_env() -> dict[str, str]:
    """Return ``os.environ`` augmented with the agent's bin directory on ``PATH``."""
    bin_dir = get_bin_dir()
    path_key = next((k for k in os.environ if k.lower() == "path"), "PATH")
    current_path = os.environ.get(path_key, "")
    path_entries = [p for p in current_path.split(os.pathsep) if p]
    if bin_dir not in path_entries:
        current_path = os.pathsep.join([bin_dir, current_path]) if current_path else bin_dir
    env = dict(os.environ)
    env[path_key] = current_path
    return env


_TAB = 0x09
_LF = 0x0A
_CR = 0x0D
_CONTROL_LIMIT = 0x1F
_FORMAT_RANGE_START = 0xFFF9
_FORMAT_RANGE_END = 0xFFFB


def sanitize_binary_output(text: str) -> str:
    """Drop control/format/surrogate characters that crash terminal width helpers."""
    out: list[str] = []
    for char in text:
        try:
            code = ord(char)
        except TypeError:  # pragma: no cover — defensive
            continue
        if code in (_TAB, _LF, _CR):
            out.append(char)
            continue
        if code <= _CONTROL_LIMIT:
            continue
        if _FORMAT_RANGE_START <= code <= _FORMAT_RANGE_END:
            continue
        out.append(char)
    return "".join(out)


def kill_process_tree(pid: int) -> None:
    """Best-effort kill of ``pid`` and any descendants.

    Mirrors the upstream ``killProcessTree`` semantics: ``taskkill /F /T``
    on Windows, ``kill(-pid, SIGKILL)`` (process group) on Unix with a
    fallback to a single-process kill if the group call fails.
    """
    if sys.platform == "win32":  # pragma: no cover — Windows-only
        with contextlib.suppress(Exception):
            subprocess.Popen(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return
    try:
        os.killpg(pid, _signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, _signal.SIGKILL)


__all__ = [
    "ShellConfig",
    "get_shell_config",
    "get_shell_env",
    "kill_process_tree",
    "reset_shell_config_cache",
    "sanitize_binary_output",
]
