"""Config value resolution — direct port of ``packages/coding-agent/src/core/resolve-config-value.ts``.

API keys, header values, and other settings can be one of three things:

* a literal string,
* an environment variable name (resolved against :data:`os.environ`),
* a shell command, prefixed with ``!`` (e.g. ``!pass anthropic-api-key``).

Shell command results are cached for the lifetime of the process to
avoid re-executing on every API call. Headers go through the same
resolver so that ``Authorization: Bearer !aws sts ...`` works.
"""

from __future__ import annotations

import os
import subprocess

_command_result_cache: dict[str, str | None] = {}
_COMMAND_TIMEOUT_SECONDS = 10.0


def resolve_config_value(config: str) -> str | None:
    """Resolve ``config``: shell command, env var, or literal — cached for ``!cmd`` form."""
    if config.startswith("!"):
        return _execute_command(config)
    env_value = os.environ.get(config)
    return env_value or config


def resolve_config_value_uncached(config: str) -> str | None:
    """Same as :func:`resolve_config_value` but bypasses the shell command cache."""
    if config.startswith("!"):
        return _execute_command_uncached(config)
    env_value = os.environ.get(config)
    return env_value or config


def resolve_config_value_or_throw(config: str, description: str) -> str:
    """Resolve ``config`` or raise :class:`ValueError` (always uncached)."""
    resolved = resolve_config_value_uncached(config)
    if resolved is not None:
        return resolved
    if config.startswith("!"):
        raise ValueError(f"Failed to resolve {description} from shell command: {config[1:]}")
    # Unreachable: ``resolve_config_value_uncached`` returns the literal
    # ``config`` when the env var is unset, never ``None``.
    raise ValueError(f"Failed to resolve {description}")  # pragma: no cover


def resolve_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Apply :func:`resolve_config_value` to every value in ``headers``."""
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved_value = resolve_config_value(value)
        if resolved_value:
            resolved[key] = resolved_value
    return resolved or None


def resolve_headers_or_throw(
    headers: dict[str, str] | None,
    description: str,
) -> dict[str, str] | None:
    """Like :func:`resolve_headers` but raises if any header fails to resolve."""
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved[key] = resolve_config_value_or_throw(value, f'{description} header "{key}"')
    return resolved or None


def clear_config_value_cache() -> None:
    """Drop the shell-command result cache. Exposed for tests."""
    _command_result_cache.clear()


def _execute_command_uncached(command_config: str) -> str | None:
    command = command_config[1:]
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):  # pragma: no cover — platform/timeout edge
        return None
    if result.returncode != 0:
        return None
    output = (result.stdout or "").strip()
    return output or None


def _execute_command(command_config: str) -> str | None:
    if command_config in _command_result_cache:
        return _command_result_cache[command_config]
    result = _execute_command_uncached(command_config)
    _command_result_cache[command_config] = result
    return result


__all__ = [
    "clear_config_value_cache",
    "resolve_config_value",
    "resolve_config_value_or_throw",
    "resolve_config_value_uncached",
    "resolve_headers",
    "resolve_headers_or_throw",
]
