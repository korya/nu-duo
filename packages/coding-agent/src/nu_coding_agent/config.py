"""App-wide configuration paths — port of ``packages/coding-agent/src/config.ts``.

The TS upstream embeds the ``piConfig`` block from ``package.json`` to
let downstream packagers rebrand the binary (``pi`` → ``tau``, ``.pi`` →
``.tau``). For Nu-duo we hardcode the matching constants:

* ``APP_NAME = "nu"``
* ``CONFIG_DIR_NAME = ".nu"``
* ``ENV_AGENT_DIR = "NU_CODING_AGENT_DIR"``

Bun-binary detection is dropped (Python has no equivalent runtime). The
package-asset helpers (``get_themes_dir``, ``get_export_template_dir``,
``get_interactive_assets_dir``) point at the importable
:mod:`nu_coding_agent` package directory, which is how Python ships
data files.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "nu"
CONFIG_DIR_NAME = ".nu"
ENV_AGENT_DIR = f"{APP_NAME.upper()}_CODING_AGENT_DIR"

_DEFAULT_SHARE_VIEWER_URL = "https://pi.dev/session/"


def _expand_tilde(path: str) -> str:
    if path == "~":
        return str(Path.home())
    if path.startswith("~/"):
        return str(Path.home() / path[2:])
    return path


# ---------------------------------------------------------------------------
# Package asset paths (shipped with the wheel)
# ---------------------------------------------------------------------------


def get_package_dir() -> str:
    """Return the directory containing :mod:`nu_coding_agent`'s ``__init__.py``.

    The TS version walks up to find ``package.json``; in Python the
    package directory itself is the analogue and ``importlib.resources``
    handles asset lookups from there. The ``NU_PACKAGE_DIR`` env var
    overrides for distros that need a custom prefix.
    """
    env_dir = os.environ.get("NU_PACKAGE_DIR")
    if env_dir:
        return _expand_tilde(env_dir)
    import nu_coding_agent  # noqa: PLC0415

    return str(Path(nu_coding_agent.__file__).parent)


def get_themes_dir() -> str:
    """Path to the bundled theme JSON files (interactive mode)."""
    return str(Path(get_package_dir()) / "modes" / "interactive" / "theme")


def get_export_template_dir() -> str:
    """Path to the bundled HTML export templates."""
    return str(Path(get_package_dir()) / "core" / "export_html")


def get_interactive_assets_dir() -> str:
    """Path to the bundled interactive UI assets (icons, fonts, …)."""
    return str(Path(get_package_dir()) / "modes" / "interactive" / "assets")


def get_bundled_interactive_asset_path(name: str) -> str:
    return str(Path(get_interactive_assets_dir()) / name)


# ---------------------------------------------------------------------------
# Share URL
# ---------------------------------------------------------------------------


def get_share_viewer_url(gist_id: str) -> str:
    base_url = os.environ.get("NU_SHARE_VIEWER_URL") or _DEFAULT_SHARE_VIEWER_URL
    return f"{base_url}#{gist_id}"


# ---------------------------------------------------------------------------
# User config paths (~/.nu/agent/*)
# ---------------------------------------------------------------------------


def get_agent_dir() -> str:
    """Return the user-level agent config directory (e.g. ``~/.nu/agent``)."""
    env_dir = os.environ.get(ENV_AGENT_DIR)
    if env_dir:
        return _expand_tilde(env_dir)
    return str(Path.home() / CONFIG_DIR_NAME / "agent")


def get_custom_themes_dir() -> str:
    return str(Path(get_agent_dir()) / "themes")


def get_models_path() -> str:
    return str(Path(get_agent_dir()) / "models.json")


def get_auth_path() -> str:
    return str(Path(get_agent_dir()) / "auth.json")


def get_settings_path() -> str:
    return str(Path(get_agent_dir()) / "settings.json")


def get_tools_dir() -> str:
    return str(Path(get_agent_dir()) / "tools")


def get_bin_dir() -> str:
    return str(Path(get_agent_dir()) / "bin")


def get_prompts_dir() -> str:
    return str(Path(get_agent_dir()) / "prompts")


def get_sessions_dir() -> str:
    return str(Path(get_agent_dir()) / "sessions")


def get_debug_log_path() -> str:
    return str(Path(get_agent_dir()) / f"{APP_NAME}-debug.log")


__all__ = [
    "APP_NAME",
    "CONFIG_DIR_NAME",
    "ENV_AGENT_DIR",
    "get_agent_dir",
    "get_auth_path",
    "get_bin_dir",
    "get_bundled_interactive_asset_path",
    "get_custom_themes_dir",
    "get_debug_log_path",
    "get_export_template_dir",
    "get_interactive_assets_dir",
    "get_models_path",
    "get_package_dir",
    "get_prompts_dir",
    "get_sessions_dir",
    "get_settings_path",
    "get_share_viewer_url",
    "get_themes_dir",
    "get_tools_dir",
]
