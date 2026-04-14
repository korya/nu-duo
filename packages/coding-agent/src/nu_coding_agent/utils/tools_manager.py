"""Binary tool manager — port of ``packages/coding-agent/src/utils/tools-manager.ts``.

Downloads and manages ``fd`` and ``rg`` (ripgrep) binary tools from GitHub
releases.  Cached binaries live under :func:`~nu_coding_agent.config.get_bin_dir`
(typically ``~/.nu/agent/bin/``).  A system-wide installation on ``$PATH`` is
used when available, and downloads are skipped entirely when the
``PI_OFFLINE`` or ``NU_OFFLINE`` environment variable is truthy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform as _platform
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from nu_coding_agent.config import APP_NAME, get_bin_dir

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NETWORK_TIMEOUT_S = 10
DOWNLOAD_TIMEOUT_S = 120

ToolName = Literal["fd", "rg"]

# ---------------------------------------------------------------------------
# Offline detection
# ---------------------------------------------------------------------------


def _is_offline_mode() -> bool:
    """Return *True* when ``PI_OFFLINE`` or ``NU_OFFLINE`` indicates offline."""
    for var in ("PI_OFFLINE", "NU_OFFLINE"):
        val = os.environ.get(var, "")
        if val.lower() in ("1", "true", "yes"):
            return True
    return False


# ---------------------------------------------------------------------------
# Tool configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ToolConfig:
    name: str
    repo: str
    binary_name: str
    tag_prefix: str
    get_asset_name: Callable[[str, str, str], str | None]


def _fd_asset_name(version: str, plat: str, arch: str) -> str | None:
    if plat == "darwin":
        arch_str = "aarch64" if arch == "arm64" else "x86_64"
        return f"fd-v{version}-{arch_str}-apple-darwin.tar.gz"
    if plat == "linux":
        arch_str = "aarch64" if arch == "arm64" else "x86_64"
        return f"fd-v{version}-{arch_str}-unknown-linux-gnu.tar.gz"
    if plat == "win32":
        arch_str = "aarch64" if arch == "arm64" else "x86_64"
        return f"fd-v{version}-{arch_str}-pc-windows-msvc.zip"
    return None


def _rg_asset_name(version: str, plat: str, arch: str) -> str | None:
    if plat == "darwin":
        arch_str = "aarch64" if arch == "arm64" else "x86_64"
        return f"ripgrep-{version}-{arch_str}-apple-darwin.tar.gz"
    if plat == "linux":
        if arch == "arm64":
            return f"ripgrep-{version}-aarch64-unknown-linux-gnu.tar.gz"
        return f"ripgrep-{version}-x86_64-unknown-linux-musl.tar.gz"
    if plat == "win32":
        arch_str = "aarch64" if arch == "arm64" else "x86_64"
        return f"ripgrep-{version}-{arch_str}-pc-windows-msvc.zip"
    return None


_TOOLS: dict[str, _ToolConfig] = {
    "fd": _ToolConfig(
        name="fd",
        repo="sharkdp/fd",
        binary_name="fd",
        tag_prefix="v",
        get_asset_name=_fd_asset_name,
    ),
    "rg": _ToolConfig(
        name="ripgrep",
        repo="BurntSushi/ripgrep",
        binary_name="rg",
        tag_prefix="",
        get_asset_name=_rg_asset_name,
    ),
}

# Termux package names for when we're on Android.
_TERMUX_PACKAGES: dict[str, str] = {"fd": "fd", "rg": "ripgrep"}

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _get_platform() -> str:
    """Map Python's :func:`sys.platform` to the Node.js-style names used by
    the upstream asset-name functions (``darwin``, ``linux``, ``win32``,
    ``android``)."""
    plat = sys.platform
    if plat.startswith("linux"):
        # Detect Android/Termux via env or uname
        if "ANDROID_ROOT" in os.environ or "com.termux" in os.environ.get("PREFIX", ""):
            return "android"
        return "linux"
    if plat == "darwin":
        return "darwin"
    if plat in ("win32", "cygwin"):
        return "win32"
    return plat


def _get_arch() -> str:
    """Map Python's :func:`platform.machine` to the Node ``os.arch()`` names
    (``arm64``, ``x64``, etc.)."""
    machine = _platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "x64"
    return machine


# ---------------------------------------------------------------------------
# Synchronous helpers (run via asyncio.to_thread when needed)
# ---------------------------------------------------------------------------


def _find_binary_recursively(root: Path, binary_name: str) -> Path | None:
    """Walk *root* looking for a file named *binary_name*."""
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and entry.name == binary_name:
                return entry
            if entry.is_dir():
                stack.append(entry)
    return None


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


async def _get_latest_version(repo: str) -> str:
    """Fetch the latest release tag from GitHub and strip a leading ``v``."""
    import httpx  # noqa: PLC0415 — lazy import to keep module load fast

    url = f"https://api.github.com/repos/{repo}/releases/latest"
    async with httpx.AsyncClient(timeout=NETWORK_TIMEOUT_S) as client:
        resp = await client.get(url, headers={"User-Agent": f"{APP_NAME}-coding-agent"})
        resp.raise_for_status()
    tag: str = resp.json()["tag_name"]
    return tag.lstrip("v")


async def _download_file(url: str, dest: Path) -> None:
    """Stream-download *url* into *dest*."""
    import httpx  # noqa: PLC0415

    async with (
        httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client,
        client.stream("GET", url) as resp,
    ):
        resp.raise_for_status()
        with dest.open("wb") as fh:
            async for chunk in resp.aiter_bytes(chunk_size=65_536):
                fh.write(chunk)


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------


async def _download_tool(tool: ToolName) -> str:
    """Download the latest release of *tool* and return the installed path."""
    config = _TOOLS[tool]
    plat = _get_platform()
    arch = _get_arch()

    version = await _get_latest_version(config.repo)

    asset_name = config.get_asset_name(version, plat, arch)
    if asset_name is None:
        raise RuntimeError(f"Unsupported platform: {plat}/{arch}")

    bin_dir = Path(get_bin_dir())
    await asyncio.to_thread(bin_dir.mkdir, parents=True, exist_ok=True)

    binary_ext = ".exe" if plat == "win32" else ""
    binary_path = bin_dir / (config.binary_name + binary_ext)

    download_url = f"https://github.com/{config.repo}/releases/download/{config.tag_prefix}{version}/{asset_name}"

    # Use a per-invocation temp directory so concurrent downloads don't clash.
    with tempfile.TemporaryDirectory(dir=bin_dir, prefix=f"dl_{config.binary_name}_") as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / asset_name

        await _download_file(download_url, archive_path)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        # Extract ---------------------------------------------------------
        def _extract_and_install() -> None:
            if asset_name.endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:gz") as tf:
                    tf.extractall(extract_dir)
            elif asset_name.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(extract_dir)
            else:
                raise RuntimeError(f"Unsupported archive format: {asset_name}")

            binary_file_name = config.binary_name + binary_ext

            # Try well-known candidate locations first.
            stem = asset_name.removesuffix(".tar.gz").removesuffix(".zip")
            candidates = [
                extract_dir / stem / binary_file_name,
                extract_dir / binary_file_name,
            ]
            found = next((c for c in candidates if c.is_file()), None)

            if found is None:
                found = _find_binary_recursively(extract_dir, binary_file_name)

            if found is None:
                raise RuntimeError(f"Binary not found in archive: expected {binary_file_name} under {extract_dir}")

            shutil.move(str(found), str(binary_path))

            if plat != "win32":
                binary_path.chmod(binary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        await asyncio.to_thread(_extract_and_install)

    return str(binary_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tool_path(tool: ToolName) -> str | None:
    """Return the path to *tool*'s binary, or ``None`` if it is not available.

    Checks the local cache directory first, then the system ``$PATH``.
    """
    config = _TOOLS.get(tool)
    if config is None:
        return None

    # 1. Cached binary in our bin dir.
    ext = ".exe" if sys.platform in ("win32", "cygwin") else ""
    local_path = Path(get_bin_dir()) / (config.binary_name + ext)
    if local_path.is_file():
        return str(local_path)

    # 2. System-wide installation.
    which = shutil.which(config.binary_name)
    if which is not None:
        return which

    return None


async def ensure_tool(tool: ToolName, *, silent: bool = False) -> str | None:
    """Ensure *tool* is available, downloading it from GitHub if necessary.

    Returns the path to the binary, or ``None`` when the tool could not be
    obtained (offline mode, unsupported platform, download failure, …).
    """
    existing = get_tool_path(tool)
    if existing is not None:
        return existing

    config = _TOOLS.get(tool)
    if config is None:
        return None

    if _is_offline_mode():
        if not silent:
            logger.warning("%s not found. Offline mode enabled, skipping download.", config.name)
        return None

    # Android / Termux — pre-built Linux binaries are incompatible with Bionic.
    if _get_platform() == "android":
        pkg = _TERMUX_PACKAGES.get(tool, tool)
        if not silent:
            logger.warning("%s not found. Install with: pkg install %s", config.name, pkg)
        return None

    if not silent:
        logger.info("%s not found. Downloading...", config.name)

    try:
        path = await _download_tool(tool)
    except Exception as exc:
        if not silent:
            logger.warning("Failed to download %s: %s", config.name, exc)
        return None

    if not silent:
        logger.info("%s installed to %s", config.name, path)
    return path


__all__ = [
    "ToolName",
    "ensure_tool",
    "get_tool_path",
]
