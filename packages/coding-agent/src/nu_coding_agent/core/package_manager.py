"""Extension package management — port of ``packages/coding-agent/src/core/package-manager.ts``.

The TS upstream (2254 LoC) uses npm to install extension "packages" that
carry tools, skills, prompts, and themes. Python's packaging ecosystem
gives us a much shorter path:

* **pip packages** are installed with ``uv pip install`` (preferred) or
  ``pip install`` as a fallback.
* **Local packages** are recorded by path and resolved directly.
* **Git packages** are cloned and installed in editable mode.
* **Resource discovery** uses :mod:`importlib.metadata` entry points
  (group ``nu_coding_agent.extensions``) and directory scanning for
  ``skills/``, ``prompts/``, ``themes/`` subdirs inside each package.

Settings persistence goes through :class:`SettingsManager` (the
``packages`` key in ``settings.json``). The precedence order for
resource resolution is: user-level > project-level > package-installed.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from nu_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir

if TYPE_CHECKING:
    from nu_coding_agent.core.settings_manager import SettingsManager
    from nu_coding_agent.core.source_info import PathMetadataLike, SourceScope

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entry-point group used for resource discovery
# ---------------------------------------------------------------------------

EXTENSION_EP_GROUP = "nu_coding_agent.extensions"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

type SourceType = Literal["pip", "git", "local"]
type ResourceType = Literal["extension", "skill", "prompt", "theme"]


@dataclass(slots=True)
class ConfiguredPackage:
    """A package the user has configured (installed or pending)."""

    source: str  # pip package name, git URL, or local path
    source_type: SourceType
    enabled: bool = True


@dataclass(slots=True)
class ResolvedResource:
    """A resource (extension, skill, prompt, theme) found in a package."""

    name: str
    path: str
    package_source: str
    resource_type: ResourceType


@dataclass(slots=True)
class ResolvedPaths:
    """All resolved resources, grouped by type."""

    extensions: list[ResolvedResource] = field(default_factory=list)
    skills: list[ResolvedResource] = field(default_factory=list)
    prompts: list[ResolvedResource] = field(default_factory=list)
    themes: list[ResolvedResource] = field(default_factory=list)

    def merge(self, other: ResolvedPaths) -> None:
        """Append all resources from *other* into *self*."""
        self.extensions.extend(other.extensions)
        self.skills.extend(other.skills)
        self.prompts.extend(other.prompts)
        self.themes.extend(other.themes)


@dataclass(slots=True)
class PackageUpdate:
    """Information about an available update for an installed package."""

    source: str
    current_version: str
    latest_version: str


@dataclass(slots=True)
class PathMetadata:
    """Satisfies :class:`PathMetadataLike` from :mod:`source_info`."""

    source: str
    scope: SourceScope
    origin: Literal["package", "top-level"]
    base_dir: str | None


# Verify structural compatibility with the protocol at import time.
_pm_check: PathMetadataLike = PathMetadata(source="", scope="user", origin="package", base_dir=None)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PackageManager(Protocol):
    """Interface for package management."""

    def get_configured_packages(self) -> list[ConfiguredPackage]: ...
    def get_resolved_paths(self) -> ResolvedPaths: ...
    async def install(self, source: str) -> None: ...
    async def remove(self, source: str) -> None: ...
    async def check_updates(self) -> list[PackageUpdate]: ...
    def resolve(self) -> ResolvedPaths: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GIT_URL_RE = re.compile(
    r"^(?:git\+|(?:https?|ssh|git)://)|\.git(?:$|#|@)|^git@",
    re.IGNORECASE,
)


def classify_source(source: str) -> SourceType:
    """Determine whether *source* is a pip name, a git URL, or a local path."""
    if _GIT_URL_RE.search(source):
        return "git"
    expanded = os.path.expanduser(source)
    if os.path.exists(expanded) or source.startswith(("./", "../", "/")):
        return "local"
    return "pip"


def _find_pip_command() -> list[str]:
    """Return the preferred pip invocation (``uv pip`` or ``pip``)."""
    if shutil.which("uv"):
        return ["uv", "pip"]
    if shutil.which("pip"):
        return ["pip"]
    raise RuntimeError(
        "Neither 'uv' nor 'pip' found on PATH. "
        "Install uv (https://docs.astral.sh/uv/) or pip to manage packages."
    )


async def _run_pip(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a pip command asynchronously."""
    cmd = [*_find_pip_command(), *args]
    logger.debug("Running: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    result = subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode or 0,
        stdout=(stdout_bytes or b"").decode(),
        stderr=(stderr_bytes or b"").decode(),
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"pip command failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result


def _scan_resource_dirs(base: str | Path, source: str) -> ResolvedPaths:
    """Scan a package directory for ``skills/``, ``prompts/``, ``themes/`` subdirs."""
    base = Path(base)
    paths = ResolvedPaths()
    for subdir, rtype in [
        ("skills", "skill"),
        ("prompts", "prompt"),
        ("themes", "theme"),
    ]:
        resource_type: ResourceType = rtype  # type: ignore[assignment]
        d = base / subdir
        if not d.is_dir():
            continue
        for child in sorted(d.iterdir()):
            if child.name.startswith((".", "_")):
                continue
            # Accept .py files and directories with __init__.py
            if child.is_file() and child.suffix in (".py", ".md", ".json", ".yaml", ".yml"):
                paths_list = getattr(paths, f"{rtype}s")
                paths_list.append(
                    ResolvedResource(
                        name=child.stem,
                        path=str(child),
                        package_source=source,
                        resource_type=resource_type,
                    )
                )
            elif child.is_dir() and (child / "__init__.py").exists():
                paths_list = getattr(paths, f"{rtype}s")
                paths_list.append(
                    ResolvedResource(
                        name=child.name,
                        path=str(child),
                        package_source=source,
                        resource_type=resource_type,
                    )
                )
    return paths


def _resolve_user_paths(
    raw_paths: list[str],
    resource_type: ResourceType,
    source_label: str,
) -> list[ResolvedResource]:
    """Expand a list of user-configured paths into :class:`ResolvedResource` items."""
    result: list[ResolvedResource] = []
    for raw in raw_paths:
        expanded = os.path.expanduser(raw)
        p = Path(expanded)
        if not p.exists():
            logger.warning("Configured %s path does not exist: %s", resource_type, raw)
            continue
        if p.is_file():
            result.append(
                ResolvedResource(
                    name=p.stem,
                    path=str(p.resolve()),
                    package_source=source_label,
                    resource_type=resource_type,
                )
            )
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.name.startswith((".", "_")):
                    continue
                if child.is_file() or (child.is_dir() and (child / "__init__.py").exists()):
                    result.append(
                        ResolvedResource(
                            name=child.stem if child.is_file() else child.name,
                            path=str(child.resolve()),
                            package_source=source_label,
                            resource_type=resource_type,
                        )
                    )
    return result


def _installed_version(package_name: str) -> str | None:
    """Return the installed version of *package_name*, or ``None``."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_name_from_source(source: str) -> str:
    """Best-effort extraction of a pip package name from a source string.

    Handles ``package==1.0``, ``package>=1.0``, bare ``package``, etc.
    For git/local sources this is a heuristic (last path component).
    """
    # Strip version specifiers
    name = re.split(r"[>=<!~\[]", source, maxsplit=1)[0].strip()
    # For URLs / paths, grab the last meaningful segment
    if "/" in name:
        name = name.rstrip("/").rsplit("/", maxsplit=1)[-1]
    # Strip .git suffix
    if name.endswith(".git"):
        name = name[:-4]
    # Normalise per PEP 503
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------------------------------------------------------------------------
# Settings serialisation helpers
# ---------------------------------------------------------------------------


def _packages_from_settings(raw: list[Any]) -> list[ConfiguredPackage]:
    """Parse the ``packages`` array from settings.json."""
    result: list[ConfiguredPackage] = []
    for item in raw:
        if isinstance(item, str):
            result.append(
                ConfiguredPackage(
                    source=item,
                    source_type=classify_source(item),
                )
            )
        elif isinstance(item, dict) and "source" in item:
            result.append(
                ConfiguredPackage(
                    source=item["source"],
                    source_type=item.get("sourceType", classify_source(item["source"])),
                    enabled=item.get("enabled", True),
                )
            )
    return result


def _packages_to_settings(packages: list[ConfiguredPackage]) -> list[dict[str, Any]]:
    """Serialise configured packages back to a settings-compatible list."""
    return [
        {
            "source": p.source,
            "sourceType": p.source_type,
            "enabled": p.enabled,
        }
        for p in packages
    ]


# ---------------------------------------------------------------------------
# DefaultPackageManager
# ---------------------------------------------------------------------------


class DefaultPackageManager:
    """Default implementation using pip/uv for package management.

    Packages are persisted via :class:`SettingsManager` under the
    ``packages`` key. Resolved resources combine three layers in
    precedence order:

    1. **User-level** paths from ``~/.nu/agent/{extensions,skills,prompts,themes}``
       and from settings' ``extensions``/``skills``/``prompts``/``themes`` arrays.
    2. **Project-level** paths from ``<cwd>/.nu/`` subdirectories.
    3. **Package-installed** resources discovered via entry points and
       directory scanning.
    """

    def __init__(
        self,
        settings: SettingsManager,
        *,
        cwd: str | None = None,
        agent_dir: str | None = None,
    ) -> None:
        self._settings = settings
        self._cwd = cwd or str(Path.cwd())
        self._agent_dir = agent_dir or get_agent_dir()
        self._packages_dir = str(Path(self._agent_dir) / "packages")
        self._cached_resolved: ResolvedPaths | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_configured_packages(self) -> list[ConfiguredPackage]:
        return _packages_from_settings(self._settings.get_packages())

    def _save_packages(self, packages: list[ConfiguredPackage]) -> None:
        self._settings.set_packages(_packages_to_settings(packages))
        self._cached_resolved = None  # bust cache

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    async def install(self, source: str) -> None:
        """Install a package and add it to the configured list.

        * **pip**: runs ``uv pip install <source>``
        * **local**: validates the path exists, no install needed
        * **git**: runs ``uv pip install git+<url>``
        """
        source_type = classify_source(source)
        packages = self.get_configured_packages()

        # Check for duplicates
        normalised = _package_name_from_source(source)
        for pkg in packages:
            if _package_name_from_source(pkg.source) == normalised:
                if not pkg.enabled:
                    pkg.enabled = True
                    self._save_packages(packages)
                    logger.info("Re-enabled existing package: %s", source)
                    return
                logger.info("Package already installed: %s", source)
                return

        # Perform the actual installation
        if source_type == "pip":
            await _run_pip("install", source)
        elif source_type == "git":
            install_source = source if source.startswith("git+") else f"git+{source}"
            await _run_pip("install", install_source)
        elif source_type == "local":
            expanded = os.path.expanduser(source)  # noqa: ASYNC240
            if not os.path.exists(expanded):  # noqa: ASYNC240
                raise FileNotFoundError(f"Local package path does not exist: {source}")
            # Install in editable mode if it looks like a Python package
            has_pyproject = os.path.exists(os.path.join(expanded, "pyproject.toml"))  # noqa: ASYNC240
            has_setup = os.path.exists(os.path.join(expanded, "setup.py"))  # noqa: ASYNC240
            if has_pyproject or has_setup:
                await _run_pip("install", "-e", expanded)
            else:
                logger.info("Local path recorded (no pyproject.toml/setup.py): %s", source)

        packages.append(ConfiguredPackage(source=source, source_type=source_type, enabled=True))
        self._save_packages(packages)
        logger.info("Installed package: %s (%s)", source, source_type)

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    async def remove(self, source: str) -> None:
        """Uninstall a package and remove it from the configured list."""
        packages = self.get_configured_packages()
        normalised = _package_name_from_source(source)

        removed = False
        remaining: list[ConfiguredPackage] = []
        for pkg in packages:
            if _package_name_from_source(pkg.source) == normalised:
                removed = True
                # Actually uninstall pip/git packages
                if pkg.source_type in ("pip", "git"):
                    try:
                        await _run_pip("uninstall", "-y", _package_name_from_source(pkg.source))
                    except RuntimeError:
                        logger.warning("pip uninstall failed for %s (may already be removed)", pkg.source)
            else:
                remaining.append(pkg)

        if not removed:
            raise ValueError(f"Package not found in configuration: {source}")

        self._save_packages(remaining)
        logger.info("Removed package: %s", source)

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def enable(self, source: str) -> None:
        """Enable a previously disabled package."""
        self._set_enabled(source, enabled=True)

    def disable(self, source: str) -> None:
        """Disable a package without uninstalling it."""
        self._set_enabled(source, enabled=False)

    def _set_enabled(self, source: str, *, enabled: bool) -> None:
        packages = self.get_configured_packages()
        normalised = _package_name_from_source(source)
        found = False
        for pkg in packages:
            if _package_name_from_source(pkg.source) == normalised:
                pkg.enabled = enabled
                found = True
                break
        if not found:
            raise ValueError(f"Package not found in configuration: {source}")
        self._save_packages(packages)

    # ------------------------------------------------------------------
    # Update checking
    # ------------------------------------------------------------------

    async def check_updates(self) -> list[PackageUpdate]:
        """Check for available updates for all pip-installed packages."""
        packages = self.get_configured_packages()
        updates: list[PackageUpdate] = []

        for pkg in packages:
            if pkg.source_type != "pip" or not pkg.enabled:
                continue
            name = _package_name_from_source(pkg.source)
            current = _installed_version(name)
            if current is None:
                continue
            latest = await self._fetch_latest_version(name)
            if latest and latest != current:
                updates.append(
                    PackageUpdate(
                        source=pkg.source,
                        current_version=current,
                        latest_version=latest,
                    )
                )
        return updates

    @staticmethod
    async def _fetch_latest_version(package_name: str) -> str | None:
        """Query PyPI for the latest version of a package."""
        try:
            result = await _run_pip("index", "versions", package_name, check=False)
            if result.returncode != 0:
                return None
            # Output format: "package_name (X.Y.Z)"
            match = re.search(r"\(([^)]+)\)", result.stdout)
            return match.group(1) if match else None
        except Exception:
            logger.debug("Failed to check latest version for %s", package_name, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self) -> ResolvedPaths:
        """Resolve all resources from all sources, respecting precedence."""
        self._cached_resolved = None
        return self.get_resolved_paths()

    def get_resolved_paths(self) -> ResolvedPaths:
        """Return cached or freshly-resolved resource paths."""
        if self._cached_resolved is not None:
            return self._cached_resolved

        result = ResolvedPaths()

        # Layer 1: User-level paths (from settings + agent_dir wellknown dirs)
        user_resolved = self._resolve_user_level()
        result.merge(user_resolved)

        # Layer 2: Project-level paths (from <cwd>/.nu/)
        project_resolved = self._resolve_project_level()
        result.merge(project_resolved)

        # Layer 3: Package-installed resources
        package_resolved = self._resolve_packages()
        result.merge(package_resolved)

        self._cached_resolved = result
        return result

    def _resolve_user_level(self) -> ResolvedPaths:
        """Resolve resources from user-level configuration."""
        paths = ResolvedPaths()
        agent = Path(self._agent_dir)

        # Well-known directories under agent_dir + explicit settings paths
        _resource_map: list[tuple[str, ResourceType, list[str]]] = [
            ("tools", "extension", self._settings.get_extension_paths()),
            ("skills", "skill", self._settings.get_skill_paths()),
            ("prompts", "prompt", self._settings.get_prompt_template_paths()),
            ("themes", "theme", self._settings.get_theme_paths()),
        ]
        for subdir, rtype, settings_paths in _resource_map:
            # Scan well-known agent_dir subdirectory
            d = agent / subdir
            if d.is_dir():
                getattr(paths, f"{rtype}s").extend(
                    _resolve_user_paths([str(d)], rtype, "user")
                )
            # Explicit paths from settings
            if settings_paths:
                getattr(paths, f"{rtype}s").extend(
                    _resolve_user_paths(settings_paths, rtype, "user-settings")
                )

        return paths

    def _resolve_project_level(self) -> ResolvedPaths:
        """Resolve resources from project-level ``.nu/`` directory."""
        paths = ResolvedPaths()
        project_dir = Path(self._cwd) / CONFIG_DIR_NAME

        if not project_dir.is_dir():
            return paths

        for subdir, rtype in [
            ("extensions", "extension"),
            ("skills", "skill"),
            ("prompts", "prompt"),
            ("themes", "theme"),
        ]:
            d = project_dir / subdir
            if d.is_dir():
                resource_type: ResourceType = rtype  # type: ignore[assignment]
                items = _resolve_user_paths(
                    [str(d)],
                    resource_type,
                    source_label="project",
                )
                getattr(paths, f"{rtype}s").extend(items)

        return paths

    def _resolve_packages(self) -> ResolvedPaths:
        """Resolve resources from installed packages."""
        paths = ResolvedPaths()
        packages = self.get_configured_packages()

        for pkg in packages:
            if not pkg.enabled:
                continue

            # Discover extensions via entry points
            _package_name_from_source(pkg.source)
            self._resolve_entry_points(paths, pkg)

            # Scan package directory for resource subdirectories
            pkg_dir = self._find_package_dir(pkg)
            if pkg_dir:
                scanned = _scan_resource_dirs(pkg_dir, pkg.source)
                paths.merge(scanned)

        return paths

    def _resolve_entry_points(self, paths: ResolvedPaths, pkg: ConfiguredPackage) -> None:
        """Find extensions registered via entry points for a package."""
        try:
            eps = importlib.metadata.entry_points(group=EXTENSION_EP_GROUP)
        except TypeError:
            return

        pkg_name = _package_name_from_source(pkg.source)
        for ep in eps:
            # Match entry points belonging to this package's distribution
            try:
                ep_dist = ep.dist
                if ep_dist is None:
                    continue
                dist_name = re.sub(r"[-_.]+", "-", ep_dist.name).lower()
                if dist_name != pkg_name:
                    continue
            except Exception:
                # If we can't determine the distribution, skip matching
                continue

            paths.extensions.append(
                ResolvedResource(
                    name=ep.name,
                    path=ep.value,
                    package_source=pkg.source,
                    resource_type="extension",
                )
            )

    @staticmethod
    def _find_package_dir(pkg: ConfiguredPackage) -> Path | None:
        """Locate the on-disk directory for a package."""
        if pkg.source_type == "local":
            expanded = Path(os.path.expanduser(pkg.source))
            return expanded if expanded.is_dir() else None

        # For pip/git packages, use importlib.metadata to find the package location
        pkg_name = _package_name_from_source(pkg.source)
        try:
            dist = importlib.metadata.distribution(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            return None

        # Walk the recorded files to find the package's root directory
        files = dist.files
        if files:
            for f in files:
                located = f.locate()
                if Path(located).name == "__init__.py":
                    return Path(located).parent
        return None

    # ------------------------------------------------------------------
    # Metadata for source_info integration
    # ------------------------------------------------------------------

    def get_path_metadata(
        self,
        path: str,
        *,
        scope: SourceScope = "user",
    ) -> PathMetadata:
        """Build a :class:`PathMetadata` for the given resource *path*.

        This satisfies :class:`PathMetadataLike` from
        :mod:`nu_coding_agent.core.source_info` and lets the extension
        loader attach proper provenance to loaded resources.
        """
        # Determine if the path belongs to a package
        for pkg in self.get_configured_packages():
            pkg_dir = self._find_package_dir(pkg)
            if pkg_dir and path.startswith(str(pkg_dir)):
                return PathMetadata(
                    source=pkg.source,
                    scope=scope,
                    origin="package",
                    base_dir=str(pkg_dir),
                )

        # Determine scope from path location
        if path.startswith(self._agent_dir):
            scope = "user"
        elif path.startswith(str(Path(self._cwd) / CONFIG_DIR_NAME)):
            scope = "project"

        return PathMetadata(
            source="local",
            scope=scope,
            origin="top-level",
            base_dir=str(Path(path).parent) if os.path.exists(path) else None,
        )

    # ------------------------------------------------------------------
    # Summary / listing
    # ------------------------------------------------------------------

    def list_packages(self) -> list[dict[str, Any]]:
        """Return a summary of all configured packages with version info."""
        result: list[dict[str, Any]] = []
        for pkg in self.get_configured_packages():
            name = _package_name_from_source(pkg.source)
            version = _installed_version(name)
            result.append({
                "source": pkg.source,
                "sourceType": pkg.source_type,
                "enabled": pkg.enabled,
                "name": name,
                "version": version,
            })
        return result

    def invalidate_cache(self) -> None:
        """Force the next :meth:`get_resolved_paths` call to re-resolve."""
        self._cached_resolved = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_package_manager(
    settings: SettingsManager,
    *,
    cwd: str | None = None,
    agent_dir: str | None = None,
) -> DefaultPackageManager:
    """Convenience factory for :class:`DefaultPackageManager`."""
    return DefaultPackageManager(settings, cwd=cwd, agent_dir=agent_dir)


__all__ = [
    "EXTENSION_EP_GROUP",
    "ConfiguredPackage",
    "DefaultPackageManager",
    "PackageManager",
    "PackageUpdate",
    "PathMetadata",
    "ResolvedPaths",
    "ResolvedResource",
    "classify_source",
    "create_package_manager",
]
