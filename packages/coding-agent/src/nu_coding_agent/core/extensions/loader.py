"""Extension loader — Python-flavoured port of ``packages/coding-agent/src/core/extensions/loader.ts``.

The TS loader uses ``jiti`` to compile TypeScript extension modules at
runtime, plus a ``virtualModules`` table so a compiled Bun binary can
serve bundled deps to extensions. Neither concept ports cleanly to
Python — the natural analogue is **Python entry points**: extensions
register themselves under a known group, get discovered via
:mod:`importlib.metadata`, and the loader simply imports the module and
calls its factory function. The porting plan called this out as the
"npm package loading → Python entry-point loading" mapping.

Three loading paths exist in this slice:

* :func:`load_extensions_from_factories` — accepts in-process factory
  callables. The simplest path; used by every test in this slice and
  the recommended pattern for embedding extensions inside an
  application that doesn't want to package them.
* :func:`load_extensions_from_paths` — accepts file paths to ``.py``
  modules. ``importlib.util`` loads each file, looks for a top-level
  ``register`` callable, and feeds it to the runner. Mirrors the TS
  ``loadExtensions(paths, …)`` pattern.
* :func:`discover_and_load_extensions` — walks the
  ``nu_coding_agent.extensions`` Python entry point group plus an
  optional list of explicit paths and combines them. The closest
  analogue to TS ``discoverAndLoadExtensions``.

Out of scope for this slice (follow-up):

* The TS loader's ``package.json`` "pi" manifest discovery — replaced
  here by the entry point group, which is the natural Python idiom.
* The ``virtualModules`` machinery — Python doesn't need it; extensions
  just ``import nu_coding_agent`` directly.
* The runtime "queue then bind" provider-registration dance — that
  ships alongside the model registry binding in the next slice.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from nu_coding_agent.core.extensions.types import (
    Extension,
    ExtensionAPI,
    ExtensionFactory,
    ExtensionHandler,
    ExtensionRuntime,
    LoadExtensionsResult,
)
from nu_coding_agent.core.source_info import create_synthetic_source_info

#: Entry point group name extensions register themselves under.
#: Python distributions add this to their ``[project.entry-points]``
#: table to be discovered automatically.
ENTRY_POINT_GROUP = "nu_coding_agent.extensions"


# ---------------------------------------------------------------------------
# ExtensionAPI implementation
# ---------------------------------------------------------------------------


class _ExtensionAPI:
    """Concrete :class:`ExtensionAPI` handed to a factory function.

    Mirrors the TS ``createExtensionAPI`` closure: registration methods
    write to the supplied :class:`Extension`, action methods would
    delegate to the runtime (deferred — currently no-ops).
    """

    def __init__(self, extension: Extension, runtime: ExtensionRuntime) -> None:
        self._extension = extension
        self._runtime = runtime

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    def on(self, event: str, handler: ExtensionHandler) -> None:
        bucket = self._extension.handlers.setdefault(event, [])
        bucket.append(handler)

    # ------------------------------------------------------------------
    # Tool / command / shortcut / flag / renderer registration
    # ------------------------------------------------------------------

    def register_tool(self, tool: Any) -> None:
        # Tools are stored by name so a follow-up slice can wrap them
        # into AgentTools via tool-definition-wrapper. For now we just
        # accept and remember them.
        name = getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None)
        if not name:
            return
        self._extension.tools[str(name)] = tool

    def register_command(self, name: str, options: dict[str, Any]) -> None:
        self._extension.commands[name] = {
            "name": name,
            "source_info": self._extension.source_info,
            **options,
        }

    def register_shortcut(self, shortcut: str, options: dict[str, Any]) -> None:
        self._extension.shortcuts[shortcut] = {
            "shortcut": shortcut,
            "extension_path": self._extension.path,
            **options,
        }

    def register_flag(self, name: str, options: dict[str, Any]) -> None:
        self._extension.flags[name] = {
            "name": name,
            "extension_path": self._extension.path,
            **options,
        }
        default = options.get("default")
        if default is not None and name not in self._runtime.flag_values:
            self._runtime.flag_values[name] = default

    def register_message_renderer(self, custom_type: str, renderer: Any) -> None:
        self._extension.message_renderers[custom_type] = renderer

    # ------------------------------------------------------------------
    # Flag access
    # ------------------------------------------------------------------

    def get_flag(self, name: str) -> bool | str | None:
        if name not in self._extension.flags:
            return None
        return self._runtime.flag_values.get(name)

    # ------------------------------------------------------------------
    # Action methods — delegate to the runtime slots that
    # ``ExtensionRunner.bind_core`` populates with session-aware
    # implementations. Until then, the runtime slots are throwing
    # stubs from ``types._unbound`` so a misuse fires loudly.
    # ------------------------------------------------------------------

    def set_label(self, entry_id: str, label: str | None) -> None:
        self._runtime.set_label(entry_id, label)

    def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return self._runtime.append_custom_entry(custom_type, data)

    def set_session_name(self, name: str) -> None:
        self._runtime.set_session_name(name)

    def get_session_name(self) -> str | None:
        return self._runtime.get_session_name()

    def get_active_tools(self) -> list[str]:
        return self._runtime.get_active_tools()

    def get_all_tools(self) -> list[dict[str, Any]]:
        return self._runtime.get_all_tools()

    def set_active_tools(self, tool_names: list[str]) -> None:
        self._runtime.set_active_tools(tool_names)

    async def set_model(self, model: Any) -> bool:
        return await self._runtime.set_model(model)

    def get_thinking_level(self) -> str:
        return self._runtime.get_thinking_level()

    def set_thinking_level(self, level: str) -> None:
        self._runtime.set_thinking_level(level)


# ---------------------------------------------------------------------------
# Extension creation
# ---------------------------------------------------------------------------


def _create_extension(path: str, resolved_path: str) -> Extension:
    """Build an empty :class:`Extension` for ``path``.

    The synthetic ``SourceInfo`` mirrors what TS ``createExtension``
    produces — we don't have a package manager yet to attach a real
    metadata record, so a synthetic one keeps the slot populated for
    downstream code (e.g. ``register_command`` writes ``source_info``
    onto each command record).
    """
    base_dir = None if path.startswith("<") else str(Path(resolved_path).parent)
    source = (
        path[1:-1].split(":", maxsplit=1)[0] or "temporary" if path.startswith("<") and path.endswith(">") else "local"
    )
    return Extension(
        path=path,
        resolved_path=resolved_path,
        source_info=create_synthetic_source_info(path, source=source, base_dir=base_dir),
    )


# ---------------------------------------------------------------------------
# Loader entry points
# ---------------------------------------------------------------------------


async def _invoke_factory(api: ExtensionAPI, factory: ExtensionFactory) -> None:
    result = factory(api)
    if inspect.isawaitable(result):
        await result


async def load_extensions_from_factories(
    factories: list[tuple[str, ExtensionFactory]],
    runtime: ExtensionRuntime | None = None,
) -> LoadExtensionsResult:
    """Load a list of in-process extension factories.

    Each entry is a ``(name, factory)`` tuple. ``name`` becomes the
    extension's ``path`` (the conventional ``"<inline:foo>"`` form is
    fine for tests). The factory is called with a fresh
    :class:`_ExtensionAPI`; any handlers / tools / commands it registers
    end up on the returned :class:`Extension`.
    """
    runtime = runtime or ExtensionRuntime()
    extensions: list[Extension] = []
    errors: list[dict[str, str]] = []

    for name, factory in factories:
        extension = _create_extension(name, name)
        api = _ExtensionAPI(extension, runtime)
        try:
            await _invoke_factory(api, factory)
        except Exception as exc:
            errors.append({"path": name, "error": f"Failed to load extension: {exc}"})
            continue
        extensions.append(extension)

    return LoadExtensionsResult(extensions=extensions, errors=errors, runtime=runtime)


async def load_extensions_from_paths(
    paths: list[str],
    runtime: ExtensionRuntime | None = None,
) -> LoadExtensionsResult:
    """Load extensions from filesystem paths to ``.py`` modules.

    Each path must point at a Python module that exposes a top-level
    ``register`` callable matching :type:`ExtensionFactory`. The module
    is loaded via :func:`importlib.util.spec_from_file_location`, so it
    does not need to be on ``sys.path``.
    """
    runtime = runtime or ExtensionRuntime()
    extensions: list[Extension] = []
    errors: list[dict[str, str]] = []

    for path in paths:
        try:
            factory = _load_factory_from_path(path)
        except Exception as exc:
            errors.append({"path": path, "error": f"Failed to load extension: {exc}"})
            continue
        if factory is None:
            errors.append(
                {
                    "path": path,
                    "error": (f"Extension does not export a 'register' callable: {path}"),
                }
            )
            continue

        extension = _create_extension(path, str(Path(path).resolve()))  # noqa: ASYNC240
        api = _ExtensionAPI(extension, runtime)
        try:
            await _invoke_factory(api, factory)
        except Exception as exc:
            errors.append({"path": path, "error": f"Failed to load extension: {exc}"})
            continue
        extensions.append(extension)

    return LoadExtensionsResult(extensions=extensions, errors=errors, runtime=runtime)


def _load_factory_from_path(path: str) -> ExtensionFactory | None:
    resolved = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"_nu_extension_{resolved.stem}", str(resolved))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "register", None)
    if not callable(factory):
        return None
    return factory  # type: ignore[return-value]


async def discover_and_load_extensions(
    explicit_paths: list[str] | None = None,
    runtime: ExtensionRuntime | None = None,
    *,
    entry_point_group: str = ENTRY_POINT_GROUP,
) -> LoadExtensionsResult:
    """Discover extensions via Python entry points + explicit paths.

    The discovery walks ``importlib.metadata.entry_points(group=...)``
    and treats every match as an importable callable that mirrors the
    :type:`ExtensionFactory` shape. Errors are recorded but never raise.
    Explicit paths from the ``--extension`` CLI flag (eventually) are
    appended after the entry-point set so a project-local extension can
    take precedence.
    """
    runtime = runtime or ExtensionRuntime()
    extensions: list[Extension] = []
    errors: list[dict[str, str]] = []

    for ep in _safe_entry_points(entry_point_group):
        try:
            loaded = ep.load()
        except Exception as exc:
            errors.append({"path": f"{ep.name} ({ep.value})", "error": f"Failed to load extension: {exc}"})
            continue
        if not callable(loaded):
            errors.append(
                {
                    "path": f"{ep.name} ({ep.value})",
                    "error": "Entry point did not resolve to a callable",
                }
            )
            continue
        factory: ExtensionFactory = loaded  # type: ignore[assignment]
        path_label = f"<entry_point:{ep.name}>"
        extension = _create_extension(path_label, ep.value)
        api = _ExtensionAPI(extension, runtime)
        try:
            await _invoke_factory(api, factory)
        except Exception as exc:
            errors.append({"path": path_label, "error": f"Failed to load extension: {exc}"})
            continue
        extensions.append(extension)

    if explicit_paths:
        path_result = await load_extensions_from_paths(explicit_paths, runtime=runtime)
        extensions.extend(path_result.extensions)
        errors.extend(path_result.errors)

    return LoadExtensionsResult(extensions=extensions, errors=errors, runtime=runtime)


def _safe_entry_points(group: str) -> list[Any]:
    """Return the entry points for ``group`` or an empty list on stdlib variance."""
    try:
        eps = entry_points(group=group)
    except TypeError:
        # Older importlib.metadata doesn't accept ``group=`` as a kwarg.
        # We're on Python 3.13 so this branch shouldn't fire, but it's
        # cheap defensive cover.
        return []
    return list(eps)


# ---------------------------------------------------------------------------
# Imports kept alive for downstream consumers
# ---------------------------------------------------------------------------


_ = importlib  # pragma: no cover
