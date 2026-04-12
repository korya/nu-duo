"""Tests for ``nu_coding_agent.core.extensions`` (slice-1 foundation).

Covers:

* ``load_extensions_from_factories`` round-trip — multiple factories,
  per-extension handler registration, error capture for crashing
  factories.
* ``load_extensions_from_paths`` — file-based discovery using a
  generated ``.py`` module under ``tmp_path``.
* ``ExtensionRunner.emit`` — sequential dispatch within an extension
  and across extensions, dataclass + raw-dict event payloads, sync
  + async handlers.
* Error capture: a misbehaving handler is reported via the listener
  and recorded in ``drain_errors`` without bringing the runner down.
* ``shutdown`` semantics: ``session_shutdown`` is broadcast iff there
  is at least one matching handler, and is idempotent.
* The deferred surface: ``register_tool`` / ``register_command`` /
  ``register_shortcut`` / ``register_flag`` / ``register_message_renderer``
  / ``get_flag`` accept input without crashing (data-only registration).
* ``wrapper.wrap_registered_tool`` raises ``NotImplementedError`` so
  the deferred surface is loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nu_coding_agent.core.extensions import (
    AgentEndEvent,
    AgentStartEvent,
    Extension,
    ExtensionAPI,
    ExtensionContext,
    ExtensionError,
    ExtensionRunner,
    ExtensionRuntime,
    MessageEndEvent,
    SessionShutdownEvent,
    discover_and_load_extensions,
    load_extensions_from_factories,
    load_extensions_from_paths,
)
from nu_coding_agent.core.extensions.wrapper import (
    wrap_registered_tool,
    wrap_registered_tools,
)

# ---------------------------------------------------------------------------
# load_extensions_from_factories
# ---------------------------------------------------------------------------


async def test_load_factory_registers_handlers() -> None:
    def register(api: ExtensionAPI) -> None:
        api.on("agent_start", lambda event, ctx: None)
        api.on("agent_end", lambda event, ctx: None)

    result = await load_extensions_from_factories([("<inline:test>", register)])
    assert result.errors == []
    assert len(result.extensions) == 1
    ext = result.extensions[0]
    assert ext.path == "<inline:test>"
    assert "agent_start" in ext.handlers
    assert "agent_end" in ext.handlers
    assert len(ext.handlers["agent_start"]) == 1


async def test_load_multiple_factories_in_order() -> None:
    def first(api: ExtensionAPI) -> None:
        api.on("message_end", lambda event, ctx: None)

    def second(api: ExtensionAPI) -> None:
        api.on("message_end", lambda event, ctx: None)

    result = await load_extensions_from_factories([("<inline:first>", first), ("<inline:second>", second)])
    assert [ext.path for ext in result.extensions] == ["<inline:first>", "<inline:second>"]


async def test_factory_exception_recorded_as_error() -> None:
    def crash(_api: ExtensionAPI) -> None:
        raise RuntimeError("boom")

    result = await load_extensions_from_factories([("<inline:crash>", crash)])
    assert result.extensions == []
    assert len(result.errors) == 1
    assert result.errors[0]["path"] == "<inline:crash>"
    assert "boom" in result.errors[0]["error"]


async def test_factory_runtime_is_shared_across_extensions() -> None:
    """All factories load against the same ``ExtensionRuntime`` instance."""
    runtime = ExtensionRuntime()

    def first(api: ExtensionAPI) -> None:
        api.register_flag("verbose", {"type": "boolean", "default": True})

    def second(api: ExtensionAPI) -> None:
        api.register_flag("dry_run", {"type": "boolean", "default": False})

    result = await load_extensions_from_factories(
        [("<inline:first>", first), ("<inline:second>", second)],
        runtime=runtime,
    )
    assert result.runtime is runtime
    assert runtime.flag_values == {"verbose": True, "dry_run": False}


async def test_async_factory_is_awaited() -> None:
    captured: list[str] = []

    async def register(api: ExtensionAPI) -> None:
        captured.append("registered")
        api.on("agent_start", lambda event, ctx: None)

    result = await load_extensions_from_factories([("<inline:async>", register)])
    assert captured == ["registered"]
    assert "agent_start" in result.extensions[0].handlers


# ---------------------------------------------------------------------------
# load_extensions_from_paths
# ---------------------------------------------------------------------------


_EXTENSION_MODULE_TEMPLATE = '''
"""Test extension."""

CALLED = []


def register(api):
    CALLED.append("registered")
    api.on("agent_start", lambda event, ctx: None)
'''


async def test_load_extensions_from_paths_loads_register_callable(tmp_path: Path) -> None:
    module = tmp_path / "ext_a.py"
    module.write_text(_EXTENSION_MODULE_TEMPLATE)

    result = await load_extensions_from_paths([str(module)])
    assert result.errors == []
    assert len(result.extensions) == 1
    assert result.extensions[0].path == str(module)
    assert "agent_start" in result.extensions[0].handlers


async def test_load_extensions_from_paths_missing_register_records_error(
    tmp_path: Path,
) -> None:
    module = tmp_path / "no_register.py"
    module.write_text('"""Empty extension."""\n')

    result = await load_extensions_from_paths([str(module)])
    assert result.extensions == []
    assert len(result.errors) == 1
    assert "register" in result.errors[0]["error"]


async def test_load_extensions_from_paths_invalid_file_records_error(
    tmp_path: Path,
) -> None:
    module = tmp_path / "broken.py"
    module.write_text("this is not valid python {{{\n")

    result = await load_extensions_from_paths([str(module)])
    assert result.extensions == []
    assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# discover_and_load_extensions (entry-point group)
# ---------------------------------------------------------------------------


async def test_discover_with_no_entry_points_returns_empty() -> None:
    """No registered entry points + no explicit paths → empty result."""
    result = await discover_and_load_extensions(
        entry_point_group="nu_coding_agent.extensions.does_not_exist",
    )
    assert result.extensions == []
    assert result.errors == []


async def test_discover_appends_explicit_paths(tmp_path: Path) -> None:
    module = tmp_path / "explicit_ext.py"
    module.write_text(_EXTENSION_MODULE_TEMPLATE)

    result = await discover_and_load_extensions(
        explicit_paths=[str(module)],
        entry_point_group="nu_coding_agent.extensions.does_not_exist",
    )
    assert len(result.extensions) == 1
    assert result.extensions[0].path == str(module)


# ---------------------------------------------------------------------------
# ExtensionRunner.emit — dispatch
# ---------------------------------------------------------------------------


async def test_runner_dispatches_dataclass_event() -> None:
    seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        api.on("agent_start", lambda event, ctx: seen.append(event))

    result = await load_extensions_from_factories([("<inline:test>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime, cwd="/work")
    await runner.emit(AgentStartEvent())
    assert len(seen) == 1
    assert isinstance(seen[0], AgentStartEvent)


async def test_runner_dispatches_dict_event() -> None:
    seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        api.on("custom_event", lambda event, ctx: seen.append(event))

    result = await load_extensions_from_factories([("<inline:test>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)
    await runner.emit({"type": "custom_event", "payload": {"x": 1}})
    assert seen == [{"type": "custom_event", "payload": {"x": 1}}]


async def test_runner_runs_handlers_in_registration_order() -> None:
    order: list[str] = []

    def first(api: ExtensionAPI) -> None:
        api.on("agent_end", lambda event, ctx: order.append("first"))

    def second(api: ExtensionAPI) -> None:
        api.on("agent_end", lambda event, ctx: order.append("second"))

    result = await load_extensions_from_factories([("<inline:first>", first), ("<inline:second>", second)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)
    await runner.emit(AgentEndEvent())
    assert order == ["first", "second"]


async def test_runner_supports_async_handlers() -> None:
    seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        async def handler(event: Any, ctx: ExtensionContext) -> None:
            seen.append((event, ctx.cwd))

        api.on("message_end", handler)

    result = await load_extensions_from_factories([("<inline:async>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime, cwd="/cwd")
    await runner.emit(MessageEndEvent(message={"role": "user"}))
    assert len(seen) == 1
    event, cwd = seen[0]
    assert isinstance(event, MessageEndEvent)
    assert cwd == "/cwd"


async def test_runner_skips_extensions_without_matching_handler() -> None:
    seen: list[str] = []

    def with_handler(api: ExtensionAPI) -> None:
        api.on("agent_start", lambda event, ctx: seen.append("with"))

    def without_handler(api: ExtensionAPI) -> None:
        api.on("agent_end", lambda event, ctx: seen.append("never"))

    result = await load_extensions_from_factories(
        [("<inline:with>", with_handler), ("<inline:without>", without_handler)]
    )
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)
    await runner.emit(AgentStartEvent())
    assert seen == ["with"]


async def test_runner_emit_no_handlers_is_noop() -> None:
    runner = ExtensionRunner.create()
    await runner.emit(AgentStartEvent())  # should not raise
    assert runner.drain_errors() == []


async def test_runner_emit_event_without_type_is_noop() -> None:
    runner = ExtensionRunner.create()
    await runner.emit({})  # type: ignore[arg-type]
    assert runner.drain_errors() == []


# ---------------------------------------------------------------------------
# ExtensionRunner — error tracking
# ---------------------------------------------------------------------------


async def test_handler_exception_is_captured_not_raised() -> None:
    def register(api: ExtensionAPI) -> None:
        def bad(event: Any, ctx: ExtensionContext) -> None:
            raise RuntimeError("kaboom")

        api.on("agent_start", bad)

    result = await load_extensions_from_factories([("<inline:bad>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)
    await runner.emit(AgentStartEvent())  # must not raise

    errors = runner.drain_errors()
    assert len(errors) == 1
    assert errors[0].extension_path == "<inline:bad>"
    assert errors[0].event == "agent_start"
    assert "kaboom" in errors[0].error


async def test_error_listener_is_invoked() -> None:
    def register(api: ExtensionAPI) -> None:
        def bad(event: Any, ctx: ExtensionContext) -> None:
            raise ValueError("nope")

        api.on("agent_start", bad)

    result = await load_extensions_from_factories([("<inline:bad>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)

    received: list[ExtensionError] = []
    unsubscribe = runner.on_error(received.append)

    await runner.emit(AgentStartEvent())
    assert len(received) == 1
    assert received[0].event == "agent_start"

    # Unsubscribed listeners stop hearing about new errors.
    unsubscribe()
    await runner.emit(AgentStartEvent())
    assert len(received) == 1


async def test_listener_exceptions_are_swallowed() -> None:
    """A broken listener must not crash the runner."""

    def register(api: ExtensionAPI) -> None:
        api.on("agent_start", lambda event, ctx: (_ for _ in ()).throw(RuntimeError("x")))

    result = await load_extensions_from_factories([("<inline:bad>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)

    def bad_listener(_err: ExtensionError) -> None:
        raise RuntimeError("listener exploded")

    runner.on_error(bad_listener)
    # Must not raise even though both handler and listener crash.
    await runner.emit(AgentStartEvent())
    assert len(runner.drain_errors()) == 1


async def test_one_bad_handler_does_not_block_others() -> None:
    seen: list[str] = []

    def register(api: ExtensionAPI) -> None:
        def bad(event: Any, ctx: ExtensionContext) -> None:
            raise RuntimeError("first handler exploded")

        def good(event: Any, ctx: ExtensionContext) -> None:
            seen.append("ran")

        api.on("agent_start", bad)
        api.on("agent_start", good)

    result = await load_extensions_from_factories([("<inline:both>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)
    await runner.emit(AgentStartEvent())
    assert seen == ["ran"]
    assert len(runner.drain_errors()) == 1


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def test_shutdown_emits_session_shutdown_event() -> None:
    seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        api.on("session_shutdown", lambda event, ctx: seen.append(event))

    result = await load_extensions_from_factories([("<inline:test>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)
    await runner.shutdown()
    assert len(seen) == 1
    assert isinstance(seen[0], SessionShutdownEvent)


async def test_shutdown_is_idempotent() -> None:
    seen: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        api.on("session_shutdown", lambda event, ctx: seen.append(event))

    result = await load_extensions_from_factories([("<inline:test>", register)])
    runner = ExtensionRunner.create(extensions=result.extensions, runtime=result.runtime)
    await runner.shutdown()
    await runner.shutdown()  # second call must be a no-op
    assert len(seen) == 1


async def test_shutdown_no_handler_is_noop() -> None:
    runner = ExtensionRunner.create()
    await runner.shutdown()  # must not raise
    assert runner.drain_errors() == []


# ---------------------------------------------------------------------------
# Deferred surface — registration methods accept input without crashing
# ---------------------------------------------------------------------------


async def test_register_tool_command_shortcut_flag_renderer() -> None:
    def register(api: ExtensionAPI) -> None:
        api.register_tool({"name": "todo", "description": "todo tool"})
        api.register_command("/snapshot", {"description": "snapshot the session"})
        api.register_shortcut("ctrl+shift+s", {"description": "save", "handler": lambda ctx: None})
        api.register_flag("verbose", {"type": "boolean", "default": True})
        api.register_message_renderer("custom_md", lambda msg: "rendered")

    result = await load_extensions_from_factories([("<inline:reg>", register)])
    ext = result.extensions[0]

    assert "todo" in ext.tools
    assert "/snapshot" in ext.commands
    assert "ctrl+shift+s" in ext.shortcuts
    assert "verbose" in ext.flags
    assert ext.flags["verbose"]["default"] is True
    assert "custom_md" in ext.message_renderers
    assert result.runtime.flag_values["verbose"] is True


async def test_get_flag_returns_runtime_value() -> None:
    runtime = ExtensionRuntime()
    runtime.flag_values["my_flag"] = "hello"

    captured: list[Any] = []

    def register(api: ExtensionAPI) -> None:
        api.register_flag("my_flag", {"type": "string", "default": "default-value"})
        captured.append(api.get_flag("my_flag"))
        captured.append(api.get_flag("not_registered"))

    await load_extensions_from_factories([("<inline:flag>", register)], runtime=runtime)
    # Pre-existing runtime value wins over the default ("hello", not "default-value").
    assert captured[0] == "hello"
    # Unregistered flags return None even if they exist in the runtime.
    assert captured[1] is None


async def test_register_tool_skips_anonymous_tools() -> None:
    def register(api: ExtensionAPI) -> None:
        api.register_tool({})  # no name → silently dropped

    result = await load_extensions_from_factories([("<inline:anon>", register)])
    assert result.extensions[0].tools == {}


# ---------------------------------------------------------------------------
# wrapper.py — deferred surface raises NotImplementedError loudly
# ---------------------------------------------------------------------------


def test_wrap_registered_tool_raises() -> None:
    runner = ExtensionRunner.create()
    with pytest.raises(NotImplementedError, match="extension tools"):
        wrap_registered_tool({"name": "x"}, runner)


def test_wrap_registered_tools_raises_on_first_entry() -> None:
    runner = ExtensionRunner.create()
    with pytest.raises(NotImplementedError):
        wrap_registered_tools([{"name": "x"}], runner)


# ---------------------------------------------------------------------------
# Runtime / context plumbing
# ---------------------------------------------------------------------------


def test_extension_runtime_default_state() -> None:
    runtime = ExtensionRuntime()
    assert runtime.flag_values == {}
    assert runtime.pending_provider_registrations == []


def test_extension_runner_create_context_uses_cwd() -> None:
    runner = ExtensionRunner.create(cwd="/work")
    ctx = runner.create_context("<inline:test>")
    assert isinstance(ctx, ExtensionContext)
    assert ctx.cwd == "/work"
    assert ctx.extension_path == "<inline:test>"


def test_extension_runner_extensions_property_is_a_copy() -> None:
    from nu_coding_agent.core.source_info import create_synthetic_source_info  # noqa: PLC0415

    ext = Extension(
        path="<inline:x>",
        resolved_path="<inline:x>",
        source_info=create_synthetic_source_info("<inline:x>", source="local"),
    )
    runner = ExtensionRunner.create(extensions=[ext])
    snapshot = runner.extensions
    snapshot.append(ext)  # mutating the snapshot must not affect the runner
    assert len(runner.extensions) == 1


def test_extension_runner_has_handlers_false_when_unregistered() -> None:
    runner = ExtensionRunner.create()
    assert runner.has_handlers("agent_start") is False


def test_extension_api_is_runtime_checkable_protocol() -> None:
    """``isinstance`` against the ``ExtensionAPI`` protocol is the contract
    a TS-shaped factory relies on; pin it so we never accidentally remove
    ``@runtime_checkable``."""
    from nu_coding_agent.core.extensions.loader import (  # noqa: PLC0415
        _ExtensionAPI,  # pyright: ignore[reportPrivateUsage]
    )

    runtime = ExtensionRuntime()
    from nu_coding_agent.core.source_info import create_synthetic_source_info  # noqa: PLC0415

    ext = Extension(
        path="<inline:x>",
        resolved_path="<inline:x>",
        source_info=create_synthetic_source_info("<inline:x>", source="local"),
    )
    api = _ExtensionAPI(ext, runtime)
    assert isinstance(api, ExtensionAPI)
