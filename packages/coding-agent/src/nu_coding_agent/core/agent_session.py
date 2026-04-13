"""Simplified ``AgentSession`` — port of ``packages/coding-agent/src/core/agent-session.ts``.

This is a *focused* port. The upstream class is 3059 LoC and pulls in
extensions, retry, navigation, HTML export, bash exec, slash command
expansion, and skill-block parsing. Many of those depend on subsystems
the Python port hasn't reached yet (extensions/, package_manager,
modes/interactive, …). This port covers the core lifecycle that
``modes/print_mode`` and the eventual ``modes/rpc`` actually need:

* Wraps an existing :class:`nu_agent_core.agent.Agent` together with
  a :class:`nu_coding_agent.core.session_manager.SessionManager`,
  :class:`ModelRegistry`, and :class:`AuthStorage`.
* ``prompt(text, images)`` runs one full turn end-to-end:
  validates credentials → appends a user message to the session →
  calls ``agent.prompt`` → persists every assistant + tool-result
  message that lands.
* ``set_model(model)`` swaps the active model on the agent's state
  and emits a ``model_change`` session entry so a resume picks the
  same model.
* ``get_stats()`` returns the same :class:`SessionStats` shape the
  upstream's ``/session`` command shows.
* The active event listener subscribes once on construction and
  cleans itself up via :meth:`close`.

Once :mod:`nu_coding_agent.core.extensions` and the runtime port
land, this module gets folded back together with the full upstream
surface (compaction triggers, retry, slash-command dispatch, …).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nu_coding_agent.core.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    calculate_context_tokens,
    compact,
    estimate_context_tokens,
    get_last_assistant_usage,
    prepare_compaction,
    should_compact,
)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable

    from nu_agent_core.agent import Agent
    from nu_agent_core.types import AgentEvent
    from nu_ai.types import ImageContent, Model

    from nu_coding_agent.core.auth_storage import AuthStorage
    from nu_coding_agent.core.compaction import CompactionPreparation, CompactionResult
    from nu_coding_agent.core.extensions import ExtensionRunner, LifecycleEvent
    from nu_coding_agent.core.model_registry import ModelRegistry
    from nu_coding_agent.core.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionStats:
    """The same shape the upstream ``/session`` command emits."""

    session_file: str | None
    session_id: str
    user_messages: int
    assistant_messages: int
    tool_calls: int
    tool_results: int
    total_messages: int
    tokens_input: int
    tokens_output: int
    tokens_cache_read: int
    tokens_cache_write: int
    tokens_total: int
    cost: float


@dataclass(slots=True)
class AgentSessionConfig:
    """Constructor knobs for :class:`AgentSession`."""

    agent: Agent
    session_manager: SessionManager
    model_registry: ModelRegistry
    auth_storage: AuthStorage
    cwd: str
    compaction_settings: CompactionSettings = field(default_factory=lambda: DEFAULT_COMPACTION_SETTINGS)
    #: Optional :class:`ExtensionRunner` that receives lifecycle events as
    #: the agent loop runs. ``None`` (the default) preserves the
    #: pre-extensions behaviour exactly — no events are dispatched.
    extension_runner: ExtensionRunner | None = None


# Listener: receives every :class:`AgentEvent` synchronously after the
# session has already persisted it. Mirrors the upstream's "subscribe to
# session-aware events" pattern. Returning ``None`` is fine; returning
# an awaitable is also fine — it gets awaited.
type AgentSessionListener = Callable[[AgentEvent], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# AgentSession
# ---------------------------------------------------------------------------


class AgentSession:
    """Glues an :class:`Agent` to a :class:`SessionManager` lifecycle."""

    def __init__(self, config: AgentSessionConfig) -> None:
        self._agent = config.agent
        self._session_manager = config.session_manager
        self._model_registry = config.model_registry
        self._auth_storage = config.auth_storage
        self._cwd = config.cwd
        self._compaction_settings = config.compaction_settings
        self._extension_runner = config.extension_runner

        self._listeners: list[AgentSessionListener] = []
        self._closed = False
        # Tracks whether ``session_start`` has fired so we only emit it
        # once per AgentSession lifetime, on the first prompt.
        self._extensions_started = False
        self._is_compacting = False

        # Session-level state (matches upstream properties)
        self._session_name: str | None = None
        self._thinking_level: str = "off"
        self._steering_mode: str = "all"
        self._follow_up_mode: str = "all"
        self._auto_compaction_enabled: bool = True
        self._auto_retry_enabled: bool = True

        # The persisting listener runs first so user-supplied subscribers
        # always observe events that are already on disk.
        self._unsubscribe_agent: Callable[[], None] | None = self._agent.subscribe(self._handle_agent_event)

        # Wire the runtime action methods to this session so attached
        # extensions can call back into us. Safe to call before any
        # extensions exist — bind_core just replaces the throwing-stub
        # action slots on the runtime with bound functions.
        if self._extension_runner is not None:
            self._extension_runner.bind_core(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        agent: Agent,
        session_manager: SessionManager,
        model_registry: ModelRegistry,
        auth_storage: AuthStorage,
        cwd: str,
        compaction_settings: CompactionSettings | None = None,
        extension_runner: ExtensionRunner | None = None,
    ) -> AgentSession:
        """Build an :class:`AgentSession` from the four required collaborators."""
        return cls(
            AgentSessionConfig(
                agent=agent,
                session_manager=session_manager,
                model_registry=model_registry,
                auth_storage=auth_storage,
                cwd=cwd,
                compaction_settings=compaction_settings or DEFAULT_COMPACTION_SETTINGS,
                extension_runner=extension_runner,
            )
        )

    def close(self) -> None:
        """Detach the session listener from the agent. Safe to call twice.

        Sync close: only unsubscribes from the agent. To also broadcast
        ``session_shutdown`` to attached extensions, use
        :meth:`shutdown` (which is async). Existing call sites that
        don't use extensions can keep calling ``close()`` directly.
        """
        if self._closed:
            return
        self._closed = True
        if self._unsubscribe_agent is not None:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None

    async def shutdown(self) -> None:
        """Async close: emit ``session_shutdown`` to extensions, then detach.

        Idempotent — safe to call multiple times. Always invokes
        :meth:`close` at the end so the agent listener is detached even
        if there is no extension runner attached.
        """
        if self._extension_runner is not None and self._extensions_started:
            await self._extension_runner.shutdown()
        self.close()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    def set_session_manager(self, sm: SessionManager) -> None:
        """Replace the session manager (used by runtime for switch/fork)."""
        self._session_manager = sm

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry

    @property
    def auth_storage(self) -> AuthStorage:
        return self._auth_storage

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def extension_runner(self) -> ExtensionRunner | None:
        """Return the attached :class:`ExtensionRunner`, if any."""
        return self._extension_runner

    @property
    def model(self) -> Model | None:
        """Active model, or ``None`` if the agent is using its placeholder default."""
        candidate = self._agent.state.model
        if candidate.api == "unknown":
            return None
        return candidate

    # ------------------------------------------------------------------
    # Session info / state
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The session's unique identifier."""
        return self._session_manager.get_session_id()

    @property
    def session_name(self) -> str | None:
        """User-assigned display name for this session."""
        return self._session_name

    def set_session_name(self, name: str) -> None:
        """Set the session's display name."""
        self._session_name = name

    @property
    def session_file(self) -> str | None:
        """Path to the session JSONL file, if persisted."""
        return self._session_manager.get_session_file()

    @property
    def is_streaming(self) -> bool:
        """Whether the agent is currently generating a response."""
        # The agent has a signal Event that exists while running
        return self._agent.signal is not None

    @property
    def is_compacting(self) -> bool:
        """Whether compaction is in progress."""
        return self._is_compacting

    @property
    def pending_message_count(self) -> int:
        """Number of queued steering/follow-up messages."""
        return 0  # Queuing is managed by the agent itself

    @property
    def messages(self) -> list[Any]:
        """Shortcut for ``agent.state.messages``."""
        return list(self._agent.state.messages)

    # ------------------------------------------------------------------
    # Thinking level
    # ------------------------------------------------------------------

    @property
    def thinking_level(self) -> str:
        """Current thinking level (off, low, medium, high)."""
        return self._thinking_level

    def set_thinking_level(self, level: str) -> None:
        """Set the thinking level."""
        self._thinking_level = level

    def cycle_thinking_level(self) -> str | None:
        """Cycle through available thinking levels. Returns new level or None."""
        levels = ["off", "low", "medium", "high"]
        try:
            idx = levels.index(self._thinking_level)
        except ValueError:
            idx = 0
        next_idx = (idx + 1) % len(levels)
        self._thinking_level = levels[next_idx]
        return self._thinking_level

    # ------------------------------------------------------------------
    # Queue modes
    # ------------------------------------------------------------------

    @property
    def steering_mode(self) -> str:
        """How steering messages are processed: 'all' or 'one-at-a-time'."""
        return self._steering_mode

    def set_steering_mode(self, mode: str) -> None:
        self._steering_mode = mode

    @property
    def follow_up_mode(self) -> str:
        """How follow-up messages are processed: 'all' or 'one-at-a-time'."""
        return self._follow_up_mode

    def set_follow_up_mode(self, mode: str) -> None:
        self._follow_up_mode = mode

    # ------------------------------------------------------------------
    # Auto-compaction / auto-retry
    # ------------------------------------------------------------------

    @property
    def auto_compaction_enabled(self) -> bool:
        return self._auto_compaction_enabled

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        self._auto_compaction_enabled = enabled

    @property
    def auto_retry_enabled(self) -> bool:
        return self._auto_retry_enabled

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        self._auto_retry_enabled = enabled

    def abort_retry(self) -> None:
        """Abort any in-progress retry."""
        # Retry state management is a future enhancement

    # ------------------------------------------------------------------
    # Steer / follow-up / abort
    # ------------------------------------------------------------------

    def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Send a steering message to interrupt the current agent turn."""
        import time as _time  # noqa: PLC0415

        from nu_ai import TextContent, UserMessage  # noqa: PLC0415

        content: list[Any] = [TextContent(type="text", text=text)]
        msg = UserMessage(role="user", content=content, timestamp=int(_time.time() * 1000))
        self._agent.steer(msg)

    def follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Queue a follow-up message for after the agent finishes."""
        import time as _time  # noqa: PLC0415

        from nu_ai import TextContent, UserMessage  # noqa: PLC0415

        content: list[Any] = [TextContent(type="text", text=text)]
        msg = UserMessage(role="user", content=content, timestamp=int(_time.time() * 1000))
        self._agent.follow_up(msg)

    def abort(self) -> None:
        """Stop the current agent turn."""
        self._agent.abort()

    # ------------------------------------------------------------------
    # Bash execution
    # ------------------------------------------------------------------

    async def execute_bash(self, command: str) -> dict[str, Any]:
        """Execute a bash command and return the result.

        Returns a dict with output, exit_code, cancelled, truncated fields.
        """
        from nu_coding_agent.core.bash_executor import execute_bash as _exec_bash  # noqa: PLC0415

        result = await _exec_bash(command)
        return {
            "output": result.output,
            "exitCode": result.exit_code,
            "cancelled": result.cancelled,
            "truncated": result.truncated,
        }

    def abort_bash(self) -> None:
        """Abort the running bash command, if any."""
        # Bash abort is a future enhancement

    # ------------------------------------------------------------------
    # Session utilities
    # ------------------------------------------------------------------

    def get_last_assistant_text(self) -> str | None:
        """Get the text content of the last assistant message."""
        from nu_ai import AssistantMessage, TextContent  # noqa: PLC0415

        for msg in reversed(self._agent.state.messages):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextContent):
                        return block.text
        return None

    def get_user_messages_for_forking(self) -> list[dict[str, str]]:
        """List user messages suitable for the fork point selector.

        Returns a list of dicts with ``entryId`` and ``text`` keys.
        """
        result: list[dict[str, str]] = []
        for entry in self._session_manager.get_entries():
            if entry.get("type") != "message":
                continue
            msg = entry.get("message")
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            break
                entry_id = entry.get("id", "")
                if entry_id and text:
                    result.append({"entryId": entry_id, "text": text})
        return result

    async def reload(self) -> None:
        """Reload all resources and extensions at runtime."""
        # Extension reload is a future enhancement

    async def bind_extensions(self, **kwargs: Any) -> None:
        """Bind extension UI context and command handlers.

        This is a simplified version — full extension binding
        with UI context, command actions, and shutdown handlers
        will be expanded as the extension system matures.
        """
        # Accept but don't use UI context for now

    # ------------------------------------------------------------------
    # HTML export
    # ------------------------------------------------------------------

    async def export_to_html(
        self,
        output_path: str | None = None,
        theme_name: str | None = None,
    ) -> str:
        """Export this session to a self-contained HTML file.

        Returns the path of the written HTML file.
        """
        from nu_coding_agent.core.export_html import ExportOptions, export_session_to_html  # noqa: PLC0415

        opts = ExportOptions(output_path=output_path, theme_name=theme_name)
        return await export_session_to_html(self.session_manager, self.agent.state, opts)

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    def subscribe(self, listener: AgentSessionListener) -> Callable[[], None]:
        """Register a listener for every :class:`AgentEvent` (after persistence)."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    async def _handle_agent_event(self, event: AgentEvent, _signal: asyncio.Event) -> None:
        """Persist the event into the session, dispatch to extensions, then user listeners.

        Order is load-bearing:

        1. **Persist first** — the session JSONL is the source of truth.
           User listeners and extension hooks must never observe events
           that aren't already durably on disk.
        2. **Then dispatch to extensions** — they get a consistent view
           of the session state and can react with their own writes.
        3. **Then user listeners** — anything subscribed via
           :meth:`subscribe` runs last so it can rely on both
           persistence and extension reactions having completed.
        """
        await self._persist_event(event)
        await self._dispatch_extension_event(event)
        for listener in list(self._listeners):
            try:
                result = listener(event)
                if result is not None:
                    await result
            except Exception as exc:
                # Mirrors the upstream's "swallow + log" semantics so a
                # broken subscriber doesn't take the agent down.
                import sys  # noqa: PLC0415

                print(f"AgentSession listener error: {exc}", file=sys.stderr)

    async def _dispatch_extension_event(self, event: AgentEvent) -> None:
        """Translate an :class:`AgentEvent` and emit it to the extension runner.

        The agent loop's :class:`AgentEvent` types and the extension
        :class:`LifecycleEvent` dataclasses share the same ``type``
        discriminator strings (``agent_start`` / ``message_end`` / ...),
        so this is a straight 1:1 mapping. Returns immediately when no
        runner is attached so the per-event hot path stays free for
        the no-extensions case.
        """
        if self._extension_runner is None:
            return
        translated = _translate_to_extension_event(event)
        if translated is not None:
            await self._extension_runner.emit(translated)

    async def _persist_event(self, event: AgentEvent) -> None:
        """Persist ``message_end`` events into the session JSONL.

        The agent loop emits a ``message_end`` for every message it
        processes — user prompts, assistant replies, AND
        :class:`ToolResultMessage` entries. Catching that single event
        type is enough to mirror the conversation tree onto disk; the
        ``tool_execution_end`` event is just metadata for UI renderers
        and would double-write tool results if persisted here too.
        """
        if event["type"] != "message_end":
            return
        self._session_manager.append_message(event["message"])

    # ------------------------------------------------------------------
    # prompt() / set_model()
    # ------------------------------------------------------------------

    async def prompt(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
    ) -> None:
        """Run one full turn against the agent, persisting along the way.

        Validates that:
        1. A model is selected on the agent.
        2. The model's provider has credentials configured (api key,
           OAuth token, or fallback resolver).

        The user message is appended to the session manually before
        ``agent.prompt`` runs because the agent's persist-on-event
        pipeline only catches the assistant + tool result entries —
        the prompt itself never produces a ``message_end`` event.
        """
        if self._closed:
            raise RuntimeError("AgentSession is closed")

        model = self._agent.state.model
        # The Agent class swaps in a placeholder "unknown" model when
        # constructed without one — treat that as "no model selected".
        if model.api == "unknown":
            raise ValueError("No model selected on the agent.")
        if not self._model_registry.has_configured_auth(model):
            raise ValueError(
                f'No API key configured for provider "{model.provider}". Set the corresponding env var or run /login.'
            )

        # Fire ``session_start`` exactly once, lazily, on the first
        # prompt. We can't do this from ``__init__`` because it's sync
        # and emit() is async; deferring to the first prompt also
        # mirrors the upstream lifecycle (a session that's constructed
        # but never prompted should not produce a session_start event).
        await self._ensure_extensions_started()

        # The agent loop emits a ``message_end`` for the user prompt
        # itself; the listener catches it and persists it. We don't
        # build the UserMessage manually here because doing so would
        # double-write it into the session.
        if images:
            # Image attachments aren't yet wired through Agent.prompt(),
            # so this is a forward-compat check rather than a feature.
            raise NotImplementedError("Image attachments are not yet supported by AgentSession.prompt().")
        await self._agent.prompt(text)

    async def _ensure_extensions_started(self) -> None:
        """Lazily emit ``session_start`` to the runner on the first prompt."""
        if self._extension_runner is None or self._extensions_started:
            return
        self._extensions_started = True
        from nu_coding_agent.core.extensions import SessionStartEvent  # noqa: PLC0415

        await self._extension_runner.emit(
            SessionStartEvent(
                cwd=self._cwd,
                session_id=self._session_manager.get_session_id(),
            )
        )

    def set_model(self, model: Model) -> None:
        """Swap the active model and persist a model_change session entry."""
        self._agent.state.model = model
        self._session_manager.append_model_change(model.provider, model.id)

    def apply_extension_tools(self) -> int:
        """Append every extension-registered tool to ``agent.state.tools``.

        Walks the attached extension runner (if any), wraps every
        registered tool via :func:`wrap_registered_tool`, and appends
        the result to the agent's active tool list. Tools whose name
        already exists in the agent's tool list are *replaced*, so
        extensions can override built-in tools by name. Returns the
        number of tools appended/overridden so callers can log it.

        Idempotent: calling this twice with the same set of extensions
        replaces the same tools by name without appending duplicates.
        Safe to call before or after the first prompt; the agent loop
        reads ``agent.state.tools`` per turn.

        No-op if no extension runner is attached.
        """
        if self._extension_runner is None:
            return 0
        from nu_coding_agent.core.extensions.wrapper import (  # noqa: PLC0415
            wrap_registered_tool,
        )

        existing = list(self._agent.state.tools)
        existing_by_name: dict[str, int] = {tool.name: idx for idx, tool in enumerate(existing)}
        applied = 0
        for raw in self._extension_runner.get_all_registered_tools():
            wrapped = wrap_registered_tool(raw, self._extension_runner)
            name = getattr(wrapped, "name", None)
            if not name:
                continue
            if name in existing_by_name:
                existing[existing_by_name[name]] = wrapped
            else:
                existing_by_name[name] = len(existing)
                existing.append(wrapped)
            applied += 1
        # The agent state copies the list on assignment so future
        # mutations to ``existing`` after this point would not bleed
        # through.
        self._agent.state.tools = existing
        return applied

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def should_compact(self) -> bool:
        """Heuristic check used by autonomous runtimes to decide whether to compact."""
        model = self._agent.state.model
        if model.api == "unknown":
            return False
        usage = get_last_assistant_usage(self._session_manager.get_entries())
        if usage is None:
            return False
        context_tokens = calculate_context_tokens(usage)
        return should_compact(context_tokens, model.context_window, self._compaction_settings)

    def prepare_compaction(self) -> CompactionPreparation | None:
        """Pure helper that pickets a cut point on the current branch."""
        path = self._session_manager.get_branch()
        return prepare_compaction(path, self._compaction_settings)

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult | None:
        """Drive the LLM to summarise + persist a compaction entry.

        Extension hooks fire around the LLM call:

        * ``session_before_compact`` is emitted with the prepared
          :class:`CompactionPreparation` and the current branch.
          Handlers may return a :class:`SessionBeforeCompactResult`
          (or a plain dict with ``cancel`` / ``compaction`` keys)  to
          either cancel the compaction entirely or supply a custom
          :class:`CompactionResult` that bypasses the LLM
          summarisation.

        * ``session_compact`` is emitted after the compaction entry
          is persisted, with ``from_extension=True`` if an extension
          handler supplied the result.
        """
        prep = self.prepare_compaction()
        if prep is None:
            return None

        self._is_compacting = True
        try:
            return await self._run_compact(prep, custom_instructions)
        finally:
            self._is_compacting = False

    async def _run_compact(
        self,
        prep: CompactionPreparation,
        custom_instructions: str | None,
    ) -> CompactionResult | None:
        # Hook 1: before_compact — handlers may cancel or replace.
        result, from_extension = await self._dispatch_before_compact(prep, custom_instructions)
        if result is None and from_extension is False:
            # Standard path: drive the LLM via compact().
            model = self._agent.state.model
            if model.api == "unknown":
                raise ValueError("No model selected on the agent.")
            api_key = await self._auth_storage.get_api_key(model.provider)
            if api_key is None:
                raise ValueError(f'No API key for provider "{model.provider}" — cannot compact.')
            result = await compact(
                preparation=prep,
                model=model,
                api_key=api_key,
                custom_instructions=custom_instructions,
            )
        elif result is None:
            # Cancelled by an extension.
            return None

        compaction_id = self._session_manager.append_compaction(
            summary=result.summary,
            first_kept_entry_id=result.first_kept_entry_id,
            tokens_before=result.tokens_before,
            details=(
                {
                    "readFiles": result.details.read_files,
                    "modifiedFiles": result.details.modified_files,
                }
                if result.details
                else None
            ),
        )

        # Hook 2: after-compact (informational) — handlers can read
        # the persisted entry and react.
        await self._dispatch_after_compact(compaction_id, from_extension)

        return result

    async def _dispatch_before_compact(
        self,
        prep: CompactionPreparation,
        custom_instructions: str | None,
    ) -> tuple[CompactionResult | None, bool]:
        """Fire ``session_before_compact``, merge results.

        Returns ``(result, from_extension)``:

        * ``(None, False)`` — no extension intervention; the standard
          LLM compaction path runs.
        * ``(None, True)`` — an extension cancelled the compaction;
          ``compact()`` should bail out and return ``None``.
        * ``(result, True)`` — an extension supplied a custom
          :class:`CompactionResult` that should be persisted as-is.
        """
        if self._extension_runner is None or not self._extension_runner.has_handlers("session_before_compact"):
            return None, False

        from nu_coding_agent.core.extensions import (  # noqa: PLC0415
            SessionBeforeCompactEvent,
        )

        branch_entries = self._session_manager.get_branch()
        results = await self._extension_runner.emit_with_results(
            SessionBeforeCompactEvent(
                preparation=prep,
                branch_entries=branch_entries,
                custom_instructions=custom_instructions,
            )
        )
        for raw in results:
            cancel, compaction = _normalize_before_compact_result(raw)
            if cancel:
                return None, True  # cancelled by extension
            if compaction is not None:
                return compaction, True  # extension supplied custom result
        return None, False

    async def _dispatch_after_compact(self, compaction_id: str, from_extension: bool) -> None:
        """Fire ``session_compact`` so handlers can react to the persisted entry."""
        if self._extension_runner is None or not self._extension_runner.has_handlers("session_compact"):
            return

        from nu_coding_agent.core.extensions import SessionCompactEvent  # noqa: PLC0415

        compaction_entry = self._session_manager.get_entry(compaction_id)
        await self._extension_runner.emit(
            SessionCompactEvent(
                compaction_entry=compaction_entry,
                from_extension=from_extension,
            )
        )

    # ------------------------------------------------------------------
    # Session statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> SessionStats:
        """Aggregate :class:`SessionStats` from the session JSONL."""
        entries = self._session_manager.get_entries()

        user_messages = 0
        assistant_messages = 0
        tool_calls = 0
        tool_results = 0
        tokens_input = 0
        tokens_output = 0
        tokens_cache_read = 0
        tokens_cache_write = 0
        cost = 0.0

        for entry in entries:
            if entry.get("type") != "message":
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                user_messages += 1
            elif role == "assistant":
                assistant_messages += 1
                content = message.get("content")
                if isinstance(content, list):
                    tool_calls += sum(
                        1 for block in content if isinstance(block, dict) and block.get("type") == "toolCall"
                    )
                usage = message.get("usage")
                if isinstance(usage, dict):
                    tokens_input += int(usage.get("input", 0) or 0)
                    tokens_output += int(usage.get("output", 0) or 0)
                    tokens_cache_read += int(usage.get("cacheRead", 0) or 0)
                    tokens_cache_write += int(usage.get("cacheWrite", 0) or 0)
                    cost_block = usage.get("cost")
                    if isinstance(cost_block, dict):
                        cost += float(cost_block.get("total", 0) or 0)
            elif role == "toolResult":
                tool_results += 1

        total_tokens = tokens_input + tokens_output + tokens_cache_read + tokens_cache_write
        total_messages = user_messages + assistant_messages + tool_results

        return SessionStats(
            session_file=self._session_manager.get_session_file(),
            session_id=self._session_manager.get_session_id(),
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
            total_messages=total_messages,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_cache_read=tokens_cache_read,
            tokens_cache_write=tokens_cache_write,
            tokens_total=total_tokens,
            cost=cost,
        )

    # ------------------------------------------------------------------
    # Convenience: estimate the current context size for the live LLM
    # ------------------------------------------------------------------

    def estimate_context_tokens(self) -> int:
        """Use ``estimate_context_tokens`` from the compaction module."""
        context = self._session_manager.build_session_context()
        return estimate_context_tokens(context.messages).tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_before_compact_result(raw: Any) -> tuple[bool, CompactionResult | None]:
    """Coerce a ``session_before_compact`` handler's return value.

    Handlers may return either a :class:`SessionBeforeCompactResult`
    dataclass or a plain dict with ``cancel`` / ``compaction`` keys.
    Returns ``(cancel, compaction)`` so the caller can decide what to
    do without doing the type-check itself.
    """
    if raw is None:
        return False, None
    cancel = bool(getattr(raw, "cancel", None) or (raw.get("cancel") if isinstance(raw, dict) else False))
    compaction = getattr(raw, "compaction", None) or (raw.get("compaction") if isinstance(raw, dict) else None)
    return cancel, compaction


# ---------------------------------------------------------------------------
# AgentEvent → extension LifecycleEvent translation
# ---------------------------------------------------------------------------


def _translate_to_extension_event(event: AgentEvent) -> LifecycleEvent | None:
    """Map an :class:`AgentEvent` TypedDict into an extension dataclass.

    The two surfaces share the same ``type`` discriminator strings, so
    this is just a per-type field projection. Unknown types return
    ``None`` so :meth:`AgentSession._dispatch_extension_event` simply
    skips dispatch (forward-compatible with new agent loop events).
    """
    from nu_coding_agent.core.extensions import (  # noqa: PLC0415
        AgentEndEvent,
        AgentStartEvent,
        MessageEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        ToolExecutionEndEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        TurnEndEvent,
        TurnStartEvent,
    )

    event_type = event["type"]
    if event_type == "agent_start":
        return AgentStartEvent()
    if event_type == "agent_end":
        return AgentEndEvent()
    if event_type == "turn_start":
        return TurnStartEvent()
    if event_type == "turn_end":
        return TurnEndEvent()
    if event_type == "message_start":
        message = event.get("message")  # type: ignore[union-attr]
        role = getattr(message, "role", "") if message is not None else ""
        return MessageStartEvent(role=str(role) if role else "")
    if event_type == "message_update":
        return MessageUpdateEvent(payload=event.get("assistant_message_event"))  # type: ignore[union-attr]
    if event_type == "message_end":
        return MessageEndEvent(message=event.get("message"))  # type: ignore[union-attr]
    if event_type == "tool_execution_start":
        return ToolExecutionStartEvent(
            tool_name=str(event.get("tool_name", "")),  # type: ignore[union-attr]
            arguments=event.get("args") or {},  # type: ignore[union-attr]
            tool_call_id=str(event.get("tool_call_id", "")),  # type: ignore[union-attr]
        )
    if event_type == "tool_execution_update":
        return ToolExecutionUpdateEvent(
            tool_name=str(event.get("tool_name", "")),  # type: ignore[union-attr]
            update=event.get("partial_result"),  # type: ignore[union-attr]
            tool_call_id=str(event.get("tool_call_id", "")),  # type: ignore[union-attr]
        )
    if event_type == "tool_execution_end":
        return ToolExecutionEndEvent(
            tool_name=str(event.get("tool_name", "")),  # type: ignore[union-attr]
            is_error=bool(event.get("is_error", False)),  # type: ignore[union-attr]
            result=event.get("result"),  # type: ignore[union-attr]
            tool_call_id=str(event.get("tool_call_id", "")),  # type: ignore[union-attr]
        )
    return None


__all__ = [
    "AgentSession",
    "AgentSessionConfig",
    "AgentSessionListener",
    "SessionStats",
]


# Keep referenced symbols alive for static analyzers that strip
# unused imports across edits.
_ = (Any,)
