# Nu-duo

A Python port of [badlogic/pi-mono](https://github.com/badlogic/pi-mono), Mario
Zechner's minimalist, layered AI-agent stack.

The project is a `uv` workspace that mirrors the upstream TypeScript monorepo
1:1. Every upstream package becomes one Python package under `packages/`,
prefixed with `nu-` (PyPI distribution name) / `nu_` (Python import name):

| Upstream package | Python package | Status |
| --- | --- | --- |
| `@mariozechner/pi-ai` | `nu-ai` | **done** — types, event stream, registry, transforms, faux + Anthropic + OpenAI Chat Completions + Google + Ollama providers, top-level stream/complete. Deferred: OpenAI Responses, Bedrock, Mistral, Vertex, OAuth flows |
| `@mariozechner/pi-agent-core` | `nu-agent-core` | **done** — types, agent loop with sequential/parallel tool execution, stateful Agent class, hooks, steering/follow-up queues |
| `@mariozechner/pi-tui` | `nu-tui` | **done** — Textual-backed TUI wrapper, 12 components (Spacer, Text, TruncatedText, Box, SelectList, Input, Editor, Loader, CancellableLoader, Markdown, SettingsList, Image), theme, terminal capabilities, autocomplete, stdin buffer, overlay types, component widget bridge. ~53% LoC parity (intentional: Textual replaces the hand-rolled renderer) |
| `@mariozechner/pi-coding-agent` | `nu-coding-agent` | **done** — 7 tools, session manager (JSONL byte-compat with TS), compaction (full surface + extension hooks), agent session (extensions + compaction + tool wrapping + action binding), extensions (types + runner + loader + lifecycle hooks + tool registration + action methods + compaction hooks), agent session runtime (switch/new/fork), SDK entry point, print mode (`nu --print` with `--continue`/`--session`/`--ephemeral`/`--base-url`/`--api`), interactive mode (Textual REPL with streaming + Rich Markdown + auto-compaction + extensions + 10 slash commands + 4 selectors + specialized message widgets + footer) |
| `@mariozechner/pi-mom` | `nu-mom` | **done** — Slack bot (slack-bolt socket mode), per-channel agent sessions, Docker sandbox execution, cron event watcher, channel history download, 6 sandbox-aware tool wrappers, structured logging |
| `@mariozechner/pi-pods` | `nu-pods` | **done** — types, config, ssh wrappers, model catalogue, all commands, CLI, agent bridge into `nu_coding_agent` via `--base-url`/`--api` |
| `@mariozechner/pi-web-ui` | `nu-web-ui` | **done** — FastAPI backend with SQLite storage (sessions, settings, provider keys, custom providers), WebSocket agent event streaming, model discovery (Ollama + LM Studio), CORS proxy. Frontend: vendored TS/Lit served as static files |

## Design goals

- Maximum fidelity to the upstream TS code: same package boundaries, same abstraction names, same API shapes, same tool behaviors, same session JSONL format.
- Pi's philosophy preserved: *primitives, not features*; minimal core; four tools (Read, Write, Edit, Bash); provider abstraction at the lowest layer; event-driven streaming; file-based state; observability over hidden orchestration.
- Modern Python: `>=3.13`, `uv`, `ruff`, `pyright` strict, `pydantic` v2, `asyncio`.

## Development

```bash
uv sync                       # install workspace + dev dependencies
just lint                     # ruff check --fix + ruff format
just test                     # pytest across all packages
just test-cov                 # per-package + total coverage
just run-example-one-shot     # one-shot sample (defaults to OpenAI gpt-4o-mini)
just run-example-interactive  # interactive REPL sample (defaults to OpenAI gpt-4o-mini)
```

The full toolchain (`ruff`, `pyright` strict, `pytest` with `--import-mode=
importlib`) is configured at the workspace root in `pyproject.toml`. Type
checking, lint and tests all run against the entire workspace by default.

## Examples

The `examples/` directory ships several runnable demos against real LLM
providers:

| File | What it does |
|---|---|
| `one-shot-openai.py` | Minimal single-prompt example against OpenAI `gpt-4o-mini`. Shortest path from import to a working agent. |
| `one-shot-anthropic.py` | Same shape, but against Anthropic `claude-haiku-4-5`. |
| `one-shot.py` | Single-prompt sample with provider switching (`--openai`/`--anthropic`), the four core tools wired in, and live streaming output. |
| `interactive-openai.py` | Minimal multi-turn REPL against OpenAI `gpt-4o-mini`. |
| `interactive-anthropic.py` | Same, against Anthropic `claude-haiku-4-5`. |
| `interactive.py` | Multi-turn REPL with provider switching, the four core tools, and live streaming output. |

```bash
# One-shot sample (defaults to OpenAI gpt-4o-mini, four core tools wired in)
just run-example-one-shot
just run-example-one-shot "summarize the project README in three bullet points"
just run-example-one-shot --anthropic "explain the agent loop"
just run-example-one-shot "list TODO comments" /path/to/some/repo

# Interactive REPL sample (defaults to OpenAI gpt-4o-mini)
just run-example-interactive
just run-example-interactive --anthropic

# Single-provider minimal variants (no flag handling, easiest to read)
uv run python examples/one-shot-openai.py
uv run python examples/one-shot-anthropic.py "what is the capital of Iceland?"
uv run python examples/interactive-openai.py
uv run python examples/interactive-anthropic.py
```

API keys are loaded from a top-level `.env` file via `python-dotenv` (see
`.env.example`). `OPENAI_API_KEY` is required for the OpenAI examples,
`ANTHROPIC_API_KEY` (or `ANTHROPIC_OAUTH_TOKEN`) for the Anthropic ones.
Override the model id used by `one-shot.py` / `interactive.py` via
`NU_SAMPLE_OPENAI_MODEL` / `NU_SAMPLE_ANTHROPIC_MODEL`.

## Deviations from pi-mono

The port aims for maximum fidelity — same package boundaries, same
abstraction names, same on-disk formats — but a handful of conscious
deviations were unavoidable or actively beneficial. Each is listed
below so the upstream blog posts and AGENTS.md still resolve cleanly,
and so reviewers know which differences are intentional rather than
oversights.

### Workspace-wide

- **Naming.** The Python distribution is branded *Nu-duo*. Upstream
  package `pi-X` becomes `nu-X` (PyPI distribution name) and `nu_X`
  (Python import name); class names are kept verbatim, function names
  switch from `camelCase` to `snake_case` per Python convention. File
  names map 1:1 with the TS originals (e.g. `agent-loop.ts` →
  `agent_loop.py`) so cross-references in the upstream blog posts
  still locate the right module.
- **Schema validation.** TS uses TypeBox + ajv to declare tool
  parameter schemas; the Python port uses Pydantic v2 dataclasses /
  models and emits JSON Schema via `model_json_schema()`. The wire
  format and the LLM-visible schemas are identical; only the
  authoring API differs.
- **Async model.** TS `AsyncIterable<Event>` becomes Python
  `AsyncIterator[Event]`; subscriber callbacks become async
  callables on an `EventStream`/`EventBus`. Same push/end/iter
  semantics, idiomatic asyncio surface.

### `nu-ai`

- **Provider scope.** Only Anthropic, OpenAI Chat Completions, Google
  Generative AI, Ollama, and the faux test provider are ported so
  far. OpenAI Responses, Bedrock, Mistral, Vertex, Cerebras, xAI,
  z.ai, OpenRouter, Together, Fireworks, DeepSeek, Nvidia,
  Hyperbolic, Lambda Labs, Sambanova, Novita, and the LM Studio
  client are still deferred. Provider-specific OAuth flows
  (Anthropic, Google, etc.) are also pending.
- **`models_generated.py`.** The catalogue snapshot is hand-maintained
  for now; the port of the `generate-models.ts` script that
  refreshes it from provider APIs is deferred.

### `nu-tui`

- **Implementation strategy.** This is the one package where the
  deviation is *implementation-level*, not API-level. Upstream pi-tui
  is a hand-rolled differential renderer; nu-tui will eventually
  rebuild the public API on top of [Textual](https://textual.textualize.io/)
  rather than reimplementing the diff engine. Currently only the
  pure-logic utilities (`UndoStack`, `KillRing`, `Fuzzy`, `Keys`,
  `Keybindings`) are ported; the Textual-backed renderer + components
  land alongside `nu_coding_agent` interactive mode.

### `nu-coding-agent`

- **Tool authoring.** Tool parameter schemas are Pydantic models, not
  TypeBox `Type.Object` literals (see workspace-wide note above). The
  resulting JSON Schemas are identical, but the authoring code differs.
- **Subprocess execution.** The Bash tool uses `asyncio.create_subprocess_exec`
  with the same timeout / cancellation semantics as the TS
  `child_process.spawn`-based version. Output capture, exit code
  handling, and signal propagation match.
- **Coverage policy.** `nu_coding_agent/cli.py` is excluded from
  coverage as a thin glue layer (matches the `nu_pods/cli.py` policy).
  The library code stays under the 90 % per-package gate.
- **Deferred surface area.** Skills, agent_session_runtime, modes/rpc,
  modes/interactive, and most upstream CLI flags are still deferred.
  Extensions now have foundation + AgentSession lifecycle integration
  (every event in the agent loop reaches attached extensions) +
  extension-registered tool wrapping (extensions can ship `AgentTool`
  instances and the agent loop calls them like built-ins, with name-
  based override) + action-method runtime binding (extensions read+
  mutate session state via `set_label` / `append_custom_entry` /
  `set_session_name` / `get_session_name` / `get_active_tools` /
  `get_all_tools` / `set_active_tools` / `set_model` /
  `get_thinking_level` / `set_thinking_level`) + `before_compact` /
  `compact` hooks during compaction (extensions can observe, cancel,
  or supply a custom `CompactionResult` that bypasses the LLM). The
  remaining extension surface — `sendMessage` / `sendUserMessage`
  (need the steering queue), `refresh_tools`, `get_commands`, command
  / shortcut / message-renderer consumers (need slash commands and
  interactive mode), and provider registration via extensions — is
  still deferred and tracked as follow-up sub-slices. The minimal
  `nu` CLI supports `--print` mode with session persistence
  (`--continue`, `--session FILE`, `--ephemeral`); the remaining 30+
  TS flags will land alongside interactive mode.
- **Extension loader.** Replaces upstream's `jiti`-based TypeScript
  loading + `package.json` "pi" manifest discovery with the natural
  Python idiom: extensions register themselves under the
  `nu_coding_agent.extensions` Python entry point group (or are
  loaded from explicit `.py` paths exporting a `register` callable).
  Conceptually identical to "load a packaged extension" — the
  porting plan called this mapping out explicitly.
- **`nu` CLI uses an in-memory `AuthStorage`.** Print-mode runs only
  read credentials from environment variables and `--api-key`; they
  do not touch `~/.nu/auth.json`. The on-disk credential store will
  land alongside interactive mode and the `nu auth` subcommand.

### `nu-pods`

- **SSH transport.** We shell out to the user's local `ssh` / `scp`
  binaries via `asyncio.create_subprocess_exec` instead of pulling in
  `asyncssh`. This matches the TS version's `child_process.spawn`
  approach exactly and means existing host configs, jump hosts,
  ssh-agent, and key files keep working unchanged. (The original
  port plan called for `asyncssh`; this was a deliberate change after
  reading the upstream code.)
- **Config location.** Config defaults to `~/.nu/pods.json` (override
  via `NU_PODS_CONFIG_DIR`) instead of the upstream `~/.pi/pods.json`
  (`PI_CONFIG_DIR`). The on-disk JSON shape is byte-compatible
  (camelCase keys preserved), so a config file can be moved between
  the two implementations without conversion.
- **API key env var.** `NU_API_KEY` is read first; `PI_API_KEY` is
  honored as a legacy fallback so existing pods setups keep working.
- **Log streaming during `start`.** The TS implementation runs a
  long-lived `tail -f` over SSH while watching for the
  `Application startup complete` marker. The Python port runs a
  bounded `tail -n 200` instead — full liveness monitoring requires
  a streaming SSH subprocess that's hostile to unit testing, and the
  observable behaviour (startup detection + classification of OOM /
  engine init / runner exit failures, plus config rollback) is
  identical for everything except the live progress bars during the
  startup window. The decision is documented inline in
  `commands/models.py`.
- **`apply_memory_and_context` quirk.** Faithfully reproduces a TS
  bug: the function strips `--gpu-memory-utilization` and
  `--max-model-len` flag *names* by substring match only, leaving
  any orphaned values behind. The bundled `models.json` configs
  never pre-set these flags so it never bites in practice; the
  fidelity is intentional and documented in a comment so the
  next reader knows it's not an oversight.
- **`nu-pods agent` launcher.** `build_invocation` resolves a
  deployed model, picks the right `--api` (responses for `gpt-oss`
  models, completions otherwise), and assembles a complete argv list,
  which the default launcher hands off to `nu_coding_agent.cli`
  in-process. The `nu` CLI honours `--base-url` / `--api` flags so
  the same Model construction path works for any OpenAI-compatible
  endpoint, not just bundled providers.
- **Bug fix.** `parse_context_size`: a naive transliteration of the
  TS `_CONTEXT_SIZES.get(value.lower(), int(value))` would crash on
  `"4k"` because Python's `dict.get` eagerly evaluates the default
  argument. Fixed to a two-step lookup.

### Deferred packages

`nu-mom`, `nu-web-ui` are scaffolds only. Their port plans (Slack
bolt + Docker SDK for mom, FastAPI backend + vendored Lit frontend
for web-ui) are documented in the porting plan but no production
code has been written yet.

## Project layout

Each upstream TS package is mirrored 1:1 under `packages/<name>/`, with
`src/nu_<name>/` for the Python module and `tests/` for the test suite.
Cross-package dependencies are declared in each package's `pyproject.toml`
and resolved by the `[tool.uv.workspace]` section in the root
`pyproject.toml`.
