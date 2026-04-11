# Nu-duo

A Python port of [badlogic/pi-mono](https://github.com/badlogic/pi-mono), Mario
Zechner's minimalist, layered AI-agent stack.

The project is a `uv` workspace that mirrors the upstream TypeScript monorepo
1:1. Every upstream package becomes one Python package under `packages/`,
prefixed with `nu-` (PyPI distribution name) / `nu_` (Python import name):

| Upstream package | Python package | Status |
| --- | --- | --- |
| `@mariozechner/pi-ai` | `nu-ai` | partial — types, event stream, registry, transforms, faux + Anthropic + OpenAI Chat Completions + Google providers, top-level stream/complete; OpenAI Responses, Bedrock, Mistral, Vertex, OAuth flows still deferred |
| `@mariozechner/pi-agent-core` | `nu-agent-core` | done — types, agent loop with sequential/parallel tool execution, stateful Agent class, hooks, steering/follow-up queues |
| `@mariozechner/pi-tui` | `nu-tui` | partial — pure utilities (UndoStack, KillRing, fuzzy, keys, keybindings); Textual-backed renderer + components deferred until consumed by interactive mode |
| `@mariozechner/pi-coding-agent` | `nu-coding-agent` | partial — four core tools (read, write, edit, bash) with shared helpers; session manager, compaction, extensions, agent_session, modes/print, modes/rpc, CLI entry point still deferred |
| `@mariozechner/pi-mom` | `nu-mom` | scaffold only |
| `@mariozechner/pi-pods` | `nu-pods` | scaffold only |
| `@mariozechner/pi-web-ui` | `nu-web-ui` | scaffold only |

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

## Project layout

Each upstream TS package is mirrored 1:1 under `packages/<name>/`, with
`src/nu_<name>/` for the Python module and `tests/` for the test suite.
Cross-package dependencies are declared in each package's `pyproject.toml`
and resolved by the `[tool.uv.workspace]` section in the root
`pyproject.toml`.
