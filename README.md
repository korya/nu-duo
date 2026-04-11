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
uv sync          # install workspace + dev dependencies
just lint        # ruff check --fix + ruff format
just test        # pytest across all packages
just test-cov    # per-package + total coverage
just run-sample  # end-to-end sample against a real provider (see below)
```

The full toolchain (`ruff`, `pyright` strict, `pytest` with `--import-mode=
importlib`) is configured at the workspace root in `pyproject.toml`. Type
checking, lint and tests all run against the entire workspace by default.

## Sample app

`examples/sample.py` exercises the four core tools end-to-end via
`nu_agent_core.Agent` against a real LLM provider. Run it through the
`just run-sample` recipe:

```bash
# OpenAI (default — gpt-4o-mini)
just run-sample
just run-sample "summarize the project README in three bullet points"

# Anthropic
just run-sample --anthropic "explain the agent loop architecture"

# Custom prompt + repo path
just run-sample "list TODO comments" /path/to/some/repo
```

API keys are loaded from a top-level `.env` file via `python-dotenv` (see
`.env.example`). `OPENAI_API_KEY` is required for `--openai`,
`ANTHROPIC_API_KEY` (or `ANTHROPIC_OAUTH_TOKEN`) for `--anthropic`. Override
the model id via `PI_SAMPLE_OPENAI_MODEL` / `PI_SAMPLE_ANTHROPIC_MODEL`.

## Project layout

Each upstream TS package is mirrored 1:1 under `packages/<name>/`, with
`src/nu_<name>/` for the Python module and `tests/` for the test suite.
Cross-package dependencies are declared in each package's `pyproject.toml`
and resolved by the `[tool.uv.workspace]` section in the root
`pyproject.toml`.
