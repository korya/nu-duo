# pi-mono (Python port)

Python port of [badlogic/pi-mono](https://github.com/badlogic/pi-mono), a minimalist, layered AI-agent stack by Mario Zechner.

This is a `uv` workspace mirroring the upstream monorepo 1:1. Each TypeScript package becomes one Python package under `packages/`:

| Upstream package            | Python package    | Status      |
| --------------------------- | ----------------- | ----------- |
| `@mariozechner/pi-ai`       | `pi-ai`           | in progress |
| `@mariozechner/pi-agent-core` | `pi-agent-core` | pending     |
| `@mariozechner/pi-tui`      | `pi-tui`          | pending     |
| `@mariozechner/pi-coding-agent` | `pi-coding-agent` | pending |
| `@mariozechner/pi-mom`      | `pi-mom`          | pending     |
| `@mariozechner/pi-pods`     | `pi-pods`         | pending     |
| `@mariozechner/pi-web-ui`   | `pi-web-ui`       | pending     |

## Design goals

- Maximum fidelity to the upstream TS code: same package boundaries, same abstraction names, same API shapes, same tool behaviors, same session JSONL format.
- Pi's philosophy preserved: *primitives, not features*; minimal core; four tools (Read, Write, Edit, Bash); provider abstraction at the lowest layer; event-driven streaming; file-based state; observability over hidden orchestration.
- Modern Python: `>=3.13`, `uv`, `ruff`, `pyright` strict, `pydantic` v2, `asyncio`.

## Development

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

See `/Users/dmitri/.claude/plans/magical-yawning-elephant.md` for the full port plan.
