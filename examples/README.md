# Nu-duo Examples

Runnable demos that showcase different capabilities of the Nu-duo stack.
Each example is self-contained and can be run with `uv run python examples/<file>.py`.

## Prerequisites

All examples load API keys from a top-level `.env` file via `python-dotenv`.
Create one from the template:

```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=AI...
```

## Quick Start Examples

These are the shortest path from zero to a working agent.

| File | Provider | What it does |
|---|---|---|
| [`one-shot-openai.py`](one-shot-openai.py) | OpenAI | Minimal single-prompt against `gpt-4o-mini`. ~30 lines. |
| [`one-shot-anthropic.py`](one-shot-anthropic.py) | Anthropic | Same shape against `claude-haiku-4-5`. |
| [`interactive-openai.py`](interactive-openai.py) | OpenAI | Minimal multi-turn REPL. |
| [`interactive-anthropic.py`](interactive-anthropic.py) | Anthropic | Same shape. |

```bash
uv run python examples/one-shot-openai.py "what is 2+2?"
uv run python examples/interactive-openai.py
```

## Full-Featured Examples

These wire in tools, provider switching, streaming, and the system prompt.

| File | What it demonstrates |
|---|---|
| [`one-shot.py`](one-shot.py) | Single prompt with `--openai`/`--anthropic` flag, four core tools (read/write/edit/bash), live streaming output, `--cwd` for targeting a specific directory. |
| [`interactive.py`](interactive.py) | Multi-turn REPL with provider switching, tools, streaming, `/model` command for runtime model switching. |

```bash
just run-example-one-shot "summarize the project README"
just run-example-one-shot --anthropic "explain the agent loop"
just run-example-interactive
```

## SDK & Session Examples

Demonstrate the programmatic API and session persistence.

| File | What it demonstrates |
|---|---|
| [`sdk-session.py`](sdk-session.py) | The high-level `create_agent_session()` factory — one call builds auth, registry, session manager, tools, agent. The recommended way to embed `nu_coding_agent` in other applications. |
| [`session-continue.py`](session-continue.py) | Session persistence: the first run creates a session on disk; the second run resumes it automatically, so the model sees the full conversation history. Run it twice to see continuation in action. |

```bash
# SDK: one-call session creation
uv run python examples/sdk-session.py "list Python files here"

# Session continuation: run twice
uv run python examples/session-continue.py "what is 2+2?"
uv run python examples/session-continue.py "and what about 3+3?"
# ^ The second run resumes the session from the first
```

## Extension Examples

Demonstrate the extension system: lifecycle hooks and custom tools.

| File | What it demonstrates |
|---|---|
| [`extension-hello.py`](extension-hello.py) | A simple extension that subscribes to `agent_start`, `message_end`, and `session_shutdown` events. Shows the extension API, factory pattern, and how to attach extensions to an `AgentSession`. |
| [`custom-tool.py`](custom-tool.py) | An extension that registers a custom `weather` tool the LLM can call. Demonstrates `api.register_tool()`, `session.apply_extension_tools()`, and how the agent loop invokes extension-registered tools. |

```bash
# Extension hooks
uv run python examples/extension-hello.py

# Custom tool
uv run python examples/custom-tool.py
```

## Provider Examples

Demonstrate different LLM providers.

| File | Provider | What it demonstrates |
|---|---|---|
| [`one-shot-openai.py`](one-shot-openai.py) | OpenAI | Chat Completions API via `gpt-4o-mini` |
| [`one-shot-anthropic.py`](one-shot-anthropic.py) | Anthropic | Messages API via `claude-haiku-4-5` |
| [`google-provider.py`](google-provider.py) | Google | Generative AI API via `gemini-2.0-flash` |

```bash
uv run python examples/google-provider.py "explain quantum computing"
```

## Web UI Example

| File | What it demonstrates |
|---|---|
| [`web-ui-server.py`](web-ui-server.py) | Launches the FastAPI backend with SQLite storage, model discovery, and WebSocket streaming. When the vendored frontend is present, serves it as static files. |

```bash
uv run python examples/web-ui-server.py --port 3000
# Then open http://localhost:3000
```

## Nu CLI

The `nu` CLI itself is the most full-featured example — it wires everything together (tools, sessions, extensions, compaction, streaming, markdown rendering).

```bash
# Interactive REPL (Textual TUI)
uv run nu

# Single-shot print mode
uv run nu --print "what is 2+2"

# Resume most recent session
uv run nu -c "follow up on our last conversation"

# Target a specific session file
uv run nu --session ~/.nu/agent/sessions/.../sess.jsonl "continue"

# Ephemeral (no session persistence)
uv run nu --ephemeral "quick question"

# Custom endpoint (e.g. a vLLM pod)
uv run nu --base-url http://pod:8001/v1 --model Qwen/Qwen3-Coder-30B --api-key x "hello"
```

## Nu-Pods CLI

The `nu-pods` CLI manages vLLM deployments on GPU pods.

```bash
# Setup a pod
nu-pods pods setup mypod "ssh root@gpu-host" --models-path /mnt/sfs

# Deploy a model
nu-pods start Qwen/Qwen3-Coder-30B-A3B-Instruct --name coder

# Chat with a deployed model
nu-pods agent coder -- --print "explain this codebase"

# List known models
nu-pods models
```
