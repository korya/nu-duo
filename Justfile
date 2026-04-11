# Forward quoted CLI arguments to recipes via "$@" instead of plain string
# substitution. Without this, ``just run-example-one-shot "a b" .`` would
# word-split the prompt into separate argv entries.
set positional-arguments := true

default:
    @just --list

# Run ruff with auto-fix on all packages
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Run pytest across all packages
test:
    uv run pytest

# Run pytest with per-package and total coverage reporting.
# Library code (everything under packages/*/src) must stay above 90 %.
test-cov:
    uv run pytest \
        --cov=nu_ai --cov=nu_agent_core --cov=nu_tui \
        --cov=nu_coding_agent --cov=nu_mom --cov=nu_pods --cov=nu_web_ui \
        --cov-report=term-missing \
        --cov-fail-under=90

# Run the one-shot sample (defaults to OpenAI gpt-4o-mini, uses .env).
# Pass --anthropic to switch providers.
# Usage: just run-example-one-shot
#        just run-example-one-shot "your prompt here"
#        just run-example-one-shot --anthropic "your prompt here"
run-example-one-shot *ARGS:
    uv run python examples/one-shot.py "$@"

# Run the interactive REPL sample (defaults to OpenAI gpt-4o-mini, uses .env).
# Pass --anthropic to switch providers.
# Usage: just run-example-interactive
#        just run-example-interactive --anthropic
run-example-interactive *ARGS:
    uv run python examples/interactive.py "$@"

# Run the `nu` CLI through uv (uses .env for API keys).
# Usage: just nu --help
#        just nu "what is 2+2"
#        just nu -q "give me the answer only"
#        just nu --anthropic "explain the agent loop"
nu *ARGS:
    uv run nu "$@"
