# Forward quoted CLI arguments to recipes via "$@" instead of plain string
# substitution. Without this, ``just run-sample "a b" .`` would word-split
# the prompt into separate argv entries.
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

# Run the end-to-end sample app (uses .env for API keys).
# Usage: just run-sample
#        just run-sample "your prompt here"
#        just run-sample --anthropic "your prompt here" /path/to/repo
run-sample *ARGS:
    uv run python examples/sample.py "$@"

# Run the `nu` CLI through uv (uses .env for API keys).
# Usage: just nu --help
#        just nu "what is 2+2"
#        just nu -q "give me the answer only"
#        just nu --anthropic "explain the agent loop"
nu *ARGS:
    uv run nu "$@"
