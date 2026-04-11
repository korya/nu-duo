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

# Run pytest with per-package and total coverage reporting
test-cov:
    uv run pytest \
        --cov=pi_ai --cov=pi_agent_core --cov=pi_tui \
        --cov=pi_coding_agent --cov=pi_mom --cov=pi_pods --cov=pi_web_ui \
        --cov-report=term-missing

# Run the end-to-end sample app (uses .env for API keys).
# Usage: just run-sample
#        just run-sample "your prompt here"
#        just run-sample --anthropic "your prompt here" /path/to/repo
run-sample *ARGS:
    uv run python examples/sample.py "$@"
