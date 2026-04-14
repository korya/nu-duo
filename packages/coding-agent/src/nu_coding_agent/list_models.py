"""List available models — direct port of ``packages/coding-agent/src/cli/list-models.ts``.

Prints a formatted table of models from the :class:`ModelRegistry`,
optionally filtered by a fuzzy search pattern. Used by the
``--list-models [search]`` CLI flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nu_tui.fuzzy import fuzzy_filter

if TYPE_CHECKING:
    from nu_ai.types import Model

    from nu_coding_agent.core.model_registry import ModelRegistry


def _format_token_count(count: int) -> str:
    """Format a token count as human-readable (e.g. 200000 -> "200K")."""
    if count >= 1_000_000:
        millions = count / 1_000_000
        return f"{millions:.0f}M" if millions % 1 == 0 else f"{millions:.1f}M"
    if count >= 1_000:
        thousands = count / 1_000
        return f"{thousands:.0f}K" if thousands % 1 == 0 else f"{thousands:.1f}K"
    return str(count)


def list_models(registry: ModelRegistry, *, search: str | None = None) -> None:
    """Print a table of available models to stdout.

    Parameters
    ----------
    registry:
        The :class:`ModelRegistry` to query.
    search:
        Optional fuzzy search pattern. When provided, only models whose
        ``"<provider> <id>"`` text matches are shown.
    """
    models: list[Model] = registry.get_available()

    if not models:
        print("No models available. Set API keys in environment variables.")
        return

    # Apply fuzzy filter when a search pattern is provided.
    filtered: list[Model] = models
    if search:
        filtered = fuzzy_filter(models, search, lambda m: f"{m.provider} {m.id}")

    if not filtered:
        print(f'No models matching "{search}"')
        return

    # Sort by provider, then by model id.
    filtered.sort(key=lambda m: (m.provider, m.id))

    # Build row data.
    rows = [
        {
            "provider": m.provider,
            "model": m.id,
            "context": _format_token_count(m.context_window),
            "max_out": _format_token_count(m.max_tokens),
            "thinking": "yes" if m.reasoning else "no",
            "images": "yes" if "image" in (m.input or []) else "no",
        }
        for m in filtered
    ]

    headers = {
        "provider": "provider",
        "model": "model",
        "context": "context",
        "max_out": "max-out",
        "thinking": "thinking",
        "images": "images",
    }

    # Calculate column widths.
    widths = {key: max(len(headers[key]), *(len(r[key]) for r in rows)) for key in headers}

    # Print header.
    header_line = "  ".join(headers[k].ljust(widths[k]) for k in headers)
    print(header_line)

    # Print rows.
    for row in rows:
        line = "  ".join(row[k].ljust(widths[k]) for k in headers)
        print(line)


__all__ = [
    "list_models",
]
