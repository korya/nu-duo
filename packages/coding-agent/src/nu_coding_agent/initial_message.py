"""Combine stdin + ``@file`` text + CLI message into a single initial prompt.

Direct port of ``packages/coding-agent/src/cli/initial-message.ts``.

Non-interactive (print) mode collects content from up to three sources
before the agent loop starts:

1. Piped stdin (e.g. ``cat README.md | nu -p "summarise this"``).
2. ``@file`` arguments processed by :mod:`nu_coding_agent.file_processor`.
3. The first positional CLI message.

This module joins them into one prompt string (parts separated by
empty lines) and passes through any image attachments from ``@file``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class InitialMessageResult:
    """The assembled initial prompt and any image attachments."""

    text: str | None = None
    images: list[dict[str, Any]] = field(default_factory=list)


def build_initial_message(
    *,
    stdin_text: str | None = None,
    file_text: str | None = None,
    cli_message: str | None = None,
    images: list[dict[str, Any]] | None = None,
) -> InitialMessageResult | None:
    """Build the initial user message from the available content sources.

    Parameters
    ----------
    stdin_text:
        Text read from stdin (when piped).
    file_text:
        Combined ``<file>`` XML blocks from :func:`process_file_arguments`.
    cli_message:
        The first positional argument (prompt) from the CLI.
    images:
        Image attachments produced by :func:`process_file_arguments`.

    Returns
    -------
    InitialMessageResult | None
        ``None`` when all sources are empty — the caller should either
        launch interactive mode or print an error.

    Notes
    -----
    The upstream joins parts with no separator (empty string ``""``) because
    the ``<file>`` blocks already end with a newline. We follow suit so that
    prompts render identically.
    """
    parts: list[str] = []
    if stdin_text is not None:
        parts.append(stdin_text)
    if file_text:
        parts.append(file_text)
    if cli_message:
        parts.append(cli_message)

    if not parts and not images:
        return None

    resolved_images = images if images and len(images) > 0 else []

    return InitialMessageResult(
        text="".join(parts) if parts else None,
        images=resolved_images,
    )


__all__ = [
    "InitialMessageResult",
    "build_initial_message",
]
