"""Visual line truncation utility — port of ``visual-truncate.ts``.

Truncates text to a maximum number of visual lines (accounting for
terminal-width wrapping).  Used by tool and bash execution widgets for
consistent tail-display behavior.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass


@dataclass
class VisualTruncateResult:
    """Result of :func:`truncate_to_visual_lines`."""

    visual_lines: list[str]
    """The visual lines to display (may be wrapped fragments)."""
    skipped_count: int
    """Number of visual lines that were hidden."""


def truncate_to_visual_lines(
    text: str,
    max_visual_lines: int,
    width: int | None = None,
    padding_x: int = 0,
) -> VisualTruncateResult:
    """Return the last *max_visual_lines* visual lines of *text*.

    Port of ``truncateToVisualLines`` (visual-truncate.ts).  Each logical
    line in *text* may wrap into multiple visual lines when it is longer than
    *width - 2·padding_x*.

    Parameters
    ----------
    text:
        The raw text (may contain ``\\n``).
    max_visual_lines:
        Maximum number of visual lines to return.
    width:
        Terminal width used for wrap calculation.  Defaults to
        ``shutil.get_terminal_size().columns``.
    padding_x:
        Horizontal padding on each side (subtracted from usable width).
    """
    if not text:
        return VisualTruncateResult(visual_lines=[], skipped_count=0)

    if width is None:
        width = shutil.get_terminal_size().columns

    usable = max(1, width - 2 * padding_x)

    # Expand every logical line into visual (wrapped) lines.
    all_visual: list[str] = []
    for logical in text.splitlines():
        if not logical:
            all_visual.append("")
            continue
        # Split into chunks of *usable* characters (simple wrap, no ANSI awareness).
        for start in range(0, max(1, len(logical)), usable):
            all_visual.append(logical[start : start + usable])

    if len(all_visual) <= max_visual_lines:
        return VisualTruncateResult(visual_lines=all_visual, skipped_count=0)

    skipped = len(all_visual) - max_visual_lines
    return VisualTruncateResult(
        visual_lines=all_visual[-max_visual_lines:],
        skipped_count=skipped,
    )


# Number of visual lines that *text* would occupy at *width*.
def count_visual_lines(text: str, width: int | None = None, padding_x: int = 0) -> int:
    if not text:
        return 0
    if width is None:
        width = shutil.get_terminal_size().columns
    usable = max(1, width - 2 * padding_x)
    count = 0
    for logical in text.splitlines() or [""]:
        count += max(1, math.ceil(len(logical) / usable))
    return count


__all__ = ["VisualTruncateResult", "count_visual_lines", "truncate_to_visual_lines"]
