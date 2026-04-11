"""Diff and edit-application primitives.

Direct port of ``packages/coding-agent/src/core/tools/edit-diff.ts``.
The TS version uses the ``diff`` npm package; the Python port uses
:mod:`difflib` for the line-by-line diff and ports the rest of the
fuzzy-matching / multi-edit application logic by hand.

Single-edit *or* multi-edit calls are supported. All edits are matched
against the same original content; if any one needs fuzzy matching,
the entire operation runs in fuzzy-normalized space (matches upstream).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Line-ending detection / normalization
# ---------------------------------------------------------------------------


def detect_line_ending(content: str) -> str:
    """Return ``"\\r\\n"`` if the content uses CRLF, otherwise ``"\\n"``."""
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    """Convert ``\\r\\n`` and standalone ``\\r`` to ``\\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


# ---------------------------------------------------------------------------
# Fuzzy normalization
# ---------------------------------------------------------------------------


_SMART_SINGLE = re.compile(r"[\u2018\u2019\u201a\u201b]")
_SMART_DOUBLE = re.compile(r"[\u201c\u201d\u201e\u201f]")
_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_SPECIAL_SPACES = re.compile(r"[\u00a0\u2002-\u200a\u202f\u205f\u3000]")


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize ``text`` for fuzzy match comparison.

    Mirrors the upstream sequence:

    1. NFKC Unicode normalization.
    2. Strip trailing whitespace per line.
    3. Smart quotes → straight quotes.
    4. Various Unicode dashes → ASCII hyphen.
    5. Various Unicode spaces → regular space.
    """
    nfkc = unicodedata.normalize("NFKC", text)
    stripped = "\n".join(line.rstrip() for line in nfkc.split("\n"))
    quoted = _SMART_SINGLE.sub("'", stripped)
    quoted = _SMART_DOUBLE.sub('"', quoted)
    dashed = _DASHES.sub("-", quoted)
    return _SPECIAL_SPACES.sub(" ", dashed)


@dataclass(slots=True)
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find ``old_text`` in ``content`` — exact first, fuzzy fallback."""
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old)
    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )
    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def strip_bom(content: str) -> tuple[str, str]:
    """Return ``(bom, text_without_bom)`` for content that may start with U+FEFF."""
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Edit:
    old_text: str
    new_text: str


@dataclass(slots=True)
class _MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


@dataclass(slots=True)
class AppliedEditsResult:
    base_content: str
    new_content: str


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    return len(fuzzy_content.split(fuzzy_old)) - 1


def _not_found_error(path: str, edit_index: int, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"Could not find the exact text in {path}. The old text must match exactly "
            f"including all whitespace and newlines."
        )
    return ValueError(
        f"Could not find edits[{edit_index}] in {path}. The oldText must match exactly "
        f"including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            f"Please provide more context to make it unique."
        )
    return ValueError(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be "
        f"unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(f"oldText must not be empty in {path}.")
    return ValueError(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _no_change_error(path: str, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"No changes made to {path}. The replacement produced identical content. This might "
            f"indicate an issue with special characters or the text not existing as expected."
        )
    return ValueError(f"No changes made to {path}. The replacements produced identical content.")


def apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[Edit],
    path: str,
) -> AppliedEditsResult:
    """Apply ``edits`` to ``normalized_content`` (LF-normalized).

    All edits are matched against the same baseline (``base_content``).
    Replacements are applied in reverse-order so prior offsets stay
    stable. If any edit needs fuzzy matching the operation runs in the
    fuzzy-normalized content space.
    """
    normalized_edits = [Edit(old_text=normalize_to_lf(e.old_text), new_text=normalize_to_lf(e.new_text)) for e in edits]

    for i, edit in enumerate(normalized_edits):
        if not edit.old_text:
            raise _empty_old_text_error(path, i, len(normalized_edits))

    initial_matches = [fuzzy_find_text(normalized_content, e.old_text) for e in normalized_edits]
    base_content = (
        normalize_for_fuzzy_match(normalized_content)
        if any(m.used_fuzzy_match for m in initial_matches)
        else normalized_content
    )

    matched: list[_MatchedEdit] = []
    for i, edit in enumerate(normalized_edits):
        result = fuzzy_find_text(base_content, edit.old_text)
        if not result.found:
            raise _not_found_error(path, i, len(normalized_edits))

        occurrences = _count_occurrences(base_content, edit.old_text)
        if occurrences > 1:
            raise _duplicate_error(path, i, len(normalized_edits), occurrences)

        matched.append(
            _MatchedEdit(
                edit_index=i,
                match_index=result.index,
                match_length=result.match_length,
                new_text=edit.new_text,
            )
        )

    matched.sort(key=lambda m: m.match_index)
    for i in range(1, len(matched)):
        previous = matched[i - 1]
        current = matched[i]
        if previous.match_index + previous.match_length > current.match_index:
            raise ValueError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in "
                f"{path}. Merge them into one edit or target disjoint regions."
            )

    new_content = base_content
    for m in reversed(matched):
        new_content = new_content[: m.match_index] + m.new_text + new_content[m.match_index + m.match_length :]

    if base_content == new_content:
        raise _no_change_error(path, len(normalized_edits))

    return AppliedEditsResult(base_content=base_content, new_content=new_content)


# ---------------------------------------------------------------------------
# Unified diff with line numbers + context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GeneratedDiff:
    diff: str
    first_changed_line: int | None


def generate_diff_string(
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> GeneratedDiff:
    """Build a custom unified diff with line numbers and bounded context.

    Mirrors the format the TS version emits via ``Diff.diffLines``:
    each line is prefixed by ``+``/``-``/`` `` plus a right-padded line
    number, and runs of unchanged lines are collapsed to ``context_lines``
    on each side of a change with ``...`` markers in the middle.
    """
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    matcher = SequenceMatcher(a=old_lines, b=new_lines)
    parts: list[tuple[str, list[str]]] = []  # (kind, lines) where kind ∈ {"equal","add","remove"}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(("equal", old_lines[i1:i2]))
        elif tag == "delete":
            parts.append(("remove", old_lines[i1:i2]))
        elif tag == "insert":
            parts.append(("add", new_lines[j1:j2]))
        else:  # "replace"
            parts.append(("remove", old_lines[i1:i2]))
            parts.append(("add", new_lines[j1:j2]))

    output: list[str] = []
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))
    pad_blank = " " * line_num_width

    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: int | None = None

    for i, (kind, lines) in enumerate(parts):
        if kind in {"add", "remove"}:
            if first_changed_line is None:
                first_changed_line = new_line_num
            for line in lines:
                if kind == "add":
                    output.append(f"+{str(new_line_num).rjust(line_num_width)} {line}")
                    new_line_num += 1
                else:
                    output.append(f"-{str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
            last_was_change = True
        else:
            next_part_is_change = i < len(parts) - 1 and parts[i + 1][0] in {"add", "remove"}
            has_leading_change = last_was_change
            has_trailing_change = next_part_is_change

            if has_leading_change and has_trailing_change:
                if len(lines) <= context_lines * 2:
                    for line in lines:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
                else:
                    leading = lines[:context_lines]
                    trailing = lines[-context_lines:]
                    skipped = len(lines) - len(leading) - len(trailing)
                    for line in leading:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
                    output.append(f" {pad_blank} ...")
                    old_line_num += skipped
                    new_line_num += skipped
                    for line in trailing:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
            elif has_leading_change:
                shown = lines[:context_lines]
                skipped = len(lines) - len(shown)
                for line in shown:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
                if skipped > 0:
                    output.append(f" {pad_blank} ...")
                    old_line_num += skipped
                    new_line_num += skipped
            elif has_trailing_change:
                skipped = max(0, len(lines) - context_lines)
                if skipped > 0:
                    output.append(f" {pad_blank} ...")
                    old_line_num += skipped
                    new_line_num += skipped
                for line in lines[skipped:]:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
            else:
                old_line_num += len(lines)
                new_line_num += len(lines)

            last_was_change = False

    return GeneratedDiff(diff="\n".join(output), first_changed_line=first_changed_line)


__all__ = [
    "AppliedEditsResult",
    "Edit",
    "FuzzyMatchResult",
    "GeneratedDiff",
    "apply_edits_to_normalized_content",
    "detect_line_ending",
    "fuzzy_find_text",
    "generate_diff_string",
    "normalize_for_fuzzy_match",
    "normalize_to_lf",
    "restore_line_endings",
    "strip_bom",
]
