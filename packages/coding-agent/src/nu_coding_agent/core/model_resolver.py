"""Model resolution + scoping — direct port of ``packages/coding-agent/src/core/model-resolver.ts``.

Glob patterns and partial id/name matching pulled straight from the
upstream. The Python port replaces ``minimatch`` with :mod:`fnmatch`
(case-insensitive) and the upstream's chalk-styled warnings with plain
strings, since the warning rendering is the caller's job.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from nu_coding_agent.core.defaults import DEFAULT_THINKING_LEVEL

if TYPE_CHECKING:
    from nu_ai.types import Model

    from nu_coding_agent.core.model_registry import ModelRegistry


type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


_VALID_THINKING_LEVELS: frozenset[str] = frozenset({"off", "minimal", "low", "medium", "high", "xhigh"})


def _is_valid_thinking_level(value: str) -> bool:
    """Match the upstream ``cli/args.ts`` ``isValidThinkingLevel`` helper."""
    return value in _VALID_THINKING_LEVELS


# ---------------------------------------------------------------------------
# Default model id per known provider
# ---------------------------------------------------------------------------


DEFAULT_MODEL_PER_PROVIDER: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-5.4",
    "azure-openai-responses": "gpt-5.2",
    "openai-codex": "gpt-5.4",
    "google": "gemini-2.5-pro",
    "google-gemini-cli": "gemini-2.5-pro",
    "google-antigravity": "gemini-3.1-pro-high",
    "google-vertex": "gemini-3-pro-preview",
    "github-copilot": "gpt-4o",
    "openrouter": "openai/gpt-5.1-codex",
    "vercel-ai-gateway": "anthropic/claude-opus-4-6",
    "xai": "grok-4-fast-non-reasoning",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "zai-glm-4.7",
    "zai": "glm-5",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "huggingface": "moonshotai/Kimi-K2.5",
    "opencode": "claude-opus-4-6",
    "opencode-go": "kimi-k2.5",
    "kimi-coding": "kimi-k2-thinking",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScopedModel:
    model: Model
    thinking_level: ThinkingLevel | None = None


@dataclass(slots=True)
class ParsedModelResult:
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None
    warning: str | None = None


@dataclass(slots=True)
class ResolveCliModelResult:
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None
    warning: str | None = None
    error: str | None = None


@dataclass(slots=True)
class InitialModelResult:
    model: Model | None
    thinking_level: ThinkingLevel
    fallback_message: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def _is_alias(model_id: str) -> bool:
    """Return ``True`` if ``model_id`` looks like an alias (no ``-YYYYMMDD``)."""
    if model_id.endswith("-latest"):
        return True
    return not _DATE_SUFFIX_RE.search(model_id)


def find_exact_model_reference_match(
    model_reference: str,
    available_models: list[Model],
) -> Model | None:
    """Look up a model by canonical ``provider/id`` or bare ``id``.

    Returns ``None`` if no match is found *or* if a bare-id lookup is
    ambiguous across providers (the upstream rejects ambiguous matches).
    """
    trimmed = model_reference.strip()
    if not trimmed:
        return None
    normalized = trimmed.lower()

    canonical_matches = [m for m in available_models if f"{m.provider}/{m.id}".lower() == normalized]
    if len(canonical_matches) == 1:
        return canonical_matches[0]
    if len(canonical_matches) > 1:
        return None

    slash_index = trimmed.find("/")
    if slash_index != -1:
        provider = trimmed[:slash_index].strip()
        model_id = trimmed[slash_index + 1 :].strip()
        if provider and model_id:
            provider_matches = [
                m
                for m in available_models
                if m.provider.lower() == provider.lower() and m.id.lower() == model_id.lower()
            ]
            if len(provider_matches) == 1:
                return provider_matches[0]
            if len(provider_matches) > 1:
                return None

    id_matches = [m for m in available_models if m.id.lower() == normalized]
    return id_matches[0] if len(id_matches) == 1 else None


def _try_match_model(model_pattern: str, available_models: list[Model]) -> Model | None:
    exact = find_exact_model_reference_match(model_pattern, available_models)
    if exact is not None:
        return exact

    lowered = model_pattern.lower()
    matches = [m for m in available_models if lowered in m.id.lower() or (m.name and lowered in m.name.lower())]
    if not matches:
        return None

    aliases = [m for m in matches if _is_alias(m.id)]
    dated = [m for m in matches if not _is_alias(m.id)]

    if aliases:
        # Prefer the alias whose id sorts highest (matches `b.localeCompare(a)`).
        aliases.sort(key=lambda m: m.id, reverse=True)
        return aliases[0]
    dated.sort(key=lambda m: m.id, reverse=True)
    return dated[0]


def parse_model_pattern(
    pattern: str,
    available_models: list[Model],
    *,
    allow_invalid_thinking_level_fallback: bool = True,
) -> ParsedModelResult:
    """Recursive ``model[:thinking_level]`` parser — port of ``parseModelPattern``.

    The recursion handles model ids that themselves contain colons (e.g.
    OpenRouter's ``provider/model:exacto``) by progressively stripping
    colon-suffixes until a match is found.
    """
    exact = _try_match_model(pattern, available_models)
    if exact is not None:
        return ParsedModelResult(model=exact)

    last_colon = pattern.rfind(":")
    if last_colon == -1:
        return ParsedModelResult()

    prefix = pattern[:last_colon]
    suffix = pattern[last_colon + 1 :]

    if _is_valid_thinking_level(suffix):
        result = parse_model_pattern(
            prefix,
            available_models,
            allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback,
        )
        if result.model is not None:
            return ParsedModelResult(
                model=result.model,
                thinking_level=None if result.warning else suffix,  # type: ignore[arg-type]
                warning=result.warning,
            )
        return result

    if not allow_invalid_thinking_level_fallback:
        # Strict mode: don't accidentally resolve to a different model.
        return ParsedModelResult()

    result = parse_model_pattern(
        prefix,
        available_models,
        allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback,
    )
    if result.model is not None:
        return ParsedModelResult(
            model=result.model,
            thinking_level=None,
            warning=f'Invalid thinking level "{suffix}" in pattern "{pattern}". Using default instead.',
        )
    return result


def _build_fallback_model(
    provider: str,
    model_id: str,
    available_models: list[Model],
) -> Model | None:
    provider_models = [m for m in available_models if m.provider == provider]
    if not provider_models:
        return None
    default_id = DEFAULT_MODEL_PER_PROVIDER.get(provider)
    base = next((m for m in provider_models if m.id == default_id), provider_models[0])
    fallback = base.model_copy(deep=True)
    fallback.id = model_id
    fallback.name = model_id
    return fallback


# ---------------------------------------------------------------------------
# resolve_model_scope (glob + thinking-level patterns from --models)
# ---------------------------------------------------------------------------


def _models_are_equal(a: Model, b: Model) -> bool:
    return a.provider == b.provider and a.id == b.id


def resolve_model_scope(
    patterns: list[str],
    model_registry: ModelRegistry,
) -> tuple[list[ScopedModel], list[str]]:
    """Expand a list of ``--models`` patterns into :class:`ScopedModel` entries.

    Returns ``(scoped_models, warnings)``. The upstream prints warnings
    via chalk; we surface them as a list so the caller decides how to
    render them.
    """
    available_models = model_registry.get_available()
    scoped: list[ScopedModel] = []
    warnings: list[str] = []

    for pattern in patterns:
        if any(c in pattern for c in "*?["):
            colon_idx = pattern.rfind(":")
            glob_pattern = pattern
            thinking_level: ThinkingLevel | None = None
            if colon_idx != -1:
                suffix = pattern[colon_idx + 1 :]
                if _is_valid_thinking_level(suffix):
                    thinking_level = suffix  # type: ignore[assignment]
                    glob_pattern = pattern[:colon_idx]
            matching = [
                m
                for m in available_models
                if fnmatch.fnmatch(f"{m.provider}/{m.id}".lower(), glob_pattern.lower())
                or fnmatch.fnmatch(m.id.lower(), glob_pattern.lower())
            ]
            if not matching:
                warnings.append(f'No models match pattern "{pattern}"')
                continue
            for model in matching:
                if not any(_models_are_equal(sm.model, model) for sm in scoped):
                    scoped.append(ScopedModel(model=model, thinking_level=thinking_level))
            continue

        result = parse_model_pattern(pattern, available_models)
        if result.warning:
            warnings.append(result.warning)
        if result.model is None:
            warnings.append(f'No models match pattern "{pattern}"')
            continue
        if not any(_models_are_equal(sm.model, result.model) for sm in scoped):
            scoped.append(ScopedModel(model=result.model, thinking_level=result.thinking_level))

    return scoped, warnings


# ---------------------------------------------------------------------------
# resolve_cli_model
# ---------------------------------------------------------------------------


def resolve_cli_model(
    *,
    cli_provider: str | None,
    cli_model: str | None,
    model_registry: ModelRegistry,
) -> ResolveCliModelResult:
    """Resolve a single model from CLI ``--provider``/``--model`` flags."""
    if not cli_model:
        return ResolveCliModelResult()

    available_models = model_registry.get_all()
    if not available_models:
        return ResolveCliModelResult(error="No models available. Check your installation or add models to models.json.")

    provider_map: dict[str, str] = {}
    for m in available_models:
        provider_map[m.provider.lower()] = m.provider

    provider = provider_map.get(cli_provider.lower()) if cli_provider else None
    if cli_provider and provider is None:
        return ResolveCliModelResult(
            error=f'Unknown provider "{cli_provider}". Use --list-models to see available providers/models.'
        )

    pattern = cli_model
    inferred_provider = False

    if provider is None:
        slash_index = cli_model.find("/")
        if slash_index != -1:
            maybe_provider = cli_model[:slash_index]
            canonical = provider_map.get(maybe_provider.lower())
            if canonical is not None:
                provider = canonical
                pattern = cli_model[slash_index + 1 :]
                inferred_provider = True

    if provider is None:
        lower = cli_model.lower()
        exact = next(
            (m for m in available_models if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower),
            None,
        )
        if exact is not None:
            return ResolveCliModelResult(model=exact)

    if cli_provider and provider:
        prefix = f"{provider}/"
        if cli_model.lower().startswith(prefix.lower()):
            pattern = cli_model[len(prefix) :]

    candidates = [m for m in available_models if m.provider == provider] if provider else available_models
    parsed = parse_model_pattern(
        pattern,
        candidates,
        allow_invalid_thinking_level_fallback=False,
    )

    if parsed.model is not None:
        return ResolveCliModelResult(
            model=parsed.model,
            thinking_level=parsed.thinking_level,
            warning=parsed.warning,
        )

    if inferred_provider:
        lower = cli_model.lower()
        exact = next(
            (m for m in available_models if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower),
            None,
        )
        if exact is not None:
            return ResolveCliModelResult(model=exact)
        fallback = parse_model_pattern(
            cli_model,
            available_models,
            allow_invalid_thinking_level_fallback=False,
        )
        if fallback.model is not None:
            return ResolveCliModelResult(
                model=fallback.model,
                thinking_level=fallback.thinking_level,
                warning=fallback.warning,
            )

    if provider:
        fallback_model = _build_fallback_model(provider, pattern, available_models)
        if fallback_model is not None:
            warning = parsed.warning
            extra = f'Model "{pattern}" not found for provider "{provider}". Using custom model id.'
            combined_warning = f"{warning} {extra}" if warning else extra
            return ResolveCliModelResult(
                model=fallback_model,
                warning=combined_warning,
            )

    display = f"{provider}/{pattern}" if provider else cli_model
    return ResolveCliModelResult(
        warning=parsed.warning,
        error=f'Model "{display}" not found. Use --list-models to see available models.',
    )


# ---------------------------------------------------------------------------
# find_initial_model
# ---------------------------------------------------------------------------


def find_initial_model(
    *,
    cli_provider: str | None,
    cli_model: str | None,
    scoped_models: list[ScopedModel],
    is_continuing: bool,
    default_provider: str | None,
    default_model_id: str | None,
    default_thinking_level: ThinkingLevel | None,
    model_registry: ModelRegistry,
) -> InitialModelResult:
    """Pick the model to start a session with, mirroring the upstream priority.

    Order:

    1. ``--provider`` + ``--model`` flags.
    2. First scoped model (skipped when resuming a session).
    3. Saved default from settings.
    4. First available model that has credentials configured.
    """
    if cli_provider and cli_model:
        resolved = resolve_cli_model(
            cli_provider=cli_provider,
            cli_model=cli_model,
            model_registry=model_registry,
        )
        if resolved.error is not None:
            raise ValueError(resolved.error)
        if resolved.model is not None:
            return InitialModelResult(model=resolved.model, thinking_level=DEFAULT_THINKING_LEVEL)

    if scoped_models and not is_continuing:
        first = scoped_models[0]
        return InitialModelResult(
            model=first.model,
            thinking_level=first.thinking_level or default_thinking_level or DEFAULT_THINKING_LEVEL,
        )

    if default_provider and default_model_id:
        found = model_registry.find(default_provider, default_model_id)
        if found is not None:
            return InitialModelResult(
                model=found,
                thinking_level=default_thinking_level or DEFAULT_THINKING_LEVEL,
            )

    available_models = model_registry.get_available()
    if available_models:
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next(
                (m for m in available_models if m.provider == provider and m.id == default_id),
                None,
            )
            if match is not None:
                return InitialModelResult(model=match, thinking_level=DEFAULT_THINKING_LEVEL)
        return InitialModelResult(model=available_models[0], thinking_level=DEFAULT_THINKING_LEVEL)

    return InitialModelResult(model=None, thinking_level=DEFAULT_THINKING_LEVEL)


# ---------------------------------------------------------------------------
# restore_model_from_session
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RestoredModelResult:
    model: Model | None
    fallback_message: str | None = None


def restore_model_from_session(
    saved_provider: str,
    saved_model_id: str,
    current_model: Model | None,
    model_registry: ModelRegistry,
) -> RestoredModelResult:
    """Re-resolve a model that was persisted in a session, with auth re-check."""
    restored = model_registry.find(saved_provider, saved_model_id)
    has_auth = model_registry.has_configured_auth(restored) if restored is not None else False

    if restored is not None and has_auth:
        return RestoredModelResult(model=restored)

    reason = "model no longer exists" if restored is None else "no auth configured"

    if current_model is not None:
        return RestoredModelResult(
            model=current_model,
            fallback_message=(
                f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
                f"Using {current_model.provider}/{current_model.id}."
            ),
        )

    available_models = model_registry.get_available()
    if available_models:
        fallback_model: Model | None = None
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next(
                (m for m in available_models if m.provider == provider and m.id == default_id),
                None,
            )
            if match is not None:
                fallback_model = match
                break
        if fallback_model is None:
            fallback_model = available_models[0]
        return RestoredModelResult(
            model=fallback_model,
            fallback_message=(
                f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
                f"Using {fallback_model.provider}/{fallback_model.id}."
            ),
        )

    return RestoredModelResult(model=None)


__all__ = [
    "DEFAULT_MODEL_PER_PROVIDER",
    "InitialModelResult",
    "ParsedModelResult",
    "ResolveCliModelResult",
    "RestoredModelResult",
    "ScopedModel",
    "ThinkingLevel",
    "find_exact_model_reference_match",
    "find_initial_model",
    "parse_model_pattern",
    "resolve_cli_model",
    "resolve_model_scope",
    "restore_model_from_session",
]
