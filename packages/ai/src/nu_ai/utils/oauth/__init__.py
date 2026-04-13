"""OAuth credential management for AI providers.

This module handles login, token refresh, and credential storage
for OAuth-based providers:
- Anthropic (Claude Pro/Max)
- GitHub Copilot
- Google Cloud Code Assist (Gemini CLI)
- Antigravity (Gemini 3, Claude, GPT-OSS via Google Cloud)
- OpenAI Codex (ChatGPT OAuth)
"""

from __future__ import annotations

from nu_ai.utils.oauth.types import (
    OAuthAuthInfo,
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthProviderInterface,
)

# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------

_BUILT_IN_OAUTH_PROVIDERS: list[OAuthProviderInterface] = []
_oauth_provider_registry: dict[str, OAuthProviderInterface] = {}
_built_ins_loaded = False


def _ensure_built_ins() -> None:
    """Lazily load built-in OAuth providers on first registry access."""
    global _built_ins_loaded
    if _built_ins_loaded:
        return
    _built_ins_loaded = True

    from nu_ai.utils.oauth.anthropic import anthropic_oauth_provider
    from nu_ai.utils.oauth.github_copilot import github_copilot_oauth_provider
    from nu_ai.utils.oauth.google_antigravity import antigravity_oauth_provider
    from nu_ai.utils.oauth.google_gemini_cli import gemini_cli_oauth_provider
    from nu_ai.utils.oauth.openai_codex import openai_codex_oauth_provider

    _BUILT_IN_OAUTH_PROVIDERS.clear()
    _BUILT_IN_OAUTH_PROVIDERS.extend(
        [
            anthropic_oauth_provider,
            github_copilot_oauth_provider,
            gemini_cli_oauth_provider,
            antigravity_oauth_provider,
            openai_codex_oauth_provider,
        ]
    )

    _oauth_provider_registry.clear()
    for provider in _BUILT_IN_OAUTH_PROVIDERS:
        _oauth_provider_registry[provider.id] = provider


def get_oauth_provider(provider_id: str) -> OAuthProviderInterface | None:
    """Get an OAuth provider by ID."""
    _ensure_built_ins()
    return _oauth_provider_registry.get(provider_id)


def register_oauth_provider(provider: OAuthProviderInterface) -> None:
    """Register a custom OAuth provider."""
    _ensure_built_ins()
    _oauth_provider_registry[provider.id] = provider


def unregister_oauth_provider(provider_id: str) -> None:
    """Unregister an OAuth provider.

    If the provider is built-in, restores the built-in implementation.
    Custom providers are removed completely.
    """
    _ensure_built_ins()
    built_in = next((p for p in _BUILT_IN_OAUTH_PROVIDERS if p.id == provider_id), None)
    if built_in is not None:
        _oauth_provider_registry[provider_id] = built_in
        return
    _oauth_provider_registry.pop(provider_id, None)


def reset_oauth_providers() -> None:
    """Reset OAuth providers to built-ins."""
    _ensure_built_ins()
    _oauth_provider_registry.clear()
    for provider in _BUILT_IN_OAUTH_PROVIDERS:
        _oauth_provider_registry[provider.id] = provider


def get_oauth_providers() -> list[OAuthProviderInterface]:
    """Get all registered OAuth providers."""
    _ensure_built_ins()
    return list(_oauth_provider_registry.values())


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


async def refresh_oauth_token(
    provider_id: str,
    credentials: OAuthCredentials,
) -> OAuthCredentials:
    """Refresh token for any OAuth provider.

    .. deprecated:: Use ``get_oauth_provider(id).refresh_token()`` instead.
    """
    provider = get_oauth_provider(provider_id)
    if provider is None:
        raise ValueError(f"Unknown OAuth provider: {provider_id}")
    return await provider.refresh_token(credentials)


async def get_oauth_api_key(
    provider_id: str,
    credentials: dict[str, OAuthCredentials],
) -> tuple[OAuthCredentials, str] | None:
    """Get API key for a provider from OAuth credentials.

    Automatically refreshes expired tokens.

    Returns:
        Tuple of (updated credentials, api_key), or None if no credentials.

    Raises:
        ValueError: If provider is unknown.
        RuntimeError: If refresh fails.
    """
    import time

    provider = get_oauth_provider(provider_id)
    if provider is None:
        raise ValueError(f"Unknown OAuth provider: {provider_id}")

    creds = credentials.get(provider_id)
    if creds is None:
        return None

    # Refresh if expired
    if time.time() * 1000 >= creds.expires:
        try:
            creds = await provider.refresh_token(creds)
        except Exception:
            raise RuntimeError(f"Failed to refresh OAuth token for {provider_id}") from None

    api_key = provider.get_api_key(creds)
    return (creds, api_key)


__all__ = [
    "OAuthAuthInfo",
    "OAuthCredentials",
    "OAuthLoginCallbacks",
    "OAuthPrompt",
    "OAuthProviderInterface",
    "get_oauth_api_key",
    "get_oauth_provider",
    "get_oauth_providers",
    "refresh_oauth_token",
    "register_oauth_provider",
    "reset_oauth_providers",
    "unregister_oauth_provider",
]
