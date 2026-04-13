"""OAuth type definitions.

Port of ``utils/oauth/types.ts``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from nu_ai.types import Api, Model


class OAuthCredentials(BaseModel):
    """OAuth credentials for a provider."""

    refresh: str
    access: str
    expires: float
    """Expiry timestamp in milliseconds since epoch."""

    # Provider-specific extra fields (e.g. projectId, enterpriseUrl, accountId, email)
    extra_data: dict[str, object] = {}
    """Provider-specific extra data stored alongside the standard fields."""

    model_config = {"extra": "forbid"}


class OAuthPrompt(BaseModel):
    """Prompt shown to the user during OAuth login."""

    message: str
    placeholder: str | None = None
    allow_empty: bool = False


class OAuthAuthInfo(BaseModel):
    """Auth info shown to the user (URL to open, instructions)."""

    url: str
    instructions: str | None = None


class OAuthLoginCallbacks(BaseModel):
    """Callbacks for the OAuth login flow.

    Note: In Python we store these as plain callables rather than
    using Pydantic validation, since they are protocol callbacks.
    """

    model_config = {"arbitrary_types_allowed": True}

    on_auth: object  # Callable[[OAuthAuthInfo], None]
    on_prompt: object  # Callable[[OAuthPrompt], Awaitable[str]]
    on_progress: object | None = None  # Callable[[str], None] | None
    on_manual_code_input: object | None = None  # Callable[[], Awaitable[str]] | None
    signal: object | None = None  # Not used in Python (use asyncio cancellation)


@runtime_checkable
class OAuthProviderInterface(Protocol):
    """Interface for an OAuth provider."""

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def uses_callback_server(self) -> bool: ...

    async def login(
        self,
        *,
        on_auth: object,
        on_prompt: object,
        on_progress: object | None = None,
        on_manual_code_input: object | None = None,
    ) -> OAuthCredentials:
        """Run the login flow, return credentials to persist."""
        ...

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        """Refresh expired credentials, return updated credentials."""
        ...

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        """Convert credentials to API key string for the provider."""
        ...

    def modify_models(self, models: list[Model[Api]], credentials: OAuthCredentials) -> list[Model[Api]]:
        """Optional: modify models for this provider (e.g., update base_url)."""
        ...
