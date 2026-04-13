"""Antigravity OAuth flow (Gemini 3, Claude, GPT-OSS via Google Cloud).

Port of ``utils/oauth/google-antigravity.ts``. Uses different OAuth
credentials than google-gemini-cli for access to additional models.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx

from nu_ai.utils.oauth.callback_server import CallbackServer
from nu_ai.utils.oauth.pkce import generate_pkce
from nu_ai.utils.oauth.types import OAuthCredentials, OAuthProviderInterface

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_CLIENT_ID = base64.b64decode(
    "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ=="
).decode()
_CLIENT_SECRET = base64.b64decode("R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=").decode()
_REDIRECT_URI = "http://localhost:51121/oauth-callback"
_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_DEFAULT_PROJECT_ID = "rising-fact-p41fc"


def _parse_redirect_url(value: str) -> dict[str, str | None]:
    value = value.strip()
    if not value:
        return {"code": None, "state": None}
    try:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(value)
        params = parse_qs(parsed.query)
        return {
            "code": params.get("code", [None])[0],
            "state": params.get("state", [None])[0],
        }
    except Exception:
        return {"code": None, "state": None}


async def _discover_project(
    access_token: str,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Discover or provision a project for the user."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps(
            {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        ),
    }

    endpoints = [
        "https://cloudcode-pa.googleapis.com",
        "https://daily-cloudcode-pa.sandbox.googleapis.com",
    ]

    if on_progress:
        on_progress("Checking for existing project...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint in endpoints:
            try:
                response = await client.post(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    headers=headers,
                    json={
                        "metadata": {
                            "ideType": "IDE_UNSPECIFIED",
                            "platform": "PLATFORM_UNSPECIFIED",
                            "pluginType": "GEMINI",
                        },
                    },
                )

                if response.is_success:
                    data = response.json()
                    project = data.get("cloudaicompanionProject")
                    if isinstance(project, str) and project:
                        return project
                    if isinstance(project, dict) and project.get("id"):
                        return project["id"]
            except Exception:
                continue

    if on_progress:
        on_progress("Using default project...")
    return _DEFAULT_PROJECT_ID


async def _get_user_email(access_token: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.is_success:
                return response.json().get("email")
    except Exception:
        pass
    return None


async def refresh_antigravity_token(refresh_token: str, project_id: str) -> OAuthCredentials:
    """Refresh Antigravity token."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            content=urlencode(
                {
                    "client_id": _CLIENT_ID,
                    "client_secret": _CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            ),
        )
        response.raise_for_status()
        data = response.json()

    return OAuthCredentials(
        refresh=data.get("refresh_token", refresh_token),
        access=data["access_token"],
        expires=time.time() * 1000 + data["expires_in"] * 1000 - 5 * 60 * 1000,
        extra_data={"projectId": project_id},
    )


async def login_antigravity(
    on_auth: Callable[[dict[str, Any]], None],
    on_progress: Callable[[str], None] | None = None,
    on_manual_code_input: Callable[[], Any] | None = None,
) -> OAuthCredentials:
    """Login with Antigravity OAuth."""
    verifier, challenge = generate_pkce()

    if on_progress:
        on_progress("Starting local server for OAuth callback...")

    server = CallbackServer(
        port=51121,
        path="/oauth-callback",
        provider_name="Google",
    )
    await server.start()

    code: str | None = None

    try:
        auth_params = urlencode(
            {
                "client_id": _CLIENT_ID,
                "response_type": "code",
                "redirect_uri": _REDIRECT_URI,
                "scope": " ".join(_SCOPES),
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": verifier,
                "access_type": "offline",
                "prompt": "consent",
            }
        )

        on_auth(
            {
                "url": f"{_AUTH_URL}?{auth_params}",
                "instructions": "Complete the sign-in in your browser.",
            }
        )

        if on_progress:
            on_progress("Waiting for OAuth callback...")

        if on_manual_code_input is not None:
            manual_input: str | None = None
            manual_error: Exception | None = None

            async def _get_manual() -> None:
                nonlocal manual_input, manual_error
                try:
                    manual_input = await on_manual_code_input()
                    server.cancel_wait()
                except Exception as exc:
                    manual_error = exc
                    server.cancel_wait()

            manual_task = asyncio.create_task(_get_manual())
            result = await server.wait_for_code()

            if manual_error is not None:
                raise manual_error

            if result is not None:
                if result.state != verifier:
                    raise ValueError("OAuth state mismatch - possible CSRF attack")
                code = result.code
            elif manual_input is not None:
                parsed = _parse_redirect_url(manual_input)
                if parsed["state"] and parsed["state"] != verifier:
                    raise ValueError("OAuth state mismatch - possible CSRF attack")
                code = parsed["code"]

            if code is None:
                await manual_task
                if manual_error is not None:
                    raise manual_error
                if manual_input is not None:
                    parsed = _parse_redirect_url(manual_input)
                    if parsed["state"] and parsed["state"] != verifier:
                        raise ValueError("OAuth state mismatch - possible CSRF attack")
                    code = parsed["code"]
        else:
            result = await server.wait_for_code()
            if result is not None:
                if result.state != verifier:
                    raise ValueError("OAuth state mismatch - possible CSRF attack")
                code = result.code

        if code is None:
            raise ValueError("No authorization code received")

        if on_progress:
            on_progress("Exchanging authorization code for tokens...")

        async with httpx.AsyncClient(timeout=30.0) as client:
            token_response = await client.post(
                _TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                content=urlencode(
                    {
                        "client_id": _CLIENT_ID,
                        "client_secret": _CLIENT_SECRET,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": _REDIRECT_URI,
                        "code_verifier": verifier,
                    }
                ),
            )
            token_response.raise_for_status()
            token_data = token_response.json()

        if not token_data.get("refresh_token"):
            raise ValueError("No refresh token received. Please try again.")

        if on_progress:
            on_progress("Getting user info...")
        email = await _get_user_email(token_data["access_token"])

        project_id = await _discover_project(token_data["access_token"], on_progress)

        expires_at = time.time() * 1000 + token_data["expires_in"] * 1000 - 5 * 60 * 1000

        extra: dict[str, object] = {"projectId": project_id}
        if email:
            extra["email"] = email
        return OAuthCredentials(
            refresh=token_data["refresh_token"],
            access=token_data["access_token"],
            expires=expires_at,
            extra_data=extra,
        )
    finally:
        server.close()


class _AntigravityOAuthProvider:
    """Antigravity OAuth provider implementation."""

    @property
    def id(self) -> str:
        return "google-antigravity"

    @property
    def name(self) -> str:
        return "Antigravity (Gemini 3, Claude, GPT-OSS)"

    @property
    def uses_callback_server(self) -> bool:
        return True

    async def login(
        self,
        *,
        on_auth: Any,
        on_prompt: Any,
        on_progress: Any | None = None,
        on_manual_code_input: Any | None = None,
    ) -> OAuthCredentials:
        return await login_antigravity(on_auth, on_progress, on_manual_code_input)

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        project_id = credentials.extra_data.get("projectId")
        if not project_id:
            raise ValueError("Antigravity credentials missing projectId")
        return await refresh_antigravity_token(credentials.refresh, str(project_id))

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        project_id = credentials.extra_data.get("projectId")
        return json.dumps({"token": credentials.access, "projectId": project_id})

    def modify_models(self, models: list[Any], credentials: OAuthCredentials) -> list[Any]:
        return models


antigravity_oauth_provider: OAuthProviderInterface = _AntigravityOAuthProvider()  # type: ignore[assignment]
