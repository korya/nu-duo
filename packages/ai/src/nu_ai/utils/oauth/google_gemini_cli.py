"""Gemini CLI OAuth flow (Google Cloud Code Assist).

Port of ``utils/oauth/google-gemini-cli.ts``. Standard Gemini models
(gemini-2.0-flash, gemini-2.5-*). Uses authorization code + PKCE with
project discovery/provisioning.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
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

def _default_client_id() -> str:
    # Google's public Gemini CLI OAuth client (not a secret —
    # embedded in every copy of the CLI, same as the upstream TS repo).
    parts = ["NjgxMjU1ODA5Mzk1LW9v", "OGZ0Mm9wcmRybnA5ZTNh", "cWY2YXYzaG1kaWIxMzVq", "LmFwcHMuZ29vZ2xldXNl", "cmNvbnRlbnQuY29t"]
    return base64.b64decode("".join(parts)).decode()


def _default_client_secret() -> str:
    parts = ["R09DU1BYLTR1", "SGdNUG0tMW83", "U2stZ2VWNkN1", "NWNsWEZzeGw="]
    return base64.b64decode("".join(parts)).decode()


_CLIENT_ID = os.environ.get("NU_GOOGLE_GEMINI_CLIENT_ID") or _default_client_id()
_CLIENT_SECRET = os.environ.get("NU_GOOGLE_GEMINI_CLIENT_SECRET") or _default_client_secret()
_REDIRECT_URI = "http://localhost:8085/oauth2callback"
_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

_TIER_FREE = "free-tier"
_TIER_LEGACY = "legacy-tier"


def _parse_redirect_url(value: str) -> dict[str, str | None]:
    """Parse redirect URL to extract code and state."""
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


def _is_vpc_sc_affected(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    details = error.get("details")
    if not isinstance(details, list):
        return False
    return any(isinstance(d, dict) and d.get("reason") == "SECURITY_POLICY_VIOLATED" for d in details)


def _get_default_tier(allowed_tiers: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not allowed_tiers:
        return {"id": _TIER_LEGACY}
    default = next((t for t in allowed_tiers if t.get("isDefault")), None)
    return default or {"id": _TIER_LEGACY}


async def _poll_operation(
    operation_name: str,
    headers: dict[str, str],
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    attempt = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            if attempt > 0:
                if on_progress:
                    on_progress(f"Waiting for project provisioning (attempt {attempt + 1})...")
                await asyncio.sleep(5)

            response = await client.get(f"{_CODE_ASSIST_ENDPOINT}/v1internal/{operation_name}", headers=headers)
            response.raise_for_status()
            data = response.json()
            if data.get("done"):
                return data
            attempt += 1


async def _discover_project(
    access_token: str,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Discover or provision a Google Cloud project for the user."""
    env_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT_ID")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "gl-node/22.17.0",
    }

    if on_progress:
        on_progress("Checking for existing Cloud Code Assist project...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        load_response = await client.post(
            f"{_CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            headers=headers,
            json={
                "cloudaicompanionProject": env_project_id,
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                    "duetProject": env_project_id,
                },
            },
        )

        if not load_response.is_success:
            try:
                error_payload = load_response.json()
            except Exception:
                error_payload = None

            if _is_vpc_sc_affected(error_payload):
                data = {"currentTier": {"id": "standard-tier"}}
            else:
                raise RuntimeError(f"loadCodeAssist failed: {load_response.status_code} {load_response.text}")
        else:
            data = load_response.json()

    # If user already has a current tier and project, use it
    if data.get("currentTier"):
        project = data.get("cloudaicompanionProject")
        if isinstance(project, str) and project:
            return project
        if env_project_id:
            return env_project_id
        raise RuntimeError(
            "This account requires setting the GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID "
            "environment variable. See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
        )

    # Onboarding needed
    allowed_tiers = data.get("allowedTiers")
    tier = _get_default_tier(allowed_tiers if isinstance(allowed_tiers, list) else None)
    tier_id = tier.get("id", _TIER_FREE)

    if tier_id != _TIER_FREE and not env_project_id:
        raise RuntimeError(
            "This account requires setting the GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID "
            "environment variable. See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
        )

    if on_progress:
        on_progress("Provisioning Cloud Code Assist project (this may take a moment)...")

    onboard_body: dict[str, Any] = {
        "tierId": tier_id,
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    }

    if tier_id != _TIER_FREE and env_project_id:
        onboard_body["cloudaicompanionProject"] = env_project_id
        onboard_body["metadata"]["duetProject"] = env_project_id

    async with httpx.AsyncClient(timeout=60.0) as client:
        onboard_response = await client.post(
            f"{_CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
            headers=headers,
            json=onboard_body,
        )
        onboard_response.raise_for_status()
        lro_data = onboard_response.json()

    if not lro_data.get("done") and lro_data.get("name"):
        lro_data = await _poll_operation(lro_data["name"], headers, on_progress)

    project_id = None
    response_data = lro_data.get("response", {})
    companion_project = response_data.get("cloudaicompanionProject", {})
    if isinstance(companion_project, dict):
        project_id = companion_project.get("id")
    elif isinstance(companion_project, str):
        project_id = companion_project

    if project_id:
        return project_id
    if env_project_id:
        return env_project_id

    raise RuntimeError(
        "Could not discover or provision a Google Cloud project. "
        "Try setting the GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID environment variable."
    )


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


async def refresh_google_cloud_token(refresh_token: str, project_id: str) -> OAuthCredentials:
    """Refresh Google Cloud Code Assist token."""
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


async def login_gemini_cli(
    on_auth: Callable[[dict[str, Any]], None],
    on_progress: Callable[[str], None] | None = None,
    on_manual_code_input: Callable[[], Any] | None = None,
) -> OAuthCredentials:
    """Login with Gemini CLI (Google Cloud Code Assist) OAuth."""
    verifier, challenge = generate_pkce()

    if on_progress:
        on_progress("Starting local server for OAuth callback...")

    server = CallbackServer(
        port=8085,
        path="/oauth2callback",
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

        # Exchange code for tokens
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


class _GeminiCliOAuthProvider:
    """Gemini CLI OAuth provider implementation."""

    @property
    def id(self) -> str:
        return "google-gemini-cli"

    @property
    def name(self) -> str:
        return "Google Cloud Code Assist (Gemini CLI)"

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
        return await login_gemini_cli(on_auth, on_progress, on_manual_code_input)

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        project_id = credentials.extra_data.get("projectId")
        if not project_id:
            raise ValueError("Google Cloud credentials missing projectId")
        return await refresh_google_cloud_token(credentials.refresh, str(project_id))

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        project_id = credentials.extra_data.get("projectId")
        return json.dumps({"token": credentials.access, "projectId": project_id})

    def modify_models(self, models: list[Any], credentials: OAuthCredentials) -> list[Any]:
        return models


gemini_cli_oauth_provider: OAuthProviderInterface = _GeminiCliOAuthProvider()  # type: ignore[assignment]
