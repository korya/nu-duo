"""OpenAI Codex (ChatGPT OAuth) flow.

Port of ``utils/oauth/openai-codex.ts``. Uses authorization code + PKCE
with a local callback server.
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

from nu_ai.utils.oauth.callback_server import CallbackServer, parse_authorization_input
from nu_ai.utils.oauth.pkce import generate_pkce
from nu_ai.utils.oauth.types import OAuthCredentials, OAuthProviderInterface

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_REDIRECT_URI = "http://localhost:1455/auth/callback"
_SCOPE = "openid profile email offline_access"
_JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _create_state() -> str:
    return os.urandom(16).hex()


def _decode_jwt(token: str) -> dict[str, Any] | None:
    """Decode JWT payload (no verification)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        # Add padding
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        decoded = base64.b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return None


def _get_account_id(access_token: str) -> str | None:
    """Extract chatgpt_account_id from JWT."""
    payload = _decode_jwt(access_token)
    if payload is None:
        return None
    auth = payload.get(_JWT_CLAIM_PATH)
    if not isinstance(auth, dict):
        return None
    account_id = auth.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


async def _exchange_code(code: str, verifier: str, redirect_uri: str = _REDIRECT_URI) -> dict[str, Any] | None:
    """Exchange authorization code for tokens. Returns None on failure."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            content=urlencode(
                {
                    "grant_type": "authorization_code",
                    "client_id": _CLIENT_ID,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": redirect_uri,
                }
            ),
        )

        if not response.is_success:
            logger.error("[openai-codex] code->token failed: %s %s", response.status_code, response.text)
            return None

        data = response.json()
        if not data.get("access_token") or not data.get("refresh_token") or not isinstance(data.get("expires_in"), int):
            logger.error("[openai-codex] token response missing fields: %s", data)
            return None

        return {
            "access": data["access_token"],
            "refresh": data["refresh_token"],
            "expires": time.time() * 1000 + data["expires_in"] * 1000,
        }


async def _refresh_access_token(refresh_token: str) -> dict[str, Any] | None:
    """Refresh access token. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                _TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                content=urlencode(
                    {
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": _CLIENT_ID,
                    }
                ),
            )

            if not response.is_success:
                logger.error("[openai-codex] Token refresh failed: %s %s", response.status_code, response.text)
                return None

            data = response.json()
            if (
                not data.get("access_token")
                or not data.get("refresh_token")
                or not isinstance(data.get("expires_in"), int)
            ):
                logger.error("[openai-codex] Token refresh response missing fields: %s", data)
                return None

            return {
                "access": data["access_token"],
                "refresh": data["refresh_token"],
                "expires": time.time() * 1000 + data["expires_in"] * 1000,
            }
    except Exception:
        logger.exception("[openai-codex] Token refresh error")
        return None


async def login_openai_codex(
    *,
    on_auth: Callable[[dict[str, Any]], None],
    on_prompt: Callable[[dict[str, Any]], Awaitable[str]],
    on_progress: Callable[[str], None] | None = None,
    on_manual_code_input: Callable[[], Awaitable[str]] | None = None,
    originator: str = "nu",
) -> OAuthCredentials:
    """Login with OpenAI Codex OAuth."""
    verifier, challenge = generate_pkce()
    state = _create_state()

    params = urlencode(
        {
            "response_type": "code",
            "client_id": _CLIENT_ID,
            "redirect_uri": _REDIRECT_URI,
            "scope": _SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": originator,
        }
    )
    auth_url = f"{_AUTHORIZE_URL}?{params}"

    server = CallbackServer(
        port=1455,
        path="/auth/callback",
        expected_state=state,
        provider_name="OpenAI",
    )
    await server.start()

    on_auth({"url": auth_url, "instructions": "A browser window should open. Complete login to finish."})

    code: str | None = None
    try:
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
                code = result.code
            elif manual_input is not None:
                parsed = parse_authorization_input(manual_input)
                if parsed["state"] and parsed["state"] != state:
                    raise ValueError("State mismatch")
                code = parsed["code"]

            if code is None:
                await manual_task
                if manual_error is not None:
                    raise manual_error
                if manual_input is not None:
                    parsed = parse_authorization_input(manual_input)
                    if parsed["state"] and parsed["state"] != state:
                        raise ValueError("State mismatch")
                    code = parsed["code"]
        else:
            result = await server.wait_for_code()
            if result is not None:
                code = result.code

        # Fallback to prompt
        if code is None:
            input_value = await on_prompt({"message": "Paste the authorization code (or full redirect URL):"})
            parsed = parse_authorization_input(input_value)
            if parsed["state"] and parsed["state"] != state:
                raise ValueError("State mismatch")
            code = parsed["code"]

        if code is None:
            raise ValueError("Missing authorization code")

        token_result = await _exchange_code(code, verifier)
        if token_result is None:
            raise RuntimeError("Token exchange failed")

        account_id = _get_account_id(token_result["access"])
        if not account_id:
            raise RuntimeError("Failed to extract accountId from token")

        return OAuthCredentials(
            access=token_result["access"],
            refresh=token_result["refresh"],
            expires=token_result["expires"],
            extra_data={"accountId": account_id},
        )
    finally:
        server.close()


async def refresh_openai_codex_token(refresh_token: str) -> OAuthCredentials:
    """Refresh OpenAI Codex OAuth token."""
    result = await _refresh_access_token(refresh_token)
    if result is None:
        raise RuntimeError("Failed to refresh OpenAI Codex token")

    account_id = _get_account_id(result["access"])
    if not account_id:
        raise RuntimeError("Failed to extract accountId from token")

    return OAuthCredentials(
        access=result["access"],
        refresh=result["refresh"],
        expires=result["expires"],
        extra_data={"accountId": account_id},
    )


class _OpenAICodexOAuthProvider:
    """OpenAI Codex OAuth provider implementation."""

    @property
    def id(self) -> str:
        return "openai-codex"

    @property
    def name(self) -> str:
        return "ChatGPT Plus/Pro (Codex Subscription)"

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
        return await login_openai_codex(
            on_auth=on_auth,
            on_prompt=on_prompt,
            on_progress=on_progress,
            on_manual_code_input=on_manual_code_input,
        )

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return await refresh_openai_codex_token(credentials.refresh)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access

    def modify_models(self, models: list[Any], credentials: OAuthCredentials) -> list[Any]:
        return models


openai_codex_oauth_provider: OAuthProviderInterface = _OpenAICodexOAuthProvider()  # type: ignore[assignment]
