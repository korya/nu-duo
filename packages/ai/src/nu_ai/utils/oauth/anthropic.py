"""Anthropic OAuth flow (Claude Pro/Max).

Port of ``utils/oauth/anthropic.ts``. Uses authorization code + PKCE
with a local callback server for the redirect.
"""

from __future__ import annotations

import base64
import logging
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

_CLIENT_ID = base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode()
_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_CALLBACK_PORT = 53692
_CALLBACK_PATH = "/callback"
_REDIRECT_URI = f"http://localhost:{_CALLBACK_PORT}{_CALLBACK_PATH}"
_SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"


async def _post_json(url: str, body: dict[str, Any]) -> str:
    """POST JSON and return response text."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response.raise_for_status()
        return response.text


async def _exchange_authorization_code(
    code: str,
    state: str,
    verifier: str,
    redirect_uri: str,
) -> OAuthCredentials:
    """Exchange authorization code for tokens."""
    try:
        response_body = await _post_json(
            _TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "client_id": _CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
        )
    except Exception as exc:
        raise RuntimeError(
            f"Token exchange request failed. url={_TOKEN_URL}; redirect_uri={redirect_uri}; details={exc}"
        ) from exc

    import json

    try:
        data = json.loads(response_body)
    except Exception as exc:
        raise RuntimeError(
            f"Token exchange returned invalid JSON. url={_TOKEN_URL}; body={response_body}; details={exc}"
        ) from exc

    return OAuthCredentials(
        refresh=data["refresh_token"],
        access=data["access_token"],
        expires=time.time() * 1000 + data["expires_in"] * 1000 - 5 * 60 * 1000,
    )


async def login_anthropic(
    *,
    on_auth: Callable[[dict[str, Any]], None],
    on_prompt: Callable[[dict[str, Any]], Awaitable[str]],
    on_progress: Callable[[str], None] | None = None,
    on_manual_code_input: Callable[[], Awaitable[str]] | None = None,
) -> OAuthCredentials:
    """Login with Anthropic OAuth (authorization code + PKCE)."""
    verifier, challenge = generate_pkce()

    server = CallbackServer(
        port=_CALLBACK_PORT,
        path=_CALLBACK_PATH,
        expected_state=verifier,
        provider_name="Anthropic",
    )
    await server.start()

    code: str | None = None
    state: str | None = None
    redirect_uri_for_exchange = _REDIRECT_URI

    try:
        auth_params = urlencode(
            {
                "code": "true",
                "client_id": _CLIENT_ID,
                "response_type": "code",
                "redirect_uri": _REDIRECT_URI,
                "scope": _SCOPES,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": verifier,
            }
        )

        on_auth(
            {
                "url": f"{_AUTHORIZE_URL}?{auth_params}",
                "instructions": "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here.",
            }
        )

        if on_manual_code_input is not None:
            import asyncio

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
                state = result.state
                redirect_uri_for_exchange = _REDIRECT_URI
            elif manual_input is not None:
                parsed = parse_authorization_input(manual_input)
                if parsed["state"] and parsed["state"] != verifier:
                    raise ValueError("OAuth state mismatch")
                code = parsed["code"]
                state = parsed["state"] or verifier

            if code is None:
                await manual_task
                if manual_error is not None:
                    raise manual_error
                if manual_input is not None:
                    parsed = parse_authorization_input(manual_input)
                    if parsed["state"] and parsed["state"] != verifier:
                        raise ValueError("OAuth state mismatch")
                    code = parsed["code"]
                    state = parsed["state"] or verifier
        else:
            result = await server.wait_for_code()
            if result is not None:
                code = result.code
                state = result.state
                redirect_uri_for_exchange = _REDIRECT_URI

        # Fallback to prompt
        if code is None:
            input_value = await on_prompt(
                {
                    "message": "Paste the authorization code or full redirect URL:",
                    "placeholder": _REDIRECT_URI,
                }
            )
            parsed = parse_authorization_input(input_value)
            if parsed["state"] and parsed["state"] != verifier:
                raise ValueError("OAuth state mismatch")
            code = parsed["code"]
            state = parsed["state"] or verifier

        if code is None:
            raise ValueError("Missing authorization code")
        if state is None:
            raise ValueError("Missing OAuth state")

        if on_progress:
            on_progress("Exchanging authorization code for tokens...")

        return await _exchange_authorization_code(code, state, verifier, redirect_uri_for_exchange)
    finally:
        server.close()


async def refresh_anthropic_token(refresh_token: str) -> OAuthCredentials:
    """Refresh Anthropic OAuth token."""
    try:
        response_body = await _post_json(
            _TOKEN_URL,
            {
                "grant_type": "refresh_token",
                "client_id": _CLIENT_ID,
                "refresh_token": refresh_token,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Anthropic token refresh request failed. url={_TOKEN_URL}; details={exc}") from exc

    import json

    try:
        data = json.loads(response_body)
    except Exception as exc:
        raise RuntimeError(
            f"Anthropic token refresh returned invalid JSON. url={_TOKEN_URL}; body={response_body}; details={exc}"
        ) from exc

    return OAuthCredentials(
        refresh=data["refresh_token"],
        access=data["access_token"],
        expires=time.time() * 1000 + data["expires_in"] * 1000 - 5 * 60 * 1000,
    )


class _AnthropicOAuthProvider:
    """Anthropic OAuth provider implementation."""

    @property
    def id(self) -> str:
        return "anthropic"

    @property
    def name(self) -> str:
        return "Anthropic (Claude Pro/Max)"

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
        return await login_anthropic(
            on_auth=on_auth,
            on_prompt=on_prompt,
            on_progress=on_progress,
            on_manual_code_input=on_manual_code_input,
        )

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return await refresh_anthropic_token(credentials.refresh)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access

    def modify_models(self, models: list[Any], credentials: OAuthCredentials) -> list[Any]:
        return models


anthropic_oauth_provider: OAuthProviderInterface = _AnthropicOAuthProvider()  # type: ignore[assignment]
