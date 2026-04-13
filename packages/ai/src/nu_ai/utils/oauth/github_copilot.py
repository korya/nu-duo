"""GitHub Copilot OAuth flow (device code flow).

Port of ``utils/oauth/github-copilot.ts``. Uses the device code flow
for authentication, with optional enterprise domain support.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

from nu_ai.utils.oauth.types import OAuthCredentials, OAuthProviderInterface

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode()

_COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

_INITIAL_POLL_MULTIPLIER = 1.2
_SLOW_DOWN_POLL_MULTIPLIER = 1.4


def normalize_domain(value: str) -> str | None:
    """Normalize a GitHub enterprise domain input."""
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        from urllib.parse import urlparse

        url_str = trimmed if "://" in trimmed else f"https://{trimmed}"
        parsed = urlparse(url_str)
        return parsed.hostname
    except Exception:
        return None


def _get_urls(domain: str) -> dict[str, str]:
    return {
        "device_code_url": f"https://{domain}/login/device/code",
        "access_token_url": f"https://{domain}/login/oauth/access_token",
        "copilot_token_url": f"https://api.{domain}/copilot_internal/v2/token",
    }


def _get_base_url_from_token(token: str) -> str | None:
    """Parse proxy-ep from a Copilot token and convert to API base URL."""
    import re

    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    proxy_host = match.group(1)
    api_host = re.sub(r"^proxy\.", "api.", proxy_host)
    return f"https://{api_host}"


def get_github_copilot_base_url(token: str | None = None, enterprise_domain: str | None = None) -> str:
    """Get the GitHub Copilot API base URL."""
    if token:
        url = _get_base_url_from_token(token)
        if url:
            return url
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return "https://api.individual.githubcopilot.com"


async def _fetch_json(url: str, **kwargs: Any) -> Any:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(**kwargs, url=url)
        response.raise_for_status()
        return response.json()


async def _start_device_flow(domain: str) -> dict[str, Any]:
    urls = _get_urls(domain)
    data = await _fetch_json(
        urls["device_code_url"],
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GitHubCopilotChat/0.35.0",
        },
        content=f"client_id={_CLIENT_ID}&scope=read%3Auser",
    )

    required = ["device_code", "user_code", "verification_uri", "interval", "expires_in"]
    for field in required:
        if field not in data:
            raise ValueError(f"Invalid device code response: missing {field}")

    return data


async def _poll_for_github_access_token(
    domain: str,
    device_code: str,
    interval_seconds: int,
    expires_in: int,
    cancelled: asyncio.Event | None = None,
) -> str:
    urls = _get_urls(domain)
    deadline = time.time() + expires_in
    interval_ms = max(1000, int(interval_seconds * 1000))
    interval_multiplier = _INITIAL_POLL_MULTIPLIER
    slow_down_count = 0

    while time.time() < deadline:
        if cancelled and cancelled.is_set():
            raise RuntimeError("Login cancelled")

        remaining_ms = (deadline - time.time()) * 1000
        wait_ms = min(int(interval_ms * interval_multiplier), remaining_ms)
        await asyncio.sleep(wait_ms / 1000)

        raw = await _fetch_json(
            urls["access_token_url"],
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            content=f"client_id={_CLIENT_ID}&device_code={device_code}&grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code",
        )

        if isinstance(raw, dict) and isinstance(raw.get("access_token"), str):
            return raw["access_token"]

        if isinstance(raw, dict) and isinstance(raw.get("error"), str):
            error = raw["error"]
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                slow_down_count += 1
                new_interval = raw.get("interval")
                if isinstance(new_interval, int) and new_interval > 0:
                    interval_ms = new_interval * 1000
                else:
                    interval_ms = max(1000, interval_ms + 5000)
                interval_multiplier = _SLOW_DOWN_POLL_MULTIPLIER
                continue
            desc = raw.get("error_description", "")
            suffix = f": {desc}" if desc else ""
            raise RuntimeError(f"Device flow failed: {error}{suffix}")

    if slow_down_count > 0:
        raise RuntimeError(
            "Device flow timed out after one or more slow_down responses. "
            "This is often caused by clock drift in WSL or VM environments. "
            "Please sync or restart the VM clock and try again."
        )
    raise RuntimeError("Device flow timed out")


async def refresh_github_copilot_token(
    refresh_token: str,
    enterprise_domain: str | None = None,
) -> OAuthCredentials:
    """Refresh GitHub Copilot token."""
    domain = enterprise_domain or "github.com"
    urls = _get_urls(domain)

    raw = await _fetch_json(
        urls["copilot_token_url"],
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {refresh_token}",
            **_COPILOT_HEADERS,
        },
    )

    if not isinstance(raw, dict):
        raise ValueError("Invalid Copilot token response")

    token = raw.get("token")
    expires_at = raw.get("expires_at")
    if not isinstance(token, str) or not isinstance(expires_at, (int, float)):
        raise ValueError("Invalid Copilot token response fields")

    extra: dict[str, object] = {}
    if enterprise_domain:
        extra["enterpriseUrl"] = enterprise_domain
    return OAuthCredentials(
        refresh=refresh_token,
        access=token,
        expires=expires_at * 1000 - 5 * 60 * 1000,
        extra_data=extra,
    )


async def _enable_model(token: str, model_id: str, enterprise_domain: str | None = None) -> bool:
    base_url = get_github_copilot_base_url(token, enterprise_domain)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base_url}/models/{model_id}/policy",
                json={"state": "enabled"},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    **_COPILOT_HEADERS,
                    "openai-intent": "chat-policy",
                    "x-interaction-type": "chat-policy",
                },
            )
            return response.is_success
    except Exception:
        return False


async def _enable_all_models(
    token: str,
    enterprise_domain: str | None = None,
) -> None:
    from nu_ai.models import get_models

    models = get_models("github-copilot")
    tasks = [_enable_model(token, m.id, enterprise_domain) for m in models]
    await asyncio.gather(*tasks, return_exceptions=True)


async def login_github_copilot(
    *,
    on_auth: Callable[[str, str | None], None],
    on_prompt: Callable[[dict[str, Any]], Awaitable[str]],
    on_progress: Callable[[str], None] | None = None,
    cancelled: asyncio.Event | None = None,
) -> OAuthCredentials:
    """Login with GitHub Copilot OAuth (device code flow)."""
    input_value = await on_prompt(
        {
            "message": "GitHub Enterprise URL/domain (blank for github.com)",
            "placeholder": "company.ghe.com",
            "allowEmpty": True,
        }
    )

    if cancelled and cancelled.is_set():
        raise RuntimeError("Login cancelled")

    trimmed = input_value.strip()
    enterprise_domain = normalize_domain(input_value)
    if trimmed and not enterprise_domain:
        raise ValueError("Invalid GitHub Enterprise URL/domain")
    domain = enterprise_domain or "github.com"

    device = await _start_device_flow(domain)
    on_auth(device["verification_uri"], f"Enter code: {device['user_code']}")

    github_access_token = await _poll_for_github_access_token(
        domain,
        device["device_code"],
        device["interval"],
        device["expires_in"],
        cancelled=cancelled,
    )

    credentials = await refresh_github_copilot_token(github_access_token, enterprise_domain)

    if on_progress:
        on_progress("Enabling models...")
    await _enable_all_models(credentials.access, enterprise_domain)

    return credentials


class _GitHubCopilotOAuthProvider:
    """GitHub Copilot OAuth provider implementation."""

    @property
    def id(self) -> str:
        return "github-copilot"

    @property
    def name(self) -> str:
        return "GitHub Copilot"

    @property
    def uses_callback_server(self) -> bool:
        return False

    async def login(
        self,
        *,
        on_auth: Any,
        on_prompt: Any,
        on_progress: Any | None = None,
        on_manual_code_input: Any | None = None,
    ) -> OAuthCredentials:
        return await login_github_copilot(
            on_auth=lambda url, instructions=None: on_auth({"url": url, "instructions": instructions}),
            on_prompt=on_prompt,
            on_progress=on_progress,
        )

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        enterprise_url = credentials.extra_data.get("enterpriseUrl")
        return await refresh_github_copilot_token(credentials.refresh, str(enterprise_url) if enterprise_url else None)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access

    def modify_models(self, models: list[Any], credentials: OAuthCredentials) -> list[Any]:
        enterprise_url = credentials.extra_data.get("enterpriseUrl")
        domain = normalize_domain(str(enterprise_url)) if enterprise_url else None
        base_url = get_github_copilot_base_url(credentials.access, domain)
        result = []
        for m in models:
            if getattr(m, "provider", None) == "github-copilot":
                # Create a copy with updated base_url
                m_dict = m.model_dump() if hasattr(m, "model_dump") else dict(m)
                m_dict["base_url"] = base_url
                result.append(type(m)(**m_dict))
            else:
                result.append(m)
        return result


github_copilot_oauth_provider: OAuthProviderInterface = _GitHubCopilotOAuthProvider()  # type: ignore[assignment]
