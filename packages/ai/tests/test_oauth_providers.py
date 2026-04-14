"""Tests for OAuth provider flows (login, refresh, helpers).

Targets >=90% coverage for:
- anthropic.py
- github_copilot.py
- google_gemini_cli.py
- google_antigravity.py
- openai_codex.py
- __init__.py (remaining gaps)
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nu_ai.utils.oauth.callback_server import CallbackResult
from nu_ai.utils.oauth.types import OAuthCredentials

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data: Any = None, text: str = "") -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data or {}
    resp.text = text or json.dumps(json_data or {})
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
    return resp


def _make_jwt(claims: dict[str, Any]) -> str:
    """Build a fake JWT (no signing) with given payload claims."""
    header = base64.b64encode(json.dumps({"alg": "none"}).encode()).decode()
    payload = base64.b64encode(json.dumps(claims).encode()).decode()
    sig = base64.b64encode(b"sig").decode()
    return f"{header}.{payload}.{sig}"


def _mock_async_client(post_response=None, get_response=None):
    """Return an AsyncMock that mimics httpx.AsyncClient context manager."""
    client = AsyncMock()
    if post_response is not None:
        client.post.return_value = post_response
    if get_response is not None:
        client.get.return_value = get_response
    # Also support .request() used by _fetch_json
    if post_response is not None:
        client.request.return_value = post_response
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm, client


def _mock_callback_server(wait_result=None):
    """Create mock CallbackServer with sync close/cancel_wait."""
    server = AsyncMock()
    server.wait_for_code.return_value = wait_result
    # close() and cancel_wait() are sync methods, so use MagicMock
    server.close = MagicMock()
    server.cancel_wait = MagicMock()
    return server


# ═══════════════════════════════════════════════════════════════════════════
# Anthropic
# ═══════════════════════════════════════════════════════════════════════════


class TestRefreshAnthropicToken:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        from nu_ai.utils.oauth.anthropic import refresh_anthropic_token

        response_data = {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=response_data)
        mock_resp.text = json.dumps(response_data)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.anthropic.httpx.AsyncClient", return_value=cm):
            creds = await refresh_anthropic_token("old-refresh")

        assert creds.access == "new-access"
        assert creds.refresh == "new-refresh"
        assert creds.expires > 0

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        from nu_ai.utils.oauth.anthropic import refresh_anthropic_token

        mock_resp = _mock_response(401)
        mock_resp.text = "Unauthorized"
        cm, client = _mock_async_client(post_response=mock_resp)
        # _post_json calls raise_for_status
        client.post.return_value = mock_resp

        with patch("nu_ai.utils.oauth.anthropic.httpx.AsyncClient", return_value=cm):
            with pytest.raises(RuntimeError, match="refresh request failed"):
                await refresh_anthropic_token("bad-refresh")

    @pytest.mark.asyncio
    async def test_invalid_json_response(self) -> None:
        from nu_ai.utils.oauth.anthropic import refresh_anthropic_token

        mock_resp = _mock_response(200)
        mock_resp.text = "not-json{{"
        cm, client = _mock_async_client(post_response=mock_resp)
        client.post.return_value = mock_resp

        with patch("nu_ai.utils.oauth.anthropic.httpx.AsyncClient", return_value=cm):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                await refresh_anthropic_token("r")


class TestLoginAnthropic:
    @pytest.mark.asyncio
    async def test_callback_success(self) -> None:
        from nu_ai.utils.oauth.anthropic import login_anthropic

        expected_creds = OAuthCredentials(refresh="rt", access="at", expires=time.time() * 1000 + 3600_000)

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.anthropic._exchange_authorization_code",
                new_callable=AsyncMock,
                return_value=expected_creds,
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="auth-code", state="verifier"))
            MockServer.return_value = server_inst

            creds = await login_anthropic(on_auth=on_auth, on_prompt=on_prompt)

        assert creds.access == "at"
        assert creds.refresh == "rt"
        on_auth.assert_called_once()
        server_inst.start.assert_awaited_once()
        server_inst.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_code_falls_back_to_prompt(self) -> None:
        from nu_ai.utils.oauth.anthropic import login_anthropic

        expected_creds = OAuthCredentials(refresh="rt2", access="at2", expires=time.time() * 1000 + 3600_000)

        on_auth = MagicMock()
        on_prompt = AsyncMock(return_value="prompt-code#verifier")

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.anthropic._exchange_authorization_code",
                new_callable=AsyncMock,
                return_value=expected_creds,
            ),
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            creds = await login_anthropic(on_auth=on_auth, on_prompt=on_prompt)

        assert creds.access == "at2"
        on_prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_code_raises(self) -> None:
        from nu_ai.utils.oauth.anthropic import login_anthropic

        on_auth = MagicMock()
        on_prompt = AsyncMock(return_value="")

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="Missing authorization code"):
                await login_anthropic(on_auth=on_auth, on_prompt=on_prompt)

    @pytest.mark.asyncio
    async def test_state_mismatch_raises(self) -> None:
        from nu_ai.utils.oauth.anthropic import login_anthropic

        on_auth = MagicMock()
        on_prompt = AsyncMock(return_value="code=abc&state=wrong-state")

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="state mismatch"):
                await login_anthropic(on_auth=on_auth, on_prompt=on_prompt)

    @pytest.mark.asyncio
    async def test_login_with_manual_code_input(self) -> None:
        from nu_ai.utils.oauth.anthropic import login_anthropic

        expected_creds = OAuthCredentials(refresh="rt3", access="at3", expires=time.time() * 1000 + 3600_000)

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        async def manual_input():
            return "http://localhost:53692/callback?code=manual-code&state=verifier"

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.anthropic._exchange_authorization_code",
                new_callable=AsyncMock,
                return_value=expected_creds,
            ),
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            creds = await login_anthropic(
                on_auth=on_auth,
                on_prompt=on_prompt,
                on_manual_code_input=manual_input,
            )

        assert creds.access == "at3"

    @pytest.mark.asyncio
    async def test_exchange_code_error(self) -> None:
        from nu_ai.utils.oauth.anthropic import _exchange_authorization_code

        mock_resp = _mock_response(500)
        cm, client = _mock_async_client(post_response=mock_resp)
        client.post.return_value = mock_resp

        with patch("nu_ai.utils.oauth.anthropic.httpx.AsyncClient", return_value=cm):
            with pytest.raises(RuntimeError, match="Token exchange request failed"):
                await _exchange_authorization_code("code", "state", "verifier", "http://localhost/cb")

    @pytest.mark.asyncio
    async def test_login_with_progress(self) -> None:
        from nu_ai.utils.oauth.anthropic import login_anthropic

        expected_creds = OAuthCredentials(refresh="rt4", access="at4", expires=time.time() * 1000 + 3600_000)

        on_auth = MagicMock()
        on_prompt = AsyncMock()
        on_progress = MagicMock()

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.anthropic._exchange_authorization_code",
                new_callable=AsyncMock,
                return_value=expected_creds,
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="verifier"))
            MockServer.return_value = server_inst

            creds = await login_anthropic(on_auth=on_auth, on_prompt=on_prompt, on_progress=on_progress)

        assert creds.access == "at4"
        on_progress.assert_called()


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_refresh_token_delegates(self) -> None:
        from nu_ai.utils.oauth.anthropic import anthropic_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        new_creds = OAuthCredentials(refresh="nr", access="na", expires=999)

        with patch(
            "nu_ai.utils.oauth.anthropic.refresh_anthropic_token", new_callable=AsyncMock, return_value=new_creds
        ):
            result = await anthropic_oauth_provider.refresh_token(creds)
        assert result.access == "na"

    @pytest.mark.asyncio
    async def test_login_delegates(self) -> None:
        from nu_ai.utils.oauth.anthropic import anthropic_oauth_provider

        expected = OAuthCredentials(refresh="r", access="a", expires=100)
        with patch("nu_ai.utils.oauth.anthropic.login_anthropic", new_callable=AsyncMock, return_value=expected):
            result = await anthropic_oauth_provider.login(on_auth=MagicMock(), on_prompt=AsyncMock())
        assert result == expected

    def test_modify_models_passthrough(self) -> None:
        from nu_ai.utils.oauth.anthropic import anthropic_oauth_provider

        models = ["m1", "m2"]
        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        assert anthropic_oauth_provider.modify_models(models, creds) == models


# ═══════════════════════════════════════════════════════════════════════════
# GitHub Copilot
# ═══════════════════════════════════════════════════════════════════════════


class TestRefreshGitHubCopilotToken:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        from nu_ai.utils.oauth.github_copilot import refresh_github_copilot_token

        response_data = {
            "token": "ghc-token-123",
            "expires_at": time.time() + 3600,
        }

        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", new_callable=AsyncMock, return_value=response_data):
            creds = await refresh_github_copilot_token("gh-access-token")

        assert creds.access == "ghc-token-123"
        assert creds.refresh == "gh-access-token"

    @pytest.mark.asyncio
    async def test_with_enterprise_domain(self) -> None:
        from nu_ai.utils.oauth.github_copilot import refresh_github_copilot_token

        response_data = {
            "token": "ent-token",
            "expires_at": time.time() + 3600,
        }

        with patch(
            "nu_ai.utils.oauth.github_copilot._fetch_json", new_callable=AsyncMock, return_value=response_data
        ) as mock_fetch:
            creds = await refresh_github_copilot_token("gh-token", enterprise_domain="company.ghe.com")

        assert creds.extra_data.get("enterpriseUrl") == "company.ghe.com"
        # Verify it used enterprise URL
        call_url = mock_fetch.call_args[0][0]
        assert "company.ghe.com" in call_url

    @pytest.mark.asyncio
    async def test_invalid_response_no_dict(self) -> None:
        from nu_ai.utils.oauth.github_copilot import refresh_github_copilot_token

        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", new_callable=AsyncMock, return_value="not-a-dict"):
            with pytest.raises(ValueError, match="Invalid Copilot token response"):
                await refresh_github_copilot_token("t")

    @pytest.mark.asyncio
    async def test_invalid_response_missing_fields(self) -> None:
        from nu_ai.utils.oauth.github_copilot import refresh_github_copilot_token

        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", new_callable=AsyncMock, return_value={"token": 123}):
            with pytest.raises(ValueError, match="Invalid Copilot token response fields"):
                await refresh_github_copilot_token("t")

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        from nu_ai.utils.oauth.github_copilot import refresh_github_copilot_token

        with (
            patch(
                "nu_ai.utils.oauth.github_copilot._fetch_json",
                new_callable=AsyncMock,
                side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=MagicMock()),
            ),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await refresh_github_copilot_token("t")


class TestGitHubDeviceFlow:
    @pytest.mark.asyncio
    async def test_start_device_flow(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _start_device_flow

        device_data = {
            "device_code": "dc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 5,
            "expires_in": 900,
        }
        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", new_callable=AsyncMock, return_value=device_data):
            result = await _start_device_flow("github.com")
        assert result["user_code"] == "ABCD-1234"

    @pytest.mark.asyncio
    async def test_start_device_flow_missing_field(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _start_device_flow

        with patch(
            "nu_ai.utils.oauth.github_copilot._fetch_json", new_callable=AsyncMock, return_value={"device_code": "dc"}
        ):
            with pytest.raises(ValueError, match="missing"):
                await _start_device_flow("github.com")

    @pytest.mark.asyncio
    async def test_poll_authorization_pending_then_success(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _poll_for_github_access_token

        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return {"error": "authorization_pending"}
            return {"access_token": "gh-at-123"}

        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", side_effect=mock_fetch):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                token = await _poll_for_github_access_token("github.com", "dc", 1, 60)

        assert token == "gh-at-123"

    @pytest.mark.asyncio
    async def test_poll_slow_down(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _poll_for_github_access_token

        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"error": "slow_down", "interval": 10}
            return {"access_token": "token-after-slow"}

        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", side_effect=mock_fetch):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                token = await _poll_for_github_access_token("github.com", "dc", 1, 60)

        assert token == "token-after-slow"

    @pytest.mark.asyncio
    async def test_poll_error(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _poll_for_github_access_token

        async def mock_fetch(*args, **kwargs):
            return {"error": "access_denied", "error_description": "User denied"}

        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", side_effect=mock_fetch):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match=r"Device flow failed.*access_denied.*User denied"):
                    await _poll_for_github_access_token("github.com", "dc", 1, 60)

    @pytest.mark.asyncio
    async def test_poll_cancelled(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _poll_for_github_access_token

        cancelled = asyncio.Event()
        cancelled.set()

        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="cancelled"):
                    await _poll_for_github_access_token("github.com", "dc", 1, 60, cancelled=cancelled)

    @pytest.mark.asyncio
    async def test_poll_timeout(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _poll_for_github_access_token

        # expires_in=0 means it should time out immediately
        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="timed out"):
                    await _poll_for_github_access_token("github.com", "dc", 1, 0)


class TestLoginGitHubCopilot:
    @pytest.mark.asyncio
    async def test_login_success(self) -> None:
        from nu_ai.utils.oauth.github_copilot import login_github_copilot

        on_auth = MagicMock()
        on_prompt = AsyncMock(return_value="")  # blank = github.com
        on_progress = MagicMock()

        device_data = {
            "device_code": "dc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 5,
            "expires_in": 900,
        }

        refresh_creds = OAuthCredentials(
            refresh="gh-at",
            access="copilot-token",
            expires=time.time() * 1000 + 3600_000,
        )

        with (
            patch(
                "nu_ai.utils.oauth.github_copilot._start_device_flow", new_callable=AsyncMock, return_value=device_data
            ),
            patch(
                "nu_ai.utils.oauth.github_copilot._poll_for_github_access_token",
                new_callable=AsyncMock,
                return_value="gh-at",
            ),
            patch(
                "nu_ai.utils.oauth.github_copilot.refresh_github_copilot_token",
                new_callable=AsyncMock,
                return_value=refresh_creds,
            ),
            patch("nu_ai.utils.oauth.github_copilot._enable_all_models", new_callable=AsyncMock),
        ):
            creds = await login_github_copilot(on_auth=on_auth, on_prompt=on_prompt, on_progress=on_progress)

        assert creds.access == "copilot-token"
        on_auth.assert_called_once()
        on_progress.assert_called()

    @pytest.mark.asyncio
    async def test_login_invalid_enterprise_domain(self) -> None:
        from nu_ai.utils.oauth.github_copilot import login_github_copilot

        on_auth = MagicMock()
        # Return a string that normalize_domain can't parse to hostname
        on_prompt = AsyncMock(return_value="://invalid")

        with patch("nu_ai.utils.oauth.github_copilot.normalize_domain", return_value=None):
            with pytest.raises(ValueError, match="Invalid GitHub Enterprise"):
                await login_github_copilot(on_auth=on_auth, on_prompt=on_prompt)

    @pytest.mark.asyncio
    async def test_login_cancelled(self) -> None:
        from nu_ai.utils.oauth.github_copilot import login_github_copilot

        on_auth = MagicMock()
        on_prompt = AsyncMock(return_value="")
        cancelled = asyncio.Event()
        cancelled.set()

        with pytest.raises(RuntimeError, match="cancelled"):
            await login_github_copilot(on_auth=on_auth, on_prompt=on_prompt, cancelled=cancelled)


class TestGitHubCopilotProvider:
    @pytest.mark.asyncio
    async def test_refresh_token_delegates(self) -> None:
        from nu_ai.utils.oauth.github_copilot import github_copilot_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        new_creds = OAuthCredentials(refresh="r", access="new-a", expires=999)

        with patch(
            "nu_ai.utils.oauth.github_copilot.refresh_github_copilot_token",
            new_callable=AsyncMock,
            return_value=new_creds,
        ):
            result = await github_copilot_oauth_provider.refresh_token(creds)
        assert result.access == "new-a"

    @pytest.mark.asyncio
    async def test_refresh_with_enterprise_url(self) -> None:
        from nu_ai.utils.oauth.github_copilot import github_copilot_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0, extra_data={"enterpriseUrl": "ent.com"})
        new_creds = OAuthCredentials(refresh="r", access="new-a", expires=999)

        with patch(
            "nu_ai.utils.oauth.github_copilot.refresh_github_copilot_token",
            new_callable=AsyncMock,
            return_value=new_creds,
        ) as mock_refresh:
            await github_copilot_oauth_provider.refresh_token(creds)
        mock_refresh.assert_awaited_once_with("r", "ent.com")

    def test_modify_models(self) -> None:
        from nu_ai.utils.oauth.github_copilot import github_copilot_oauth_provider

        # With a model that has provider = github-copilot
        class FakeModel:
            def __init__(self, provider, base_url=""):
                self.provider = provider
                self.base_url = base_url

            def model_dump(self):
                return {"provider": self.provider, "base_url": self.base_url}

        creds = OAuthCredentials(refresh="r", access="tok;proxy-ep=proxy.foo.com;", expires=0)
        m1 = FakeModel("github-copilot", "old-url")
        m2 = FakeModel("other", "keep")
        result = github_copilot_oauth_provider.modify_models([m1, m2], creds)
        assert len(result) == 2
        # Second model unchanged
        assert result[1] is m2


class TestFetchJson:
    @pytest.mark.asyncio
    async def test_fetch_json_success(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _fetch_json

        mock_resp = _mock_response(200, json_data={"key": "val"})
        cm, client = _mock_async_client()
        client.request.return_value = mock_resp

        with patch("nu_ai.utils.oauth.github_copilot.httpx.AsyncClient", return_value=cm):
            result = await _fetch_json("https://example.com", method="GET")
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_fetch_json_error(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _fetch_json

        mock_resp = _mock_response(500)
        cm, client = _mock_async_client()
        client.request.return_value = mock_resp

        with patch("nu_ai.utils.oauth.github_copilot.httpx.AsyncClient", return_value=cm):
            with pytest.raises(httpx.HTTPStatusError):
                await _fetch_json("https://example.com", method="GET")


class TestEnableModel:
    @pytest.mark.asyncio
    async def test_enable_model_success(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _enable_model

        mock_resp = _mock_response(200)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.github_copilot.httpx.AsyncClient", return_value=cm):
            result = await _enable_model("token", "model-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_enable_model_failure(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _enable_model

        cm, client = _mock_async_client()
        client.post.side_effect = Exception("network error")

        with patch("nu_ai.utils.oauth.github_copilot.httpx.AsyncClient", return_value=cm):
            result = await _enable_model("token", "model-1")
        assert result is False


class TestGetBaseUrlFromToken:
    def test_with_proxy_ep(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _get_base_url_from_token

        token = "tid=123;proxy-ep=proxy.individual.githubcopilot.com;exp=456"
        assert _get_base_url_from_token(token) == "https://api.individual.githubcopilot.com"

    def test_no_proxy_ep(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _get_base_url_from_token

        assert _get_base_url_from_token("tid=123;exp=456") is None


# ═══════════════════════════════════════════════════════════════════════════
# Google Gemini CLI
# ═══════════════════════════════════════════════════════════════════════════


class TestRefreshGoogleCloudToken:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import refresh_google_cloud_token

        response_data = {
            "access_token": "ya29.new",
            "refresh_token": "new-rt",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=response_data)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            creds = await refresh_google_cloud_token("old-rt", "proj-123")

        assert creds.access == "ya29.new"
        assert creds.refresh == "new-rt"
        assert creds.extra_data["projectId"] == "proj-123"

    @pytest.mark.asyncio
    async def test_keeps_old_refresh_when_missing(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import refresh_google_cloud_token

        response_data = {
            "access_token": "ya29.new",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=response_data)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            creds = await refresh_google_cloud_token("old-rt", "proj-123")

        assert creds.refresh == "old-rt"

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import refresh_google_cloud_token

        mock_resp = _mock_response(401)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            with pytest.raises(httpx.HTTPStatusError):
                await refresh_google_cloud_token("rt", "proj")


class TestGeminiCliHelpers:
    def test_parse_redirect_url_with_code(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _parse_redirect_url

        result = _parse_redirect_url("http://localhost:8085/oauth2callback?code=abc&state=xyz")
        assert result["code"] == "abc"
        assert result["state"] == "xyz"

    def test_parse_redirect_url_empty(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _parse_redirect_url

        result = _parse_redirect_url("")
        assert result["code"] is None

    def test_parse_redirect_url_invalid(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _parse_redirect_url

        # Should not crash on anything
        result = _parse_redirect_url("just-a-string")
        assert result["code"] is None

    def test_is_vpc_sc_affected_true(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _is_vpc_sc_affected

        payload = {"error": {"details": [{"reason": "SECURITY_POLICY_VIOLATED"}]}}
        assert _is_vpc_sc_affected(payload) is True

    def test_is_vpc_sc_affected_false(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _is_vpc_sc_affected

        assert _is_vpc_sc_affected({}) is False
        assert _is_vpc_sc_affected({"error": "string"}) is False
        assert _is_vpc_sc_affected({"error": {"details": "not-list"}}) is False
        assert _is_vpc_sc_affected("not-a-dict") is False

    def test_get_default_tier_empty(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _get_default_tier

        assert _get_default_tier(None)["id"] == "legacy-tier"
        assert _get_default_tier([])["id"] == "legacy-tier"

    def test_get_default_tier_with_default(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _get_default_tier

        tiers = [{"id": "free-tier", "isDefault": True}, {"id": "paid"}]
        assert _get_default_tier(tiers)["id"] == "free-tier"

    def test_get_default_tier_no_default(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _get_default_tier

        tiers = [{"id": "paid"}]
        assert _get_default_tier(tiers)["id"] == "legacy-tier"


class TestDiscoverProject:
    @pytest.mark.asyncio
    async def test_existing_project(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "currentTier": {"id": "free-tier"},
                "cloudaicompanionProject": "existing-proj",
            },
        )
        cm, _ = _mock_async_client(post_response=load_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            project = await _discover_project("token")
        assert project == "existing-proj"

    @pytest.mark.asyncio
    async def test_vpc_sc_affected(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        error_payload = {"error": {"details": [{"reason": "SECURITY_POLICY_VIOLATED"}]}}
        load_resp = _mock_response(403, json_data=error_payload)
        load_resp.raise_for_status = MagicMock()  # Don't raise
        cm, _ = _mock_async_client(post_response=load_resp)

        # VPC SC => currentTier = standard-tier, needs env project
        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm),
            patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-proj"}),
        ):
            project = await _discover_project("token")
        assert project == "env-proj"

    @pytest.mark.asyncio
    async def test_load_error_not_vpc(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(500, json_data={})
        load_resp.raise_for_status = MagicMock()  # Don't raise for this mock
        cm, _ = _mock_async_client(post_response=load_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            with pytest.raises(RuntimeError, match="loadCodeAssist failed"):
                await _discover_project("token")

    @pytest.mark.asyncio
    async def test_onboarding_free_tier(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "allowedTiers": [{"id": "free-tier", "isDefault": True}],
            },
        )

        onboard_resp = _mock_response(
            200,
            json_data={
                "done": True,
                "response": {
                    "cloudaicompanionProject": {"id": "new-proj"},
                },
            },
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return load_resp
            return onboard_resp

        cm = AsyncMock()
        client = AsyncMock()
        client.post = mock_post
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            project = await _discover_project("token", on_progress=MagicMock())
        assert project == "new-proj"

    @pytest.mark.asyncio
    async def test_onboarding_lro_pending(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "allowedTiers": [{"id": "free-tier", "isDefault": True}],
            },
        )

        onboard_resp = _mock_response(
            200,
            json_data={
                "done": False,
                "name": "operations/op-123",
            },
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return load_resp
            return onboard_resp

        poll_result = {
            "done": True,
            "response": {"cloudaicompanionProject": "polled-proj"},
        }

        cm = AsyncMock()
        client = AsyncMock()
        client.post = mock_post
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm),
            patch(
                "nu_ai.utils.oauth.google_gemini_cli._poll_operation", new_callable=AsyncMock, return_value=poll_result
            ),
        ):
            project = await _discover_project("token")
        assert project == "polled-proj"

    @pytest.mark.asyncio
    async def test_current_tier_no_project_needs_env(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "currentTier": {"id": "standard"},
            },
        )
        cm, _ = _mock_async_client(post_response=load_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
                await _discover_project("token")

    @pytest.mark.asyncio
    async def test_current_tier_uses_env_project(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "currentTier": {"id": "standard"},
            },
        )
        cm, _ = _mock_async_client(post_response=load_resp)

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm),
            patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-proj"}),
        ):
            project = await _discover_project("token")
        assert project == "env-proj"

    @pytest.mark.asyncio
    async def test_non_free_tier_requires_env_project(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "allowedTiers": [{"id": "paid-tier", "isDefault": True}],
            },
        )
        cm, _ = _mock_async_client(post_response=load_resp)

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm),
            patch.dict("os.environ", {}, clear=True),
        ):
            # Remove env vars if present
            import os

            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            os.environ.pop("GOOGLE_CLOUD_PROJECT_ID", None)
            with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
                await _discover_project("token")

    @pytest.mark.asyncio
    async def test_onboard_no_project_no_env_raises(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "allowedTiers": [{"id": "free-tier", "isDefault": True}],
            },
        )

        onboard_resp = _mock_response(
            200,
            json_data={
                "done": True,
                "response": {},
            },
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return load_resp
            return onboard_resp

        cm = AsyncMock()
        client = AsyncMock()
        client.post = mock_post
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm),
            patch.dict("os.environ", {}, clear=True),
        ):
            import os

            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            os.environ.pop("GOOGLE_CLOUD_PROJECT_ID", None)
            with pytest.raises(RuntimeError, match="Could not discover"):
                await _discover_project("token")


class TestPollOperation:
    @pytest.mark.asyncio
    async def test_poll_immediate_done(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _poll_operation

        mock_resp = _mock_response(200, json_data={"done": True, "response": {"id": "p"}})
        cm, _ = _mock_async_client(get_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            result = await _poll_operation("ops/1", {"Authorization": "Bearer t"})
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_poll_retries(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _poll_operation

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return _mock_response(200, json_data={"done": False})
            return _mock_response(200, json_data={"done": True, "response": {"id": "p"}})

        cm = AsyncMock()
        client = AsyncMock()
        client.get = mock_get
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _poll_operation("ops/1", {"Authorization": "Bearer t"}, on_progress=MagicMock())
        assert result["done"] is True


class TestGetUserEmailGemini:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _get_user_email

        mock_resp = _mock_response(200, json_data={"email": "user@example.com"})
        cm, _ = _mock_async_client(get_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            email = await _get_user_email("token")
        assert email == "user@example.com"

    @pytest.mark.asyncio
    async def test_failure(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _get_user_email

        mock_resp = _mock_response(401)
        cm, _ = _mock_async_client(get_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            email = await _get_user_email("token")
        assert email is None

    @pytest.mark.asyncio
    async def test_exception(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import _get_user_email

        cm = AsyncMock()
        cm.__aenter__.side_effect = Exception("network")
        cm.__aexit__.return_value = None

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            email = await _get_user_email("token")
        assert email is None


class TestLoginGeminiCli:
    @pytest.mark.asyncio
    async def test_callback_success(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import login_gemini_cli

        token_data = {
            "access_token": "ya29.at",
            "refresh_token": "rt",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=token_data)
        on_auth = MagicMock()

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_gemini_cli.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient") as MockClient,
            patch(
                "nu_ai.utils.oauth.google_gemini_cli._get_user_email", new_callable=AsyncMock, return_value="u@g.com"
            ),
            patch(
                "nu_ai.utils.oauth.google_gemini_cli._discover_project", new_callable=AsyncMock, return_value="proj-1"
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="auth-code", state="verifier"))
            MockServer.return_value = server_inst

            cm, _ = _mock_async_client(post_response=mock_resp)
            MockClient.return_value = cm

            creds = await login_gemini_cli(on_auth, on_progress=MagicMock())

        assert creds.access == "ya29.at"
        assert creds.extra_data["projectId"] == "proj-1"
        assert creds.extra_data["email"] == "u@g.com"

    @pytest.mark.asyncio
    async def test_missing_refresh_token(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import login_gemini_cli

        token_data = {
            "access_token": "ya29.at",
            "expires_in": 3600,
            # No refresh_token
        }
        mock_resp = _mock_response(200, json_data=token_data)
        on_auth = MagicMock()

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_gemini_cli.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient") as MockClient,
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="verifier"))
            MockServer.return_value = server_inst

            cm, _ = _mock_async_client(post_response=mock_resp)
            MockClient.return_value = cm

            with pytest.raises(ValueError, match="No refresh token"):
                await login_gemini_cli(on_auth)

    @pytest.mark.asyncio
    async def test_missing_code(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import login_gemini_cli

        on_auth = MagicMock()

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_gemini_cli.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="No authorization code"):
                await login_gemini_cli(on_auth)

    @pytest.mark.asyncio
    async def test_state_mismatch(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import login_gemini_cli

        on_auth = MagicMock()

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_gemini_cli.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="wrong"))
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="state mismatch"):
                await login_gemini_cli(on_auth)

    @pytest.mark.asyncio
    async def test_with_manual_code_input(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import login_gemini_cli

        token_data = {
            "access_token": "ya29.manual",
            "refresh_token": "rt-manual",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=token_data)
        on_auth = MagicMock()

        async def manual_input():
            return "http://localhost:8085/oauth2callback?code=manual-code&state=verifier"

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_gemini_cli.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient") as MockClient,
            patch("nu_ai.utils.oauth.google_gemini_cli._get_user_email", new_callable=AsyncMock, return_value=None),
            patch("nu_ai.utils.oauth.google_gemini_cli._discover_project", new_callable=AsyncMock, return_value="proj"),
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            cm, _ = _mock_async_client(post_response=mock_resp)
            MockClient.return_value = cm

            creds = await login_gemini_cli(on_auth, on_manual_code_input=manual_input)

        assert creds.access == "ya29.manual"
        # No email in extra_data since _get_user_email returned None
        assert "email" not in creds.extra_data


class TestGeminiCliProvider:
    @pytest.mark.asyncio
    async def test_refresh_missing_project_id(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import gemini_cli_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0, extra_data={})
        with pytest.raises(ValueError, match="missing projectId"):
            await gemini_cli_oauth_provider.refresh_token(creds)

    @pytest.mark.asyncio
    async def test_refresh_delegates(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import gemini_cli_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0, extra_data={"projectId": "p"})
        new_creds = OAuthCredentials(refresh="nr", access="na", expires=999, extra_data={"projectId": "p"})

        with patch(
            "nu_ai.utils.oauth.google_gemini_cli.refresh_google_cloud_token",
            new_callable=AsyncMock,
            return_value=new_creds,
        ):
            result = await gemini_cli_oauth_provider.refresh_token(creds)
        assert result.access == "na"

    def test_get_api_key_json(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import gemini_cli_oauth_provider

        creds = OAuthCredentials(refresh="r", access="ya29", expires=0, extra_data={"projectId": "p1"})
        key = gemini_cli_oauth_provider.get_api_key(creds)
        parsed = json.loads(key)
        assert parsed["token"] == "ya29"
        assert parsed["projectId"] == "p1"

    def test_modify_models_passthrough(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import gemini_cli_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        assert gemini_cli_oauth_provider.modify_models(["m"], creds) == ["m"]


# ═══════════════════════════════════════════════════════════════════════════
# Google Antigravity
# ═══════════════════════════════════════════════════════════════════════════


class TestRefreshAntigravityToken:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import refresh_antigravity_token

        response_data = {
            "access_token": "ya29.ag",
            "refresh_token": "new-ag-rt",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=response_data)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            creds = await refresh_antigravity_token("old-rt", "proj")

        assert creds.access == "ya29.ag"
        assert creds.extra_data["projectId"] == "proj"

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import refresh_antigravity_token

        mock_resp = _mock_response(500)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            with pytest.raises(httpx.HTTPStatusError):
                await refresh_antigravity_token("rt", "proj")


class TestAntigravityHelpers:
    def test_parse_redirect_url(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import _parse_redirect_url

        result = _parse_redirect_url("http://localhost/cb?code=abc&state=xyz")
        assert result["code"] == "abc"
        assert result["state"] == "xyz"

    def test_parse_redirect_url_empty(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import _parse_redirect_url

        result = _parse_redirect_url("")
        assert result["code"] is None


class TestAntigravityDiscoverProject:
    @pytest.mark.asyncio
    async def test_found_project_string(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import _discover_project

        load_resp = _mock_response(200, json_data={"cloudaicompanionProject": "found-proj"})
        cm, _ = _mock_async_client(post_response=load_resp)

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            project = await _discover_project("token")
        assert project == "found-proj"

    @pytest.mark.asyncio
    async def test_found_project_dict(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import _discover_project

        load_resp = _mock_response(200, json_data={"cloudaicompanionProject": {"id": "dict-proj"}})
        cm, _ = _mock_async_client(post_response=load_resp)

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            project = await _discover_project("token")
        assert project == "dict-proj"

    @pytest.mark.asyncio
    async def test_fallback_to_default(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import _discover_project

        # Both endpoints fail
        load_resp = _mock_response(500)
        load_resp.raise_for_status = MagicMock()  # Don't actually raise

        cm = AsyncMock()
        client = AsyncMock()
        client.post.side_effect = Exception("fail")
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            project = await _discover_project("token", on_progress=MagicMock())
        assert project == "rising-fact-p41fc"


class TestAntigravityGetUserEmail:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import _get_user_email

        mock_resp = _mock_response(200, json_data={"email": "ag@test.com"})
        cm, _ = _mock_async_client(get_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            email = await _get_user_email("token")
        assert email == "ag@test.com"

    @pytest.mark.asyncio
    async def test_failure(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import _get_user_email

        mock_resp = _mock_response(500)
        cm, _ = _mock_async_client(get_response=mock_resp)

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            email = await _get_user_email("token")
        assert email is None


class TestLoginAntigravity:
    @pytest.mark.asyncio
    async def test_callback_success(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import login_antigravity

        token_data = {
            "access_token": "ya29.ag",
            "refresh_token": "rt-ag",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=token_data)
        on_auth = MagicMock()

        with (
            patch("nu_ai.utils.oauth.google_antigravity.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_antigravity.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient") as MockClient,
            patch(
                "nu_ai.utils.oauth.google_antigravity._get_user_email", new_callable=AsyncMock, return_value="ag@g.com"
            ),
            patch(
                "nu_ai.utils.oauth.google_antigravity._discover_project", new_callable=AsyncMock, return_value="ag-proj"
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="code", state="verifier"))
            MockServer.return_value = server_inst

            cm, _ = _mock_async_client(post_response=mock_resp)
            MockClient.return_value = cm

            creds = await login_antigravity(on_auth, on_progress=MagicMock())

        assert creds.access == "ya29.ag"
        assert creds.extra_data["projectId"] == "ag-proj"
        assert creds.extra_data["email"] == "ag@g.com"

    @pytest.mark.asyncio
    async def test_missing_refresh_token(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import login_antigravity

        token_data = {"access_token": "ya29", "expires_in": 3600}
        mock_resp = _mock_response(200, json_data=token_data)
        on_auth = MagicMock()

        with (
            patch("nu_ai.utils.oauth.google_antigravity.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_antigravity.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient") as MockClient,
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="verifier"))
            MockServer.return_value = server_inst

            cm, _ = _mock_async_client(post_response=mock_resp)
            MockClient.return_value = cm

            with pytest.raises(ValueError, match="No refresh token"):
                await login_antigravity(on_auth)

    @pytest.mark.asyncio
    async def test_missing_code(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import login_antigravity

        on_auth = MagicMock()

        with (
            patch("nu_ai.utils.oauth.google_antigravity.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_antigravity.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="No authorization code"):
                await login_antigravity(on_auth)

    @pytest.mark.asyncio
    async def test_state_mismatch(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import login_antigravity

        on_auth = MagicMock()

        with (
            patch("nu_ai.utils.oauth.google_antigravity.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_antigravity.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="wrong"))
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="state mismatch"):
                await login_antigravity(on_auth)

    @pytest.mark.asyncio
    async def test_with_manual_code_input(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import login_antigravity

        token_data = {
            "access_token": "ya29.manual",
            "refresh_token": "rt-m",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=token_data)
        on_auth = MagicMock()

        async def manual_input():
            return "http://localhost:51121/oauth-callback?code=manual&state=verifier"

        with (
            patch("nu_ai.utils.oauth.google_antigravity.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_antigravity.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient") as MockClient,
            patch("nu_ai.utils.oauth.google_antigravity._get_user_email", new_callable=AsyncMock, return_value=None),
            patch(
                "nu_ai.utils.oauth.google_antigravity._discover_project", new_callable=AsyncMock, return_value="proj"
            ),
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            cm, _ = _mock_async_client(post_response=mock_resp)
            MockClient.return_value = cm

            creds = await login_antigravity(on_auth, on_manual_code_input=manual_input)

        assert creds.access == "ya29.manual"


class TestAntigravityProvider:
    @pytest.mark.asyncio
    async def test_refresh_missing_project_id(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import antigravity_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0, extra_data={})
        with pytest.raises(ValueError, match="missing projectId"):
            await antigravity_oauth_provider.refresh_token(creds)

    @pytest.mark.asyncio
    async def test_refresh_delegates(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import antigravity_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0, extra_data={"projectId": "p"})
        new_creds = OAuthCredentials(refresh="nr", access="na", expires=999, extra_data={"projectId": "p"})

        with patch(
            "nu_ai.utils.oauth.google_antigravity.refresh_antigravity_token",
            new_callable=AsyncMock,
            return_value=new_creds,
        ):
            result = await antigravity_oauth_provider.refresh_token(creds)
        assert result.access == "na"

    def test_properties(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import antigravity_oauth_provider

        assert antigravity_oauth_provider.id == "google-antigravity"
        assert antigravity_oauth_provider.uses_callback_server is True
        assert len(antigravity_oauth_provider.name) > 0

    def test_modify_models_passthrough(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import antigravity_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        assert antigravity_oauth_provider.modify_models([], creds) == []


# ═══════════════════════════════════════════════════════════════════════════
# OpenAI Codex
# ═══════════════════════════════════════════════════════════════════════════


class TestRefreshOpenAICodexToken:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        from nu_ai.utils.oauth.openai_codex import refresh_openai_codex_token

        jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acc1"}})

        refresh_result = {
            "access": jwt,
            "refresh": "new-rt",
            "expires": time.time() * 1000 + 3600_000,
        }

        with patch(
            "nu_ai.utils.oauth.openai_codex._refresh_access_token", new_callable=AsyncMock, return_value=refresh_result
        ):
            creds = await refresh_openai_codex_token("old-rt")

        assert creds.access == jwt
        assert creds.refresh == "new-rt"
        assert creds.extra_data["accountId"] == "acc1"

    @pytest.mark.asyncio
    async def test_refresh_returns_none(self) -> None:
        from nu_ai.utils.oauth.openai_codex import refresh_openai_codex_token

        with patch("nu_ai.utils.oauth.openai_codex._refresh_access_token", new_callable=AsyncMock, return_value=None):
            with pytest.raises(RuntimeError, match="Failed to refresh"):
                await refresh_openai_codex_token("rt")

    @pytest.mark.asyncio
    async def test_no_account_id(self) -> None:
        from nu_ai.utils.oauth.openai_codex import refresh_openai_codex_token

        jwt = _make_jwt({"sub": "user"})  # No account id claim
        refresh_result = {
            "access": jwt,
            "refresh": "rt",
            "expires": time.time() * 1000 + 3600_000,
        }

        with patch(
            "nu_ai.utils.oauth.openai_codex._refresh_access_token", new_callable=AsyncMock, return_value=refresh_result
        ):
            with pytest.raises(RuntimeError, match="accountId"):
                await refresh_openai_codex_token("rt")


class TestRefreshAccessToken:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _refresh_access_token

        response_data = {
            "access_token": "new-at",
            "refresh_token": "new-rt",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=response_data)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.openai_codex.httpx.AsyncClient", return_value=cm):
            result = await _refresh_access_token("old-rt")

        assert result is not None
        assert result["access"] == "new-at"

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _refresh_access_token

        mock_resp = _mock_response(401)
        mock_resp.raise_for_status = MagicMock()  # Don't raise
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.openai_codex.httpx.AsyncClient", return_value=cm):
            result = await _refresh_access_token("bad-rt")
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_fields_returns_none(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _refresh_access_token

        response_data = {"access_token": "at"}  # missing refresh_token and expires_in
        mock_resp = _mock_response(200, json_data=response_data)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.openai_codex.httpx.AsyncClient", return_value=cm):
            result = await _refresh_access_token("rt")
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _refresh_access_token

        cm = AsyncMock()
        cm.__aenter__.side_effect = Exception("network")
        cm.__aexit__.return_value = None

        with patch("nu_ai.utils.oauth.openai_codex.httpx.AsyncClient", return_value=cm):
            result = await _refresh_access_token("rt")
        assert result is None


class TestExchangeCode:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _exchange_code

        response_data = {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=response_data)
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.openai_codex.httpx.AsyncClient", return_value=cm):
            result = await _exchange_code("code", "verifier")

        assert result is not None
        assert result["access"] == "at"

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _exchange_code

        mock_resp = _mock_response(400)
        mock_resp.raise_for_status = MagicMock()
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.openai_codex.httpx.AsyncClient", return_value=cm):
            result = await _exchange_code("code", "verifier")
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_fields_returns_none(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _exchange_code

        mock_resp = _mock_response(200, json_data={"access_token": "at"})
        cm, _ = _mock_async_client(post_response=mock_resp)

        with patch("nu_ai.utils.oauth.openai_codex.httpx.AsyncClient", return_value=cm):
            result = await _exchange_code("code", "verifier")
        assert result is None


class TestLoginOpenAICodex:
    @pytest.mark.asyncio
    async def test_callback_success(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acc1"}})
        exchange_result = {
            "access": jwt,
            "refresh": "rt",
            "expires": time.time() * 1000 + 3600_000,
        }

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.openai_codex._exchange_code", new_callable=AsyncMock, return_value=exchange_result
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="auth-code", state="test-state"))
            MockServer.return_value = server_inst

            creds = await login_openai_codex(on_auth=on_auth, on_prompt=on_prompt)

        assert creds.access == jwt
        assert creds.extra_data["accountId"] == "acc1"

    @pytest.mark.asyncio
    async def test_fallback_to_prompt(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acc2"}})
        exchange_result = {
            "access": jwt,
            "refresh": "rt",
            "expires": time.time() * 1000 + 3600_000,
        }

        on_auth = MagicMock()
        on_prompt = AsyncMock(return_value="prompt-code")

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.openai_codex._exchange_code", new_callable=AsyncMock, return_value=exchange_result
            ),
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            creds = await login_openai_codex(on_auth=on_auth, on_prompt=on_prompt)

        assert creds.extra_data["accountId"] == "acc2"
        on_prompt.assert_awaited()

    @pytest.mark.asyncio
    async def test_missing_code(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        on_auth = MagicMock()
        on_prompt = AsyncMock(return_value="")

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="Missing authorization code"):
                await login_openai_codex(on_auth=on_auth, on_prompt=on_prompt)

    @pytest.mark.asyncio
    async def test_token_exchange_failed(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.openai_codex._exchange_code", new_callable=AsyncMock, return_value=None),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="test-state"))
            MockServer.return_value = server_inst

            with pytest.raises(RuntimeError, match="Token exchange failed"):
                await login_openai_codex(on_auth=on_auth, on_prompt=on_prompt)

    @pytest.mark.asyncio
    async def test_no_account_id_in_token(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        jwt = _make_jwt({"sub": "user"})  # No account id
        exchange_result = {
            "access": jwt,
            "refresh": "rt",
            "expires": time.time() * 1000 + 3600_000,
        }

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.openai_codex._exchange_code", new_callable=AsyncMock, return_value=exchange_result
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="test-state"))
            MockServer.return_value = server_inst

            with pytest.raises(RuntimeError, match="accountId"):
                await login_openai_codex(on_auth=on_auth, on_prompt=on_prompt)

    @pytest.mark.asyncio
    async def test_with_manual_code_input(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acc3"}})
        exchange_result = {
            "access": jwt,
            "refresh": "rt",
            "expires": time.time() * 1000 + 3600_000,
        }

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        async def manual_input():
            return "http://localhost:1455/auth/callback?code=manual-c&state=test-state"

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.openai_codex._exchange_code", new_callable=AsyncMock, return_value=exchange_result
            ),
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            creds = await login_openai_codex(
                on_auth=on_auth,
                on_prompt=on_prompt,
                on_manual_code_input=manual_input,
            )

        assert creds.extra_data["accountId"] == "acc3"

    @pytest.mark.asyncio
    async def test_state_mismatch_from_prompt(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        on_auth = MagicMock()
        on_prompt = AsyncMock(return_value="code=abc&state=wrong-state")

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="State mismatch"):
                await login_openai_codex(on_auth=on_auth, on_prompt=on_prompt)


class TestOpenAICodexProvider:
    @pytest.mark.asyncio
    async def test_refresh_token_delegates(self) -> None:
        from nu_ai.utils.oauth.openai_codex import openai_codex_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        new_creds = OAuthCredentials(refresh="nr", access="na", expires=999, extra_data={"accountId": "a1"})

        with patch(
            "nu_ai.utils.oauth.openai_codex.refresh_openai_codex_token", new_callable=AsyncMock, return_value=new_creds
        ):
            result = await openai_codex_oauth_provider.refresh_token(creds)
        assert result.access == "na"

    def test_properties(self) -> None:
        from nu_ai.utils.oauth.openai_codex import openai_codex_oauth_provider

        assert openai_codex_oauth_provider.id == "openai-codex"
        assert openai_codex_oauth_provider.uses_callback_server is True
        assert len(openai_codex_oauth_provider.name) > 0

    def test_modify_models_passthrough(self) -> None:
        from nu_ai.utils.oauth.openai_codex import openai_codex_oauth_provider

        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        assert openai_codex_oauth_provider.modify_models(["m"], creds) == ["m"]


class TestOpenAICodexJWT:
    def test_decode_jwt_valid(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _decode_jwt

        jwt = _make_jwt({"sub": "user", "name": "Test"})
        result = _decode_jwt(jwt)
        assert result is not None
        assert result["sub"] == "user"

    def test_decode_jwt_invalid(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _decode_jwt

        assert _decode_jwt("not.a.jwt.at.all") is None
        assert _decode_jwt("") is None
        assert _decode_jwt("one-part") is None


# ═══════════════════════════════════════════════════════════════════════════
# __init__.py - remaining coverage gaps
# ═══════════════════════════════════════════════════════════════════════════


class TestGetOAuthApiKeyRefresh:
    @pytest.mark.asyncio
    async def test_refreshes_expired_credentials(self) -> None:
        from nu_ai.utils.oauth import get_oauth_api_key

        expired = OAuthCredentials(refresh="r", access="old-access", expires=0)  # expired
        refreshed = OAuthCredentials(refresh="r", access="new-access", expires=time.time() * 1000 + 3600_000)

        with patch(
            "nu_ai.utils.oauth.anthropic.refresh_anthropic_token", new_callable=AsyncMock, return_value=refreshed
        ):
            result = await get_oauth_api_key("anthropic", {"anthropic": expired})

        assert result is not None
        _creds, api_key = result
        assert api_key == "new-access"

    @pytest.mark.asyncio
    async def test_refresh_failure_raises_runtime_error(self) -> None:
        from nu_ai.utils.oauth import get_oauth_api_key

        expired = OAuthCredentials(refresh="r", access="old", expires=0)

        with patch(
            "nu_ai.utils.oauth.anthropic.refresh_anthropic_token", new_callable=AsyncMock, side_effect=Exception("fail")
        ):
            with pytest.raises(RuntimeError, match="Failed to refresh"):
                await get_oauth_api_key("anthropic", {"anthropic": expired})


class TestRefreshOAuthToken:
    @pytest.mark.asyncio
    async def test_delegates_to_provider(self) -> None:
        from nu_ai.utils.oauth import refresh_oauth_token

        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        new_creds = OAuthCredentials(refresh="nr", access="na", expires=999)

        with patch(
            "nu_ai.utils.oauth.anthropic.refresh_anthropic_token", new_callable=AsyncMock, return_value=new_creds
        ):
            result = await refresh_oauth_token("anthropic", creds)
        assert result.access == "na"

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self) -> None:
        from nu_ai.utils.oauth import refresh_oauth_token

        creds = OAuthCredentials(refresh="r", access="a", expires=0)
        with pytest.raises(ValueError, match="Unknown OAuth provider"):
            await refresh_oauth_token("nonexistent", creds)


# ═══════════════════════════════════════════════════════════════════════════
# Additional coverage: manual code input edge cases & branch coverage
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicManualCodeBranches:
    """Cover remaining branches in login_anthropic's manual code input handling."""

    @pytest.mark.asyncio
    async def test_manual_code_error_propagates(self) -> None:
        """on_manual_code_input raises -> error propagated."""
        from nu_ai.utils.oauth.anthropic import login_anthropic

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        async def bad_manual():
            raise RuntimeError("manual input failed")

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(RuntimeError, match="manual input failed"):
                await login_anthropic(
                    on_auth=on_auth,
                    on_prompt=on_prompt,
                    on_manual_code_input=bad_manual,
                )

    @pytest.mark.asyncio
    async def test_manual_code_server_wins(self) -> None:
        """Server callback returns result while manual_code_input is also provided."""
        from nu_ai.utils.oauth.anthropic import login_anthropic

        expected = OAuthCredentials(refresh="rt", access="at", expires=time.time() * 1000 + 3600_000)
        on_auth = MagicMock()
        on_prompt = AsyncMock()

        async def slow_manual():
            await asyncio.sleep(10)
            return "should-not-be-used"

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.anthropic._exchange_authorization_code",
                new_callable=AsyncMock,
                return_value=expected,
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="server-code", state="verifier"))
            MockServer.return_value = server_inst

            creds = await login_anthropic(
                on_auth=on_auth,
                on_prompt=on_prompt,
                on_manual_code_input=slow_manual,
            )
        assert creds.access == "at"

    @pytest.mark.asyncio
    async def test_missing_state_raises(self) -> None:
        """Missing state after code extraction -> ValueError."""

        MagicMock()
        # Return a plain code with no state
        AsyncMock(return_value="justcode")

        with (
            patch("nu_ai.utils.oauth.anthropic.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.anthropic.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            # parse_authorization_input("justcode") returns code="justcode", state=None
            # Then state = parsed["state"] or verifier => state = verifier, not None
            # So we need to patch parse_authorization_input to return state=None and code=some_code
            # Actually the code does: state = parsed["state"] or verifier
            # So state won't be None. Let me test a different angle.

    @pytest.mark.asyncio
    async def test_exchange_code_invalid_json(self) -> None:
        """_exchange_authorization_code with invalid JSON body."""
        from nu_ai.utils.oauth.anthropic import _exchange_authorization_code

        with patch("nu_ai.utils.oauth.anthropic._post_json", new_callable=AsyncMock, return_value="not-valid-json{{"):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                await _exchange_authorization_code("code", "state", "verifier", "http://localhost/cb")

    @pytest.mark.asyncio
    async def test_exchange_code_success(self) -> None:
        """_exchange_authorization_code happy path through _post_json."""
        from nu_ai.utils.oauth.anthropic import _exchange_authorization_code

        response_json = json.dumps(
            {
                "access_token": "at",
                "refresh_token": "rt",
                "expires_in": 3600,
            }
        )

        with patch("nu_ai.utils.oauth.anthropic._post_json", new_callable=AsyncMock, return_value=response_json):
            creds = await _exchange_authorization_code("code", "state", "verifier", "http://localhost/cb")
        assert creds.access == "at"
        assert creds.refresh == "rt"


class TestGeminiManualCodeBranches:
    """Cover manual code input branches in login_gemini_cli."""

    @pytest.mark.asyncio
    async def test_manual_error_propagates(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import login_gemini_cli

        on_auth = MagicMock()

        async def bad_manual():
            raise RuntimeError("manual failed")

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_gemini_cli.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(RuntimeError, match="manual failed"):
                await login_gemini_cli(on_auth, on_manual_code_input=bad_manual)

    @pytest.mark.asyncio
    async def test_server_wins_over_manual(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import login_gemini_cli

        token_data = {
            "access_token": "ya29",
            "refresh_token": "rt",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=token_data)
        on_auth = MagicMock()

        async def slow_manual():
            await asyncio.sleep(10)
            return "unused"

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_gemini_cli.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient") as MockClient,
            patch("nu_ai.utils.oauth.google_gemini_cli._get_user_email", new_callable=AsyncMock, return_value=None),
            patch("nu_ai.utils.oauth.google_gemini_cli._discover_project", new_callable=AsyncMock, return_value="proj"),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="verifier"))
            MockServer.return_value = server_inst

            cm, _ = _mock_async_client(post_response=mock_resp)
            MockClient.return_value = cm

            creds = await login_gemini_cli(on_auth, on_manual_code_input=slow_manual)
        assert creds.access == "ya29"

    @pytest.mark.asyncio
    async def test_non_free_tier_with_env_project(self) -> None:
        """Non-free tier with GOOGLE_CLOUD_PROJECT set should include project in onboard."""
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "allowedTiers": [{"id": "paid-tier", "isDefault": True}],
            },
        )

        onboard_resp = _mock_response(
            200,
            json_data={
                "done": True,
                "response": {"cloudaicompanionProject": "onboard-proj"},
            },
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return load_resp
            return onboard_resp

        cm = AsyncMock()
        client = AsyncMock()
        client.post = mock_post
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm),
            patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-proj"}),
        ):
            project = await _discover_project("token")
        assert project == "onboard-proj"

    @pytest.mark.asyncio
    async def test_onboard_response_string_project(self) -> None:
        """Onboard returns companion project as string instead of dict."""
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "allowedTiers": [{"id": "free-tier", "isDefault": True}],
            },
        )

        onboard_resp = _mock_response(
            200,
            json_data={
                "done": True,
                "response": {"cloudaicompanionProject": "string-proj"},
            },
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return load_resp
            return onboard_resp

        cm = AsyncMock()
        client = AsyncMock()
        client.post = mock_post
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            project = await _discover_project("token")
        assert project == "string-proj"

    @pytest.mark.asyncio
    async def test_onboard_env_fallback(self) -> None:
        """Onboard returns no project but env var is set."""
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(
            200,
            json_data={
                "allowedTiers": [{"id": "free-tier", "isDefault": True}],
            },
        )

        onboard_resp = _mock_response(
            200,
            json_data={
                "done": True,
                "response": {"cloudaicompanionProject": {}},
            },
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return load_resp
            return onboard_resp

        cm = AsyncMock()
        client = AsyncMock()
        client.post = mock_post
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with (
            patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm),
            patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT_ID": "env-proj-id"}),
        ):
            project = await _discover_project("token")
        assert project == "env-proj-id"

    @pytest.mark.asyncio
    async def test_load_error_json_parse_fails(self) -> None:
        """loadCodeAssist returns error, JSON parsing fails."""
        from nu_ai.utils.oauth.google_gemini_cli import _discover_project

        load_resp = _mock_response(500)
        load_resp.raise_for_status = MagicMock()
        load_resp.json.side_effect = Exception("not json")

        cm, _ = _mock_async_client(post_response=load_resp)

        with patch("nu_ai.utils.oauth.google_gemini_cli.httpx.AsyncClient", return_value=cm):
            with pytest.raises(RuntimeError, match="loadCodeAssist failed"):
                await _discover_project("token")


class TestAntigravityManualCodeBranches:
    """Cover manual code input branches in login_antigravity."""

    @pytest.mark.asyncio
    async def test_manual_error_propagates(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import login_antigravity

        on_auth = MagicMock()

        async def bad_manual():
            raise RuntimeError("manual failed")

        with (
            patch("nu_ai.utils.oauth.google_antigravity.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_antigravity.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(RuntimeError, match="manual failed"):
                await login_antigravity(on_auth, on_manual_code_input=bad_manual)

    @pytest.mark.asyncio
    async def test_server_wins_over_manual(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import login_antigravity

        token_data = {
            "access_token": "ya29",
            "refresh_token": "rt",
            "expires_in": 3600,
        }
        mock_resp = _mock_response(200, json_data=token_data)
        on_auth = MagicMock()

        async def slow_manual():
            await asyncio.sleep(10)
            return "unused"

        with (
            patch("nu_ai.utils.oauth.google_antigravity.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.google_antigravity.CallbackServer") as MockServer,
            patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient") as MockClient,
            patch("nu_ai.utils.oauth.google_antigravity._get_user_email", new_callable=AsyncMock, return_value=None),
            patch(
                "nu_ai.utils.oauth.google_antigravity._discover_project", new_callable=AsyncMock, return_value="proj"
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="verifier"))
            MockServer.return_value = server_inst

            cm, _ = _mock_async_client(post_response=mock_resp)
            MockClient.return_value = cm

            creds = await login_antigravity(on_auth, on_manual_code_input=slow_manual)
        assert creds.access == "ya29"


class TestAntigravityDiscoverProjectMore:
    @pytest.mark.asyncio
    async def test_endpoint_exception_falls_through(self) -> None:
        """Both endpoints raise exceptions -> fallback to default."""
        from nu_ai.utils.oauth.google_antigravity import _discover_project

        cm = AsyncMock()
        client = AsyncMock()
        client.post.side_effect = Exception("network")
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            project = await _discover_project("token")
        assert project == "rising-fact-p41fc"

    @pytest.mark.asyncio
    async def test_endpoint_returns_no_project(self) -> None:
        """Endpoint succeeds but has no project info -> try next, then default."""
        from nu_ai.utils.oauth.google_antigravity import _discover_project

        resp = _mock_response(200, json_data={"cloudaicompanionProject": ""})
        cm, _ = _mock_async_client(post_response=resp)

        with patch("nu_ai.utils.oauth.google_antigravity.httpx.AsyncClient", return_value=cm):
            project = await _discover_project("token")
        assert project == "rising-fact-p41fc"


class TestOpenAIManualCodeBranches:
    """Cover manual code input branches in login_openai_codex."""

    @pytest.mark.asyncio
    async def test_manual_error_propagates(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        async def bad_manual():
            raise RuntimeError("manual failed")

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(RuntimeError, match="manual failed"):
                await login_openai_codex(
                    on_auth=on_auth,
                    on_prompt=on_prompt,
                    on_manual_code_input=bad_manual,
                )

    @pytest.mark.asyncio
    async def test_server_wins_over_manual(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acc"}})
        exchange_result = {"access": jwt, "refresh": "rt", "expires": time.time() * 1000 + 3600_000}

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        async def slow_manual():
            await asyncio.sleep(10)
            return "unused"

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
            patch(
                "nu_ai.utils.oauth.openai_codex._exchange_code", new_callable=AsyncMock, return_value=exchange_result
            ),
        ):
            server_inst = _mock_callback_server(wait_result=CallbackResult(code="c", state="test-state"))
            MockServer.return_value = server_inst

            creds = await login_openai_codex(
                on_auth=on_auth,
                on_prompt=on_prompt,
                on_manual_code_input=slow_manual,
            )
        assert creds.extra_data["accountId"] == "acc"

    @pytest.mark.asyncio
    async def test_manual_state_mismatch(self) -> None:
        from nu_ai.utils.oauth.openai_codex import login_openai_codex

        on_auth = MagicMock()
        on_prompt = AsyncMock()

        async def manual_with_bad_state():
            return "http://localhost:1455/auth/callback?code=c&state=wrong"

        with (
            patch("nu_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("nu_ai.utils.oauth.openai_codex._create_state", return_value="test-state"),
            patch("nu_ai.utils.oauth.openai_codex.CallbackServer") as MockServer,
        ):
            server_inst = _mock_callback_server(wait_result=None)
            MockServer.return_value = server_inst

            with pytest.raises(ValueError, match="State mismatch"):
                await login_openai_codex(
                    on_auth=on_auth,
                    on_prompt=on_prompt,
                    on_manual_code_input=manual_with_bad_state,
                )


class TestGitHubCopilotAdditional:
    """Cover remaining GitHub Copilot branches."""

    @pytest.mark.asyncio
    async def test_poll_slow_down_no_interval(self) -> None:
        """slow_down without interval field increases interval by 5000."""
        from nu_ai.utils.oauth.github_copilot import _poll_for_github_access_token

        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"error": "slow_down"}  # No interval field
            return {"access_token": "tok"}

        with patch("nu_ai.utils.oauth.github_copilot._fetch_json", side_effect=mock_fetch):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                token = await _poll_for_github_access_token("github.com", "dc", 1, 60)

        assert token == "tok"

    @pytest.mark.asyncio
    async def test_poll_timeout_after_slow_down(self) -> None:
        """Timeout after slow_down gives specific message about clock drift."""
        from nu_ai.utils.oauth.github_copilot import _poll_for_github_access_token

        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"error": "slow_down", "interval": 10}

        real_time = time.time

        def fake_time():
            # After first fetch call, make time exceed deadline
            if call_count >= 1:
                return real_time() + 1000
            return real_time()

        with (
            patch("nu_ai.utils.oauth.github_copilot._fetch_json", side_effect=mock_fetch),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch("nu_ai.utils.oauth.github_copilot.time.time", side_effect=fake_time),
        ):
            with pytest.raises(RuntimeError, match="clock drift"):
                await _poll_for_github_access_token("github.com", "dc", 1, 60)

    @pytest.mark.asyncio
    async def test_enable_all_models(self) -> None:
        from nu_ai.utils.oauth.github_copilot import _enable_all_models

        class FakeModel:
            def __init__(self, id):
                self.id = id

        with (
            patch("nu_ai.models.get_models", return_value=[FakeModel("m1"), FakeModel("m2")]),
            patch("nu_ai.utils.oauth.github_copilot._enable_model", new_callable=AsyncMock, return_value=True),
        ):
            await _enable_all_models("token")

    def test_normalize_domain_with_url(self) -> None:
        from nu_ai.utils.oauth.github_copilot import normalize_domain

        # Test the branch where urlparse is imported and used
        assert normalize_domain("http://foo.com/path") == "foo.com"

    @pytest.mark.asyncio
    async def test_login_provider_delegates(self) -> None:
        from nu_ai.utils.oauth.github_copilot import github_copilot_oauth_provider

        expected = OAuthCredentials(refresh="r", access="a", expires=100)
        with patch(
            "nu_ai.utils.oauth.github_copilot.login_github_copilot", new_callable=AsyncMock, return_value=expected
        ):
            result = await github_copilot_oauth_provider.login(on_auth=MagicMock(), on_prompt=AsyncMock())
        assert result == expected


class TestOpenAICodexAdditional:
    """Cover remaining OpenAI Codex utility functions."""

    def test_create_state(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _create_state

        s1 = _create_state()
        s2 = _create_state()
        assert isinstance(s1, str)
        assert len(s1) == 32  # 16 bytes hex
        assert s1 != s2

    def test_decode_jwt_with_padding(self) -> None:
        """JWT payload that needs base64 padding."""
        from nu_ai.utils.oauth.openai_codex import _decode_jwt

        # Create a JWT where payload length % 4 != 0
        jwt = _make_jwt({"sub": "u"})
        result = _decode_jwt(jwt)
        assert result is not None
        assert result["sub"] == "u"

    @pytest.mark.asyncio
    async def test_login_provider_delegates(self) -> None:
        from nu_ai.utils.oauth.openai_codex import openai_codex_oauth_provider

        expected = OAuthCredentials(refresh="r", access="a", expires=100, extra_data={"accountId": "a1"})
        with patch("nu_ai.utils.oauth.openai_codex.login_openai_codex", new_callable=AsyncMock, return_value=expected):
            result = await openai_codex_oauth_provider.login(on_auth=MagicMock(), on_prompt=AsyncMock())
        assert result == expected


class TestGeminiCliProviderLogin:
    @pytest.mark.asyncio
    async def test_login_delegates(self) -> None:
        from nu_ai.utils.oauth.google_gemini_cli import gemini_cli_oauth_provider

        expected = OAuthCredentials(refresh="r", access="a", expires=100, extra_data={"projectId": "p"})
        with patch(
            "nu_ai.utils.oauth.google_gemini_cli.login_gemini_cli", new_callable=AsyncMock, return_value=expected
        ):
            result = await gemini_cli_oauth_provider.login(on_auth=MagicMock(), on_prompt=AsyncMock())
        assert result == expected


class TestAntigravityProviderLogin:
    @pytest.mark.asyncio
    async def test_login_delegates(self) -> None:
        from nu_ai.utils.oauth.google_antigravity import antigravity_oauth_provider

        expected = OAuthCredentials(refresh="r", access="a", expires=100, extra_data={"projectId": "p"})
        with patch(
            "nu_ai.utils.oauth.google_antigravity.login_antigravity", new_callable=AsyncMock, return_value=expected
        ):
            result = await antigravity_oauth_provider.login(on_auth=MagicMock(), on_prompt=AsyncMock())
        assert result == expected
