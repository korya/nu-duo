"""Tests for the OAuth module.

Tests cover:
- PKCE generation
- HTML page rendering
- Authorization input parsing
- Callback server lifecycle
- Provider registry (register, unregister, reset)
- High-level get_oauth_api_key
- Provider interface conformance (all 5 providers)
- GitHub Copilot domain normalization / base URL extraction
- OpenAI Codex JWT decoding
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest
from nu_ai.utils.oauth.callback_server import CallbackServer, parse_authorization_input
from nu_ai.utils.oauth.oauth_page import oauth_error_html, oauth_success_html
from nu_ai.utils.oauth.pkce import generate_pkce
from nu_ai.utils.oauth.types import OAuthCredentials

# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------


class TestPKCE:
    def test_generate_pkce_returns_two_strings(self) -> None:
        verifier, challenge = generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) > 20
        assert len(challenge) > 20

    def test_generate_pkce_deterministic_challenge(self) -> None:
        """Same verifier always produces the same challenge."""
        import base64
        import hashlib

        verifier, challenge = generate_pkce()
        expected_hash = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(expected_hash).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_generate_pkce_unique(self) -> None:
        """Two calls produce different verifiers."""
        v1, _ = generate_pkce()
        v2, _ = generate_pkce()
        assert v1 != v2

    def test_pkce_base64url_no_padding(self) -> None:
        """Verifier and challenge should not contain padding or non-URL chars."""
        verifier, challenge = generate_pkce()
        for s in (verifier, challenge):
            assert "=" not in s
            assert "+" not in s
            assert "/" not in s


# ---------------------------------------------------------------------------
# OAuth HTML pages
# ---------------------------------------------------------------------------


class TestOAuthPage:
    def test_success_html_contains_message(self) -> None:
        html = oauth_success_html("Login complete!")
        assert "Login complete!" in html
        assert "Authentication successful" in html

    def test_error_html_contains_message(self) -> None:
        html = oauth_error_html("Something went wrong")
        assert "Something went wrong" in html
        assert "Authentication failed" in html

    def test_error_html_with_details(self) -> None:
        html = oauth_error_html("Failed", details="Error: timeout")
        assert "Error: timeout" in html

    def test_html_escapes_special_chars(self) -> None:
        html = oauth_success_html("<script>alert('xss')</script>")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Authorization input parsing
# ---------------------------------------------------------------------------


class TestParseAuthorizationInput:
    def test_empty_string(self) -> None:
        result = parse_authorization_input("")
        assert result["code"] is None
        assert result["state"] is None

    def test_plain_code(self) -> None:
        result = parse_authorization_input("abc123")
        assert result["code"] == "abc123"
        assert result["state"] is None

    def test_url_with_code_and_state(self) -> None:
        url = "http://localhost:53692/callback?code=mycode&state=mystate"
        result = parse_authorization_input(url)
        assert result["code"] == "mycode"
        assert result["state"] == "mystate"

    def test_code_hash_state(self) -> None:
        result = parse_authorization_input("mycode#mystate")
        assert result["code"] == "mycode"
        assert result["state"] == "mystate"

    def test_query_string_format(self) -> None:
        result = parse_authorization_input("code=abc&state=xyz")
        assert result["code"] == "abc"
        assert result["state"] == "xyz"

    def test_whitespace_trimmed(self) -> None:
        result = parse_authorization_input("  abc123  ")
        assert result["code"] == "abc123"


# ---------------------------------------------------------------------------
# Callback server
# ---------------------------------------------------------------------------


class TestCallbackServer:
    async def test_server_start_and_close(self) -> None:
        server = CallbackServer(port=19876, path="/cb", provider_name="Test")
        await server.start()
        assert server.redirect_uri == "http://localhost:19876/cb"
        server.close()

    async def test_cancel_wait_returns_none(self) -> None:
        server = CallbackServer(port=19877, path="/cb", provider_name="Test")
        await server.start()
        try:
            server.cancel_wait()
            result = await asyncio.wait_for(server.wait_for_code(), timeout=2.0)
            assert result is None
        finally:
            server.close()

    async def test_callback_delivers_code(self) -> None:
        server = CallbackServer(port=19878, path="/cb", provider_name="Test")
        await server.start()
        try:
            # Hit the callback endpoint
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://127.0.0.1:19878/cb?code=testcode&state=teststate")
                assert resp.status_code == 200
                assert "Authentication successful" in resp.text

            result = await asyncio.wait_for(server.wait_for_code(), timeout=2.0)
            assert result is not None
            assert result.code == "testcode"
            assert result.state == "teststate"
        finally:
            server.close()

    async def test_callback_wrong_path_returns_404(self) -> None:
        server = CallbackServer(port=19879, path="/cb", provider_name="Test")
        await server.start()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://127.0.0.1:19879/wrong?code=x&state=y")
                assert resp.status_code == 404
        finally:
            server.close()

    async def test_callback_missing_code(self) -> None:
        server = CallbackServer(port=19880, path="/cb", provider_name="Test")
        await server.start()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://127.0.0.1:19880/cb?state=only")
                assert resp.status_code == 400
        finally:
            server.close()

    async def test_callback_state_mismatch(self) -> None:
        server = CallbackServer(port=19881, path="/cb", expected_state="correct", provider_name="Test")
        await server.start()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://127.0.0.1:19881/cb?code=x&state=wrong")
                assert resp.status_code == 400
                assert "State mismatch" in resp.text
        finally:
            server.close()

    async def test_callback_error_param(self) -> None:
        server = CallbackServer(port=19882, path="/cb", provider_name="Test")
        await server.start()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://127.0.0.1:19882/cb?error=access_denied")
                assert resp.status_code == 400
                assert "access_denied" in resp.text
        finally:
            server.close()


# ---------------------------------------------------------------------------
# OAuthCredentials model
# ---------------------------------------------------------------------------


class TestOAuthCredentials:
    def test_basic_fields(self) -> None:
        creds = OAuthCredentials(refresh="r", access="a", expires=123.0)
        assert creds.refresh == "r"
        assert creds.access == "a"
        assert creds.expires == 123.0
        assert creds.extra_data == {}

    def test_extra_data(self) -> None:
        creds = OAuthCredentials(refresh="r", access="a", expires=123.0, extra_data={"projectId": "p1"})
        assert creds.extra_data["projectId"] == "p1"

    def test_serialization_round_trip(self) -> None:
        creds = OAuthCredentials(refresh="r", access="a", expires=99.0, extra_data={"k": "v"})
        data = creds.model_dump()
        restored = OAuthCredentials(**data)
        assert restored == creds


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_get_known_providers(self) -> None:
        from nu_ai.utils.oauth import get_oauth_provider

        for pid in ["anthropic", "github-copilot", "google-gemini-cli", "google-antigravity", "openai-codex"]:
            provider = get_oauth_provider(pid)
            assert provider is not None, f"Missing provider: {pid}"
            assert provider.id == pid

    def test_get_unknown_returns_none(self) -> None:
        from nu_ai.utils.oauth import get_oauth_provider

        assert get_oauth_provider("nonexistent") is None

    def test_get_all_providers(self) -> None:
        from nu_ai.utils.oauth import get_oauth_providers

        providers = get_oauth_providers()
        ids = {p.id for p in providers}
        assert "anthropic" in ids
        assert "github-copilot" in ids
        assert len(providers) >= 5

    def test_register_custom_provider(self) -> None:
        from nu_ai.utils.oauth import (
            get_oauth_provider,
            register_oauth_provider,
            reset_oauth_providers,
        )

        class FakeProvider:
            @property
            def id(self) -> str:
                return "fake"

            @property
            def name(self) -> str:
                return "Fake"

            @property
            def uses_callback_server(self) -> bool:
                return False

            async def login(self, **kwargs: Any) -> OAuthCredentials:
                return OAuthCredentials(refresh="", access="", expires=0)

            async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
                return credentials

            def get_api_key(self, credentials: OAuthCredentials) -> str:
                return ""

            def modify_models(self, models: list[Any], credentials: OAuthCredentials) -> list[Any]:
                return models

        try:
            register_oauth_provider(FakeProvider())  # type: ignore[arg-type]
            assert get_oauth_provider("fake") is not None
        finally:
            reset_oauth_providers()
            assert get_oauth_provider("fake") is None

    def test_unregister_builtin_restores(self) -> None:
        from nu_ai.utils.oauth import get_oauth_provider, reset_oauth_providers, unregister_oauth_provider

        try:
            unregister_oauth_provider("anthropic")
            # Built-in should be restored, not removed
            provider = get_oauth_provider("anthropic")
            assert provider is not None
            assert provider.id == "anthropic"
        finally:
            reset_oauth_providers()

    def test_unregister_custom_removes(self) -> None:
        from nu_ai.utils.oauth import get_oauth_provider, reset_oauth_providers, unregister_oauth_provider

        try:
            unregister_oauth_provider("nonexistent-custom")
            assert get_oauth_provider("nonexistent-custom") is None
        finally:
            reset_oauth_providers()


# ---------------------------------------------------------------------------
# High-level get_oauth_api_key
# ---------------------------------------------------------------------------


class TestGetOAuthApiKey:
    async def test_returns_none_when_no_credentials(self) -> None:
        from nu_ai.utils.oauth import get_oauth_api_key

        result = await get_oauth_api_key("anthropic", {})
        assert result is None

    async def test_returns_key_when_not_expired(self) -> None:
        from nu_ai.utils.oauth import get_oauth_api_key

        far_future = time.time() * 1000 + 3600 * 1000
        creds = OAuthCredentials(refresh="r", access="my-access-token", expires=far_future)
        result = await get_oauth_api_key("anthropic", {"anthropic": creds})
        assert result is not None
        _new_creds, api_key = result
        assert api_key == "my-access-token"

    async def test_raises_for_unknown_provider(self) -> None:
        from nu_ai.utils.oauth import get_oauth_api_key

        with pytest.raises(ValueError, match="Unknown OAuth provider"):
            await get_oauth_api_key("nonexistent", {})


# ---------------------------------------------------------------------------
# GitHub Copilot specifics
# ---------------------------------------------------------------------------


class TestGitHubCopilot:
    def test_normalize_domain_basic(self) -> None:
        from nu_ai.utils.oauth.github_copilot import normalize_domain

        assert normalize_domain("company.ghe.com") == "company.ghe.com"
        assert normalize_domain("https://company.ghe.com") == "company.ghe.com"
        assert normalize_domain("https://company.ghe.com/foo") == "company.ghe.com"
        assert normalize_domain("") is None
        assert normalize_domain("   ") is None

    def test_get_base_url_default(self) -> None:
        from nu_ai.utils.oauth.github_copilot import get_github_copilot_base_url

        assert get_github_copilot_base_url() == "https://api.individual.githubcopilot.com"

    def test_get_base_url_enterprise(self) -> None:
        from nu_ai.utils.oauth.github_copilot import get_github_copilot_base_url

        url = get_github_copilot_base_url(enterprise_domain="company.ghe.com")
        assert url == "https://copilot-api.company.ghe.com"

    def test_get_base_url_from_token(self) -> None:
        from nu_ai.utils.oauth.github_copilot import get_github_copilot_base_url

        token = "tid=123;exp=456;proxy-ep=proxy.individual.githubcopilot.com;foo=bar"
        url = get_github_copilot_base_url(token=token)
        assert url == "https://api.individual.githubcopilot.com"


# ---------------------------------------------------------------------------
# OpenAI Codex specifics
# ---------------------------------------------------------------------------


class TestOpenAICodex:
    def test_decode_jwt_valid(self) -> None:
        import base64
        import json

        # Build a simple JWT with the expected claim
        header = base64.b64encode(json.dumps({"alg": "none"}).encode()).decode()
        payload = base64.b64encode(
            json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acc123"}}).encode()
        ).decode()
        sig = base64.b64encode(b"sig").decode()
        token = f"{header}.{payload}.{sig}"

        from nu_ai.utils.oauth.openai_codex import _get_account_id

        assert _get_account_id(token) == "acc123"

    def test_decode_jwt_no_claim(self) -> None:
        import base64
        import json

        header = base64.b64encode(json.dumps({"alg": "none"}).encode()).decode()
        payload = base64.b64encode(json.dumps({"sub": "user"}).encode()).decode()
        sig = base64.b64encode(b"sig").decode()
        token = f"{header}.{payload}.{sig}"

        from nu_ai.utils.oauth.openai_codex import _get_account_id

        assert _get_account_id(token) is None

    def test_decode_jwt_invalid(self) -> None:
        from nu_ai.utils.oauth.openai_codex import _get_account_id

        assert _get_account_id("not-a-jwt") is None
        assert _get_account_id("") is None


# ---------------------------------------------------------------------------
# Provider interface conformance
# ---------------------------------------------------------------------------


class TestProviderConformance:
    """Verify all providers expose the expected interface."""

    def test_all_providers_have_required_properties(self) -> None:
        from nu_ai.utils.oauth import get_oauth_providers

        for provider in get_oauth_providers():
            assert isinstance(provider.id, str)
            assert len(provider.id) > 0
            assert isinstance(provider.name, str)
            assert len(provider.name) > 0
            assert isinstance(provider.uses_callback_server, bool)
            assert callable(provider.login)
            assert callable(provider.refresh_token)
            assert callable(provider.get_api_key)
            assert callable(provider.modify_models)

    def test_provider_ids_are_unique(self) -> None:
        from nu_ai.utils.oauth import get_oauth_providers

        providers = get_oauth_providers()
        ids = [p.id for p in providers]
        assert len(ids) == len(set(ids))

    def test_anthropic_get_api_key(self) -> None:
        from nu_ai.utils.oauth import get_oauth_provider

        provider = get_oauth_provider("anthropic")
        assert provider is not None
        creds = OAuthCredentials(refresh="r", access="sk-ant-xxx", expires=0)
        assert provider.get_api_key(creds) == "sk-ant-xxx"

    def test_openai_codex_get_api_key(self) -> None:
        from nu_ai.utils.oauth import get_oauth_provider

        provider = get_oauth_provider("openai-codex")
        assert provider is not None
        creds = OAuthCredentials(refresh="r", access="oacc-xxx", expires=0)
        assert provider.get_api_key(creds) == "oacc-xxx"

    def test_gemini_cli_get_api_key_is_json(self) -> None:
        import json

        from nu_ai.utils.oauth import get_oauth_provider

        provider = get_oauth_provider("google-gemini-cli")
        assert provider is not None
        creds = OAuthCredentials(refresh="r", access="ya29.xxx", expires=0, extra_data={"projectId": "my-proj"})
        key = provider.get_api_key(creds)
        parsed = json.loads(key)
        assert parsed["token"] == "ya29.xxx"
        assert parsed["projectId"] == "my-proj"

    def test_antigravity_get_api_key_is_json(self) -> None:
        import json

        from nu_ai.utils.oauth import get_oauth_provider

        provider = get_oauth_provider("google-antigravity")
        assert provider is not None
        creds = OAuthCredentials(refresh="r", access="ya29.yyy", expires=0, extra_data={"projectId": "ag-proj"})
        key = provider.get_api_key(creds)
        parsed = json.loads(key)
        assert parsed["token"] == "ya29.yyy"
        assert parsed["projectId"] == "ag-proj"
