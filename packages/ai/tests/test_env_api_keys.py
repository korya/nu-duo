"""Tests for pi_ai.env_api_keys.

Ported from the documented behaviour of
``packages/ai/src/env-api-keys.ts`` — upstream has no dedicated test file.
Verifies every mapped provider plus the special cases (GitHub Copilot,
Anthropic OAuth precedence, Vertex ADC, Bedrock multi-credential).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pi_ai.env_api_keys import get_env_api_key

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    # Strip every variable the module inspects so tests don't leak real
    # credentials from the host environment.
    for var in (
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
        "AI_GATEWAY_API_KEY",
        "ZAI_API_KEY",
        "MISTRAL_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_CN_API_KEY",
        "HF_TOKEN",
        "OPENCODE_API_KEY",
        "KIMI_API_KEY",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_OAUTH_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(var, raising=False)


class TestEnvMap:
    @pytest.mark.parametrize(
        ("provider", "env_var"),
        [
            ("openai", "OPENAI_API_KEY"),
            ("azure-openai-responses", "AZURE_OPENAI_API_KEY"),
            ("google", "GEMINI_API_KEY"),
            ("groq", "GROQ_API_KEY"),
            ("cerebras", "CEREBRAS_API_KEY"),
            ("xai", "XAI_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("vercel-ai-gateway", "AI_GATEWAY_API_KEY"),
            ("zai", "ZAI_API_KEY"),
            ("mistral", "MISTRAL_API_KEY"),
            ("minimax", "MINIMAX_API_KEY"),
            ("minimax-cn", "MINIMAX_CN_API_KEY"),
            ("huggingface", "HF_TOKEN"),
            ("opencode", "OPENCODE_API_KEY"),
            ("opencode-go", "OPENCODE_API_KEY"),
            ("kimi-coding", "KIMI_API_KEY"),
        ],
    )
    def test_mapped_provider(self, monkeypatch: pytest.MonkeyPatch, provider: str, env_var: str) -> None:
        monkeypatch.setenv(env_var, "secret-key")
        assert get_env_api_key(provider) == "secret-key"

    def test_missing_returns_none(self) -> None:
        assert get_env_api_key("openai") is None

    def test_unknown_provider_returns_none(self) -> None:
        assert get_env_api_key("completely-unknown") is None


class TestGithubCopilot:
    def test_copilot_github_token_preferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "primary")
        monkeypatch.setenv("GH_TOKEN", "secondary")
        monkeypatch.setenv("GITHUB_TOKEN", "tertiary")
        assert get_env_api_key("github-copilot") == "primary"

    def test_gh_token_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GH_TOKEN", "secondary")
        assert get_env_api_key("github-copilot") == "secondary"

    def test_github_token_final_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tertiary")
        assert get_env_api_key("github-copilot") == "tertiary"


class TestAnthropic:
    def test_oauth_token_preferred_over_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "oauth")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "api")
        assert get_env_api_key("anthropic") == "oauth"

    def test_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "api")
        assert get_env_api_key("anthropic") == "api"


class TestGoogleVertex:
    def test_explicit_api_key_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "explicit")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        assert get_env_api_key("google-vertex") == "explicit"

    def test_adc_present_returns_placeholder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text("{}")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        assert get_env_api_key("google-vertex") == "<authenticated>"

    def test_adc_missing_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No GOOGLE_CLOUD_API_KEY, no ADC file, no env → None.
        # Point GOOGLE_APPLICATION_CREDENTIALS at a non-existent path to
        # deterministically disable discovery regardless of host state.
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/creds.json")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        assert get_env_api_key("google-vertex") is None

    def test_missing_project_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text("{}")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        assert get_env_api_key("google-vertex") is None


class TestAmazonBedrock:
    @pytest.mark.parametrize(
        "env",
        [
            {"AWS_PROFILE": "default"},
            {"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"},
            {"AWS_BEARER_TOKEN_BEDROCK": "tok"},
            {"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/creds"},
            {"AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://169.254.170.2/creds"},
            {"AWS_WEB_IDENTITY_TOKEN_FILE": "/token"},
        ],
    )
    def test_any_credential_source_returns_placeholder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert get_env_api_key("amazon-bedrock") == "<authenticated>"

    def test_access_key_without_secret_not_authenticated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
        # Without the matching secret → no auth.
        assert get_env_api_key("amazon-bedrock") is None

    def test_no_credentials_returns_none(self) -> None:
        assert get_env_api_key("amazon-bedrock") is None
