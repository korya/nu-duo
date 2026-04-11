"""Environment-variable credential discovery.

Port of ``packages/ai/src/env-api-keys.ts``. Returns the API key (or the
``"<authenticated>"`` placeholder used by providers that use ambient
credentials) for a known provider, or ``None`` if no credential is
configured.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
    "zai": "ZAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "huggingface": "HF_TOKEN",
    "opencode": "OPENCODE_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
    "kimi-coding": "KIMI_API_KEY",
}


def _has_vertex_adc_credentials() -> bool:
    """Return ``True`` iff ADC credentials are discoverable.

    Checks ``GOOGLE_APPLICATION_CREDENTIALS`` first (standard override), then
    falls back to the default gcloud location
    ``~/.config/gcloud/application_default_credentials.json``.
    """
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_path:
        return Path(gac_path).exists()
    default_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return default_path.exists()


def get_env_api_key(provider: str) -> str | None:
    """Return the configured API key (or authentication marker) for ``provider``.

    Providers that rely on ambient credentials (google-vertex ADC,
    amazon-bedrock) return the literal string ``"<authenticated>"`` when
    discovery succeeds — the string is opaque, downstream callers only
    branch on presence/absence.
    """
    if provider == "github-copilot":
        return os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

    if provider == "anthropic":
        # ANTHROPIC_OAUTH_TOKEN takes precedence over ANTHROPIC_API_KEY.
        return os.environ.get("ANTHROPIC_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")

    if provider == "google-vertex":
        if os.environ.get("GOOGLE_CLOUD_API_KEY"):
            return os.environ["GOOGLE_CLOUD_API_KEY"]

        has_credentials = _has_vertex_adc_credentials()
        has_project = bool(os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT"))
        has_location = bool(os.environ.get("GOOGLE_CLOUD_LOCATION"))

        if has_credentials and has_project and has_location:
            return "<authenticated>"
        return None

    if provider == "amazon-bedrock":
        if (
            os.environ.get("AWS_PROFILE")
            or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
        ):
            return "<authenticated>"
        return None

    env_var = _ENV_MAP.get(provider)
    return os.environ.get(env_var) if env_var else None


__all__ = ["get_env_api_key"]
