"""Register nu_ai's built-in API providers.

Port of ``packages/ai/src/providers/register-builtins.ts``. Registers
every built-in provider into the global API registry on import.

Registered providers:

* ``anthropic-messages`` — Anthropic Messages API
* ``openai-completions`` — OpenAI Chat Completions (+ Ollama, Groq, etc.)
* ``openai-responses`` — OpenAI Responses API (``/v1/responses``)
* ``openai-codex-responses`` — OpenAI Codex (ChatGPT backend API)
* ``azure-openai-responses`` — Azure OpenAI Responses
* ``google-generative-ai`` — Google Gemini
* ``google-vertex-ai`` — Google Vertex AI
* ``google-gemini-cli`` — Google Gemini CLI variant
* ``amazon-bedrock`` — Amazon Bedrock (Converse API)
* ``mistral`` — Mistral AI
"""

from __future__ import annotations

from nu_ai.api_registry import ApiProvider, register_api_provider

_BUILTIN_SOURCE_ID = "nu-ai/builtins"


def _register(api: str, stream_mod: str, stream_fn: str, simple_fn: str | None = None) -> None:
    """Lazy-register a provider by importing only when needed."""
    import importlib

    mod = importlib.import_module(stream_mod)
    stream = getattr(mod, stream_fn)
    simple = getattr(mod, simple_fn) if simple_fn else stream
    register_api_provider(
        ApiProvider(api=api, stream=stream, stream_simple=simple),  # type: ignore[arg-type]
        source_id=_BUILTIN_SOURCE_ID,
    )


def register_builtin_providers() -> None:
    """Install every built-in provider into the global API registry."""

    # Anthropic
    _register(
        "anthropic-messages",
        "nu_ai.providers.anthropic",
        "stream_anthropic",
    )

    # OpenAI Chat Completions
    _register(
        "openai-completions",
        "nu_ai.providers.openai_completions",
        "stream_openai_completions",
        "stream_simple_openai_completions",
    )

    # OpenAI Responses
    _register(
        "openai-responses",
        "nu_ai.providers.openai_responses",
        "stream_openai_responses",
        "stream_simple_openai_responses",
    )

    # OpenAI Codex Responses
    _register(
        "openai-codex-responses",
        "nu_ai.providers.openai_codex_responses",
        "stream_openai_codex_responses",
        "stream_simple_openai_codex_responses",
    )

    # Azure OpenAI Responses
    _register(
        "azure-openai-responses",
        "nu_ai.providers.azure_openai_responses",
        "stream_azure_openai_responses",
        "stream_simple_azure_openai_responses",
    )

    # Google Generative AI
    _register(
        "google-generative-ai",
        "nu_ai.providers.google",
        "stream_google",
        "stream_simple_google",
    )

    # Google Vertex AI
    _register(
        "google-vertex-ai",
        "nu_ai.providers.google_vertex",
        "stream_google_vertex",
        "stream_simple_google_vertex",
    )

    # Google Gemini CLI
    _register(
        "google-gemini-cli",
        "nu_ai.providers.google_gemini_cli",
        "stream_google_gemini_cli",
        "stream_simple_google_gemini_cli",
    )

    # Amazon Bedrock
    _register(
        "amazon-bedrock",
        "nu_ai.providers.amazon_bedrock",
        "stream_bedrock",
        "stream_simple_bedrock",
    )

    # Mistral
    _register(
        "mistral",
        "nu_ai.providers.mistral",
        "stream_mistral",
        "stream_simple_mistral",
    )


# Side-effect: register on import
register_builtin_providers()

__all__ = ["register_builtin_providers"]
