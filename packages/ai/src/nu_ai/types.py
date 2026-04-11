"""Core types for nu_ai.

Direct port of ``packages/ai/src/types.ts`` from the upstream TypeScript
monorepo. Conventions:

* **Python attributes are snake_case**, but **wire format is camelCase** — the
  original TS field names. All types inherit from :class:`_Model` which sets
  ``populate_by_name=True`` and a global ``alias_generator=to_camel`` so that
  ``model_dump(by_alias=True)`` produces JSON byte-compatible with the
  upstream TS output (critical for the JSONL session format).
* Discriminated unions use the ``type`` tag for content and the ``role`` tag
  for messages, matching the TS ``interface`` shape.
* ``extra="forbid"`` everywhere; unknown fields are a bug, not a feature.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# ---------------------------------------------------------------------------
# Base model — shared config for every type in nu_ai.
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    """Base model: snake_case attrs ↔ camelCase wire format, forbid extras."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        # Keep validation strict but allow the common case of constructing
        # nested models from plain dicts.
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# API / provider / thinking taxonomies.
# ---------------------------------------------------------------------------

type KnownApi = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-gemini-cli",
    "google-vertex",
]
"""Built-in API identifiers. Extensions may use arbitrary strings."""

type Api = str
"""An API identifier — a :data:`KnownApi` literal or a custom string."""

type KnownProvider = Literal[
    "amazon-bedrock",
    "anthropic",
    "google",
    "google-gemini-cli",
    "google-antigravity",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "mistral",
    "minimax",
    "minimax-cn",
    "huggingface",
    "opencode",
    "opencode-go",
    "kimi-coding",
]

type Provider = str
"""A provider identifier — a :data:`KnownProvider` literal or a custom string."""

type ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]

type CacheRetention = Literal["none", "short", "long"]

type Transport = Literal["sse", "websocket", "auto"]

type StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


class ThinkingBudgets(_Model):
    """Token budgets for each thinking level (token-based providers only)."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


# ---------------------------------------------------------------------------
# Content blocks.
# ---------------------------------------------------------------------------


class TextContent(_Model):
    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = None


class ThinkingContent(_Model):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool | None = None


class ImageContent(_Model):
    type: Literal["image"] = "image"
    data: str
    """Base64-encoded image data."""
    mime_type: str
    """e.g. ``"image/jpeg"``, ``"image/png"``."""


class ToolCall(_Model):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None
    """Google-specific: opaque signature for reusing thought context."""


type Content = Annotated[
    TextContent | ThinkingContent | ImageContent | ToolCall,
    Field(discriminator="type"),
]
"""All content-block variants, dispatched on the ``type`` tag."""

type UserContent = Annotated[
    TextContent | ImageContent,
    Field(discriminator="type"),
]

type AssistantContent = Annotated[
    TextContent | ThinkingContent | ToolCall,
    Field(discriminator="type"),
]

type ToolResultContent = Annotated[
    TextContent | ImageContent,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Usage and cost.
# ---------------------------------------------------------------------------


class Cost(_Model):
    input: float
    output: float
    cache_read: float
    cache_write: float
    total: float


class Usage(_Model):
    input: int
    output: int
    cache_read: int
    cache_write: int
    total_tokens: int
    cost: Cost


# ---------------------------------------------------------------------------
# Messages.
# ---------------------------------------------------------------------------


class UserMessage(_Model):
    role: Literal["user"] = "user"
    content: str | list[UserContent]
    timestamp: int
    """Unix timestamp in milliseconds."""


class AssistantMessage(_Model):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent]
    api: Api
    provider: Provider
    model: str
    response_id: str | None = None
    """Provider-specific response/message identifier when the upstream API exposes one."""
    usage: Usage
    stop_reason: StopReason
    error_message: str | None = None
    timestamp: int


class ToolResultMessage(_Model):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContent]
    details: Any = None
    is_error: bool
    timestamp: int


type Message = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]


# ---------------------------------------------------------------------------
# Tools and context.
# ---------------------------------------------------------------------------


class Tool(_Model):
    """An LLM tool definition — name, description, and JSON-schema parameters.

    The upstream TS version is generic over ``TParameters extends TSchema``
    (TypeBox). In Python we store the JSON Schema as a plain ``dict`` — pi_agent
    converts Pydantic models to JSON Schema via ``model_json_schema()`` before
    populating this field.
    """

    name: str
    description: str
    parameters: dict[str, Any]


class Context(_Model):
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] | None = None


# ---------------------------------------------------------------------------
# Stream options.
# ---------------------------------------------------------------------------


class OpenRouterMaxPrice(_Model):
    prompt: float | str | None = None
    completion: float | str | None = None
    image: float | str | None = None
    audio: float | str | None = None
    request: float | str | None = None


class OpenRouterThroughput(_Model):
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p99: float | None = None


class OpenRouterSort(_Model):
    by: str | None = None
    partition: str | None = None


class OpenRouterRouting(_Model):
    """OpenRouter provider routing preferences.

    Sent as the ``provider`` field in the OpenRouter API request body.
    Field names are preserved in ``snake_case`` on the wire (OpenRouter's own
    convention), so this class opts out of the camelCase alias generator.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    data_collection: Literal["deny", "allow"] | None = None
    zdr: bool | None = None
    enforce_distillable_text: bool | None = None
    order: list[str] | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    quantizations: list[str] | None = None
    sort: str | OpenRouterSort | None = None
    max_price: OpenRouterMaxPrice | None = None
    preferred_min_throughput: float | OpenRouterThroughput | None = None
    preferred_max_latency: float | OpenRouterThroughput | None = None


class VercelGatewayRouting(_Model):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    only: list[str] | None = None
    order: list[str] | None = None


class StreamOptions(_Model):
    temperature: float | None = None
    max_tokens: int | None = None
    # signal: AbortSignal — not serialized. Exposed separately on the stream API.
    api_key: str | None = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] | None = None
    # onPayload is a callback; not serialized. Passed separately in Python.


class SimpleStreamOptions(StreamOptions):
    reasoning: ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = None


# ---------------------------------------------------------------------------
# Provider-specific stream option subtypes.
# ---------------------------------------------------------------------------


type AnthropicEffort = Literal["low", "medium", "high", "max"]

type AnthropicToolChoice = Literal["auto", "any", "none"] | dict[str, str]
"""Either a bare string choice or ``{"type": "tool", "name": <tool_name>}``."""


class AnthropicOptions(StreamOptions):
    """Anthropic-specific extension of :class:`StreamOptions`.

    Mirrors ``AnthropicOptions`` from ``packages/ai/src/providers/anthropic.ts``.
    The ``client`` field is intentionally omitted — in Python it's an
    out-of-band parameter to :func:`nu_ai.providers.anthropic.stream_anthropic`
    since :class:`anthropic.Anthropic` is not serializable into a Pydantic model.
    """

    thinking_enabled: bool | None = None
    """Enable extended thinking.

    For Opus 4.6 / Sonnet 4.6: adaptive thinking (model decides when/how much).
    For older models: budget-based thinking via ``thinking_budget_tokens``.
    """

    thinking_budget_tokens: int | None = None
    """Token budget for extended thinking (older models only)."""

    effort: AnthropicEffort | None = None
    """Effort level for adaptive thinking on Opus 4.6 / Sonnet 4.6."""

    interleaved_thinking: bool | None = None

    tool_choice: AnthropicToolChoice | None = None


# ---------------------------------------------------------------------------
# OpenAI Chat Completions options.
# ---------------------------------------------------------------------------

type OpenAIToolChoiceString = Literal["auto", "none", "required"]

type OpenAICompletionsToolChoice = OpenAIToolChoiceString | dict[str, Any]
"""Either a bare string choice or ``{"type": "function", "function": {"name": ...}}``."""


class OpenAICompletionsOptions(StreamOptions):
    """OpenAI Chat Completions-specific stream options.

    Mirrors ``OpenAICompletionsOptions`` from
    ``packages/ai/src/providers/openai-completions.ts``. Also applies to any
    provider that speaks the OpenAI Chat Completions wire format (Ollama,
    LMStudio, Groq, etc.) since they share the same endpoint shape.
    """

    tool_choice: OpenAICompletionsToolChoice | None = None
    reasoning_effort: ThinkingLevel | None = None


# ---------------------------------------------------------------------------
# Google Generative AI options.
# ---------------------------------------------------------------------------

type GoogleToolChoice = Literal["auto", "none", "any"]

type GoogleThinkingLevel = Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]


class GoogleThinkingOptions(_Model):
    """Nested ``thinking`` block on :class:`GoogleOptions`."""

    enabled: bool
    budget_tokens: int | None = None
    """``-1`` for dynamic, ``0`` to disable. Ignored if :attr:`level` is set."""
    level: GoogleThinkingLevel | None = None


class GoogleOptions(StreamOptions):
    """Google Generative AI (Gemini) stream options.

    Mirrors ``GoogleOptions`` from ``packages/ai/src/providers/google.ts``.
    """

    tool_choice: GoogleToolChoice | None = None
    thinking: GoogleThinkingOptions | None = None


# ---------------------------------------------------------------------------
# OpenAI-compat overrides (used on ``Model.compat``).
# ---------------------------------------------------------------------------


class OpenAICompletionsCompat(_Model):
    supports_store: bool | None = None
    supports_developer_role: bool | None = None
    supports_reasoning_effort: bool | None = None
    reasoning_effort_map: dict[ThinkingLevel, str] | None = None
    supports_usage_in_streaming: bool | None = None
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] | None = None
    requires_tool_result_name: bool | None = None
    requires_assistant_after_tool_result: bool | None = None
    requires_thinking_as_text: bool | None = None
    thinking_format: Literal["openai", "openrouter", "zai", "qwen", "qwen-chat-template"] | None = None
    open_router_routing: OpenRouterRouting | None = None
    vercel_gateway_routing: VercelGatewayRouting | None = None
    zai_tool_stream: bool | None = None
    supports_strict_mode: bool | None = None


class OpenAIResponsesCompat(_Model):
    """Reserved for future use (parity with upstream)."""


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------

TApi = TypeVar("TApi", bound=str)


class ModelCost(_Model):
    input: float
    """$/million tokens."""
    output: float
    cache_read: float
    cache_write: float


class Model(_Model):
    """A provider-qualified LLM model.

    Python intentionally does not reproduce the TS conditional type on
    ``compat`` — ``compat`` is simply ``OpenAICompletionsCompat``,
    ``OpenAIResponsesCompat`` or ``None``, and providers decide which is
    applicable based on :attr:`api`.
    """

    id: str
    name: str
    api: Api
    provider: Provider
    base_url: str
    reasoning: bool
    input: list[Literal["text", "image"]]
    cost: ModelCost
    context_window: int
    max_tokens: int
    headers: dict[str, str] | None = None
    compat: OpenAICompletionsCompat | OpenAIResponsesCompat | None = None


# ---------------------------------------------------------------------------
# Assistant message event protocol (streaming).
# ---------------------------------------------------------------------------


class _EventBase(_Model):
    """Base for every ``AssistantMessageEvent`` variant."""


class StartEvent(_EventBase):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStartEvent(_EventBase):
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class TextDeltaEvent(_EventBase):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEndEvent(_EventBase):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(_EventBase):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class ThinkingDeltaEvent(_EventBase):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(_EventBase):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(_EventBase):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class ToolCallDeltaEvent(_EventBase):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(_EventBase):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class DoneEvent(_EventBase):
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class ErrorEvent(_EventBase):
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


type AssistantMessageEvent = Annotated[
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="type"),
]


__all__ = [
    "AnthropicEffort",
    "AnthropicOptions",
    "AnthropicToolChoice",
    "Api",
    "AssistantContent",
    "AssistantMessage",
    "AssistantMessageEvent",
    "CacheRetention",
    "Content",
    "Context",
    "Cost",
    "DoneEvent",
    "ErrorEvent",
    "GoogleOptions",
    "GoogleThinkingLevel",
    "GoogleThinkingOptions",
    "GoogleToolChoice",
    "ImageContent",
    "KnownApi",
    "KnownProvider",
    "Message",
    "Model",
    "ModelCost",
    "OpenAICompletionsCompat",
    "OpenAICompletionsOptions",
    "OpenAICompletionsToolChoice",
    "OpenAIResponsesCompat",
    "OpenAIToolChoiceString",
    "OpenRouterRouting",
    "Provider",
    "SimpleStreamOptions",
    "StartEvent",
    "StopReason",
    "StreamOptions",
    "TextContent",
    "TextDeltaEvent",
    "TextEndEvent",
    "TextStartEvent",
    "ThinkingBudgets",
    "ThinkingContent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ThinkingLevel",
    "ThinkingStartEvent",
    "Tool",
    "ToolCall",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
    "ToolResultContent",
    "ToolResultMessage",
    "Transport",
    "Usage",
    "UserContent",
    "UserMessage",
    "VercelGatewayRouting",
]
