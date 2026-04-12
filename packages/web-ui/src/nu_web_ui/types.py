"""Pydantic models for nu_web_ui.

Mirrors the TypeScript shape from storage/types.ts and stores/custom-providers-store.ts.
camelCase aliases are used wherever the frontend expects them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Cost / Usage (mirrors upstream SessionMetadata.usage)
# ---------------------------------------------------------------------------


class CostBreakdown(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(0.0, alias="cacheRead")
    cache_write: float = Field(0.0, alias="cacheWrite")
    total: float = 0.0


class UsageStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input: int = 0
    output: int = 0
    cache_read: int = Field(0, alias="cacheRead")
    cache_write: int = Field(0, alias="cacheWrite")
    total_tokens: int = Field(0, alias="totalTokens")
    cost: CostBreakdown = Field(default_factory=CostBreakdown)


# ---------------------------------------------------------------------------
# ModelCost / ModelInfo
# ---------------------------------------------------------------------------


class ModelCostInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(0.0, alias="cacheRead")
    cache_write: float = Field(0.0, alias="cacheWrite")


class ModelInfo(BaseModel):
    """Represents a discovered or configured LLM model."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    api: str
    provider: str
    base_url: str = Field("", alias="baseUrl")
    reasoning: bool = False
    input: list[str] = Field(default_factory=lambda: ["text"])
    cost: ModelCostInfo = Field(default_factory=ModelCostInfo)
    context_window: int = Field(8192, alias="contextWindow")
    max_tokens: int = Field(4096, alias="maxTokens")


# ---------------------------------------------------------------------------
# SessionInfo (= SessionMetadata in the TS source)
# ---------------------------------------------------------------------------

type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


class SessionInfo(BaseModel):
    """Lightweight session metadata — matches TS SessionMetadata."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str = ""
    created_at: str = Field(..., alias="createdAt")
    last_modified: str = Field(..., alias="lastModified")
    message_count: int = Field(0, alias="messageCount")
    usage: UsageStats = Field(default_factory=UsageStats)
    thinking_level: ThinkingLevel = Field("off", alias="thinkingLevel")
    preview: str = ""


# ---------------------------------------------------------------------------
# Message (mirrors AgentMessage / TS message shapes at a generic level)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a session transcript.

    Kept deliberately open so any AgentMessage variant round-trips cleanly
    as JSON without losing fields.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    role: str
    content: Any = None


# ---------------------------------------------------------------------------
# SessionData (full session including messages)
# ---------------------------------------------------------------------------


class SessionData(BaseModel):
    """Full session data — matches TS SessionData."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str = ""
    model: ModelInfo | None = None
    thinking_level: ThinkingLevel = Field("off", alias="thinkingLevel")
    messages: list[Any] = Field(default_factory=list)
    created_at: str = Field(..., alias="createdAt")
    last_modified: str = Field(..., alias="lastModified")


# ---------------------------------------------------------------------------
# ProviderKey
# ---------------------------------------------------------------------------


class ProviderKey(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider: str
    key: str


# ---------------------------------------------------------------------------
# CustomProvider
# ---------------------------------------------------------------------------

type CustomProviderType = Literal[
    "ollama",
    "llama.cpp",
    "vllm",
    "lmstudio",
    "openai-completions",
    "openai-responses",
    "anthropic-messages",
]


class CustomProvider(BaseModel):
    """Custom LLM provider — matches TS CustomProvider interface."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    type: str
    base_url: str = Field(..., alias="baseUrl")
    api_key: str | None = Field(None, alias="apiKey")
    models: list[ModelInfo] | None = None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class ProxySettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    url: str = ""


class Settings(BaseModel):
    """Application settings persisted in the settings store."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    theme: str = "system"
    proxy: ProxySettings = Field(default_factory=ProxySettings)


__all__ = [
    "CostBreakdown",
    "CustomProvider",
    "CustomProviderType",
    "Message",
    "ModelCostInfo",
    "ModelInfo",
    "ProviderKey",
    "ProxySettings",
    "SessionData",
    "SessionInfo",
    "Settings",
    "ThinkingLevel",
    "UsageStats",
]
