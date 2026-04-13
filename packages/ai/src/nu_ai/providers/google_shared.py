"""Shared utilities for Google Generative AI, Gemini CLI and Vertex providers.

Direct port of ``packages/ai/src/providers/google-shared.ts``. These helpers
are consumed by both :mod:`nu_ai.providers.google_vertex` and
:mod:`nu_ai.providers.google_gemini_cli`.

The module deliberately contains no HTTP or SDK calls — it is pure Python so
it can be unit-tested without any Google SDK installed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from nu_ai.providers.transform_messages import transform_messages
from nu_ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from nu_ai.utils.sanitize_unicode import sanitize_surrogates

if TYPE_CHECKING:
    from nu_ai.types import Context, Model


# ---------------------------------------------------------------------------
# Thought-signature helpers
# ---------------------------------------------------------------------------

# Thought signatures must be base64 for Google APIs (TYPE_BYTES).
_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

# Sentinel value that tells the Gemini API to skip thought signature
# validation.  Used for unsigned function call parts (e.g. replayed from
# providers without thought signatures).
# See: https://ai.google.dev/gemini-api/docs/thought-signatures
SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def is_thinking_part(part: Any) -> bool:
    """Return ``True`` when *part* is a Gemini thinking block.

    Protocol note: ``thought_signature`` can appear on *any* part type — it
    does **not** indicate the part is thinking content.  Only ``thought: true``
    does.
    """
    if isinstance(part, dict):
        return part.get("thought") is True
    return getattr(part, "thought", None) is True


def retain_thought_signature(
    existing: str | None,
    incoming: str | None,
) -> str | None:
    """Preserve the last non-empty signature across streamed deltas.

    Some backends emit ``thought_signature`` only on the first delta; later
    deltas omit it.  This prevents overwriting a valid signature with ``None``
    within the same streamed block.
    """
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE_PATTERN.match(signature))


def _resolve_thought_signature(
    is_same_provider_and_model: bool,
    signature: str | None,
) -> str | None:
    """Return *signature* only if it comes from the same model and is valid base64."""
    if is_same_provider_and_model and _is_valid_thought_signature(signature):
        return signature
    return None


# ---------------------------------------------------------------------------
# Tool call ID requirements
# ---------------------------------------------------------------------------


def requires_tool_call_id(model_id: str) -> bool:
    """Whether *model_id* requires explicit IDs on function calls/responses.

    Claude and ``gpt-oss-`` models served via Google APIs need explicit tool
    call IDs; native Gemini models rely on positional matching.
    """
    return model_id.startswith(("claude-", "gpt-oss-"))


# ---------------------------------------------------------------------------
# Model family helpers
# ---------------------------------------------------------------------------


def _get_gemini_major_version(model_id: str) -> int | None:
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    if not match:
        return None
    return int(match.group(1))


def _supports_multimodal_function_response(model_id: str) -> bool:
    major = _get_gemini_major_version(model_id)
    if major is not None:
        return major >= 3
    return True


def is_gemini_3_pro_model(model_id: str) -> bool:
    return bool(re.search(r"gemini-3(?:\.1)?-pro", model_id.lower()))


def is_gemini_3_flash_model(model_id: str) -> bool:
    return bool(re.search(r"gemini-3(?:\.1)?-flash", model_id.lower()))


def is_gemini_3_model(model_id: str) -> bool:
    return is_gemini_3_pro_model(model_id) or is_gemini_3_flash_model(model_id)


# ---------------------------------------------------------------------------
# Stop reason / tool choice mapping
# ---------------------------------------------------------------------------


def map_stop_reason(reason: str | None) -> str:
    """Map a Gemini ``FinishReason`` enum/string to nu_ai's :data:`StopReason`.

    Accepts either the bare string name (``"STOP"``, ``"MAX_TOKENS"``, …) or a
    google-genai enum value that stringifies to the same (e.g.
    ``FinishReason.STOP``).
    """
    reason_str = reason if isinstance(reason, str) else str(reason or "")
    # google-genai SDK may produce ``"FinishReason.STOP"``
    reason_str = reason_str.split(".")[-1]
    if reason_str == "STOP":
        return "stop"
    if reason_str == "MAX_TOKENS":
        return "length"
    return "error"


def map_stop_reason_string(reason: str) -> str:
    """Map a raw string finish reason (as returned by Cloud Code Assist SSE)."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"


def map_tool_choice(choice: str) -> str:
    """Map nu_ai tool choice to Gemini ``FunctionCallingConfigMode`` name."""
    if choice == "auto":
        return "AUTO"
    if choice == "none":
        return "NONE"
    if choice == "any":
        return "ANY"
    return "AUTO"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def convert_tools(
    tools: list[Tool],
    *,
    use_parameters: bool = False,
) -> list[dict[str, Any]] | None:
    """Convert nu_ai tools to Gemini function declarations.

    Uses ``parameters_json_schema`` by default (full JSON Schema support).
    Pass ``use_parameters=True`` for Cloud Code Assist with Claude models,
    where the API translates ``parameters`` into Anthropic's ``input_schema``.
    """
    if not tools:
        return None
    function_declarations: list[dict[str, Any]] = []
    for tool in tools:
        declaration: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
        }
        if use_parameters:
            declaration["parameters"] = tool.parameters
        else:
            declaration["parameters_json_schema"] = tool.parameters
        function_declarations.append(declaration)
    return [{"function_declarations": function_declarations}]


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

_ILLEGAL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _make_tool_call_id_normalizer(model: Model) -> Any:
    def normalize(tool_call_id: str, *_: object) -> str:
        if not requires_tool_call_id(model.id):
            return tool_call_id
        return _ILLEGAL_ID_CHARS.sub("_", tool_call_id)[:64]

    return normalize


def convert_messages(
    model: Model,
    context: Context,
) -> list[dict[str, Any]]:
    """Convert nu_ai messages to the Gemini ``contents`` array.

    Runs the cross-provider :func:`transform_messages` with a Gemini-aware
    tool call id normalizer, then translates each message to the
    ``{role, parts: [...]}`` shape.
    """
    contents: list[dict[str, Any]] = []
    transformed = transform_messages(context.messages, model, _make_tool_call_id_normalizer(model))

    is_gemini_3 = "gemini-3" in model.id.lower()

    for msg in transformed:
        if isinstance(msg, UserMessage):
            _append_user_message(contents, msg, model)
        elif isinstance(msg, AssistantMessage):
            _append_assistant_message(contents, msg, model, is_gemini_3=is_gemini_3)
        else:
            # ToolResultMessage
            _append_tool_result_message(contents, msg, model)

    return contents


def _append_user_message(
    contents: list[dict[str, Any]],
    msg: UserMessage,
    model: Model,
) -> None:
    if isinstance(msg.content, str):
        contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(msg.content)}]})
        return

    parts: list[dict[str, Any]] = []
    for item in msg.content:
        if isinstance(item, TextContent):
            parts.append({"text": sanitize_surrogates(item.text)})
        else:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": item.mime_type,
                        "data": item.data,
                    },
                }
            )

    if "image" not in model.input:
        parts = [p for p in parts if "text" in p]
    if not parts:
        return
    contents.append({"role": "user", "parts": parts})


def _append_assistant_message(
    contents: list[dict[str, Any]],
    msg: AssistantMessage,
    model: Model,
    *,
    is_gemini_3: bool,
) -> None:
    is_same_provider_and_model = msg.provider == model.provider and msg.model == model.id
    parts: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextContent):
            if not block.text.strip():
                continue
            thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.text_signature)
            part: dict[str, Any] = {"text": sanitize_surrogates(block.text)}
            if thought_signature:
                part["thought_signature"] = thought_signature
            parts.append(part)
        elif isinstance(block, ThinkingContent):
            if not block.thinking.strip():
                continue
            if is_same_provider_and_model:
                thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.thinking_signature)
                thinking_part: dict[str, Any] = {
                    "thought": True,
                    "text": sanitize_surrogates(block.thinking),
                }
                if thought_signature:
                    thinking_part["thought_signature"] = thought_signature
                parts.append(thinking_part)
            else:
                # Cross-model: fall back to plain text
                parts.append({"text": sanitize_surrogates(block.thinking)})
        else:
            # ToolCall
            assert isinstance(block, ToolCall)
            thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.thought_signature)
            effective_signature = thought_signature or (SKIP_THOUGHT_SIGNATURE if is_gemini_3 else None)
            function_call: dict[str, Any] = {
                "name": block.name,
                "args": block.arguments or {},
            }
            if requires_tool_call_id(model.id):
                function_call["id"] = block.id
            part_dict: dict[str, Any] = {"function_call": function_call}
            if effective_signature:
                part_dict["thought_signature"] = effective_signature
            parts.append(part_dict)

    if not parts:
        return
    contents.append({"role": "model", "parts": parts})


def _append_tool_result_message(
    contents: list[dict[str, Any]],
    msg: ToolResultMessage,
    model: Model,
) -> None:
    text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
    text_result = "\n".join(text_parts)
    image_blocks: list[ImageContent] = (
        [c for c in msg.content if isinstance(c, ImageContent)] if "image" in model.input else []
    )
    has_text = bool(text_result)
    has_images = bool(image_blocks)

    model_supports_multimodal = _supports_multimodal_function_response(model.id)
    response_value = sanitize_surrogates(text_result) if has_text else ("(see attached image)" if has_images else "")

    image_parts: list[dict[str, Any]] = [
        {
            "inline_data": {
                "mime_type": block.mime_type,
                "data": block.data,
            },
        }
        for block in image_blocks
    ]

    include_id = requires_tool_call_id(model.id)
    function_response: dict[str, Any] = {
        "name": msg.tool_name,
        "response": {"error": response_value} if msg.is_error else {"output": response_value},
    }
    if has_images and model_supports_multimodal:
        function_response["parts"] = image_parts
    if include_id:
        function_response["id"] = msg.tool_call_id
    function_response_part: dict[str, Any] = {"function_response": function_response}

    # Collapse consecutive tool results into a single user turn.
    if (
        contents
        and contents[-1]["role"] == "user"
        and any("function_response" in p for p in contents[-1].get("parts", []))
    ):
        contents[-1]["parts"].append(function_response_part)
    else:
        contents.append({"role": "user", "parts": [function_response_part]})

    # Gemini < 3: images go in a separate user turn.
    if has_images and not model_supports_multimodal:
        contents.append(
            {
                "role": "user",
                "parts": [{"text": "Tool result image:"}, *image_parts],
            }
        )


__all__ = [
    "SKIP_THOUGHT_SIGNATURE",
    "convert_messages",
    "convert_tools",
    "is_gemini_3_flash_model",
    "is_gemini_3_model",
    "is_gemini_3_pro_model",
    "is_thinking_part",
    "map_stop_reason",
    "map_stop_reason_string",
    "map_tool_choice",
    "requires_tool_call_id",
    "retain_thought_signature",
]
