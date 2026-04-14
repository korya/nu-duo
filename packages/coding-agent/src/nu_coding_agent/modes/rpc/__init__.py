"""RPC mode: headless operation with JSON stdin/stdout protocol.

Used for embedding the agent in other applications.
Receives commands as JSON on stdin, outputs events and responses as JSON on stdout.

Protocol:
- Commands: JSON objects with ``type`` field, optional ``id`` for correlation
- Responses: JSON objects with ``type: "response"``, ``command``, ``success``, and optional ``data``/``error``
- Events: AgentSessionEvent objects streamed as they occur
- Extension UI: Extension UI requests are emitted, client responds with extension_ui_response
"""

from __future__ import annotations

from nu_coding_agent.modes.rpc.jsonl import attach_jsonl_line_reader, serialize_json_line
from nu_coding_agent.modes.rpc.rpc_client import ModelInfo, RpcClient, RpcClientError, RpcClientOptions
from nu_coding_agent.modes.rpc.rpc_mode import run_rpc_mode
from nu_coding_agent.modes.rpc.rpc_types import (
    RpcCommand,
    RpcExtensionUIRequest,
    RpcExtensionUIResponse,
    RpcResponse,
    RpcSessionState,
    RpcSlashCommand,
)

__all__ = [
    "ModelInfo",
    "RpcClient",
    "RpcClientError",
    "RpcClientOptions",
    "RpcCommand",
    "RpcExtensionUIRequest",
    "RpcExtensionUIResponse",
    "RpcResponse",
    "RpcSessionState",
    "RpcSlashCommand",
    "attach_jsonl_line_reader",
    "run_rpc_mode",
    "serialize_json_line",
]
