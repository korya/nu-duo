"""Strict JSONL protocol implementation.

Port of ``modes/rpc/jsonl.ts``.

Framing is LF-only. Payload strings may contain other Unicode separators such as
U+2028 and U+2029. Clients must split records on ``\\n`` only.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def serialize_json_line(value: Any) -> str:
    """Serialize a single strict JSONL record.

    Returns JSON string terminated by ``\\n``.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


async def attach_jsonl_line_reader(
    on_line: Callable[[str], Any],
    *,
    stream: asyncio.StreamReader | None = None,
) -> Callable[[], None]:
    """Attach an LF-only JSONL reader to an async stream.

    If ``stream`` is None, reads from stdin via ``asyncio.get_event_loop().connect_read_pipe``.

    Returns a detach function that stops reading.
    """
    cancelled = asyncio.Event()

    if stream is None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    else:
        reader = stream

    async def _read_loop() -> None:
        buffer = ""
        while not cancelled.is_set():
            try:
                chunk = await reader.read(8192)
            except (asyncio.CancelledError, ConnectionResetError):
                break
            if not chunk:
                # EOF — emit any remaining buffer
                if buffer:
                    line = buffer.rstrip("\r")
                    if line:
                        result = on_line(line)
                        if asyncio.iscoroutine(result):
                            await result
                break

            buffer += chunk.decode("utf-8", errors="replace")

            while True:
                idx = buffer.find("\n")
                if idx == -1:
                    break
                line = buffer[:idx]
                buffer = buffer[idx + 1 :]
                # Strip trailing \r (CRLF input)
                if line.endswith("\r"):
                    line = line[:-1]
                if line:
                    result = on_line(line)
                    if asyncio.iscoroutine(result):
                        await result

    task = asyncio.create_task(_read_loop())

    def detach() -> None:
        cancelled.set()
        task.cancel()

    return detach
