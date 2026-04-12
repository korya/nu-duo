"""CORS proxy — forwards LLM API requests from the browser to providers.

The frontend sends a POST to /api/proxy with a JSON body containing:
  - ``url``     — target URL (string, required)
  - ``method``  — HTTP method (string, default "POST")
  - ``headers`` — dict of request headers (object, optional)
  - ``body``    — request body (any JSON value, optional)

The proxy response mirrors the upstream status code and headers, supporting
both regular JSON responses and streaming (SSE / chunked transfer).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Headers we strip from the proxied response to avoid confusing the client.
_HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-encoding",  # httpx decompresses; let FastAPI re-encode if needed
    ]
)


async def proxy_request(request: Request) -> Response:
    """Handle a proxy request from the frontend.

    Reads the target URL, method, headers, and body from the JSON payload,
    forwards them to the target, and streams the response back.
    """
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        return Response(content="Invalid JSON body", status_code=400)

    target_url: str | None = payload.get("url")
    if not target_url:
        return Response(content="Missing 'url' field", status_code=400)

    method: str = str(payload.get("method", "POST")).upper()
    forward_headers: dict[str, str] = dict(payload.get("headers") or {})
    body: Any = payload.get("body")

    # Serialize body back to JSON bytes if present.
    body_bytes: bytes | None = None
    if body is not None:
        import json

        body_bytes = json.dumps(body).encode()
        forward_headers.setdefault("Content-Type", "application/json")

    logger.debug("Proxying %s %s", method, target_url)

    async def _stream_response() -> Any:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(None)) as client,
            client.stream(
                method,
                target_url,
                headers=forward_headers,
                content=body_bytes,
            ) as upstream,
        ):
            # Yield status line as a special first chunk so the caller can
            # reconstruct the response — standard SSE doesn't carry this,
            # but the proxy contract here is streaming raw bytes.
            async for chunk in upstream.aiter_raw():
                yield chunk

    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as probe_client:
        try:
            upstream_resp = await probe_client.request(
                method,
                target_url,
                headers=forward_headers,
                content=body_bytes,
            )
        except httpx.RequestError as exc:
            logger.warning("Proxy upstream error: %s", exc)
            return Response(content=f"Upstream error: {exc}", status_code=502)

    # Filter hop-by-hop headers.
    response_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in _HOP_BY_HOP}

    content_type = upstream_resp.headers.get("content-type", "")
    is_streaming = "text/event-stream" in content_type or "application/octet-stream" in content_type

    if is_streaming:
        # Re-issue the request as a streaming call so we don't buffer.
        async def _iter() -> Any:
            async with (
                httpx.AsyncClient(timeout=httpx.Timeout(None)) as sc,
                sc.stream(
                    method,
                    target_url,
                    headers=forward_headers,
                    content=body_bytes,
                ) as sr,
            ):
                async for chunk in sr.aiter_raw():
                    yield chunk

        return StreamingResponse(
            _iter(),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type=content_type,
        )

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=response_headers,
        media_type=content_type or "application/octet-stream",
    )


__all__ = ["proxy_request"]
