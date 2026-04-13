"""Local HTTP callback server for OAuth authorization code flows.

Shared infrastructure used by Anthropic, OpenAI Codex, Gemini CLI,
and Antigravity OAuth providers.

Uses stdlib ``http.server`` in a daemon thread.
"""

from __future__ import annotations

import asyncio
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

from nu_ai.utils.oauth.oauth_page import oauth_error_html, oauth_success_html

logger = logging.getLogger(__name__)


class CallbackResult(NamedTuple):
    """Result from the OAuth callback."""

    code: str
    state: str


class CallbackServer:
    """Local HTTP server that listens for an OAuth callback.

    Parameters:
        host: Bind address (default ``127.0.0.1``).
        port: Bind port.
        path: Callback path to listen on (e.g. ``/callback``).
        expected_state: If set, validates the ``state`` parameter.
        provider_name: Provider name for HTML messages.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int,
        path: str,
        expected_state: str | None = None,
        provider_name: str = "OAuth",
    ) -> None:
        self._host = host
        self._port = port
        self._path = path
        self._expected_state = expected_state
        self._provider_name = provider_name
        self._future: asyncio.Future[CallbackResult | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self._port}{self._path}"

    async def start(self) -> None:
        """Start the callback server in a background thread."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        future: asyncio.Future[CallbackResult | None] = loop.create_future()
        self._future = future

        host = self._host
        port = self._port
        path = self._path
        expected_state = self._expected_state
        provider_name = self._provider_name

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass  # Suppress default logging

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != path:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(oauth_error_html("Callback route not found.").encode())
                    return

                params = parse_qs(parsed.query)
                error = params.get("error", [None])[0]
                code = params.get("code", [None])[0]
                state = params.get("state", [None])[0]

                if error:
                    self.send_response(400)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        oauth_error_html(
                            f"{provider_name} authentication did not complete.",
                            f"Error: {error}",
                        ).encode()
                    )
                    return

                if not code or not state:
                    self.send_response(400)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(oauth_error_html("Missing code or state parameter.").encode())
                    return

                if expected_state and state != expected_state:
                    self.send_response(400)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(oauth_error_html("State mismatch.").encode())
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    oauth_success_html(f"{provider_name} authentication completed. You can close this window.").encode()
                )

                def _resolve() -> None:
                    if not future.done():
                        future.set_result(CallbackResult(code=code, state=state))

                loop.call_soon_threadsafe(_resolve)

        server = HTTPServer((host, port), Handler)
        server.timeout = 0.5
        self._server = server

        def serve() -> None:
            server.serve_forever(poll_interval=0.5)

        thread = Thread(target=serve, daemon=True)
        thread.start()
        self._thread = thread

    def cancel_wait(self) -> None:
        """Cancel waiting for the callback (resolve future with None)."""
        if self._future is not None and not self._future.done() and self._loop is not None:

            def _cancel() -> None:
                if self._future is not None and not self._future.done():
                    self._future.set_result(None)

            self._loop.call_soon_threadsafe(_cancel)

    async def wait_for_code(self) -> CallbackResult | None:
        """Wait for the callback to arrive. Returns None if cancelled."""
        if self._future is None:
            raise RuntimeError("Server not started")
        return await self._future

    def close(self) -> None:
        """Shut down the server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def parse_authorization_input(value: str) -> dict[str, str | None]:
    """Parse user-pasted authorization code or redirect URL.

    Returns dict with ``code`` and ``state`` keys (may be None).
    """
    value = value.strip()
    if not value:
        return {"code": None, "state": None}

    # Try as URL
    try:
        parsed = urlparse(value)
        if parsed.scheme in ("http", "https"):
            params = parse_qs(parsed.query)
            return {
                "code": params.get("code", [None])[0],
                "state": params.get("state", [None])[0],
            }
    except Exception:
        pass

    # Try as "code#state"
    if "#" in value:
        parts = value.split("#", 1)
        return {"code": parts[0], "state": parts[1]}

    # Try as query string
    if "code=" in value:
        params = parse_qs(value)
        return {
            "code": params.get("code", [None])[0],
            "state": params.get("state", [None])[0],
        }

    # Plain code
    return {"code": value, "state": None}
