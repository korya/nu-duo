"""Web UI server example: launch the FastAPI backend.

Starts the nu_web_ui FastAPI server with model discovery and
SQLite storage. The frontend (when present) is served as static
files.

Run with::

    uv run python examples/web-ui-server.py
    uv run python examples/web-ui-server.py --port 3000

Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import sys


def main() -> int:
    import uvicorn

    port = 8000
    host = "127.0.0.1"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        else:
            i += 1

    from nu_web_ui import create_app

    app = create_app()

    print(f"Starting nu-web-ui on http://{host}:{port}")
    print("Endpoints:")
    print(f"  GET  http://{host}:{port}/api/sessions")
    print(f"  GET  http://{host}:{port}/api/settings")
    print(f"  GET  http://{host}:{port}/api/models/discover")
    print(f"  WS   ws://{host}:{port}/api/chat")
    print()

    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
