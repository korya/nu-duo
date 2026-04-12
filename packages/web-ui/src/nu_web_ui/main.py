"""CLI entry point for nu-web-ui.

Usage::

    nu-web-ui [--host HOST] [--port PORT] [--db DB] [--reload]

Starts a uvicorn server hosting the FastAPI application.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nu-web-ui",
        description="Start the nu-web-ui FastAPI backend.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port (default: 8080)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite database file.  Defaults to ~/.nu-web-ui/data.db.",
    )
    parser.add_argument(
        "--frontend",
        default=None,
        help="Path to compiled frontend static files directory.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Set DB path before importing the app module so the lifespan picks it up.
    import os

    if args.db:
        os.environ["NU_WEB_UI_DB"] = args.db

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required.  Install it with: pip install uvicorn[standard]", file=sys.stderr)
        return 1

    # Import here so the env var override above takes effect first.
    from nu_web_ui.app import create_app

    app = create_app(db_path=args.db, frontend_dir=args.frontend)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
