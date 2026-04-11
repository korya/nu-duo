# pi-web-ui

Python port of [`@mariozechner/pi-web-ui`](https://github.com/badlogic/pi-mono/tree/main/packages/web-ui) — browser chat UI.

**Hybrid structure:** the Lit/TS frontend is vendored verbatim from upstream under `frontend/`; the Python package here provides the FastAPI backend, SQLite storage stores, WebSocket streaming, and CORS proxy. The frontend is served as static assets.

See the root `README.md` for the overall port plan and status.
