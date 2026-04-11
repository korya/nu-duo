# pi-tui

Python port of [`@mariozechner/pi-tui`](https://github.com/badlogic/pi-mono/tree/main/packages/tui) — terminal UI primitives.

**Deviation from upstream:** rendering is implemented as a thin wrapper over [Textual](https://github.com/Textualize/textual), not the TS differential ANSI renderer. The public API (class names, method names, component model) matches pi-tui so downstream packages port without changes.

See the root `README.md` for the overall port plan and status.
