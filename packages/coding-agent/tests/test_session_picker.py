"""Tests for ``nu_coding_agent.session_picker``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nu_coding_agent.session_picker import _SessionPickerApp, select_session


class TestSessionPickerApp:
    def test_instantiate(self) -> None:
        app = _SessionPickerApp(cwd="/tmp/test")
        assert app._cwd == "/tmp/test"
        assert app._session_dir is None

    def test_instantiate_with_session_dir(self) -> None:
        app = _SessionPickerApp(cwd="/tmp/test", session_dir="/tmp/sessions")
        assert app._session_dir == "/tmp/sessions"

    def test_css_defined(self) -> None:
        app = _SessionPickerApp(cwd="/tmp")
        assert app.CSS is not None
        assert "background" in app.CSS

    @pytest.mark.asyncio
    async def test_on_mount(self) -> None:
        app = _SessionPickerApp(cwd="/tmp/test")
        app.push_screen_wait = AsyncMock(return_value="/tmp/session.jsonl")  # type: ignore[method-assign]
        app.exit = MagicMock()  # type: ignore[method-assign]
        await app.on_mount()
        app.exit.assert_called_once_with("/tmp/session.jsonl")


class TestSelectSession:
    @pytest.mark.asyncio
    async def test_returns_result(self) -> None:
        with patch.object(_SessionPickerApp, "run_async", new_callable=AsyncMock, return_value="/tmp/s.jsonl"):
            result = await select_session("/tmp/test")
            assert result == "/tmp/s.jsonl"

    @pytest.mark.asyncio
    async def test_returns_none(self) -> None:
        with patch.object(_SessionPickerApp, "run_async", new_callable=AsyncMock, return_value=None):
            result = await select_session("/tmp/test", session_dir="/tmp/sessions")
            assert result is None
