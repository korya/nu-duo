"""Tests for ``nu_coding_agent.config_selector``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nu_coding_agent.config_selector import (
    ConfigSelectorOptions,
    _ConfigSelectorApp,
    select_config,
)


def _make_opts(**overrides: object) -> ConfigSelectorOptions:
    sm = MagicMock()
    sm.get_enabled_skills = MagicMock(return_value=["skill1"])
    sm.get_enabled_extensions = MagicMock(return_value=["ext1"])
    defaults = {"cwd": "/tmp/test", "agent_dir": "/tmp/agent", "settings_manager": sm}
    defaults.update(overrides)
    return ConfigSelectorOptions(**defaults)  # type: ignore[arg-type]


class TestConfigSelectorApp:
    def test_instantiate(self) -> None:
        opts = _make_opts()
        app = _ConfigSelectorApp(opts)
        assert app._options is opts

    def test_options_dataclass(self) -> None:
        sm = MagicMock()
        opts = ConfigSelectorOptions(cwd="/x", agent_dir="/y", settings_manager=sm)
        assert opts.cwd == "/x"
        assert opts.agent_dir == "/y"
        assert opts.settings_manager is sm

    def test_bindings(self) -> None:
        opts = _make_opts()
        app = _ConfigSelectorApp(opts)
        binding_keys = [b[0] for b in app.BINDINGS]
        assert "q" in binding_keys
        assert "escape" in binding_keys

    def test_css_defined(self) -> None:
        opts = _make_opts()
        app = _ConfigSelectorApp(opts)
        assert "background" in app.CSS

    def test_compose_returns_widgets(self) -> None:
        opts = _make_opts()
        app = _ConfigSelectorApp(opts)
        widgets = list(app.compose())
        # Should yield Header, VerticalScroll, Footer
        assert len(widgets) == 3

    def test_build_resource_list_with_skills_and_extensions(self) -> None:
        opts = _make_opts()
        app = _ConfigSelectorApp(opts)
        lv = app._build_resource_list()
        from textual.widgets import ListView

        assert isinstance(lv, ListView)

    def test_build_resource_list_empty(self) -> None:
        sm = MagicMock()
        sm.get_enabled_skills = MagicMock(return_value=[])
        sm.get_enabled_extensions = MagicMock(return_value=[])
        opts = ConfigSelectorOptions(cwd="/tmp", agent_dir="/tmp", settings_manager=sm)
        app = _ConfigSelectorApp(opts)
        # Items list starts with 2 header items (Skills, Extensions), so it
        # won't be empty and the "No configuration" branch won't fire.
        lv = app._build_resource_list()
        from textual.widgets import ListView

        assert isinstance(lv, ListView)

    def test_build_resource_list_no_getattr(self) -> None:
        sm = MagicMock(spec=[])  # No attributes at all
        opts = ConfigSelectorOptions(cwd="/tmp", agent_dir="/tmp", settings_manager=sm)
        app = _ConfigSelectorApp(opts)
        lv = app._build_resource_list()
        from textual.widgets import ListView

        assert isinstance(lv, ListView)

    @pytest.mark.asyncio
    async def test_action_quit(self) -> None:
        opts = _make_opts()
        app = _ConfigSelectorApp(opts)
        # Mock exit so we don't actually try to exit
        app.exit = MagicMock()  # type: ignore[method-assign]
        await app.action_quit()
        app.exit.assert_called_once()


class TestSelectConfig:
    @pytest.mark.asyncio
    async def test_select_config(self) -> None:
        opts = _make_opts()
        with patch.object(_ConfigSelectorApp, "run_async", new_callable=AsyncMock) as mock_run:
            await select_config(opts)
            mock_run.assert_called_once()
