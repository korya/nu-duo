"""Tests for ``nu_coding_agent.core.package_manager``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nu_coding_agent.core.package_manager import (
    ConfiguredPackage,
    DefaultPackageManager,
    ResolvedPaths,
    ResolvedResource,
    _package_name_from_source,
    _packages_from_settings,
    _packages_to_settings,
    _scan_resource_dirs,
    classify_source,
    create_package_manager,
)

# ---------------------------------------------------------------------------
# classify_source
# ---------------------------------------------------------------------------


class TestClassifySource:
    def test_pip_package(self) -> None:
        assert classify_source("requests") == "pip"
        assert classify_source("my-package>=1.0") == "pip"
        assert classify_source("some_pkg==2.3.4") == "pip"

    def test_git_url(self) -> None:
        assert classify_source("git+https://github.com/user/repo.git") == "git"
        assert classify_source("https://github.com/user/repo.git") == "git"
        assert classify_source("git@github.com:user/repo.git") == "git"
        assert classify_source("ssh://git@github.com/user/repo") == "git"

    def test_local_path(self, tmp_path: Path) -> None:
        assert classify_source(str(tmp_path)) == "local"
        assert classify_source("./local_pkg") == "local"
        assert classify_source("../sibling") == "local"
        assert classify_source("/absolute/path") == "local"


# ---------------------------------------------------------------------------
# _package_name_from_source
# ---------------------------------------------------------------------------


class TestPackageNameFromSource:
    def test_simple(self) -> None:
        assert _package_name_from_source("requests") == "requests"

    def test_with_version(self) -> None:
        assert _package_name_from_source("my-pkg>=1.0") == "my-pkg"

    def test_git_url(self) -> None:
        assert _package_name_from_source("https://github.com/user/my-extension.git") == "my-extension"

    def test_normalisation(self) -> None:
        assert _package_name_from_source("My_Package") == "my-package"


# ---------------------------------------------------------------------------
# _packages_from_settings / _packages_to_settings
# ---------------------------------------------------------------------------


class TestSettingsSerialization:
    def test_from_settings_string(self) -> None:
        result = _packages_from_settings(["requests", "flask"])
        assert len(result) == 2
        assert result[0].source == "requests"
        assert result[0].source_type == "pip"
        assert result[0].enabled is True

    def test_from_settings_dict(self) -> None:
        result = _packages_from_settings([{"source": "my-pkg", "sourceType": "pip", "enabled": False}])
        assert len(result) == 1
        assert result[0].source == "my-pkg"
        assert result[0].enabled is False

    def test_from_settings_dict_defaults(self) -> None:
        result = _packages_from_settings([{"source": "./local"}])
        assert result[0].source_type == "local"
        assert result[0].enabled is True

    def test_to_settings(self) -> None:
        pkgs = [ConfiguredPackage(source="foo", source_type="pip", enabled=True)]
        serialized = _packages_to_settings(pkgs)
        assert serialized == [{"source": "foo", "sourceType": "pip", "enabled": True}]

    def test_from_settings_ignores_invalid(self) -> None:
        result = _packages_from_settings([42, None, {"no_source": True}])
        assert result == []


# ---------------------------------------------------------------------------
# _scan_resource_dirs
# ---------------------------------------------------------------------------


class TestScanResourceDirs:
    def test_scans_skills_prompts_themes(self, tmp_path: Path) -> None:
        (tmp_path / "skills").mkdir()
        (tmp_path / "skills" / "hello.py").write_text("# skill")
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "greet.md").write_text("# prompt")
        (tmp_path / "themes").mkdir()
        (tmp_path / "themes" / "dark.json").write_text("{}")

        result = _scan_resource_dirs(tmp_path, "test-pkg")
        assert len(result.skills) == 1
        assert result.skills[0].name == "hello"
        assert len(result.prompts) == 1
        assert result.prompts[0].name == "greet"
        assert len(result.themes) == 1
        assert result.themes[0].name == "dark"

    def test_skips_hidden_and_underscore(self, tmp_path: Path) -> None:
        (tmp_path / "skills").mkdir()
        (tmp_path / "skills" / ".hidden.py").write_text("")
        (tmp_path / "skills" / "_private.py").write_text("")
        (tmp_path / "skills" / "visible.py").write_text("")

        result = _scan_resource_dirs(tmp_path, "src")
        assert len(result.skills) == 1
        assert result.skills[0].name == "visible"

    def test_directory_with_init(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "complex_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "__init__.py").write_text("")

        result = _scan_resource_dirs(tmp_path, "src")
        assert len(result.skills) == 1
        assert result.skills[0].name == "complex_skill"

    def test_no_subdirs(self, tmp_path: Path) -> None:
        result = _scan_resource_dirs(tmp_path, "src")
        assert result.extensions == []
        assert result.skills == []
        assert result.prompts == []
        assert result.themes == []


# ---------------------------------------------------------------------------
# ResolvedPaths
# ---------------------------------------------------------------------------


class TestResolvedPaths:
    def test_merge(self) -> None:
        a = ResolvedPaths(skills=[ResolvedResource(name="s1", path="/a", package_source="x", resource_type="skill")])
        b = ResolvedPaths(skills=[ResolvedResource(name="s2", path="/b", package_source="y", resource_type="skill")])
        a.merge(b)
        assert len(a.skills) == 2


# ---------------------------------------------------------------------------
# DefaultPackageManager
# ---------------------------------------------------------------------------


def _make_settings_mock(packages: list | None = None) -> MagicMock:
    sm = MagicMock()
    sm.get_packages.return_value = packages or []
    sm.get_extension_paths.return_value = []
    sm.get_skill_paths.return_value = []
    sm.get_prompt_template_paths.return_value = []
    sm.get_theme_paths.return_value = []
    return sm


class TestDefaultPackageManager:
    def test_get_configured_packages_empty(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")
        assert pm.get_configured_packages() == []

    def test_get_configured_packages(self) -> None:
        sm = _make_settings_mock(["requests"])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")
        pkgs = pm.get_configured_packages()
        assert len(pkgs) == 1
        assert pkgs[0].source == "requests"

    @pytest.mark.asyncio
    async def test_install_pip(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock) as mock_pip:
            await pm.install("my-package")
            mock_pip.assert_called_once_with("install", "my-package")
            sm.set_packages.assert_called_once()

    @pytest.mark.asyncio
    async def test_install_git(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock) as mock_pip:
            await pm.install("git+https://github.com/user/repo.git")
            mock_pip.assert_called_once_with("install", "git+https://github.com/user/repo.git")

    @pytest.mark.asyncio
    async def test_install_local(self, tmp_path: Path) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        pkg_dir = tmp_path / "local_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "pyproject.toml").write_text("[project]\nname='foo'\n")

        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock) as mock_pip:
            await pm.install(str(pkg_dir))
            mock_pip.assert_called_once_with("install", "-e", str(pkg_dir))

    @pytest.mark.asyncio
    async def test_install_local_no_pyproject(self, tmp_path: Path) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        pkg_dir = tmp_path / "local_pkg"
        pkg_dir.mkdir()

        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock) as mock_pip:
            await pm.install(str(pkg_dir))
            mock_pip.assert_not_called()
            sm.set_packages.assert_called_once()

    @pytest.mark.asyncio
    async def test_install_local_not_found(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with pytest.raises(FileNotFoundError):
            await pm.install("/nonexistent/path/pkg")

    @pytest.mark.asyncio
    async def test_install_duplicate_reenables(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": False}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock):
            await pm.install("my-pkg")
            sm.set_packages.assert_called_once()

    @pytest.mark.asyncio
    async def test_install_already_installed(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock) as mock_pip:
            await pm.install("my-pkg")
            mock_pip.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_pip(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock) as mock_pip:
            await pm.remove("my-pkg")
            mock_pip.assert_called_once_with("uninstall", "-y", "my-pkg")
            sm.set_packages.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_not_found(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="not found"):
                await pm.remove("nonexistent")

    def test_enable(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": False}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")
        pm.enable("my-pkg")
        sm.set_packages.assert_called_once()

    def test_disable(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")
        pm.disable("my-pkg")
        sm.set_packages.assert_called_once()

    def test_enable_not_found(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")
        with pytest.raises(ValueError, match="not found"):
            pm.enable("nonexistent")

    @pytest.mark.asyncio
    async def test_check_updates(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with (
            patch("nu_coding_agent.core.package_manager._installed_version", return_value="1.0.0"),
            patch.object(
                DefaultPackageManager,
                "_fetch_latest_version",
                new_callable=AsyncMock,
                return_value="2.0.0",
            ),
        ):
            updates = await pm.check_updates()
            assert len(updates) == 1
            assert updates[0].current_version == "1.0.0"
            assert updates[0].latest_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_check_updates_no_updates(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with (
            patch("nu_coding_agent.core.package_manager._installed_version", return_value="1.0.0"),
            patch.object(
                DefaultPackageManager,
                "_fetch_latest_version",
                new_callable=AsyncMock,
                return_value="1.0.0",
            ),
        ):
            updates = await pm.check_updates()
            assert updates == []

    @pytest.mark.asyncio
    async def test_check_updates_skips_disabled(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": False}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with patch("nu_coding_agent.core.package_manager._installed_version") as mock_ver:
            updates = await pm.check_updates()
            mock_ver.assert_not_called()
            assert updates == []

    def test_resolve_empty(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/nonexistent", agent_dir="/tmp/agent_ne")
        result = pm.resolve()
        assert isinstance(result, ResolvedPaths)

    def test_resolve_caches(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/nonexistent", agent_dir="/tmp/agent_ne")
        r1 = pm.get_resolved_paths()
        r2 = pm.get_resolved_paths()
        assert r1 is r2

    def test_invalidate_cache(self) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd="/tmp/nonexistent", agent_dir="/tmp/agent_ne")
        r1 = pm.get_resolved_paths()
        pm.invalidate_cache()
        r2 = pm.get_resolved_paths()
        assert r1 is not r2

    def test_resolve_with_entry_points(self) -> None:
        sm = _make_settings_mock([{"source": "my-ext", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp/nonexistent", agent_dir="/tmp/agent_ne")

        mock_ep = MagicMock()
        mock_ep.name = "my_tool"
        mock_ep.value = "my_ext.tools:MyTool"
        mock_dist = MagicMock()
        mock_dist.name = "my-ext"
        mock_ep.dist = mock_dist

        with (
            patch("importlib.metadata.entry_points", return_value=[mock_ep]),
            patch.object(DefaultPackageManager, "_find_package_dir", return_value=None),
        ):
            result = pm.resolve()
            assert len(result.extensions) == 1
            assert result.extensions[0].name == "my_tool"

    def test_list_packages(self) -> None:
        sm = _make_settings_mock([{"source": "my-pkg", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp/test", agent_dir="/tmp/agent")

        with patch("nu_coding_agent.core.package_manager._installed_version", return_value="1.0"):
            result = pm.list_packages()
            assert len(result) == 1
            assert result[0]["version"] == "1.0"

    def test_get_path_metadata_package(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()

        sm = _make_settings_mock([{"source": str(pkg_dir), "sourceType": "local", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"))

        meta = pm.get_path_metadata(str(pkg_dir / "skills" / "foo.py"))
        assert meta.origin == "package"
        assert meta.source == str(pkg_dir)

    def test_get_path_metadata_local(self, tmp_path: Path) -> None:
        sm = _make_settings_mock()
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        pm = DefaultPackageManager(sm, cwd=str(tmp_path), agent_dir=str(agent_dir))

        meta = pm.get_path_metadata(str(agent_dir / "skills" / "foo.py"))
        assert meta.scope == "user"


# ---------------------------------------------------------------------------
# create_package_manager
# ---------------------------------------------------------------------------


class TestFindPipCommand:
    def test_uv_preferred(self) -> None:
        from nu_coding_agent.core.package_manager import _find_pip_command

        with patch("shutil.which", side_effect=lambda x: "/usr/bin/uv" if x == "uv" else None):
            assert _find_pip_command() == ["uv", "pip"]

    def test_pip_fallback(self) -> None:
        from nu_coding_agent.core.package_manager import _find_pip_command

        with patch("shutil.which", side_effect=lambda x: "/usr/bin/pip" if x == "pip" else None):
            assert _find_pip_command() == ["pip"]

    def test_neither_found(self) -> None:
        from nu_coding_agent.core.package_manager import _find_pip_command

        with patch("shutil.which", return_value=None), pytest.raises(RuntimeError, match="Neither"):
            _find_pip_command()


class TestRunPip:
    @pytest.mark.asyncio
    async def test_run_pip_success(self) -> None:
        from nu_coding_agent.core.package_manager import _run_pip

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"output", b"")
        mock_proc.returncode = 0

        with (
            patch("shutil.which", side_effect=lambda x: "/usr/bin/uv" if x == "uv" else None),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await _run_pip("install", "pkg")
            assert result.returncode == 0
            assert result.stdout == "output"

    @pytest.mark.asyncio
    async def test_run_pip_failure(self) -> None:
        from nu_coding_agent.core.package_manager import _run_pip

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"error msg")
        mock_proc.returncode = 1

        with (
            patch("shutil.which", side_effect=lambda x: "/usr/bin/uv" if x == "uv" else None),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            with pytest.raises(RuntimeError, match="pip command failed"):
                await _run_pip("install", "pkg")


class TestResolveUserPaths:
    def test_resolves_files(self, tmp_path: Path) -> None:
        from nu_coding_agent.core.package_manager import _resolve_user_paths

        (tmp_path / "skill.py").write_text("# skill")
        result = _resolve_user_paths([str(tmp_path / "skill.py")], "skill", "user")
        assert len(result) == 1
        assert result[0].name == "skill"

    def test_resolves_directory(self, tmp_path: Path) -> None:
        from nu_coding_agent.core.package_manager import _resolve_user_paths

        (tmp_path / "a.py").write_text("")
        (tmp_path / ".hidden.py").write_text("")
        result = _resolve_user_paths([str(tmp_path)], "skill", "user")
        assert len(result) == 1

    def test_warns_on_missing(self) -> None:
        from nu_coding_agent.core.package_manager import _resolve_user_paths

        result = _resolve_user_paths(["/nonexistent/path"], "skill", "user")
        assert result == []


class TestResolveProjectLevel:
    def test_resolves_project_skills(self, tmp_path: Path) -> None:
        sm = _make_settings_mock()
        cwd = tmp_path / "project"
        cwd.mkdir()
        skills = cwd / ".nu" / "skills"
        skills.mkdir(parents=True)
        (skills / "my_skill.py").write_text("# skill")

        pm = DefaultPackageManager(sm, cwd=str(cwd), agent_dir=str(tmp_path / "agent"))
        result = pm._resolve_project_level()
        assert len(result.skills) == 1

    def test_no_project_dir(self, tmp_path: Path) -> None:
        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"))
        result = pm._resolve_project_level()
        assert result.skills == []


class TestInstalledVersion:
    def test_found(self) -> None:
        from nu_coding_agent.core.package_manager import _installed_version

        with patch("importlib.metadata.version", return_value="1.2.3"):
            assert _installed_version("pkg") == "1.2.3"

    def test_not_found(self) -> None:
        import importlib.metadata as _im

        from nu_coding_agent.core.package_manager import _installed_version

        with patch("importlib.metadata.version", side_effect=_im.PackageNotFoundError("x")):
            assert _installed_version("nonexistent") is None


class TestFetchLatestVersion:
    @pytest.mark.asyncio
    async def test_parses_output(self) -> None:
        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock) as mock_pip:
            mock_pip.return_value = MagicMock(returncode=0, stdout="my-pkg (2.0.0)")
            result = await DefaultPackageManager._fetch_latest_version("my-pkg")
            assert result == "2.0.0"

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self) -> None:
        with patch("nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock) as mock_pip:
            mock_pip.return_value = MagicMock(returncode=1, stdout="")
            result = await DefaultPackageManager._fetch_latest_version("my-pkg")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self) -> None:
        with patch(
            "nu_coding_agent.core.package_manager._run_pip", new_callable=AsyncMock, side_effect=Exception("boom")
        ):
            result = await DefaultPackageManager._fetch_latest_version("my-pkg")
            assert result is None


class TestResolveUserLevel:
    def test_with_user_dirs(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "agent"
        (agent_dir / "tools").mkdir(parents=True)
        (agent_dir / "tools" / "my_ext.py").write_text("")
        (agent_dir / "skills").mkdir()
        (agent_dir / "skills" / "s.py").write_text("")

        sm = _make_settings_mock()
        pm = DefaultPackageManager(sm, cwd=str(tmp_path), agent_dir=str(agent_dir))
        result = pm._resolve_user_level()
        assert len(result.extensions) >= 1 or len(result.skills) >= 1


class TestResolveEntryPointsEdge:
    def test_entry_point_no_dist(self) -> None:
        sm = _make_settings_mock([{"source": "my-ext", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp", agent_dir="/tmp/agent")

        mock_ep = MagicMock()
        mock_ep.name = "tool"
        mock_ep.dist = None

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            paths = ResolvedPaths()
            pkg = ConfiguredPackage(source="my-ext", source_type="pip")
            pm._resolve_entry_points(paths, pkg)
            assert paths.extensions == []

    def test_entry_point_type_error(self) -> None:
        sm = _make_settings_mock([{"source": "my-ext", "sourceType": "pip", "enabled": True}])
        pm = DefaultPackageManager(sm, cwd="/tmp", agent_dir="/tmp/agent")

        with patch("importlib.metadata.entry_points", side_effect=TypeError):
            paths = ResolvedPaths()
            pkg = ConfiguredPackage(source="my-ext", source_type="pip")
            pm._resolve_entry_points(paths, pkg)
            assert paths.extensions == []


class TestFindPackageDir:
    def test_local_existing(self, tmp_path: Path) -> None:
        pkg = ConfiguredPackage(source=str(tmp_path), source_type="local")
        result = DefaultPackageManager._find_package_dir(pkg)
        assert result == tmp_path

    def test_local_nonexistent(self) -> None:
        pkg = ConfiguredPackage(source="/nonexistent", source_type="local")
        result = DefaultPackageManager._find_package_dir(pkg)
        assert result is None

    def test_pip_not_installed(self) -> None:
        import importlib.metadata as _im

        pkg = ConfiguredPackage(source="nonexistent-pkg", source_type="pip")
        with patch("importlib.metadata.distribution", side_effect=_im.PackageNotFoundError("x")):
            result = DefaultPackageManager._find_package_dir(pkg)
            assert result is None


class TestGetPathMetadataProject:
    def test_project_scope(self, tmp_path: Path) -> None:
        sm = _make_settings_mock()
        cwd = tmp_path / "proj"
        cwd.mkdir()
        nu_dir = cwd / ".nu"
        nu_dir.mkdir()
        skill_path = str(nu_dir / "skills" / "foo.py")

        pm = DefaultPackageManager(sm, cwd=str(cwd), agent_dir=str(tmp_path / "agent"))
        meta = pm.get_path_metadata(skill_path)
        assert meta.scope == "project"


class TestCreatePackageManager:
    def test_factory(self) -> None:
        sm = _make_settings_mock()
        pm = create_package_manager(sm, cwd="/tmp/x", agent_dir="/tmp/a")
        assert isinstance(pm, DefaultPackageManager)
