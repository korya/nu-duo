"""Tests for ``nu_coding_agent.utils.tools_manager``."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nu_coding_agent.utils.tools_manager import (
    _download_file,
    _download_tool,
    _fd_asset_name,
    _find_binary_recursively,
    _get_arch,
    _get_latest_version,
    _get_platform,
    _is_offline_mode,
    _rg_asset_name,
    ensure_tool,
    get_tool_path,
)

# ---------------------------------------------------------------------------
# _is_offline_mode
# ---------------------------------------------------------------------------


class TestIsOfflineMode:
    def test_not_offline(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _is_offline_mode() is False

    def test_pi_offline(self) -> None:
        with patch.dict(os.environ, {"PI_OFFLINE": "1"}, clear=True):
            assert _is_offline_mode() is True

    def test_nu_offline_true(self) -> None:
        with patch.dict(os.environ, {"NU_OFFLINE": "true"}, clear=True):
            assert _is_offline_mode() is True

    def test_nu_offline_yes(self) -> None:
        with patch.dict(os.environ, {"NU_OFFLINE": "yes"}, clear=True):
            assert _is_offline_mode() is True

    def test_nu_offline_false(self) -> None:
        with patch.dict(os.environ, {"NU_OFFLINE": "false"}, clear=True):
            assert _is_offline_mode() is False


# ---------------------------------------------------------------------------
# Asset name functions
# ---------------------------------------------------------------------------


class TestFdAssetName:
    def test_darwin_arm(self) -> None:
        name = _fd_asset_name("10.2.0", "darwin", "arm64")
        assert name == "fd-v10.2.0-aarch64-apple-darwin.tar.gz"

    def test_darwin_x64(self) -> None:
        name = _fd_asset_name("10.2.0", "darwin", "x64")
        assert name == "fd-v10.2.0-x86_64-apple-darwin.tar.gz"

    def test_linux(self) -> None:
        name = _fd_asset_name("10.2.0", "linux", "x64")
        assert name == "fd-v10.2.0-x86_64-unknown-linux-gnu.tar.gz"

    def test_win32(self) -> None:
        name = _fd_asset_name("10.2.0", "win32", "x64")
        assert name == "fd-v10.2.0-x86_64-pc-windows-msvc.zip"

    def test_unsupported(self) -> None:
        assert _fd_asset_name("10.2.0", "freebsd", "x64") is None


class TestRgAssetName:
    def test_darwin_arm(self) -> None:
        name = _rg_asset_name("14.1.1", "darwin", "arm64")
        assert name == "ripgrep-14.1.1-aarch64-apple-darwin.tar.gz"

    def test_linux_arm(self) -> None:
        name = _rg_asset_name("14.1.1", "linux", "arm64")
        assert name == "ripgrep-14.1.1-aarch64-unknown-linux-gnu.tar.gz"

    def test_linux_x64(self) -> None:
        name = _rg_asset_name("14.1.1", "linux", "x64")
        assert name == "ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz"

    def test_win32(self) -> None:
        name = _rg_asset_name("14.1.1", "win32", "x64")
        assert name == "ripgrep-14.1.1-x86_64-pc-windows-msvc.zip"

    def test_unsupported(self) -> None:
        assert _rg_asset_name("14.1.1", "freebsd", "x64") is None


# ---------------------------------------------------------------------------
# _get_platform / _get_arch
# ---------------------------------------------------------------------------


class TestGetPlatform:
    def test_darwin(self) -> None:
        with patch("nu_coding_agent.utils.tools_manager.sys.platform", "darwin"):
            assert _get_platform() == "darwin"

    def test_linux(self) -> None:
        with (
            patch("nu_coding_agent.utils.tools_manager.sys.platform", "linux"),
            patch.dict(os.environ, {}, clear=True),
        ):
            assert _get_platform() == "linux"

    def test_android(self) -> None:
        with (
            patch("nu_coding_agent.utils.tools_manager.sys.platform", "linux"),
            patch.dict(os.environ, {"ANDROID_ROOT": "/system"}, clear=True),
        ):
            assert _get_platform() == "android"

    def test_win32(self) -> None:
        with patch("nu_coding_agent.utils.tools_manager.sys.platform", "win32"):
            assert _get_platform() == "win32"


class TestGetArch:
    def test_aarch64(self) -> None:
        with patch("nu_coding_agent.utils.tools_manager._platform.machine", return_value="aarch64"):
            assert _get_arch() == "arm64"

    def test_x86_64(self) -> None:
        with patch("nu_coding_agent.utils.tools_manager._platform.machine", return_value="x86_64"):
            assert _get_arch() == "x64"

    def test_arm64(self) -> None:
        with patch("nu_coding_agent.utils.tools_manager._platform.machine", return_value="arm64"):
            assert _get_arch() == "arm64"


# ---------------------------------------------------------------------------
# get_tool_path
# ---------------------------------------------------------------------------


class TestGetToolPath:
    def test_cached_binary(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "fd").touch()
        with patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)):
            result = get_tool_path("fd")
            assert result == str(bin_dir / "fd")

    def test_system_which(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager.shutil.which", return_value="/usr/bin/rg"),
        ):
            result = get_tool_path("rg")
            assert result == "/usr/bin/rg"

    def test_not_found(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager.shutil.which", return_value=None),
        ):
            result = get_tool_path("fd")
            assert result is None

    def test_unknown_tool(self) -> None:
        result = get_tool_path("unknown_tool")  # type: ignore[arg-type]
        assert result is None


# ---------------------------------------------------------------------------
# ensure_tool
# ---------------------------------------------------------------------------


class TestEnsureTool:
    @pytest.mark.asyncio
    async def test_existing_tool_returned(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "fd").touch()
        with patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)):
            result = await ensure_tool("fd")
            assert result == str(bin_dir / "fd")

    @pytest.mark.asyncio
    async def test_offline_mode_returns_none(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager.shutil.which", return_value=None),
            patch.dict(os.environ, {"NU_OFFLINE": "1"}, clear=True),
        ):
            result = await ensure_tool("fd", silent=True)
            assert result is None

    @pytest.mark.asyncio
    async def test_android_returns_none(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager.shutil.which", return_value=None),
            patch("nu_coding_agent.utils.tools_manager._get_platform", return_value="android"),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = await ensure_tool("rg", silent=True)
            assert result is None

    @pytest.mark.asyncio
    async def test_download_failure_returns_none(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager.shutil.which", return_value=None),
            patch("nu_coding_agent.utils.tools_manager._get_platform", return_value="linux"),
            patch(
                "nu_coding_agent.utils.tools_manager._download_tool",
                side_effect=RuntimeError("network error"),
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = await ensure_tool("fd", silent=True)
            assert result is None

    @pytest.mark.asyncio
    async def test_unknown_tool(self) -> None:
        result = await ensure_tool("unknown_tool")  # type: ignore[arg-type]
        assert result is None

    @pytest.mark.asyncio
    async def test_download_success(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager.shutil.which", return_value=None),
            patch("nu_coding_agent.utils.tools_manager._get_platform", return_value="linux"),
            patch(
                "nu_coding_agent.utils.tools_manager._download_tool",
                return_value="/home/user/.nu/agent/bin/fd",
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = await ensure_tool("fd", silent=True)
            assert result == "/home/user/.nu/agent/bin/fd"


# ---------------------------------------------------------------------------
# _find_binary_recursively
# ---------------------------------------------------------------------------


class TestFindBinaryRecursively:
    def test_finds_file(self, tmp_path: Path) -> None:
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        binary = subdir / "fd"
        binary.touch()
        result = _find_binary_recursively(tmp_path, "fd")
        assert result is not None
        assert result.name == "fd"

    def test_not_found(self, tmp_path: Path) -> None:
        result = _find_binary_recursively(tmp_path, "missing")
        assert result is None

    def test_file_at_root(self, tmp_path: Path) -> None:
        binary = tmp_path / "rg"
        binary.touch()
        result = _find_binary_recursively(tmp_path, "rg")
        assert result is not None


# ---------------------------------------------------------------------------
# _get_latest_version
# ---------------------------------------------------------------------------


class TestGetLatestVersion:
    @pytest.mark.asyncio
    async def test_strips_v_prefix(self) -> None:
        import httpx as _httpx

        mock_response = MagicMock()
        mock_response.json.return_value = {"tag_name": "v14.1.1"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock(spec=_httpx)
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            version = await _get_latest_version("sharkdp/fd")
            assert version == "14.1.1"

    @pytest.mark.asyncio
    async def test_no_v_prefix(self) -> None:
        import httpx as _httpx

        mock_response = MagicMock()
        mock_response.json.return_value = {"tag_name": "14.1.1"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock(spec=_httpx)
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            version = await _get_latest_version("BurntSushi/ripgrep")
            assert version == "14.1.1"


# ---------------------------------------------------------------------------
# _download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    @pytest.mark.asyncio
    async def test_downloads_to_dest(self, tmp_path: Path) -> None:
        import httpx as _httpx

        dest = tmp_path / "archive.tar.gz"
        chunks = [b"chunk1", b"chunk2"]

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_bytes = MagicMock(return_value=AsyncIterator(chunks))
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_stream = MagicMock(return_value=mock_resp)

        mock_client = AsyncMock()
        mock_client.stream = mock_stream
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock(spec=_httpx)
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            await _download_file("https://example.com/file.tar.gz", dest)
            assert dest.read_bytes() == b"chunk1chunk2"


class AsyncIterator:
    """Helper to create an async iterator from a list."""

    def __init__(self, items: list[bytes]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


# ---------------------------------------------------------------------------
# _download_tool (integration-ish, with mocked network)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestDownloadTool:
    @pytest.mark.asyncio
    async def test_downloads_and_extracts_tar_gz(self, tmp_path: Path) -> None:
        import io
        import tarfile

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        # Create a tar.gz in memory with fd binary inside
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tf:
            info = tarfile.TarInfo(name="fd-v10.2.0-x86_64-apple-darwin/fd")
            info.size = 4
            tf.addfile(info, io.BytesIO(b"fake"))
        tar_bytes = tar_buf.getvalue()

        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager._get_platform", return_value="darwin"),
            patch("nu_coding_agent.utils.tools_manager._get_arch", return_value="x64"),
            patch("nu_coding_agent.utils.tools_manager._get_latest_version", return_value="10.2.0"),
            patch("nu_coding_agent.utils.tools_manager._download_file") as mock_dl,
        ):
            # Make _download_file write the tar to the archive path
            async def fake_download(url: str, dest: Path) -> None:
                dest.write_bytes(tar_bytes)

            mock_dl.side_effect = fake_download

            result = await _download_tool("fd")
            assert result == str(bin_dir / "fd")
            assert (bin_dir / "fd").exists()

    @pytest.mark.asyncio
    async def test_unsupported_platform(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager._get_platform", return_value="freebsd"),
            patch("nu_coding_agent.utils.tools_manager._get_arch", return_value="x64"),
            patch("nu_coding_agent.utils.tools_manager._get_latest_version", return_value="10.2.0"),
        ):
            with pytest.raises(RuntimeError, match="Unsupported platform"):
                await _download_tool("fd")

    @pytest.mark.asyncio
    async def test_downloads_zip(self, tmp_path: Path) -> None:
        import io
        import zipfile

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("fd-v10.2.0-x86_64-pc-windows-msvc/fd.exe", b"fake")
        zip_bytes = zip_buf.getvalue()

        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager._get_platform", return_value="win32"),
            patch("nu_coding_agent.utils.tools_manager._get_arch", return_value="x64"),
            patch("nu_coding_agent.utils.tools_manager._get_latest_version", return_value="10.2.0"),
            patch("nu_coding_agent.utils.tools_manager._download_file") as mock_dl,
        ):

            async def fake_download(url: str, dest: Path) -> None:
                dest.write_bytes(zip_bytes)

            mock_dl.side_effect = fake_download
            result = await _download_tool("fd")
            assert result == str(bin_dir / "fd.exe")

    @pytest.mark.asyncio
    async def test_binary_not_found_in_archive(self, tmp_path: Path) -> None:
        import io
        import tarfile

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tf:
            info = tarfile.TarInfo(name="somedir/other_binary")
            info.size = 4
            tf.addfile(info, io.BytesIO(b"fake"))
        tar_bytes = tar_buf.getvalue()

        with (
            patch("nu_coding_agent.utils.tools_manager.get_bin_dir", return_value=str(bin_dir)),
            patch("nu_coding_agent.utils.tools_manager._get_platform", return_value="darwin"),
            patch("nu_coding_agent.utils.tools_manager._get_arch", return_value="x64"),
            patch("nu_coding_agent.utils.tools_manager._get_latest_version", return_value="10.2.0"),
            patch("nu_coding_agent.utils.tools_manager._download_file") as mock_dl,
        ):

            async def fake_download(url: str, dest: Path) -> None:
                dest.write_bytes(tar_bytes)

            mock_dl.side_effect = fake_download
            with pytest.raises(RuntimeError, match="Binary not found"):
                await _download_tool("fd")
