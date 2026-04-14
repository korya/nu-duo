"""Tests for ``nu_coding_agent.utils.clipboard`` and ``nu_coding_agent.utils.clipboard_image``."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
from nu_coding_agent.utils.clipboard import _copy_sync, _copy_to_x11_clipboard, copy_to_clipboard
from nu_coding_agent.utils.clipboard_image import (
    ClipboardImage,
    _base_mime_type,
    _read_clipboard_image_sync,
    _select_preferred_image_mime_type,
    extension_for_image_mime_type,
    is_wayland_session,
    read_clipboard_image,
)

# ---- clipboard.py ---------------------------------------------------------


class TestCopyToX11Clipboard:
    def test_xclip_success(self) -> None:
        with patch("nu_coding_agent.utils.clipboard.subprocess.run") as mock_run:
            _copy_to_x11_clipboard("hello")
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["xclip", "-selection", "clipboard"]

    def test_xclip_fails_falls_back_to_xsel(self) -> None:
        with patch("nu_coding_agent.utils.clipboard.subprocess.run") as mock_run:
            mock_run.side_effect = [FileNotFoundError, MagicMock()]
            _copy_to_x11_clipboard("hello")
            assert mock_run.call_count == 2
            assert mock_run.call_args_list[1][0][0] == ["xsel", "--clipboard", "--input"]


class TestCopySync:
    def test_osc52_emitted(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout") as mock_stdout,
            patch("nu_coding_agent.utils.clipboard.sys.platform", "linux"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run"),
            patch.dict("os.environ", {"DISPLAY": "x"}, clear=True),
        ):
            _copy_sync("test")
            written = mock_stdout.write.call_args[0][0]
            encoded = base64.b64encode(b"test").decode("ascii")
            assert f"\033]52;c;{encoded}\a" == written

    def test_darwin_uses_pbcopy(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "darwin"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run") as mock_run,
        ):
            _copy_sync("hi")
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["pbcopy"]

    def test_win32_uses_clip(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "win32"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run") as mock_run,
        ):
            _copy_sync("hi")
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["clip"]

    def test_linux_termux(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "linux"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run") as mock_run,
            patch.dict("os.environ", {"TERMUX_VERSION": "1"}, clear=True),
        ):
            _copy_sync("hi")
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["termux-clipboard-set"]

    def test_linux_termux_failure_falls_through(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "linux"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run", side_effect=FileNotFoundError),
            patch("nu_coding_agent.utils.clipboard.subprocess.Popen") as mock_popen,
            patch.dict("os.environ", {"TERMUX_VERSION": "1", "WAYLAND_DISPLAY": "w0"}, clear=True),
        ):
            mock_popen_inst = MagicMock()
            mock_popen.return_value = mock_popen_inst
            # termux fails, falls to wayland which also fails (run raises), but no crash
            _copy_sync("hi")

    def test_linux_wayland_broken_pipe(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "linux"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run"),
            patch("nu_coding_agent.utils.clipboard.subprocess.Popen") as mock_popen,
            patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"}, clear=True),
        ):
            mock_stdin = MagicMock()
            mock_stdin.write.side_effect = BrokenPipeError
            mock_popen.return_value = MagicMock(stdin=mock_stdin)
            # Should not raise
            _copy_sync("hi")

    def test_linux_wayland_fails_falls_to_x11(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "linux"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run") as mock_run,
            patch.dict("os.environ", {"WAYLAND_DISPLAY": "w0", "DISPLAY": ":0"}, clear=True),
        ):
            import subprocess as sp

            # First call (which wl-copy) raises, falling to x11
            mock_run.side_effect = [sp.CalledProcessError(1, "which"), MagicMock()]
            _copy_sync("hi")

    def test_linux_wayland(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "linux"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run") as mock_run,
            patch("nu_coding_agent.utils.clipboard.subprocess.Popen") as mock_popen,
            patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"}, clear=True),
        ):
            mock_popen_inst = MagicMock()
            mock_popen.return_value = mock_popen_inst
            _copy_sync("hi")
            # First run call is `which wl-copy`
            mock_run.assert_called_once()
            mock_popen.assert_called_once()

    def test_linux_x11(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "linux"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run") as mock_run,
            patch.dict("os.environ", {"DISPLAY": ":0"}, clear=True),
        ):
            _copy_sync("hi")
            # Should try xclip
            assert mock_run.call_count >= 1

    def test_subprocess_error_swallowed(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard.sys.stdout"),
            patch("nu_coding_agent.utils.clipboard.sys.platform", "darwin"),
            patch("nu_coding_agent.utils.clipboard.subprocess.run", side_effect=OSError("boom")),
        ):
            # Should not raise
            _copy_sync("hi")


@pytest.mark.asyncio
async def test_copy_to_clipboard_async() -> None:
    with (
        patch("nu_coding_agent.utils.clipboard._copy_sync") as mock_sync,
    ):
        await copy_to_clipboard("text")
        mock_sync.assert_called_once_with("text")


# ---- clipboard_image.py ---------------------------------------------------


class TestIsWaylandSession:
    def test_wayland_display(self) -> None:
        assert is_wayland_session({"WAYLAND_DISPLAY": "wayland-0"}) is True

    def test_xdg_session_type(self) -> None:
        assert is_wayland_session({"XDG_SESSION_TYPE": "wayland"}) is True

    def test_x11(self) -> None:
        assert is_wayland_session({"DISPLAY": ":0"}) is False

    def test_empty(self) -> None:
        assert is_wayland_session({}) is False


class TestExtensionForImageMimeType:
    def test_png(self) -> None:
        assert extension_for_image_mime_type("image/png") == "png"

    def test_jpeg(self) -> None:
        assert extension_for_image_mime_type("image/jpeg") == "jpg"

    def test_webp(self) -> None:
        assert extension_for_image_mime_type("image/webp") == "webp"

    def test_gif(self) -> None:
        assert extension_for_image_mime_type("image/gif") == "gif"

    def test_with_params(self) -> None:
        assert extension_for_image_mime_type("image/png; charset=utf-8") == "png"

    def test_unsupported(self) -> None:
        assert extension_for_image_mime_type("image/bmp") is None

    def test_not_image(self) -> None:
        assert extension_for_image_mime_type("text/plain") is None


class TestBaseMimeType:
    def test_strips_params(self) -> None:
        assert _base_mime_type("image/png; charset=utf-8") == "image/png"

    def test_lowercases(self) -> None:
        assert _base_mime_type("IMAGE/PNG") == "image/png"

    def test_plain(self) -> None:
        assert _base_mime_type("image/jpeg") == "image/jpeg"


class TestSelectPreferredImageMimeType:
    def test_prefers_png(self) -> None:
        assert _select_preferred_image_mime_type(["image/gif", "image/png"]) == "image/png"

    def test_falls_back_to_any_image(self) -> None:
        assert _select_preferred_image_mime_type(["image/bmp"]) == "image/bmp"

    def test_empty(self) -> None:
        assert _select_preferred_image_mime_type([]) is None

    def test_no_image(self) -> None:
        assert _select_preferred_image_mime_type(["text/plain"]) is None


class TestReadClipboardImageSync:
    def test_termux_returns_none(self) -> None:
        result = _read_clipboard_image_sync(env={"TERMUX_VERSION": "1"}, platform="linux")
        assert result is None

    def test_darwin_pngpaste(self) -> None:
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with patch(
            "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_pngpaste",
            return_value=ClipboardImage(bytes=fake_png, mime_type="image/png"),
        ):
            result = _read_clipboard_image_sync(env={}, platform="darwin")
            assert result is not None
            assert result.mime_type == "image/png"

    def test_darwin_fallback_osascript(self) -> None:
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with (
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_pngpaste",
                return_value=None,
            ),
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_osascript",
                return_value=ClipboardImage(bytes=fake_png, mime_type="image/png"),
            ),
        ):
            result = _read_clipboard_image_sync(env={}, platform="darwin")
            assert result is not None

    def test_unsupported_format_converted_to_png(self) -> None:
        with (
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_pngpaste",
                return_value=ClipboardImage(bytes=b"bmpdata", mime_type="image/bmp"),
            ),
            patch(
                "nu_coding_agent.utils.clipboard_image._convert_to_png",
                return_value=b"pngdata",
            ),
        ):
            result = _read_clipboard_image_sync(env={}, platform="darwin")
            assert result is not None
            assert result.mime_type == "image/png"
            assert result.bytes == b"pngdata"

    def test_unsupported_format_conversion_fails(self) -> None:
        with (
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_pngpaste",
                return_value=ClipboardImage(bytes=b"bmpdata", mime_type="image/bmp"),
            ),
            patch(
                "nu_coding_agent.utils.clipboard_image._convert_to_png",
                return_value=None,
            ),
        ):
            result = _read_clipboard_image_sync(env={}, platform="darwin")
            assert result is None

    def test_linux_wayland(self) -> None:
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with patch(
            "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_wl_paste",
            return_value=ClipboardImage(bytes=fake_png, mime_type="image/png"),
        ):
            result = _read_clipboard_image_sync(env={"WAYLAND_DISPLAY": "wayland-0"}, platform="linux")
            assert result is not None

    def test_linux_x11(self) -> None:
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with (
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_wl_paste",
                return_value=None,
            ),
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_xclip",
                return_value=ClipboardImage(bytes=fake_png, mime_type="image/png"),
            ),
        ):
            result = _read_clipboard_image_sync(env={"DISPLAY": ":0"}, platform="linux")
            assert result is not None

    def test_unsupported_platform_returns_none(self) -> None:
        result = _read_clipboard_image_sync(env={}, platform="win32")
        assert result is None

    def test_no_image_returns_none(self) -> None:
        with (
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_pngpaste",
                return_value=None,
            ),
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_osascript",
                return_value=None,
            ),
        ):
            result = _read_clipboard_image_sync(env={}, platform="darwin")
            assert result is None


class TestRunCommand:
    def test_success(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _run_command

        with patch("nu_coding_agent.utils.clipboard_image.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"output")
            result = _run_command("echo", ["hello"])
            assert result.ok is True
            assert result.stdout == b"output"

    def test_nonzero_exit(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _run_command

        with patch("nu_coding_agent.utils.clipboard_image.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout=b"")
            result = _run_command("bad", [])
            assert result.ok is False

    def test_file_not_found(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _run_command

        with patch("nu_coding_agent.utils.clipboard_image.subprocess.run", side_effect=FileNotFoundError):
            result = _run_command("missing", [])
            assert result.ok is False

    def test_timeout(self) -> None:
        import subprocess

        from nu_coding_agent.utils.clipboard_image import _run_command

        with patch(
            "nu_coding_agent.utils.clipboard_image.subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 1.0),
        ):
            result = _run_command("slow", [])
            assert result.ok is False

    def test_over_max_buffer(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _run_command

        with patch("nu_coding_agent.utils.clipboard_image.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"x" * 100)
            result = _run_command("big", [], max_buffer=50)
            assert result.ok is False


class TestReadViaWlPaste:
    def test_success(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_wl_paste

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.side_effect = [
                _CmdResult(ok=True, stdout=b"image/png\ntext/plain\n"),
                _CmdResult(ok=True, stdout=b"pngdata"),
            ]
            result = _read_clipboard_image_via_wl_paste()
            assert result is not None
            assert result.mime_type == "image/png"

    def test_list_types_fails(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_wl_paste

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=False, stdout=b"")
            result = _read_clipboard_image_via_wl_paste()
            assert result is None

    def test_no_image_types(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_wl_paste

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=True, stdout=b"text/plain\n")
            result = _read_clipboard_image_via_wl_paste()
            assert result is None

    def test_data_read_fails(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_wl_paste

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.side_effect = [
                _CmdResult(ok=True, stdout=b"image/png\n"),
                _CmdResult(ok=False, stdout=b""),
            ]
            result = _read_clipboard_image_via_wl_paste()
            assert result is None


class TestReadViaXclip:
    def test_success_with_targets(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_xclip

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.side_effect = [
                _CmdResult(ok=True, stdout=b"image/png\ntext/plain\n"),
                _CmdResult(ok=True, stdout=b"pngdata"),
            ]
            result = _read_clipboard_image_via_xclip()
            assert result is not None
            assert result.mime_type == "image/png"

    def test_targets_fail_tries_all_types(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_xclip

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            # targets fail, then first supported type succeeds
            mock_cmd.side_effect = [
                _CmdResult(ok=False, stdout=b""),
                _CmdResult(ok=True, stdout=b"pngdata"),
            ]
            result = _read_clipboard_image_via_xclip()
            assert result is not None

    def test_nothing_found(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_xclip

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=False, stdout=b"")
            result = _read_clipboard_image_via_xclip()
            assert result is None


class TestReadViaPngpaste:
    def test_success(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_pngpaste

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=True, stdout=b"pngdata")
            result = _read_clipboard_image_via_pngpaste()
            assert result is not None
            assert result.mime_type == "image/png"

    def test_failure(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_pngpaste

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=False, stdout=b"")
            result = _read_clipboard_image_via_pngpaste()
            assert result is None

    def test_empty_output(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_pngpaste

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=True, stdout=b"")
            result = _read_clipboard_image_via_pngpaste()
            assert result is None


class TestReadViaOsascript:
    def test_success(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_osascript

        hex_data = b"\x89PNG".hex()
        output = f"\u00abdata PNGf{hex_data}\u00bb".encode()
        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=True, stdout=output)
            result = _read_clipboard_image_via_osascript()
            assert result is not None
            assert result.mime_type == "image/png"

    def test_failure(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_osascript

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=False, stdout=b"")
            result = _read_clipboard_image_via_osascript()
            assert result is None

    def test_no_hex_marker(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_osascript

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=True, stdout=b"no data here")
            result = _read_clipboard_image_via_osascript()
            assert result is None

    def test_no_end_marker(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_osascript

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=True, stdout="\u00abdata PNGfAABBCC".encode())
            result = _read_clipboard_image_via_osascript()
            assert result is None

    def test_bad_hex(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_osascript

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=True, stdout="\u00abdata PNGfZZZZ\u00bb".encode())
            result = _read_clipboard_image_via_osascript()
            assert result is None

    def test_empty_hex(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_osascript

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=True, stdout="\u00abdata PNGf\u00bb".encode())
            result = _read_clipboard_image_via_osascript()
            assert result is None


class TestIsWsl:
    def test_wsl_distro_name(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _is_wsl

        assert _is_wsl({"WSL_DISTRO_NAME": "Ubuntu"}) is True

    def test_wslenv(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _is_wsl

        assert _is_wsl({"WSLENV": "PATH"}) is True

    def test_not_wsl(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _is_wsl

        with patch("pathlib.Path.read_text", return_value="Linux version 5.15.0-generic"):
            assert _is_wsl({}) is False

    def test_proc_version_microsoft(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _is_wsl

        with patch("pathlib.Path.read_text", return_value="Linux version 5.15.0-Microsoft"):
            assert _is_wsl({}) is True

    def test_proc_version_oserror(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _is_wsl

        with patch("pathlib.Path.read_text", side_effect=OSError):
            assert _is_wsl({}) is False


class TestConvertToPng:
    def test_success(self) -> None:
        import io

        from nu_coding_agent.utils.clipboard_image import _convert_to_png
        from PIL import Image

        img = Image.new("RGB", (5, 5), "blue")
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        result = _convert_to_png(buf.getvalue())
        assert result is not None
        assert result[:4] == b"\x89PNG"

    def test_invalid_data(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _convert_to_png

        result = _convert_to_png(b"not an image")
        assert result is None


class TestReadViaPowershell:
    def test_success(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_powershell

        with (
            patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd,
            patch("nu_coding_agent.utils.clipboard_image.Path") as mock_path_cls,
        ):
            mock_cmd.side_effect = [
                _CmdResult(ok=True, stdout=b"C:\\tmp\\file.png\n"),  # wslpath
                _CmdResult(ok=True, stdout=b"ok\n"),  # powershell
            ]
            mock_path_cls.return_value.read_bytes.return_value = b"pngdata"
            result = _read_clipboard_image_via_powershell()
            assert result is not None
            assert result.mime_type == "image/png"

    def test_wslpath_fails(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_powershell

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.return_value = _CmdResult(ok=False, stdout=b"")
            result = _read_clipboard_image_via_powershell()
            assert result is None

    def test_powershell_returns_empty(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_powershell

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.side_effect = [
                _CmdResult(ok=True, stdout=b"C:\\tmp\\file.png\n"),
                _CmdResult(ok=True, stdout=b"empty\n"),
            ]
            result = _read_clipboard_image_via_powershell()
            assert result is None

    def test_wslpath_empty_path(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_powershell

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.side_effect = [
                _CmdResult(ok=True, stdout=b"\n"),  # empty path
            ]
            result = _read_clipboard_image_via_powershell()
            assert result is None

    def test_powershell_fails(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_powershell

        with patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd:
            mock_cmd.side_effect = [
                _CmdResult(ok=True, stdout=b"C:\\tmp\\file.png\n"),
                _CmdResult(ok=False, stdout=b""),
            ]
            result = _read_clipboard_image_via_powershell()
            assert result is None

    def test_empty_file(self) -> None:
        from nu_coding_agent.utils.clipboard_image import _CmdResult, _read_clipboard_image_via_powershell

        with (
            patch("nu_coding_agent.utils.clipboard_image._run_command") as mock_cmd,
            patch("nu_coding_agent.utils.clipboard_image.Path") as mock_path_cls,
        ):
            mock_cmd.side_effect = [
                _CmdResult(ok=True, stdout=b"C:\\tmp\\file.png\n"),
                _CmdResult(ok=True, stdout=b"ok\n"),
            ]
            mock_path_cls.return_value.read_bytes.return_value = b""
            result = _read_clipboard_image_via_powershell()
            assert result is None


class TestReadClipboardImageSyncLinuxWsl:
    def test_wsl_powershell_fallback(self) -> None:
        with (
            patch("nu_coding_agent.utils.clipboard_image._is_wsl", return_value=True),
            patch("nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_wl_paste", return_value=None),
            patch("nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_xclip", return_value=None),
            patch(
                "nu_coding_agent.utils.clipboard_image._read_clipboard_image_via_powershell",
                return_value=ClipboardImage(bytes=b"png", mime_type="image/png"),
            ),
        ):
            result = _read_clipboard_image_sync(env={"WSL_DISTRO_NAME": "Ubuntu"}, platform="linux")
            assert result is not None


@pytest.mark.asyncio
async def test_read_clipboard_image_async() -> None:
    with patch(
        "nu_coding_agent.utils.clipboard_image._read_clipboard_image_sync",
        return_value=ClipboardImage(bytes=b"data", mime_type="image/png"),
    ):
        result = await read_clipboard_image(env={}, platform="darwin")
        assert result is not None
        assert result.mime_type == "image/png"
