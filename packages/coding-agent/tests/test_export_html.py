"""Tests for the HTML export module — port of export-html tests.

Tests cover:
- ANSI to HTML conversion (ansi_to_html, ansi_lines_to_html)
- Theme helpers (get_resolved_theme_colors, get_theme_export_colors)
- Colour utilities (_parse_color, _derive_export_colors)
- export_from_file / export_session_to_html integration
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from nu_coding_agent.core.export_html import (
    ExportOptions,
    ansi_lines_to_html,
    ansi_to_html,
    export_from_file,
    export_session_to_html,
)
from nu_coding_agent.core.export_html.index import (
    _adjust_brightness,
    _derive_export_colors,
    _parse_color,
)
from nu_coding_agent.core.session_manager import SessionManager
from nu_coding_agent.modes.interactive.theme.theme import (
    get_resolved_theme_colors,
    get_theme_export_colors,
)

# ===========================================================================
# ansi_to_html
# ===========================================================================


class TestAnsiToHtml:
    def test_plain_text_passthrough(self) -> None:
        assert ansi_to_html("hello world") == "hello world"

    def test_html_escaping(self) -> None:
        assert ansi_to_html("<b>&foo</b>") == "&lt;b&gt;&amp;foo&lt;/b&gt;"

    def test_html_escaping_quotes(self) -> None:
        assert ansi_to_html("\"double\" 'single'") == "&quot;double&quot; &#039;single&#039;"

    def test_bold(self) -> None:
        html = ansi_to_html("\x1b[1mBOLD\x1b[0m")
        assert '<span style="font-weight:bold">BOLD</span>' in html

    def test_italic(self) -> None:
        html = ansi_to_html("\x1b[3mITALIC\x1b[0m")
        assert '<span style="font-style:italic">ITALIC</span>' in html

    def test_underline(self) -> None:
        html = ansi_to_html("\x1b[4mUNDERLINE\x1b[0m")
        assert '<span style="text-decoration:underline">UNDERLINE</span>' in html

    def test_dim(self) -> None:
        html = ansi_to_html("\x1b[2mDIM\x1b[0m")
        assert '<span style="opacity:0.6">DIM</span>' in html

    def test_standard_fg_color(self) -> None:
        # Code 31 = red
        html = ansi_to_html("\x1b[31mRED\x1b[0m")
        assert "color:#800000" in html
        assert "RED" in html

    def test_standard_bg_color(self) -> None:
        # Code 41 = red background
        html = ansi_to_html("\x1b[41mBGRED\x1b[0m")
        assert "background-color:#800000" in html

    def test_bright_fg_color(self) -> None:
        # Code 91 = bright red
        html = ansi_to_html("\x1b[91mBRIGHT\x1b[0m")
        assert "color:#ff0000" in html

    def test_bright_bg_color(self) -> None:
        # Code 101 = bright red background
        html = ansi_to_html("\x1b[101mBRIGHTBG\x1b[0m")
        assert "background-color:#ff0000" in html

    def test_256_fg_color(self) -> None:
        # 38;5;196 = 256-colour index 196 (bright red in cube)
        html = ansi_to_html("\x1b[38;5;196mCOLOR\x1b[0m")
        assert "color:" in html
        assert "COLOR" in html

    def test_rgb_fg_color(self) -> None:
        html = ansi_to_html("\x1b[38;2;255;128;0mORANGE\x1b[0m")
        assert "color:rgb(255,128,0)" in html
        assert "ORANGE" in html

    def test_rgb_bg_color(self) -> None:
        html = ansi_to_html("\x1b[48;2;0;128;255mBLUEBG\x1b[0m")
        assert "background-color:rgb(0,128,255)" in html

    def test_reset_closes_span(self) -> None:
        html = ansi_to_html("\x1b[1mBOLD\x1b[0m normal")
        assert "</span>" in html
        assert " normal" in html

    def test_nested_styles(self) -> None:
        # Bold then color — each escape closes previous span
        html = ansi_to_html("\x1b[1m\x1b[31mBOLD_RED\x1b[0m")
        assert "BOLD_RED" in html
        assert "</span>" in html

    def test_no_trailing_span_without_style(self) -> None:
        html = ansi_to_html("plain")
        assert "<span" not in html

    def test_empty_string(self) -> None:
        assert ansi_to_html("") == ""

    def test_default_fg_reset(self) -> None:
        # Code 39 resets foreground
        html = ansi_to_html("\x1b[31mRED\x1b[39m after")
        assert "after" in html

    def test_256_color_standard_range(self) -> None:
        # Index 0 = #000000, index 15 = #ffffff
        from nu_coding_agent.core.export_html.ansi_to_html import _color256_to_hex

        assert _color256_to_hex(0) == "#000000"
        assert _color256_to_hex(15) == "#ffffff"

    def test_256_color_grayscale(self) -> None:
        from nu_coding_agent.core.export_html.ansi_to_html import _color256_to_hex

        # Index 232 = gray 8 (#080808)
        assert _color256_to_hex(232) == "#080808"
        # Index 255 = gray 238 (#eeeeee)
        assert _color256_to_hex(255) == "#eeeeee"

    def test_256_color_cube(self) -> None:
        from nu_coding_agent.core.export_html.ansi_to_html import _color256_to_hex

        # Index 16 = first cube color (0,0,0 → #000000)
        assert _color256_to_hex(16) == "#000000"
        # Index 231 = last cube color (5,5,5 → #ffffff)
        assert _color256_to_hex(231) == "#ffffff"


class TestAnsiLinesToHtml:
    def test_single_line(self) -> None:
        result = ansi_lines_to_html(["hello"])
        assert result == '<div class="ansi-line">hello</div>'

    def test_empty_line_becomes_nbsp(self) -> None:
        result = ansi_lines_to_html([""])
        assert result == '<div class="ansi-line">&nbsp;</div>'

    def test_multiple_lines(self) -> None:
        result = ansi_lines_to_html(["a", "b"])
        assert '<div class="ansi-line">a</div>' in result
        assert '<div class="ansi-line">b</div>' in result

    def test_ansi_in_lines(self) -> None:
        result = ansi_lines_to_html(["\x1b[1mBOLD\x1b[0m"])
        assert "BOLD" in result
        assert "font-weight:bold" in result


# ===========================================================================
# Theme helpers
# ===========================================================================


class TestGetResolvedThemeColors:
    def test_dark_theme_returns_dict(self) -> None:
        colors = get_resolved_theme_colors("dark")
        assert isinstance(colors, dict)
        assert len(colors) > 10

    def test_light_theme_returns_dict(self) -> None:
        colors = get_resolved_theme_colors("light")
        assert isinstance(colors, dict)
        assert len(colors) > 10

    def test_all_values_are_strings(self) -> None:
        colors = get_resolved_theme_colors("dark")
        for key, value in colors.items():
            assert isinstance(value, str), f"{key} = {value!r} is not str"

    def test_all_values_start_with_hash_or_rgb(self) -> None:
        colors = get_resolved_theme_colors("dark")
        for key, value in colors.items():
            assert value.startswith(("#", "rgb(")), f"{key} = {value!r}"

    def test_accent_color_present(self) -> None:
        colors = get_resolved_theme_colors("dark")
        assert "accent" in colors

    def test_unknown_theme_falls_back_to_dark(self) -> None:
        # Should not raise; falls back to dark
        colors = get_resolved_theme_colors("does-not-exist")
        assert "accent" in colors

    def test_none_uses_default(self) -> None:
        colors = get_resolved_theme_colors(None)
        assert isinstance(colors, dict)
        assert len(colors) > 0


class TestGetThemeExportColors:
    def test_dark_theme_has_export_colors(self) -> None:
        export = get_theme_export_colors("dark")
        assert export["pageBg"] is not None
        assert export["cardBg"] is not None
        assert export["infoBg"] is not None

    def test_light_theme_has_export_colors(self) -> None:
        export = get_theme_export_colors("light")
        assert export["pageBg"] is not None
        assert export["cardBg"] is not None

    def test_values_are_strings(self) -> None:
        export = get_theme_export_colors("dark")
        for key, value in export.items():
            if value is not None:
                assert isinstance(value, str), f"{key} = {value!r}"

    def test_unknown_theme_returns_nones(self) -> None:
        export = get_theme_export_colors("does-not-exist")
        assert export == {"pageBg": None, "cardBg": None, "infoBg": None}


# ===========================================================================
# Colour utilities
# ===========================================================================


class TestParseColor:
    def test_hex_color(self) -> None:
        result = _parse_color("#ff8800")
        assert result == (255, 136, 0)

    def test_hex_uppercase(self) -> None:
        result = _parse_color("#FF8800")
        assert result == (255, 136, 0)

    def test_rgb_color(self) -> None:
        result = _parse_color("rgb(255, 128, 0)")
        assert result == (255, 128, 0)

    def test_rgb_no_spaces(self) -> None:
        result = _parse_color("rgb(10,20,30)")
        assert result == (10, 20, 30)

    def test_invalid_returns_none(self) -> None:
        assert _parse_color("not-a-color") is None
        assert _parse_color("") is None

    def test_invalid_hex_short(self) -> None:
        assert _parse_color("#fff") is None


class TestAdjustBrightness:
    def test_lighten(self) -> None:
        result = _adjust_brightness("#808080", 1.5)
        assert result.startswith("rgb(")

    def test_darken(self) -> None:
        result = _adjust_brightness("#808080", 0.5)
        assert result.startswith("rgb(")

    def test_invalid_color_passthrough(self) -> None:
        result = _adjust_brightness("not-a-color", 1.5)
        assert result == "not-a-color"

    def test_clamp_at_255(self) -> None:
        result = _adjust_brightness("#ffffff", 2.0)
        assert "255" in result


class TestDeriveExportColors:
    def test_dark_base_produces_three_colors(self) -> None:
        colors = _derive_export_colors("#343541")
        assert "pageBg" in colors
        assert "cardBg" in colors
        assert "infoBg" in colors

    def test_light_base_produces_lighter_colors(self) -> None:
        colors = _derive_export_colors("#ffffff")
        assert colors["cardBg"] == "#ffffff"  # cardBg is base for light

    def test_invalid_color_uses_defaults(self) -> None:
        colors = _derive_export_colors("invalid")
        assert colors["pageBg"] == "rgb(24, 24, 30)"


def _write_minimal_jsonl(path: Path) -> None:
    """Write a minimal valid session JSONL file (header entry only)."""
    header = {
        "type": "session",
        "version": 3,
        "id": uuid.uuid4().hex,
        "timestamp": "2024-01-01T00:00:00.000Z",
        "cwd": "/tmp",
    }
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")


# ===========================================================================
# export_from_file integration
# ===========================================================================


class TestExportFromFile:
    async def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await export_from_file(str(tmp_path / "nonexistent.jsonl"))

    async def test_exports_minimal_session(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"
        _write_minimal_jsonl(session_file)
        out = str(tmp_path / "output.html")
        result = await export_from_file(str(session_file), ExportOptions(output_path=out))
        assert result == out
        html = Path(out).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "Session Export" in html

    async def test_default_output_path(self, tmp_path: Path) -> None:
        session_file = tmp_path / "mysession.jsonl"
        _write_minimal_jsonl(session_file)
        original_cwd = Path.cwd()
        import os

        os.chdir(tmp_path)
        try:
            result = await export_from_file(str(session_file))
            assert result == "nu-session-mysession.html"
            assert (tmp_path / result).exists()
        finally:
            os.chdir(original_cwd)

    async def test_html_contains_session_data_script(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"
        _write_minimal_jsonl(session_file)
        out = str(tmp_path / "out.html")
        await export_from_file(str(session_file), ExportOptions(output_path=out))
        html = Path(out).read_text(encoding="utf-8")
        assert 'id="session-data"' in html
        assert 'type="application/json"' in html

    async def test_dark_theme(self, tmp_path: Path) -> None:
        session_file = tmp_path / "s.jsonl"
        _write_minimal_jsonl(session_file)
        out = str(tmp_path / "out.html")
        await export_from_file(str(session_file), ExportOptions(output_path=out, theme_name="dark"))
        html = Path(out).read_text(encoding="utf-8")
        assert "#18181e" in html  # dark theme pageBg

    async def test_light_theme(self, tmp_path: Path) -> None:
        session_file = tmp_path / "s.jsonl"
        _write_minimal_jsonl(session_file)
        out = str(tmp_path / "out.html")
        await export_from_file(str(session_file), ExportOptions(output_path=out, theme_name="light"))
        html = Path(out).read_text(encoding="utf-8")
        assert "#f8f8f8" in html  # light theme pageBg

    async def test_string_options(self, tmp_path: Path) -> None:
        session_file = tmp_path / "s.jsonl"
        _write_minimal_jsonl(session_file)
        out = str(tmp_path / "out.html")
        result = await export_from_file(str(session_file), out)
        assert result == out


# ===========================================================================
# export_session_to_html integration
# ===========================================================================


class TestExportSessionToHtml:
    async def test_raises_for_in_memory_session(self) -> None:
        sm = SessionManager.in_memory()
        with pytest.raises(ValueError, match="in-memory"):
            await export_session_to_html(sm)

    async def test_raises_when_no_file_yet(self, tmp_path: Path) -> None:
        sm = SessionManager.open(str(tmp_path / "new.jsonl"))
        # File hasn't been created yet
        with pytest.raises(ValueError, match="Nothing to export"):
            await export_session_to_html(sm)

    async def test_exports_existing_session_file(self, tmp_path: Path) -> None:
        session_file = tmp_path / "chat.jsonl"
        _write_minimal_jsonl(session_file)
        sm = SessionManager.open(str(session_file))
        out = str(tmp_path / "out.html")
        result = await export_session_to_html(sm, options=ExportOptions(output_path=out))
        assert result == out
        html = Path(out).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html

    async def test_state_system_prompt_included(self, tmp_path: Path) -> None:
        session_file = tmp_path / "chat.jsonl"
        _write_minimal_jsonl(session_file)
        sm = SessionManager.open(str(session_file))

        class FakeState:
            system_prompt = "You are a helpful assistant."
            tools: list = []

        out = str(tmp_path / "out.html")
        await export_session_to_html(sm, state=FakeState(), options=ExportOptions(output_path=out))
        html = Path(out).read_text(encoding="utf-8")
        # System prompt is base64-encoded in the session data blob
        import base64
        import json as json_mod

        script_start = html.index('id="session-data"')
        script_end = html.index("</script>", script_start)
        b64 = html[script_start:script_end].split(">", 1)[1].strip()
        data = json_mod.loads(base64.b64decode(b64).decode())
        assert data["systemPrompt"] == "You are a helpful assistant."
