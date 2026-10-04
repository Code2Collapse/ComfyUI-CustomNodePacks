"""Static contract checks for the Advanced Paint Canvas c2c_ui front-end."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
PAINT_JS = PACK / "js" / "mec_advanced_paint.js"
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def paint_source() -> str:
    assert PAINT_JS.is_file(), f"missing {PAINT_JS}"
    return PAINT_JS.read_text(encoding="utf-8")


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_mec_advanced_paint_js_parses():
    p = subprocess.run(
        [NODE, "--check", str(PAINT_JS)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode == 0, p.stderr[:600]


def test_paint_imports_c2c_ui_barrel(paint_source: str):
    assert "./c2c_ui/index.js" in paint_source


def test_paint_uses_mount_panel_and_open_editor(paint_source: str):
    assert "mountPanel" in paint_source
    assert "openEditor" in paint_source


def test_paint_dropped_var_black_fallback(paint_source: str):
    assert "var(--c2c-black)" not in paint_source


def test_a_stroke_ends_on_the_canvas_itself_not_only_on_window():
    """The full-screen editor stops pointerup at its overlay (so input never
    leaks to the graph). With a window-only listener the stroke never ended:
    no undo step, and the brush kept painting on every later mouse move.
    Found live 2026-09-27: Ctrl+Z after one stroke left the canvas unchanged."""
    src = (Path(__file__).resolve().parents[1] / "js" / "mec_advanced_paint.js").read_text(encoding="utf-8")
    bind = src[src.index("    _bind() {"):src.index("    dispose() {")]
    for ev in ("pointerup", "pointercancel", "lostpointercapture"):
        assert f'c.addEventListener("{ev}", this._onUpBound)' in bind, ev
    shell = (Path(__file__).resolve().parents[1] / "js" / "c2c_ui" / "editor.js").read_text(encoding="utf-8")
    assert '"pointerup"' in shell[shell.index("const blockToGraph"):][:300], \
        "if the shell stops blocking pointerup this test documents a constraint that no longer exists"


def test_black_and_white_paint_both_read_on_the_transparency_checkerboard():
    """User, 2026-09-27: "black isn't visible". The checkerboard was the night
    ground (#07081a) - the default black brush vanished on it."""
    import re
    src = (Path(__file__).resolve().parents[1] / "js" / "mec_advanced_paint.js").read_text(encoding="utf-8")

    def lum(hx):
        def ch(v):
            v /= 255
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = (ch(int(hx[i:i + 2], 16)) for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for name in ("CHECKER_A", "CHECKER_B"):
        hx = re.search(rf'const {name} = "#([0-9a-fA-F]{{6}})"', src).group(1)
        black = (lum(hx) + 0.05) / 0.05
        white = 1.05 / (lum(hx) + 0.05)
        assert black >= 2.5 and white >= 2.5, f"{name} #{hx}: black {black:.1f}:1, white {white:.1f}:1"
