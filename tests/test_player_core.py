"""L2.53: the Load Video (C2C) player's pure logic (js/_c2c_player_core.js, run under Node): selection <-> widgets,
loop modes incl. ping-pong, J/K/L shuttle, speeds, the ruler and the frame box parser."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
CORE = PACK / "js" / "_c2c_player_core.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")


def _js(expr):
    script = f"import * as P from {json.dumps(CORE.as_uri())};\nconsole.log(JSON.stringify({expr}));"
    p = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_core_has_no_dom_or_comfy_imports():
    src = CORE.read_text(encoding="utf-8")
    assert "import " not in src.split("*/", 1)[1]
    assert "document" not in src and "window" not in src


def test_timecode_and_seconds():
    assert _js("[P.timecode(0, 24), P.timecode(25, 24), P.timecode(24 * 3661 + 5, 24)]") == [
        "00:00:00:00", "00:00:01:01", "01:01:01:05"]
    assert _js("[P.formatSeconds(4.04), P.formatSeconds(12.25), P.formatSeconds(62.5)]") == ["4.04 s", "12.3 s", "1:02.5"]


def test_selection_round_trip_matches_the_loaded_count():
    # 97 frames, every 2nd, skip 10, cap 20 -> 20 frames: 10, 12, ..., 48
    sel = _js("P.selectionFromWidgets(10, 20, 2, 97)")
    assert sel == {"in": 10, "out": 48, "count": 20}
    assert _js("P.widgetsFromSelection(10, 48, 2, 97)") == {"skip": 10, "cap": 20}
    # the whole clip writes cap 0 ("all")
    assert _js("P.widgetsFromSelection(0, 96, 1, 97)") == {"skip": 0, "cap": 0}
    # handles crossed over still give a valid range
    assert _js("P.widgetsFromSelection(40, 30, 1, 97)") == {"skip": 30, "cap": 11}


@pytest.mark.parametrize("mode,frame,dir_,expect", [
    ("selection", 51, 1, {"frame": 10, "dir": 1, "stop": False}),   # wraps to in
    ("selection", 9, -1, {"frame": 50, "dir": -1, "stop": False}),  # backwards wraps to out
    ("pingpong", 51, 1, {"frame": 50, "dir": -1, "stop": False}),   # turns around at out
    ("pingpong", 9, -1, {"frame": 10, "dir": 1, "stop": False}),    # turns around at in
    ("clip", 97, 1, {"frame": 0, "dir": 1, "stop": False}),         # whole clip wraps
    ("once", 97, 1, {"frame": 96, "dir": 1, "stop": True}),         # stops on the last frame
    ("selection", 30, 1, {"frame": 30, "dir": 1, "stop": False}),   # inside: unchanged
])
def test_loop_step(mode, frame, dir_, expect):
    got = _js(f"P.loopStep({{ frame: {frame}, dir: {dir_}, mode: {json.dumps(mode)}, inF: 10, outF: 50, frames: 97 }})")
    assert got == expect


def test_shuttle_and_speeds():
    seq = _js("(() => { let r = 0; const out = []; for (const k of ['l','l','l','l','k','j','j','l']) "
              "{ r = P.shuttle(r, k); out.push(r); } return out; })()")
    assert seq == [1, 2, 4, 4, 0, -1, -2, 1]
    assert _js("[P.nextSpeed(1, 1), P.nextSpeed(4, 1), P.nextSpeed(0.25, -1), P.nextSpeed(7, -1)]") == [2, 4, 0.25, 0.5]
    assert _js("P.LOOP_MODES.map((m) => P.nextLoopMode(m))") == ["clip", "once", "pingpong", "selection"]


def test_ruler_labels_are_spaced_and_inside_the_clip():
    for frames, width, unit in ((97, 400, "frames"), (2400, 300, "frames"), (97, 400, "time"), (14400, 600, "time"), (2, 50, "frames")):
        ticks = _js(f"P.rulerTicks({frames}, 24, {width}, {json.dumps(unit)})")
        majors = [t for t in ticks if t["major"]]
        assert majors and majors[0]["frame"] == 0
        assert all(0 <= t["frame"] <= frames - 1 + 1e-6 for t in ticks)
        if len(majors) > 1:
            px = (majors[1]["frame"] - majors[0]["frame"]) * width / (frames - 1)
            assert px >= 64 - 1e-6, (frames, width, unit, px)
        assert all(t["label"] for t in majors)
        assert len(ticks) < 400


def test_frame_box_parser():
    assert _js("[P.parseFrameInput('120', 24), P.parseFrameInput('00:00:01:12', 24), P.parseFrameInput('1:12', 24),"
               " P.parseFrameInput('', 24), P.parseFrameInput('abc', 24)]") == [120, 36, 36, None, None]


def test_crop_pixels_match_the_python_handle():
    """The label shows exactly the pixels nodes/c2c_video/handle.py crops (Python round = half to even)."""
    cases = [((0.25, 0.125, 0.5, 0.5), 256, 256), ((0.1, 0.1, 0.8, 0.8), 1920, 1080), ((0.0, 0.0, 1.0, 1.0), 640, 360),
             ((0.5 / 7, 0.3, 0.5, 0.25), 7, 10), ((0.9, 0.9, 0.5, 0.5), 100, 100)]
    for (x, y, w, h), W, H in cases:
        got = _js(f"P.cropPixels({{ x: {x}, y: {y}, w: {w}, h: {h} }}, {W}, {H})")
        fx, fy = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        fw, fh = min(max(w, 0.0), 1.0 - fx), min(max(h, 0.0), 1.0 - fy)
        px, py = int(round(fx * W)), int(round(fy * H))
        pw = max(1, min(W - px, int(round(fw * W))))
        ph = max(1, min(H - py, int(round(fh * H))))
        full = (px, py, pw, ph) == (0, 0, W, H) or fw <= 0 or fh <= 0
        exp = (0, 0, W, H) if full else (px, py, pw, ph)
        assert (got["x"], got["y"], got["w"], got["h"]) == exp, ((x, y, w, h), W, H)
        assert got["full"] == full


def test_aspect_presets_and_fit():
    assert _js("[P.presetRatio('16:9', 1, 1), P.presetRatio('Original', 1920, 1080), P.presetRatio('Free', 9, 9)]") == [
        16 / 9, 16 / 9, None]
    r = _js("P.fitAspect(1, 1920, 1080)")              # square in a 16:9 frame: full height, centred
    assert abs(r["h"] - 1) < 1e-9 and abs(r["w"] * 1920 - 1080) < 1e-6 and abs(r["x"] - (1 - r["w"]) / 2) < 1e-9
    r = _js("P.fitAspect(16 / 9, 1080, 1920)")         # 16:9 in a portrait frame: full width
    assert abs(r["w"] - 1) < 1e-9 and abs(r["h"] * 1920 - 1080 * 9 / 16) < 1e-6


def test_drag_rect_stays_inside_and_keeps_the_ratio():
    # free: a corner drag past the frame edge clamps at the edge
    r = _js("P.dragRect({ x: 0.25, y: 0.25, w: 0.5, h: 0.5 }, 'se', 0.6, 0.6)")
    assert abs(r["x"] - 0.25) < 1e-9 and abs(r["x"] + r["w"] - 1) < 1e-9 and abs(r["y"] + r["h"] - 1) < 1e-9
    # move clamps inside
    r = _js("P.dragRect({ x: 0.25, y: 0.25, w: 0.5, h: 0.5 }, 'move', -1, 2)")
    assert r == {"x": 0, "y": 0.5, "w": 0.5, "h": 0.5}
    # locked 16:9 on a 1920x1080 frame (fractions w/h == 1): every drag keeps w == h in fractions, inside the frame
    for part, dx, dy in (("se", 0.3, 0.1), ("nw", -0.4, -0.1), ("e", 0.2, 0), ("n", 0, -0.3), ("sw", -0.9, 0.9)):
        r = _js(f"P.dragRect({{ x: 0.3, y: 0.3, w: 0.4, h: 0.4 }}, {json.dumps(part)}, {dx}, {dy}, 16 / 9, 1920, 1080)")
        assert abs(r["w"] - r["h"]) < 1e-9, (part, r)
        assert r["x"] >= -1e-9 and r["y"] >= -1e-9 and r["x"] + r["w"] <= 1 + 1e-9 and r["y"] + r["h"] <= 1 + 1e-9, (part, r)
