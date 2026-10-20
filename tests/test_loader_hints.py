"""L2.53 (owner A14): Load Video (C2C) settings UX logic (js/_c2c_loader_hints.js, run under Node) - frame-grid
stepping, grey real values, the details line, seconds <-> frames follow, the size fields leaving a preset."""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
HINTS = PACK / "js" / "_c2c_loader_hints.js"
LOADER = PACK / "js" / "c2c_video_loader.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")


def _js(expr):
    script = f"import * as H from {json.dumps(HINTS.as_uri())};\nconsole.log(JSON.stringify({expr}));"
    p = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def _python_presets():
    tree = ast.parse((PACK / "nodes" / "c2c_video" / "nodes_load.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "FORMAT_PRESETS":
            return ast.literal_eval(node.value)
    raise AssertionError("FORMAT_PRESETS not found")


def test_hints_module_is_pure():
    src = HINTS.read_text(encoding="utf-8")
    assert "import " not in src.split("*/", 1)[1]
    assert "document" not in src and "window" not in src and "app." not in src


@needs_node
def test_format_tables_mirror_the_python_presets():
    py = _python_presets()
    grid = _js("H.FORMAT_GRID")
    rate = _js("H.FORMAT_RATE")
    assert set(grid) == set(py)
    for name, spec in py.items():
        assert grid[name] == (list(spec["frames"]) if "frames" in spec else None), name
        assert rate.get(name) == spec.get("rate"), name


@needs_node
@pytest.mark.parametrize("value,prev,expect", [
    (17, 0, 22),     # arrow up from "all" (step2 = 17): the first legal H3 count above
    (39, 22, 39),    # already legal
    (5, 22, 5),      # arrow down
    (0, 5, 0),       # down past the smallest: back to "all"
    (30, 0, 22),     # typed: nearest (ties go down)
    (31, 0, 39),
    (3, 0, 5),       # typed below the smallest legal count: the smallest
    (23, 22, 39),    # one frame up (a drag): next legal
    (21, 22, 5),     # one frame down
])
def test_snap_cap_minimax_h3(value, prev, expect):
    assert _js(f"H.snapCap({value}, {prev}, [17, 5])") == expect


@needs_node
def test_snap_cap_without_a_grid_and_ltxv():
    assert _js("[H.snapCap(23, 0, null), H.snapCap(10, 0, [8, 1]), H.snapCap(8, 0, [8, 1]), H.capStep('LTXV'), "
               "H.capStep('None')]") == [23, 9, 9, 1, 1]


@needs_node
def test_typed_and_arrow_counts_survive_the_frontends_int_rounding():
    # frontend 1.53.10 onValueChange rounds to a multiple of step2 BEFORE the node's callback (measured live,
    # docs/evidence/L2.53/v4k_live.json C2: step2 = 17 turned a typed 60 into 73)
    got = _js("(() => { const t = H.capStep('MiniMax H3'); const g = [17, 5];"
              " const fe = (v) => t === 1 ? Math.round(v) : Math.round(v / t) * t;"
              " return [H.snapCap(fe(60), 39, g), H.snapCap(fe(47), 39, g), H.snapCap(fe(0 + t), 0, g),"
              " H.snapCap(fe(5 + t), 5, g), H.snapCap(fe(22 + t), 22, g), H.snapCap(fe(39 - t), 39, g)]; })()")
    assert got == [56, 39, 5, 22, 39, 22]


@needs_node
def test_grey_hints_are_the_probes_numbers():
    load = {"frames": 141, "duration": 5.875, "width": 1280, "height": 720, "fps": 24, "start_seconds": 0,
            "end_seconds": 5.875}
    h = _js(f"H.hints({{frame_load_cap: 158, seconds_cap: 0, custom_width: 0, custom_height: 0, force_rate: 0, "
            f"resolution: '720p', skip_first_frames: 3, select_every_nth: 1}}, {json.dumps(load)})")
    assert h["frame_load_cap"] == {"shown": 141, "differs": True, "canReset": True}       # "141 <- 158"
    assert h["seconds_cap"] == {"shown": "5.8", "differs": True, "canReset": False}
    assert h["custom_width"]["shown"] == 1280 and h["custom_width"]["canReset"] is True   # a preset is set
    assert h["force_rate"] == {"shown": 24, "differs": True, "canReset": False}
    assert h["skip_first_frames"]["canReset"] is True and h["select_every_nth"]["canReset"] is False
    same = _js(f"H.hints({{frame_load_cap: 141}}, {json.dumps(load)})")
    assert same["frame_load_cap"]["differs"] is False


@needs_node
def test_follow_rules():
    assert _js("[H.capEdit('frame_load_cap', 65), H.capEdit('seconds_cap', 2.5), H.capEdit('seconds_cap', 0)]") == [
        {"seconds_cap": 0}, {"frame_load_cap": 0}, None]
    assert _js("[H.sizeEdit('custom_width', {resolution: '1080p'}), H.sizeEdit('custom_height', {resolution: 'custom'})]"
               ) == [{"resolution": "custom", "custom_height": 0}, None]


@needs_node
def test_details_line_and_rate_note():
    probe = {"file": "shot.mp4", "width": 1920, "height": 1080, "fps": 30, "duration": 10, "frame_count": 300,
             "audio": None,
             "load": {"frames": 56, "duration": 1.8666, "start_seconds": 1.5, "end_seconds": 3.3666, "width": 1280,
                      "height": 704, "fps": 30, "audio": "silence",
                      "decoder": {"asked": "auto", "used": "fast (ffmpeg, identical conversion)", "note": ""}}}
    lines = _js(f"H.describe({json.dumps(probe)})")
    assert lines[0] == "shot.mp4  1920x1080  30 fps  10.0s  300 frames  no audio"
    assert lines[1] == "loads 1.5s to 3.3s (1.8s, 56 frames) at 1280x704, 30 fps"
    assert lines[2] == "decoder: fast (ffmpeg, identical conversion)"
    assert "silent" in lines[3]
    assert _js(f"H.rateNote('MiniMax H3', {json.dumps(probe['load'])}, 1)") == (
        "MiniMax H3 expects 24 fps; this loads 30 fps (set force_rate to 24)")
    assert _js("H.rateNote('MiniMax H3', {fps: 12}, 2)") == ""          # 24 fps read, every 2nd frame


def test_loader_wires_the_ux_only_into_our_nodes():
    js = LOADER.read_text(encoding="utf-8")
    assert js.count("settingsUx: true") == 2                            # Load Video (C2C) + Load Video Path (C2C)
    assert 'from "./_c2c_loader_hints.js"' in js
    # the preview (the whole clip) never carries the load-only settings: moving the crop must not reload it
    assert "for (const name of LOAD_ONLY_WIDGETS) d.delete(name);" in js
    assert '"#aaa"' in js and "var(--" not in js.split("function drawExtras", 1)[1].split("\n}\n", 1)[0]
