"""Video Comparer must release what it decodes.

A <video> keeps its decoder and buffered frames alive after the last reference
is dropped, and a PLAYING one keeps decoding: the comparer used to leak one per
source swap and leave both running after the node was deleted. It also asked
for a CPU-backed visible canvas it never reads back, and allocated a fresh
read-back canvas on every render in the diff modes.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "js" / "video_comparer_c2c.js"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_swapping_a_source_releases_the_old_video_only():
    src = SRC.read_text(encoding="utf-8")
    helpers = src[src.index("function releaseMedia(v)"):src.index("function loadSource(filename)")]
    body = """
const mk = () => ({ calls: [], pause() { this.calls.push("pause"); },
  removeAttribute(a) { this.calls.push("rm:" + a); }, load() { this.calls.push("load"); } });
const oldV = { kind: "video", el: mk() }, img = { kind: "image", el: mk() }, next = { kind: "video", el: mk() };
const r1 = swapSource(oldV, next) === next;
const r2 = swapSource(next, next) === next;          // same source: must stay alive
swapSource(img, null);                                // images have nothing to release
process.stdout.write(JSON.stringify({ r1, r2, old: oldV.el.calls, kept: next.el.calls, img: img.el.calls }));
"""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "vc.mjs"
        f.write_text(helpers + body, encoding="utf-8")
        p = subprocess.run([NODE, str(f)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:800]
    out = json.loads(p.stdout)
    assert out["r1"] and out["r2"]
    assert out["old"] == ["pause", "rm:src", "load"]
    assert out["kept"] == [] and out["img"] == []


def test_every_source_assignment_goes_through_swap_and_removal_releases_all():
    src = SRC.read_text(encoding="utf-8")
    raw = re.findall(r"S\.src[AB] = await loadSource\(", src)
    assert not raw, "a source assigned without swapSource() leaks the previous <video>"
    assert src.count("swapSource(S.src") >= 6
    removed = src[src.index("nodeType.prototype.onRemoved = function"):]
    assert "releaseAll" in removed[:400]


def test_visible_canvas_stays_on_the_gpu_and_the_diff_canvas_is_reused():
    src = SRC.read_text(encoding="utf-8")
    assert 'cvs.getContext("2d", { willReadFrequently: true })' not in src
    assert "S._diffCanvas ||=" in src
    diff = src[src.index('mode === "diff" || mode === "per_channel"'):]
    assert 'document.createElement("canvas")' not in diff[:200].replace("S._diffCanvas ||= document.createElement(\"canvas\")", "")
