"""The progress HUD must not redraw the canvas at 60 fps for a whole generation.

Each frame of its animation loop requests a foreground-only redraw via
runtime.requestRedraw() — never background invalidation. It used to call
dirtyAllGraphs() which forced full bg blits. The loop now runs at full rate
only while a bar glides to a new value (~12 fps otherwise for shimmer + ETA).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HUD = Path(__file__).resolve().parents[1] / "js" / "c2c_progress_hud.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

HARNESS = r"""
let clock = 0, rafQ = [], timeQ = [], fgRedraws = 0, bgRedraws = 0;
globalThis.requestAnimationFrame = (f) => { rafQ.push(f); return rafQ.length; };
globalThis.setTimeout = (f, ms) => { timeQ.push({ f, at: clock + ms }); return timeQ.length; };
globalThis.document = { hidden: false, addEventListener() {} };
const PROGRESS = new Map();
const _now = () => clock;
globalThis.__c2cRuntime = {
  requestRedraw({ bg = false } = {}) {
    if (bg) bgRedraws++;
    else fgRedraws++;
  },
};
function getRuntime() { return globalThis.__c2cRuntime; }
function _requestFgRedraw() { getRuntime().requestRedraw(); }
%LOOP%
// advance simulated time one display frame (16 ms) at a time
const run = (ms) => { const end = clock + ms; while (clock < end) { clock += 16;
  const due = timeQ.filter((t) => t.at <= clock); timeQ = timeQ.filter((t) => t.at > clock);
  due.forEach((t) => t.f()); const q = rafQ; rafQ = []; q.forEach((f) => f(clock)); } };
PROGRESS.set("5", { value: 10, max: 20, smoothed: 0 });
_ensure_anim();
run(1000); const glide = fgRedraws; const glideBg = bgRedraws;
fgRedraws = 0; bgRedraws = 0; run(10000); const steady = fgRedraws; const steadyBg = bgRedraws;
PROGRESS.get("5").value = 11;
fgRedraws = 0; bgRedraws = 0; run(1000); const nextStep = fgRedraws; const nextStepBg = bgRedraws;
PROGRESS.clear();
fgRedraws = 0; bgRedraws = 0; run(1000); const done = fgRedraws; const doneBg = bgRedraws;
process.stdout.write(JSON.stringify({ glide, glideBg, steady, steadyBg, nextStep, nextStepBg, done, doneBg }));
"""


def _run() -> dict:
    src = HUD.read_text(encoding="utf-8")
    start = src.index("const IDLE_FRAME_MS")
    end = src.index("try {\n    document.addEventListener")
    loop = src[start:end]
    probe = Path(__import__("tempfile").mkdtemp()) / "hud.mjs"
    probe.write_text(HARNESS.replace("%LOOP%", loop), encoding="utf-8")
    r = subprocess.run([NODE, str(probe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[:800]
    return json.loads(r.stdout)


def test_steady_progress_redraws_at_about_12_fps_not_60():
    out = _run()
    # 10 s at 60 fps was ~625 redraws; ~12 fps is ~100-120
    assert out["steady"] <= 150, out
    assert out["steady"] >= 60, "the shimmer and ETA must keep moving"
    assert out["steadyBg"] == 0, "progress path must not request background invalidation"


def test_a_new_value_glides_at_full_rate_and_the_loop_ends_with_the_run():
    out = _run()
    assert out["glide"] >= 30, out
    assert out["glideBg"] == 0, out
    assert out["nextStep"] >= 12, out
    assert out["nextStepBg"] == 0, out
    assert out["done"] <= 1, out
    assert out["doneBg"] == 0, out
