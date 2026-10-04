"""_c2c_runtime.js - executable contract tests under Node (no browser).

The runtime is what every C2C front-end hook leans on to never take ComfyUI
down, so its guarantees are pinned here:
* safePatch never runs core's method twice, falls back to core when our wrapper
  throws, and removes itself (restoring core) after repeated errors;
* every() pauses when the tab is hidden and its circuit breaker really trips;
* requestRedraw coalesces any number of requests into one foreground redraw
  per frame (the progress-HUD storm: 2,541 fg+bg redraws in 20 s);
* a browser that has never been probed runs FULL; only a probe that saw
  software rendering selects LITE.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
RUNTIME = PACK / "js" / "_c2c_runtime.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")

_PRELUDE = r"""
globalThis.window = globalThis;
const _ls = new Map(Object.entries(globalThis.__LS__ || {}));
globalThis.localStorage = { getItem: (k) => (_ls.has(k) ? _ls.get(k) : null), setItem: (k, v) => _ls.set(k, String(v)) };
globalThis.document = { hidden: false, documentElement: { classList: { _s: new Set(),
    toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); }, contains(c) { return this._s.has(c); } } },
    addEventListener() {} };
const _rafq = [];
globalThis.requestAnimationFrame = (fn) => { _rafq.push(fn); return _rafq.length; };
globalThis.__flushFrame = () => { const q = _rafq.splice(0); for (const f of q) f(performance.now()); };
const _timers = [];
globalThis.setTimeout = (fn, ms) => { _timers.push({ fn, ms }); return _timers.length; };
globalThis.clearTimeout = () => {};
globalThis.__runTimers = (n = 1) => { for (let i = 0; i < n; i++) { const t = _timers.shift(); if (t) t.fn(); } };
globalThis.__timerCount = () => _timers.length;
globalThis.__redraws = [];
globalThis.__listeners = {};
globalThis.__reports = [];
"""

_STUBS = {
    'import { app } from "../../scripts/app.js";':
        "const app = { canvas: { setDirty(fg, bg) { globalThis.__redraws.push([fg, bg]); } },"
        " extensions: [], registerExtension() {} };",
    'import { api } from "../../scripts/api.js";':
        "const api = { addEventListener(t, fn) { (globalThis.__listeners[t] ||= []).push(fn); } };",
    'import { reportFailure } from "./_c2c_report.js";':
        "const reportFailure = (where, err) => globalThis.__reports.push([where, String(err && err.message || err)]);",
}


def _run(tmp_path: Path, body: str, ls: dict | None = None) -> dict:
    src = RUNTIME.read_text(encoding="utf-8")
    for old, new in _STUBS.items():
        assert old in src, f"runtime import changed: {old}"
        src = src.replace(old, new)
    mod = tmp_path / "runtime.mjs"
    mod.write_text(src, encoding="utf-8")
    probe = tmp_path / "probe.mjs"
    probe.write_text(
        f"globalThis.__LS__ = {json.dumps(ls or {})};\n" + _PRELUDE
        + f'const {{ getRuntime }} = await import({json.dumps(mod.resolve().as_uri())});\n'
        + "const rt = getRuntime(); const out = {};\n" + body
        + "\nconsole.log(JSON.stringify(out));\n",
        encoding="utf-8")
    r = subprocess.run([NODE, str(probe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_safepatch_never_runs_core_twice(tmp_path):
    out = _run(tmp_path, r"""
      const obj = { calls: 0, draw(x) { this.calls++; return x * 2; } };
      rt.safePatch(obj, "draw", (orig) => function (x) { const r = orig.call(this, x); throw new Error("ours broke after core ran"); }, { id: "t1" });
      out.result = obj.draw(21);
      out.calls = obj.calls;
    """)
    assert out == {"result": 42, "calls": 1}


def test_safepatch_falls_back_when_wrapper_fails_before_core(tmp_path):
    out = _run(tmp_path, r"""
      const obj = { calls: 0, draw(x) { this.calls++; return x + 1; } };
      rt.safePatch(obj, "draw", (orig) => function (x) { throw new Error("ours broke first"); }, { id: "t2" });
      out.result = obj.draw(1);
      out.calls = obj.calls;
    """)
    assert out == {"result": 2, "calls": 1}


def test_safepatch_removes_itself_after_repeated_errors(tmp_path):
    out = _run(tmp_path, r"""
      const obj = { draw() { return "core"; } };
      const orig = obj.draw;
      rt.safePatch(obj, "draw", () => function () { throw new Error("always"); }, { id: "t3", maxErrors: 3 });
      for (let i = 0; i < 3; i++) obj.draw();
      out.restored = obj.draw === orig;
      out.registered = rt.patches().has("t3");
      out.lastReport = globalThis.__reports.at(-1)[1];
    """)
    assert out["restored"] is True and out["registered"] is False
    assert "core behaviour restored" in out["lastReport"]


def test_every_breaker_trips_and_stops_calling(tmp_path):
    out = _run(tmp_path, r"""
      let n = 0;
      rt.every("bad", 100, () => { n++; throw new Error("boom"); });
      __runTimers(12);
      out.calls = n;
      out.disabled = globalThis.__reports.some(r => r[1].includes("disabled after repeated errors"));
    """)
    assert out["disabled"] is True
    assert out["calls"] == 5          # the default breaker: 5 errors, then no more calls


def test_every_pauses_while_hidden(tmp_path):
    out = _run(tmp_path, r"""
      let n = 0;
      rt.every("t", 100, () => { n++; });
      document.hidden = true; __runTimers(5); out.hidden = n;
      document.hidden = false; __runTimers(2); out.visible = n;
    """)
    assert out == {"hidden": 0, "visible": 2}


def test_every_starts_no_timer_until_used(tmp_path):
    out = _run(tmp_path, "out.timers = __timerCount();")
    assert out["timers"] == 0


def test_redraw_requests_coalesce_to_one_foreground_redraw(tmp_path):
    out = _run(tmp_path, r"""
      for (let i = 0; i < 50; i++) rt.requestRedraw();
      __flushFrame();
      out.fg = globalThis.__redraws.slice();
      rt.requestRedraw(); rt.requestRedraw({ bg: true }); __flushFrame();
      out.mixed = globalThis.__redraws.slice(1);
    """)
    assert out["fg"] == [[True, False]]
    assert out["mixed"] == [[True, True]]


def test_redraws_capped_while_running(tmp_path):
    out = _run(tmp_path, r"""
      for (const f of globalThis.__listeners["execution_start"]) f({});
      for (let i = 0; i < 40; i++) { rt.requestRedraw(); __flushFrame(); }
      out.redraws = globalThis.__redraws.length;
      out.runningClass = document.documentElement.classList.contains("c2c-running");
    """)
    assert out["runningClass"] is True
    assert out["redraws"] <= 15


@pytest.mark.parametrize("ls,expected", [
    ({}, "full"),                                                       # never probed: normal behaviour
    ({"c2c.perf.gpu": json.dumps({"software": False})}, "full"),
    ({"c2c.perf.gpu": json.dumps({"software": True, "renderer": "llvmpipe"})}, "lite"),
    ({"c2c.perf.mode": "Full", "c2c.perf.gpu": json.dumps({"software": True})}, "full"),
    ({"c2c.perf.mode": "Lite"}, "lite"),
    ({"c2c.lite": "1"}, "lite"),                                         # migrated from the old toggle
])
def test_tier_decision(tmp_path, ls, expected):
    assert _run(tmp_path, "out.tier = rt.tier();", ls)["tier"] == expected


def test_run_state_ends_only_when_queue_is_empty(tmp_path):
    out = _run(tmp_path, r"""
      const fire = (t, d) => { for (const f of globalThis.__listeners[t] || []) f({ detail: d }); };
      fire("execution_start", {});
      fire("status", { exec_info: { queue_remaining: 2 } });
      fire("executing", { node: null });
      out.midQueue = rt.runState.isRunning();
      fire("status", { exec_info: { queue_remaining: 0 } });
      out.after = rt.runState.isRunning();
    """)
    assert out == {"midQueue": True, "after": False}


def test_capped_redraws_keep_one_pending_retry(tmp_path):
    """While capped, 100 more requests must not make 100 timers."""
    out = _run(tmp_path, r"""
      for (const f of globalThis.__listeners["execution_start"]) f({});
      for (let i = 0; i < 20; i++) { rt.requestRedraw(); __flushFrame(); }
      const before = __timerCount();
      for (let i = 0; i < 100; i++) rt.requestRedraw();
      out.newTimers = __timerCount() - before;
    """)
    assert out["newTimers"] <= 1


def test_quiet_controller_pauses_and_resumes_our_animations(tmp_path):
    out = _run(tmp_path, r"""
      globalThis.CSSAnimation = class CSSAnimation {};
      // real CSSAnimation instances: the runtime (rightly) ignores anything else
      const ours = Object.assign(new CSSAnimation(), { animationName: "c2c-dash", playState: "running",
        pause() { this.playState = "paused"; globalThis.__paused.push("c2c-dash"); },
        play() { this.playState = "running"; globalThis.__played.push("c2c-dash"); } });
      const core = Object.assign(new CSSAnimation(), { animationName: "comfy-spinner", playState: "running",
        pause() { this.playState = "paused"; globalThis.__paused.push("core"); },
        play() { this.playState = "running"; globalThis.__played.push("core"); } });
      globalThis.__paused = [];
      globalThis.__played = [];
      globalThis.__anims = [ours, core];
      document.getAnimations = () => globalThis.__anims;
      document.removeEventListener = () => {};
      document.addEventListener = (t, fn, cap) => {
        if (t === "animationstart") globalThis.__animStart = fn;
      };
      for (const f of globalThis.__listeners["execution_start"]) f({});
      out.paused = globalThis.__paused.slice();
      out.quiet = document.documentElement.classList.contains("c2c-quiet");
      for (const f of globalThis.__listeners["execution_success"]) f({});
      out.played = globalThis.__played.slice();
      out.afterQuiet = document.documentElement.classList.contains("c2c-quiet");
    """)
    assert out["quiet"] is True
    assert "c2c-dash" in out["paused"]
    assert "core" not in out["paused"]
    assert out["afterQuiet"] is False
    assert "c2c-dash" in out["played"]


def test_quiet_controller_inert_without_get_animations(tmp_path):
    out = _run(tmp_path, r"""
      delete document.getAnimations;
      for (const f of globalThis.__listeners["execution_start"]) f({});
      out.quiet = document.documentElement.classList.contains("c2c-quiet");
      out.threw = false;
      try { rt.onQuiet(() => {}); } catch (e) { out.threw = true; }
    """)
    assert out["quiet"] is True
    assert out["threw"] is False
