"""c2c_browser_guard.js — Node probe harness (no browser required)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
GUARD = PACK / "js" / "c2c_browser_guard.js"
NODE = shutil.which("node")

_PRELUDE = r"""
globalThis.window = globalThis;
globalThis.sessionStorage = { _m: new Map(), getItem(k){return this._m.get(k)||null;}, setItem(k,v){this._m.set(k,v);} };
globalThis.document = {
  createElement(tag) {
    const el = {
      tagName: tag.toUpperCase(),
      width: 0, height: 0,
      style: {},
      isConnected: true,
      _handlers: {},
      addEventListener(ev, fn) { this._handlers[ev] = fn; },
      getContext(type) {
        if (type === "2d") return globalThis.__TEST_2D_CTX__;
        return null;
      },
    };
    return el;
  },
  hidden: false,
};
globalThis.HTMLCanvasElement = function HTMLCanvasElement() {};
globalThis.HTMLCanvasElement.prototype = {
  getContext(type, attrs) { return globalThis.__ORIG_GET_CONTEXT__(this, type, attrs); },
};
globalThis.__ORIG_GET_CONTEXT__ = function(canvas, type) {
  if (/webgl/i.test(type)) return globalThis.__TEST_WEBGL_CTX__;
  return null;
};
globalThis.__TEST_WEBGL_CTX__ = {
  isContextLost: () => false,
  getExtension(name) {
    if (name === "WEBGL_debug_renderer_info") return { UNMASKED_RENDERER_WEBGL: globalThis.__TEST_RENDERER__ || "NVIDIA" };
    if (name === "WEBGL_lose_context") return { loseContext: () => { globalThis.__LOSE_CALLS__ = (globalThis.__LOSE_CALLS__||0)+1; } };
    return null;
  },
  getParameter(p) { return globalThis.__TEST_RENDERER__ || "NVIDIA"; },
  canvas: null,
};
globalThis.__TEST_2D_CTX__ = null;
globalThis.__TOASTS__ = [];
globalThis.app = {
  extensionManager: { toast: { add(o) { globalThis.__TOASTS__.push(o); } } },
  ui: { settings: { getSettingValue() { return 0; } } },
  registerExtension(ext) { globalThis.__REGISTERED_EXT__ = ext; if (ext.setup) ext.setup(); },
};
globalThis.fetch = async () => ({ ok: true });
globalThis.clearTimeout = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
globalThis.console = { ...console, debug() {}, warn() {} };
"""


def _guard_copy(tmp_path: Path) -> str:
    """The guard with ComfyUI's app import swapped for the stub in the prelude
    (../../scripts/app.js only exists inside a running ComfyUI)."""
    src = GUARD.read_text(encoding="utf-8").replace(
        'import { app } from "../../scripts/app.js";', "const app = globalThis.app;")
    # The guard patches getContext through the REAL runtime (safePatch); put
    # it beside the guard with ComfyUI's imports stubbed, as test_c2c_runtime does.
    runtime = (GUARD.parent / "_c2c_runtime.js").read_text(encoding="utf-8")
    for old, new in {
        'import { app } from "../../scripts/app.js";': "const app = globalThis.app;",
        'import { api } from "../../scripts/api.js";': "const api = { addEventListener() {} };",
        'import { reportFailure } from "./_c2c_report.js";': "const reportFailure = () => {};",
    }.items():
        assert old in runtime, f"runtime import changed: {old}"
        runtime = runtime.replace(old, new)
    (tmp_path / "_c2c_runtime.js").write_text(runtime, encoding="utf-8")
    dst = tmp_path / "c2c_browser_guard.mjs"
    dst.write_text(src, encoding="utf-8")
    return dst.resolve().as_uri()


def _run(tmp_path: Path, body: str, gc: bool = False) -> dict:
    probe = tmp_path / "probe_browser_guard.mjs"
    uri = _guard_copy(tmp_path)
    lines = [
        "delete globalThis.__C2C_BROWSER_GUARD__;",
        "const M1 = await import('" + uri + "');",
    ]
    probe.write_text(_PRELUDE + "\n".join(lines) + "\n" + body, encoding="utf-8")
    args = [NODE] + (["--expose-gc"] if gc else []) + [str(probe)]
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[:800]
    return json.loads(p.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_getcontext_wrapped_once_on_double_load(tmp_path):
    uri = _guard_copy(tmp_path)
    probe = tmp_path / "probe_double.mjs"
    lines = [
        "delete globalThis.__C2C_BROWSER_GUARD__;",
        "await import('" + uri + "');",
        "const after1 = HTMLCanvasElement.prototype.getContext;",
        "await import('" + uri + "?again');",   # a second copy of the module
        "const after2 = HTMLCanvasElement.prototype.getContext;",
        "process.stdout.write(JSON.stringify({ same: after1 === after2, "
        "wrapped: after1 !== globalThis.__ORIG_GET_CONTEXT__ }));",
    ]
    probe.write_text(_PRELUDE + "\n".join(lines) + "\n", encoding="utf-8")
    p = subprocess.run([NODE, str(probe)], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[:800]
    assert json.loads(p.stdout) == {"same": True, "wrapped": True}


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_contextlost_prevent_default_and_toast_once(tmp_path):
    out = _run(tmp_path, r"""
const G = globalThis.__C2C_BROWSER_GUARD__;
const canvas = document.createElement("canvas");
const ctx = globalThis.__TEST_WEBGL_CTX__;
ctx.canvas = canvas;
G.trackWebGLContext(canvas, ctx);
let prevented = 0;
const fn = canvas._handlers["webglcontextlost"];
const e = { preventDefault() { prevented++; } };
fn(e); fn(e);
process.stdout.write(JSON.stringify({ prevented, toasts: globalThis.__TOASTS__.length }));
""")
    assert out["prevented"] >= 1
    assert out["toasts"] == 1


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_janitor_never_touches_connected_contexts(tmp_path):
    out = _run(tmp_path, r"""
const G = globalThis.__C2C_BROWSER_GUARD__;
globalThis.__LOSE_CALLS__ = 0;
const keep = [];
for (let i = 0; i < 14; i++) {
  const canvas = document.createElement("canvas");
  canvas.isConnected = true;
  const ctx = { ...globalThis.__TEST_WEBGL_CTX__, isContextLost: () => false, canvas };
  keep.push(ctx, canvas);
  G.live.add({ ctxRef: new WeakRef(ctx), canvasRef: new WeakRef(canvas), disconnectedPasses: 5 });
}
G.runJanitor();
process.stdout.write(JSON.stringify({ lose: globalThis.__LOSE_CALLS__, live: G.live.size }));
""")
    assert out["lose"] == 0
    assert out["live"] == 14


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_janitor_releases_only_double_disconnected_orphans(tmp_path):
    out = _run(tmp_path, r"""
const G = globalThis.__C2C_BROWSER_GUARD__;
globalThis.__LOSE_CALLS__ = 0;
const keep = [];
const mk = (connected, passes) => {
  const canvas = document.createElement("canvas");
  canvas.isConnected = connected;
  const ctx = { ...globalThis.__TEST_WEBGL_CTX__, isContextLost: () => false, canvas };
  keep.push(ctx, canvas);
  const entry = { ctxRef: new WeakRef(ctx), canvasRef: new WeakRef(canvas), disconnectedPasses: passes };
  G.live.add(entry);
  return entry;
};
for (let i = 0; i < 11; i++) mk(true, 0);      // on screen
mk(false, 0);                                    // removed from the page
G.runJanitor();                                  // 1st pass: could be a re-mount
const afterOne = globalThis.__LOSE_CALLS__;
G.runJanitor();                                  // 2nd consecutive pass: orphaned
process.stdout.write(JSON.stringify({ afterOne, afterTwo: globalThis.__LOSE_CALLS__, live: G.live.size }));
""")
    assert out["afterOne"] == 0, "a single disconnected pass may be a re-mount"
    assert out["afterTwo"] == 1, "only the orphan is released"
    assert out["live"] == 11


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_farbling_detected_when_readback_differs(tmp_path):
    out = _run(tmp_path, r"""
const G = globalThis.__C2C_BROWSER_GUARD__;
const W=64,H=64;
const expected = new Uint8ClampedArray(W*H*4);
for (let y=0;y<H;y++) for (let x=0;x<W;x++) {
  const i=(y*W+x)*4; expected[i]=(x*4)&255; expected[i+1]=(y*4)&255; expected[i+2]=128; expected[i+3]=255;
}
globalThis.__TEST_2D_CTX__ = {
  fillStyle:"", fillRect(){},
  getImageData(){ return { data: expected } },
};
const ok = G.checkCanvasFarbling();
expected[0] = 1;
globalThis.__TEST_2D_CTX__.getImageData = () => ({ data: expected });
const bad = G.checkCanvasFarbling();
process.stdout.write(JSON.stringify({ ok, bad }));
""")
    assert out["ok"] is False
    assert out["bad"] is True


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_software_renderer_toast_once(tmp_path):
    out = _run(tmp_path, r"""
globalThis.__TEST_RENDERER__ = "Google SwiftShader";
const G = globalThis.__C2C_BROWSER_GUARD__;
const canvas = document.createElement("canvas");
const ctx = globalThis.__TEST_WEBGL_CTX__;
G.trackWebGLContext(canvas, ctx);
G.trackWebGLContext(canvas, ctx);
process.stdout.write(JSON.stringify({ toasts: globalThis.__TOASTS__.length, detail: globalThis.__TOASTS__[0]?.detail || "" }));
""")
    assert out["toasts"] == 1
    assert "hardware acceleration" in out["detail"].lower() or "software" in out["detail"].lower()


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_tracking_never_keeps_a_context_alive(tmp_path):
    """The guard must not be the leak it exists to prevent: an orphaned context
    that nothing else references is collected, and the tracker forgets it."""
    out = _run(tmp_path, r"""
const G = globalThis.__C2C_BROWSER_GUARD__;
(function () {
  const canvas = document.createElement("canvas");
  const ctx = { ...globalThis.__TEST_WEBGL_CTX__, isContextLost: () => false, canvas };
  G.trackWebGLContext(canvas, ctx);
})();
const before = G.live.size;
await new Promise((r) => setImmediate(r));
globalThis.gc(); await new Promise((r) => setImmediate(r)); globalThis.gc();
G.runJanitor();
process.stdout.write(JSON.stringify({ before, after: G.live.size }));
""", gc=True)
    assert out["before"] == 1
    assert out["after"] == 0
