"""The shared c2c_ui framework and its byte-identical copies in every pack.

Canonical source: ComfyUI-CustomNodePacks/js/c2c_ui/. Each pack ships an
identical copy because cross-pack imports 404 on a standalone install.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
CANONICAL_DIR = WORKSPACE / "ComfyUI-CustomNodePacks" / "js" / "c2c_ui"
MODULES = sorted(p.name for p in CANONICAL_DIR.glob("*.js")) if CANONICAL_DIR.is_dir() else []
# The adapted Pixaroma code is MIT: its notice must travel with every copy.
SHIPPED = MODULES + ["LICENSE-PIXAROMA.txt"]

COPIES = {
    "ComfyUI-CustomNodePacks": "js",
    "ComfyUI-NukeMaxNodes": "web",
    "ComfyUI-WanNodeExperiments": "web",
    "ComfyUI-MiniMaxSuite": "web",
    "ComfyUI-WanAnimatePreprocessV2": "js",
    "ComfyUI-GLM_Image": "web",
    "ComfyUI-WanAnimalPreprocessor": "web",
}


def _copy_dir(pack: str) -> Path:
    return WORKSPACE / pack / COPIES[pack] / "c2c_ui"


@pytest.mark.parametrize("pack", sorted(COPIES))
def test_every_pack_ships_the_same_c2c_ui(pack):
    dest = _copy_dir(pack)
    if not (WORKSPACE / pack).exists():
        pytest.skip(f"{pack} is not in this workspace")
    assert dest.is_dir(), f"{pack} has no c2c_ui/ directory"
    for name in SHIPPED:
        src = CANONICAL_DIR / name
        copy = dest / name
        assert copy.exists(), f"{pack} is missing c2c_ui/{name}"
        assert copy.read_text(encoding="utf-8") == src.read_text(encoding="utf-8"), (
            f"{pack}'s c2c_ui/{name} has drifted from CustomNodePacks'"
        )


@pytest.mark.parametrize("pack", sorted(COPIES))
def test_the_copy_sits_at_the_web_root_its_pack_declares(pack):
    if not (WORKSPACE / pack).exists():
        pytest.skip(f"{pack} is not in this workspace")
    init = (WORKSPACE / pack / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'WEB_DIRECTORY\s*=\s*["\']\.?/?([\w/]+)["\']', init)
    assert m, f"{pack} declares no WEB_DIRECTORY"
    assert m.group(1).strip("/") == COPIES[pack]


def test_every_import_in_c2c_ui_is_relative():
    bad = []
    for path in CANONICAL_DIR.glob("*.js"):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"""(?:import|export)\s+.*?\s+from\s+['"]([^'"]+)['"]""", text):
            spec = m.group(1)
            if not spec.startswith("./"):
                bad.append(f"{path.name}: {spec}")
        for m in re.finditer(r"""import\s+['"]([^'"]+)['"]""", text):
            spec = m.group(1)
            if not spec.startswith("./"):
                bad.append(f"{path.name}: {spec}")
    assert not bad, "non-relative imports found: " + "; ".join(bad)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_each_module_imports_without_top_level_dom_access():
    for name in MODULES:
        path = (CANONICAL_DIR / name).resolve()
        p = subprocess.run(
            [shutil.which("node"), "--input-type=module", "-e",
             f"import '{path.as_uri()}'"],
            capture_output=True, text=True, timeout=30,
        )
        assert p.returncode == 0, f"{name} failed to import: {p.stderr[:600]}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_node_syntax_check():
    node = shutil.which("node")
    for name in MODULES:
        path = CANONICAL_DIR / name
        p = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, f"{name}: {p.stderr[:400]}"


# Browser globals the modules read AT CALL TIME. The modules are side-effect
# free, so the probes set these first and then import the real files - no
# source slicing.
_PRELUDE = """
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.devicePixelRatio = 2;
globalThis.LiteGraph = { vueNodesMode: false };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.document = {
  getElementById: () => null,
  createElement: () => ({ id: "", textContent: "", style: {}, appendChild() {}, classList: { add() {} } }),
  head: { appendChild() {} },
};
"""


def _run_probe(tmp_path, name, module, body):
    uri = (CANONICAL_DIR / module).resolve().as_uri()
    probe = tmp_path / f"{name}.mjs"
    probe.write_text(_PRELUDE + f'const M = await import("{uri}");\n' + body, encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[:800]
    return json.loads(p.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_adaptive_canvas_only_is_live_not_decided_once(tmp_path):
    """A static canvasOnly:true hides a DOM widget completely in Nodes 2.0;
    the getter must follow the renderer on every read."""
    data = _run_probe(tmp_path, "adaptive", "nodes2.js", """
const w = { options: {} };
M.adaptiveCanvasOnly(w);
const seen = [];
for (const mode of [false, true, false]) { LiteGraph.vueNodesMode = mode; seen.push(w.options.canvasOnly); }
process.stdout.write(JSON.stringify(seen));
""")
    assert data == [True, False, True]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_mount_panel_reserves_the_dom_margin(tmp_path):
    """ComfyUI sizes the element to computedHeight - 4 - (2*margin - 4); the
    panel must reserve exactly that, or it hangs past the node edge."""
    data = _run_probe(tmp_path, "mount", "node_panel.js", """
let opts = null;
const widget = { options: {} };
const node = { addDOMWidget(name, type, root, o) { opts = o; return widget; }, onRemoved: null };
const root = { classList: { add() {} }, style: {}, children: [] };
const w = M.mountPanel(node, "panel", root, { minHeight: 120, margin: 4 });
const slot = w.computeSize(300)[1];
process.stdout.write(JSON.stringify({
  margin: opts.margin, minH: opts.getMinHeight(), slot,
  element: slot - 4 - (opts.margin ? 2 * opts.margin - 4 : 0),
  canvasOnlyLegacy: w.options.canvasOnly,
  cleanup: typeof node.onRemoved === "function",
}));
""")
    assert data == {"margin": 4, "minH": 128, "slot": 128, "element": 120,
                    "canvasOnlyLegacy": True, "cleanup": True}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_canvas_backing_scale_caps_and_never_below_dpr(tmp_path):
    data = _run_probe(tmp_path, "scale", "nodes2.js", """
globalThis.comfyAPI = { app: { app: { canvas: { ds: { scale: 10 } } } } };
const capped = M.canvasBackingScale(1000, 1000);
comfyAPI.app.app.canvas.ds.scale = 0.5;
const floored = M.canvasBackingScale(100, 100);
process.stdout.write(JSON.stringify({ capped, floored }));
""")
    assert data["capped"] == 6          # 6000 / 1000: the long side is capped
    assert data["floored"] == 2         # zoomed out: never below devicePixelRatio


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_on_renderer_change_fires_on_flip_and_stops_polling(tmp_path):
    data = _run_probe(tmp_path, "renderer", "nodes2.js", """
let live = 0;
const si = setInterval, ci = clearInterval;
globalThis.setInterval = (...a) => { live += 1; return si(...a); };
globalThis.clearInterval = (...a) => { live -= 1; return ci(...a); };
const fires = [];
const unsub = M.onRendererChange((v) => fires.push(v));
await new Promise((r) => setTimeout(r, 350));   // no flip yet: must not fire
LiteGraph.vueNodesMode = true;
await new Promise((r) => setTimeout(r, 350));
unsub();
process.stdout.write(JSON.stringify({ live, fires }));
""")
    assert data == {"live": 0, "fires": [True]}


_CHART_PRELUDE = """
globalThis.window = globalThis;
globalThis.devicePixelRatio = 2;
globalThis.LiteGraph = { vueNodesMode: false };
globalThis.comfyAPI = { app: { app: { canvas: { ds: { scale: 1 } } } } };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = (fn) => { fn(); return 1; };
globalThis.cancelAnimationFrame = () => {};
Object.defineProperty(globalThis, "navigator", { value: { clipboard: { writeText: async () => {} } }, configurable: true, writable: true });
globalThis.getComputedStyle = () => ({
  getPropertyValue: (p) => ({
    "--cu-series-1": "#b494ff",
    "--cu-sunken": "#0c0d23",
    "--cu-edge": "#2a2a57",
    "--cu-ink-dim": "#6f6d9b",
    "--cu-ink-soft": "#bab7db",
    "--cu-danger": "#f27a92",
  }[p] || ""),
});
function makeEl(tag = "div") {
  const kids = [];
  const el = {
    tagName: tag.toUpperCase(),
    className: "",
    hidden: false,
    textContent: "",
    style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    dataset: {},
    children: kids,
    appendChild(c) { kids.push(c); c.parentElement = el; return c; },
    listeners: {},
    addEventListener(type, fn) { this.listeners[type] = fn; },
    removeEventListener(type) { delete this.listeners[type]; },
    click() { this.listeners.click?.(); },
    querySelector(sel) {
      const cls = sel.startsWith(".") ? sel.slice(1) : sel;
      if (String(this.className).split(/\s+/).includes(cls)) return this;
      for (const c of this.children) {
        const hit = c.querySelector?.(sel);
        if (hit) return hit;
      }
      return null;
    },
    querySelectorAll(sel) {
      const out = [];
      const cls = sel.startsWith(".") ? sel.slice(1) : sel;
      if (String(this.className).split(/\s+/).includes(cls)) out.push(this);
      for (const c of this.children) {
        const nested = c.querySelectorAll?.(sel) || [];
        for (const n of nested) out.push(n);
      }
      return out;
    },
    getBoundingClientRect() { return { left: 0, top: 0, width: 300, height: 160 }; },
    clientWidth: 300,
    clientHeight: 160,
    parentElement: null,
    setAttribute() {},
    getAttribute() { return null; },
  };
  if (tag === "canvas") {
    el.width = 300; el.height = 160;
    el.getContext = () => ({
      setTransform() {}, clearRect() {}, fillRect() {}, strokeRect() {},
      beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {},
      arc() {}, save() {}, restore() {}, translate() {}, rotate() {},
      fillText() {}, setLineDash() {}, measureText(s) { return { width: String(s).length * 6 }; },
      fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", globalAlpha: 1,
      textAlign: "", textBaseline: "",
    });
  }
  return el;
}
globalThis.document = {
  getElementById: () => null,
  createElement: (tag) => makeEl(tag),
  createElementNS: (_ns, tag) => makeEl(tag),
  head: { appendChild() {} },
  body: { appendChild() {} },
};
"""


def _run_chart_probe(tmp_path, name, body):
    chart_uri = (CANONICAL_DIR / "chart.js").resolve().as_uri()
    probe = tmp_path / f"{name}.mjs"
    probe.write_text(_CHART_PRELUDE + f'const M = await import("{chart_uri}");\n' + body, encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[:800]
    return json.loads(p.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_nice_ticks_unit_interval(tmp_path):
    data = _run_chart_probe(tmp_path, "ticks1", """
process.stdout.write(JSON.stringify(M.niceTicks(0, 1, 5)));
""")
    assert data["step"] == 0.2
    assert data["ticks"] == [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert data["decimals"] == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_nice_ticks_drift_range(tmp_path):
    data = _run_chart_probe(tmp_path, "ticks2", """
process.stdout.write(JSON.stringify(M.niceTicks(0, 3.4, 5)));
""")
    assert 0 < data["step"] <= 1.0
    assert len(data["ticks"]) <= 6
    assert data["ticks"][0] >= 0
    assert data["ticks"][-1] <= 3.4 + data["step"]
    assert data["decimals"] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_nearest_index_edges(tmp_path):
    data = _run_chart_probe(tmp_path, "near", """
const xs = [0, 2, 4, 6];
process.stdout.write(JSON.stringify({
  empty: M.nearestIndex([], 1),
  before: M.nearestIndex(xs, -1),
  exact: M.nearestIndex(xs, 4),
  between: M.nearestIndex(xs, 3.1),
  after: M.nearestIndex(xs, 99),
}));
""")
    assert data == {"empty": -1, "before": 0, "exact": 2, "between": 2, "after": 3}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_series_to_csv_shape(tmp_path):
    data = _run_chart_probe(tmp_path, "csv", """
const csv = M.seriesToCSV("step", [0, 1], [
  { label: "a", values: [1.0, null] },
  { label: "b", values: [2.0, 3.0] },
]);
const lines = csv.split("\\n");
process.stdout.write(JSON.stringify({ header: lines[0], rows: lines.length - 1, row1: lines[1], row2: lines[2] }));
""")
    assert data["header"] == "step,a,b"
    assert data["rows"] == 2
    assert data["row1"] == "0,1,2"      # JavaScript number text: 1.0 prints as 1
    assert data["row2"] == "1,,3"       # a missing value is an empty cell


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_line_chart_build_toggle_and_csv(tmp_path):
    data = _run_chart_probe(tmp_path, "chart", """
const chart = M.lineChart({
  minHeight: 120,
  xLabel: "step",
  series: [{ id: "a", label: "A", color: "--cu-series-1", axis: "y" }],
});
chart.setState("ready");
chart.setData([0, 1], { a: [1, 2] });
const chip = chart.el.querySelector(".c2c-ui-chart__chip");
const before = chart.toCSV().split("\\n").length;
chip.click();
const after = chart.toCSV().split("\\n").length;
chart.destroy();
process.stdout.write(JSON.stringify({ before, after, hasEl: !!chart.el }));
""")
    assert data["hasEl"]
    assert data["before"] >= 2
    assert data["after"] >= 2


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_line_chart_hides_readout_on_pointerleave(tmp_path):
    data = _run_chart_probe(tmp_path, "hover", """
const chart = M.lineChart({
  minHeight: 120,
  xLabel: "step",
  series: [{ id: "a", label: "A", color: "--cu-series-1", axis: "y" }],
});
chart.setState("ready");
chart.setData([0, 1, 2], { a: [1, 2, 3] });
const plotWrap = chart.el.querySelector(".c2c-ui-chart__plot-wrap");
const readout = chart.el.querySelector(".c2c-ui-chart__readout");
plotWrap.listeners.pointermove({ clientX: 150 });
const shown = !readout.hidden;
plotWrap.listeners.pointerleave();
const hidden = readout.hidden;
chart.destroy();
process.stdout.write(JSON.stringify({ shown, hidden }));
""")
    assert data["shown"] is True
    assert data["hidden"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_line_chart_skips_plot_false_legend_chips(tmp_path):
    data = _run_chart_probe(tmp_path, "legend", """
const chart = M.lineChart({
  series: [
    { id: "a", label: "A", color: "--cu-series-1", axis: "y" },
    { id: "carry", label: "carry", color: "--cu-series-2", axis: "y", plot: false, readout: true },
  ],
});
const chips = chart.el.querySelectorAll(".c2c-ui-chart__chip");
const hasCarry = chips.some((c) => (c.children[1]?.textContent || "").includes("carry"));
chart.destroy();
process.stdout.write(JSON.stringify({ count: chips.length, hasCarry }));
""")
    assert data["count"] == 1
    assert data["hasCarry"] is False


_ZOOM_PRELUDE = """
globalThis.window = globalThis;
globalThis.devicePixelRatio = 2;
globalThis.LiteGraph = { vueNodesMode: false };
let scale = 1;
globalThis.comfyAPI = { app: { app: { canvas: { ds: { get scale() { return scale; }, set scale(v) { scale = v; } } } } } };
const listeners = new Map();
globalThis.addEventListener = (type, fn, opts) => {
  if (!listeners.has(type)) listeners.set(type, []);
  listeners.get(type).push({ fn, opts });
};
globalThis.removeEventListener = (type, fn, opts) => {
  const arr = listeners.get(type) || [];
  const i = arr.findIndex((e) => e.fn === fn);
  if (i >= 0) arr.splice(i, 1);
};
globalThis.dispatchWheel = () => {
  for (const { fn } of listeners.get("wheel") || []) fn({});
};
globalThis.requestAnimationFrame = (fn) => { fn(); return 1; };
globalThis.cancelAnimationFrame = () => {};
globalThis.document = { getElementById: () => null, createElement: () => ({ style: {}, classList: { add() {} } }), head: { appendChild() {} } };
"""


def _run_zoom_probe(tmp_path, name, body):
    uri = (CANONICAL_DIR / "nodes2.js").resolve().as_uri()
    probe = tmp_path / f"{name}.mjs"
    probe.write_text(_ZOOM_PRELUDE + f'const M = await import("{uri}");\n' + body, encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[:800]
    return json.loads(p.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_on_zoom_change_shared_listeners(tmp_path):
    data = _run_zoom_probe(tmp_path, "zoomshare", """
const listenerCount = () => ["wheel","pointerup","keyup","resize"]
  .reduce((n, t) => n + (listeners.get(t) || []).length, 0);
const a = M.onZoomChange(() => {});
const b = M.onZoomChange(() => {});
const shared = listenerCount() === 4;
a(); b();
const torn = listenerCount() === 0;
process.stdout.write(JSON.stringify({ shared, torn }));
""")
    assert data["shared"]
    assert data["torn"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_on_zoom_change_fires_once_on_scale_change(tmp_path):
    data = _run_zoom_probe(tmp_path, "zoomfire", """
let calls = 0;
const off = M.onZoomChange(() => { calls += 1; });
dispatchWheel();
const unchanged = calls;
scale = 1.2;
dispatchWheel();
const changed = calls;
off();
process.stdout.write(JSON.stringify({ unchanged, changed }));
""")
    assert data["unchanged"] == 0
    assert data["changed"] == 1
