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
