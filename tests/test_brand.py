"""The house look on the canvas, and the copy of it in every pack.

Each Code2Collapse pack ships _c2c_brand.js and colours ONLY its own nodes -
the packs cannot import each other (a cross-pack import 404s on a standalone
install and takes the pack's whole front-end down). So the six copies must
stay identical, and the one decision that matters - "is this node mine?" - is
exercised here through the real file, not a Python paraphrase of it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
CANONICAL = WORKSPACE / "ComfyUI-CustomNodePacks" / "js" / "_c2c_brand.js"
COPIES = {
    "ComfyUI-CustomNodePacks": "js",
    "ComfyUI-NukeMaxNodes": "web",
    "ComfyUI-WanNodeExperiments": "web",
    "ComfyUI-MiniMaxSuite": "web",
    "ComfyUI-WanAnimatePreprocessV2": "js",
    "ComfyUI-GLM_Image": "web",
    "ComfyUI-WanAnimalPreprocessor": "web",
}


def _copy(pack):
    return WORKSPACE / pack / COPIES[pack] / "_c2c_brand.js"


@pytest.mark.parametrize("pack", sorted(COPIES))
def test_every_pack_ships_the_same_brand(pack):
    path = _copy(pack)
    if not (WORKSPACE / pack).exists():
        pytest.skip(f"{pack} is not in this workspace")
    assert path.exists(), f"{pack} has no _c2c_brand.js - its nodes lose the house look"
    assert path.read_text(encoding="utf-8") == CANONICAL.read_text(encoding="utf-8"), (
        f"{pack}'s brand has drifted from CustomNodePacks'")


@pytest.mark.parametrize("pack", sorted(COPIES))
def test_the_copy_sits_at_the_web_root_its_import_expects(pack):
    """Every copy imports ../../scripts/app.js, which is right only at the root
    of the pack's WEB_DIRECTORY. One level deeper is a 404."""
    if not (WORKSPACE / pack).exists():
        pytest.skip(f"{pack} is not in this workspace")
    init = (WORKSPACE / pack / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'WEB_DIRECTORY\s*=\s*["\']\.?/?([\w/]+)["\']', init)
    assert m, f"{pack} declares no WEB_DIRECTORY, so the brand never loads"
    assert m.group(1).strip("/") == COPIES[pack]


def test_node_colours_are_literal_hex():
    """Node colours are painted on the canvas, which cannot parse var() - it
    paints black and throws nothing."""
    src = CANONICAL.read_text(encoding="utf-8")
    block = src[src.index("export const BRAND"):src.index("});", src.index("export const BRAND"))]
    for key in ("title", "body", "ink"):
        m = re.search(rf'{key}:\s*"(#[0-9a-fA-F]{{6}})"', block)
        assert m, f"BRAND.{key} is not a literal hex colour"
    assert "var(" not in block


def test_the_users_colour_always_wins():
    """A colour from a saved workflow (restored before onNodeCreated) or from
    the Colors menu must never be overwritten."""
    src = CANONICAL.read_text(encoding="utf-8")
    assert "if (!this.color) this.color = BRAND.title" in src
    assert "if (!this.bgcolor) this.bgcolor = BRAND.body" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_a_pack_recognises_its_own_nodes_and_nobody_elses(tmp_path):
    src = CANONICAL.read_text(encoding="utf-8")
    start = src.index("/** \"/extensions/")
    end = src.index("function enabled()")
    probe = tmp_path / "probe.mjs"
    probe.write_text(src[start:end] + r'''
const cases = [
  // [served url, python_module, expected]
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", "custom_nodes.ComfyUI-NukeMaxNodes", true],
  // the Linux install uses the GitHub folder names
  ["http://h/extensions/ComfyUI-NukeNodePack/_c2c_brand.js", "custom_nodes.ComfyUI-NukeNodePack", true],
  // served under a pyproject project name, module still the folder
  ["http://h/extensions/comfyui-nukemax-nodes/_c2c_brand.js", "custom_nodes.ComfyUI-NukeMaxNodes", true],
  // another pack's node
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", "custom_nodes.ComfyUI-MiniMaxSuite", false],
  // a core node
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", "nodes", false],
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", "comfy_extras.nodes_video", false],
  // no python_module: left alone rather than guessed at
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", undefined, false],
  // a folder name with a space, url-encoded by the browser
  ["http://h/extensions/My%20Pack/_c2c_brand.js", "custom_nodes.My Pack", true],
];
const out = cases.map(([u, m, want]) => {
  const got = belongsTo({ python_module: m }, servedFolder(u));
  return { u, m, want, got };
});
process.stdout.write(JSON.stringify(out));
''', encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:600]
    wrong = [c for c in json.loads(p.stdout) if c["got"] != c["want"]]
    assert not wrong, wrong


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_a_node_smaller_than_its_content_is_grown_and_never_shrunk(tmp_path):
    """DOM widgets arrive in onNodeCreated, after sizing: 275 fresh nodes came
    out 8-19px short with their bottom row clipped, small ones narrower than
    their own title. Grow to computeSize, never shrink."""
    src = CANONICAL.read_text(encoding="utf-8")
    start = src.index("/** Grow `node`")
    end = src.index("function enabled()")
    probe = tmp_path / "fit.mjs"
    probe.write_text(src[start:end] + r'''
const strip = { element: { tagName: "DIV" } };
const mk = (w, h, mw, mh, widgets = [strip]) => ({ size: [w, h], computeSize: () => [mw, mh],
  widgets, setSize(s) { this.size = s; }, setDirtyCanvas() {} });
// no DOM widget / only a textarea / a virtual reroute: small on purpose, untouched
const plain = mk(154, 45, 210, 53, []);
const texty = mk(154, 45, 210, 53, [{ element: { tagName: "TEXTAREA" } }]);
const reroute = Object.assign(mk(40, 30, 270, 26), { isVirtualNode: true });
const shortNode = mk(270, 233, 210, 241);
const tallNode = mk(400, 600, 210, 241);       // the user made it bigger
const narrow = mk(154, 45, 210, 53);           // "Invert (Nuke..." - both sides short
const broken = { size: [1, 1], widgets: [strip], computeSize() { throw new Error("x"); } };
const out = [fitToContent(shortNode), shortNode.size, fitToContent(tallNode), tallNode.size,
             fitToContent(narrow), narrow.size, fitToContent(broken), fitToContent(null),
             fitToContent(plain), plain.size, fitToContent(texty), fitToContent(reroute), reroute.size];
process.stdout.write(JSON.stringify(out));
''', encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:600]
    assert json.loads(p.stdout) == [True, [270, 241], False, [400, 600],
                                    True, [210, 53], False, False,
                                    False, [154, 45], False, False, [40, 30]]
    # deferred one frame: after other onNodeCreated hooks and after configure
    assert re.search(r"requestAnimationFrame\(\(\) => \{[^}]*fitToContent\(node\)", src)


def test_in_nodes2_only_the_width_is_grown(tmp_path):
    """Nodes 2.0 sizes a node to its content and treats size[1] as a minimum;
    computeSize also counts the advanced inputs it hides. Growing the height
    there left ~250-500 px of dead space (ReLight2D 1226 -> 1470, measured)."""
    src = CANONICAL.read_text(encoding="utf-8")
    start = src.index("/** Grow `node`")
    end = src.index("function enabled()")
    probe = tmp_path / "fit2.mjs"
    probe.write_text("globalThis.LiteGraph = { vueNodesMode: true };\n" + src[start:end] + r'''
const n = { size: [154, 45], computeSize: () => [210, 900], widgets: [{ element: { tagName: "DIV" } }],
  setSize(s) { this.size = s; }, setDirtyCanvas() {} };
process.stdout.write(JSON.stringify([fitToContent(n), n.size]));
''', encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:600]
    assert json.loads(p.stdout) == [True, [210, 45]]


def test_the_setting_is_registered_once_and_read_by_every_copy():
    src = CANONICAL.read_text(encoding="utf-8")
    assert "if (!window.__C2C_BRAND_REG__)" in src
    assert 'getSettingValue?.("c2c.brand.nodeColors", true)' in src


def _chrome_probe(tmp_path, body):
    """Run the chrome helpers (everything between the palette and enabled())
    in node, with a fake LiteGraph."""
    src = CANONICAL.read_text(encoding="utf-8")
    start = src.index("export const BRAND")
    end = src.index("function enabled()")
    probe = tmp_path / "chrome.mjs"
    probe.write_text("globalThis.window = globalThis;\n" + src[start:end] + "\n" + body, encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:800]
    return json.loads(p.stdout)


def _lum(hx):
    hx = hx.lstrip("#")

    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(int(hx[i:i + 2], 16)) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _brand(key):
    src = CANONICAL.read_text(encoding="utf-8")
    block = src[src.index("export const BRAND"):src.index("});", src.index("export const BRAND"))]
    return re.search(rf'\b{key}:\s*"(#[0-9a-fA-F]{{6}})"', block).group(1).lower()


def test_every_node_shade_is_a_step_of_the_night_ramp():
    """The user's night palette is a ramp from #0f1030 to #3b3a68 (authored as
    the night CORE in _c2c_theme.js). The node's dark shades come from it, not
    from a colour picked by eye - that is how the look drifted to "full purple"."""
    theme = (CANONICAL.parent / "_c2c_theme.js").read_text(encoding="utf-8")
    night = theme[theme.index("night: {"):theme.index("},", theme.index("night: {"))]
    ramp = {"#0f1030"} | {
        m.group(2).lower() for m in re.finditer(r'\b(bg|bg2|bg3|surface0|surface1|surface2):\s*"(#[0-9a-fA-F]{6})"', night)}
    assert len(ramp) == 7, ramp
    for key in ("title", "body", "titleTop", "titleBottom", "badge", "widgetBg", "widgetOutline"):
        assert _brand(key) in ramp, f"BRAND.{key} {_brand(key)} is not on the night ramp"


def test_the_body_is_dark_and_the_rim_carries_the_edge():
    """Dark body (no brighter than ComfyUI's canvas, #141414), so the node is
    told apart by its edge in the pack's colour: every pack colour must stand
    well clear of both the body and the canvas, and the rim never fades out."""
    body = _brand("body")
    assert _lum(body) <= _lum("#141414"), f"body {body} is lighter than the canvas: no darkness"
    src = CANONICAL.read_text(encoding="utf-8")
    stripes = set(re.findall(r'"(#[0-9a-f]{6})"\]', src[src.index("const PACK_STRIPE"):src.index("];", src.index("const PACK_STRIPE"))]))
    stripes.add("#b494ff")
    for s in stripes:
        assert _contrast(s, body) >= 7, f"pack colour {s} is only {_contrast(s, body):.1f}:1 on the body"
        assert _contrast(s, "#141414") >= 6, f"pack colour {s} is only {_contrast(s, '#141414'):.1f}:1 on the canvas"
    rim = src[src.index("function drawOutline"):src.index("const WIDGET_KEYS")]
    alphas = [float(a) for a in re.findall(r"rgba\(stripe,\s*([0-9.]+)\)", rim)]
    assert alphas and min(alphas) >= 0.25, alphas


PACKS_PROBE = """
const legacy = { color: "#26275a", bgcolor: "#17182f" };
upgradeLegacy(legacy);
const indigo = { color: "#33357c", bgcolor: "#282a56" };
upgradeLegacy(indigo);
const mine = { color: "#aa3355", bgcolor: "#101010" };
upgradeLegacy(mine);
process.stdout.write(JSON.stringify({
  stripes: ["ComfyUI-NukeMaxNodes", "ComfyUI-NukeNodePack", "ComfyUI-MiniMaxSuite",
            "ComfyUI-WanNodeExperiments", "ComfyUI-WanAnimatePreprocessV2",
            "ComfyUI-WanAnimalPreprocessor", "ComfyUI-GLM_Image", "ComfyUI-CustomNodePacks"].map(stripeFor),
  legacy, indigo, mine, brand: { title: BRAND.title, body: BRAND.body },
  wears: [wearsBrand({}), wearsBrand({ color: BRAND.title }), wearsBrand({ color: "#26275a" }),
          wearsBrand({ color: "#aa3355" })],
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_pack_stripes_legacy_upgrade_and_user_colours(tmp_path):
    data = _chrome_probe(tmp_path, PACKS_PROBE)
    assert data["stripes"] == ["#f1aa7b", "#f1aa7b", "#76dccb", "#7fd4f2", "#eba2de",
                               "#f3d288", "#8ee09d", "#b494ff"]
    current = {"color": data["brand"]["title"], "bgcolor": data["brand"]["body"]}
    assert data["legacy"] == current
    assert data["indigo"] == current, "a workflow saved with the old indigo must come up dark"
    assert data["mine"] == {"color": "#aa3355", "bgcolor": "#101010"}, "a user's own colour must win"
    assert data["wears"] == [True, True, True, False]


WIDGETS_PROBE = """
globalThis.LiteGraph = { WIDGET_BGCOLOR: "#222", WIDGET_OUTLINE_COLOR: "#666",
                         WIDGET_TEXT_COLOR: "#DDD", WIDGET_SECONDARY_TEXT_COLOR: "#999" };
const during = withNightWidgets(() => ({ ...LiteGraph }));
let threw = false;
try { withNightWidgets(() => { throw new Error("draw failed"); }); } catch (e) { threw = true; }
// a fake 2D context: roundRect lives on the prototype, like the real one
const radii = [];
const ctx = Object.create({ roundRect(x, y, w, h, r) { radii.push(r); } });
const tinted = withNightWidgets(() => {
  ctx.roundRect(15, 0, 200, 20, [10]);   // LiteGraph's widget pill: squared off
  ctx.roundRect(15, 0, 200, 20, [4]);    // somebody's own radius: untouched
  ctx.roundRect(0, 0, 200, 80, [40]);    // a big panel that happens to be h/2: untouched
  return LiteGraph.WIDGET_OUTLINE_COLOR;
}, "#76dccb", ctx);
let threw2 = false;
try { withNightWidgets(() => { throw new Error("x"); }, "#76dccb", ctx); } catch (e) { threw2 = true; }
process.stdout.write(JSON.stringify({ during, after: { ...LiteGraph }, threw, radii, tinted,
  ctxRestored: !Object.prototype.hasOwnProperty.call(ctx, "roundRect") && threw2,
  body: BRAND.body, widgetBg: BRAND.widgetBg }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_night_widgets_are_scoped_and_always_restored(tmp_path):
    data = _chrome_probe(tmp_path, WIDGETS_PROBE)
    assert data["during"]["WIDGET_BGCOLOR"] == data["widgetBg"]
    assert data["after"] == {"WIDGET_BGCOLOR": "#222", "WIDGET_OUTLINE_COLOR": "#666",
                             "WIDGET_TEXT_COLOR": "#DDD", "WIDGET_SECONDARY_TEXT_COLOR": "#999"}
    assert data["threw"], "a draw error must still propagate"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_fields_are_squared_and_tinted_only_for_the_draw(tmp_path):
    """Our fields are fields, not LiteGraph pills, with a rim in the pack's
    colour - and the context's roundRect is back to the real one afterwards,
    even when the draw throws, or every other node on the canvas would change."""
    data = _chrome_probe(tmp_path, WIDGETS_PROBE)
    assert data["radii"] == [[5], [4], [40]]
    assert data["tinted"] == "rgba(118,220,203,0.3)"
    assert data["ctxRestored"]
    assert _lum(data["widgetBg"]) < _lum(data["body"]), "fields sit below the body, not on top of it"


def _vue_probe(tmp_path, body):
    """Run the Nodes 2.0 half (palette .. FOLDER) in node with a tiny fake DOM."""
    src = CANONICAL.read_text(encoding="utf-8")
    start = src.index("export const BRAND")
    end = src.index("const FOLDER = servedFolder")
    prelude = """
globalThis.window = globalThis;
globalThis.requestAnimationFrame = (f) => f();
class El {
  constructor(id) { this.nodeType = 1; this.dataset = id == null ? {} : { nodeId: String(id) };
    this.children = []; }
  get firstElementChild() { return this.children[0] || null; }
  matches(sel) { return sel === ".lg-node[data-node-id]" && "nodeId" in this.dataset; }
  querySelectorAll(sel) { const out = [];
    const walk = (e) => { for (const c of e.children) { if (c.matches(sel)) out.push(c); walk(c); } };
    walk(this); return out; }
}
const body = new El(); const head = { kids: [], appendChild(n) { this.kids.push(n); } };
globalThis.document = { body, head, createElement: () => ({}),
  querySelectorAll: (sel) => body.querySelectorAll(sel) };
let observer = null;
globalThis.MutationObserver = class { constructor(cb) { this.cb = cb; observer = this; }
  observe() {} disconnect() {} };
const nodes = new Map();
const app = { graph: { getNodeById: (id) => nodes.get(Number(id)) ?? null } };
"""
    probe = tmp_path / "vue.mjs"
    probe.write_text(prelude + src[start:end] + "\n" + body, encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:800]
    return json.loads(p.stdout)


VUE_PROBE = """
nodes.set(1, { __c2cPack: "minimax" });
nodes.set(2, {});                               // somebody else's node
const a = new El(1), b = new El(2), wrap = new El(), c = new El(3);
wrap.children.push(c);
body.children.push(a, b);
globalThis.LiteGraph = undefined;
installVueChrome(); installVueChrome();         // second copy of the file: no-op
const afterInstall = [a.dataset.c2cPack ?? null, b.dataset.c2cPack ?? null];
nodes.set(3, { __c2cPack: "nukemax" });
globalThis.LiteGraph = { vueNodesMode: false };
observer.cb([{ addedNodes: [wrap] }]);
const classicIgnored = c.dataset.c2cPack ?? null;
globalThis.LiteGraph.vueNodesMode = true;
observer.cb([{ addedNodes: [wrap, { nodeType: 3 }] }]);
const added = c.dataset.c2cPack ?? null;
nodes.set(1, {});                               // another workflow reused id 1
retagVueNodes();
process.stdout.write(JSON.stringify({
  afterInstall, classicIgnored, added, reused: a.dataset.c2cPack ?? null,
  sheets: head.kids.length, css: head.kids[0]?.textContent ?? "",
  keys: ["ComfyUI-NukeMaxNodes", "ComfyUI-NukeNodePack", "ComfyUI-MiniMaxSuite", "ComfyUI-GLM_Image",
         "ComfyUI-WanAnimalPreprocessor", "ComfyUI-WanAnimatePreprocessV2", "ComfyUI-CustomNodePacks"].map(packKey),
}));
"""


def test_nodes2_elements_are_tagged_once_per_page_and_retagged_on_reuse(tmp_path):
    """Nodes 2.0 draws nodes as DOM, so the look is CSS keyed on data-c2c-pack.
    One sheet and one observer serve all seven copies; the tag follows the node
    that owns the id NOW (Vue reuses an element when a new workflow reuses the id)."""
    out = _vue_probe(tmp_path, VUE_PROBE)
    assert out["afterInstall"] == ["minimax", None]
    assert out["classicIgnored"] is None, "the observer must stay idle in the classic renderer"
    assert out["added"] == "nukemax"
    assert out["reused"] is None, "a stale tag would paint our chrome on somebody else's node"
    assert out["sheets"] == 1
    assert out["keys"] == ["nukemax", "nukemax", "minimax", "glm", "wananimalpreprocess",
                           "wananimatepreprocess", "core"]


REUSE_PROBE = """
globalThis.LiteGraph = { vueNodesMode: true };
nodes.set(1, { __c2cPack: "minimax" });
const a = new El(1), inner = new El();
inner.closest = (sel) => (sel === ".lg-node[data-node-id]" ? a : null);
a.children.push(inner);
body.children.push(a);
installVueChrome();
const first = a.dataset.c2cPack ?? null;
// graph cleared and a NukeMax node added in the same tick: it gets id 1 again
// and Vue keeps the element, re-rendering only what is inside it
nodes.set(1, { __c2cPack: "nukemax" });
observer.cb([{ target: inner, addedNodes: [] }]);
const reused = a.dataset.c2cPack ?? null;
// inside a subgraph the ids are the subgraph's own: look them up there
const sub = new Map([[1, {}]]);
app.canvas = { graph: { getNodeById: (id) => sub.get(Number(id)) ?? null } };
observer.cb([{ target: inner, addedNodes: [] }]);
const inSubgraph = a.dataset.c2cPack ?? null;
process.stdout.write(JSON.stringify({ first, reused, inSubgraph }));
"""


def test_nodes2_a_reused_element_is_retagged_from_the_graph_on_screen(tmp_path):
    """Found live: clearing the graph and adding a node in one tick hands the
    new node id 1 and Vue keeps node 1's element - no element is added, so the
    tag went stale and our node showed no chrome at all."""
    out = _vue_probe(tmp_path, REUSE_PROBE)
    assert out["first"] == "minimax"
    assert out["reused"] == "nukemax"
    assert out["inSubgraph"] is None, "a subgraph's node 1 is not the root's node 1"


def test_nodes2_css_keeps_the_users_colour_and_every_pack_has_a_stripe(tmp_path):
    out = _vue_probe(tmp_path, VUE_PROBE)
    css = out["css"]
    for key in ("core", "nukemax", "minimax", "wannodeexperiments", "wananimatepreprocess",
                "wananimalpreprocess", "glm"):
        assert f'.lg-node[data-c2c-pack="{key}"]' in css
    # the title sheen is translucent (never an opaque colour) so a colour the
    # user picked for the node still shows; the only opaque fill is the stripe
    sheen = re.search(r"linear-gradient\(180deg,[^;]*\)", css).group(0)
    assert "#" not in sheen, sheen
    assert "background-color" not in css, "the node's own colours come from Vue's inline style"
    assert "::after" in css and "pointer-events: none" in css
    # the same language as the classic canvas: aurora in the header, pack-tinted
    # fields, and core's azure accent (slider fills, toggles) in the pack colour
    assert "linear-gradient(90deg, color-mix(in srgb, var(--c2c-node-stripe)" in css
    assert "--primary-background: var(--c2c-node-stripe)" in css
    assert re.search(r"\.bg-component-node-widget-background \{\s*box-shadow: inset", css)
