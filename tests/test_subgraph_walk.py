"""js/_subgraph_walk.js - the helpers every C2C extension uses to work INSIDE subgraphs (owner 2026-10-14, L2.47).
Runs the shipped module in node with its three imports stubbed, against a mock root graph holding a subgraph node
(frontend 1.53 shape: string ids, links in a Map)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
WALK = PACK / "js" / "_subgraph_walk.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")

_STUBS = {
    'import { app } from "../../scripts/app.js";': "const app = globalThis.__app;",
    'import { reportFailure as __c2cReport } from "./_c2c_report.js";': "const __c2cReport = () => {};",
    'import { getRuntime } from "./_c2c_runtime.js";': "const getRuntime = () => ({ requestRedraw() {} });",
}

_SETUP = r"""
const mk = (id, extra = {}) => ({ id: String(id), _nodes: undefined, ...extra });
const inner = { _nodes: [mk(5, { type: "Inner" }), mk(6, { type: "Deep-holder" })], links: new Map([[1, { id: 1, origin_id: "5" }]]) };
const deep = { _nodes: [mk(9, { type: "Deep" })], links: new Map() };
inner._nodes[1].subgraph = deep;
for (const n of inner._nodes) n.graph = inner;
deep._nodes[0].graph = deep;
const root = { isRootGraph: true, _nodes: [mk(1, { type: "A" }), mk(12, { type: "SG", subgraph: inner })], links: { 7: { id: 7 } } };
for (const n of root._nodes) n.graph = root;
const calls = [];
globalThis.__app = { graph: root, canvas: { graph: root, subgraph: undefined,
  setGraph(g) { calls.push(["setGraph", g === root ? "root" : g === inner ? "inner" : "deep"]); this.graph = g; },
  deselectAllNodes() {}, selectNode(n) { calls.push(["select", n.id]); }, centerOnNode(n) { calls.push(["center", n.id]); },
  setDirty() {} } };
"""


def _run(tmp_path: Path, body: str) -> dict:
    src = WALK.read_text(encoding="utf-8")
    for old, new in _STUBS.items():
        assert old in src, f"import changed: {old}"
        src = src.replace(old, new)
    mod = tmp_path / "walk.mjs"
    mod.write_text(src, encoding="utf-8")
    probe = tmp_path / "probe.mjs"
    probe.write_text(_SETUP + f"const W = await import({json.dumps(mod.resolve().as_uri())});\nconst out = {{}};\n"
                     + body + "\nconsole.log(JSON.stringify(out));\n", encoding="utf-8")
    r = subprocess.run([NODE, str(probe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_composite_ids_resolve_through_nested_subgraphs(tmp_path):
    # INVARIANT: the server's "12:6:9" names Deep inside Deep-holder inside SG; "12:5" names Inner; a bad path is null.
    out = _run(tmp_path, """
      out.inner = W.findNodeAnywhere("12:5")?.node?.type;
      out.deep = W.findNodeAnywhere("12:6:9")?.node?.type;
      out.missing = W.findNodeAnywhere("12:77");
      out.root = W.findNodeAnywhere(1)?.node?.type;
    """)
    assert out == {"inner": "Inner", "deep": "Deep", "missing": None, "root": "A"}


def test_current_graph_and_map_safe_links(tmp_path):
    # INVARIANT: currentGraph() is the graph on screen; linkOf / linksOf read both a Map and a plain object.
    out = _run(tmp_path, """
      out.rootIsCurrent = W.currentGraph() === __app.graph;
      __app.canvas.graph = inner;
      out.innerIsCurrent = W.currentGraph() === inner;
      out.mapLink = W.linkOf(inner, 1)?.origin_id;
      out.objLink = W.linkOf(root, 7)?.id;
      out.counts = [W.linksOf(inner).length, W.linksOf(root).length, W.linksOf(null).length];
    """)
    assert out == {"rootIsCurrent": True, "innerIsCurrent": True, "mapLink": "5", "objLink": 7, "counts": [1, 1, 0]}


def test_reveal_opens_the_holding_subgraph_then_centres(tmp_path):
    # INVARIANT: revealNode on a node inside a subgraph switches the canvas to that graph (canvas.subgraph set, as the
    # frontend's navigateToGraph does) BEFORE selecting / centring; for a root node from inside it goes back to root.
    out = _run(tmp_path, """
      W.revealNode("12:5");
      out.first = calls.splice(0);
      out.subgraphSet = __app.canvas.subgraph === inner;
      W.revealNode("1");
      out.second = calls.splice(0);
      out.subgraphCleared = __app.canvas.subgraph === undefined;
    """)
    assert out["first"] == [["setGraph", "inner"], ["select", "5"], ["center", "5"]] and out["subgraphSet"]
    assert out["second"] == [["setGraph", "root"], ["select", "1"], ["center", "1"]] and out["subgraphCleared"]


def test_for_all_nodes_visits_every_graph_once(tmp_path):
    # INVARIANT: the whole-workflow walk reaches nested subgraph nodes.
    out = _run(tmp_path, """
      const seen = []; W.forAllNodes((n) => { seen.push(n.type); }); out.seen = seen.sort();
    """)
    assert out["seen"] == sorted(["A", "SG", "Inner", "Deep-holder", "Deep"])
