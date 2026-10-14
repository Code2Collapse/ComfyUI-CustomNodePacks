"""js/_vault_flatten.js - the vault payload for selections on the root graph, INSIDE subgraphs and CONTAINING
subgraph nodes (owner 2026-10-14, ledger L2.45 / L2.46). Runs the shipped module in node against mock LiteGraph
graphs shaped like frontend 1.53 (string ids, links in a Map, subgraph IO nodes "-10" / "-20")."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

MOD = Path(__file__).resolve().parents[1] / "js" / "_vault_flatten.js"

HARNESS = r"""
import { flattenSelection } from "%s";

let lid = 1;
function graph(nodes, links, io) {
  const g = { _nodes: nodes, links: new Map(), getNodeById: (id) => nodes.find((n) => String(n.id) === String(id)) };
  if (io) { g.inputNode = { slots: io.in }; g.outputNode = { slots: io.out }; }
  for (const [o, os, t, ts, type] of links) {
    const l = { id: lid++, origin_id: String(o), origin_slot: os, target_id: String(t), target_slot: ts, type };
    g.links.set(l.id, l);
    const on = nodes.find((n) => String(n.id) === String(o)); if (on) on.outputs[os].links.push(l.id);
    const tn = nodes.find((n) => String(n.id) === String(t)); if (tn) tn.inputs[ts].link = l.id;
  }
  return g;
}
function node(id, type, ins = [], outs = [], widgets = []) {
  return { id: String(id), type, comfyClass: type,
           inputs: ins.map(([name, t, w]) => ({ name, type: t, link: null, widget: w ? { name } : undefined })),
           outputs: outs.map(([name, t]) => ({ name, type: t, links: [] })),
           widgets: widgets.map(([name, value]) => ({ name, value })) };
}
function sgnode(id, sg, ins, outs, widgets = []) {
  const n = node(id, "uuid-" + id, ins, outs, widgets); n.subgraph = sg; n.isSubgraphNode = () => true; return n;
}
function reroute(id) { const n = node(id, "Reroute", [["", "*"]], [["", "*"]]); n.isVirtualNode = true; return n; }
const pick = (g, ids) => ids.map((i) => g.getNodeById(i));
const out = {};

// 1 root: A(ext) -> B -> C -> D(ext)
{ const n = [node(1, "Load", [], [["IMAGE", "IMAGE"]]), node(2, "Inv", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]], [["k", 3]]),
             node(3, "Blur", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]]), node(4, "Save", [["images", "IMAGE"]], [])];
  const g = graph(n, [[1, 0, 2, 0, "IMAGE"], [2, 0, 3, 0, "IMAGE"], [3, 0, 4, 0, "IMAGE"]]);
  out.root = flattenSelection(pick(g, [2, 3]), g); }

// 2 inside a subgraph: IO_IN -> B -> C -> IO_OUT
{ const n = [node(2, "Inv", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]]), node(3, "Blur", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]])];
  const g = graph(n, [["-10", 0, 2, 0, "IMAGE"], [2, 0, 3, 0, "IMAGE"], [3, 0, "-20", 0, "IMAGE"]],
                  { in: [{ name: "pic", type: "IMAGE" }], out: [{ name: "result", type: "IMAGE" }] });
  out.inside = flattenSelection(pick(g, [2, 3]), g); }

// 3 selection CONTAINS a subgraph node S (inner X -> Y), outer A -> S -> D; S has a promoted widget input "amount"
{ const inner = graph([node(10, "X", [["image", "IMAGE"], ["amount", "INT", true]], [["IMAGE", "IMAGE"]], [["amount", 1]]),
                       node(11, "Y", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]])],
                      [["-10", 0, 10, 0, "IMAGE"], ["-10", 1, 10, 1, "INT"], [10, 0, 11, 0, "IMAGE"], [11, 0, "-20", 0, "IMAGE"]],
                      { in: [{ name: "image", type: "IMAGE" }, { name: "amount", type: "INT" }], out: [{ name: "IMAGE", type: "IMAGE" }] });
  const S = sgnode(5, inner, [["image", "IMAGE"], ["amount", "INT", true]], [["IMAGE", "IMAGE"]], [["amount", 7]]);
  const n = [node(1, "Load", [], [["IMAGE", "IMAGE"]]), S, node(4, "Save", [["images", "IMAGE"]], [])];
  const g = graph(n, [[1, 0, 5, 0, "IMAGE"], [5, 0, 4, 0, "IMAGE"]]);
  out.contains = flattenSelection(pick(g, [5]), g); }

// 4 nested: S(outer) contains T contains Z
{ const tIn = graph([node(20, "Z", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]])],
                    [["-10", 0, 20, 0, "IMAGE"], [20, 0, "-20", 0, "IMAGE"]],
                    { in: [{ name: "image", type: "IMAGE" }], out: [{ name: "IMAGE", type: "IMAGE" }] });
  const T = sgnode(30, tIn, [["image", "IMAGE"]], [["IMAGE", "IMAGE"]]);
  const sIn = graph([T], [["-10", 0, 30, 0, "IMAGE"], [30, 0, "-20", 0, "IMAGE"]],
                    { in: [{ name: "image", type: "IMAGE" }], out: [{ name: "IMAGE", type: "IMAGE" }] });
  const S = sgnode(5, sIn, [["image", "IMAGE"]], [["IMAGE", "IMAGE"]]);
  const n = [node(1, "Load", [], [["IMAGE", "IMAGE"]]), S, node(4, "Save", [["images", "IMAGE"]], [])];
  const g = graph(n, [[1, 0, 5, 0, "IMAGE"], [5, 0, 4, 0, "IMAGE"]]);
  out.nested = flattenSelection(pick(g, [5]), g); }

// 5 a Reroute between two selected nodes
{ const n = [node(2, "Inv", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]]), reroute(9), node(3, "Blur", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]]),
             node(1, "Load", [], [["IMAGE", "IMAGE"]]), node(4, "Save", [["images", "IMAGE"]], [])];
  const g = graph(n, [[1, 0, 2, 0, "IMAGE"], [2, 0, 9, 0, "IMAGE"], [9, 0, 3, 0, "IMAGE"], [3, 0, 4, 0, "IMAGE"]]);
  out.reroute = flattenSelection(pick(g, [2, 9, 3]), g); }

// 6 two external inputs both named "image", one output feeding two outside nodes
{ const n = [node(1, "LoadA", [], [["IMAGE", "IMAGE"]]), node(2, "LoadB", [], [["IMAGE", "IMAGE"]]),
             node(3, "Blend", [["image", "IMAGE"], ["image2", "IMAGE"]], [["IMAGE", "IMAGE"]]),
             node(4, "Inv", [["image", "IMAGE"]], [["IMAGE", "IMAGE"]]),
             node(6, "SaveA", [["images", "IMAGE"]], []), node(7, "SaveB", [["images", "IMAGE"]], [])];
  n[2].inputs[1].name = "image";                       // the same input name twice, as two nodes would give
  const g = graph(n, [[1, 0, 3, 0, "IMAGE"], [2, 0, 3, 1, "IMAGE"], [3, 0, 4, 0, "IMAGE"], [4, 0, 6, 0, "IMAGE"], [4, 0, 7, 0, "IMAGE"]]);
  out.names = flattenSelection(pick(g, [3, 4]), g); }

// 7 a Reroute alone carries an outside value to an outside node
{ const n = [node(1, "Load", [], [["IMAGE", "IMAGE"]]), reroute(9), node(4, "Save", [["images", "IMAGE"]], [])];
  const g = graph(n, [[1, 0, 9, 0, "IMAGE"], [9, 0, 4, 0, "IMAGE"]]);
  out.passthrough = flattenSelection(pick(g, [9]), g); }

process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def R():
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH")
    with tempfile.TemporaryDirectory() as d:
        js = Path(d) / "h.mjs"
        js.write_text(HARNESS % MOD.as_uri(), encoding="utf-8")
        p = subprocess.run([shutil.which("node"), str(js)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:2000]
    return json.loads(p.stdout)


def test_root_selection_has_one_entering_and_one_leaving_wire(R):
    # INVARIANT: the plain case keeps working: A -> [B -> C] -> D gives 1 boundary input, 1 output, 1 inner link.
    r = R["root"]
    assert [n["id"] for n in r["nodes"]] == ["2", "3"] and r["nodes"][0]["widgets"] == {"k": 3}
    assert r["links"] == [{"from": "2", "from_slot": 0, "to": "3", "to_slot": 0, "to_name": "image"}]
    assert r["boundary_in"] == [{"name": "image", "to": "2", "to_slot": 0, "to_name": "image"}]
    assert r["externalSources"] == [{"kind": "node", "id": "1", "slot": 0}]
    assert r["externalTargets"] == [[{"kind": "node", "id": "4", "slot": 0}]] and not r["problems"]


def test_selection_inside_a_subgraph_keeps_its_io_wires(R):
    # INVARIANT: the measured bug - links from the subgraph's input node ("-10") and to its output node ("-20") become
    # boundary wires the vault reconnects, typed from the IO slots.
    r = R["inside"]
    assert r["externalSources"] == [{"kind": "io", "slot": 0}] and r["inTypes"] == ["IMAGE"]
    assert r["externalTargets"] == [[{"kind": "io", "slot": 0}]]
    assert r["boundary_out"] == [{"name": "result", "from": "3", "from_slot": 0}]


def test_a_subgraph_node_in_the_selection_is_flattened(R):
    # INVARIANT: a subgraph node's inner nodes enter the payload ("5:10", "5:11"), its IO wires map to them, and a
    # value typed on the subgraph node (promoted widget) reaches the inner widget.
    r = R["contains"]
    assert sorted(n["id"] for n in r["nodes"]) == ["5:10", "5:11"] and r["flattenedSubgraphs"] == 1
    assert {"from": "5:10", "from_slot": 0, "to": "5:11", "to_slot": 0, "to_name": "image"} in r["links"]
    assert r["boundary_in"] == [{"name": "image", "to": "5:10", "to_slot": 0, "to_name": "image"}]
    assert r["boundary_out"][0]["from"] == "5:11"
    x = next(n for n in r["nodes"] if n["id"] == "5:10")
    assert x["widgets"]["amount"] == 7 and x["class_type"] == "X"


def test_nested_subgraphs_flatten_recursively(R):
    # INVARIANT: S containing T containing Z yields the one real node "5:30:20", wired in and out.
    r = R["nested"]
    assert [n["id"] for n in r["nodes"]] == ["5:30:20"] and r["flattenedSubgraphs"] == 2
    assert r["boundary_in"][0]["to"] == "5:30:20" and r["boundary_out"][0]["from"] == "5:30:20"


def test_reroute_is_followed_not_stored(R):
    # INVARIANT: Reroute is front-end only: the payload links B -> C directly and holds no Reroute node.
    r = R["reroute"]
    assert [n["type"] if "type" in n else n["class_type"] for n in r["nodes"]] == ["Inv", "Blur"]
    assert {"from": "2", "from_slot": 0, "to": "3", "to_slot": 0, "to_name": "image"} in r["links"]


def test_boundary_names_are_unique_and_fan_out_is_one_output(R):
    # INVARIANT: the executor keys boundary inputs by name, so two "image" inputs get distinct names; one value used
    # by two outside nodes is ONE vault output with two targets.
    r = R["names"]
    assert [b["name"] for b in r["boundary_in"]] == ["image", "image_2"]
    assert len(r["boundary_out"]) == 1 and len(r["externalTargets"][0]) == 2


def test_a_pass_through_reroute_is_refused_with_a_reason(R):
    # INVARIANT: a Reroute that only carries an outside value would drop that wire when removed - it is reported.
    assert any("Reroute" in p for p in R["passthrough"]["problems"])
