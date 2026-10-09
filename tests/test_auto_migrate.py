"""L7.65: automatic migration of removed / merged C2C nodes on workflow load (js/_c2c_migrate_core.js, run under Node).

Semantics are core 1.52.7's replaceWithMapping (links move with their slot, values by old_widget_ids position,
set_value writes a constant, outputs by index), applied to the saved JSON before configure; nothing dropped silently.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
CORE = PACK / "js" / "_c2c_migrate_core.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")

# The successor as LiteGraph would build it: inputs (sockets + widget sockets), outputs, widget names and defaults.
SHAPES = {
    "NewFx": {
        "inputs": [{"name": "image", "type": "IMAGE"}, {"name": "mask", "type": "MASK"},
                   {"name": "effect", "type": "COMBO", "widget": {"name": "effect"}},
                   {"name": "size", "type": "INT", "widget": {"name": "size"}}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE"}, {"name": "MASK", "type": "MASK"}],
        "widgets": ["effect", "size"], "values": ["glow", 4],
    },
}
TABLE = [{
    "old_node_id": "OldShadow", "new_node_id": "NewFx", "old_widget_ids": ["radius"],
    "input_mapping": [{"new_id": "image", "old_id": "image"}, {"new_id": "size", "old_id": "radius"},
                      {"new_id": "effect", "set_value": "drop shadow"}],
    "output_mapping": [{"old_idx": 0, "new_idx": 0}],
}]


def _run(workflow, registered=("NewFx", "Src", "Dst")):
    script = f"""
import {{ migrateWorkflow }} from {json.dumps(CORE.as_uri())};
const shapes = {json.dumps(SHAPES)};
const reg = new Set({json.dumps(list(registered))});
const wf = {json.dumps(workflow)};
const r = migrateWorkflow(wf, {json.dumps(TABLE)}, (t) => reg.has(t), (t) => shapes[t] ?? null);
console.log(JSON.stringify({{ wf, r }}));
"""
    p = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def _wf(links, extra_input_link=None):
    old = {"id": 2, "type": "OldShadow", "pos": [0, 0], "size": [200, 100],
           "inputs": [{"name": "image", "type": "IMAGE", "link": 1},
                      {"name": "guide", "type": "IMAGE", "link": extra_input_link}],
           "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]},
                       {"name": "debug", "type": "IMAGE", "links": [3] if extra_input_link else []}],
           "properties": {"Node name for S&R": "OldShadow"}, "widgets_values": [9]}
    nodes = [{"id": 1, "type": "Src", "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1] + ([4] if extra_input_link else [])}],
              "inputs": []}, old,
             {"id": 3, "type": "Dst", "inputs": [{"name": "images", "type": "IMAGE", "link": 2}], "outputs": []}]
    if extra_input_link:
        nodes.append({"id": 4, "type": "Dst", "inputs": [{"name": "images", "type": "IMAGE", "link": 3}], "outputs": []})
    return {"nodes": nodes, "links": links}


def test_type_values_links_and_set_value_move():
    out = _run(_wf([[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]]))
    node = next(n for n in out["wf"]["nodes"] if n["id"] == 2)
    assert node["type"] == "NewFx" and node["properties"]["Node name for S&R"] == "NewFx"
    assert node["widgets_values"] == ["drop shadow", 9]                     # set_value + old radius -> size
    assert node["inputs"][0]["link"] == 1 and node["outputs"][0]["links"] == [2]
    assert out["wf"]["links"] == [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]]
    assert out["r"]["migrated"] == ["OldShadow → NewFx (#2)"] and out["r"]["problems"] == []


def test_unmapped_links_are_removed_and_reported():
    links = [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"], [3, 2, 1, 4, 0, "IMAGE"], [4, 1, 0, 2, 1, "IMAGE"]]
    out = _run(_wf(links, extra_input_link=4))
    ids = [l[0] for l in out["wf"]["links"]]
    assert ids == [1, 2]                                                      # guide input + debug output dropped
    probs = " ".join(out["r"]["problems"])
    assert 'input "guide"' in probs and 'output "debug"' in probs
    src = next(n for n in out["wf"]["nodes"] if n["id"] == 1)
    dst = next(n for n in out["wf"]["nodes"] if n["id"] == 4)
    assert src["outputs"][0]["links"] == [1] and dst["inputs"][0]["link"] is None


def test_object_links_and_retargeted_slot():
    TABLE_SLOT = SHAPES["NewFx"]["inputs"]
    wf = _wf([{"id": 1, "origin_id": 1, "origin_slot": 0, "target_id": 2, "target_slot": 0, "type": "IMAGE"},
              {"id": 2, "origin_id": 2, "origin_slot": 0, "target_id": 3, "target_slot": 0, "type": "IMAGE"}])
    out = _run(wf)
    assert out["wf"]["links"][0]["target_slot"] == 0 and TABLE_SLOT[0]["name"] == "image"


def test_existing_types_and_missing_successors_are_left_alone():
    wf = _wf([[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]])
    out = _run(wf, registered=("NewFx", "Src", "Dst", "OldShadow"))        # the old type still exists: untouched
    assert next(n for n in out["wf"]["nodes"] if n["id"] == 2)["type"] == "OldShadow" and not out["r"]["migrated"]
    out = _run(wf, registered=("Src", "Dst"))                               # successor missing: reported, untouched
    assert next(n for n in out["wf"]["nodes"] if n["id"] == 2)["type"] == "OldShadow"
    assert "NewFx is not available" in out["r"]["problems"][0]


def test_subgraph_definitions_are_migrated():
    inner = _wf([{"id": 1, "origin_id": 1, "origin_slot": 0, "target_id": 2, "target_slot": 0, "type": "IMAGE"}])
    root = {"nodes": [], "links": [], "definitions": {"subgraphs": [inner]}}
    out = _run(root)
    sub = out["wf"]["definitions"]["subgraphs"][0]
    assert next(n for n in sub["nodes"] if n["id"] == 2)["type"] == "NewFx" and len(out["r"]["migrated"]) == 1
