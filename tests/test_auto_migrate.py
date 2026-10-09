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


def _run(workflow, registered=("NewFx", "Src", "Dst"), table=None, shapes=None):
    script = f"""
import {{ migrateWorkflow }} from {json.dumps(CORE.as_uri())};
const shapes = {json.dumps(shapes or SHAPES)};
const reg = new Set({json.dumps(list(registered))});
const wf = {json.dumps(workflow)};
const r = migrateWorkflow(wf, {json.dumps(table or TABLE)}, (t) => reg.has(t), (t) => shapes[t] ?? null);
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


# ── L7.65 wave 1: value translation, seed control, removals, required sockets, notes ────────────────────────────

SAMPLER = {
    "inputs": [{"name": "model", "type": "MODEL"}, {"name": "latent_image", "type": "LATENT"},
               {"name": "seed", "type": "INT", "widget": {"name": "seed"}},
               {"name": "space", "type": "COMBO", "widget": {"name": "space"}},
               {"name": "stops", "type": "FLOAT", "widget": {"name": "stops"}}],
    "outputs": [{"name": "LATENT", "type": "LATENT"}],
    "widgets": ["seed", "control_after_generate", "space", "stops"], "values": [0, "randomize", "srgb", 0.0],
    "required": ["model", "latent_image", "seed", "space"],
}
WAVE_TABLE = [
    {"old_node_id": "OldSampler", "new_node_id": "NewSampler",
     "old_widget_ids": ["source", "exposure", "seed", "control_after_generate"],
     "input_mapping": [{"new_id": "model", "old_id": "model"}, {"new_id": "seed", "old_id": "seed"},
                       {"new_id": "space", "old_id": "source"}],
     "output_mapping": [{"old_idx": 0, "new_idx": 0}],
     "c2c_values": [{"new_id": "space", "old_id": "source", "map": {"sRGB": "srgb", "Log C3": "logc3"}},
                    {"new_id": "stops", "old_id": "exposure", "fn": "log2", "clamp": [-5, 5]}],
     "c2c_note": "one constant CFG now."},
    {"old_node_id": "OldNoControl", "new_node_id": "NewSampler", "old_widget_ids": ["seed"],
     "input_mapping": [{"new_id": "seed", "old_id": "seed"}], "output_mapping": []},
    {"old_node_id": "OldStatus", "c2c_remove": True, "c2c_note": "see the panel."},
]
WAVE_SHAPES = {"NewSampler": SAMPLER}


def _wave(nodes, links):
    return _run({"nodes": nodes, "links": links}, registered=("NewSampler", "Src", "Dst"), table=WAVE_TABLE,
                shapes=WAVE_SHAPES)


def _old_sampler(vals, named=None, model_link=1):
    n = {"id": 2, "type": "OldSampler", "inputs": [{"name": "model", "type": "MODEL", "link": model_link}],
         "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}], "widgets_values": vals}
    if named is not None:
        n["widgets_values_named"] = named
    return n


def _src():
    return {"id": 1, "type": "Src", "inputs": [], "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1]}]}


def test_value_maps_log2_clamp_and_seed_control_travel():
    out = _wave([_src(), _old_sampler(["Log C3", 4.0, 77, "fixed"])], [[1, 1, 0, 2, 0, "MODEL"]])
    node = next(n for n in out["wf"]["nodes"] if n["id"] == 2)
    assert node["type"] == "NewSampler"
    assert node["widgets_values"] == [77, "fixed", "logc3", 2.0]            # seed, its control, mapped, log2(4)
    out = _wave([_src(), _old_sampler(["sRGB", 1000.0, 5, "increment"])], [[1, 1, 0, 2, 0, "MODEL"]])
    node = next(n for n in out["wf"]["nodes"] if n["id"] == 2)
    assert node["widgets_values"] == [5, "increment", "srgb", 5]            # log2(1000) clamped to 5


def test_a_seed_without_control_stays_fixed():
    old = {"id": 2, "type": "OldNoControl", "inputs": [], "outputs": [], "widgets_values": [42]}
    out = _wave([old], [])
    assert next(n for n in out["wf"]["nodes"] if n["id"] == 2)["widgets_values"][:2] == [42, "fixed"]


def test_an_untranslatable_value_is_reported_and_left_at_default():
    out = _wave([_src(), _old_sampler(["Rec.2020", 1.0, 1, "fixed"])], [[1, 1, 0, 2, 0, "MODEL"]])
    node = next(n for n in out["wf"]["nodes"] if n["id"] == 2)
    assert node["widgets_values"][2] == "srgb"
    assert any('source = "Rec.2020" has no equivalent' in p for p in out["r"]["problems"])


def test_required_socket_left_empty_is_reported_and_note_given_once():
    a = _old_sampler(["sRGB", 1.0, 1, "fixed"])
    b = dict(_old_sampler(["sRGB", 1.0, 2, "fixed"], model_link=None), id=3)
    out = _wave([_src(), a, b], [[1, 1, 0, 2, 0, "MODEL"]])
    probs = out["r"]["problems"]
    assert "OldSampler → NewSampler (#2): required input not connected: latent_image" in probs
    assert "OldSampler → NewSampler (#3): required inputs not connected: model, latent_image" in probs
    assert out["r"]["notes"] == ["OldSampler → NewSampler: one constant CFG now."]


def test_named_values_follow_the_new_widgets():
    out = _wave([_src(), _old_sampler(["sRGB", 2.0, 9, "fixed"], named={"source": "sRGB", "seed": 9})],
                [[1, 1, 0, 2, 0, "MODEL"]])
    node = next(n for n in out["wf"]["nodes"] if n["id"] == 2)
    assert node["widgets_values_named"] == {"seed": 9, "control_after_generate": "fixed", "space": "srgb", "stops": 1.0}


def test_a_removed_node_leaves_the_graph_with_its_links_and_is_reported():
    status = {"id": 5, "type": "OldStatus", "inputs": [{"name": "x", "type": "MODEL", "link": 1}],
              "outputs": [{"name": "status", "type": "STRING", "links": [2]}], "widgets_values": [False]}
    dst = {"id": 6, "type": "Dst", "inputs": [{"name": "text", "type": "STRING", "link": 2}], "outputs": []}
    out = _wave([_src(), status, dst], [[1, 1, 0, 5, 0, "MODEL"], [2, 5, 0, 6, 0, "STRING"]])
    assert [n["id"] for n in out["wf"]["nodes"]] == [1, 6]
    assert out["wf"]["links"] == []
    assert next(n for n in out["wf"]["nodes"] if n["id"] == 1)["outputs"][0]["links"] == []
    assert next(n for n in out["wf"]["nodes"] if n["id"] == 6)["inputs"][0]["link"] is None
    assert out["r"]["removed"] == ["OldStatus (#5) - see the panel."]
    assert any('output "status" fed other nodes' in p for p in out["r"]["problems"])
