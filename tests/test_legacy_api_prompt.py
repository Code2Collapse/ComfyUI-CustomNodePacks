"""API prompts that still name a C2C legacy node id (L7.65 P05 live check, 2026-10-15).

Core's NodeReplaceManager.apply_replacements (core 0.36 - 0.39) indexes node_struct["inputs"][old_id] directly: an
API prompt that omits an optional input of a legacy node (e.g. Background Remover's keep_model_loaded) raised KeyError
and POST /prompt answered HTTP 500 (measured on the owner-stack testbed). nodes/_legacy_replacements.py now migrates
C2C's own ids in an on_prompt handler, which core runs first, with the same semantics but skipping absent inputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from nodes import _legacy_replacements as L

PACK = Path(__file__).resolve().parents[1]
ROWS = {r["old_node_id"]: r for r in json.loads((PACK / "nodes" / "_legacy_replacements.json").read_text(encoding="utf-8"))}


def _bg_prompt():
    return {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 96, "height": 64, "batch_size": 1, "color": 0}},
        "2": {"class_type": "BackgroundRemoverMEC", "_meta": {"title": "BG"},
              "inputs": {"image": ["1", 0], "model_name": "rmbg_2.0", "threshold": 0.5, "invert": False, "mask_blur": 0}},
        "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 1]}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
    }


def test_optional_input_absent_is_skipped_not_a_crash():
    p = _bg_prompt()
    assert L.migrate_api_prompt(p) == 1
    n = p["2"]
    row = ROWS["BackgroundRemoverMEC"]
    assert n["class_type"] == row["new_node_id"] == "MaskOpsMEC"
    assert n["_meta"] == {"title": "BG"}
    assert "keep_model_loaded" not in n["inputs"]                    # absent in the prompt -> the new default applies
    assert n["inputs"]["image"] == ["1", 0]
    for m in row["input_mapping"]:
        if "set_value" in m:
            assert n["inputs"][m["new_id"]] == m["set_value"]
        elif m["old_id"] in _bg_prompt()["2"]["inputs"]:
            assert n["inputs"][m["new_id"]] == _bg_prompt()["2"]["inputs"][m["old_id"]]


def test_outputs_are_relinked_by_the_row():
    p = _bg_prompt()
    L.migrate_api_prompt(p)
    out = {o["old_idx"]: o["new_idx"] for o in ROWS["BackgroundRemoverMEC"]["output_mapping"]}
    assert p["3"]["inputs"]["source"] == ["2", out[1]]
    assert p["4"]["inputs"]["source"] == ["2", out[0]]
    assert p["1"] == _bg_prompt()["1"]                               # untouched


def test_unregistered_successor_and_unknown_ids_are_left_alone():
    p = _bg_prompt()
    assert L.migrate_api_prompt(copy.deepcopy(p), registered={"EmptyImage", "PreviewAny"}) == 0
    q = {"1": {"class_type": "SomethingElse", "inputs": {}}}
    assert L.migrate_api_prompt(q) == 0 and q["1"]["class_type"] == "SomethingElse"
    assert L.migrate_api_prompt(None) == 0


def test_every_row_migrates_a_prompt_with_no_inputs_at_all():
    """The worst case for core's KeyError: every legacy id, every optional input omitted."""
    for old, row in ROWS.items():
        p = {"9": {"class_type": old, "inputs": {}}}
        assert L.migrate_api_prompt(p) == 1, old
        assert p["9"]["class_type"] == row["new_node_id"]
        assert set(p["9"]["inputs"]) == {m["new_id"] for m in row.get("input_mapping") or [] if "set_value" in m}


def test_on_prompt_hook_never_raises():
    assert L._on_prompt({"prompt": "not a dict"}) == {"prompt": "not a dict"}
    data = {"prompt": _bg_prompt(), "client_id": "x"}
    assert L._on_prompt(data) is data and data["prompt"]["2"]["class_type"] == "MaskOpsMEC"


def test_successor_required_widgets_get_defaults_sockets_never():
    class New:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",), "mode": (["a", "b"],), "level": ("INT", {"default": 7}),
                                 "pick": ("COMBO", {"options": ["x", "y"]}), "hook": ("INT", {"forceInput": True}),
                                 "kept": ("FLOAT", {"default": 0.5})},
                    "optional": {"extra": ("STRING", {"default": "z"})}}
    node = {"class_type": "New", "inputs": {"image": ["1", 0], "kept": 0.9}}
    filled = L.fill_required_defaults(node, New)
    assert sorted(filled) == ["level", "mode", "pick"]
    assert node["inputs"] == {"image": ["1", 0], "kept": 0.9, "mode": "a", "level": 7, "pick": "x"}
