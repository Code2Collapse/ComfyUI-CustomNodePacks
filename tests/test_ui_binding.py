"""A node that asks a human to type structured data needs an editor.

The audit that produced this file scanned all 638 registered nodes for the
sharpest symptom of a missing front-end: a multiline STRING widget whose name
says it holds coordinates, points, a spline or a timeline, on a node with no
JS bound to it. Nobody types a bezier, and nobody types
{"x":100,"y":200,"label":1} forty times.

Twelve nodes matched. Most were fine - provenance a machine writes, a subgraph
you PASTE rather than author, an optional numeric override with an example in
its own tooltip. Two were not: the matting pipelines shipped point prompts as
a text box, when the point prompts are places you click on a picture.

They already read exactly the format the existing editor writes, so the fix
was a binding, not new code. This test pins that binding, and pins the
FORMAT AGREEMENT underneath it - if either side changes shape, a click stops
reaching the backend and the node quietly does nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

EDITOR = PACK / "js" / "points_bbox_editor.js"

#: Nodes whose point prompts are CLICKS, so a bare text box is not a usable
#: front-end for them.
NEEDS_POINT_EDITOR = (
    "MaskEditMEC",
    "SAMMaskGeneratorMEC",
    "SAMViTMattePipelineMEC",
    "SeCMatAnyonePipelineMEC",
)


def editor_source() -> str:
    assert EDITOR.exists(), f"{EDITOR.name} is gone - every point node lost its editor"
    return EDITOR.read_text(encoding="utf-8")


def target_list() -> list[str]:
    m = re.search(r"const TARGET_NODES\s*=\s*\[(.*?)\]", editor_source(), re.S)
    assert m, "TARGET_NODES is no longer a literal array - the binding cannot be checked"
    return re.findall(r"""["']([^"']+)["']""", m.group(1))


@pytest.mark.parametrize("node_id", NEEDS_POINT_EDITOR)
def test_every_click_driven_node_has_the_editor_bound(node_id):
    assert node_id in target_list(), (
        f"{node_id} takes point prompts but no editor is bound to it, so its "
        "points_json is a raw text box. Add it to TARGET_NODES in "
        "points_bbox_editor.js.")


#: The node modules, read as SOURCE rather than imported.
#:
#: Importing them drags in ComfyUI's folder_paths, which has no base_path
#: until a real server boot - so the test would skip on the machine where it
#: matters most. The contract being checked is a widget NAME, which is right
#: there in INPUT_TYPES, so reading the file tests the same thing and always
#: runs.
BACKENDS = (
    ("nodes/sam_vitmatte_pipeline.py", "SAMViTMattePipelineMEC"),
    ("nodes/sec_matanyone_pipeline.py", "SeCMatAnyonePipelineMEC"),
)


@pytest.mark.parametrize("path,cls_name", BACKENDS)
def test_the_backend_still_takes_the_fields_the_editor_writes(path, cls_name):
    """The editor writes points_json and bbox_json BY NAME. Rename either and
    the clicks stop arriving, with no error - the node simply behaves as
    though nothing was clicked."""
    src = (PACK / path).read_text(encoding="utf-8")
    assert f"class {cls_name}" in src, f"{cls_name} is not in {path}"
    assert '"points_json"' in src, f"{cls_name} no longer reads points_json"
    assert '"bbox_json"' in src, f"{cls_name} no longer reads bbox_json"


def test_the_editor_writes_both_fields_by_name():
    src = editor_source()
    assert 'find("points_json")' in src, "the editor stopped writing points_json"
    assert 'find("bbox_json")' in src, "the editor stopped writing bbox_json"


@pytest.mark.parametrize("path,cls_name", BACKENDS)
def test_the_point_format_is_documented_on_the_widget(path, cls_name):
    """The tooltip is the contract a user reads. It has to name the shape the
    editor actually writes, or someone hand-editing produces something the
    backend silently ignores."""
    src = (PACK / path).read_text(encoding="utf-8")
    i = src.index('"points_json"')
    window = src[i:i + 400]
    assert '"x"' in window and "label" in window, (
        f"{cls_name}'s points_json widget does not document its shape")


def test_the_audit_threshold_is_recorded():
    """Twelve nodes matched the scan; ten were judged fine and two were not.
    Recording which two stops the next audit re-opening the same argument."""
    assert set(NEEDS_POINT_EDITOR) >= {
        "SAMViTMattePipelineMEC", "SeCMatAnyonePipelineMEC"}, (
        "the two nodes the audit actually found were dropped from the list")
