"""roto_json <-> the spline editor's shape format.

VectorRotoMEC shipped its shapes as a TEXT BOX. Roto is closed cubic beziers
with per-point tangent handles - nobody types one - so the node was unusable
as shipped. Rather than write a second roto editor beside the 1,428-line one
this pack already has, an adapter converts between the two formats.

The conversion has one trap that does not raise, and these tests exist for it:
the editor stores handles as an OFFSET from the point, roto_json stores them
as a POSITION on the canvas. Feed one to the other unconverted and every
tangent lands near the origin - the shape collapses toward the top-left and it
reads as "the editor mangled my roto".

The JS is executed through Node, so the real shipped module is tested rather
than a Python reimplementation that could agree with itself while the file
that actually runs disagrees.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
MODULE = PACK / "js" / "_roto_format.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is not on PATH; the adapter is ES module JS and is run for real")


def run_js(body: str):
    """Execute `body` against the real module and return its JSON result."""
    src = (
        # as_uri(), not as_posix(): on Windows an ESM import of an absolute
        # path needs a file:// URL, or node refuses it outright with
        # ERR_UNSUPPORTED_ESM_URL_SCHEME ("Received protocol 'd:'").
        "import * as R from " + json.dumps(MODULE.as_uri()) + ";\n"
        "const out = (() => { " + body + " })();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    tmp = PACK / "tests" / "_roto_probe.mjs"
    tmp.write_text(src, encoding="utf-8")
    try:
        p = subprocess.run([shutil.which("node"), str(tmp)],
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            raise AssertionError("node failed: " + p.stderr[:600])
        return json.loads(p.stdout)
    finally:
        tmp.unlink(missing_ok=True)


def J(obj):
    """A JSON string, embedded as a JS string literal."""
    return json.dumps(json.dumps(obj))


def roto(points, frame=0, w=1024, h=1024):
    """A roto document with one spline. Handles ABSOLUTE, as the file stores."""
    spline = [{"x": x, "y": y, "in": [x - 40, y], "out": [x + 40, y]}
              for x, y in points]
    return {"canvas": {"w": w, "h": h},
            "frames": [{"frame": frame, "splines": [spline]}]}


SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]


# -- the trap ---------------------------------------------------------------

def test_handles_come_back_relative():
    """The editor wants an OFFSET. Handed an absolute position it would put
    every tangent near the origin and the shape would collapse."""
    out = run_js("return R.rotoToShapes(" + J(roto(SQUARE)) + ", 0);")
    hs = out["shapes"][0]["handles"]
    assert hs[0]["in"]["x"] == pytest.approx(-40)
    assert hs[0]["out"]["x"] == pytest.approx(40)
    assert hs[0]["in"]["y"] == pytest.approx(0)


def test_handles_go_out_absolute():
    """And back the other way: the renderer reads a position, not an offset."""
    shapes = [{"points": [{"x": 100, "y": 100}, {"x": 300, "y": 100}],
               "handles": [{"in": {"x": -40, "y": 0}, "out": {"x": 40, "y": 0}},
                           {"in": {"x": -40, "y": 0}, "out": {"x": 40, "y": 0}}],
               "closed": True}]
    out = run_js("return R.shapesToRoto(" + json.dumps(shapes) + ", {frame:0});")
    p0 = out["frames"][0]["splines"][0][0]
    assert p0["in"][0] == pytest.approx(60)
    assert p0["in"][1] == pytest.approx(100)
    assert p0["out"][0] == pytest.approx(140)


def test_a_full_round_trip_is_lossless():
    """THE test. Load a roto, save it back untouched, and the shape must be
    identical - otherwise every open-and-close of the editor drifts the matte."""
    doc = roto(SQUARE)
    out = run_js(
        "const s = R.rotoToShapes(" + J(doc) + ", 0);"
        "return R.shapesToRoto(s.shapes, {previous:" + J(doc) +
        ", frame:0, canvas:s.canvas});")
    before = doc["frames"][0]["splines"][0]
    after = out["frames"][0]["splines"][0]
    assert len(after) == len(before)
    for a, b in zip(after, before):
        assert a["x"] == pytest.approx(b["x"])
        assert a["y"] == pytest.approx(b["y"])
        assert a["in"][0] == pytest.approx(b["in"][0])
        assert a["in"][1] == pytest.approx(b["in"][1])
        assert a["out"][0] == pytest.approx(b["out"][0])


# -- frames -----------------------------------------------------------------

def test_the_canvas_survives():
    """The renderer rescales by width/canvas.w. Drop it and every shape
    silently changes size."""
    out = run_js("return R.rotoToShapes(" + J(roto(SQUARE, w=1920, h=1080)) + ", 0);")
    assert out["canvas"] == {"w": 1920, "h": 1080}


def test_an_unkeyed_frame_shows_the_key_before_it():
    """What the renderer does is hold the previous key, so the editor shows
    that rather than an empty canvas - otherwise scrubbing to frame 12 looks
    like the roto vanished."""
    doc = roto(SQUARE, frame=0)
    doc["frames"].append({"frame": 20, "splines": [
        [{"x": 500, "y": 500, "in": [460, 500], "out": [540, 500]},
         {"x": 700, "y": 500, "in": [660, 500], "out": [740, 500]}]]})
    out = run_js("return R.rotoToShapes(" + J(doc) + ", 12);")
    assert out["shapes"][0]["points"][0]["x"] == pytest.approx(100), \
        "frame 12 did not fall back to the key at frame 0"
    assert out["frames"] == [0, 20]


def test_saving_one_frame_leaves_the_others_alone():
    """Editing frame 20 must not wipe frame 0."""
    doc = roto(SQUARE, frame=0)
    doc["frames"].append({"frame": 20, "splines": []})
    shapes = [{"points": [{"x": 9, "y": 9}, {"x": 19, "y": 9}],
               "handles": [{"in": {"x": 0, "y": 0}, "out": {"x": 0, "y": 0}},
                           {"in": {"x": 0, "y": 0}, "out": {"x": 0, "y": 0}}],
               "closed": True}]
    out = run_js("return R.shapesToRoto(" + json.dumps(shapes) +
                 ", {previous:" + J(doc) + ", frame:20});")
    assert [f["frame"] for f in out["frames"]] == [0, 20]
    assert out["frames"][0]["splines"][0][0]["x"] == pytest.approx(100)
    assert out["frames"][1]["splines"][0][0]["x"] == pytest.approx(9)


def test_a_new_frame_is_inserted_in_order():
    doc = roto(SQUARE, frame=0)
    doc["frames"].append({"frame": 50, "splines": []})
    shapes = [{"points": [{"x": 1, "y": 1}, {"x": 2, "y": 2}],
               "handles": [{"in": {"x": 0, "y": 0}, "out": {"x": 0, "y": 0}},
                           {"in": {"x": 0, "y": 0}, "out": {"x": 0, "y": 0}}]}]
    out = run_js("return R.shapesToRoto(" + json.dumps(shapes) +
                 ", {previous:" + J(doc) + ", frame:25});")
    assert [f["frame"] for f in out["frames"]] == [0, 25, 50]


def test_the_last_keyframe_cannot_be_deleted():
    """Zero frames rasterises nothing at all - an empty matte with no way back."""
    out = run_js("return R.removeKeyframe(" + J(roto(SQUARE)) + ", 0);")
    assert len(out["frames"]) == 1


def test_deleting_one_of_several_keyframes_works():
    doc = roto(SQUARE, frame=0)
    doc["frames"].append({"frame": 10, "splines": []})
    out = run_js("return R.removeKeyframe(" + J(doc) + ", 10);")
    assert [f["frame"] for f in out["frames"]] == [0]


# -- the tearing check ------------------------------------------------------

def test_matching_point_counts_are_accepted():
    doc = roto(SQUARE, frame=0)
    doc["frames"].append({"frame": 10, "splines": [
        [{"x": i * 10, "y": i * 10, "in": [0, 0], "out": [0, 0]} for i in range(4)]]})
    out = run_js("return R.keyframesAgree(" + J(doc) + ");")
    assert out["ok"] is True


def test_a_point_count_mismatch_is_caught_with_the_fix():
    """The renderer matches point 1 to point 1. A key with a different count
    interpolates against the wrong point and the shape TEARS between keys -
    and it does not raise, it just renders wrongly."""
    doc = roto(SQUARE, frame=0)                                   # 4 points
    doc["frames"].append({"frame": 10, "splines": [
        [{"x": 0, "y": 0, "in": [0, 0], "out": [0, 0]} for _ in range(3)]]})
    out = run_js("return R.keyframesAgree(" + J(doc) + ");")
    assert out["ok"] is False
    assert "tear" in out["why"]
    assert "EVERY keyframe" in out["why"], "the fix is not stated"


def test_a_shape_missing_from_a_keyframe_is_caught():
    doc = roto(SQUARE, frame=0)
    doc["frames"].append({"frame": 10, "splines": []})
    out = run_js("return R.keyframesAgree(" + J(doc) + ");")
    assert out["ok"] is False
    assert "pop in and out" in out["why"]


def test_a_single_keyframe_always_agrees():
    out = run_js("return R.keyframesAgree(" + J(roto(SQUARE)) + ");")
    assert out["ok"] is True


# -- not losing work --------------------------------------------------------

def test_corrupt_previous_json_does_not_lose_the_edit():
    """A previous value that will not parse must not take the shape being
    saved down with it."""
    shapes = [{"points": [{"x": 5, "y": 5}, {"x": 6, "y": 6}],
               "handles": [{"in": {"x": 0, "y": 0}, "out": {"x": 0, "y": 0}},
                           {"in": {"x": 0, "y": 0}, "out": {"x": 0, "y": 0}}]}]
    out = run_js("return R.shapesToRoto(" + json.dumps(shapes) +
                 ", {previous:" + json.dumps("{not json") + ", frame:0});")
    assert out["frames"][0]["splines"][0][0]["x"] == pytest.approx(5)


def test_unparseable_roto_gives_an_empty_editor_not_a_crash():
    out = run_js("return R.rotoToShapes(" + json.dumps("{nope") + ", 0);")
    assert out["shapes"] == []


def test_a_one_point_spline_is_dropped():
    """Two points is the minimum the rasteriser draws; one renders nothing and
    would sit in the file looking like a shape."""
    shapes = [{"points": [{"x": 1, "y": 1}],
               "handles": [{"in": {"x": 0, "y": 0}, "out": {"x": 0, "y": 0}}]}]
    out = run_js("return R.shapesToRoto(" + json.dumps(shapes) + ", {frame:0});")
    assert out["frames"][0]["splines"] == []


def test_an_empty_document_is_still_valid():
    out = run_js("return R.emptyRoto(1920, 1080);")
    assert out["canvas"] == {"w": 1920, "h": 1080}
    assert out["frames"] == [{"frame": 0, "splines": []}]


def test_the_node_this_targets_still_reads_roto_json():
    """The adapter writes roto_json by name. Rename it and the editor saves
    into nothing, with no error at all."""
    src = (PACK / "nodes" / "roto.py").read_text(encoding="utf-8")
    assert "class VectorRotoMEC" in src
    assert '"roto_json"' in src
    assert "VectorRotoMEC" in MODULE.read_text(encoding="utf-8")


# -- the binding ------------------------------------------------------------

EDITOR = PACK / "js" / "spline_mask_editor.js"


def test_the_editor_is_bound_to_the_roto_node():
    """Without this the adapter exists and nothing uses it."""
    src = EDITOR.read_text(encoding="utf-8")
    assert "ROTO_NODES" in src
    assert "_roto_format.js" in src


def test_the_editor_switches_format_on_both_read_and_write():
    """Saving in one format and loading in the other would round-trip a
    shape into the wrong place on every reopen."""
    src = EDITOR.read_text(encoding="utf-8")
    assert "if (this.isRoto) return this.saveRoto();" in src
    assert "if (this.isRoto) return this.loadRoto();" in src


def test_the_raw_json_widget_is_hidden():
    """It is authored entirely by the editor. Left visible it is a 40-line
    textarea that desynchronises from the shapes the moment anyone edits it."""
    src = EDITOR.read_text(encoding="utf-8")
    assert 'hideWidget(node.widgets?.find(w => w.name === "roto_json"));' in src


def test_the_node_has_the_frame_widget_the_editor_reads():
    src = (PACK / "nodes" / "roto.py").read_text(encoding="utf-8")
    assert '"roto_frame"' in src
    assert '"roto_frame"' in EDITOR.read_text(encoding="utf-8")


def test_the_frame_widget_is_declared_last():
    """Widget values serialise as a flat POSITIONAL array. A widget inserted
    anywhere but the end shifts every value after it, so an existing workflow
    silently re-reads its numbers into the wrong parameters."""
    src = (PACK / "nodes" / "roto.py").read_text(encoding="utf-8")
    i = src.index("class VectorRotoMEC")
    block = src[i:src.index("def rasterize", i)]
    names = re.findall(r'"([a-z_]+)":\s*\(', block)
    assert names[-1] == "roto_frame", f"roto_frame is not last: {names}"


def test_the_renderer_ignores_the_editor_cursor():
    """Reading it would make the render depend on where the editor happened
    to be parked when the workflow was saved."""
    src = (PACK / "nodes" / "roto.py").read_text(encoding="utf-8")
    body = src[src.index("def rasterize"):]
    body = body[:body.index("\n    def ") if "\n    def " in body else len(body)]
    # The signature wraps, so "def rasterize" is not on the line that declares
    # the parameter. A declaration is `roto_frame: int` (annotated); anything
    # else mentioning it is a USE.
    uses = [ln for ln in body.splitlines()
            if "roto_frame" in ln
            and not ln.strip().startswith("#")
            and "roto_frame: int" not in ln]
    assert not uses, f"rasterize() reads the editor cursor: {uses}"
