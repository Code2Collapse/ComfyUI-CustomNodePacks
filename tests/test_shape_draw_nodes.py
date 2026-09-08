"""Shape drawing tests — retargeted onto the consolidated DrawShapeMEC node.

This file previously imported five wrapper classes (DrawCircleMEC,
DrawRectangleMEC, DrawEllipseMEC, DrawPolygonMEC, DrawLineMEC). Those were
deliberately consolidated into a single `DrawShapeMEC` with a `shape` dropdown
covering 12 shapes (nodes/mask_draw_frame.py:744-754) — a product decision, not a
regression. The import of the removed names aborted pytest COLLECTION for the
whole repo, so none of the other 16 test files ran either.

Every assertion below is carried over unchanged in strength; only the call site
moved. `_parse_coords_for_batch` still exists (mask_draw_frame.py:711) so those
tests are untouched.
"""

import sys
from pathlib import Path

import pytest
import torch

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from nodes.mask_draw_frame import DrawShapeMEC, _parse_coords_for_batch  # noqa: E402


def _draw(shape, width=64, height=64, batch_size=1, **over):
    """Call DrawShapeMEC with defaults, overriding only what a test cares about."""
    kw = dict(
        width=width, height=height, shape=shape,
        cx=32.0, cy=32.0, radius=10.0,
        size_w=30.0, size_h=20.0, rx=20.0, ry=10.0,
        top_left_x=20.0, top_left_y=20.0, x2=40.0, y2=40.0, thickness=5.0,
        outer_r=20.0, inner_r=8.0, num_points=5, corner_radius=5.0,
        cross_size=20.0, arrow_length=30.0, head_length=10.0, head_width=10.0,
        # polygon is the one shape driven purely by explicit points: _build_params
        # returns points_json raw (mask_draw_frame.py:888-891) and ignores
        # cx/cy/radius/num_points entirely. A square spanning (20,20)-(45,45).
        points_json="[[20,20],[45,20],[45,45],[20,45]]",
        value=1.0, feather=0.0, rotation=0.0,
        operation="replace", batch_size=batch_size, coords_json="",
    )
    kw.update(over)
    (mask,) = DrawShapeMEC().draw(**kw)
    return mask


# ── coords_json batch parsing (helper unchanged, tests unchanged) ────────────

def test_parse_coords_constant_when_single_entry():
    result = _parse_coords_for_batch('[{"cx": 100, "cy": 100}]', 3)
    assert len(result) == 3
    assert all(r["cx"] == 100 for r in result)


def test_parse_coords_cycles_when_shorter_than_batch():
    result = _parse_coords_for_batch('[{"cx": 10}, {"cx": 20}]', 4)
    assert len(result) == 4
    assert result[0]["cx"] == 10
    assert result[1]["cx"] == 20
    assert result[2]["cx"] == 10  # cycles


def test_parse_coords_reads_both_axes():
    result = _parse_coords_for_batch('[{"cx": 50, "cy": 60}]', 1)
    assert result[0]["cx"] == 50
    assert result[0]["cy"] == 60


def test_parse_coords_empty_string_yields_blank_frames():
    result = _parse_coords_for_batch("", 3)
    assert len(result) == 3
    assert all(r == {} for r in result)


def test_parse_coords_respects_batch_size():
    result = _parse_coords_for_batch('[{"cx": 1}]', 2)
    assert len(result) == 2


# ── shapes ──────────────────────────────────────────────────────────────────

def test_circle_fills_its_centre():
    mask = _draw("circle", cx=32.0, cy=32.0, radius=10.0)
    assert mask.shape == (1, 64, 64)
    assert mask.sum() > 0
    assert mask[0, 32, 32] == 1.0


def test_circle_supports_batches():
    mask = _draw("circle", width=32, height=32, batch_size=4)
    assert mask.shape == (4, 32, 32)


def test_circle_leaves_outside_empty():
    mask = _draw("circle", cx=10.0, cy=10.0, radius=6.0)
    assert mask[0, 10, 10] > 0
    assert mask[0, 50, 50] == 0.0


def test_rectangle_inside_filled_outside_empty():
    mask = _draw("rectangle", top_left_x=20.0, top_left_y=20.0, size_w=20.0, size_h=10.0)
    assert mask.shape == (1, 64, 64)
    assert mask[0, 25, 25] == 1.0   # inside
    assert mask[0, 0, 0] == 0.0     # outside


def test_rectangle_supports_batches():
    mask = _draw("rectangle", width=32, height=32, batch_size=3)
    assert mask.shape == (3, 32, 32)


def test_ellipse_fills_its_centre():
    mask = _draw("ellipse", cx=32.0, cy=32.0, rx=20.0, ry=10.0)
    assert mask.shape == (1, 64, 64)
    assert mask[0, 32, 32] > 0


def test_polygon_fills_its_interior():
    mask = _draw("polygon")  # square (20,20)-(45,45) from the helper's points_json
    assert mask.shape == (1, 64, 64)
    assert mask[0, 30, 30] > 0


def test_polygon_ignores_num_points_despite_its_tooltip():
    """Documents a real mismatch found during this triage.

    The `num_points` tooltip reads "Number of points/sides — star, polygon"
    (mask_draw_frame.py:798), but `_build_params` returns points_json raw for
    polygon and never consults num_points (mask_draw_frame.py:888-891). Only
    `star` uses it. Pinned as CURRENT BEHAVIOUR, not endorsed — the tooltip is
    what should change. Kept as a test so a future tooltip fix has to notice this.
    """
    a = _draw("polygon", num_points=3)
    b = _draw("polygon", num_points=9)
    assert torch.equal(a, b), (
        "num_points now affects polygon — good, but the tooltip and this test "
        "must be updated together"
    )


def test_line_marks_its_path_and_not_the_corner():
    mask = _draw("line", top_left_x=32.0, top_left_y=20.0, x2=32.0, y2=45.0, thickness=5.0)
    assert mask.shape == (1, 64, 64)
    assert mask[0, 32, 32] > 0   # on the line
    assert mask[0, 0, 0] == 0.0  # off it


def test_line_supports_batches():
    mask = _draw("line", width=32, height=32, batch_size=5)
    assert mask.shape == (5, 32, 32)


def test_shapes_are_actually_distinct():
    """The consolidation's central risk: `shape` must change the output.

    A dropdown that silently ignores its value would still produce a plausible
    mask for every selection, so this is the assertion that guards the refactor.
    """
    mask_c = _draw("circle")
    mask_r = _draw("rectangle")
    assert not torch.equal(mask_c, mask_r), "circle and rectangle produced identical masks"


@pytest.mark.parametrize("shape", DrawShapeMEC.SHAPES)
def test_every_advertised_shape_draws_something(shape):
    """All 12 dropdown entries must produce a non-empty mask.

    New: the old five-wrapper suite could not cover the seven shapes the
    consolidation added, so they shipped untested.
    """
    mask = _draw(shape)
    assert mask.shape == (1, 64, 64)
    assert mask.sum() > 0, f"shape {shape!r} drew nothing"
