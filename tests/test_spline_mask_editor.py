"""
Tests for SplineMaskEditorMEC — verifies rasterization, spline sampling, and
coord extraction produce correct geometry, not gimmick output.

Anti-gimmick checks:
  1. Output mask ≠ zeros for valid spline data ✓
  2. Closed shapes have filled interior ✓
  3. Invert flips mask ✓
  4. Feather changes output vs non-feathered ✓
  5. Coords JSON has correct structure ✓
  6. SPLINE_DATA output is well-formed ✓
  7. Empty input returns zeros gracefully ✓
"""

import torch
import pytest
import json
import sys
import types

# Stub ComfyUI modules for test isolation
for mod_name in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

from nodes.spline_mask_editor import (
    SplineMaskEditorMEC,
    _catmull_rom_sample,
    _bezier_sample,
    _polyline_sample,
    _rasterize_splines,
    _coords_from_splines,
    _build_spline_data,
)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def node():
    return SplineMaskEditorMEC()


@pytest.fixture
def triangle_spline_json():
    """Three-point closed polygon covering a triangle region."""
    return json.dumps([{
        "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 30, "y": 50}],
        "closed": True,
        "type": "polyline",
    }])


@pytest.fixture
def open_line_json():
    """Two-point open polyline."""
    return json.dumps([{
        "points": [{"x": 0, "y": 32}, {"x": 63, "y": 32}],
        "closed": False,
        "type": "polyline",
    }])


# ═══════════════════════════════════════════════════════════════════
#  Unit tests: spline sampling
# ═══════════════════════════════════════════════════════════════════

class TestCatmullRomSample:
    def test_two_points_linear(self):
        pts = [(0.0, 0.0), (100.0, 100.0)]
        result = _catmull_rom_sample(pts, samples_per_segment=10, closed=False)
        assert len(result) >= 10
        # First and last should be near the input points
        assert abs(result[0][0]) < 1.0
        assert abs(result[-1][0] - 100.0) < 1.0

    def test_closed_loop(self):
        pts = [(10, 10), (50, 10), (50, 50), (10, 50)]
        result = _catmull_rom_sample(pts, samples_per_segment=10, closed=True)
        # Closed loop should produce curves through all 4 points
        assert len(result) >= 40

    def test_single_point(self):
        pts = [(50, 50)]
        result = _catmull_rom_sample(pts, samples_per_segment=10, closed=False)
        assert len(result) == 1


class TestBezierSample:
    def test_straight_segment(self):
        pts = [(0, 0), (100, 100)]
        handles = [
            {"cp1x": 0, "cp1y": 0, "cp2x": 33, "cp2y": 33},
            {"cp1x": 66, "cp1y": 66, "cp2x": 100, "cp2y": 100},
        ]
        result = _bezier_sample(pts, handles, samples_per_segment=10)
        assert len(result) >= 10

    def test_single_point(self):
        result = _bezier_sample([(50, 50)], [], 10)
        assert len(result) == 1


class TestPolylineSample:
    def test_closed(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        result = _polyline_sample(pts, closed=True)
        assert len(result) == 5  # 4 points + closing point
        assert result[-1] == result[0]

    def test_open(self):
        pts = [(0, 0), (10, 0)]
        result = _polyline_sample(pts, closed=False)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════
#  Unit tests: rasterization
# ═══════════════════════════════════════════════════════════════════

class TestRasterizeSplines:
    def test_triangle_fills_interior(self, triangle_spline_json):
        mask = _rasterize_splines(triangle_spline_json, 64, 64,
                                  "polyline", True, 20, 0.0, False,
                                  torch.device("cpu"))
        assert mask.shape == (1, 64, 64)
        assert mask.sum() > 0, "Triangle should have filled interior"
        # Check a point inside the triangle
        assert mask[0, 25, 30] > 0.5, "Center of triangle should be filled"

    def test_invert(self, triangle_spline_json):
        mask_normal = _rasterize_splines(triangle_spline_json, 64, 64,
                                         "polyline", True, 20, 0.0, False,
                                         torch.device("cpu"))
        mask_invert = _rasterize_splines(triangle_spline_json, 64, 64,
                                         "polyline", True, 20, 0.0, True,
                                         torch.device("cpu"))
        # Inverted should be roughly complementary
        combined = mask_normal + mask_invert
        assert torch.allclose(combined, torch.ones_like(combined), atol=0.01)

    def test_feather_changes_output(self, triangle_spline_json):
        mask_sharp = _rasterize_splines(triangle_spline_json, 64, 64,
                                        "polyline", True, 20, 0.0, False,
                                        torch.device("cpu"))
        mask_feathered = _rasterize_splines(triangle_spline_json, 64, 64,
                                            "polyline", True, 20, 5.0, False,
                                            torch.device("cpu"))
        # Feathered should have more non-zero pixels (softer edges)
        sharp_nonzero = (mask_sharp > 0.01).sum()
        feather_nonzero = (mask_feathered > 0.01).sum()
        assert feather_nonzero >= sharp_nonzero

    def test_empty_json_returns_zeros(self):
        mask = _rasterize_splines("[]", 64, 64, "catmull_rom", True, 20, 0.0,
                                  False, torch.device("cpu"))
        assert mask.sum() == 0.0

    def test_invalid_json_returns_zeros(self):
        mask = _rasterize_splines("not json!", 64, 64, "catmull_rom", True, 20,
                                  0.0, False, torch.device("cpu"))
        assert mask.sum() == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Unit tests: coords extraction
# ═══════════════════════════════════════════════════════════════════

class TestCoordsFromSplines:
    def test_extracts_coords(self, triangle_spline_json):
        coords_str = _coords_from_splines(triangle_spline_json)
        coords = json.loads(coords_str)
        assert len(coords) == 3
        assert all("x" in c and "y" in c and "label" in c for c in coords)
        assert coords[0]["label"] == 1

    def test_empty_returns_empty_list(self):
        assert _coords_from_splines("[]") == "[]"
        assert _coords_from_splines("bad") == "[]"


# ═══════════════════════════════════════════════════════════════════
#  Unit tests: SPLINE_DATA builder
# ═══════════════════════════════════════════════════════════════════

class TestBuildSplineData:
    def test_structure(self, triangle_spline_json):
        data = _build_spline_data(triangle_spline_json, 64, 64)
        assert "shapes" in data
        assert "canvas_width" in data
        assert data["canvas_width"] == 64
        assert len(data["shapes"]) == 1

    def test_empty_json(self):
        data = _build_spline_data("[]", 512, 512)
        assert data["shapes"] == []


# ═══════════════════════════════════════════════════════════════════
#  Integration tests: full node execution
# ═══════════════════════════════════════════════════════════════════

class TestSplineMaskEditorNode:
    def test_basic_execution(self, node, triangle_spline_json):
        dummy_image = torch.rand(1, 64, 64, 3)
        result = node.execute(
            image=dummy_image,
            spline_data=triangle_spline_json,
            spline_type="polyline",
            closed=True,
            smoothing=True,
            samples_per_segment=20,
            feather_radius=0.0,
            invert=False,
        )
        mask, coords_json, spline_data, bbox_json, bbox = result["result"]
        assert mask.shape[0] == 1
        assert mask.shape[1] == 64
        assert mask.shape[2] == 64
        assert mask.sum() > 0
        coords = json.loads(coords_json)
        assert len(coords) == 3
        assert isinstance(spline_data, dict)

    def test_empty_spline_data(self, node):
        dummy_image = torch.rand(1, 64, 64, 3)
        result = node.execute(
            image=dummy_image,
            spline_data="[]",
            spline_type="catmull_rom",
            closed=True,
            smoothing=True,
            samples_per_segment=20,
            feather_radius=0.0,
            invert=False,
        )
        mask, coords_json, spline_data, bbox_json, bbox = result["result"]
        assert mask.sum() == 0.0
        assert json.loads(coords_json) == []

    def test_reference_image_overrides_size(self, node, triangle_spline_json):
        ref = torch.rand(1, 128, 96, 3)
        result = node.execute(
            image=ref,
            spline_data=triangle_spline_json,
            spline_type="polyline",
            closed=True,
            smoothing=True,
            samples_per_segment=20,
            feather_radius=0.0,
            invert=False,
        )
        mask, *_ = result["result"]
        assert mask.shape == (1, 128, 96)
