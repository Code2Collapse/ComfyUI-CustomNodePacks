"""MaskOps folds to what the chosen PIPELINE reads.

pipeline="onyx" is one fixed pipeline (SAM 3.1 video session + tiled
ViTMatte): the segmenter picker, the accuracy stages, the luma pre-key and the
pixel-sized trimap knobs read nothing there, and ONYX's own controls read
nothing on the legacy cascade. The server spec carries both lists; these check
the reference implementation (`visible`) the front-end mirrors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

vis = pytest.importorskip("nodes.mask_matting._visibility",
                          reason="mask_matting registry unavailable")

ONYX_ONLY = {"scene_prompts", "band_scale", "matte_tile", "temporal_stabilise"}


def test_onyx_shows_its_controls_and_drops_the_cascade_machinery():
    shown = set(vis.visible("auto_best", "vitmatte", {"pipeline": "onyx"}))
    assert ONYX_ONLY <= shown
    assert "pipeline" in shown and "model" in shown and "matter" in shown
    for gone in ("segmenter", "tta_flip", "multiscale", "auto_quality",
                 "enable_luma_key", "trimap_dilate", "robust_propagation"):
        assert gone not in shown, gone
    # the single-object shorthand still reads the tracking window
    assert {"frame_annotation", "tracking_direction"} <= shown


def test_the_cascade_folds_onyx_controls_away():
    shown = set(vis.visible("sam3", "vitmatte", {"pipeline": "cascade (legacy)"}))
    assert not (ONYX_ONLY & shown)
    assert "pipeline" in shown


def test_the_spec_tells_the_front_end_about_onyx():
    spec = vis.build_spec()
    assert set(spec["onyx"]["only"]) == ONYX_ONLY
    assert "segmenter" in spec["onyx"]["ignores"]
    assert "pipeline" in spec["always"]
    for name in ONYX_ONLY:
        assert spec["toggles"][name] == ["pipeline", "cascade (legacy)"]


def test_no_onyx_rule_names_a_widget_the_node_lacks():
    node = pytest.importorskip("nodes.mask_matting.node", reason="MaskOps unavailable")
    spec_in = node.MaskOpsMEC.INPUT_TYPES()
    real = set(spec_in.get("required") or {}) | set(spec_in.get("optional") or {})
    named = set(vis.ONYX_IGNORES) | set(vis.ONYX_KEEPS) | ONYX_ONLY | {"pipeline"}
    assert not (named - real), sorted(named - real)
