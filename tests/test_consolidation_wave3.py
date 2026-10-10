"""L7.65 consolidation wave 3: merged nodes give the same results through their successor.

Same method as test_consolidation_wave2.py: the old node's inputs go through its row in nodes/_legacy_replacements.json
(as the workflow migration moves them) into the successor, and every mapped output is compared with the old node's.
P13: the eight Layer Effect nodes are kept as the engines of Layer Effects (C2C), so the reference is the engine called
the way the old node was.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.test_consolidation_wave2 import check, migrate_kwargs

LAYER_EFFECTS = {
    "LayerEffectDropShadowMEC": "drop_shadow", "LayerEffectInnerShadowMEC": "inner_shadow",
    "LayerEffectOuterGlowMEC": "outer_glow", "LayerEffectInnerGlowMEC": "inner_glow",
    "LayerEffectStrokeMEC": "stroke", "LayerEffectColorOverlayMEC": "color_overlay",
    "LayerEffectGradientOverlayMEC": "gradient_overlay", "LayerEffectGradientMapMEC": "gradient_map",
}


def _plate(seed=0, b=2, h=24, w=32):
    g = torch.Generator().manual_seed(seed)
    layer = torch.rand(b, h, w, 4, generator=g)
    layer[..., 3] = (torch.rand(b, h, w, generator=g) > 0.45).float()
    return layer, torch.rand(b, h, w, 3, generator=g)


def _old_inputs(engine, layer, bg, overrides):
    """The old node's inputs: its own defaults (first combo entry when no default), the plate, the overrides."""
    spec = engine.INPUT_TYPES()
    vals = {}
    for name, s in {**spec["required"], **spec.get("optional", {})}.items():
        opts = s[1] if len(s) > 1 else {}
        if "default" in opts:
            vals[name] = opts["default"]
        elif isinstance(s[0], (list, tuple)):
            vals[name] = s[0][0]
    if "image" in spec["required"]:
        vals["image"] = layer
    else:
        vals["layer_image"] = layer
        vals["background_image"] = bg
    vals.update(overrides)
    return vals


@pytest.mark.parametrize("old_id", sorted(LAYER_EFFECTS))
@pytest.mark.parametrize("overrides", [{}, {"opacity": 60, "blend_mode": "multiply"}])
def test_layer_effect_through_layer_effects(old_id, overrides):
    from nodes.layer_effects import nodes as LE

    engine = getattr(LE, old_id)
    if old_id == "LayerEffectGradientMapMEC":
        overrides = {k: v for k, v in overrides.items() if k != "blend_mode"}
    layer, bg = _plate()
    old = _old_inputs(engine, layer, bg, overrides)
    expected = engine().execute(**old)
    defaults = {k: (v[1].get("default") if len(v) > 1 and "default" in v[1] else
                    (v[0][0] if isinstance(v[0], (list, tuple)) else None))
                for k, v in LE.LayerEffectsMEC.INPUT_TYPES()["required"].items()}
    defaults.pop("layer_image")

    def run_new(**k):
        return LE.LayerEffectsMEC().execute(**{**defaults, **k})

    check(old_id, old, expected, run_new)


def test_layer_effects_stack_equals_chaining_the_old_nodes():
    from nodes.layer_effects import nodes as LE

    layer, bg = _plate(seed=3)
    kw = {k: (v[1].get("default") if len(v) > 1 and "default" in v[1] else
              (v[0][0] if isinstance(v[0], (list, tuple)) else None))
          for k, v in LE.LayerEffectsMEC.INPUT_TYPES()["required"].items()}
    kw.pop("layer_image")
    for key, _e, _p in LE._EFFECTS:
        kw[key] = key in ("drop_shadow", "color_overlay", "stroke")
    kw["color_overlay_opacity"] = 40
    out, em, report = LE.LayerEffectsMEC().execute(layer_image=layer, background_image=bg, **kw)
    _b, _l, matte, _n, _bb = LE._prepare_compositing("t", layer, bg, None, kw["invert_mask"])
    p = lambda key: {q: kw[LE.effect_widget_name(key, q)] for q in dict((k, ps) for k, _e, ps in LE._EFFECTS)[key]}
    a = LE.LayerEffectDropShadowMEC().execute(layer, kw["invert_mask"], dissolve_seed=0, background_image=bg, **p("drop_shadow"))
    b = LE.LayerEffectColorOverlayMEC().execute(a[0], False, dissolve_seed=0, background_image=a[0], layer_mask=matte,
                                                **p("color_overlay"))
    c = LE.LayerEffectStrokeMEC().execute(b[0], False, dissolve_seed=0, background_image=b[0], layer_mask=matte,
                                          **p("stroke"))
    assert torch.equal(out, c[0])
    assert torch.equal(em, torch.maximum(torch.maximum(a[1], b[1]), c[1]))
    assert report.count("\n") == 2


def test_order_is_respected_and_checked():
    from nodes.layer_effects.nodes import DEFAULT_ORDER, LayerEffectsMEC

    seq = LayerEffectsMEC.sequence
    assert seq(DEFAULT_ORDER, ["stroke", "drop_shadow"]) == ["drop_shadow", "stroke"]
    assert seq("stroke, drop_shadow", ["stroke", "drop_shadow"]) == ["stroke", "drop_shadow"]
    assert seq("stroke", ["stroke", "inner_glow", "drop_shadow"]) == ["stroke", "drop_shadow", "inner_glow"]
    with pytest.raises(ValueError, match="unknown effect"):
        seq("stroke, bevel", ["stroke"])


def test_no_effect_on_puts_the_layer_over_the_background():
    from nodes.layer_effects import nodes as LE

    layer, bg = _plate(seed=4)
    kw = {k: (v[1].get("default") if len(v) > 1 and "default" in v[1] else
              (v[0][0] if isinstance(v[0], (list, tuple)) else None))
          for k, v in LE.LayerEffectsMEC.INPUT_TYPES()["required"].items()}
    kw.pop("layer_image")
    for key, _e, _p in LE._EFFECTS:
        kw[key] = False
    out, em, report = LE.LayerEffectsMEC().execute(layer_image=layer, background_image=bg, **kw)
    _b, _l, matte, _n, _bb = LE._prepare_compositing("t", layer, bg, None, kw["invert_mask"])
    m = matte.unsqueeze(-1)
    assert torch.allclose(out, layer[..., :3] * m + bg * (1 - m), atol=1e-6)
    assert "no effect switched on" in report


def test_gradient_map_alone_needs_no_matte():
    from nodes.layer_effects import nodes as LE

    img = torch.rand(1, 8, 8, 3, generator=torch.Generator().manual_seed(5))       # RGB, no alpha, no mask
    kw = {k: (v[1].get("default") if len(v) > 1 and "default" in v[1] else
              (v[0][0] if isinstance(v[0], (list, tuple)) else None))
          for k, v in LE.LayerEffectsMEC.INPUT_TYPES()["required"].items()}
    kw.pop("layer_image")
    for key, _e, _p in LE._EFFECTS:
        kw[key] = key == "gradient_map"
    out, _em, _r = LE.LayerEffectsMEC().execute(layer_image=img, **kw)
    ref = LE.LayerEffectGradientMapMEC().execute(img, *[kw[LE.effect_widget_name("gradient_map", q)] for q in
                                                        ("start_color", "mid_color", "end_color", "mid_point", "opacity")])
    assert torch.equal(out, ref[0])


# ── P29: Batch Version Manager -> Folder Version Incrementer, layout "show / shot / task / version" ────────────────

@pytest.mark.parametrize("reserve", [False, True])
@pytest.mark.parametrize("forward_slash", [True, False])
@pytest.mark.parametrize("existing", [[], ["v001", "v002", "v007"], ["v001", "x", "v12"]])
def test_batch_version_manager_through_folder_incrementer(tmp_path, reserve, forward_slash, existing):
    import shutil

    from folder_incrementer import FolderIncrementer
    from nodes.batch_version_manager import BatchVersionManagerMEC

    task_dir = tmp_path / "proj" / "sh020" / "comp"
    for d in existing:
        (task_dir / d).mkdir(parents=True)
    old = {"root": str(tmp_path), "show": "proj", "shot": "sh020", "task": "comp", "reserve": reserve, "padding": 3,
           "max_retries": 5, "min_version": 2, "forward_slash": forward_slash, "write_manifest": True}
    expected = BatchVersionManagerMEC().allocate(**old)
    if reserve:                                        # put the tree back exactly as it was before the second run
        shutil.rmtree(task_dir / expected[2])
    check("BatchVersionManagerMEC", old, expected, lambda **k: FolderIncrementer().increment(**k))


def test_folder_incrementer_default_layout_keeps_its_ten_outputs(tmp_path):
    from folder_incrementer import FolderIncrementer

    out = FolderIncrementer().increment(source_filename="shotA.mov", base_path=str(tmp_path), path_style="linux")
    assert len(out) == 12
    version_string, num, folder, sub, prefix, filename = out[:6]
    assert (version_string, num, folder) == ("v001", 1, "shotA")
    assert out[10] == "/".join([str(tmp_path).replace("\\", "/"), sub])          # absolute twin of subfolder_path
    import json as _json
    info = _json.loads(out[11])
    assert info["path"] == out[10] and info["label"] == "v001" and info["layout"].startswith("source")


# ── P03: eight mask nodes -> Mask Tools (C2C) ─────────────────────────────────────────────────────────────────────

def _mask_tools_defaults():
    from nodes.mask_toolkit.nodes import MaskToolsMEC

    return {k: (v[1].get("default") if len(v) > 1 and "default" in v[1] else
                (v[0][0] if isinstance(v[0], (list, tuple)) else None))
            for k, v in MaskToolsMEC.INPUT_TYPES()["required"].items()}


def _engine_call(old_id, inputs):
    from nodes.helpers.helpers import MaskBatchCombineMEC
    from nodes.luminance_keyer import LuminanceKeyerMEC
    from nodes.mask_toolkit import nodes as MT
    from nodes.shuffle import ShuffleMEC

    if old_id == "LuminanceKeyerMEC":
        return LuminanceKeyerMEC().key_luminance(**inputs)
    if old_id == "MaskBatchCombineMEC":
        return MaskBatchCombineMEC().combine(**inputs)
    if old_id == "ShuffleMEC":
        return ShuffleMEC().shuffle(**inputs)
    return getattr(MT, old_id)().execute(**inputs)


_G = torch.Generator().manual_seed(11)
_IMG = torch.rand(2, 18, 22, 3, generator=_G)
_MASK = torch.rand(2, 18, 22, generator=_G)
_MASK_B = torch.rand(2, 18, 22, generator=_G)
MASK_CASES = [
    ("MaskFromColorMEC", {"image": _IMG, "color": "#33AA55", "colorspace": "lab", "tolerance": 30.0, "soft_falloff": 12.0,
                          "invert": False}),
    ("MaskFromColorMEC", {"image": _IMG, "color": "#FFFFFF", "colorspace": "rgb", "tolerance": 50.0, "soft_falloff": 10.0,
                          "invert": True}),
    ("LuminanceKeyerMEC", {"image": _IMG, "mode": "custom", "low": 0.2, "high": 0.7, "gamma": 1.3, "falloff": 0.5,
                           "invert": False, "channel": "luma", "low_soft": 0.05, "high_soft": 0.1, "invert_key": False}),
    ("LuminanceKeyerMEC", {"image": _IMG, "mode": "highlights", "low": 0.0, "high": 1.0, "gamma": 1.0, "falloff": 1.0,
                           "invert": True, "channel": "saturation", "low_soft": 0.0, "high_soft": 0.0, "invert_key": True}),
    ("MaskGradientMEC", {"width": 64, "height": 40, "gradient_type": "radial", "angle": 30.0, "center_x": 0.3,
                         "center_y": 0.6, "start": 0.1, "end": 0.9}),
    ("MaskGradientMEC", {"width": 512, "height": 512, "gradient_type": "linear", "angle": 0.0, "center_x": 0.5,
                         "center_y": 0.5, "start": 0.0, "end": 1.0, "size_as": _IMG}),
    ("MaskGrainMEC", {"mask": _MASK, "amount": 9, "seed": 1234, "grain_size": 3, "invert": False}),
    ("MaskMotionBlurMEC", {"mask": _MASK, "angle": 45.0, "distance": 7, "invert": True}),
    ("EdgeSpreadMEC", {"image": _IMG, "spread": 3, "invert_mask": False, "mask": (_MASK > 0.5).float()}),
    ("MaskBatchCombineMEC", {"mask_a": _MASK, "mask_b": _MASK_B, "op": "xor"}),
    ("MaskBatchCombineMEC", {"mask_a": _MASK, "mask_b": _MASK_B, "op": "subtract"}),
    ("ShuffleMEC", {"image": _IMG, "out_R": "B", "out_G": "Lum", "out_B": "InvR", "out_A": "G",
                    "premultiply_output": True}),
]


@pytest.mark.parametrize("old_id,inputs", MASK_CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(MASK_CASES)])
def test_mask_node_through_mask_tools(old_id, inputs):
    from nodes.mask_toolkit.nodes import MaskToolsMEC

    expected = _engine_call(old_id, inputs)
    defaults = _mask_tools_defaults()
    check(old_id, inputs, expected, lambda **k: MaskToolsMEC().execute(**{**defaults, **k}))


def _run_mode(mode, **kw):
    from nodes.mask_toolkit.nodes import MaskToolsMEC

    return MaskToolsMEC().execute(**{**_mask_tools_defaults(), "mode": mode, **kw})


def test_grade_gain_gamma_clamp_invert():
    m = torch.tensor([[[0.0, 0.25, 0.5, 1.0]]])
    g, img, _r = _run_mode("grade", mask=m, grade_gain=2.0, grade_gamma=1.0, grade_clamp=True, grade_invert=False)
    assert torch.allclose(g, torch.tensor([[[0.0, 0.5, 1.0, 1.0]]]))
    g, _i, _r = _run_mode("grade", mask=m, grade_gain=1.0, grade_gamma=2.0, grade_clamp=True, grade_invert=True)
    assert torch.allclose(g, 1.0 - m.sqrt())
    g, _i, _r = _run_mode("grade", mask=m, grade_gain=3.0, grade_gamma=1.0, grade_clamp=False, grade_invert=False)
    assert float(g.max()) == 3.0
    assert img.shape == (1, 1, 4, 3)                                  # the matte as a grey image


def test_grow_and_shrink_each_side():
    m = torch.zeros(1, 12, 12)
    m[0, 4:8, 4:8] = 1.0
    cols = lambda r: r[0, 5].nonzero().flatten().tolist()
    rows = lambda r: r[0, :, 5].nonzero().flatten().tolist()
    assert cols(_run_mode("grow / shrink", mask=m, grow_left=2)[0]) == [2, 3, 4, 5, 6, 7]
    assert cols(_run_mode("grow / shrink", mask=m, grow_right=3)[0]) == [4, 5, 6, 7, 8, 9, 10]
    assert cols(_run_mode("grow / shrink", mask=m, grow_left=-1, grow_right=-1)[0]) == [5, 6]
    assert rows(_run_mode("grow / shrink", mask=m, grow_up=1, grow_down=-2)[0]) == [3, 4, 5]
    edge = torch.zeros(1, 6, 6)
    edge[0, :, :3] = 1.0                                                # runs off the left of the frame
    assert _run_mode("grow / shrink", mask=edge, grow_left=-2)[0][0, 2].tolist() == [1, 1, 1, 0, 0, 0]
    soft, _i, _r = _run_mode("grow / shrink", mask=m, grow_feather=2)
    assert 0.0 < float(soft[0, 3, 5]) < 1.0                              # feathered edge


def test_difference_key_marks_what_changed():
    plate = torch.full((1, 16, 16, 3), 0.4)
    shot = plate.clone()
    shot[0, 4:10, 4:10] = torch.tensor([0.9, 0.1, 0.1])
    m, _i, _r = _run_mode("key: difference", image=shot, plate=plate, difference_key_tolerance=0.05,
                          difference_key_softness=0.05)
    assert float(m[0, 6, 6]) == 1.0 and float(m[0, 0, 0]) == 0.0
    small = torch.full((1, 8, 8, 3), 0.4)                                # a plate of another size is resized
    m2, _i, rep = _run_mode("key: difference", image=shot, plate=small)
    assert m2.shape == (1, 16, 16) and "plate resized" in rep
    mi, _i, _r = _run_mode("key: difference", image=shot, plate=plate, difference_key_invert=True)
    assert torch.allclose(mi, 1.0 - m)


def test_each_mode_says_what_it_needs():
    for mode, what in (("key: colour", "an image"), ("grade", "a mask"), ("combine", "a mask"),
                       ("key: difference", "an image")):
        with pytest.raises(ValueError, match=what):
            _run_mode(mode)
    with pytest.raises(ValueError, match="mask_b"):
        _run_mode("combine", mask=_MASK)
    with pytest.raises(ValueError, match="clean plate"):
        _run_mode("key: difference", image=_IMG)


# ── P07: Mask Temporal -> Mask Track, mode "stabilize" ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("temporal_mode", ["none", "gaussian"])
def test_mask_temporal_through_mask_track(temporal_mode):
    from nodes.mask_matting.temporal_node import MaskTemporalMEC
    from nodes.mask_tracker_mec import MaskTrackerMEC

    g = torch.Generator().manual_seed(21)
    video = torch.rand(6, 20, 28, 3, generator=g)
    mask = (torch.rand(6, 20, 28, generator=g) > 0.4).float()
    old = {"image": video, "mask": mask, "temporal_mode": temporal_mode, "blend": 0.6, "sigma": 1.5, "device": "cpu",
           "drop_threshold": 0.35, "jump_threshold": 0.2}
    expected = MaskTemporalMEC().run(**old)
    spec = MaskTrackerMEC.INPUT_TYPES()
    defaults = {k: (v[1].get("default") if len(v) > 1 and "default" in v[1] else
                    (v[0][0] if isinstance(v[0], (list, tuple)) else None))
                for k, v in {**spec["required"], **spec["optional"]}.items() if k not in ("mask", "video", "sam_model")}
    check("MaskTemporalMEC", old, expected, lambda **k: MaskTrackerMEC().execute(**{**defaults, **k}))


def test_mask_track_other_modes_gain_an_empty_warning():
    from nodes.mask_tracker_mec import MaskTrackerMEC

    spec = MaskTrackerMEC.INPUT_TYPES()
    kw = {k: (v[1].get("default") if len(v) > 1 and "default" in v[1] else
              (v[0][0] if isinstance(v[0], (list, tuple)) else None))
          for k, v in {**spec["required"], **spec["optional"]}.items() if k not in ("mask", "video", "sam_model")}
    g = torch.Generator().manual_seed(22)
    out = MaskTrackerMEC().execute(**{**kw, "mode": "consistency_check", "video": torch.rand(4, 16, 16, 3, generator=g),
                                      "mask": torch.rand(4, 16, 16, generator=g)})
    assert len(out) == len(MaskTrackerMEC.RETURN_TYPES) == 6 and out[5] == ""
    with pytest.raises(ValueError, match="connect the video"):
        MaskTrackerMEC().execute(**{**kw, "mode": "stabilize"})


def test_the_tracker_front_end_shows_every_stabilize_control():
    import re

    from nodes.mask_tracker_mec import MaskTrackerMEC

    src = (Path(__file__).resolve().parents[1] / "js" / "motion_mask_tracker.js").read_text(encoding="utf-8")
    group = re.findall(r'"([a-z_]+)"', src.split("const STABILIZE = [", 1)[1].split("];", 1)[0])
    backend = [n for n in MaskTrackerMEC.INPUT_TYPES()["optional"] if n.startswith("stabilize_")]
    assert group == backend
