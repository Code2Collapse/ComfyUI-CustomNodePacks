"""L7.65 consolidation wave 3: merged nodes give the same results through their successor.

Same method as test_consolidation_wave2.py: the old node's inputs go through its row in nodes/_legacy_replacements.json
(as the workflow migration moves them) into the successor, and every mapped output is compared with the old node's.
P13: the eight Layer Effect nodes are kept as the engines of Layer Effects (C2C), so the reference is the engine called
the way the old node was.
"""
from __future__ import annotations

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
