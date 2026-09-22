"""Tests for MEC Layer Effects nodes."""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest
import torch

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

for mod_name in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

from nodes.layer_effects._blend import BLEND_MODE_NAMES, BLEND_EPS, blend_rgba
from nodes.layer_effects._ops import align_batch, blur_mask, expand_mask
from nodes.layer_effects.nodes import (
    NODE_CLASS_MAPPINGS,
    LayerEffectDropShadowMEC,
    LayerEffectGradientMapMEC,
    LayerEffectGradientOverlayMEC,
    LayerEffectInnerGlowMEC,
    LayerEffectOuterGlowMEC,
    LayerEffectStrokeMEC,
    LayerEffectColorOverlayMEC,
    LayerEffectInnerShadowMEC,
)

ALL_NODE_IDS = (
    "LayerEffectDropShadowMEC",
    "LayerEffectInnerShadowMEC",
    "LayerEffectOuterGlowMEC",
    "LayerEffectInnerGlowMEC",
    "LayerEffectStrokeMEC",
    "LayerEffectColorOverlayMEC",
    "LayerEffectGradientOverlayMEC",
    "LayerEffectGradientMapMEC",
)

def _mask_square(batch: int = 1, size: int = 64) -> torch.Tensor:
    m = torch.zeros(batch, size, size)
    m[:, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
    return m


def _layer(batch: int = 1, size: int = 64, *, device=None, dtype=torch.float32) -> torch.Tensor:
    img = torch.zeros(batch, size, size, 3, device=device, dtype=dtype)
    img[:, :, :, 0] = 1.0
    return img


def _bg(batch: int = 1, size: int = 64, *, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.ones(batch, size, size, 3, device=device, dtype=dtype) * 0.5


# ── numpy reference (ported from upstream blendmodes.py formulas) ─────

def _ref_blend_rgb(a, b, mode):
    import numpy as np
    eps = BLEND_EPS
    if mode == "normal":
        return b
    if mode == "multiply":
        return a * b
    if mode == "screen":
        return 1.0 - (1.0 - a) * (1.0 - b)
    if mode == "overlay":
        return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "hard light":
        return np.where(b < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "difference":
        return np.abs(a - b)
    if mode == "darken":
        return np.minimum(a, b)
    if mode == "lighten":
        return np.maximum(a, b)
    if mode == "linear dodge(add)":
        return a + b
    if mode == "dodge" or mode == "color dodge":
        return np.clip(a / np.maximum(1.0 - b, eps), 0, 1)
    if mode == "color burn":
        return np.clip(1.0 - (1.0 - a) / np.maximum(b, eps), 0, 1)
    if mode == "linear burn":
        return np.clip(a + b - 1.0, 0, 1)
    if mode == "linear light":
        # NOT clipped here. blendmodes.py:236 computes
        # `blend = backdrop + 2*source - 1` and clamps only AFTER the opacity
        # composite (line 261), exactly like linear burn / subtract / exclusion
        # / pin light / grain, which this reference already leaves unclamped.
        # Clipping here made the reference disagree with both the upstream and
        # the port by up to 0.23.
        return a + 2 * b - 1.0
    if mode == "exclusion":
        return a + b - 2 * a * b
    if mode == "subtract":
        return a - b
    if mode == "divide":
        return np.clip(a / np.maximum(b, eps), 0, 1)
    if mode == "grain extract":
        return a - b + 0.5
    if mode == "grain merge":
        return a + b - 0.5
    if mode == "vivid light":
        return np.where(
            b <= 0.5,
            np.clip(a / np.maximum(1 - 2 * b, eps), 0, 1),
            np.clip(1 - (1 - a) / np.maximum(2 * b - 0.5, eps), 0, 1),
        )
    if mode == "pin light":
        return np.where(
            b <= 0.5,
            np.minimum(a, 2 * b),
            np.maximum(a, 2 * (b - 0.5)),
        )
    if mode == "hard mix":
        return np.round(np.clip(a + 2 * b - 1.0, 0, 1))
    if mode == "soft light":
        return np.where(
            b <= 0.5,
            a - (1.0 - 2.0 * b) * a * (1.0 - a),
            a + (2.0 * b - 1.0) * (np.sqrt(np.clip(a, 0.0, 1.0)) - a),
        )
    raise KeyError(mode)


def _ref_apply(backdrop, source, mode, opacity):
    import numpy as np
    bg = backdrop[..., :3]
    src = source[..., :3]
    sa = source[..., 3:4]
    if mode in ("darker color", "lighter color", "hue", "saturation", "color", "luminosity", "dissolve"):
        return backdrop
    blend = _ref_blend_rgb(bg, src, mode)
    w = sa * opacity
    rgb = (1 - w) * bg + w * blend
    rgb = np.clip(rgb, 0, 1)
    alpha = np.maximum(backdrop[..., 3:4], source[..., 3:4])
    return np.concatenate([rgb, alpha], axis=-1)


class TestAlignBatch:
    def test_align_batch_preserves_none_position(self):
        a = torch.zeros(2, 4, 4)
        c = torch.ones(3, 4, 4)
        aligned, batch = align_batch([a, None, c])
        assert len(aligned) == 3
        assert aligned[1] is None
        assert batch == 3
        assert aligned[0] is not None and aligned[0].shape[0] == 3
        assert aligned[2] is not None and aligned[2].shape[0] == 3


class TestRegistration:
    def test_registration_all_eight_ids(self):
        for nid in ALL_NODE_IDS:
            assert nid in NODE_CLASS_MAPPINGS


class TestDropShadowDefects:
    @pytest.fixture
    def node(self):
        return LayerEffectDropShadowMEC()

    def test_drop_shadow_zero_offset_no_crash(self, node):
        layer = _layer(1)
        mask = _mask_square(1)
        bg = _bg(1)
        out, em, rep = node.execute(
            layer, True, "multiply", 50, 0, 0, 2, 4, "#000000", 0, bg, mask,
        )
        assert out.shape == (1, 64, 64, 3)
        assert em.shape == (1, 64, 64)

    def test_mask_mismatch_resizes_not_drops(self, node):
        layer = _layer(1, 64)
        mask = _mask_square(1, 32)
        bg = _bg(1, 64)
        out, em, rep = node.execute(
            layer, False, "normal", 50, 0, 0, 0, 0, "#000000", 0, bg, mask,
        )
        assert out.shape[1:3] == (64, 64)
        assert "mask resized" in rep

    def test_missing_mask_raises_value_error(self, node):
        layer = _layer(1)
        bg = _bg(1)
        with pytest.raises(ValueError, match="needs a layer mask"):
            node.execute(layer, True, "multiply", 50, 0, 0, 0, 0, "#000000", 0, bg, None)

    def test_output_rgb_plus_mask_not_rgba_image(self, node):
        layer = _layer(1)
        mask = _mask_square(1)
        bg = _bg(1)
        out, em, rep = node.execute(
            layer, True, "screen", 80, 10, 10, 4, 6, "#000000", 0, bg, mask,
        )
        assert out.shape[-1] == 3
        assert em.ndim == 3


class TestBatch:
    NODES = (
        LayerEffectDropShadowMEC(),
        LayerEffectInnerShadowMEC(),
        LayerEffectOuterGlowMEC(),
        LayerEffectInnerGlowMEC(),
        LayerEffectStrokeMEC(),
        LayerEffectColorOverlayMEC(),
        LayerEffectGradientOverlayMEC(),
    )

    def test_batch_four_in_four_out_all_nodes(self):
        layer = _layer(4)
        mask = _mask_square(4)
        bg = _bg(4)
        for node in self.NODES:
            if isinstance(node, LayerEffectGradientOverlayMEC):
                out, em, rep = node.execute(
                    layer, True, "overlay", 100, "#FF0000", 255, "#0000FF", 255, 45, 0, bg, mask,
                )
            elif isinstance(node, (LayerEffectOuterGlowMEC, LayerEffectInnerGlowMEC)):
                out, em, rep = node.execute(
                    layer, True, "screen", 100, 3, 20, 8, "#FFFF00", "#FF0000", 0, bg, mask,
                )
            elif isinstance(node, LayerEffectStrokeMEC):
                out, em, rep = node.execute(
                    layer, True, "normal", 100, 0, 4, 0, "#FF0000", 0, bg, mask,
                )
            elif isinstance(node, LayerEffectColorOverlayMEC):
                out, em, rep = node.execute(
                    layer, True, "multiply", 50, "#00FF00", 0, bg, mask,
                )
            else:
                out, em, rep = node.execute(
                    layer, True, "multiply", 50, 5, 5, 2, 4, "#000000", 0, bg, mask,
                )
            assert out.shape[0] == 4, type(node).__name__
            assert em.shape[0] == 4

        gm = LayerEffectGradientMapMEC()
        img = _layer(4)
        out, em, rep = gm.execute(img, "#000", "#888", "#FFF", 0.5, 100, _mask_square(4))
        assert out.shape[0] == 4

    def test_batch_hold_last_frame(self):
        node = LayerEffectDropShadowMEC()
        layer = _layer(4)
        mask = _mask_square(4)
        bg = _bg(1)
        out, em, rep = node.execute(
            layer, True, "multiply", 50, 0, 0, 2, 2, "#000000", 0, bg, mask,
        )
        assert out.shape[0] == 4


class TestBlurBounds:
    def test_large_blur_clamped_fast_and_finite(self):
        mask = _mask_square(1, 64)
        notes: list[str] = []
        t0 = time.perf_counter()
        out = blur_mask(mask, 2048, notes)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"blur took {elapsed:.2f}s"
        assert torch.isfinite(out).all()
        assert any("blur 2048 exceeded" in n for n in notes)

    def test_grow_clamped_reported(self):
        mask = _mask_square(1, 64)
        notes: list[str] = []
        expand_mask(mask, 9999, 0, notes, grow_label="grow")
        assert any("grow 9999 exceeded" in n for n in notes)


class TestBlendModes:
    @pytest.mark.parametrize("mode", [m for m in BLEND_MODE_NAMES if m not in (
        "dissolve", "darker color", "lighter color", "hue", "saturation", "color", "luminosity",
    )])
    def test_blend_modes_match_numpy_reference(self, mode):
        torch.manual_seed(0)
        b, h, w = 1, 32, 32
        backdrop = torch.rand(b, h, w, 4)
        source = torch.rand(b, h, w, 4)
        out = blend_rgba(backdrop, source, mode, 0.75)
        ref = _ref_apply(backdrop.numpy(), source.numpy(), mode, 0.75)
        diff = (out.numpy() - ref).max()
        assert diff < 1e-4, f"{mode} max diff {diff}"

    @pytest.mark.parametrize("mode", BLEND_MODE_NAMES)
    def test_all_modes_finite_on_extreme_inputs(self, mode):
        b, h, w = 1, 8, 8
        backdrop = torch.zeros(b, h, w, 4)
        backdrop[..., 3] = 1.0
        source = torch.ones(b, h, w, 4)
        out = blend_rgba(backdrop, source, mode, 1.0, dissolve_seed=42)
        assert torch.isfinite(out).all(), mode

    def test_dissolve_seed_deterministic(self):
        b, h, w = 1, 32, 32
        bd = torch.ones(b, h, w, 4) * 0.5
        src = torch.ones(b, h, w, 4)
        src[..., 3] = 0.8
        a = blend_rgba(bd.clone(), src.clone(), "dissolve", 0.5, dissolve_seed=7)
        b_out = blend_rgba(bd.clone(), src.clone(), "dissolve", 0.5, dissolve_seed=7)
        c = blend_rgba(bd.clone(), src.clone(), "dissolve", 0.5, dissolve_seed=8)
        assert torch.equal(a, b_out)
        assert not torch.equal(a, c)


class TestPILSpotCheck:
    def test_pil_multiply_screen_difference(self):
        pytest.importorskip("PIL")
        from PIL import ImageChops, Image
        import numpy as np
        a = Image.new("RGB", (4, 4), (128, 64, 32))
        b = Image.new("RGB", (4, 4), (64, 128, 200))
        for chop, mode in ((ImageChops.multiply, "multiply"), (ImageChops.screen, "screen"), (ImageChops.difference, "difference")):
            pil = np.asarray(chop(a, b)).astype(np.float32) / 255.0
            t_a = torch.tensor([[[[128 / 255, 64 / 255, 32 / 255]]]]).expand(1, 4, 4, 3)
            t_b = torch.tensor([[[[64 / 255, 128 / 255, 200 / 255]]]]).expand(1, 4, 4, 3)
            bd = torch.cat([t_a, torch.ones(1, 4, 4, 1)], dim=-1)
            src = torch.cat([t_b, torch.ones(1, 4, 4, 1)], dim=-1)
            out = blend_rgba(bd, src, mode, 1.0)[0, 0, 0, :3].numpy()
            np.testing.assert_allclose(out, pil[0, 0], atol=1 / 255)

    def test_pil_overlay_if_available(self):
        pytest.importorskip("PIL")
        from PIL import ImageChops, Image
        if not hasattr(ImageChops, "overlay"):
            pytest.skip("ImageChops.overlay not available on this Pillow")
        import numpy as np
        a = Image.new("RGB", (4, 4), (100, 150, 200))
        b = Image.new("RGB", (4, 4), (50, 100, 250))
        pil = np.asarray(ImageChops.overlay(a, b)).astype(np.float32) / 255.0
        t_a = torch.full((1, 4, 4, 3), 0.0)
        t_a[..., 0] = 100 / 255
        t_a[..., 1] = 150 / 255
        t_a[..., 2] = 200 / 255
        t_b = torch.zeros(1, 4, 4, 3)
        t_b[..., 0] = 50 / 255
        t_b[..., 1] = 100 / 255
        t_b[..., 2] = 250 / 255
        bd = torch.cat([t_a, torch.ones(1, 4, 4, 1)], dim=-1)
        src = torch.cat([t_b, torch.ones(1, 4, 4, 1)], dim=-1)
        out = blend_rgba(bd, src, "overlay", 1.0)[0, 0, 0, :3].numpy()
        np.testing.assert_allclose(out, pil[0, 0], atol=1 / 255)


class TestDeviceDtype:
    def test_output_preserves_dtype_and_device(self):
        node = LayerEffectDropShadowMEC()
        layer = _layer(1).to(dtype=torch.float32)
        mask = _mask_square(1)
        bg = _bg(1)
        out, em, rep = node.execute(
            layer, True, "multiply", 50, 2, 2, 1, 2, "#000000", 0, bg, mask,
        )
        assert out.dtype == layer.dtype
        assert out.device == layer.device
        assert em.dtype == layer.dtype
        assert em.device == layer.device


class TestEffectMaskSemantics:
    def test_stroke_ring_is_hollow_and_the_right_width(self):
        # The stroke is CENTRED on the mask edge, Photoshop's "Center" position:
        # inner = stroke_grow - width/2, outer = inner + width. On a 64px frame
        # whose mask square starts at row 16, a width-8 stroke occupies rows
        # 12..19 and NOTHING else.
        #
        # The first version of this test sampled row 4 and demanded coverage
        # there. Row 4 is twelve pixels clear of the edge - three times the
        # stroke's outer reach - so it asserted the stroke was four times wider
        # than it is. Sampling inside the band, outside the band and at the
        # centre pins the width instead of just "something is non-zero".
        node = LayerEffectStrokeMEC()
        out, em, rep = node.execute(
            _layer(1), False, "normal", 100, 0, 8, 0, "#FF0000", 0, _bg(1),
            _mask_square(1, 64),
        )
        assert em[0, 14, 32].item() > 0.5, "the band itself is empty"
        assert em[0, 32, 32].item() < 0.1, "the ring is filled in, not hollow"
        assert em[0, 2, 32].item() < 0.1, "the stroke bleeds past its width"

    def test_stroke_width_controls_the_band(self):
        # INVARIANT: the counter-test. A stroke that ignores stroke_width would
        # still pass the shape test above.
        node = LayerEffectStrokeMEC()
        thin_em = node.execute(
            _layer(1), False, "normal", 100, 0, 4, 0, "#FF0000", 0, _bg(1),
            _mask_square(1, 64),
        )[1]
        wide_em = node.execute(
            _layer(1), False, "normal", 100, 0, 16, 0, "#FF0000", 0, _bg(1),
            _mask_square(1, 64),
        )[1]
        assert wide_em.sum() > thin_em.sum() * 1.5

    def test_outer_glow_sits_outside_and_inner_glow_inside(self):
        # The first version asked only that the two masks DIFFER, with
        # glow_range 24 and blur 10 on a 64px frame - which expands the glow
        # 34px from a mask edge 16px from the border, so both masks saturated to
        # all-ones and were identical. It was testing nothing, and it said so by
        # failing.
        #
        # What actually distinguishes them is WHERE the glow lives: outer glow
        # outside the subject, inner glow inside it. Pin that, at a range the
        # frame can hold.
        layer, bg = _layer(1), _bg(1)
        mask = _mask_square(1, 64)          # the square occupies rows 16..47
        em_o = LayerEffectOuterGlowMEC().execute(
            layer, False, "screen", 100, 4, 6, 2, "#FFFF00", "#FF0000", 0, bg, mask,
        )[1]
        em_i = LayerEffectInnerGlowMEC().execute(
            layer, False, "screen", 100, 4, 6, 2, "#FFFF00", "#FF0000", 0, bg, mask,
        )[1]
        assert not torch.allclose(em_o, em_i)
        # just outside the top edge
        assert em_o[0, 13, 32].item() > em_i[0, 13, 32].item()
        # just inside it
        assert em_i[0, 19, 32].item() > em_o[0, 19, 32].item()
        # neither may swallow the whole frame at this range
        assert em_o.min().item() < 0.5 and em_i.min().item() < 0.5

    def test_glow_range_controls_the_spread(self):
        layer, bg, mask = _layer(1), _bg(1), _mask_square(1, 64)
        tight = LayerEffectOuterGlowMEC().execute(
            layer, False, "screen", 100, 4, 3, 0, "#FFFF00", "#FF0000", 0, bg, mask,
        )[1]
        loose = LayerEffectOuterGlowMEC().execute(
            layer, False, "screen", 100, 4, 12, 0, "#FFFF00", "#FF0000", 0, bg, mask,
        )[1]
        assert loose.sum() > tight.sum() * 1.2

    def test_color_overlay_effect_mask_is_layer_mask(self):
        node = LayerEffectColorOverlayMEC()
        layer = _layer(1)
        mask = _mask_square(1)
        bg = _bg(1)
        _, em, rep = node.execute(layer, False, "multiply", 100, "#FF0000", 0, bg, mask)
        assert torch.allclose(em, mask)


class TestEffectMaskPaintedFootprint:
    """effect_mask must be ~0 wherever the output still equals the background."""

    FLAT_BG = 0.5

    @staticmethod
    def _flat_scene():
        bg = torch.full((1, 64, 64, 3), TestEffectMaskPaintedFootprint.FLAT_BG)
        layer = torch.zeros(1, 64, 64, 3)
        layer[..., 0] = 1.0
        mask = _mask_square(1, 64)
        return layer, bg, mask

    @staticmethod
    def _assert_mask_zero_on_untouched(out, bg, em):
        untouched = (out - bg).abs().amax(dim=-1) < 1e-4
        if untouched.any():
            assert em[untouched].max().item() < 0.05

    def test_effect_mask_matches_what_was_actually_painted(self):
        layer, bg, mask = self._flat_scene()

        cases = (
            (
                LayerEffectDropShadowMEC(),
                lambda n: n.execute(
                    layer, False, "multiply", 100, 8, 8, 4, 4, "#000000", 0, bg, mask,
                ),
            ),
            (
                LayerEffectInnerShadowMEC(),
                lambda n: n.execute(
                    layer, False, "multiply", 100, 4, 4, 2, 3, "#000000", 0, bg, mask,
                ),
            ),
            (
                LayerEffectOuterGlowMEC(),
                lambda n: n.execute(
                    layer, False, "screen", 100, 4, 6, 2, "#FFFF00", "#FF0000", 0, bg, mask,
                ),
            ),
            (
                LayerEffectInnerGlowMEC(),
                lambda n: n.execute(
                    layer, False, "screen", 100, 4, 6, 2, "#FFFF00", "#FF0000", 0, bg, mask,
                ),
            ),
            (
                LayerEffectStrokeMEC(),
                lambda n: n.execute(
                    layer, False, "normal", 100, 0, 6, 0, "#FF0000", 0, bg, mask,
                ),
            ),
        )
        for node, run in cases:
            out, em, _ = run(node)
            self._assert_mask_zero_on_untouched(out, bg, em)


class TestGradientMap:
    def test_gradient_map_luminance_mapping(self):
        node = LayerEffectGradientMapMEC()
        img = torch.ones(1, 8, 8, 3) * 0.5
        out, em, rep = node.execute(img, "#000000", "#808080", "#FFFFFF", 0.5, 100)
        assert "ramp" in rep
        assert out.shape == (1, 8, 8, 3)
        assert torch.isfinite(out).all()

    def test_gradient_overlay_report_has_ramp(self):
        node = LayerEffectGradientOverlayMEC()
        layer = _layer(1)
        mask = _mask_square(1)
        bg = _bg(1)
        _, _, rep = node.execute(
            layer, False, "overlay", 100, "#FF0000", 255, "#0000FF", 255, 90, 0, bg, mask,
        )
        assert "ramp" in rep
        assert "angle 90" in rep


class TestSignConvention:
    def test_positive_distance_moves_shadow_right(self):
        node = LayerEffectDropShadowMEC()
        layer = _layer(1, 64)
        mask = torch.zeros(1, 64, 64)
        mask[:, 20:30, 10:20] = 1.0
        bg = _bg(1, 64)
        _, em_left, _ = node.execute(
            layer, False, "multiply", 100, -10, 0, 0, 0, "#000000", 0, bg, mask,
        )
        _, em_right, _ = node.execute(
            layer, False, "multiply", 100, 10, 0, 0, 0, "#000000", 0, bg, mask,
        )
        cx = lambda t: (t[0].sum(dim=0).argmax().item())
        assert cx(em_right) > cx(em_left)


class TestReportAndCoverage:
    def test_effect_mask_nonzero_coverage(self):
        node = LayerEffectDropShadowMEC()
        layer = _layer(1)
        mask = _mask_square(1)
        bg = _bg(1)
        _, em, _ = node.execute(
            layer, False, "multiply", 100, 0, 0, 4, 4, "#000000", 0, bg, mask,
        )
        assert em.max() > 0.0

    def test_report_mentions_batch_and_adjustments(self):
        node = LayerEffectDropShadowMEC()
        layer = _layer(4)
        mask = _mask_square(4, 32)
        bg = _bg(4, 64)
        _, _, rep = node.execute(
            layer, False, "multiply", 50, 0, 0, 0, 0, "#000000", 0, bg, mask,
        )
        assert "4 frame" in rep
        assert "mask resized" in rep


class TestLivePackRegistration:
    def test_ids_merged_in_pack_init(self):
        init_src = (PACK_ROOT / "__init__.py").read_text(encoding="utf-8")
        assert "_LAYERFX_MAPPINGS" in init_src
        for nid in ALL_NODE_IDS:
            assert nid in NODE_CLASS_MAPPINGS

# ── the front-end ───────────────────────────────────────────────────────────
#
# js/c2c_layer_effects.js hardcodes widget names: distance_x/distance_y for the
# direction dial, and start_color/mid_color/end_color/start_alpha/end_alpha/
# angle/mid_point for the gradient ramps. Rename one of those in nodes.py and
# the control silently reads `undefined` and draws a black ramp or a dial stuck
# at the centre. Nothing in Python notices, and the node still "works".

import json as _json
import re as _re
import shutil as _shutil
import subprocess as _subprocess

_PACK = Path(__file__).resolve().parents[1]
_FX_JS = _PACK / "js" / "c2c_layer_effects.js"
_NODE_BIN = _shutil.which("node")


class TestFrontEndMatchesTheBackend:
    def test_the_widget_file_exists_and_is_served(self):
        # WEB_DIRECTORY is ./js and ComfyUI loads every .js in it, so being in
        # that directory IS the wiring - there is no import list to forget.
        assert _FX_JS.is_file()
        init = (_PACK / "__init__.py").read_text(encoding="utf-8")
        assert 'WEB_DIRECTORY = "./js"' in init

    @pytest.mark.skipif(_NODE_BIN is None, reason="node is not installed")
    def test_the_widget_file_parses(self):
        # A syntax error here does not break this node alone - ComfyUI loads
        # every file in js/, and a broken module is a broken extension.
        r = _subprocess.run([_NODE_BIN, "--check", str(_FX_JS)],
                            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr

    def test_every_node_the_front_end_targets_actually_exists(self):
        from nodes.layer_effects import NODE_CLASS_MAPPINGS

        src = _FX_JS.read_text(encoding="utf-8")
        block = src.split("const NODES = new Set([", 1)[1].split("]);", 1)[0]
        targeted = set(_re.findall(r'"([^"]+)"', block))
        assert targeted == set(NODE_CLASS_MAPPINGS), (
            "front-end targets and registered ids disagree: "
            f"only in JS {sorted(targeted - set(NODE_CLASS_MAPPINGS))}, "
            f"only in Python {sorted(set(NODE_CLASS_MAPPINGS) - targeted)}"
        )

    def test_the_dial_writes_widgets_that_exist(self):
        from nodes.layer_effects import NODE_CLASS_MAPPINGS

        src = _FX_JS.read_text(encoding="utf-8")
        block = src.split("const DIAL_NODES = new Set([", 1)[1].split("]);", 1)[0]
        for node_id in _re.findall(r'"([^"]+)"', block):
            spec = NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()
            names = set(spec.get("required", {})) | set(spec.get("optional", {}))
            for needed in ("distance_x", "distance_y"):
                assert needed in names, (
                    f"the dial writes {needed} on {node_id}, which has no such "
                    "widget - the control would silently do nothing"
                )

    def test_the_ramp_reads_widgets_that_exist(self):
        from nodes.layer_effects import NODE_CLASS_MAPPINGS

        src = _FX_JS.read_text(encoding="utf-8")
        block = src.split("const RAMP_NODES = {", 1)[1].split("\n};", 1)[0]
        # one entry per node: NodeId: { stops: [...], alphas: [...], angle: "x" }
        for node_id, body in _re.findall(r"(\w+):\s*\{(.*?)\n  \}", block, _re.S):
            spec = NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()
            names = set(spec.get("required", {})) | set(spec.get("optional", {}))
            for needed in _re.findall(r'"([^"]+)"', body):
                assert needed in names, (
                    f"the ramp reads {needed} on {node_id}, which has no such "
                    "widget - the ramp would draw from undefined"
                )

    def test_every_colour_widget_is_a_string_the_picker_can_drive(self):
        # The swatch writes a "#rrggbb" STRING back into the widget. A colour
        # widget that is not a STRING would take the value and then fail
        # server-side on the next run.
        from nodes.layer_effects import NODE_CLASS_MAPPINGS

        for node_id, cls in NODE_CLASS_MAPPINGS.items():
            spec = cls.INPUT_TYPES()
            for section in ("required", "optional"):
                for name, entry in spec.get(section, {}).items():
                    if name != "color" and not name.endswith("_color"):
                        continue
                    assert entry[0] == "STRING", f"{node_id}.{name} is {entry[0]}"
                    default = entry[1].get("default", "") if len(entry) > 1 else ""
                    assert _re.fullmatch(r"#[0-9a-fA-F]{6}", str(default)), (
                        f"{node_id}.{name} default {default!r} is not #rrggbb, so "
                        "the swatch opens on the wrong colour"
                    )

    def test_the_hex_pattern_in_the_js_accepts_the_python_defaults(self):
        # Belt and braces: the JS validates with its own regex before writing.
        # If the two disagree, a legitimate default shows as an error outline.
        # The pattern lives in the SHARED control module, because the mask
        # toolkit's colour row is the same control.
        from nodes.layer_effects import NODE_CLASS_MAPPINGS

        shared = _PACK / "js" / "_c2c_fx_controls.js"
        assert "const HEX_RE = /^#?[0-9a-fA-F]{6}$/;" in shared.read_text(
            encoding="utf-8")
        for cls in NODE_CLASS_MAPPINGS.values():
            spec = cls.INPUT_TYPES()
            for name, entry in spec.get("required", {}).items():
                if name == "color" or name.endswith("_color"):
                    assert _re.fullmatch(
                        r"#?[0-9a-fA-F]{6}", str(entry[1].get("default", ""))
                    )
        _json.dumps({})          # keep the import honest

    def test_the_shared_control_module_resolves_for_both_families(self):
        # INVARIANT: a relative import of a file that is not there 404s, and the
        # browser discards the whole module graph below it - so one bad path
        # takes out BOTH front-ends, not just the one that named it. Everything
        # in js/ is auto-loaded by WEB_DIRECTORY, so there is no import list to
        # catch this; only the path itself.
        shared = _PACK / "js" / "_c2c_fx_controls.js"
        assert shared.is_file()
        for consumer in ("c2c_layer_effects.js", "c2c_mask_toolkit.js"):
            src = (_PACK / "js" / consumer).read_text(encoding="utf-8")
            assert './_c2c_fx_controls.js"' in src, f"{consumer} lost the import"

    def test_the_shared_module_registers_nothing_of_its_own(self):
        # INVARIANT: WEB_DIRECTORY loads every .js in js/, this one included. A
        # helper that also called registerExtension would run twice-over and
        # attach controls to nodes it knows nothing about.
        shared = (_PACK / "js" / "_c2c_fx_controls.js").read_text(encoding="utf-8")
        assert "registerExtension" not in shared
        assert "scripts/app.js" not in shared
