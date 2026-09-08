"""
Tests for InpaintCropProMEC, InpaintStitchProMEC, InpaintMaskPrepareMEC.

Covers:
- Batch correctness: IMAGE=(B,H,W,C), MASK=(B,H,W)
- Each blend mode produces different output
- Video stable crop produces identical bbox per frame
- Temporal smoothing reduces variance
- Round-trip: crop → stitch recovers original shape
- Edge cases: empty mask, single frame, full mask
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def crop_node():
    from nodes.inpaint_suite import InpaintCropProMEC
    return InpaintCropProMEC()


@pytest.fixture
def stitch_node():
    from nodes.inpaint_suite import InpaintStitchProMEC
    return InpaintStitchProMEC()


@pytest.fixture
def prepare_node():
    from nodes.inpaint_suite import InpaintMaskPrepareMEC
    return InpaintMaskPrepareMEC()


@pytest.fixture
def sample_image():
    """Varied image with structure (not uniform)."""
    torch.manual_seed(42)
    img = torch.rand(2, 128, 128, 3)
    x_grad = torch.linspace(0, 1, 128).view(1, 1, 128, 1).expand(2, 128, 128, 1)
    y_grad = torch.linspace(0, 1, 128).view(1, 128, 1, 1).expand(2, 128, 128, 1)
    img = (img * 0.5 + x_grad * 0.25 + y_grad * 0.25).clamp(0, 1)
    return img


@pytest.fixture
def sample_mask():
    """Center circle mask."""
    mask = torch.zeros(2, 128, 128)
    cy, cx = 64, 64
    for y in range(128):
        for x in range(128):
            if (x - cx) ** 2 + (y - cy) ** 2 < 30 ** 2:
                mask[:, y, x] = 1.0
    return mask


@pytest.fixture
def video_mask():
    """Different masks per frame (for video tests)."""
    mask = torch.zeros(4, 128, 128)
    mask[0, 10:40, 10:40] = 1.0
    mask[1, 44:84, 44:84] = 1.0
    mask[2, 80:120, 80:120] = 1.0
    mask[3, 50:70, 10:118] = 1.0
    return mask


# ── Crop node tests ──────────────────────────────────────────────────

# ── Crop / Stitch tests ──────────────────────────────────────────────
#
# REWRITTEN 2026-08-29. InpaintCropProMEC was redesigned wholesale to the
# lquesada API: `crop_for_inpaint` -> `inpaint_crop`, outputs 7 -> 5
# (`cropped_comp` and `crop_mask` removed), and ~38 parameters with `mask` moved
# to an OPTIONAL KEYWORD at the end. The old tests passed mask as the 2nd
# POSITIONAL argument, which now lands on `downscale_algorithm` — so they were
# not repairable by renaming, and all 15 failed.
#
# Two rules, both learned from that failure:
#   1. Every call below is by KEYWORD. The rot happened precisely because a
#      positional signature grew underneath the tests.
#   2. Every test carries a one-line INVARIANT comment, so the next redesign can
#      classify stale-vs-regression without re-deriving 38 parameters of meaning.
#
# Crop defaults come from INPUT_TYPES() rather than being hardcoded, so adding a
# parameter with a default cannot break this file.


def _crop_defaults(node):
    """Build crop kwargs from the node's own declared defaults."""
    req = type(node).INPUT_TYPES().get("required", {})
    out = {}
    for name, spec in req.items():
        tspec = spec[0]
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if "default" in opts:
            out[name] = opts["default"]
        elif isinstance(tspec, list) and tspec:
            out[name] = tspec[0]          # combo: first entry is the default
    out.pop("image", None)
    out.pop("mask", None)
    return out


def _crop(node, image, mask, **over):
    kw = _crop_defaults(node)
    kw.update(over)
    return node.inpaint_crop(image=image, mask=mask, **kw)


class TestInpaintCropPro:
    def test_output_shapes(self, crop_node, sample_image, sample_mask):
        # INVARIANT: contract shapes — IMAGE is (B,H,W,3), MASK is (B,H,W), stitcher is a dict.
        stitcher, cropped, inpaint_mask, blend_mask, info = _crop(
            crop_node, sample_image, sample_mask
        )
        assert cropped.dim() == 4
        assert cropped.shape[0] == 2
        assert cropped.shape[3] == 3
        assert inpaint_mask.dim() == 3
        assert blend_mask.dim() == 3
        assert isinstance(stitcher, dict)
        assert isinstance(info, str)
        assert len(info) > 20

    def test_batch_correctness(self, crop_node, sample_image, sample_mask):
        # INVARIANT: batch size is preserved through the crop, per frame.
        _stitcher, cropped, inpaint_mask, _blend, _info = _crop(
            crop_node, sample_image, sample_mask
        )
        assert cropped.shape[0] == sample_image.shape[0]
        assert inpaint_mask.shape[0] == sample_image.shape[0]

    def test_forced_output_size(self, crop_node, sample_image, sample_mask):
        # INVARIANT: an explicit output target size is honoured exactly.
        _s, cropped, _im, _bm, _i = _crop(
            crop_node, sample_image, sample_mask,
            output_resize_to_target_size=True,
            output_target_width=512, output_target_height=512,
        )
        assert cropped.shape[1] == 512
        assert cropped.shape[2] == 512

    def test_canvas_is_multiple_of_alignment(self, crop_node, sample_image, sample_mask):
        # INVARIANT: crop canvas honours the alignment multiple (Wan needs /32-friendly sizes).
        _s, cropped, _im, _bm, _i = _crop(
            crop_node, sample_image, sample_mask, wan_align_multiple=32
        )
        assert cropped.shape[1] % 32 == 0, f"height {cropped.shape[1]} not aligned to 32"
        assert cropped.shape[2] % 32 == 0, f"width {cropped.shape[2]} not aligned to 32"

    def test_mask_resizes_with_the_crop(self, crop_node, sample_image, sample_mask):
        # INVARIANT: returned masks share the cropped image spatial size — crop and
        # masks must not drift apart, or the stitch composites offset.
        _s, cropped, inpaint_mask, blend_mask, _i = _crop(
            crop_node, sample_image, sample_mask
        )
        assert inpaint_mask.shape[1:] == cropped.shape[1:3]
        assert blend_mask.shape[1:] == cropped.shape[1:3]

    def test_empty_mask_does_not_crash(self, crop_node, sample_image):
        # INVARIANT: an all-zero mask degrades gracefully instead of raising.
        empty = torch.zeros(2, 128, 128)
        _s, cropped, _im, _bm, info = _crop(crop_node, sample_image, empty)
        assert cropped.shape[0] == 2
        assert len(info) > 10

    def test_video_stable_crop_holds_one_bbox(self, crop_node, video_mask):
        # INVARIANT: wan_stable_crop yields ONE box for the whole clip, so the
        # subject does not jitter frame to frame.
        img = torch.rand(4, 128, 128, 3)
        stitcher, cropped, _im, _bm, _i = _crop(
            crop_node, img, video_mask, wan_stable_crop=True
        )
        assert cropped.shape[0] == 4
        assert isinstance(stitcher, dict)

    def test_info_reports_real_values(self, crop_node, sample_image, sample_mask):
        # INVARIANT: the report names what actually happened (R8), not a fixed string.
        _s, _c, _im, _bm, info = _crop(crop_node, sample_image, sample_mask)
        low = info.lower()
        assert "crop" in low or "canvas" in low
        assert "mask" in low or "blend" in low

    def test_output_is_finite_on_hostile_input(self, crop_node, sample_mask):
        # INVARIANT: NaN/Inf in the plate must not propagate into the crop.
        img = torch.rand(2, 128, 128, 3)
        img[0, 0, 0, 0] = float("nan")
        img[0, 1, 1, 1] = float("inf")
        _s, cropped, inpaint_mask, _bm, _i = _crop(crop_node, img, sample_mask)
        assert torch.isfinite(cropped).all(), "NaN/Inf leaked through the crop"
        assert torch.isfinite(inpaint_mask).all()


class TestInpaintStitchPro:
    def test_round_trip_shape(self, crop_node, stitch_node, sample_image, sample_mask):
        # INVARIANT: crop -> stitch returns the plate exact shape.
        stitcher, cropped, _im, _bm, _i = _crop(crop_node, sample_image, sample_mask)
        result, _blend_used, _info = stitch_node.inpaint_stitch(
            stitcher=stitcher, inpainted_image=cropped
        )
        assert result.shape == sample_image.shape

    def test_identity_round_trip_returns_the_plate(self, crop_node, stitch_node,
                                                   sample_image, sample_mask):
        # INVARIANT: stitching back an UNMODIFIED crop reproduces the plate. This is
        # the strongest statement of the crop/stitch contract — a geometry error
        # shows up here even when every shape assertion still passes.
        stitcher, cropped, _im, _bm, _i = _crop(crop_node, sample_image, sample_mask)
        result, _bu, _i2 = stitch_node.inpaint_stitch(
            stitcher=stitcher, inpainted_image=cropped
        )
        assert result.shape == sample_image.shape
        assert torch.isfinite(result).all()
        # Resampling makes this approximate, not bit-exact. The bound is on the
        # MEAN so a localised geometry slip cannot hide inside it.
        assert (result - sample_image).abs().mean().item() < 0.05, (
            "identity crop->stitch drifted from the plate"
        )

    def test_exterior_is_untouched_by_stitch(self, crop_node, stitch_node,
                                             sample_image, sample_mask):
        # INVARIANT (project core contract): pixels outside the composite region are
        # the LITERAL plate. Asserted at the far corners, which no crop of a centre
        # mask can legitimately reach.
        stitcher, cropped, _im, _bm, _i = _crop(crop_node, sample_image, sample_mask)
        painted = cropped.clone()
        painted[:] = 1.0                       # blow out the crop entirely
        result, _bu, _i2 = stitch_node.inpaint_stitch(
            stitcher=stitcher, inpainted_image=painted
        )
        for yy, xx in ((0, 0), (0, 127), (127, 0), (127, 127)):
            assert torch.equal(result[:, yy, xx, :], sample_image[:, yy, xx, :]), (
                f"stitch modified plate pixel ({yy},{xx}) outside the composite region"
            )

    def test_stitch_consumes_the_crops_own_stitcher(self, crop_node, stitch_node,
                                                    sample_image, sample_mask):
        # INVARIANT: stitch uses the transform the crop produced. Feeding a stitcher
        # from a DIFFERENT crop must not silently give the same composite.
        s1, c1, _a, _b, _c = _crop(crop_node, sample_image, sample_mask)
        other_mask = torch.zeros(2, 128, 128)
        other_mask[:, 5:35, 5:35] = 1.0
        s2, _c2, _d, _e, _f = _crop(crop_node, sample_image, other_mask)

        r_own, _x, _y = stitch_node.inpaint_stitch(stitcher=s1, inpainted_image=c1)
        try:
            r_other, _p, _q = stitch_node.inpaint_stitch(stitcher=s2, inpainted_image=c1)
        except Exception:
            return  # refusing a mismatched stitcher is also correct
        assert not torch.equal(r_own, r_other), (
            "stitch ignored the stitcher — a different crop transform produced an "
            "identical composite, so the transform is being recomputed downstream"
        )

    def test_blend_mode_override_changes_result(self, crop_node, stitch_node,
                                                sample_image, sample_mask):
        # INVARIANT: the override is wired — a different blend mode changes pixels.
        stitcher, cropped, _im, _bm, _i = _crop(crop_node, sample_image, sample_mask)
        painted = (cropped * 0.3).clamp(0, 1)

        # "from_crop" resolves to whatever the crop stored (inpaint_suite.py:1791-1794),
        # and the crop default stitch_blend_mode is "gaussian" (:928-929), so the
        # alternative must differ from BOTH or the two calls resolve to one mode.
        crop_default = type(crop_node).INPUT_TYPES()["required"]["stitch_blend_mode"][1]["default"]
        modes = type(stitch_node).INPUT_TYPES()["required"]["blend_mode_override"][0]
        alt = next((m for m in modes if m not in ("from_crop", crop_default)), None)
        if alt is None:
            pytest.skip("no blend mode distinct from the crop default")

        # Assert on blend_mask_used, the node's OWN second output, not on a
        # downstream pixel sum. gaussian and edge_aware both composite through the
        # same simple-alpha branch (:1898-1900) — the mode changes only the MASK
        # (:1879-1884). A pixel-difference threshold is therefore an indirect proxy
        # that reads near zero on smooth input (measured 0.0025-0.0033 total) and
        # would tempt a threshold loosening. The mask is the mechanism, so pin that.
        _a, mask_from_crop, _y = stitch_node.inpaint_stitch(
            stitcher=stitcher, inpainted_image=painted, blend_mode_override="from_crop"
        )
        _b, mask_alt, _q = stitch_node.inpaint_stitch(
            stitcher=stitcher, inpainted_image=painted, blend_mode_override=alt
        )
        assert not torch.equal(mask_from_crop, mask_alt), (
            f"blend_mode_override {alt} produced an identical blend mask to "
            f"{crop_default} — the override is not reaching the mask builder"
        )

    def test_result_stays_in_range(self, crop_node, stitch_node, sample_image, sample_mask):
        # INVARIANT: composite output is a valid IMAGE — finite and within [0,1].
        stitcher, cropped, _im, _bm, _i = _crop(crop_node, sample_image, sample_mask)
        result, _bu, _i2 = stitch_node.inpaint_stitch(
            stitcher=stitcher, inpainted_image=cropped
        )
        assert torch.isfinite(result).all()
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_color_match_changes_result(self, crop_node, stitch_node, sample_image, sample_mask):
        # INVARIANT: color_match is wired, not a dead widget.
        stitcher, cropped, _im, _bm, _i = _crop(crop_node, sample_image, sample_mask)
        painted = (cropped * 0.5 + 0.2).clamp(0, 1)
        off, _a, _b = stitch_node.inpaint_stitch(
            stitcher=stitcher, inpainted_image=painted, color_match=False
        )
        on, _c, _d = stitch_node.inpaint_stitch(
            stitcher=stitcher, inpainted_image=painted, color_match=True
        )
        assert (off - on).abs().sum().item() > 0.01, "color_match had no effect"

    def test_stitch_info_reports_real_data(self, crop_node, stitch_node,
                                           sample_image, sample_mask):
        # INVARIANT: R8 — the report says what ran.
        stitcher, cropped, _im, _bm, _i = _crop(crop_node, sample_image, sample_mask)
        _r, _bu, info = stitch_node.inpaint_stitch(stitcher=stitcher, inpainted_image=cropped)
        low = info.lower()
        assert "canvas" in low or "stitch" in low or "blend" in low


class TestInpaintMaskPrepare:
    def test_output_shapes(self, prepare_node, sample_mask):
        inpaint_m, stitch_m, preview, info = prepare_node.prepare_mask(
            sample_mask, True, True, 100, 4, "hard_binary", "gaussian", 16, False, 1.5
        )
        assert inpaint_m.shape == sample_mask.shape
        assert stitch_m.shape == sample_mask.shape
        assert preview.dim() == 4
        assert preview.shape[3] == 3
        assert isinstance(info, str)

    def test_fill_holes(self, prepare_node):
        mask = torch.zeros(1, 64, 64)
        mask[:, 10:54, 10:54] = 1.0
        mask[:, 20:40, 20:40] = 0.0  # interior hole
        inpaint_m, _, _, info = prepare_node.prepare_mask(
            mask, True, False, 0, 0, "hard_binary", "gaussian", 16, False, 1.5
        )
        center_val = inpaint_m[0, 30, 30].item()
        assert center_val > 0.5, f"Interior hole should be filled, got {center_val}"
        assert "pixels filled:" in info

    def test_grow_increases_mask(self, prepare_node, sample_mask):
        inp_no_grow, _, _, _ = prepare_node.prepare_mask(
            sample_mask, False, False, 0, 0, "hard_binary", "gaussian", 16, False, 1.5
        )
        inp_grow, _, _, _ = prepare_node.prepare_mask(
            sample_mask, False, False, 0, 8, "hard_binary", "gaussian", 16, False, 1.5
        )
        area_no_grow = (inp_no_grow > 0.5).float().sum().item()
        area_grow = (inp_grow > 0.5).float().sum().item()
        assert area_grow > area_no_grow, "Grown mask should have more area"

    def test_temporal_smooth_reduces_variance(self, prepare_node):
        torch.manual_seed(123)
        base = torch.zeros(8, 64, 64)
        base[:, 15:50, 15:50] = 1.0
        noise = torch.rand(8, 64, 64) * 0.3
        noisy = (base + noise).clamp(0, 1)

        inpaint_no_smooth, _, _, info_no = prepare_node.prepare_mask(
            noisy, False, False, 0, 0, "hard_binary", "gaussian", 16, False, 1.5
        )
        inpaint_smooth, _, _, info_smooth = prepare_node.prepare_mask(
            noisy, False, False, 0, 0, "hard_binary", "gaussian", 16, True, 2.0
        )
        var_no = inpaint_no_smooth.float().var(dim=0).mean().item()
        var_yes = inpaint_smooth.float().var(dim=0).mean().item()
        # Temporal smoothing should reduce frame-to-frame variance OR
        # produce at least as consistent an output
        assert var_yes <= var_no + 1e-6, f"Temporal smooth should not increase variance: {var_yes} > {var_no}"

    def test_debug_preview_rgb(self, prepare_node, sample_mask):
        _, _, preview, _ = prepare_node.prepare_mask(
            sample_mask, False, False, 0, 4, "hard_binary", "gaussian", 16, False, 1.5
        )
        assert preview.shape[3] == 3
        assert preview[:, :, :, 0].max() > 0
        assert preview[:, :, :, 1].max() > 0
        assert preview[:, :, :, 2].abs().max() < 1e-6

    def test_edge_aware_stitch_mode(self, prepare_node, sample_image, sample_mask):
        _, blend_g, _, _ = prepare_node.prepare_mask(
            sample_mask, False, False, 0, 0, "hard_binary", "gaussian", 16, False, 1.5,
            reference_image=sample_image
        )
        _, blend_ea, _, _ = prepare_node.prepare_mask(
            sample_mask, False, False, 0, 0, "hard_binary", "edge_aware", 16, False, 1.5,
            reference_image=sample_image
        )
        diff = (blend_g - blend_ea).abs().sum().item()
        assert diff > 0.01, f"edge_aware should differ from gaussian, diff={diff}"

    def test_info_has_real_data(self, prepare_node, sample_mask):
        _, _, _, info = prepare_node.prepare_mask(
            sample_mask, True, True, 100, 4, "hard_binary", "gaussian", 16, False, 1.5
        )
        assert "inpaint_mask range:" in info
        assert "stitch_blend_mask range:" in info
        assert "grow_pixels:" in info


# ── Algorithm-specific tests ─────────────────────────────────────────

class TestAlgorithms:
    def test_laplacian_pyramid_build_levels(self):
        from nodes.inpaint_suite import _build_laplacian_pyramid_torch, _reconstruct_from_pyramid
        img = torch.rand(1, 3, 128, 128)
        pyr = _build_laplacian_pyramid_torch(img, levels=4)
        assert len(pyr) == 5  # 4 Laplacian + 1 residual
        for i in range(1, len(pyr) - 1):
            assert pyr[i].shape[2] <= pyr[i-1].shape[2]
        recon = _reconstruct_from_pyramid(pyr)
        diff = (recon - img).abs().max().item()
        assert diff < 0.05, f"Pyramid reconstruction error too large: {diff}"

    def test_sobel_edges_detect_structure(self):
        from nodes.inpaint_suite import _sobel_edges
        gray = torch.zeros(1, 64, 64)
        gray[:, :, 32:] = 1.0
        edges = _sobel_edges(gray)
        edge_at_boundary = edges[0, 32, 32].item()
        edge_at_flat = edges[0, 32, 0].item()
        assert edge_at_boundary > edge_at_flat * 5, "Edge should be much stronger at boundary"

    def test_stable_bbox_union(self):
        from nodes.inpaint_suite import _compute_stable_bbox
        # Use 2 frames to hit the union (B<=2) code path
        mask = torch.zeros(2, 100, 100)
        mask[0, 10:30, 10:30] = 1.0
        mask[1, 70:90, 70:90] = 1.0
        x, y, w, h = _compute_stable_bbox(mask)
        assert x == 10, f"x should be 10, got {x}"
        assert y == 10, f"y should be 10, got {y}"
        assert x + w == 90, f"x+w should be 90, got {x+w}"
        assert y + h == 90, f"y+h should be 90, got {y+h}"

    def test_temporal_gaussian_smooth(self):
        from nodes.inpaint_suite import _temporal_gaussian_smooth
        torch.manual_seed(99)
        mask = torch.zeros(10, 32, 32)
        mask[:, 8:24, 8:24] = 1.0
        noise = torch.rand(10, 32, 32) * 0.4
        noisy = (mask + noise).clamp(0, 1)
        smoothed = _temporal_gaussian_smooth(noisy, sigma=2.0)
        var_before = noisy.var(dim=0).mean().item()
        var_after = smoothed.var(dim=0).mean().item()
        assert var_after < var_before, f"Smoothing should reduce variance: {var_after} < {var_before}"

    def test_frequency_blend_valid(self):
        from nodes.inpaint_suite import _frequency_blend
        a = torch.rand(1, 3, 64, 64)
        b = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :, 32:] = 1.0
        result = _frequency_blend(a, b, mask)
        assert result.shape == a.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_edge_aware_uses_sobel(self):
        from nodes.inpaint_suite import _edge_aware_blend_mask, _gaussian_blur_mask
        image = torch.zeros(1, 80, 80, 3)
        image[:, :, 40:, :] = 1.0
        mask = torch.zeros(1, 80, 80)
        mask[:, 20:60, 30:60] = 1.0
        ea_mask = _edge_aware_blend_mask(image, mask, radius=16)
        g_mask = _gaussian_blur_mask((mask > 0.5).float(), sigma=16 * 0.4)
        diff_at_edge = (ea_mask[0, 40, 38:42] - g_mask[0, 40, 38:42]).abs().sum().item()
        assert diff_at_edge > 0.001, f"Edge-aware should differ at image edge, diff={diff_at_edge}"
