"""Smart Image Crop / Stitch.

The interesting failures here are geometric and silent. A crop can come back
squashed, offset by a few pixels, or taken from the whole frame when a mask
was there all along - and every one of those still renders something.

CPU-only, torch only, no ComfyUI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes.smart_crop import (  # noqa: E402
    MASK_EPS,
    SmartCropError,
    SmartImageCropMEC,
    SmartImageStitcherMEC,
    bbox_from_mask,
    box_feather_mask,
    color_match,
    edge_pad,
    fill_holes_torch,
    grow_or_shrink,
    mask_has_pixels,
    prepare_mask,
    snap_down,
    snap_up,
    target_size,
    widen_to_aspect,
)


def image(h=256, w=512, b=1):
    """A gradient, so a squash or an offset shows up as a value change."""
    ys = torch.linspace(0, 1, h).view(1, h, 1, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w, 1)
    return (ys * xs).repeat(b, 1, 1, 3).float()


def mask_with_box(h=256, w=512, box=(100, 50, 80, 40), value=1.0):
    m = torch.zeros(1, h, w)
    x, y, bw, bh = box
    m[0, y:y + bh, x:x + bw] = value
    return m


# ── snapping ────────────────────────────────────────────────────────────────

def test_snapping_goes_the_way_it_says():
    assert snap_up(1000, 128) == 1024
    assert snap_down(1000, 128) == 896
    assert snap_up(1024, 128) == 1024, "an exact multiple must not move"
    assert snap_down(1024, 128) == 1024


def test_snapping_never_returns_zero():
    """A zero dimension is an empty tensor far away from here."""
    assert snap_down(1, 128) == 128
    assert snap_up(0, 64) == 64


# ── target size ─────────────────────────────────────────────────────────────

def test_a_small_crop_is_scaled_up_to_the_minimum():
    """The whole reason to crop: a 200px face should arrive at the model
    large, not at 200px."""
    tw, th = target_size(200, 200, mode="Automatic", min_res=768, max_res=2048,
                         manual_w=0, manual_h=0, divisor=128)
    assert min(tw, th) >= 768


def test_scaling_up_snaps_up_so_the_minimum_is_actually_met():
    """Rounding to nearest here lands BELOW the minimum that was asked for,
    which is the bound quietly not holding."""
    for size in (100, 137, 201, 333):
        tw, th = target_size(size, size, mode="Automatic", min_res=768,
                             max_res=2048, manual_w=0, manual_h=0, divisor=128)
        assert min(tw, th) >= 768, f"{size} -> {tw}x{th}, below the minimum"


def test_a_large_crop_is_scaled_down_to_the_maximum():
    tw, th = target_size(6000, 6000, mode="Automatic", min_res=768,
                         max_res=2048, manual_w=0, manual_h=0, divisor=128)
    assert max(tw, th) <= 2048


def test_scaling_down_snaps_down_so_the_maximum_is_actually_obeyed():
    for size in (2100, 3000, 5000, 9999):
        tw, th = target_size(size, size, mode="Automatic", min_res=768,
                             max_res=2048, manual_w=0, manual_h=0, divisor=128)
        assert max(tw, th) <= 2048, f"{size} -> {tw}x{th}, over the maximum"


def test_a_crop_already_in_range_keeps_its_aspect():
    tw, th = target_size(1024, 512, mode="Automatic", min_res=256,
                         max_res=2048, manual_w=0, manual_h=0, divisor=128)
    assert tw / th == pytest.approx(2.0, abs=0.15)


def test_an_extreme_aspect_is_clamped_rather_than_demanding_the_impossible():
    """A 20:1 sliver cannot be both above the minimum on its short side and
    below the maximum on its long one. Clamping is the only answer that
    exists; the alternative is a silently enormous tensor."""
    tw, th = target_size(2000, 100, mode="Automatic", min_res=768,
                         max_res=2048, manual_w=0, manual_h=0, divisor=128)
    assert tw <= 2048 and th >= 768


@pytest.mark.parametrize("divisor", [8, 16, 32, 64, 128, 256])
def test_every_output_lands_on_the_divisor(divisor):
    for w, h in [(200, 200), (1000, 600), (6000, 400), (77, 913)]:
        tw, th = target_size(w, h, mode="Automatic", min_res=768, max_res=2048,
                             manual_w=0, manual_h=0, divisor=divisor)
        assert tw % divisor == 0 and th % divisor == 0


def test_manual_mode_uses_the_manual_numbers():
    tw, th = target_size(50, 900, mode="Manual", min_res=768, max_res=2048,
                         manual_w=1000, manual_h=500, divisor=128)
    assert (tw, th) == (snap_up(1000, 128), snap_up(500, 128))


# ── aspect matching ─────────────────────────────────────────────────────────

def test_the_crop_is_widened_to_the_output_aspect():
    """Crop tight and resize after, and the region arrives squashed - a face
    becomes a wide face, and the model obligingly generates a wide face."""
    w, h = widen_to_aspect(100, 100, 2048, 1024)
    assert w / h == pytest.approx(2.0, abs=1e-6)


def test_widening_only_ever_grows():
    """Shrinking to fit would cut off part of the region the mask asked for."""
    for cw, ch, tw, th in [(100, 100, 2048, 1024), (300, 100, 1024, 1024),
                           (50, 900, 1024, 512), (640, 480, 512, 512)]:
        nw, nh = widen_to_aspect(cw, ch, tw, th)
        assert nw >= cw - 1e-6 and nh >= ch - 1e-6


def test_a_crop_already_at_the_target_aspect_is_untouched():
    w, h = widen_to_aspect(200, 100, 1024, 512)
    assert (w, h) == pytest.approx((200, 100))


# ── mask detection ──────────────────────────────────────────────────────────

def test_a_very_light_grey_mask_is_still_a_mask():
    """Masks from paint tools, mattes and JPEG round-trips are rarely 1.0. A
    threshold of 0.5 throws the selection away and crops the whole frame,
    which looks like the node ignoring the mask."""
    faint = mask_with_box(value=4.0 / 255.0)
    assert mask_has_pixels(faint[0])
    assert bbox_from_mask(faint[0]) == [100, 50, 80, 40]


def test_a_genuinely_empty_mask_is_empty():
    assert not mask_has_pixels(torch.zeros(64, 64))
    assert bbox_from_mask(torch.zeros(64, 64)) is None


def test_the_threshold_sits_below_one_greylevel():
    assert MASK_EPS <= 1.0 / 255.0 + 1e-9


def test_the_bbox_is_tight():
    m = mask_with_box(box=(10, 20, 30, 40))
    assert bbox_from_mask(m[0]) == [10, 20, 30, 40]


# ── hole filling ────────────────────────────────────────────────────────────

def test_an_enclosed_hole_is_filled():
    """A subject mask with a gap regenerates that gap as background - the
    hole-in-the-face result."""
    m = torch.zeros(1, 64, 64)
    m[0, 10:50, 10:50] = 1.0
    m[0, 25:30, 25:30] = 0.0                  # a hole in the middle
    filled = fill_holes_torch(m)
    assert filled[0, 27, 27] == pytest.approx(1.0)


def test_a_bay_open_to_the_edge_is_not_filled():
    """An open notch is a shape, not a hole. Filling it would silently
    enlarge the selection."""
    m = torch.zeros(1, 64, 64)
    m[0, 10:50, 10:50] = 1.0
    m[0, 20:30, 10:30] = 0.0                  # opens out to the left edge
    filled = fill_holes_torch(m)
    assert filled[0, 25, 12] == pytest.approx(0.0)


def test_filling_does_not_grow_the_outline():
    m = torch.zeros(1, 64, 64)
    m[0, 10:50, 10:50] = 1.0
    filled = fill_holes_torch(m)
    assert filled[0, 9, 30] == pytest.approx(0.0)
    assert filled[0, 50, 30] == pytest.approx(0.0)


def test_an_empty_mask_survives_hole_filling():
    assert fill_holes_torch(torch.zeros(1, 32, 32)).max() == 0.0


def test_a_full_mask_survives_hole_filling():
    assert fill_holes_torch(torch.ones(1, 32, 32)).min() == pytest.approx(1.0)


# ── grow and shrink ─────────────────────────────────────────────────────────

def test_growing_enlarges_the_bbox_by_the_amount_asked_for():
    m = mask_with_box(h=256, w=256, box=(100, 100, 20, 20))
    grown = grow_or_shrink(m, 8)
    x, y, w, h = bbox_from_mask(grown[0])
    assert w == 20 + 16 and h == 20 + 16
    assert x == 92 and y == 92


def test_shrinking_reduces_it():
    m = mask_with_box(h=256, w=256, box=(100, 100, 40, 40))
    shrunk = grow_or_shrink(m, -8)
    _, _, w, h = bbox_from_mask(shrunk[0])
    assert w == 40 - 16 and h == 40 - 16


def test_zero_is_a_no_op():
    m = mask_with_box()
    assert torch.equal(grow_or_shrink(m, 0), m)


def test_prepare_mask_solidifies_a_faint_mask_before_growing():
    """Order matters: grow a faint mask first and max-pooling spreads a 0.02
    value around instead of a selection."""
    faint = mask_with_box(value=3.0 / 255.0)
    out, _ = prepare_mask(faint, patch_holes=False, grow_pixels=4)
    assert out.max() == pytest.approx(1.0)


# ── edge padding ────────────────────────────────────────────────────────────

def test_a_crop_past_the_edge_repeats_the_edge_pixel():
    """Zero-padding puts a black band in the model's input and it generates
    content to match it."""
    img = image(64, 64)
    canvas, (pad_l, pad_t) = edge_pad(img, -10, -10, 40, 40)
    assert canvas.shape == (1, 40, 40, 3)
    assert (pad_l, pad_t) == (10, 10)
    assert canvas[0, 0, 20, 0] == pytest.approx(float(canvas[0, 10, 20, 0]))
    assert canvas.max() > 0.0, "the padded region came back black"


def test_a_crop_entirely_outside_the_frame_is_blank_not_an_error():
    canvas, _ = edge_pad(image(64, 64), 500, 500, 32, 32)
    assert canvas.shape == (1, 32, 32, 3)


def test_an_interior_crop_is_exact():
    img = image(64, 64)
    canvas, pad = edge_pad(img, 10, 20, 16, 8)
    assert pad == (0, 0)
    assert torch.allclose(canvas, img[:, 20:28, 10:26, :])


# ── blending ────────────────────────────────────────────────────────────────

def test_a_box_feather_is_soft_at_the_edge_and_solid_in_the_middle():
    """The edge does not reach 0 - the blur pads by reflection, which pulls
    interior values outward - but it must be far softer than the centre or
    the seam shows."""
    m = box_feather_mask(64, 64, 16, torch.device("cpu"), torch.float32)
    assert m[0, 32, 32] > 0.9
    assert m[0, 0, 32] < 0.35
    assert m[0, 0, 32] < m[0, 32, 32] / 4


def test_a_zero_feather_is_a_hard_rectangle():
    m = box_feather_mask(32, 32, 0, torch.device("cpu"), torch.float32)
    assert m.min() == pytest.approx(1.0)


def test_a_feather_wider_than_the_crop_does_not_erase_it():
    """The margin has to be clamped or a small crop blends to nothing and the
    stitch appears to do nothing at all."""
    m = box_feather_mask(8, 8, 200, torch.device("cpu"), torch.float32)
    assert m.max() > 0.0


def test_colour_matching_moves_the_crop_toward_the_plate():
    dark = torch.full((1, 16, 16, 3), 0.2)
    bright = torch.full((1, 16, 16, 3), 0.8)
    out = color_match(dark, bright, amount=1.0)
    assert float(out.mean()) > 0.7


def test_zero_amount_changes_nothing():
    dark = torch.full((1, 16, 16, 3), 0.2)
    bright = torch.full((1, 16, 16, 3), 0.8)
    assert torch.equal(color_match(dark, bright, amount=0.0), dark)


# ── the crop node ───────────────────────────────────────────────────────────

def run_crop(**kw):
    args = dict(image=image(), mask=mask_with_box(),
                resolution_mode="Automatic", max_resolution=2048,
                min_resolution=768, manual_width=1024, manual_height=1024,
                mask_grow_pixels=32, patch_mask_holes=True,
                no_mask_mode="Bypass", force_divisibility=128)
    args.update(kw)
    return SmartImageCropMEC().crop(**args)


def test_the_crop_is_divisible_and_around_the_mask():
    crop, crop_mask, st, overlay, info = run_crop()
    assert crop.shape[1] % 128 == 0 and crop.shape[2] % 128 == 0
    assert crop_mask.shape[1:] == crop.shape[1:3]
    assert st["bypass"] is False
    assert overlay.shape == (1, 256, 512, 3)
    assert "Cropped 1 frame" in info


def test_the_crop_is_not_squashed():
    """The output aspect and the source region's aspect must agree, or the
    model sees a distorted subject."""
    _, _, st, _, _ = run_crop()
    src = st["w"][0] / st["h"][0]
    out = st["target_w"][0] / st["target_h"][0]
    assert src == pytest.approx(out, rel=0.02)


def test_an_empty_mask_bypasses_by_default():
    img = image()
    crop, _, st, _, info = run_crop(image=img, mask=torch.zeros(1, 256, 512))
    assert st["bypass"] is True
    assert torch.equal(crop, img)
    assert "empty" in info


def test_resize_full_image_mode_resizes_instead():
    crop, _, st, _, info = run_crop(mask=torch.zeros(1, 256, 512),
                                    no_mask_mode="Resize Full Image")
    assert st["full_image_resize"] is True
    assert crop.shape[2] % 128 == 0
    assert "Resized the whole frame" in info


def test_crop_full_image_mode_uses_the_whole_frame():
    crop, mask_out, st, _, _ = run_crop(mask=torch.zeros(1, 256, 512),
                                        no_mask_mode="Crop Full Image")
    assert crop.shape == (1, 256, 512, 3)
    assert mask_out.min() == pytest.approx(1.0)
    assert st["bypass"] is False


def test_impossible_limits_are_refused_rather_than_silently_swapped():
    with pytest.raises(SmartCropError, match="Nothing can satisfy both"):
        run_crop(min_resolution=2048, max_resolution=768)


def test_a_faint_mask_still_produces_a_real_crop():
    _, _, st, _, _ = run_crop(mask=mask_with_box(value=3.0 / 255.0))
    assert st["w"][0] < 512, "the whole frame was cropped - the mask was missed"


def test_a_batch_where_only_one_frame_has_a_mask_keeps_both_frames():
    """Dropping the unmasked frame would silently shorten the batch, and
    passing it through at full size cannot concatenate with a 768px crop."""
    masks = torch.zeros(2, 256, 512)
    masks[0, 50:90, 100:180] = 1.0
    crop, _, st, _, info = run_crop(image=image(b=2), mask=masks)
    assert crop.shape[0] == 2
    assert len(st["x"]) == 2
    assert "no mask" in info


def test_a_batch_with_differently_sized_masks_produces_one_tensor():
    """THE batch bug, inherited from upstream. Sizing each frame from its own
    mask is right frame by frame and then fails at torch.cat with a shape
    error that says nothing about masks."""
    masks = torch.zeros(3, 256, 512)
    masks[0, 50:90, 100:180] = 1.0       # 80x40
    masks[1, 20:200, 50:450] = 1.0       # 400x180
    masks[2, 10:30, 10:40] = 1.0         # 30x20
    crop, crop_mask, st, _, info = run_crop(image=image(b=3), mask=masks)
    assert crop.shape[0] == 3
    assert crop_mask.shape[0] == 3
    assert len(set(zip(st["target_w"], st["target_h"]))) == 1, (
        "the frames were given different output sizes")
    assert "different sizes" in info


def test_the_batch_size_comes_from_the_largest_mask():
    """Taking it from the smallest leaves the big region under-resolved,
    which is the opposite of the reason to crop."""
    small = torch.zeros(1, 256, 512)
    small[0, 50:90, 100:180] = 1.0
    big = torch.zeros(1, 256, 512)
    big[0, 20:200, 50:450] = 1.0

    _, _, st_big, _, _ = run_crop(image=image(), mask=big)
    both = torch.cat([small, big], dim=0)
    _, _, st_both, _, _ = run_crop(image=image(b=2), mask=both)
    assert st_both["target_w"][0] == st_big["target_w"][0]
    assert st_both["target_h"][0] == st_big["target_h"][0]


# ── the stitcher ────────────────────────────────────────────────────────────

def test_a_round_trip_puts_the_crop_back_where_it_came_from():
    img = image()
    crop, _, st, _, _ = run_crop(image=img)
    red = torch.zeros_like(crop)
    red[..., 0] = 1.0
    out, info = SmartImageStitcherMEC().stitch(
        original_image=img, processed_image=red, stitcher=st,
        feather_pixels=0, blend_mode="Hard Paste")
    assert out.shape == img.shape
    # the pasted region is red, and somewhere far from it is untouched
    assert float(out[0, 70, 140, 0]) > 0.9
    assert torch.allclose(out[0, 250, 500, :], img[0, 250, 500, :], atol=1e-5)
    assert "Stitched 1 region" in info


def test_a_bypassed_stitcher_returns_the_original_untouched():
    img = image()
    _, _, st, _, _ = run_crop(image=img, mask=torch.zeros(1, 256, 512))
    out, info = SmartImageStitcherMEC().stitch(
        original_image=img, processed_image=image(64, 64), stitcher=st,
        feather_pixels=8)
    assert torch.equal(out, img)
    assert "bypass" in info.lower()


def test_mask_feather_without_a_mask_says_so_instead_of_doing_nothing():
    """A silent fallback here looks exactly like the feather being broken."""
    img = image()
    crop, _, st, _, _ = run_crop(image=img)
    out, info = SmartImageStitcherMEC().stitch(
        original_image=img, processed_image=torch.zeros_like(crop), stitcher=st,
        feather_pixels=16, blend_mode="Mask Feather", mask=None)
    assert "no mask is connected" in info
    assert out.shape == img.shape


def test_the_wrong_thing_wired_into_the_stitcher_is_named():
    with pytest.raises(SmartCropError, match="not a stitcher"):
        SmartImageStitcherMEC().stitch(
            original_image=image(), processed_image=image(), stitcher="nope",
            feather_pixels=0)


def test_a_stitcher_missing_its_geometry_is_named():
    with pytest.raises(SmartCropError, match="did not come from"):
        SmartImageStitcherMEC().stitch(
            original_image=image(), processed_image=image(),
            stitcher={"bypass": False, "x": [0]}, feather_pixels=0)


def test_the_original_image_is_not_modified_in_place():
    """ComfyUI hands out inference tensors that cannot be written to, and a
    node that mutates its input corrupts whatever else is reading it."""
    img = image()
    before = img.clone()
    crop, _, st, _, _ = run_crop(image=img)
    SmartImageStitcherMEC().stitch(
        original_image=img, processed_image=torch.zeros_like(crop), stitcher=st,
        feather_pixels=0, blend_mode="Hard Paste")
    assert torch.equal(img, before), "the input image was written to"


def test_colour_matching_runs_without_changing_the_frame_shape():
    img = image()
    crop, _, st, _, _ = run_crop(image=img)
    out, info = SmartImageStitcherMEC().stitch(
        original_image=img, processed_image=torch.full_like(crop, 0.9),
        stitcher=st, feather_pixels=8, enable_color_match=True,
        color_match_amount=1.0)
    assert out.shape == img.shape
    assert "Colour matched" in info
