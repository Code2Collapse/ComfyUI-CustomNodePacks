"""L7.99 (owner A11, 2026-10-13): Inpaint Crop Pro / Stitch Pro - the crop window, the stitch round trip and the
post-run preview agree with what executes.

Reproduced before the fix (docs/evidence/L7.99/repro_before.json): a mask near the border got a window padded with
smeared edge pixels although a window of the same size fits inside the image (41 % synthetic for a 16:9 target);
an empty mask changed the image (max error 0.033) because the padded mask band leaked through the stitch feather.
"""
from __future__ import annotations

import os

import pytest
import torch

DEFAULTS = dict(
    downscale_algorithm="bilinear", upscale_algorithm="bicubic",
    preresize=False, preresize_mode="ensure minimum resolution",
    preresize_min_width=1024, preresize_min_height=1024, preresize_max_width=16384, preresize_max_height=16384,
    mask_fill_holes=True, mask_expand_pixels=0, mask_invert=False, mask_blend_pixels=32, mask_hipass_filter=0.1,
    extend_for_outpainting=False, extend_up_factor=1.0, extend_down_factor=1.0, extend_left_factor=1.0,
    extend_right_factor=1.0, context_from_mask_extend_factor=1.2, auto_context_factor=False, aspect_preset="Custom",
    output_resize_to_target_size=True, output_target_width=512, output_target_height=512, output_padding="32",
    device_mode="cpu (compatible)", wan_align_multiple=16, wan_temporal_smooth_frames=0.0, wan_stable_crop=True,
    wan_mask_polarity="regenerate_subject", inpaint_mask_mode="hard_binary", stitch_blend_mode="gaussian",
    blend_radius=32, video_stable_temporal_sigma=3.0, video_stable_dilate_px=-1, video_stable_blur_sigma=-1.0,
    fill_masked_area="none",
)
H, W = 240, 320


@pytest.fixture(scope="module")
def suite():
    return pytest.importorskip("nodes.inpaint_suite")


@pytest.fixture(autouse=True)
def temp_dir(monkeypatch, tmp_path):
    """The preview PNGs go to ComfyUI's temp folder; the test stub of folder_paths has none."""
    import sys
    fp = sys.modules.get("folder_paths")
    if fp is not None:
        monkeypatch.setattr(fp, "get_temp_directory", lambda: str(tmp_path), raising=False)
    return tmp_path


def plate(B=1, h=H, w=W, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, h).view(1, h, 1, 1).expand(B, h, w, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, w, 1).expand(B, h, w, 1)
    img = torch.cat([xx, yy, xx * yy], -1) * 0.9 + torch.rand(B, h, w, 3, generator=g) * 0.1
    if c == 4:
        img = torch.cat([img, torch.rand(B, h, w, 1, generator=g)], -1)
    return img.clamp(0, 1)


def box(B, x0, y0, x1, y1, h=H, w=W, dx=0):
    m = torch.zeros(B, h, w)
    for b in range(B):
        m[b, y0:y1, max(0, x0 + b * dx):min(w, x1 + b * dx)] = 1.0
    return m


def crop(suite, img, mask, **over):
    kw = dict(DEFAULTS)
    kw.update(over)
    out = suite.InpaintCropProMEC().inpaint_crop(img, mask=mask, **kw)
    res = out["result"] if isinstance(out, dict) else out
    return res, (out.get("ui") if isinstance(out, dict) else None)


def stitch(suite, st, inpainted):
    out = suite.InpaintStitchProMEC().inpaint_stitch(st, inpainted)
    return out["result"] if isinstance(out, dict) else out


def windows(st):
    """Each frame's crop window in original-image coordinates (x, y, w, h)."""
    out = []
    for j in range(len(st["cropped_to_canvas_x"])):
        out.append((st["cropped_to_canvas_x"][j] - st["canvas_to_orig_x"][j],
                    st["cropped_to_canvas_y"][j] - st["canvas_to_orig_y"][j],
                    st["cropped_to_canvas_w"][j], st["cropped_to_canvas_h"][j]))
    return out


# ── D2: the window stays on real pixels whenever it can ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mask_box, tw, th", [
    ((0, 80, 30, 160), 896, 512),        # left border, wide target
    ((290, 0, 320, 30), 512, 512),       # top-right corner
    ((150, 200, 170, 240), 512, 512),    # bottom border, tall mask
    ((0, 0, 20, 240), 512, 896),         # full-height strip at the left, tall target
    ((300, 100, 320, 140), 1024, 512),   # right border, very wide target
])
def test_window_inside_the_image_when_it_fits(suite, mask_box, tw, th):
    (st, *_), _ui = crop(suite, plate(), box(1, *mask_box), output_target_width=tw, output_target_height=th)
    (x, y, w, h), = windows(st)
    if w <= W:
        assert 0 <= x and x + w <= W, (x, w)
    if h <= H:
        assert 0 <= y and y + h <= H, (y, h)


def test_window_larger_than_the_image_covers_all_of_it(suite):
    # empty mask -> whole frame; a square target makes the window 320 high on a 240 high image
    (st, *_), _ui = crop(suite, plate(), box(1, 100, 100, 200, 200))
    (st2, *_), _ui = crop(suite, plate(), torch.ones(1, H, W))
    for (x, y, w, h) in windows(st) + windows(st2):
        if h > H:
            assert y <= 0 and y + h >= H
        if w > W:
            assert x <= 0 and x + w >= W


def test_window_still_contains_the_mask(suite):
    for mb in [(0, 80, 30, 160), (290, 0, 320, 30), (150, 200, 170, 240)]:
        (st, *_), _ui = crop(suite, plate(), box(1, *mb), mask_blend_pixels=0, context_from_mask_extend_factor=1.0)
        (x, y, w, h), = windows(st)
        assert x <= mb[0] and y <= mb[1] and x + w >= mb[2] and y + h >= mb[3]


# ── D3 + round trips ────────────────────────────────────────────────────────────────────────────────────────────

def test_empty_mask_is_an_exact_identity(suite):
    img = plate()
    (st, cropped, *_), _ui = crop(suite, img, torch.zeros(1, H, W))
    out, used, _info = stitch(suite, st, cropped)
    assert out.shape == img.shape
    assert torch.equal(out, img) and float(used.max()) == 0.0


@pytest.mark.parametrize("over", [
    {}, {"output_resize_to_target_size": False}, {"stitch_blend_mode": "edge_aware"},
    {"stitch_blend_mode": "laplacian_pyramid"}, {"stitch_blend_mode": "frequency_blend"},
    {"stitch_blend_mode": "video_stable"}, {"aspect_preset": "16:9"}, {"wan_align_multiple": 64},
])
@pytest.mark.parametrize("mb", [(130, 90, 190, 150), (0, 80, 30, 160), (290, 0, 320, 30), (0, 0, 320, 240)])
def test_identity_round_trip_changes_nothing_outside_the_blend(suite, over, mb):
    img = plate()
    (st, cropped, *_), _ui = crop(suite, img, box(1, *mb), **over)
    out, used, _info = stitch(suite, st, cropped)
    assert out.shape == img.shape
    outside = used == 0                                                 # blend_mask_used = the stitch's footprint
    assert torch.equal(out[outside], img[outside])


def test_no_resize_round_trip_is_exact_everywhere(suite):
    img = plate()
    for mb in [(130, 90, 190, 150), (0, 80, 30, 160), (290, 0, 320, 30)]:
        (st, cropped, *_), _ui = crop(suite, img, box(1, *mb), output_resize_to_target_size=False)
        out, _used, _info = stitch(suite, st, cropped)
        assert torch.allclose(out, img, atol=1e-6)


def test_stitch_puts_the_result_back_where_it_was_cropped(suite):
    img = plate()
    mb = (40, 60, 100, 120)
    (st, cropped, *_), _ui = crop(suite, img, box(1, *mb), stitch_blend_mode="gaussian",
                                  mask_blend_pixels=0, blend_radius=4)
    white = torch.ones_like(cropped)
    out, used, _info = stitch(suite, st, white)
    core = out[0, mb[1] + 10:mb[3] - 10, mb[0] + 10:mb[2] - 10]
    assert float(core.min()) > 0.99                                     # inside the mask: the new pixels
    far = out[0, 200:, 250:]
    assert torch.equal(far, img[0, 200:, 250:])                         # far away: the plate


def test_rgba_plate_keeps_its_alpha_and_odd_sizes_work(suite):
    img = plate(h=233, w=317, c=4)
    (st, cropped, *_), _ui = crop(suite, img, box(1, 50, 40, 90, 100, h=233, w=317))
    out, used, _info = stitch(suite, st, cropped[..., :3])
    assert out.shape == img.shape and torch.equal(out[..., 3], img[..., 3])


def test_preserve_subject_polarity_round_trip(suite):
    img = plate()
    keep = 1.0 - box(1, 130, 90, 190, 150)                              # Wan replacement: 1 = keep
    (st, cropped, *_), _ui = crop(suite, img, keep, wan_mask_polarity="preserve_subject")
    out, used, _info = stitch(suite, st, cropped)
    regenerated = used < 1.0 - 1e-6                                     # 'used' is in the user's polarity
    untouched = used > 1.0 - 1e-6
    assert torch.allclose(out[untouched], img[untouched], atol=0, rtol=0)
    assert regenerated.any()


# ── batches: per-frame windows, the stable union, the preview frame ─────────────────────────────────────────────

def test_batch_semantics(suite):
    img = plate(B=5)
    moving = box(5, 20, 100, 60, 140, dx=50)
    (st, cropped, *_), _ui = crop(suite, img, moving)
    assert len(set(windows(st))) == 1                                   # stable: one window for every frame
    (x, y, w, h) = windows(st)[0]
    assert x <= 20 and x + w >= 60 + 4 * 50                             # the union of all frames
    (st2, cropped2, *_), _ui = crop(suite, img, moving, wan_stable_crop=False)
    assert len(set(windows(st2))) == 5                                  # per frame
    out, _u, _i = stitch(suite, st2, cropped2)
    assert out.shape == img.shape


def test_preview_frame_is_display_only(suite):
    img = plate(B=4)
    moving = box(4, 20, 100, 60, 140, dx=60)
    a, ui_a = crop(suite, img, moving, wan_stable_crop=False)
    b, ui_b = crop(suite, img, moving, wan_stable_crop=False, preview_frame=3)
    c, ui_c = crop(suite, img, moving, wan_stable_crop=False, preview_frame=99)   # clamps to the last frame
    for x, y in ((a, b), (a, c)):
        assert windows(x[0]) == windows(y[0]) and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2])
    assert ui_a and ui_b and ui_a["images"] and ui_b["images"]


# ── the post-run preview is drawn from the executed window ──────────────────────────────────────────────────────

def test_preview_images_exist_and_show_the_executed_window(suite, temp_dir):
    img = plate()
    (st, cropped, *_rest), ui = crop(suite, img, box(1, 0, 80, 30, 160), output_target_width=896,
                                     output_target_height=512)
    assert ui and len(ui["images"]) == 1                                 # one picture: window + crop
    for im in ui["images"]:
        assert im["type"] == "temp" and im["filename"].endswith(".png")
        assert os.path.isfile(os.path.join(str(temp_dir), im["subfolder"], im["filename"]))
    (plan,) = ui["c2c_crop_plan"]                                       # ui values are lists (core merges them)
    assert plan["frames"][0]["window"] == list(windows(st)[0])
    assert plan["image_size"] == [W, H] and plan["output_size"] == [int(cropped.shape[2]), int(cropped.shape[1])]
