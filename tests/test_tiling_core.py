"""Tiling: split a frame, put it back, see no seam.

Three things are pinned here because each is a silent failure in a finished
shot rather than an error anyone sees while working:

  * the complementary-window identity. If `rise + fall` is not exactly 1 the
    overlap is a visible band, and it is visible on a gradient long before it
    is visible on a test pattern.
  * round-trip identity. Split and merge with no refinement must return the
    input EXACTLY. Anything else means the blend is lossy, and a lossy blend
    applied 124 times is a shot that breathes.
  * one plan per clip. A plan recomputed per frame gives slightly different
    boundaries, and a seam that MOVES is far more visible than one that does
    not - this is the video-specific bug an image-upscaler port gets wrong.

CPU-only, torch only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes.tiling._core import (  # noqa: E402
    BLEND_MODES,
    TilingError,
    build_plan,
    describe_plan,
    fall,
    merge,
    plan_for_tile_size,
    rise,
    split,
    validate_plan,
    window_1d,
    window_2d,
)


def gradient(h=512, w=768, b=1, c=3):
    """A smooth ramp in both axes - the thing a seam shows up on. A noise
    pattern hides seams; a gradient does not."""
    ys = torch.linspace(0, 1, h).view(1, h, 1, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w, 1)
    return (0.25 + 0.5 * ys * xs).repeat(b, 1, 1, c).float()


# ── the window identity ─────────────────────────────────────────────────────

@pytest.mark.parametrize("length", [1, 2, 7, 16, 64, 65, 128, 333])
def test_rise_and_fall_sum_to_exactly_one(length):
    """THE identity the whole thing rests on. A window pair that sums to
    0.98 in the middle of the overlap is a dark band down the frame."""
    total = rise(length) + fall(length)
    assert torch.allclose(total, torch.ones(length), atol=1e-6), \
        f"max deviation {float((total - 1).abs().max())}"


def test_rise_starts_at_zero_and_fall_ends_near_zero():
    r, f = rise(64), fall(64)
    assert float(r[0]) == pytest.approx(0.0, abs=1e-7)
    assert float(f[0]) == pytest.approx(1.0, abs=1e-7)
    assert float(r[-1]) < 1.0 and float(f[-1]) > 0.0


def test_a_zero_length_window_is_empty_not_an_error():
    assert rise(0).numel() == 0 and fall(0).numel() == 0


def test_two_overlapping_windows_sum_to_one_across_the_seam():
    """The identity applied the way it is actually used: tile A fades out over
    the same pixels tile B fades in over."""
    size, ov = 128, 32
    a = window_1d(size, 0, ov)           # left tile: flat then fades out
    b = window_1d(size, ov, 0)           # right tile: fades in then flat
    seam = a[size - ov:] + b[:ov]
    assert torch.allclose(seam, torch.ones(ov), atol=1e-6)


@pytest.mark.parametrize("blend", ["cosine", "linear"])
def test_every_blend_mode_is_complementary(blend):
    size, ov = 64, 16
    a = window_1d(size, 0, ov, blend=blend)
    b = window_1d(size, ov, 0, blend=blend)
    assert torch.allclose(a[size - ov:] + b[:ov], torch.ones(ov), atol=1e-6)


def test_blend_none_is_a_flat_window():
    assert torch.allclose(window_1d(32, 8, 8, blend="none"), torch.ones(32))


def test_an_unknown_blend_mode_is_named():
    with pytest.raises(TilingError, match="Unknown blend"):
        window_1d(32, 8, 8, blend="gaussian")


def test_a_2d_window_is_the_outer_product():
    plan = build_plan(256, 256, rows=2, cols=2, overlap=32)
    w = window_2d(plan.tiles[0], blend="cosine")
    assert w.shape == (plan.tiles[0].height, plan.tiles[0].width)
    assert float(w.max()) <= 1.0 + 1e-6
    assert float(w.min()) >= 0.0


# ── the plan ────────────────────────────────────────────────────────────────

def test_a_plan_covers_every_pixel():
    """A pixel in no tile is a black rectangle found on playback."""
    for h, w, r, c, ov in [(512, 768, 2, 2, 64), (1080, 1920, 3, 4, 96),
                           (333, 517, 3, 3, 17), (64, 64, 1, 1, 0)]:
        validate_plan(build_plan(h, w, rows=r, cols=c, overlap=ov))


def test_tiles_are_all_the_same_size():
    """A short final tile is a different denoise problem from its neighbours
    and shows as a band along one edge."""
    plan = build_plan(1000, 1000, rows=3, cols=3, overlap=64)
    sizes = {(t.height, t.width) for t in plan.tiles}
    assert len(sizes) == 1, f"tiles came out different sizes: {sizes}"


def test_the_last_tile_ends_exactly_at_the_frame_edge():
    """Rounded, it would leave a sliver uncovered or run past the frame."""
    plan = build_plan(1000, 1337, rows=3, cols=4, overlap=64)
    assert max(t.bottom for t in plan.tiles) == 1000
    assert max(t.right for t in plan.tiles) == 1337


def test_edge_tiles_do_not_fade_into_the_frame_border():
    """A tile that fades at the frame edge has no neighbour to make up the
    weight, so the rim of the picture goes dark - the classic tiled-upscale
    vignette."""
    plan = build_plan(512, 512, rows=2, cols=2, overlap=64)
    by_pos = {(t.row, t.col): t for t in plan.tiles}
    assert by_pos[(0, 0)].fade_top == 0 and by_pos[(0, 0)].fade_left == 0
    assert by_pos[(1, 1)].fade_bottom == 0 and by_pos[(1, 1)].fade_right == 0


def test_neighbouring_fades_are_the_same_length():
    """Two tiles whose fades differ in length do not sum to 1 across the
    overlap, which is a seam."""
    plan = build_plan(900, 1200, rows=3, cols=3, overlap=64)
    by_pos = {(t.row, t.col): t for t in plan.tiles}
    for r in range(plan.rows - 1):
        for c in range(plan.cols):
            assert by_pos[(r, c)].fade_bottom == by_pos[(r + 1, c)].fade_top
    for r in range(plan.rows):
        for c in range(plan.cols - 1):
            assert by_pos[(r, c)].fade_right == by_pos[(r, c + 1)].fade_left


def test_a_single_tile_plan_is_the_whole_frame():
    plan = build_plan(256, 384, rows=1, cols=1, overlap=64)
    t = plan.tiles[0]
    assert (t.top, t.left, t.height, t.width) == (0, 0, 256, 384)
    assert (t.fade_top, t.fade_bottom, t.fade_left, t.fade_right) == (0, 0, 0, 0)


def test_an_overlap_as_wide_as_the_frame_is_refused():
    """The only way the tile size can come out no larger than the overlap:
    size = ceil((total + ov(n-1))/n) <= ov  happens exactly when total <= ov."""
    with pytest.raises(TilingError, match="as wide as"):
        build_plan(64, 64, rows=4, cols=4, overlap=64)


def test_a_wasteful_plan_is_allowed_but_reported():
    """256px into 8x8 tiles at 64px overlap gives 88px tiles advancing 24px -
    each one redoes 73% of its neighbour. That is a bad idea, not an error: it
    still produces a correct picture, just slowly. Refusing a configuration
    that works is over-reach, so it is the report's job.
    """
    plan = build_plan(256, 256, rows=8, cols=8, overlap=64)
    validate_plan(plan)
    assert plan.redundancy > 1.0
    assert "twice the frame area" in describe_plan(plan)


def test_a_sensible_plan_is_not_caught_by_either_guard():
    """The guards must not become so eager that ordinary plans are refused."""
    for h, w, r, c, ov in [(1080, 1920, 2, 3, 64), (2048, 2048, 3, 3, 96),
                           (512, 512, 2, 2, 64), (4096, 2160, 4, 6, 128)]:
        build_plan(h, w, rows=r, cols=c, overlap=ov)


def test_a_tile_smaller_than_its_overlap_is_refused():
    with pytest.raises(TilingError, match="very slow blur"):
        plan_for_tile_size(1024, 1024, tile=64, overlap=64)


def test_a_tile_size_is_a_maximum_not_a_target():
    """Tiles are equal-sized and must tile the frame exactly, so the size that
    comes out is the largest at or below the request that does. 1024 across
    2048 at 64 overlap gives three 726px tiles - two would need 1056px, over
    the limit."""
    plan = plan_for_tile_size(2048, 2048, tile=1024, overlap=64)
    t = plan.tiles[0]
    assert t.height <= 1024 and t.width <= 1024, "the maximum was exceeded"
    assert t.height > 512, "the tiles came out needlessly small"
    validate_plan(plan)


def test_a_larger_tile_request_gives_fewer_tiles():
    small = plan_for_tile_size(2048, 2048, tile=512, overlap=64)
    large = plan_for_tile_size(2048, 2048, tile=1024, overlap=64)
    assert len(large.tiles) < len(small.tiles)


def test_a_tile_request_larger_than_the_frame_gives_one_tile():
    plan = plan_for_tile_size(512, 512, tile=4096, overlap=64)
    assert len(plan.tiles) == 1


# ── the video rule ──────────────────────────────────────────────────────────

def test_the_same_inputs_always_give_the_same_plan():
    """THE video-specific bug. A plan recomputed per frame must be identical,
    or the seam moves - and a seam that moves is the most distracting thing in
    the frame, far worse than one that does not."""
    a = build_plan(1080, 1920, rows=3, cols=4, overlap=96)
    b = build_plan(1080, 1920, rows=3, cols=4, overlap=96)
    assert a.tiles == b.tiles


def test_one_plan_drives_every_frame_of_a_clip():
    """The plan is geometry; it does not know how many frames there are."""
    plan = build_plan(256, 256, rows=2, cols=2, overlap=32)
    clip = gradient(256, 256, b=8)
    pieces = split(clip, plan)
    assert len(pieces) == 4
    assert all(p.shape[0] == 8 for p in pieces), "a frame was dropped"
    out = merge(pieces, plan)
    assert out.shape == clip.shape


# ── the round trip ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("blend", ["cosine", "linear"])
def test_split_then_merge_returns_the_original_exactly(blend):
    """No refinement in between, so anything but the input back means the
    blend is lossy - and a lossy blend applied over 124 frames is a shot that
    breathes."""
    img = gradient(512, 768)
    plan = build_plan(512, 768, rows=2, cols=3, overlap=64, blend=blend)
    out = merge(split(img, plan), plan)
    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-5), \
        f"max error {float((out - img).abs().max())}"


def test_the_round_trip_is_exact_at_the_frame_edges_too():
    """Where the windows legitimately do not sum to 1 - there is no neighbour
    - the weight division is what rescues the level. Without it the rim is
    dark."""
    img = torch.full((1, 256, 256, 3), 0.5)
    plan = build_plan(256, 256, rows=2, cols=2, overlap=48)
    out = merge(split(img, plan), plan)
    for probe in [(0, 0), (0, 255), (255, 0), (255, 255), (128, 0), (0, 128)]:
        assert float(out[0, probe[0], probe[1], 0]) == pytest.approx(0.5, abs=1e-5)


def test_there_is_no_seam_on_a_flat_field():
    """A seam on a flat grey is the clearest possible test: every pixel must
    come back at the same value, so any variation IS the seam."""
    img = torch.full((1, 400, 600, 3), 0.42)
    plan = build_plan(400, 600, rows=2, cols=3, overlap=64)
    out = merge(split(img, plan), plan)
    spread = float(out.max() - out.min())
    assert spread < 1e-5, f"the blend left a {spread:.2e} band"


def test_a_latent_round_trips_too():
    """BCHW, not BHWC - tiling the channel axis would be confident nonsense."""
    lat = torch.randn(1, 16, 64, 96)
    plan = build_plan(64, 96, rows=2, cols=2, overlap=16)
    out = merge(split(lat, plan), plan)
    assert out.shape == lat.shape
    assert torch.allclose(out, lat, atol=1e-5)


def test_a_video_latent_round_trips():
    lat = torch.randn(1, 16, 9, 64, 64)
    plan = build_plan(64, 64, rows=2, cols=2, overlap=16)
    out = merge(split(lat, plan), plan)
    assert out.shape == lat.shape
    assert torch.allclose(out, lat, atol=1e-5)


# ── upscaling ───────────────────────────────────────────────────────────────

def test_merging_upscaled_tiles_lands_at_the_right_size():
    img = gradient(256, 256)
    plan = build_plan(256, 256, rows=2, cols=2, overlap=32)
    pieces = [torch.nn.functional.interpolate(
        p.movedim(-1, 1), scale_factor=2, mode="bilinear",
        align_corners=False).movedim(1, -1) for p in split(img, plan)]
    out = merge(pieces, plan, scale=2.0)
    assert out.shape == (1, 512, 512, 3)


def test_an_upscaled_merge_has_no_seam():
    img = torch.full((1, 256, 256, 3), 0.3)
    plan = build_plan(256, 256, rows=2, cols=2, overlap=64)
    pieces = [torch.nn.functional.interpolate(
        p.movedim(-1, 1), scale_factor=2, mode="nearest").movedim(1, -1)
        for p in split(img, plan)]
    out = merge(pieces, plan, scale=2.0)
    assert float(out.max() - out.min()) < 1e-5


def test_a_scale_that_does_not_match_the_tiles_is_named():
    """Inferring the scale from one tile silently rounds a non-integer ratio
    and shifts every tile after the first, so it is passed explicitly and
    checked."""
    img = gradient(256, 256)
    plan = build_plan(256, 256, rows=2, cols=2, overlap=32)
    pieces = [torch.nn.functional.interpolate(
        p.movedim(-1, 1), scale_factor=2, mode="nearest").movedim(1, -1)
        for p in split(img, plan)]
    with pytest.raises(TilingError, match="does not match"):
        merge(pieces, plan, scale=1.0)


# ── refusing to guess ───────────────────────────────────────────────────────

def test_a_plan_for_the_wrong_frame_size_is_refused():
    """Silently tiling the wrong region of every frame is the worst outcome."""
    plan = build_plan(256, 256, rows=2, cols=2, overlap=32)
    with pytest.raises(TilingError, match="belongs to one frame size"):
        split(gradient(512, 512), plan)


def test_a_missing_tile_is_refused_rather_than_left_black():
    plan = build_plan(256, 256, rows=2, cols=2, overlap=32)
    pieces = split(gradient(256, 256), plan)
    with pytest.raises(TilingError, match="Every tile has to be returned"):
        merge(pieces[:-1], plan)


def test_an_ambiguous_tensor_layout_is_refused():
    """(1, 3, 8, 3) is equally readable as 3 rows of 8x3-channel pixels or 3
    channels of 8x3. Guessing tiles the channel axis and returns confident
    nonsense."""
    from nodes.tiling._core import _spatial_dims

    with pytest.raises(TilingError, match="both axes are small"):
        _spatial_dims(torch.randn(1, 3, 8, 3))


def test_an_unreadable_layout_is_refused():
    """Neither axis looks like a channel count."""
    from nodes.tiling._core import _spatial_dims

    with pytest.raises(TilingError, match="neither axis"):
        _spatial_dims(torch.randn(1, 128, 64, 64))


def test_an_ordinary_image_and_latent_are_read_correctly():
    """The check must not become so strict that real inputs are refused."""
    from nodes.tiling._core import _spatial_dims

    assert _spatial_dims(torch.randn(1, 512, 768, 3)) == (512, 768)   # IMAGE
    assert _spatial_dims(torch.randn(1, 4, 64, 96)) == (64, 96)       # LATENT
    assert _spatial_dims(torch.randn(1, 16, 64, 96)) == (64, 96)      # 16-ch
    assert _spatial_dims(torch.randn(1, 16, 9, 64, 96)) == (64, 96)   # video


def test_a_nonsense_frame_size_is_named():
    with pytest.raises(TilingError, match="must be positive"):
        build_plan(0, 100)


def test_a_negative_overlap_is_named():
    with pytest.raises(TilingError, match="cannot be negative"):
        build_plan(256, 256, overlap=-8)


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_states_the_cost_of_the_overlap():
    text = describe_plan(build_plan(1080, 1920, rows=3, cols=4, overlap=96))
    assert "more area is processed" in text
    assert "12 tiles" in text


def test_the_report_states_the_one_plan_rule():
    text = describe_plan(build_plan(512, 512, rows=2, cols=2, overlap=64))
    assert "ONCE" in text and "seam that moves" in text


def test_a_narrow_overlap_is_warned_about():
    text = describe_plan(build_plan(1024, 1024, rows=2, cols=2, overlap=8))
    assert "narrow overlap" in text


def test_a_wasteful_plan_is_warned_about():
    text = describe_plan(build_plan(512, 512, rows=4, cols=4, overlap=100))
    assert "twice the frame area" in text


def test_redundancy_is_zero_for_a_single_tile():
    assert build_plan(256, 256, rows=1, cols=1).redundancy == pytest.approx(0.0)


def test_every_blend_mode_is_offered_and_works():
    for mode in BLEND_MODES:
        plan = build_plan(256, 256, rows=2, cols=2, overlap=32, blend=mode)
        assert len(split(gradient(256, 256), plan)) == 4
