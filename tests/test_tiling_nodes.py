"""The tiling nodes, end to end.

The unit tests prove the geometry and the lock separately. These prove the
CHAIN, because that is where a tiled upscale actually goes wrong: split, hand
the tiles to something that refines them, merge, lock. A seam or a drifting
face only appears once all four are in a row.

The "refiner" here is deliberately hostile - it adds detail AND shifts the
picture - because a refiner that behaves proves nothing.

CPU-only, torch only, no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes.tiling._core import TilingError  # noqa: E402
from nodes.tiling._identity import IdentityLockError, drift  # noqa: E402
from nodes.tiling._nodes import (  # noqa: E402
    IdentityLockMEC,
    TileMergeMEC,
    TilePlanMEC,
    TileSplitMEC,
)


def clip(h=256, w=384, frames=1, cx=192):
    """A face-like blob on a gradient, over N frames."""
    ys = torch.arange(h).view(1, h, 1, 1).float()
    xs = torch.arange(w).view(1, 1, w, 1).float()
    ramp = 0.2 + 0.3 * (xs / w) + 0.2 * (ys / h)
    d = ((ys - h / 2) ** 2 + (xs - cx) ** 2).sqrt()
    blob = 0.4 * torch.exp(-(d / 28.0) ** 2)
    return (ramp + blob).repeat(frames, 1, 1, 3).float().clamp(0, 1)


def add_detail(x, seed=0, amount=0.05):
    g = torch.Generator().manual_seed(seed)
    return (x + (torch.rand(x.shape, generator=g) - 0.5) * amount).clamp(0, 1)


def plan_for(img, **kw):
    args = dict(mode="tile count", tile_size=1024, rows=2, cols=2,
                overlap=64, blend="cosine")
    args.update(kw)
    return TilePlanMEC().plan(image=img, **args)


# ── the plan node ───────────────────────────────────────────────────────────

def test_a_plan_is_built_from_the_image_size():
    img = clip()
    plan, info = plan_for(img)
    assert (plan.height, plan.width) == (256, 384)
    assert len(plan.tiles) == 4
    assert "4 tiles" in info


def test_tile_size_mode_picks_its_own_counts():
    plan, info = plan_for(clip(512, 512), mode="tile size", tile_size=256,
                          overlap=32)
    assert len(plan.tiles) > 1
    assert all(t.height <= 256 and t.width <= 256 for t in plan.tiles)


def test_the_plan_does_not_depend_on_which_frame_it_saw():
    """One plan for the clip. Built from frame 0 or frame 50, it must be the
    same - otherwise the seams move when the plan is rebuilt."""
    a, _ = plan_for(clip(frames=1))
    b, _ = plan_for(clip(frames=8))
    assert a.tiles == b.tiles


# ── split and merge ─────────────────────────────────────────────────────────

def test_split_emits_every_tile_of_every_frame():
    img = clip(frames=5)
    plan, _ = plan_for(img)
    tiles, count, info = TileSplitMEC().split(image=img, tile_plan=plan)
    assert count == 4
    assert tiles.shape[0] == 4 * 5, "tiles x frames did not come out"
    assert "TILE-MAJOR" in info


def test_the_batch_is_tile_major():
    """Every frame of tile 1, then every frame of tile 2. A video refiner
    reading the batch as a sequence must see one tile's whole time range
    contiguously, or it deflickers across tiles instead of across time."""
    frames = 4
    img = clip(frames=frames)
    # make each frame distinguishable
    for f in range(frames):
        img[f] = img[f] * 0 + (f + 1) / 10.0
    plan, _ = plan_for(img)
    tiles, n, _ = TileSplitMEC().split(image=img, tile_plan=plan)

    for t in range(n):
        block = tiles[t * frames:(t + 1) * frames]
        values = [round(float(block[f].mean()), 3) for f in range(frames)]
        assert values == [0.1, 0.2, 0.3, 0.4], \
            f"tile {t} frames came out {values} - the batch is not tile-major"


def test_a_round_trip_with_no_refinement_returns_the_clip():
    """Nothing in between, so anything but the input back means the chain is
    lossy - and a lossy chain applied over 124 frames is a shot that breathes."""
    img = clip(frames=3)
    plan, _ = plan_for(img, overlap=64)
    tiles, _, _ = TileSplitMEC().split(image=img, tile_plan=plan)
    out, info = TileMergeMEC().merge(tiles=tiles, tile_plan=plan)
    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-5), \
        f"max error {float((out - img).abs().max())}"


def test_there_is_no_seam_on_a_flat_field():
    """Any variation across a flat grey IS the seam."""
    flat = torch.full((2, 256, 384, 3), 0.37)
    plan, _ = plan_for(flat, overlap=96)
    tiles, _, _ = TileSplitMEC().split(image=flat, tile_plan=plan)
    out, _ = TileMergeMEC().merge(tiles=tiles, tile_plan=plan)
    assert float(out.max() - out.min()) < 1e-5


def test_an_upscaling_refiner_is_handled():
    img = clip()
    plan, _ = plan_for(img)
    tiles, _, _ = TileSplitMEC().split(image=img, tile_plan=plan)
    bigger = torch.nn.functional.interpolate(
        tiles.movedim(-1, 1), scale_factor=2, mode="bilinear",
        align_corners=False).movedim(1, -1)
    out, info = TileMergeMEC().merge(tiles=bigger, tile_plan=plan)
    assert out.shape == (1, 512, 768, 3)
    assert "2.000x" in info


def test_a_refiner_that_changed_the_tile_count_is_named():
    img = clip()
    plan, _ = plan_for(img)
    tiles, _, _ = TileSplitMEC().split(image=img, tile_plan=plan)
    with pytest.raises(TilingError, match="same count"):
        TileMergeMEC().merge(tiles=tiles[:-1], tile_plan=plan)


def test_a_non_uniform_scale_is_named_rather_than_guessed():
    """Squashed tiles would be merged at a wrong ratio and every tile after
    the first would land shifted."""
    img = clip()
    plan, _ = plan_for(img)
    tiles, _, _ = TileSplitMEC().split(image=img, tile_plan=plan)
    squashed = torch.nn.functional.interpolate(
        tiles.movedim(-1, 1), scale_factor=(2.0, 1.0), mode="nearest"
    ).movedim(1, -1)
    with pytest.raises(TilingError, match="not a uniform scale"):
        TileMergeMEC().merge(tiles=squashed, tile_plan=plan)


def test_the_wrong_plan_for_the_image_is_named():
    plan, _ = plan_for(clip(256, 384))
    with pytest.raises(TilingError, match="same footage"):
        TileSplitMEC().split(image=clip(512, 512), tile_plan=plan)


def test_something_that_is_not_a_plan_is_named():
    with pytest.raises(TilingError, match="not a tile plan"):
        TileSplitMEC().split(image=clip(), tile_plan={"rows": 2})


# ── the whole chain ─────────────────────────────────────────────────────────

def hostile_refiner(tiles, shift=12, seed=1):
    """Adds detail AND shifts the picture - exactly what a real sampler does to
    a face, and what the lock has to undo. A refiner that behaves proves
    nothing.

    12px is chosen from measurement. Across the whole chain, radius 6, two
    corrections:

        shift  4 -> raw drift 0.0082, locked 0.00049   (94% removed)
        shift 12 -> raw drift 0.0252, locked 0.00045   (98% removed)
        shift 24 -> raw drift 0.0490, locked 0.00058   (99% removed)

    The locked figure barely moves however hard the refiner pushes: the lock
    does not reduce drift proportionally, it removes it. 4px was simply too
    gentle to clear the control case's own threshold.
    """
    rolled = torch.roll(tiles, shifts=shift, dims=2)
    return add_detail(rolled, seed=seed, amount=0.06)


def test_the_chain_without_a_lock_lets_the_face_move():
    """The control case. If this did NOT drift, the lock would be untested."""
    img = clip(frames=2)
    plan, _ = plan_for(img, overlap=96)
    tiles, _, _ = TileSplitMEC().split(image=img, tile_plan=plan)
    out, _ = TileMergeMEC().merge(tiles=hostile_refiner(tiles), tile_plan=plan)
    assert drift(img, out, radius=12.0) > 0.01, \
        "the hostile refiner is not hostile enough to test anything"


def test_the_chain_with_a_lock_holds_the_face():
    """THE end-to-end promise: every pixel refined, nothing moved."""
    img = clip(frames=2)
    plan, _ = plan_for(img, overlap=96)
    tiles, _, _ = TileSplitMEC().split(image=img, tile_plan=plan)
    refined, _ = TileMergeMEC().merge(tiles=hostile_refiner(tiles),
                                      tile_plan=plan)

    locked, dmap, moved, info = IdentityLockMEC().lock(
        original=img, refined=refined, radius=6.0, strength=1.0, corrections=2)

    raw = drift(img, refined, radius=6.0)
    # The measured ratio is about 0.018. A tenth is a comfortable margin that
    # still fails loudly if the lock ever stops working.
    assert moved < raw / 10.0, f"the lock barely helped: {raw} -> {moved}"
    assert locked.shape == img.shape

    # and it is not a no-op - the detail actually arrived
    assert float((locked - img).abs().mean()) > 1e-3


def test_the_locked_result_still_carries_the_refinement():
    img = clip(frames=1)
    plan, _ = plan_for(img)
    tiles, _, _ = TileSplitMEC().split(image=img, tile_plan=plan)
    refined, _ = TileMergeMEC().merge(
        tiles=add_detail(tiles, amount=0.08), tile_plan=plan)
    locked, _, _, _ = IdentityLockMEC().lock(
        original=img, refined=refined, radius=8.0, strength=1.0, corrections=2)

    # closer to the refinement than to a plain copy of the plate
    assert float((locked - refined).abs().mean()) < \
        float((img - refined).abs().mean())


# ── the lock node ───────────────────────────────────────────────────────────

def test_the_lock_reports_a_number_and_a_map():
    img = clip()
    refined = add_detail(img)
    out, dmap, moved, info = IdentityLockMEC().lock(
        original=img, refined=refined, radius=8.0, strength=1.0,
        corrections=2, want_drift_map=True)
    assert dmap.shape == (1, 256, 384)
    assert isinstance(moved, float)
    assert "Structure moved" in info


def test_the_drift_map_output_exists_even_when_not_asked_for():
    """A MASK output that is sometimes absent breaks any graph that wires it."""
    img = clip()
    _, dmap, _, _ = IdentityLockMEC().lock(
        original=img, refined=img, radius=8.0, strength=1.0, corrections=2,
        want_drift_map=False)
    assert dmap.shape == (1, 256, 384)
    assert float(dmap.max()) == 0.0


def test_a_size_mismatch_tells_you_to_upscale_the_plate():
    with pytest.raises(IdentityLockError, match="upscale the plate"):
        IdentityLockMEC().lock(original=clip(256, 384), refined=clip(512, 768),
                               radius=8.0, strength=1.0, corrections=2)


def test_the_locked_drift_barely_grows_with_how_hard_the_refiner_pushes():
    """The lock does not reduce drift proportionally - it removes it. A refiner
    that shifts six times as far lands at almost the same locked drift, which
    is what makes the guarantee worth stating at all."""
    img = clip(frames=1)
    plan, _ = plan_for(img, overlap=96)
    tiles, _, _ = TileSplitMEC().split(image=img, tile_plan=plan)

    locked = []
    for shift in (4, 24):
        refined, _ = TileMergeMEC().merge(
            tiles=hostile_refiner(tiles, shift=shift), tile_plan=plan)
        _, _, moved, _ = IdentityLockMEC().lock(
            original=img, refined=refined, radius=6.0, strength=1.0,
            corrections=2)
        locked.append(moved)

    assert locked[1] < locked[0] * 3.0, (
        f"a 6x harder shift gave {locked[1] / locked[0]:.1f}x the drift")


def test_strength_zero_with_a_mask_locks_only_the_mask():
    """The configuration most shots want, and the one that used to be a silent
    no-op: nothing locked anywhere except the face."""
    img = clip(cx=192)
    refined = hostile_refiner(img.unsqueeze(0)[0], shift=6)

    mask = torch.zeros(1, 256, 384)
    mask[:, 96:160, 160:224] = 1.0            # over the blob

    out, _, _, info = IdentityLockMEC().lock(
        original=img, refined=refined, radius=8.0, strength=0.0,
        corrections=2, protect=mask)

    inside = (slice(None), slice(110, 146), slice(174, 210), slice(None))
    assert float((out[inside] - img[inside]).abs().mean()) < \
        float((refined[inside] - img[inside]).abs().mean()), \
        "the masked region was not locked"
    # far from the mask, the refinement passes through untouched
    assert torch.allclose(out[:, 0:20, 0:20, :], refined[:, 0:20, 0:20, :],
                          atol=1e-5)
    assert "protection mask" in info


# ── registration ────────────────────────────────────────────────────────────

def test_every_node_declares_the_contract_the_pack_expects():
    from nodes.tiling import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert set(NODE_CLASS_MAPPINGS) == set(NODE_DISPLAY_NAME_MAPPINGS)
    for name, cls in NODE_CLASS_MAPPINGS.items():
        assert isinstance(cls.RETURN_TYPES, tuple), f"{name} RETURN_TYPES"
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES), name
        assert hasattr(cls, cls.FUNCTION), f"{name} missing {cls.FUNCTION}"
        assert cls.CATEGORY.startswith("C2C/"), f"{name} category {cls.CATEGORY}"
        assert cls.DESCRIPTION, f"{name} has no description"


def test_no_node_caches_on_a_constant():
    """IS_CHANGED must actually vary with the inputs, or a second run with
    different footage silently returns the first run's result."""
    from nodes.tiling import NODE_CLASS_MAPPINGS

    a, b = clip(cx=100), clip(cx=200)
    plan_a, _ = plan_for(a)
    assert TileSplitMEC.IS_CHANGED(a, plan_a) != TileSplitMEC.IS_CHANGED(b, plan_a)
    assert IdentityLockMEC.IS_CHANGED(a, b, radius=8.0) != \
        IdentityLockMEC.IS_CHANGED(a, a, radius=8.0)
    for name, cls in NODE_CLASS_MAPPINGS.items():
        assert hasattr(cls, "IS_CHANGED"), f"{name} has no IS_CHANGED"
