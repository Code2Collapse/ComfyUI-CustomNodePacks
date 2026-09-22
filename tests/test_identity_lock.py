"""The identity lock: refine every pixel, move nothing.

This is the node the whole request rests on - "all the pixels get refined but
faces must not change" - so the tests do not check that it runs. They check
that the promise HOLDS, by constructing a refinement that deliberately moves a
face and measuring whether it still moved afterwards.

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

from nodes.tiling._identity import (  # noqa: E402
    IdentityLockError,
    describe,
    drift,
    drift_map,
    gaussian_kernel,
    identity_lock,
    low_pass,
)


def face(h=128, w=128, cx=64, cy=64, r=24):
    """A soft blob standing in for a face: a low-frequency shape whose
    POSITION is the thing that must not move."""
    ys = torch.arange(h).view(1, h, 1, 1).float()
    xs = torch.arange(w).view(1, 1, w, 1).float()
    d = ((ys - cy) ** 2 + (xs - cx) ** 2).sqrt()
    return (0.3 + 0.5 * torch.exp(-(d / r) ** 2)).repeat(1, 1, 1, 3)


def texture(h=128, w=128, seed=0):
    """High-frequency detail - the thing a refinement is supposed to add."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, 3, generator=g) * 0.08 - 0.04


# ── the low pass ────────────────────────────────────────────────────────────

def test_the_kernel_sums_to_one():
    """A kernel that does not sum to 1 changes the image's brightness, which
    then reads as the lock darkening the shot."""
    for r in (1.0, 4.0, 8.0, 32.0):
        assert float(gaussian_kernel(r).sum()) == pytest.approx(1.0, abs=1e-6)


def test_the_kernel_is_wide_enough_not_to_clip_its_tail():
    """A truncated Gaussian has a shoulder, and a shoulder in the low-pass
    becomes a halo around every hard edge in the result."""
    k = gaussian_kernel(8.0)
    assert k.numel() >= 2 * 3 * 8, f"kernel is only {k.numel()} wide for sigma 8"
    assert float(k[0]) < 0.01, "the tail is still significant at the edge"


def test_a_low_pass_preserves_the_average():
    img = face()
    assert float(low_pass(img, 8.0).mean()) == pytest.approx(
        float(img.mean()), abs=1e-3)


def test_a_low_pass_removes_high_frequency_detail():
    detailed = face() + texture()
    assert float(low_pass(detailed, 8.0).std()) < float(detailed.std())


def test_a_zero_radius_is_a_no_op():
    img = face()
    assert torch.equal(low_pass(img, 0.0), img)


def test_the_frame_edge_is_not_darkened():
    """Padding the blur with black darkens the rim, which shows in the
    recombined result as a dark border."""
    flat = torch.full((1, 64, 64, 3), 0.5)
    out = low_pass(flat, 8.0)
    assert float(out.min()) == pytest.approx(0.5, abs=1e-4)
    assert float(out.max()) == pytest.approx(0.5, abs=1e-4)


# ── THE promise ─────────────────────────────────────────────────────────────

def test_a_refinement_that_moves_the_face_is_undone():
    """THE test. A refinement that shifts the face by 6px is exactly the
    failure being guarded against: over 124 frames that is a face that swims.
    After the lock, the face must be back where it started."""
    original = face(cx=64)
    moved = face(cx=70) + texture()          # shifted AND detailed

    assert drift(original, moved, radius=12.0) > 0.02, \
        "the fixture does not actually move the face"

    locked = identity_lock(original, moved, radius=12.0, strength=1.0)
    assert drift(original, locked, radius=12.0) < 0.005, \
        "the face was still allowed to move"


def test_the_refinement_detail_still_arrives():
    """The lock must not be a no-op dressed up as a guarantee: if it threw the
    refinement away entirely the face would also not move, and the node would
    be useless."""
    original = face()
    refined = face() + texture()

    locked = identity_lock(original, refined, radius=12.0, strength=1.0)
    added = float((locked - original).abs().mean())
    assert added > 1e-3, "no detail came through at all - the lock is a no-op"

    # and the detail that came through is the refinement's, not noise
    assert float((locked - refined).abs().mean()) < added


def test_structure_and_detail_are_actually_separated():
    """The arithmetic identity the whole thing rests on: low(original) plus
    the refinement's high band."""
    original = face(cx=64)
    refined = face(cx=70) + texture()
    # corrections=0 so this compares against the PLAIN split; the correction
    # passes are tested separately as the thing that tightens it.
    locked = identity_lock(original, refined, radius=12.0, strength=1.0,
                           corrections=0)

    expected = low_pass(original, 12.0) + (refined - low_pass(refined, 12.0))
    assert torch.allclose(locked, expected, atol=1e-5)


def test_zero_strength_passes_the_refinement_straight_through():
    original, refined = face(), face(cx=70) + texture()
    assert torch.allclose(identity_lock(original, refined, strength=0.0),
                          refined, atol=1e-6)


def test_strength_fades_between_the_two():
    original = face(cx=64)
    refined = face(cx=72) + texture()
    drifts = [drift(original,
                    identity_lock(original, refined, radius=12.0, strength=s),
                    radius=12.0)
              for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert drifts == sorted(drifts, reverse=True), \
        f"more lock did not mean less drift: {drifts}"


def test_an_identical_refinement_changes_nothing():
    img = face()
    assert torch.allclose(identity_lock(img, img, radius=8.0), img, atol=1e-5)


# ── the radius ──────────────────────────────────────────────────────────────

def test_a_smaller_radius_locks_harder():
    """The trade, the right way round. A small radius puts almost the whole
    image in the low band, so almost everything is locked. A large one leaves
    only the coarsest shapes locked, so more refinement arrives - and
    medium-scale movement arrives with it.

    Measured: radius 2 drifts ~0.0000002, radius 24 drifts ~0.018.
    """
    original = face(cx=64)
    refined = face(cx=70)
    small = drift(original, identity_lock(original, refined, radius=2.0),
                  radius=12.0)
    large = drift(original, identity_lock(original, refined, radius=24.0),
                  radius=12.0)
    assert small < large, f"radius 2 drifted {small}, radius 24 drifted {large}"


def test_a_larger_radius_lets_more_refinement_through():
    """The other half of the same trade - a hard lock is not free, it costs
    the detail that was the point of refining."""
    original = face(cx=64)
    refined = face(cx=64) + texture()
    tight = identity_lock(original, refined, radius=2.0, strength=1.0)
    loose = identity_lock(original, refined, radius=24.0, strength=1.0)
    assert float((loose - original).abs().mean()) > \
        float((tight - original).abs().mean()), \
        "a larger radius did not let more detail through"


def test_corrections_tighten_the_lock():
    original = face(cx=64)
    refined = face(cx=70) + texture()
    plain = drift(original,
                  identity_lock(original, refined, radius=12.0, corrections=0),
                  radius=12.0)
    fixed = drift(original,
                  identity_lock(original, refined, radius=12.0, corrections=2),
                  radius=12.0)
    assert fixed < plain / 2.0, \
        f"corrections barely helped: {plain} -> {fixed}"


def test_corrections_barely_cost_any_detail():
    """A tighter lock that threw the refinement away would be no use."""
    original = face(cx=64)
    refined = face(cx=64) + texture()
    plain = identity_lock(original, refined, radius=12.0, corrections=0)
    fixed = identity_lock(original, refined, radius=12.0, corrections=4)
    kept_plain = float((plain - original).abs().mean())
    kept_fixed = float((fixed - original).abs().mean())
    assert kept_fixed > kept_plain * 0.9, \
        f"corrections cost {100 * (1 - kept_fixed / kept_plain):.0f}% of the detail"


def test_a_negative_radius_is_refused():
    with pytest.raises(IdentityLockError, match="cannot be negative"):
        identity_lock(face(), face(), radius=-1.0)


def test_a_strength_outside_zero_to_one_is_refused():
    with pytest.raises(IdentityLockError, match="between 0 and 1"):
        identity_lock(face(), face(), strength=1.5)


def test_mismatched_sizes_are_refused_with_the_fix():
    """Comparing different pixels would report nonsense drift and lock the
    wrong structure in."""
    with pytest.raises(IdentityLockError, match="resize the original"):
        identity_lock(face(128, 128), face(256, 256))


# ── the protection mask ─────────────────────────────────────────────────────

def test_a_protection_mask_locks_inside_and_frees_outside():
    """The shot's real shape: a hard lock on the actor, a free hand on the
    background, in one pass."""
    original = face(cx=64)
    refined = face(cx=72) + texture()

    mask = torch.zeros(1, 128, 128)
    mask[:, 32:96, 32:96] = 1.0                 # over the face

    # strength=0 WITH a mask: nothing locked anywhere except inside the mask.
    # This used to return early and ignore the mask entirely, which made the
    # single most useful configuration a silent no-op.
    out = identity_lock(original, refined, radius=12.0, strength=0.0,
                        protect=mask)

    inside_orig = original[:, 40:88, 40:88, :]
    inside_out = out[:, 40:88, 40:88, :]
    inside_ref = refined[:, 40:88, 40:88, :]
    assert float((inside_out - inside_orig).abs().mean()) < \
        float((inside_ref - inside_orig).abs().mean()), \
        "the masked region was not locked"

    # outside, with strength 0, the refinement passes through untouched
    assert torch.allclose(out[:, 0:16, 0:16, :], refined[:, 0:16, 0:16, :],
                          atol=1e-5)


def test_no_mask_means_the_global_strength_applies_everywhere():
    original, refined = face(cx=64), face(cx=72) + texture()
    a = identity_lock(original, refined, radius=12.0, strength=1.0)
    b = identity_lock(original, refined, radius=12.0, strength=1.0, protect=None)
    assert torch.allclose(a, b)


# ── measurement ─────────────────────────────────────────────────────────────

def test_drift_ignores_added_detail():
    """Comparing the images directly would report every added pore as drift,
    which is the refinement working. Only the band under the detail matters."""
    original = face()
    detailed = face() + texture()
    assert drift(original, detailed, radius=12.0) < 0.01


def test_drift_catches_a_moved_face():
    assert drift(face(cx=64), face(cx=80), radius=12.0) > 0.02


def test_drift_of_an_image_against_itself_is_zero():
    img = face()
    assert drift(img, img) == pytest.approx(0.0, abs=1e-6)


def test_drift_on_a_flat_image_does_not_divide_by_zero():
    flat = torch.full((1, 64, 64, 3), 0.5)
    assert drift(flat, flat) == 0.0


def test_the_drift_map_shows_where_it_moved():
    """A number says the lock did not hold; the map says where, which is the
    difference between knowing and finding."""
    m = drift_map(face(cx=48), face(cx=80), radius=12.0)
    assert m.shape == (1, 128, 128)
    assert float(m.max()) == pytest.approx(1.0, abs=1e-5)
    # the movement is between the two positions, not at the frame corner
    assert float(m[0, 64, 48:80].max()) > float(m[0, 5, 5])


def test_the_drift_map_of_an_unchanged_image_is_empty():
    img = face()
    assert float(drift_map(img, img).max()) == pytest.approx(0.0, abs=1e-6)


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_states_the_measured_drift():
    original, refined = face(cx=64), face(cx=64) + texture()
    locked = identity_lock(original, refined, radius=12.0, strength=1.0)
    text = describe(original, locked, radius=12.0, strength=1.0, protected=False)
    assert "Structure moved" in text
    assert "effectively unchanged" in text


def test_the_report_warns_when_the_lock_did_not_hold():
    """A radius too LARGE to catch the movement: everything below it passes
    through as 'detail', including the shift.

    The warning has to measure COARSER than the lock. At the lock's own radius
    the correction passes force the low band to match, so it always reads ~0
    and the warning could never fire - a check that can only pass is not a
    check.
    """
    original, refined = face(cx=48), face(cx=88)
    weak = identity_lock(original, refined, radius=40.0, strength=1.0)
    text = describe(original, weak, radius=40.0, strength=1.0, protected=False)
    assert "WARNING" in text
    assert "SMALLER radius" in text


def test_the_report_measures_at_two_scales():
    original = face()
    text = describe(original, original, radius=8.0, strength=1.0, protected=False)
    assert "Measured coarser" in text


def test_the_report_always_states_the_cost():
    """The trade is real and must not be discoverable only by surprise: a
    warped ear stays warped."""
    original = face()
    text = describe(original, original, radius=8.0, strength=1.0, protected=False)
    assert "CANNOT" in text
    assert "warped ear" in text or "structural" in text


def test_the_report_mentions_a_mask_when_one_is_used():
    original = face()
    text = describe(original, original, radius=8.0, strength=1.0, protected=True)
    assert "protection mask" in text


# ── layout ──────────────────────────────────────────────────────────────────

def test_bchw_input_works_too():
    """Latents are BCHW; the lock should not care."""
    a = torch.rand(1, 4, 64, 64)
    b = a + torch.rand(1, 4, 64, 64) * 0.05
    out = identity_lock(a, b, radius=8.0, strength=1.0)
    assert out.shape == a.shape


def test_a_clip_locks_frame_by_frame():
    original = face().repeat(6, 1, 1, 1)
    refined = (face(cx=72) + texture()).repeat(6, 1, 1, 1)
    out = identity_lock(original, refined, radius=12.0, strength=1.0)
    assert out.shape == original.shape
    assert drift(original, out, radius=12.0) < 0.005
