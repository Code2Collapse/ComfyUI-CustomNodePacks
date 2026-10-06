"""Diagnosing a decode that came back reddish, oversaturated or hard.

The node's value is the DIAGNOSIS, not the sliders - "reddish and
oversaturated" has four different causes that need four different fixes, and
getting the wrong one makes the shot worse. So these tests build each fault
deliberately and check it is identified as that fault and not another.

The one that matters most is clipping, because it cannot be undone. A tool
that claims to recover a clipped highlight is inventing detail, and there is a
test here that the report says so.

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

from nodes.vae_clean import (  # noqa: E402
    VAECleanError,
    VAECleanMEC,
    balance,
    ciede2000,
    clean_chroma,
    contrast,
    decode_noise,
    desaturate,
    linear_to_srgb,
    luma,
    measure,
    write_back_unchanged,
    roundtrip_report,
    soften_contrast,
    saturation,
    srgb_to_lab,
    srgb_to_linear,
)


def plate(h=64, w=64, seed=0):
    """A neutral scene: smooth gradients, real colour, a little grain.

    Built in LINEAR and converted at the end, which is the point. Two earlier
    versions of this fixture were wrong in ways worth recording:

      * per-pixel uniform noise. Maximally high-frequency in every channel, so
        a CLEAN plate measured 0.34 chroma noise and its channel means already
        differed by 1%. An image is mostly low-frequency; a fixture that is
        not cannot tell a fault from itself.
      * a zero-mean tint applied in sRGB. The node measures in LINEAR, and the
        curve between them is not linear - so a tint that averages to nothing
        on screen averages to a 4% red cast in light, and the plate failed its
        own "a clean plate has no cast" test.

    Colour and cast are different things: colour is chroma that VARIES across
    the frame, a cast is chroma in the channel MEANS. This is colourful
    everywhere and neutral on average.
    """
    g = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0.06, 0.34, h).view(1, h, 1, 1)
    xs = torch.linspace(-0.02, 0.02, w).view(1, 1, w, 1)
    lin = (ys + xs).repeat(1, 1, 1, 3)

    for cy, cx, r, amp in ((0.3, 0.35, 0.18, 0.06), (0.7, 0.68, 0.22, -0.04)):
        yy = torch.linspace(0, 1, h).view(1, h, 1, 1)
        xx = torch.linspace(0, 1, w).view(1, 1, w, 1)
        d = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()
        lin = lin + amp * torch.exp(-(d / r) ** 2).repeat(1, 1, 1, 3)

    t = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1)
    tint = torch.cat([t, -t, (2.0 * t.abs() - 1.0)], dim=-1) * 0.05
    lin = lin + tint
    lin = lin + (torch.rand(1, h, w, 3, generator=g) - 0.5) * 0.002
    return linear_to_srgb(lin.clamp(0.004, 0.75)).clamp(0.0, 1.0)


def with_cast(img, r=1.0, gr=1.0, b=1.0):
    """A per-channel gain applied in LINEAR, the way a decoder bias arrives."""
    lin = srgb_to_linear(img)
    lin = lin * torch.tensor([r, gr, b])
    return linear_to_srgb(lin).clamp(0.0, 1.0)


def run(img, **kw):
    args = dict(balance_mode="off", balance_strength=1.0, saturation=1.0,
                contrast_restore=0.0, chroma_cleanup=0.0)
    args.update(kw)
    return VAECleanMEC().clean(image=img, **args)


# ── transfer ────────────────────────────────────────────────────────────────

def test_srgb_round_trips():
    x = torch.linspace(0, 1, 1024).view(1, 32, 32, 1).repeat(1, 1, 1, 3)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_the_toe_is_the_real_piecewise_curve_not_a_power():
    """A 2.2 power shifts the darkest values, and the black point is exactly
    what this node measures."""
    dark = torch.tensor([0.01, 0.02, 0.03])
    exact = srgb_to_linear(dark)
    power = dark ** 2.2
    assert float((exact - power).abs().max()) > 1e-4, \
        "the toe is being approximated; black-point measurement will be wrong"


def test_luma_uses_rec709_weights():
    green = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    blue = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    assert float(luma(green)) > float(luma(blue)) * 5, \
        "green is not weighted far above blue - these are not Rec.709 weights"


# ── detecting each fault as itself ──────────────────────────────────────────

def test_a_red_cast_is_measured_as_a_red_cast():
    m = measure(with_cast(plate(), r=1.35))
    r, g, b = m["cast"]
    assert r > 0.03 and r > g and r > b


def test_a_clean_plate_shows_no_cast_worth_correcting():
    """The floor matters: 'correcting' measurement noise adds a cast."""
    m = measure(plate())
    assert max(abs(c) for c in m["cast"]) < 0.02


def test_oversaturation_is_measured_separately_from_a_cast():
    """They are different faults with different fixes, so a saturated image
    must not read as a colour cast."""
    lin = srgb_to_linear(plate())
    y = luma(lin)
    hot = linear_to_srgb((y + (lin - y) * 2.2).clamp(0, 1))
    m = measure(hot)
    assert m["saturation"] > measure(plate())["saturation"] * 1.4
    assert max(abs(c) for c in m["cast"]) < 0.05, \
        "saturation is being misreported as a cast"


def test_saturation_is_scale_free():
    """Exposure must not read as saturation, or every bright shot looks
    oversaturated."""
    img = plate()
    dim = linear_to_srgb(srgb_to_linear(img) * 0.5)
    a = measure(img)["saturation"]
    b = measure(dim)["saturation"]
    assert abs(a - b) < a * 0.25, f"exposure moved saturation {a:.3f} -> {b:.3f}"


def test_crushed_blacks_show_in_the_black_point():
    img = plate()
    crushed = ((img - 0.25).clamp(0, 1) / 0.75).clamp(0, 1)
    assert measure(crushed)["black"] < measure(img)["black"]


def test_chroma_noise_is_detected_and_luma_detail_is_not_mistaken_for_it():
    """Luma high-frequency is detail you want; chroma high-frequency at this
    scale is the decoder. Confusing them means smoothing the picture."""
    img = plate()
    g = torch.Generator().manual_seed(3)

    lin = srgb_to_linear(img)
    y = luma(lin)
    speckle = (torch.rand(lin.shape, generator=g) - 0.5) * 0.10
    chroma_noisy = linear_to_srgb((lin + (speckle - speckle.mean(-1, keepdim=True))).clamp(0, 1))

    luma_detail = linear_to_srgb(
        (lin + (torch.rand(y.shape, generator=g) - 0.5) * 0.10).clamp(0, 1))

    base = decode_noise(srgb_to_linear(img))
    assert decode_noise(srgb_to_linear(chroma_noisy)) > base * 2
    assert decode_noise(srgb_to_linear(luma_detail)) < \
        decode_noise(srgb_to_linear(chroma_noisy)), \
        "luma detail reads as chroma noise; cleanup would soften the picture"


# ── corrections ─────────────────────────────────────────────────────────────

def test_grey_world_removes_a_cast():
    cast = with_cast(plate(), r=1.4)
    lin, _ = balance(srgb_to_linear(cast), "grey world", reference=None,
                     strength=1.0)
    out = measure(linear_to_srgb(lin.clamp(0, 1)))
    assert max(abs(c) for c in out["cast"]) < 0.02


def test_balance_strength_fades_the_correction():
    cast = with_cast(plate(), r=1.4)
    before = max(abs(c) for c in measure(cast)["cast"])
    got = []
    for s in (0.0, 0.5, 1.0):
        lin, _ = balance(srgb_to_linear(cast), "grey world", reference=None,
                         strength=s)
        got.append(max(abs(c) for c in measure(linear_to_srgb(lin.clamp(0, 1)))["cast"]))
    assert got == sorted(got, reverse=True)
    assert got[0] == pytest.approx(before, abs=0.01)


def test_white_point_survives_a_scene_that_is_genuinely_one_colour():
    """Grey world throws a sunset away. White point does not, because a red
    room still has a neutral highlight."""
    img = plate() * torch.tensor([1.0, 0.45, 0.30])       # a red-lit scene
    img[:, :8, :8, :] = 0.95                              # with a neutral highlight
    img = img.clamp(0, 1)

    grey, _ = balance(srgb_to_linear(img), "grey world", reference=None, strength=1.0)
    white, _ = balance(srgb_to_linear(img), "white point", reference=None, strength=1.0)

    red_after_grey = measure(linear_to_srgb(grey.clamp(0, 1)))["cast"][0]
    red_after_white = measure(linear_to_srgb(white.clamp(0, 1)))["cast"][0]
    assert red_after_white > red_after_grey, \
        "white point destroyed the scene's colour the way grey world does"


def test_match_reference_copies_the_plate():
    ref = plate(seed=7)
    off = with_cast(plate(seed=7), r=1.3, b=0.8)
    lin, _ = balance(srgb_to_linear(off), "match reference", reference=ref,
                     strength=1.0)
    a = measure(linear_to_srgb(lin.clamp(0, 1)))["means"]
    b = measure(ref)["means"]
    for x, y in zip(a, b):
        assert x == pytest.approx(y, rel=0.08)


def test_match_reference_without_one_is_refused_with_the_fix():
    with pytest.raises(VAECleanError, match="Wire the plate"):
        balance(srgb_to_linear(plate()), "match reference", reference=None,
                strength=1.0)


def test_desaturating_holds_brightness():
    """About luma, not toward grey - a naive RGB lerp darkens as it
    desaturates, which is why saturation sliders feel like exposure."""
    lin = srgb_to_linear(plate())
    out = desaturate(lin, 0.4)
    assert float(luma(out).mean()) == pytest.approx(float(luma(lin).mean()), rel=1e-4)


def test_full_desaturation_is_monochrome():
    out = desaturate(srgb_to_linear(plate()), 0.0)
    assert float((out[..., 0] - out[..., 1]).abs().max()) < 1e-5


def test_chroma_cleanup_leaves_luma_alone():
    """The whole reason it can go hard without softening the picture."""
    lin = srgb_to_linear(plate())
    out = clean_chroma(lin, 0.8)
    assert torch.allclose(luma(out), luma(lin), atol=1e-4)


def test_chroma_cleanup_actually_reduces_chroma_noise():
    g = torch.Generator().manual_seed(11)
    lin = srgb_to_linear(plate())
    speckle = (torch.rand(lin.shape, generator=g) - 0.5) * 0.12
    noisy = lin + (speckle - speckle.mean(-1, keepdim=True))
    assert decode_noise(clean_chroma(noisy, 0.8)) < decode_noise(noisy) * 0.6


def test_lifting_shadows_reduces_contrast_rather_than_increasing_it():
    """The first version of this normalised the range to [0,1], which
    INCREASES contrast - the exact opposite of the complaint it exists for.
    The black point going the wrong way is what caught it."""
    img = ((plate() - 0.25).clamp(0, 1) / 0.75).clamp(0, 1)   # crushed
    m = measure(img)
    out = measure(linear_to_srgb(soften_contrast(srgb_to_linear(img), 1.0).clamp(0, 1)))
    assert out["black"] > m["black"],         "the black point went DOWN - this is adding contrast, not softening it"


def test_lifting_shadows_cannot_clip():
    """It moves everything toward white and never past it, which a range
    stretch cannot promise."""
    out = soften_contrast(srgb_to_linear(plate()), 1.0)
    assert float(out.max()) <= 1.0 + 1e-6


def test_lifting_shadows_off_is_a_no_op():
    lin = srgb_to_linear(plate())
    assert torch.equal(soften_contrast(lin, 0.0), lin)


# ── the honesty tests ───────────────────────────────────────────────────────

def test_clipping_is_reported_and_called_unrecoverable():
    """THE one that matters. A clipped highlight is gone, and a tool that
    claims to bring it back is inventing detail."""
    img = (plate() * 2.4).clamp(0, 1)
    _, report, _, _ = run(img)
    assert "CLIPPED" in report
    assert "cannot be undone" in report
    assert "inventing detail" in report


def test_a_clean_plate_is_not_accused_of_clipping():
    _, report, _, _ = run(plate())
    assert "Nothing is clipped" in report


def test_oversaturation_points_at_cfg_rather_than_the_decoder():
    """Oversaturation plus hard contrast is the classic CFG signature. Saying
    so is more use than quietly desaturating a sampler setting."""
    lin = srgb_to_linear(plate())
    y = luma(lin)
    hot = linear_to_srgb((y + (lin - y) * 4.0).clamp(0, 1))
    _, report, _, _ = run(hot)
    assert "CFG" in report
    assert "treats the symptom" in report


def test_a_large_cast_warns_it_might_be_the_lighting():
    """A sunset measures exactly like a decoder cast. Neutralising it throws
    the shot away, so the report has to raise the possibility."""
    _, report, _, _ = run(with_cast(plate(), r=1.9, b=0.5))
    assert "sunset" in report or "red-lit" in report
    assert "EVERY shot" in report


def test_a_tiny_cast_is_called_noise_rather_than_corrected():
    _, report, _, _ = run(with_cast(plate(), r=1.002))
    assert "measurement noise" in report


def test_measuring_only_says_so():
    _, report, _, _ = run(plate())
    assert "nothing was changed" in report


# ── the node ────────────────────────────────────────────────────────────────

def test_the_node_returns_the_measurements_as_numbers():
    """So a graph can branch on them instead of a human reading prose."""
    img = with_cast(plate(), r=1.4)
    out, report, cast, sat = run(img)
    assert out.shape == img.shape
    assert cast > 0.03
    assert 0.0 < sat < 3.0


def test_defaults_change_nothing():
    """Dropping the node into a graph must be observation, not a look."""
    img = plate()
    out, _, _, _ = run(img)
    assert torch.allclose(out, img, atol=2e-3)


def test_the_full_chain_fixes_a_bad_decode():
    """Cast, oversaturation and crushed blacks at once - the actual complaint."""
    lin = srgb_to_linear(plate())
    y = luma(lin)
    bad = linear_to_srgb(
        ((y + (lin - y) * 1.9) * torch.tensor([1.35, 1.0, 0.85])).clamp(0, 1))

    before = measure(bad)
    out, _, _, _ = run(bad, balance_mode="grey world", saturation=0.62,
                       contrast_restore=0.5)
    after = measure(out)

    assert max(abs(c) for c in after["cast"]) < max(abs(c) for c in before["cast"])
    assert after["saturation"] < before["saturation"]


def test_a_non_image_is_refused():
    with pytest.raises(VAECleanError, match=r"\[B,H,W,C\]"):
        run(torch.rand(64, 64))


def test_it_is_registered():
    from nodes.vae_clean import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    assert set(NODE_CLASS_MAPPINGS) == set(NODE_DISPLAY_NAME_MAPPINGS)
    assert VAECleanMEC.DESCRIPTION
    assert VAECleanMEC.CATEGORY.startswith("C2C/")


# ── R4 restore unchanged ─────────────────────────────────────────────────────

def _synthetic_decode_plate(h=64, w=64, seed=0):
    ref = plate(h, w, seed)
    g = torch.Generator().manual_seed(seed + 1)
    noise = (torch.rand(ref.shape, generator=g) - 0.5) * 0.02
    edited = ref + noise
    cy, cx = h // 2, w // 2
    half = 12
    edited[:, cy - half:cy + half, cx - half:cx + half, :] = 0.9
    return ref, edited


def test_write_back_unchanged_outside_square_matches_reference():
    # An edit reaches 3 radii: the box mean spreads it by r, the dilation by r,
    # the feather by r. Outside that band every pixel is the reference, exactly.
    ref, edited = _synthetic_decode_plate(128, 128)
    radius = 8
    out, info = write_back_unchanged(edited, ref, threshold=0.01, radius=radius)
    cy, cx = 64, 64
    half = 12
    margin = 3 * radius + 1
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    gy0, gy1 = max(0, y0 - margin), min(ref.shape[1], y1 + margin)
    gx0, gx1 = max(0, x0 - margin), min(ref.shape[2], x1 + margin)
    outside = torch.ones(ref.shape[1], ref.shape[2], dtype=torch.bool)
    outside[gy0:gy1, gx0:gx1] = False
    inside = torch.zeros(ref.shape[1], ref.shape[2], dtype=torch.bool)
    inside[y0:y1, x0:x1] = True
    assert torch.equal(out[:, outside, :], ref[:, outside, :])
    assert torch.equal(out[:, inside, :], edited[:, inside, :])
    assert info["restored_share"] > 0.5


def test_write_back_unchanged_broadcasts_one_frame_reference():
    ref, edited = _synthetic_decode_plate()
    ref1 = ref[:1]
    out, _ = write_back_unchanged(edited, ref1, threshold=0.01, radius=8)
    out_single, _ = write_back_unchanged(edited, ref, threshold=0.01, radius=8)
    assert torch.allclose(out, out_single, atol=1e-5)


def test_write_back_unchanged_shape_mismatch_raises():
    ref, edited = _synthetic_decode_plate(h=64, w=64)
    bad_ref = ref[:, :, :32, :]
    with pytest.raises(VAECleanError, match="does not match"):
        write_back_unchanged(edited, bad_ref, threshold=0.01, radius=8)


def test_restore_defaults_still_bit_identical():
    img = plate()
    out, _, _, _ = run(img, restore_unchanged=0.0)
    assert torch.allclose(out, img, atol=2e-3)


def test_restore_without_reference_raises():
    with pytest.raises(VAECleanError, match="restore_unchanged"):
        run(plate(), restore_unchanged=0.05)


# ── R5 CIEDE2000 and round-trip report ───────────────────────────────────────

# Sharma, Wu, Dalal 2005 Table 1 — verified vs skimage.deltaE_ciede2000 to 1e-4.
SHARMA_2005 = (
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
)


def test_ciede2000_sharma_reference_pairs():
    for lab1, lab2, expected in SHARMA_2005:
        a = torch.tensor(lab1, dtype=torch.float64)
        b = torch.tensor(lab2, dtype=torch.float64)
        got = float(ciede2000(a, b))
        assert got == pytest.approx(expected, abs=1e-4)


def test_ciede2000_matches_skimage_on_random_lab():
    skimage = pytest.importorskip("skimage")
    from skimage.color import deltaE_ciede2000
    g = torch.Generator().manual_seed(42)
    lab1 = torch.stack([
        torch.rand(32, generator=g) * 100.0,
        torch.rand(32, generator=g) * 200.0 - 100.0,
        torch.rand(32, generator=g) * 200.0 - 100.0,
    ], dim=-1)
    lab2 = lab1 + torch.randn(32, 3) * 5.0
    ours = ciede2000(lab1, lab2).numpy()
    ref = deltaE_ciede2000(lab1.numpy(), lab2.numpy())
    assert (ours - ref).max() < 1e-3


def test_roundtrip_report_frame_mismatch_note():
    ref = torch.cat([plate(seed=i) for i in range(3)], dim=0)
    img = torch.cat([plate(seed=i) for i in range(4)], dim=0)
    report = roundtrip_report(img, ref)
    assert report["frame_note"] is not None
    assert "4n+1" in report["frame_note"]
    assert any("4n+1" in line for line in report["lines"])


def test_roundtrip_report_broadcast_one_frame_no_frame_note():
    ref = plate()[:1]
    img = torch.cat([plate(seed=i) for i in range(4)], dim=0)
    report = roundtrip_report(img, ref)
    assert report["frame_note"] is None
    assert not any("Frame count" in line for line in report["lines"])


def test_match_reference_different_size_skips_roundtrip_in_report():
    img = plate(64, 64)
    ref = plate(32, 32, seed=7)
    _, report, _, _ = run(img, balance_mode="match reference", reference=ref)
    assert "not measured" in report


def test_roundtrip_report_psnr_uses_raw_values():
    ref = plate()
    img = ref.clone()
    img[0, 0, 0, 0] = 1.5
    report = roundtrip_report(img, ref)
    assert report["max_err"] == pytest.approx(
        abs(1.5 - float(ref[0, 0, 0, 0])), abs=1e-5)
    assert report["outside_01_share"] > 0.0
