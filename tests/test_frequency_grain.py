"""Tests for MEC Frequency / Grain nodes and ops."""
from __future__ import annotations

import math
import sys
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

from nodes.frequency_grain._ops import (
    film_grain,
    frequency_combine,
    frequency_separate,
)
from nodes.frequency_grain.nodes import (
    FilmGrainMEC,
    FrequencyCombineMEC,
    FrequencySeparateMEC,
    NODE_CLASS_MAPPINGS,
)

ALL_NODE_IDS = (
    "FrequencySeparateMEC",
    "FrequencyCombineMEC",
    "FilmGrainMEC",
)

ROUNDTRIP_TOL = 2e-2


def _gradient_image(batch: int = 1, size: int = 64) -> torch.Tensor:
    ys = torch.linspace(0.1, 0.9, size)
    xs = torch.linspace(0.2, 0.8, size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rgb = torch.stack([xx, yy, (xx + yy) * 0.5], dim=-1)
    return rgb.unsqueeze(0).expand(batch, -1, -1, -1).clone()


def _detailed_image(batch: int = 1, size: int = 64, seed: int = 7) -> torch.Tensor:
    """A ramp with real per-pixel texture on it.

    Every test about frequency SEPARATION needs an image that actually has
    high frequencies. A smooth ramp has almost none - its HF layer sits at the
    neutral value everywhere - so assertions written against a ramp pass
    whether the code works or not. The texture is chromatic (independent per
    channel) on purpose: that is what tells Luminance and RGB detail apart.
    """
    gen = torch.Generator().manual_seed(seed)
    base = _gradient_image(batch, size)
    tex = torch.rand(batch, size, size, 3, generator=gen) * 0.35 - 0.175
    return (base + tex).clamp(0.02, 0.98)


_LUMA_W = torch.tensor((0.2126, 0.7152, 0.0722))


def _luma(x: torch.Tensor) -> torch.Tensor:
    return (x[..., :3] * _LUMA_W).sum(-1)


def _saturation(rgb: torch.Tensor) -> torch.Tensor:
    mx = rgb[..., :3].max(-1).values
    mn = rgb[..., :3].min(-1).values
    return torch.where(mx > 1e-6, (mx - mn) / mx.clamp(min=1e-6), torch.zeros_like(mx))


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """rgb [...,3] in 0..1 -> hsv same shape, h in 0..1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = torch.max(rgb, dim=-1).values
    minc = torch.min(rgb, dim=-1).values
    v = maxc
    delt = maxc - minc
    s = torch.where(maxc > 1e-6, delt / maxc.clamp(min=1e-6), torch.zeros_like(maxc))
    rc = (maxc - r) / delt.clamp(min=1e-6)
    gc = (maxc - g) / delt.clamp(min=1e-6)
    bc = (maxc - b) / delt.clamp(min=1e-6)
    h = torch.zeros_like(maxc)
    h = torch.where((maxc == r) & (delt > 1e-6), (bc - gc) % 6.0, h)
    h = torch.where((maxc == g) & (delt > 1e-6), 2.0 + rc - bc, h)
    h = torch.where((maxc == b) & (delt > 1e-6), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=-1)


def _roundtrip(
    image: torch.Tensor,
    *,
    mode: str,
    detail: str,
    linear: bool = True,
) -> torch.Tensor:
    hf, lf, _, _ = frequency_separate(
        image, "Gaussian", 4, 0.1, mode, detail, linear,
    )
    out, _ = frequency_combine(hf, lf, mode, linear, 1.0)
    return out[..., :3]


def _separate_lf(image, *, mode, detail, linear=True):
    """Just the base, for tests that compare against what the detail lands on."""
    _, lf, _, _ = frequency_separate(image, "Gaussian", 4, 0.1, mode, detail, linear)
    return lf[..., :3]


def _assert_input_unchanged(inp: torch.Tensor, fn, *args, **kwargs) -> None:
    before = inp.clone()
    fn(*args, **kwargs)
    assert torch.equal(inp, before), "Input tensor was mutated in place."


@pytest.fixture
def separate_node():
    return FrequencySeparateMEC()


@pytest.fixture
def combine_node():
    return FrequencyCombineMEC()


@pytest.fixture
def grain_node():
    return FilmGrainMEC()


class TestRegistration:
    def test_registration_three_nodes_present(self):
        for nid in ALL_NODE_IDS:
            assert nid in NODE_CLASS_MAPPINGS


class TestRoundTrip:
    """Separate -> Combine, on an image that actually has detail in it."""

    @pytest.mark.parametrize("mode", ["Divide", "Subtract"])
    def test_roundtrip_with_rgb_detail_is_exact(self, mode):
        """detail=RGB carries every channel, so the round trip is lossless.

        Measured max error is ~1e-5, so the tolerance is tight on purpose: a
        loose one here would not notice the colour space being applied twice,
        which is the failure this guards.
        """
        src = _detailed_image()
        out = _roundtrip(src, mode=mode, detail="RGB")
        assert torch.allclose(out, src, atol=1e-4), (
            f"{mode}+RGB drifted by {(out - src).abs().max():.5f}"
        )

    @pytest.mark.parametrize("mode", ["Divide", "Subtract"])
    def test_roundtrip_with_luminance_detail_is_deliberately_not_exact(self, mode):
        """detail=Luminance throws the CHROMATIC detail away. That is the deal.

        The detail layer is built from luminance alone and broadcast to three
        channels, so per-channel texture cannot come back - on a chromatic
        image the error is 0.3-0.6, not a rounding difference. This is pinned
        so nobody "fixes" it into an identity and quietly turns Luminance into
        RGB, losing the property the next test depends on.
        """
        src = _detailed_image()
        out = _roundtrip(src, mode=mode, detail="Luminance")
        assert (out - src).abs().max() > 0.1

    def test_luminance_detail_still_restores_the_luminance(self):
        """What Luminance mode DOES put back: the achromatic detail."""
        src = _detailed_image()
        out = _roundtrip(src, mode="Divide", detail="Luminance")
        base = _separate_lf(src, mode="Divide", detail="Luminance")
        assert (_luma(out) - _luma(src)).abs().mean() < (
            _luma(base) - _luma(src)).abs().mean(), (
            "recombining put back no luminance detail at all"
        )


class TestLuminanceDivide:
    """Why the default is Divide + Luminance and not Subtract + Luminance.

    In Divide the detail layer is a per-pixel SCALAR the base is multiplied by,
    and scaling all three linear channels by one number leaves the chromaticity
    alone. In Subtract it is an OFFSET added equally to all three, which moves
    a saturated pixel towards grey. Measured: 0.004 mean saturation error for
    Divide against 0.107 for Subtract - a factor of 25, not a nuance.
    """

    def test_divide_luminance_keeps_the_base_saturation(self):
        src = _detailed_image()
        base = _separate_lf(src, mode="Divide", detail="Luminance")
        out = _roundtrip(src, mode="Divide", detail="Luminance")
        drift = (_saturation(out) - _saturation(base)).abs().mean()
        assert drift < 0.02, f"saturation moved by {drift:.4f}"

    def test_subtract_luminance_does_not_and_that_is_the_tradeoff(self):
        src = _detailed_image()
        base = _separate_lf(src, mode="Subtract", detail="Luminance")
        out = _roundtrip(src, mode="Subtract", detail="Luminance")
        drift = (_saturation(out) - _saturation(base)).abs().mean()
        assert drift > 0.05, (
            "Subtract+Luminance stopped desaturating - if this now matches "
            "Divide, the Divide path has probably been changed into an offset"
        )


class TestMaskBehaviour:
    def test_mask_smaller_than_image_is_resized_not_ignored(self):
        image = _detailed_image(size=64)
        small_mask = torch.zeros(1, 32, 32)
        small_mask[:, 8:24, 8:24] = 1.0
        hf_none, _, hf_masked_none, _ = frequency_separate(
            image, "Gaussian", 4, 0.1, "Divide", "Luminance", True,
        )
        hf_small, _, hf_masked_small, _ = frequency_separate(
            image, "Gaussian", 4, 0.1, "Divide", "Luminance", True, mask=small_mask,
        )
        full_mask = torch.ones(1, 64, 64)
        _, _, hf_masked_full, _ = frequency_separate(
            image, "Gaussian", 4, 0.1, "Divide", "Luminance", True, mask=full_mask,
        )
        assert not torch.allclose(hf_masked_small, hf_masked_none, atol=1e-3)
        assert not torch.allclose(hf_masked_small, hf_masked_full, atol=1e-3)
        assert torch.allclose(hf_small, hf_none, atol=1e-5)

    def test_high_frequency_masked_is_neutral_outside_mask_divide(self):
        # _detailed_image, not a ramp: a ramp's HF sits AT the neutral value
        # everywhere, so "inside the mask is not neutral" would pass vacuously.
        image = _detailed_image(size=64)
        mask = torch.zeros(1, 64, 64)
        mask[:, 16:48, 16:48] = 1.0
        _, _, hf_masked, _ = frequency_separate(
            image, "Gaussian", 4, 0.1, "Divide", "Luminance", True, mask=mask,
        )
        outside = hf_masked[0, :16, :, 0]
        inside = hf_masked[0, 32, 32, 0]
        assert torch.allclose(outside, torch.ones_like(outside), atol=1e-4)
        assert not math.isclose(float(inside), 1.0, abs_tol=0.01)

    def test_high_frequency_masked_is_neutral_outside_mask_subtract(self):
        image = _detailed_image(size=64)
        mask = torch.zeros(1, 64, 64)
        mask[:, 16:48, 16:48] = 1.0
        _, _, hf_masked, _ = frequency_separate(
            image, "Gaussian", 4, 0.1, "Subtract", "Luminance", True, mask=mask,
        )
        outside = hf_masked[0, :16, :, 0]
        assert torch.allclose(outside, torch.zeros_like(outside), atol=1e-4)


class TestCombine:
    def test_combine_detail_strength_zero_returns_base(self):
        src = _gradient_image()
        hf, lf, _, _ = frequency_separate(
            src, "Gaussian", 4, 0.1, "Divide", "Luminance", True,
        )
        out, _ = frequency_combine(hf, lf, "Divide", True, 0.0)
        assert torch.allclose(out[..., :3], lf[..., :3], atol=ROUNDTRIP_TOL, rtol=ROUNDTRIP_TOL)

    def test_combine_preserves_rgba_alpha_channel(self):
        src = _gradient_image()
        hf, lf, _, _ = frequency_separate(
            src, "Gaussian", 4, 0.1, "Divide", "Luminance", True,
        )
        alpha = torch.full((1, 64, 64, 1), 0.42)
        lf_rgba = torch.cat([lf, alpha], dim=-1)
        out, _ = frequency_combine(hf, lf_rgba, "Divide", True, 1.0)
        assert out.shape[-1] == 4
        assert torch.allclose(out[..., 3], alpha[..., 0], atol=1e-5)


class TestFilmGrain:
    def test_film_grain_amount_zero_returns_unchanged_pixels(self):
        src = _gradient_image()
        out, notes = film_grain(src, 0.0, 25.0, 50.0, 0.0, 0, animate=True)
        assert torch.allclose(out, src)
        assert any("amount 0" in n for n in notes)

    def test_film_grain_animate_true_differs_per_frame(self):
        src = _gradient_image(batch=3)
        out, _ = film_grain(src, 50.0, 25.0, 50.0, 0.0, 42, animate=True)
        diff_01 = (out[0] - out[1]).abs().mean()
        diff_02 = (out[0] - out[2]).abs().mean()
        assert diff_01 > 1e-4
        assert diff_02 > 1e-4

    def test_film_grain_animate_false_matches_across_frames(self):
        src = _gradient_image(batch=3)
        out, _ = film_grain(src, 50.0, 25.0, 50.0, 0.0, 42, animate=False)
        assert torch.allclose(out[0], out[1], atol=1e-5)
        assert torch.allclose(out[0], out[2], atol=1e-5)

    def test_film_grain_same_seed_is_deterministic(self):
        src = _gradient_image()
        a, _ = film_grain(src, 50.0, 25.0, 50.0, 0.0, 7, animate=True)
        b, _ = film_grain(src, 50.0, 25.0, 50.0, 0.0, 7, animate=True)
        assert torch.allclose(a, b)


class TestInputIsolation:
    def test_frequency_separate_does_not_mutate_input(self, separate_node):
        image = _gradient_image()
        _assert_input_unchanged(
            image,
            separate_node.execute,
            image, "Gaussian", 8, 0.1, "Divide", "Luminance", True, None,
        )

    def test_frequency_combine_does_not_mutate_inputs(self, combine_node):
        src = _gradient_image()
        hf, lf, _, _ = frequency_separate(
            src, "Gaussian", 4, 0.1, "Divide", "Luminance", True,
        )
        hf_before = hf.clone()
        lf_before = lf.clone()
        combine_node.execute(hf, lf, "Divide", True, 1.0)
        assert torch.equal(hf, hf_before)
        assert torch.equal(lf, lf_before)

    def test_film_grain_does_not_mutate_input(self, grain_node):
        image = _gradient_image()
        _assert_input_unchanged(
            image,
            grain_node.execute,
            image, 25.0, 25.0, 50.0, 0.0, True, 0, None,
        )


class TestReporting:
    def test_median_reports_the_radius_it_actually_used(self, separate_node):
        image = _gradient_image()
        _, _, _, report = separate_node.execute(
            image, "Median", 40, 0.1, "Divide", "Luminance", True,
        )
        assert "effective 7" in report
        assert "radius 40" in report
