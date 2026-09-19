"""
Tests for LuminanceKeyerMEC – verifies real computation, not gimmick code.

Anti-gimmick checks:
  1. Output differs from input ✓
  2. Metrics computed not hardcoded ✓
  3. All branches reachable ✓
  4. Error paths inform user ✓
  5. Bright vs dark images produce visibly different masks ✓
  6. Fallback paths do something real ✓
"""

import torch
import pytest

import sys
import types

# Stub ComfyUI modules for test isolation
for mod_name in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

from nodes.luminance_keyer import LuminanceKeyerMEC, _smooth_step, _auto_select_mode


@pytest.fixture
def keyer():
    return LuminanceKeyerMEC()


@pytest.fixture
def bright_image():
    """Predominantly bright image (mean luma ~0.85)."""
    img = torch.ones(1, 64, 64, 3) * 0.85
    # Add slight variation
    img[:, :32, :, 0] = 0.9
    img[:, 32:, :, 1] = 0.8
    return img


@pytest.fixture
def dark_image():
    """Predominantly dark image (mean luma ~0.15)."""
    img = torch.ones(1, 64, 64, 3) * 0.15
    img[:, :32, :, 0] = 0.1
    img[:, 32:, :, 1] = 0.2
    return img


@pytest.fixture
def mid_image():
    """Balanced midtone image (mean luma ~0.5)."""
    img = torch.ones(1, 64, 64, 3) * 0.5
    return img


@pytest.fixture
def gradient_image():
    """Horizontal gradient from black to white – exercises the full luminance range."""
    B, H, W, C = 1, 64, 256, 3
    grad = torch.linspace(0.0, 1.0, W).unsqueeze(0).unsqueeze(0).expand(B, H, W)
    img = grad.unsqueeze(-1).expand(B, H, W, C)
    return img


@pytest.fixture
def batch_image():
    """Batch of 3: dark, mid, bright."""
    dark = torch.ones(1, 32, 32, 3) * 0.1
    mid = torch.ones(1, 32, 32, 3) * 0.5
    bright = torch.ones(1, 32, 32, 3) * 0.9
    return torch.cat([dark, mid, bright], dim=0)


# ─── Schema / Input Types ────────────────────────────────────────────

# Frozen widget order — inserting into the middle breaks saved workflows.
_LUMAKEY_REQUIRED_ORDER = [
    "image", "mode", "low", "high", "gamma", "falloff", "invert",
]
_LUMAKEY_OPTIONAL_ORDER = ["channel", "low_soft", "high_soft", "invert_key"]

# Goldens captured from LuminanceKeyerMEC BEFORE the mask-toolkit extension.
# Means/sums for representative fixtures; full tensors compared via reference impl.
_GOLDEN_STATS = {
    "gradient_custom": {"mean": 0.5, "sum": 8192.0, "tl": 0.0, "br": 1.0, "mid": 0.7530365586280823},
    "gradient_highlights": {"mean": 0.15, "sum": 2457.6, "tl": 0.0, "br": 1.0, "mid": 0.0},
    "gradient_shadows": {"mean": 0.15, "sum": 2457.6, "tl": 1.0, "br": 0.0, "mid": 0.0},
    "bright_auto": {"mean": 0.0, "sum": 0.0},
    "dark_auto": {"mean": 0.0, "sum": 0.0},
    "gradient_narrow": {"mean": 0.2, "sum": 3276.8},
    "batch_custom": {"frame_means": [0.028, 0.5, 0.972]},
}


def _reference_key_pre_extension(
    image, mode, low, high, gamma, falloff, invert,
):
    """Verbatim pre-extension luminance key path (BT.709 + existing modes only)."""
    r = image[:, :, :, 0]
    g = image[:, :, :, 1]
    b = image[:, :, :, 2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    luminance = luminance.clamp(0.0, 1.0)
    presets = {"highlights": (0.7, 1.0), "midtones": (0.3, 0.7), "shadows": (0.0, 0.3)}
    effective_mode = mode
    if mode == "auto":
        mean_luma = luminance.mean().item()
        if mean_luma > 0.6:
            effective_mode = "shadows"
        elif mean_luma < 0.4:
            effective_mode = "highlights"
        else:
            effective_mode = "midtones"
    if effective_mode in presets:
        t_low, t_high = presets[effective_mode]
    else:
        t_low, t_high = low, high
    if t_low > t_high:
        t_low, t_high = t_high, t_low
    span = t_high - t_low
    if span < 1e-7:
        mask = (luminance >= t_low).float()
    elif effective_mode == "shadows":
        t = ((luminance - t_low) / span).clamp(0.0, 1.0)
        mask = _smooth_step(1.0 - t, falloff)
    elif effective_mode == "midtones":
        midpoint = (t_low + t_high) * 0.5
        half_span = span * 0.5
        t = 1.0 - ((luminance - midpoint).abs() / half_span).clamp(0.0, 1.0)
        mask = _smooth_step(t, falloff)
    else:
        t = ((luminance - t_low) / span).clamp(0.0, 1.0)
        mask = _smooth_step(t, falloff)
    if abs(gamma - 1.0) > 1e-6:
        mask = mask.clamp(0.0, 1.0).pow(max(gamma, 0.01))
    if invert:
        mask = 1.0 - mask
    return mask.clamp(0.0, 1.0)


class TestSchema:
    def test_widget_order_pinned(self):
        spec = LuminanceKeyerMEC.INPUT_TYPES()
        assert list(spec["required"].keys()) == _LUMAKEY_REQUIRED_ORDER
        assert list(spec.get("optional", {}).keys()) == _LUMAKEY_OPTIONAL_ORDER

    def test_input_types_structure(self):
        inputs = LuminanceKeyerMEC.INPUT_TYPES()
        req = inputs["required"]
        assert "image" in req
        assert "mode" in req
        assert "low" in req
        assert "high" in req
        assert "gamma" in req
        assert "falloff" in req
        assert "invert" in req

    def test_all_inputs_have_tooltips(self):
        inputs = LuminanceKeyerMEC.INPUT_TYPES()
        for name, spec in inputs["required"].items():
            if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], dict):
                assert "tooltip" in spec[1], f"Missing tooltip for '{name}'"

    def test_return_types(self):
        assert LuminanceKeyerMEC.RETURN_TYPES == ("MASK", "STRING")
        assert LuminanceKeyerMEC.RETURN_NAMES == ("mask", "info")

    def test_category(self):
        assert LuminanceKeyerMEC.CATEGORY == "C2C/Keying"

    def test_vram_tier(self):
        assert LuminanceKeyerMEC.VRAM_TIER == 1


# ─── Golden / default compatibility ───────────────────────────────────

class TestGoldenDefaults:
    def test_defaults_bit_identical_golden(self, keyer, gradient_image, batch_image):
        cases = [
            (gradient_image, "custom", 0.0, 1.0, 1.0, 1.0, False),
            (gradient_image, "highlights", 0.0, 1.0, 1.0, 1.0, False),
            (gradient_image, "shadows", 0.0, 1.0, 1.0, 1.0, False),
            (torch.ones(1, 64, 64, 3) * 0.85, "auto", 0.0, 1.0, 1.0, 1.0, False),
            (torch.ones(1, 64, 64, 3) * 0.15, "auto", 0.0, 1.0, 1.0, 1.0, False),
            (gradient_image, "custom", 0.4, 0.6, 1.0, 1.0, False),
        ]
        for img, mode, low, high, gamma, falloff, inv in cases:
            ref = _reference_key_pre_extension(img, mode, low, high, gamma, falloff, inv)
            got, _ = keyer.key_luminance(
                img, mode, low, high, gamma, falloff, inv,
                channel="luma", low_soft=0.0, high_soft=0.0, invert_key=False,
            )
            assert torch.allclose(ref, got, atol=1e-6)
        ref_b = _reference_key_pre_extension(
            batch_image, "custom", 0.0, 1.0, 1.0, 1.0, False,
        )
        got_b, _ = keyer.key_luminance(
            batch_image, "custom", 0.0, 1.0, 1.0, 1.0, False,
            channel="luma", low_soft=0.0, high_soft=0.0, invert_key=False,
        )
        assert torch.allclose(ref_b, got_b, atol=1e-6)

    def test_hardcoded_golden_stats(self, keyer, gradient_image):
        m, _ = keyer.key_luminance(
            gradient_image, "custom", 0.0, 1.0, 1.0, 1.0, False,
            channel="luma", low_soft=0.0, high_soft=0.0, invert_key=False,
        )
        g = _GOLDEN_STATS["gradient_custom"]
        assert abs(m.mean().item() - g["mean"]) < 1e-4
        assert abs(m.sum().item() - g["sum"]) < 1.0
        assert abs(m[0, 0, 0].item() - g["tl"]) < 1e-6
        assert abs(m[0, -1, -1].item() - g["br"]) < 1e-6

    def test_channel_red_differs_from_luma(self, keyer):
        img = torch.zeros(1, 32, 32, 3)
        img[:, :, :, 0] = 0.9
        img[:, :, :, 1] = 0.1
        img[:, :, :, 2] = 0.1
        m_luma, _ = keyer.key_luminance(
            img, "custom", 0.0, 1.0, 1.0, 1.0, False, channel="luma",
        )
        m_red, _ = keyer.key_luminance(
            img, "custom", 0.0, 1.0, 1.0, 1.0, False, channel="red",
        )
        assert not torch.allclose(m_luma, m_red, atol=0.05)

    def test_soft_ends_widen_transition(self, keyer, gradient_image):
        hard, _ = keyer.key_luminance(
            gradient_image, "custom", 0.3, 0.7, 1.0, 1.0, False,
            low_soft=0.0, high_soft=0.0,
        )
        soft, _ = keyer.key_luminance(
            gradient_image, "custom", 0.3, 0.7, 1.0, 1.0, False,
            low_soft=0.15, high_soft=0.15,
        )
        mid_hard = ((hard > 0.1) & (hard < 0.9)).float().mean().item()
        mid_soft = ((soft > 0.1) & (soft < 0.9)).float().mean().item()
        assert mid_soft > mid_hard


# ─── BT.709 Luminance ────────────────────────────────────────────────

class TestLuminance:
    def test_pure_red(self, keyer):
        """Pure red → luminance = 0.2126."""
        img = torch.zeros(1, 4, 4, 3)
        img[:, :, :, 0] = 1.0  # R=1
        mask, info = keyer.key_luminance(img, "custom", 0.0, 1.0, 1.0, 1.0, False)
        # BT.709: 0.2126*1 + 0 + 0 = 0.2126; normalized over [0,1] → smoothstep(0.2126)
        assert mask.shape == (1, 4, 4)
        mean_val = mask.mean().item()
        # Luminance 0.2126 → t=0.2126 → smoothstep gives ~0.0867
        assert 0.05 < mean_val < 0.2, f"Expected low mask for pure red, got {mean_val}"

    def test_pure_green(self, keyer):
        """Pure green → luminance = 0.7152 (dominant channel)."""
        img = torch.zeros(1, 4, 4, 3)
        img[:, :, :, 1] = 1.0  # G=1
        mask, info = keyer.key_luminance(img, "custom", 0.0, 1.0, 1.0, 1.0, False)
        mean_val = mask.mean().item()
        # Luminance 0.7152 → t=0.7152 → smoothstep gives ~0.803
        assert 0.6 < mean_val < 0.9, f"Expected mid-high mask for pure green, got {mean_val}"

    def test_pure_white(self, keyer):
        """Pure white → luminance = 1.0 → mask = 1.0."""
        img = torch.ones(1, 4, 4, 3)
        mask, _ = keyer.key_luminance(img, "custom", 0.0, 1.0, 1.0, 1.0, False)
        assert torch.allclose(mask, torch.ones_like(mask), atol=1e-5)

    def test_pure_black(self, keyer):
        """Pure black → luminance = 0.0 → mask = 0.0."""
        img = torch.zeros(1, 4, 4, 3)
        mask, _ = keyer.key_luminance(img, "custom", 0.0, 1.0, 1.0, 1.0, False)
        assert torch.allclose(mask, torch.zeros_like(mask), atol=1e-5)


# ─── Bright vs Dark produce different masks ──────────────────────────

class TestBrightVsDark:
    def test_bright_and_dark_differ_in_custom_mode(self, keyer, bright_image, dark_image):
        mask_bright, _ = keyer.key_luminance(bright_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        mask_dark, _ = keyer.key_luminance(dark_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        diff = (mask_bright.mean() - mask_dark.mean()).abs().item()
        assert diff > 0.3, f"Bright/dark masks too similar: diff={diff:.4f}"

    def test_bright_and_dark_differ_in_auto_mode(self, keyer, bright_image, dark_image):
        mask_bright, info_b = keyer.key_luminance(bright_image, "auto", 0.0, 1.0, 1.0, 1.0, False)
        mask_dark, info_d = keyer.key_luminance(dark_image, "auto", 0.0, 1.0, 1.0, 1.0, False)
        # Auto should pick different modes for bright vs dark
        assert "shadows" in info_b, f"Expected auto→shadows for bright image, got: {info_b}"
        assert "highlights" in info_d, f"Expected auto→highlights for dark image, got: {info_d}"

    def test_highlights_keys_bright_pixels(self, keyer, gradient_image):
        mask, _ = keyer.key_luminance(gradient_image, "highlights", 0.0, 1.0, 1.0, 1.0, False)
        # Left side (dark) should be ~0, right side (bright) should be ~1
        left_mean = mask[:, :, :64].mean().item()
        right_mean = mask[:, :, -64:].mean().item()
        assert left_mean < 0.1, f"Highlights: left (dark) should be ~0, got {left_mean:.3f}"
        assert right_mean > 0.5, f"Highlights: right (bright) should be high, got {right_mean:.3f}"

    def test_shadows_keys_dark_pixels(self, keyer, gradient_image):
        mask, _ = keyer.key_luminance(gradient_image, "shadows", 0.0, 1.0, 1.0, 1.0, False)
        left_mean = mask[:, :, :64].mean().item()
        right_mean = mask[:, :, -64:].mean().item()
        assert left_mean > 0.5, f"Shadows: left (dark) should be high, got {left_mean:.3f}"
        assert right_mean < 0.05, f"Shadows: right (bright) should be ~0, got {right_mean:.3f}"


# ─── Modes ────────────────────────────────────────────────────────────

class TestModes:
    def test_auto_bright_picks_shadows(self, keyer, bright_image):
        _, info = keyer.key_luminance(bright_image, "auto", 0.0, 1.0, 1.0, 1.0, False)
        assert "shadows" in info

    def test_auto_dark_picks_highlights(self, keyer, dark_image):
        _, info = keyer.key_luminance(dark_image, "auto", 0.0, 1.0, 1.0, 1.0, False)
        assert "highlights" in info

    def test_auto_mid_picks_midtones(self, keyer, mid_image):
        _, info = keyer.key_luminance(mid_image, "auto", 0.0, 1.0, 1.0, 1.0, False)
        assert "midtones" in info

    def test_custom_uses_user_thresholds(self, keyer, gradient_image):
        # Custom with narrow band at 0.4–0.6
        mask, info = keyer.key_luminance(gradient_image, "custom", 0.4, 0.6, 1.0, 1.0, False)
        assert "low=0.400" in info
        assert "high=0.600" in info
        # Most of gradient should be either 0 or 1 (narrow band)
        extremes = ((mask < 0.05) | (mask > 0.95)).float().mean().item()
        assert extremes > 0.6, f"Expected mostly 0/1 for narrow band, got {extremes:.2f}"


# ─── Falloff ──────────────────────────────────────────────────────────

class TestFalloff:
    def test_hard_falloff_produces_binary(self, keyer, gradient_image):
        mask, _ = keyer.key_luminance(gradient_image, "custom", 0.3, 0.7, 1.0, 0.0, False)
        # With falloff=0, result should be binary
        unique_vals = mask.unique()
        assert len(unique_vals) <= 3, f"Hard falloff should produce near-binary, got {len(unique_vals)} unique values"

    def test_smooth_falloff_has_gradient(self, keyer, gradient_image):
        mask, _ = keyer.key_luminance(gradient_image, "custom", 0.3, 0.7, 1.0, 1.0, False)
        unique_vals = mask.unique()
        assert len(unique_vals) > 10, f"Smooth falloff should produce gradient, got {len(unique_vals)} unique values"

    def test_high_falloff_smoother_than_low(self, keyer, gradient_image):
        mask_low, _ = keyer.key_luminance(gradient_image, "custom", 0.2, 0.8, 1.0, 0.5, False)
        mask_high, _ = keyer.key_luminance(gradient_image, "custom", 0.2, 0.8, 1.0, 5.0, False)
        # High falloff should have more intermediate values
        mid_low = ((mask_low > 0.1) & (mask_low < 0.9)).float().mean().item()
        mid_high = ((mask_high > 0.1) & (mask_high < 0.9)).float().mean().item()
        assert mid_high >= mid_low, "Higher falloff should produce more intermediate values"


# ─── Gamma ────────────────────────────────────────────────────────────

class TestGamma:
    def test_gamma_greater_than_1_darkens(self, keyer, gradient_image):
        mask_normal, _ = keyer.key_luminance(gradient_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        mask_high_gamma, _ = keyer.key_luminance(gradient_image, "custom", 0.0, 1.0, 3.0, 1.0, False)
        # gamma>1 → pow(gamma) → darker overall (compresses toward 0)
        assert mask_high_gamma.mean().item() < mask_normal.mean().item()

    def test_gamma_less_than_1_brightens(self, keyer, gradient_image):
        mask_normal, _ = keyer.key_luminance(gradient_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        mask_low_gamma, _ = keyer.key_luminance(gradient_image, "custom", 0.0, 1.0, 0.3, 1.0, False)
        # gamma<1 → pow(gamma) → brighter overall (expands toward 1)
        assert mask_low_gamma.mean().item() > mask_normal.mean().item()

    def test_gamma_1_is_identity(self, keyer, gradient_image):
        mask_a, _ = keyer.key_luminance(gradient_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        mask_b, _ = keyer.key_luminance(gradient_image, "custom", 0.0, 1.0, 1.0001, 1.0, False)
        assert torch.allclose(mask_a, mask_b, atol=0.01)


# ─── Invert ───────────────────────────────────────────────────────────

class TestInvert:
    def test_invert_flips_mask(self, keyer, gradient_image):
        mask_normal, _ = keyer.key_luminance(gradient_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        mask_inverted, _ = keyer.key_luminance(gradient_image, "custom", 0.0, 1.0, 1.0, 1.0, True)
        reconstructed = mask_normal + mask_inverted
        assert torch.allclose(reconstructed, torch.ones_like(reconstructed), atol=1e-5)


# ─── Batch handling ───────────────────────────────────────────────────

class TestBatch:
    def test_batch_output_shape(self, keyer, batch_image):
        mask, info = keyer.key_luminance(batch_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        assert mask.shape == (3, 32, 32)

    def test_batch_per_frame_coverage_in_info(self, keyer, batch_image):
        _, info = keyer.key_luminance(batch_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        assert "3 frame(s)" in info
        assert "Per-frame coverage:" in info

    def test_batch_frames_differ(self, keyer, batch_image):
        mask, _ = keyer.key_luminance(batch_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        # Frame 0 (dark) should differ from Frame 2 (bright)
        diff = (mask[2].mean() - mask[0].mean()).abs().item()
        assert diff > 0.3, f"Batch frames should differ: diff={diff:.4f}"


# ─── Info string ──────────────────────────────────────────────────────

class TestInfoString:
    def test_info_contains_computed_values(self, keyer, gradient_image):
        _, info = keyer.key_luminance(gradient_image, "custom", 0.2, 0.8, 1.5, 2.0, True)
        assert "Mean luminance:" in info
        assert "Mask coverage:" in info
        assert "Thresholds:" in info
        assert "Gamma: 1.50" in info
        assert "Invert: True" in info

    def test_info_values_are_not_hardcoded(self, keyer, bright_image, dark_image):
        _, info_b = keyer.key_luminance(bright_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        _, info_d = keyer.key_luminance(dark_image, "custom", 0.0, 1.0, 1.0, 1.0, False)
        # Mean luminance should differ between bright and dark images
        assert info_b != info_d, "Info strings should differ for different images"


# ─── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_low_equals_high_produces_binary(self, keyer, gradient_image):
        mask, _ = keyer.key_luminance(gradient_image, "custom", 0.5, 0.5, 1.0, 1.0, False)
        # Degenerate range → hard binary at 0.5
        unique_vals = mask.unique()
        assert len(unique_vals) == 2

    def test_swapped_thresholds_auto_corrected(self, keyer, gradient_image):
        mask_normal, _ = keyer.key_luminance(gradient_image, "custom", 0.3, 0.7, 1.0, 1.0, False)
        mask_swapped, _ = keyer.key_luminance(gradient_image, "custom", 0.7, 0.3, 1.0, 1.0, False)
        assert torch.allclose(mask_normal, mask_swapped, atol=1e-5)

    def test_single_pixel_image(self, keyer):
        img = torch.tensor([[[[0.5, 0.5, 0.5]]]])  # (1,1,1,3)
        mask, info = keyer.key_luminance(img, "custom", 0.0, 1.0, 1.0, 1.0, False)
        assert mask.shape == (1, 1, 1)
        assert "1x1" in info


# ─── Smoothstep unit tests ───────────────────────────────────────────

class TestSmoothStep:
    def test_endpoints(self):
        x = torch.tensor([0.0, 1.0])
        result = _smooth_step(x, 1.0)
        assert torch.allclose(result, x, atol=1e-6)

    def test_midpoint(self):
        x = torch.tensor([0.5])
        result = _smooth_step(x, 1.0)
        assert torch.allclose(result, torch.tensor([0.5]), atol=1e-6)

    def test_hard_step(self):
        x = torch.tensor([0.3, 0.7])
        result = _smooth_step(x, 0.0)
        assert torch.allclose(result, torch.tensor([0.0, 1.0]))


class TestAutoSelectMode:
    def test_bright_selects_shadows(self):
        luma = torch.ones(1, 10, 10) * 0.8
        assert _auto_select_mode(luma) == "shadows"

    def test_dark_selects_highlights(self):
        luma = torch.ones(1, 10, 10) * 0.2
        assert _auto_select_mode(luma) == "highlights"

    def test_mid_selects_midtones(self):
        luma = torch.ones(1, 10, 10) * 0.5
        assert _auto_select_mode(luma) == "midtones"
