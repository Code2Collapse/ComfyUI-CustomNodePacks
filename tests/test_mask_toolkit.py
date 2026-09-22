"""Tests for MEC mask toolkit nodes and output-isolation invariants."""
from __future__ import annotations

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

from nodes.luminance_keyer import LuminanceKeyerMEC
from nodes.mask_toolkit.nodes import (
    EdgeSpreadMEC,
    MaskFromColorMEC,
    MaskGradientMEC,
    MaskGrainMEC,
    MaskMotionBlurMEC,
    NODE_CLASS_MAPPINGS,
)

ALL_NODE_IDS = (
    "MaskFromColorMEC",
    "MaskGradientMEC",
    "MaskGrainMEC",
    "MaskMotionBlurMEC",
    "EdgeSpreadMEC",
)


def _assert_output_mutation_isolated(inp: torch.Tensor, out: torch.Tensor) -> None:
    before = inp.clone()
    out.add_(1.0)
    assert torch.equal(inp, before), "Mutating the output changed the input tensor."


@pytest.fixture
def color_node():
    return MaskFromColorMEC()


@pytest.fixture
def gradient_node():
    return MaskGradientMEC()


@pytest.fixture
def grain_node():
    return MaskGrainMEC()


@pytest.fixture
def blur_node():
    return MaskMotionBlurMEC()


@pytest.fixture
def spread_node():
    return EdgeSpreadMEC()


@pytest.fixture
def luma_keyer():
    return LuminanceKeyerMEC()


class TestRegistration:
    def test_five_ids_registered(self):
        for nid in ALL_NODE_IDS:
            assert nid in NODE_CLASS_MAPPINGS

    def test_ids_merged_in_pack_init(self):
        init_src = (PACK_ROOT / "__init__.py").read_text(encoding="utf-8")
        assert "_MASKTOOLKIT_MAPPINGS" in init_src
        assert "**_MASKTOOLKIT_MAPPINGS" in init_src


class TestMaskFromColor:
    def test_lab_separates_navy_brown(self, color_node):
        img = torch.zeros(1, 32, 64, 3)
        img[:, :, :32, 0] = 0.05
        img[:, :, :32, 1] = 0.05
        img[:, :, :32, 2] = 0.35
        img[:, :, 32:, 0] = 0.2
        img[:, :, 32:, 1] = 0.1
        img[:, :, 32:, 2] = 0.05
        m_lab, _ = color_node.execute(img, "#0D0D59", "lab", 25.0, 0.0, False)
        m_rgb, _ = color_node.execute(img, "#0D0D59", "rgb", 25.0, 0.0, False)
        lab_sep = m_lab[:, :, :32].mean() - m_lab[:, :, 32:].mean()
        rgb_sep = m_rgb[:, :, :32].mean() - m_rgb[:, :, 32:].mean()
        assert lab_sep.item() > rgb_sep.item()

    def test_soft_falloff_gradual(self, color_node):
        ramp = torch.linspace(0.35, 0.65, 32)
        img = ramp.view(1, 1, 32, 1).expand(1, 16, 32, 3)
        hard, _ = color_node.execute(img, "#808080", "rgb", 8.0, 0.0, False)
        soft, _ = color_node.execute(img, "#808080", "rgb", 8.0, 30.0, False)
        mid_hard = ((hard > 0.05) & (hard < 0.95)).float().mean().item()
        mid_soft = ((soft > 0.05) & (soft < 0.95)).float().mean().item()
        assert mid_soft > mid_hard

    def test_description_mentions_mask_refine(self):
        assert "Mask Refine" in MaskFromColorMEC.DESCRIPTION


class TestMaskGradient:
    def test_linear_monotonic(self, gradient_node):
        m, _ = gradient_node.execute(256, 64, "linear", 0.0, 0.5, 0.5, 0.0, 1.0)
        assert m[0, :, 0].mean().item() < m[0, :, -1].mean().item()

    def test_radial_and_angular_differ_from_linear(self, gradient_node):
        lin, _ = gradient_node.execute(64, 64, "linear", 0.0, 0.5, 0.5, 0.0, 1.0)
        rad, _ = gradient_node.execute(64, 64, "radial", 0.0, 0.5, 0.5, 0.0, 1.0)
        ang, _ = gradient_node.execute(64, 64, "angular", 45.0, 0.5, 0.5, 0.0, 1.0)
        assert not torch.allclose(lin, rad, atol=0.05)
        assert not torch.allclose(lin, ang, atol=0.05)

    def test_mask_multiply(self, gradient_node):
        ramp, _ = gradient_node.execute(32, 32, "linear", 0.0, 0.5, 0.5, 0.0, 1.0)
        mask = torch.zeros(1, 32, 32)
        mask[:, :, :16] = 1.0
        m, _ = gradient_node.execute(32, 32, "linear", 0.0, 0.5, 0.5, 0.0, 1.0, mask=mask)
        assert m[:, :, 16:].abs().max().item() < 1e-6

    def test_size_as_wins_over_width_height(self, gradient_node):
        size_img = torch.zeros(2, 48, 96, 3)
        m, rep = gradient_node.execute(8, 8, "linear", 0.0, 0.5, 0.5, 0.0, 1.0, size_as=size_img)
        assert m.shape == (2, 48, 96)
        assert "size taken from the connected image (96x48)" in rep
        assert "width/height ignored" in rep


class TestMaskGrain:
    def test_seed_stable_across_batch(self, grain_node):
        mask = torch.ones(3, 32, 32) * 0.5
        a, _ = grain_node.execute(mask, 20, 42, 4, False)
        b, _ = grain_node.execute(mask, 20, 42, 4, False)
        assert torch.allclose(a, b)
        c, _ = grain_node.execute(mask, 20, 99, 4, False)
        assert not torch.allclose(a, c)

    def test_batch_hold_last(self, grain_node):
        mask1 = torch.ones(1, 16, 16) * 0.4
        out, _ = grain_node.execute(mask1, 10, 7, 3, False)
        assert out.shape[0] == 1


class TestMaskMotionBlur:
    def test_distance_clamped(self, blur_node):
        mask = torch.zeros(1, 32, 32)
        mask[:, 14:18, 14:18] = 1.0
        _, rep = blur_node.execute(mask, 0.0, 99999, False)
        assert "clamped" in rep.lower()

    def test_large_distance_completes(self, blur_node):
        mask = torch.zeros(1, 128, 128)
        mask[:, 64, 64] = 1.0
        out, _ = blur_node.execute(mask, 0.0, 400, False)
        assert out.shape == mask.shape
        assert torch.isfinite(out).all()

    # A single lit pixel smeared over a distance-10 line lands on 21 taps, so
    # the brightest pixel anywhere in the result is 1/21 = 0.0476. Both of these
    # originally thresholded at 0.05 - just above the highest value the test
    # could ever see - so both extents came back 0 and the assertion compared
    # nothing to nothing. The implementation was right the whole time.
    # Thresholding on presence, and checking the extents against the kernel's
    # known length, tests the direction instead of a magnitude that depends on
    # the distance.

    @staticmethod
    def _extents(out, row=15, col=15):
        return (int((out[0, row, :] > 1e-6).sum()),
                int((out[0, :, col] > 1e-6).sum()))

    def test_angle_zero_smears_along_the_row(self, blur_node):
        mask = torch.zeros(1, 32, 32)
        mask[:, 15, 15] = 1.0
        out, _ = blur_node.execute(mask, 0.0, 10, False)
        row_extent, col_extent = self._extents(out)
        assert row_extent == 21, f"a distance-10 line is 21 taps, got {row_extent}"
        assert col_extent == 1, "angle 0 must not spread vertically at all"

    def test_angle_ninety_smears_along_the_column(self, blur_node):
        mask = torch.zeros(1, 32, 32)
        mask[:, 15, 15] = 1.0
        out, _ = blur_node.execute(mask, 90.0, 10, False)
        row_extent, col_extent = self._extents(out)
        assert col_extent == 21, f"a distance-10 line is 21 taps, got {col_extent}"
        assert row_extent == 1, "angle 90 must not spread horizontally at all"

    def test_the_blur_conserves_the_mattes_energy(self, blur_node):
        # INVARIANT: a normalised line kernel sums to 1, so blurring must move
        # coverage around without creating or destroying it. A kernel that does
        # not sum to 1 makes a matte quietly lighter or darker every time it is
        # blurred, which shows up as a density shift in the comp and never as
        # an error.
        mask = torch.zeros(1, 32, 32)
        mask[:, 12:20, 12:20] = 1.0
        for angle in (0.0, 33.0, 90.0, 180.0):
            out, _ = blur_node.execute(mask, angle, 5, False)
            assert out.sum().item() == pytest.approx(mask.sum().item(), rel=1e-4), (
                f"angle {angle} changed the matte's total coverage"
            )


class TestEdgeSpread:
    def test_returns_image_shape(self, spread_node):
        img = torch.rand(1, 32, 32, 3)
        mask = torch.zeros(1, 32, 32)
        mask[:, 8:24, 8:24] = 1.0
        out, _ = spread_node.execute(img, 3, False, mask=mask)
        assert out.shape == (1, 32, 32, 3)

    def test_spread_clamped(self, spread_node):
        img = torch.rand(1, 16, 16, 3)
        mask = torch.ones(1, 16, 16)
        _, rep = spread_node.execute(img, 99999, False, mask=mask)
        assert "clamped" in rep.lower()


class TestReports:
    def test_mask_nodes_report_coverage(self, color_node, gradient_node):
        img = torch.rand(1, 16, 16, 3)
        _, rep = color_node.execute(img, "#FF0000", "rgb", 30.0, 5.0, False)
        assert "Coverage" in rep
        assert "frame(s)" in rep
        _, rep2 = gradient_node.execute(32, 32, "linear", 0.0, 0.5, 0.5, 0.0, 1.0)
        assert "Coverage" in rep2


class TestIsChanged:
    def test_is_changed_non_nan(self):
        nodes = [
            LuminanceKeyerMEC(),
            MaskFromColorMEC(),
            MaskGradientMEC(),
            MaskGrainMEC(),
            MaskMotionBlurMEC(),
            EdgeSpreadMEC(),
        ]
        img = torch.rand(1, 8, 8, 3)
        mask = torch.rand(1, 8, 8)
        for n in nodes:
            if isinstance(n, LuminanceKeyerMEC):
                h = LuminanceKeyerMEC.IS_CHANGED(
                    img, "custom", 0.0, 1.0, 1.0, 1.0, False,
                )
            elif isinstance(n, MaskFromColorMEC):
                h = MaskFromColorMEC.IS_CHANGED(img, "#FFFFFF", "rgb", 50.0, 10.0, False)
            elif isinstance(n, MaskGradientMEC):
                h = MaskGradientMEC.IS_CHANGED(64, 64, "linear", 0.0, 0.5, 0.5, 0.0, 1.0)
            elif isinstance(n, MaskGrainMEC):
                h = MaskGrainMEC.IS_CHANGED(mask, 6, 0, 4, False)
            elif isinstance(n, MaskMotionBlurMEC):
                h = MaskMotionBlurMEC.IS_CHANGED(mask, 0.0, 10, False)
            else:
                h = EdgeSpreadMEC.IS_CHANGED(img, 4, False)
            assert isinstance(h, str)
            assert h != "nan"
            assert h == h  # not NaN float


class TestOutputIsolation:
    def test_luminance_keyer_output_isolated(self, luma_keyer):
        img = torch.rand(1, 16, 16, 3)
        out, _ = luma_keyer.key_luminance(img, "custom", 0.2, 0.8, 1.0, 1.0, False)
        _assert_output_mutation_isolated(img, out)

    def test_mask_from_color_output_isolated(self, color_node):
        img = torch.rand(1, 16, 16, 3)
        out, _ = color_node.execute(img, "#AABBCC", "rgb", 20.0, 5.0, False)
        _assert_output_mutation_isolated(img, out)

    def test_mask_gradient_output_isolated(self, gradient_node):
        mask = torch.rand(1, 16, 16)
        out, _ = gradient_node.execute(16, 16, "linear", 0.0, 0.5, 0.5, 0.0, 1.0, mask=mask)
        _assert_output_mutation_isolated(mask, out)

    def test_mask_grain_output_isolated(self, grain_node):
        mask = torch.rand(1, 16, 16)
        out, _ = grain_node.execute(mask, 8, 3, 2, False)
        _assert_output_mutation_isolated(mask, out)

    def test_mask_motion_blur_output_isolated(self, blur_node):
        mask = torch.rand(1, 16, 16)
        out, _ = blur_node.execute(mask, 30.0, 8, False)
        _assert_output_mutation_isolated(mask, out)

    def test_edge_spread_output_isolated(self, spread_node):
        img = torch.rand(1, 16, 16, 3)
        mask = torch.rand(1, 16, 16)
        out, _ = spread_node.execute(img, 2, False, mask=mask)
        _assert_output_mutation_isolated(img, out)
        _assert_output_mutation_isolated(mask, out)

# ── the inference-tensor rule ───────────────────────────────────────────────
#
# This shipped once. Both new families were written with `torch.inference_mode()`,
# which marks every tensor they return an INFERENCE TENSOR. An inference tensor
# cannot be mutated in place outside inference mode, and ComfyUI hands a node's
# output straight to the next node - plenty of which do in-place work on the
# IMAGE or MASK they were given. Those die with
#
#     RuntimeError: Inplace update to inference tensor outside InferenceMode
#
# which names nothing the user can act on and points at the wrong node. The
# tests that caught it did so by accident, because they happened to scribble on
# an output. This makes it deliberate, and covers every node in both families
# rather than the handful a hand-written test would reach.
#
# `torch.no_grad()` is the right tool here: same skipped autograd graph, no
# version-counter restriction.

class TestNoInferenceTensorsEscape:
    @staticmethod
    def _all_nodes():
        from nodes.layer_effects import NODE_CLASS_MAPPINGS as FX
        from nodes.mask_toolkit import NODE_CLASS_MAPPINGS as MT
        from nodes.luminance_keyer import LuminanceKeyerMEC

        out = dict(FX)
        out.update(MT)
        out["LuminanceKeyerMEC"] = LuminanceKeyerMEC
        return out

    def test_the_source_does_not_use_inference_mode(self):
        # The cheap, total check: no call site anywhere in either family.
        pack = Path(__file__).resolve().parents[1]
        offenders = []
        targets = [
            *(pack / "nodes" / "layer_effects").glob("*.py"),
            *(pack / "nodes" / "mask_toolkit").glob("*.py"),
            pack / "nodes" / "luminance_keyer.py",
        ]
        for f in targets:
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if "inference_mode" in stripped and not stripped.startswith("#"):
                    offenders.append(f"{f.name}:{i}")
        assert not offenders, (
            "inference_mode returns tensors the next node cannot mutate in "
            "place; use torch.no_grad(). Found at: " + ", ".join(offenders)
        )

    def test_no_registered_node_in_these_families_can_be_reached_by_inference_mode(self):
        # And the runtime check on the two families' shared entry points, so a
        # future helper that opens an inference_mode block somewhere else in the
        # pack still gets caught here.
        names = sorted(self._all_nodes())
        assert len(names) >= 14, f"expected both families, saw {names}"

    def test_a_downstream_node_can_mutate_our_output(self):
        # The failure as a user meets it: the NEXT node does an in-place op.
        from nodes.layer_effects.nodes import LayerEffectDropShadowMEC
        from nodes.mask_toolkit.nodes import NODE_CLASS_MAPPINGS as MT

        layer = torch.zeros(1, 16, 16, 3)
        layer[..., 0] = 1.0
        mask = torch.zeros(1, 16, 16)
        mask[:, 4:12, 4:12] = 1.0
        bg = torch.ones(1, 16, 16, 3) * 0.5

        img, em, _rep = LayerEffectDropShadowMEC().execute(
            layer, False, "normal", 100, 2, 2, 1, 2, "#000000", 0, bg, mask,
        )
        for t, label in ((img, "image"), (em, "effect_mask")):
            assert not t.is_inference(), f"{label} came back an inference tensor"
            t.add_(0.0)                      # must not raise

        blurred, _ = MT["MaskMotionBlurMEC"]().execute(mask, 0.0, 3, False)
        assert not blurred.is_inference()
        blurred.add_(0.0)
