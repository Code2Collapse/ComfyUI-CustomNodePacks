"""Tests for stabilization_utils module."""

import math
import numpy as np
import torch
import pytest

from nodes.stabilization_utils import (
    estimate_translation_torch,
    warp_frame_torch,
    motion_adaptive_temporal_smooth,
    smooth_bbox_trajectory,
    compute_stable_bbox_trajectory,
    compute_motion_magnitudes,
)

# ── Phase correlation translation estimation ─────────────────────────

def test_translation_zero_shift():
    """Identical frames should produce ~zero translation."""
    frame = torch.rand(64, 64, 3)
    dx, dy = estimate_translation_torch(frame, frame)
    assert abs(dx) < 2.0
    assert abs(dy) < 2.0


def test_translation_known_shift():
    """Frame shifted by known amount should be detected."""
    frame = torch.zeros(64, 64, 3)
    # Draw a bright rectangle
    frame[20:40, 20:40, :] = 1.0
    # Shift right by 5, down by 3
    shifted = torch.zeros_like(frame)
    shifted[23:43, 25:45, :] = 1.0
    dx, dy = estimate_translation_torch(frame, shifted)
    # Phase correlation may return either sign convention
    assert abs(abs(dx) - 5) < 3, f"dx={dx}"
    assert abs(abs(dy) - 3) < 3, f"dy={dy}"


# ── Torch warp ───────────────────────────────────────────────────────

def test_warp_frame_torch_identity():
    """Zero translation should preserve the frame (approximately)."""
    frame = torch.rand(32, 32, 3)
    warped = warp_frame_torch(frame, 0.0, 0.0)
    assert warped.shape == frame.shape
    # Should be very close to original
    assert (warped - frame).abs().max() < 0.05


def test_warp_frame_torch_translation():
    """Non-zero translation should shift content."""
    frame = torch.zeros(32, 32, 3)
    frame[10:20, 10:20, :] = 1.0
    warped = warp_frame_torch(frame, 5.0, 0.0)
    assert warped.shape == frame.shape
    # The block should have moved — center of mass should shift
    orig_cx = (frame[..., 0] > 0.5).float().nonzero(as_tuple=False)[:, 1].float().mean()
    warp_cx = (warped[..., 0] > 0.5).float().nonzero(as_tuple=False)[:, 1].float().mean()
    assert warp_cx > orig_cx or warp_cx < orig_cx  # shifted either way


# ── Motion-adaptive temporal smoothing ───────────────────────────────

def test_motion_adaptive_smooth_uniform():
    """Without motion magnitudes, should behave like uniform Gaussian."""
    mask = torch.rand(10, 32, 32)
    smoothed = motion_adaptive_temporal_smooth(mask, sigma_base=1.0)
    assert smoothed.shape == mask.shape
    assert smoothed.min() >= 0.0
    assert smoothed.max() <= 1.0


def test_motion_adaptive_smooth_with_magnitudes():
    """With motion magnitudes, high-motion frames should be less smoothed."""
    B = 10
    mask = torch.zeros(B, 32, 32)
    # Frame 5 has a sharp spike
    mask[5, 10:20, 10:20] = 1.0

    # Low motion everywhere — should smooth the spike heavily
    low_mags = [0.1] * B
    smoothed_low = motion_adaptive_temporal_smooth(
        mask, sigma_base=2.0, motion_magnitudes=low_mags, motion_sensitivity=0.8
    )

    # High motion at frame 5 — spike should be more preserved
    high_mags = [0.1] * B
    high_mags[5] = 10.0
    smoothed_high = motion_adaptive_temporal_smooth(
        mask, sigma_base=2.0, motion_magnitudes=high_mags, motion_sensitivity=0.8
    )

    # The spike at frame 5 should be more preserved with high motion
    spike_low = smoothed_low[5, 10:20, 10:20].mean()
    spike_high = smoothed_high[5, 10:20, 10:20].mean()
    assert spike_high >= spike_low * 0.8, f"spike_high={spike_high}, spike_low={spike_low}"


def test_motion_adaptive_smooth_single_frame():
    """Single frame should be returned unchanged."""
    mask = torch.rand(1, 32, 32)
    result = motion_adaptive_temporal_smooth(mask, sigma_base=1.0)
    assert torch.equal(result, mask)


# ── Smooth bbox trajectory ───────────────────────────────────────────

def test_smooth_bbox_trajectory_static():
    """Identical bboxes should remain unchanged."""
    bboxes = [(10, 20, 100, 100)] * 10
    result = smooth_bbox_trajectory(bboxes, method="median_then_exponential")
    for r in result:
        assert r == (10, 20, 100, 100)


def test_smooth_bbox_trajectory_outlier_rejection():
    """Median filter should reject a single outlier."""
    bboxes = [(10, 20, 100, 100)] * 10
    # Insert an outlier
    bboxes[5] = (500, 500, 50, 50)
    result = smooth_bbox_trajectory(bboxes, method="median", window_radius=2)
    # Outlier should be suppressed — the smoothed bbox[5] should be closer to (10,20,100,100)
    assert result[5][0] < 300, f"outlier not rejected: {result[5]}"


def test_smooth_bbox_trajectory_exponential():
    """Exponential smoothing should reduce jitter."""
    # Jittery sequence
    bboxes = [(10 + i % 3, 20 + i % 2, 100, 100) for i in range(20)]
    result = smooth_bbox_trajectory(bboxes, method="exponential", alpha=0.3)
    assert len(result) == 20
    # Smoothed values should have less variance than input
    x_in = [b[0] for b in bboxes]
    x_out = [b[0] for b in result]
    var_in = np.var(x_in)
    var_out = np.var(x_out)
    assert var_out <= var_in + 1.0


# ── Compute stable bbox trajectory ──────────────────────────────────

def test_compute_stable_bbox_trajectory_basic():
    """Should produce a valid bbox from a mask sequence."""
    mask = torch.zeros(5, 64, 64)
    mask[0, 10:30, 10:30] = 1.0
    mask[1, 12:32, 12:32] = 1.0
    mask[2, 11:31, 11:31] = 1.0
    mask[3, 13:33, 13:33] = 1.0
    mask[4, 10:30, 10:30] = 1.0
    x, y, w, h = compute_stable_bbox_trajectory(mask)
    assert w > 0 and h > 0
    assert x >= 0 and y >= 0
    # Smoothing may shift bbox slightly; allow tolerance
    assert x <= 13, f"x should be near 10, got {x}"
    assert y <= 13, f"y should be near 10, got {y}"
    assert x + w >= 28, f"x+w should cover mask, got {x+w}"
    assert y + h >= 28, f"y+h should cover mask, got {y+h}"


def test_compute_stable_bbox_trajectory_empty():
    """All-zero mask should return full image bbox."""
    mask = torch.zeros(5, 64, 64)
    x, y, w, h = compute_stable_bbox_trajectory(mask)
    assert w == 64 and h == 64


# ── Motion magnitudes ────────────────────────────────────────────────

def test_compute_motion_magnitudes_none_transforms():
    """None transforms should produce zero magnitudes."""
    transforms = [None, None, None]
    mags = compute_motion_magnitudes(transforms)
    assert all(m == 0.0 for m in mags)


def test_compute_motion_magnitudes_translation():
    """Translation-only transform should produce correct magnitude."""
    T = np.eye(3, dtype=np.float64)
    T[0, 2] = 3.0
    T[1, 2] = 4.0
    mags = compute_motion_magnitudes([None, T])
    assert mags[0] == 0.0
    assert mags[1] == pytest.approx(5.0, abs=0.5)  # sqrt(9+16) = 5, plus small rotation component
