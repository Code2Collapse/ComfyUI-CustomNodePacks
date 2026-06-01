"""Local re-implementation of temporal Context Windows for long videos.

Reference: kijai's WanVideoContextWindows node. For videos longer than
the model's training window N_train, we split the latent into
overlapping temporal windows, run the sampler on each, and blend the
overlaps so motion remains smooth at the seams.

Pure-tensor primitives:
    plan_windows(F, window, overlap)               → list[range]
        Compute the (start, end) frame index of each window.
    blend_window_weights(window, overlap) → 1-D tensor of length `window`
        Symmetric blend ramp: 1 in the middle, linear-down to ramp[0]=0
        / ramp[-1]=0 over the overlap regions on each side.
    splice_windows(window_outputs, plan, full_len) → Tensor
        Combine per-window outputs back to a single (F, ...) tensor.

This module is sampler-agnostic; the Director's wrapper loops the
sampler over the planned windows and forwards outputs here.
"""
from __future__ import annotations

from typing import List, Sequence

import torch


def plan_windows(
    n_frames: int,
    *,
    window: int,
    overlap: int,
) -> List[range]:
    """Plan a uniform-stride covering of ``[0, n_frames)`` by windows.

    Args:
        n_frames: total frame count.
        window:   window length (must be > overlap).
        overlap:  shared frames between consecutive windows.

    Returns:
        List of ``range`` objects, each ``range(start, end)`` with
        ``end <= n_frames`` and ``end - start <= window``.
    """
    if window <= 0 or n_frames <= 0:
        raise ValueError("plan_windows: window and n_frames must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError(
            f"plan_windows: overlap={overlap} must be in [0, {window})"
        )
    if n_frames <= window:
        return [range(0, n_frames)]
    stride = window - overlap
    out: List[range] = []
    start = 0
    while start < n_frames:
        end = min(start + window, n_frames)
        out.append(range(start, end))
        if end == n_frames:
            break
        start += stride
    return out


def blend_window_weights(window: int, overlap: int,
                         *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Symmetric blend ramp of length ``window``.

    Linear ramps over ``overlap`` frames at each edge, flat=1 in
    between. Used to weight overlapping window outputs.
    """
    if window <= 0:
        raise ValueError("blend_window_weights: window must be positive")
    if overlap < 0 or overlap > window // 2:
        raise ValueError(
            f"blend_window_weights: overlap={overlap} not in [0, {window//2}]"
        )
    w = torch.ones(window, device=device, dtype=dtype)
    if overlap == 0:
        return w
    ramp = torch.linspace(0.0, 1.0, overlap + 2, device=device, dtype=dtype)[1:-1]
    w[:overlap]       = ramp
    w[-overlap:]      = ramp.flip(0)
    return w


def splice_windows(
    window_outputs: Sequence[torch.Tensor],
    plan: Sequence[range],
    full_len: int,
    *,
    overlap: int = 0,
) -> torch.Tensor:
    """Blend per-window outputs into a single ``(full_len, ...)`` tensor.

    Each entry of ``window_outputs`` is ``(window_len, ...)``;
    overlapping frames between adjacent windows are blended via the
    symmetric ramp.
    """
    if len(window_outputs) != len(plan):
        raise ValueError(
            f"splice_windows: outputs={len(window_outputs)} plan={len(plan)}"
        )
    if not window_outputs:
        raise ValueError("splice_windows: no window outputs")
    sample = window_outputs[0]
    out = torch.zeros((full_len, *sample.shape[1:]),
                      device=sample.device, dtype=sample.dtype)
    acc = torch.zeros(full_len, device=sample.device, dtype=sample.dtype)
    for win_idx, (frames, r) in enumerate(zip(window_outputs, plan)):
        if frames.shape[0] != len(r):
            raise ValueError(
                f"window {win_idx}: got {frames.shape[0]} frames, "
                f"expected {len(r)}"
            )
        # Weights are flat=1 for a single window, otherwise the ramp.
        if len(plan) == 1:
            w = torch.ones(len(r), device=sample.device, dtype=sample.dtype)
        else:
            w = blend_window_weights(
                len(r), min(overlap, len(r) // 2),
                device=sample.device, dtype=sample.dtype,
            )
        for i, f in enumerate(r):
            out[f] = out[f] + frames[i] * w[i]
            acc[f] = acc[f] + w[i]
    acc = acc.clamp_min(1e-8)
    # Broadcast acc over trailing dims.
    out = out / acc.view(-1, *([1] * (out.dim() - 1)))
    return out
