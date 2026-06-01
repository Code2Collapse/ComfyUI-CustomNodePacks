"""Local re-implementation of SLG (Skip Layer Guidance).

Paper: arXiv:2411.18664 (Hong et al., 2024) "Skip-Layer Guidance: Layer
Subset Selection for Conditional Generation".

Idea: at sampler time, run TWO forward passes:
    eps_pos    = unet(x, t, cond_pos)         # full pass
    eps_skip   = unet(x, t, cond_pos, skip_layers=[...])  # some layers dropped
Then guide:
    eps_final = eps_pos + slg_scale * (eps_pos - eps_skip)

The "skip" pass is run with a subset of transformer blocks bypassed
(residual returned unchanged). The signal `(eps_pos - eps_skip)` is
what those blocks contributed; amplifying it strengthens the conditional
direction.

This module provides:
    1. SLGConfig: which layer indices to skip, the guidance scale,
       and an optional step window [start, end] when SLG is active.
    2. should_apply_slg(step, n_steps, cfg) → bool: gate by step window.
    3. make_layer_skip_predicate(cfg) → fn(layer_idx) → bool: used
       inside the transformer to decide whether to bypass a block.
    4. combine_slg(eps_pos, eps_skip, scale) → eps_final.

Pure-tensor; integration into the sampler is the Director's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch


@dataclass
class SLGConfig:
    """Skip-Layer Guidance configuration.

    Args:
        skip_layers: indices of transformer blocks to bypass in the
            "skip" branch (0-based). Empty = SLG disabled.
        slg_scale:   guidance multiplier applied to (eps_pos - eps_skip).
        start_pct:   step-window start as fraction in [0, 1].
        end_pct:     step-window end   as fraction in [0, 1] (exclusive).
    """
    skip_layers: Sequence[int] = field(default_factory=tuple)
    slg_scale:   float = 0.7
    start_pct:   float = 0.0
    end_pct:     float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.start_pct <= 1.0 and 0.0 <= self.end_pct <= 1.0):
            raise ValueError("SLG start_pct/end_pct must lie in [0, 1]")
        if self.end_pct < self.start_pct:
            raise ValueError("SLG end_pct must be >= start_pct")
        # Deduplicate + freeze.
        self.skip_layers = tuple(sorted(set(int(i) for i in self.skip_layers)))


def should_apply_slg(step: int, n_steps: int, cfg: SLGConfig) -> bool:
    """Return True iff SLG should be active on this step."""
    if not cfg.skip_layers or n_steps <= 0:
        return False
    pct = step / max(1, n_steps - 1) if n_steps > 1 else 0.0
    return cfg.start_pct <= pct < cfg.end_pct


def make_layer_skip_predicate(cfg: SLGConfig) -> Callable[[int], bool]:
    """Return a fast ``fn(layer_idx) -> bool`` for the "skip" branch."""
    skip_set = frozenset(cfg.skip_layers)
    return lambda layer_idx: layer_idx in skip_set


def combine_slg(eps_pos: torch.Tensor,
                eps_skip: torch.Tensor,
                slg_scale: float) -> torch.Tensor:
    """Combine full + skip predictions to produce the SLG-guided output.

    Args:
        eps_pos:   prediction from the full conditional pass.
        eps_skip:  prediction from the layer-skipped pass.
        slg_scale: SLG guidance strength.

    Returns:
        ``eps_pos + slg_scale * (eps_pos - eps_skip)``, same shape as inputs.
    """
    if eps_pos.shape != eps_skip.shape:
        raise ValueError(
            f"combine_slg: shape mismatch {tuple(eps_pos.shape)} vs "
            f"{tuple(eps_skip.shape)}"
        )
    return eps_pos + slg_scale * (eps_pos - eps_skip)
