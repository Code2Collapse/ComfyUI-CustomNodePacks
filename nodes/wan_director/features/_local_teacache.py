"""Local re-implementation of TeaCache (Timestep Embedding Aware Cache).

Paper: arXiv:2411.19108 (Liu et al., 2024) "TeaCache: Timestep Embedding
Aware Cache for Speeding up Video Diffusion Models".

Idea: across consecutive sampler steps, the residual added by a
transformer block changes slowly when the timestep-modulation
embedding is similar. So we can **reuse** the previous block's
residual when the modulation embedding delta is below a threshold,
and only run the full block when the embedding has changed enough.

This module provides the **state + gating logic** as a pure-tensor
class with no model dependency, so it can be unit-tested in isolation.

Integration (done by the Director / sampler wrapper):
    cache = TeaCache(rel_l1_thresh=0.1)
    for step in sampler:
        mod_emb = compute_timestep_modulation(t)
        if cache.should_skip(mod_emb):
            x = x + cache.cached_residual    # reuse
        else:
            residual = expensive_transformer(x, mod_emb)
            cache.record(mod_emb, residual)
            x = x + residual

The threshold ``rel_l1_thresh`` is the relative L1 norm change of the
modulation embedding that triggers a refresh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class TeaCache:
    """Stateful TeaCache gate.

    Args:
        rel_l1_thresh: trigger refresh when
            ``||mod_emb_t - mod_emb_prev||_1 / ||mod_emb_prev||_1`` > this.
        max_skips: hard cap on consecutive skips (prevents staleness).
    """
    rel_l1_thresh: float = 0.10
    max_skips:     int   = 5
    _prev_emb:        Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _cached_residual: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _skip_count: int = field(default=0, init=False, repr=False)
    n_hits:   int = field(default=0, init=False)
    n_misses: int = field(default=0, init=False)

    def should_skip(self, mod_emb: torch.Tensor) -> bool:
        """Return True iff the previous residual should be reused."""
        if self._prev_emb is None or self._cached_residual is None:
            return False
        if self._skip_count >= self.max_skips:
            return False
        prev_norm = self._prev_emb.abs().sum().clamp_min(1e-8)
        rel_l1   = (mod_emb - self._prev_emb).abs().sum() / prev_norm
        return bool((rel_l1 < self.rel_l1_thresh).item())

    def record(self, mod_emb: torch.Tensor, residual: torch.Tensor) -> None:
        """Record a fresh transformer pass (miss)."""
        self._prev_emb        = mod_emb.detach().clone()
        self._cached_residual = residual.detach().clone()
        self._skip_count = 0
        self.n_misses += 1

    @property
    def cached_residual(self) -> torch.Tensor:
        """Return the cached residual; bumps skip counter + hit count."""
        if self._cached_residual is None:
            raise RuntimeError("TeaCache: no residual cached yet")
        self._skip_count += 1
        self.n_hits += 1
        return self._cached_residual

    def reset(self) -> None:
        """Forget cached state (call between sampling jobs)."""
        self._prev_emb = None
        self._cached_residual = None
        self._skip_count = 0
        self.n_hits = 0
        self.n_misses = 0

    def report(self) -> str:
        total = self.n_hits + self.n_misses
        ratio = (self.n_hits / total) if total else 0.0
        return (f"TeaCache: hits={self.n_hits} misses={self.n_misses} "
                f"hit_ratio={ratio:.2%}")
