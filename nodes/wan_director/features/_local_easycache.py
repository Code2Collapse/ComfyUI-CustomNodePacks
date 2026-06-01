"""Local re-implementation of EasyCache.

Reference: kijai's ComfyUI-WanVideoWrapper "EasyCache" node (community
simplification of TeaCache for video DiTs).

Idea: simplest possible cache gate — reuse the last residual when the
**L1 distance between the current latent and the previously cached
latent** is below a fixed absolute threshold. No timestep embedding,
no magnitude EMA — just a direct latent-space difference.

This is the cheapest and most conservative cache; useful as a fallback
when neither TeaCache nor MagCache have been tuned. Pure-tensor,
state-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class EasyCache:
    """Stateful absolute-distance cache gate.

    Args:
        l1_thresh:  reuse cache when ``mean(|x - x_prev|) < l1_thresh``.
        max_skips:  cap on consecutive skips.
    """
    l1_thresh: float = 0.02
    max_skips: int   = 4
    _prev_latent:     Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _cached_residual: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _skip_count: int = field(default=0, init=False, repr=False)
    n_hits:   int = field(default=0, init=False)
    n_misses: int = field(default=0, init=False)

    def should_skip(self, latent: torch.Tensor) -> bool:
        if self._prev_latent is None or self._cached_residual is None:
            return False
        if self._skip_count >= self.max_skips:
            return False
        if latent.shape != self._prev_latent.shape:
            return False
        diff = (latent - self._prev_latent).abs().mean().item()
        return diff < self.l1_thresh

    def record(self, latent: torch.Tensor, residual: torch.Tensor) -> None:
        self._prev_latent     = latent.detach().clone()
        self._cached_residual = residual.detach().clone()
        self._skip_count = 0
        self.n_misses += 1

    @property
    def cached_residual(self) -> torch.Tensor:
        if self._cached_residual is None:
            raise RuntimeError("EasyCache: no residual cached yet")
        self._skip_count += 1
        self.n_hits += 1
        return self._cached_residual

    def reset(self) -> None:
        self._prev_latent = None
        self._cached_residual = None
        self._skip_count = 0
        self.n_hits = 0
        self.n_misses = 0

    def report(self) -> str:
        total = self.n_hits + self.n_misses
        ratio = (self.n_hits / total) if total else 0.0
        return (f"EasyCache: hits={self.n_hits} misses={self.n_misses} "
                f"hit_ratio={ratio:.2%}")
