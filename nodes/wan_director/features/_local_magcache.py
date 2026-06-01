"""Local re-implementation of MagCache (Magnitude-Aware Caching).

Paper: arXiv:2506.09045 (Ma et al., 2025) "MagCache: Fast Video
Generation with Magnitude-Aware Cache".

Idea: instead of gating on the modulation embedding (TeaCache), gate
on the **magnitude ratio of the most recent residual** vs the
expected/average residual magnitude. When current step's residual is
predicted to be small (low magnitude regime), reuse the previous one.

This is a pure-tensor, model-agnostic gate; the Director / sampler
wrapper supplies residuals after each transformer pass.

Decision rule:
    Refresh when    |residual_t|_2 / |residual_prev|_2  > mag_thresh
    OR              consecutive_skips >= max_skips
    OR              first call (no cache yet)
    Otherwise       reuse cached residual.

Differs from TeaCache: TeaCache uses the input embedding to predict
whether the output residual will change. MagCache uses an EMA of the
actual residual magnitude to predict the next step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class MagCache:
    """Stateful magnitude-aware cache gate.

    Args:
        mag_thresh: ratio threshold; if predicted residual magnitude
            exceeds ``mag_thresh * ema_mag`` the cache is refreshed.
        ema_alpha:  EMA smoothing factor for the magnitude estimator.
        max_skips:  cap on consecutive skips.
    """
    mag_thresh: float = 1.2
    ema_alpha:  float = 0.3
    max_skips:  int   = 5
    _cached_residual: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _ema_mag:    float = field(default=0.0,  init=False, repr=False)
    _skip_count: int   = field(default=0,    init=False, repr=False)
    n_hits:   int = field(default=0, init=False)
    n_misses: int = field(default=0, init=False)

    def predict_magnitude(self, predictor: Optional[torch.Tensor] = None) -> float:
        """Return the current EMA magnitude estimate, or override if given."""
        if predictor is not None:
            return float(torch.linalg.vector_norm(predictor).item())
        return self._ema_mag

    def should_skip(self, predictor: Optional[torch.Tensor] = None) -> bool:
        """Decide whether to reuse the cached residual.

        ``predictor`` (optional): a tensor whose magnitude approximates
        what the next transformer pass would produce (e.g., the input
        latent's residual proxy). If None, we use the EMA alone.
        """
        if self._cached_residual is None:
            return False
        if self._skip_count >= self.max_skips:
            return False
        if self._ema_mag <= 0.0:
            return False
        pred = self.predict_magnitude(predictor)
        # Skip when the predicted magnitude is *not* growing past
        # the threshold ratio relative to the EMA baseline.
        return pred <= self.mag_thresh * self._ema_mag

    def record(self, residual: torch.Tensor) -> None:
        """Record a freshly-computed residual."""
        self._cached_residual = residual.detach().clone()
        mag = float(torch.linalg.vector_norm(residual).item())
        if self._ema_mag == 0.0:
            self._ema_mag = mag
        else:
            self._ema_mag = (self.ema_alpha * mag
                             + (1.0 - self.ema_alpha) * self._ema_mag)
        self._skip_count = 0
        self.n_misses += 1

    @property
    def cached_residual(self) -> torch.Tensor:
        if self._cached_residual is None:
            raise RuntimeError("MagCache: no residual cached yet")
        self._skip_count += 1
        self.n_hits += 1
        return self._cached_residual

    def reset(self) -> None:
        self._cached_residual = None
        self._ema_mag = 0.0
        self._skip_count = 0
        self.n_hits = 0
        self.n_misses = 0

    def report(self) -> str:
        total = self.n_hits + self.n_misses
        ratio = (self.n_hits / total) if total else 0.0
        return (f"MagCache: hits={self.n_hits} misses={self.n_misses} "
                f"hit_ratio={ratio:.2%} ema_mag={self._ema_mag:.4f}")
