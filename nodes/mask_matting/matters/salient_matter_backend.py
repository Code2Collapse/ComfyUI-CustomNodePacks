"""Salient-model matter backends.

BiRefNet and RMBG-2.0 are dichotomous-segmentation models whose raw sigmoid
output is a *soft* foreground probability — i.e. an alpha matte. They are
therefore legitimate matter backends in their own right (no trimap needed),
not merely segmenters. This module exposes them as ``BaseMatter`` so they
appear in the matter combo of the unified node.

Both backends reuse the already-loaded segmenter classes from
``segmenters.salient_backend`` so we share weights/load paths and avoid
duplicating the HF loading code. The ``coarse_mask`` argument is honoured
as a soft *gating envelope*: the matter alpha is clipped to a slightly-
dilated version of the coarse mask so user prompts upstream still bound
the result (same convention as ``RVMMatter``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..utils import free_vram, interruptible_range, to_bhwc, to_mask
from . import BaseMatter, register

logger = logging.getLogger("MEC.MaskMatting.SalientMatter")


def _have_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except Exception:
        return False


class _SalientMatterBase(BaseMatter):
    """Shared plumbing: delegate inference to the matching segmenter class."""

    NEEDS_TRIMAP = False
    STATUS = "ready" if _have_transformers() else "missing-deps"
    _SEG_CLS_PATH: str = ""    # e.g. "BiRefNetSegmenter"

    def _segmenter(self):
        # Lazy import to avoid a circular import at package load time.
        from ..segmenters import salient_backend as sb
        return getattr(sb, self._SEG_CLS_PATH)

    def load(self) -> None:
        if self._model is not None:
            return
        cls = self._segmenter()
        seg = cls(
            model_name=self.model_name,
            device=self.device,
            precision=self.precision,
            attention=self.attention,
            offload=self.offload,
        )
        seg.load()
        self._model = seg

    @torch.no_grad()
    def matte(
        self,
        image_bhwc: torch.Tensor,
        coarse_mask: Optional[torch.Tensor],
        *,
        trimap: Optional[torch.Tensor] = None,
        edge_radius: int = 4,
        memory_size: int = 8,
    ) -> Dict[str, Any]:
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B = img.shape[0]

            # Run the salient segmenter — its output IS a soft alpha matte.
            arr = img.detach().cpu().numpy()
            outs = []
            for i in interruptible_range(B, label=self.KEY):
                outs.append(self._model._infer_one(arr[i]))
            import numpy as np
            alpha = torch.from_numpy(np.stack(outs, 0)).float().clamp(0.0, 1.0)

            # Optional gating by coarse_mask (dilated by ``edge_radius`` so
            # genuine hair / soft edges are not clipped). Mirrors RVMMatter.
            if coarse_mask is not None:
                m = to_mask(coarse_mask)
                if m.shape == alpha.shape:
                    r = max(1, int(edge_radius))
                    k = 2 * r + 1
                    grow = F.max_pool2d(m.unsqueeze(1), k, 1, r).squeeze(1)
                    alpha = torch.minimum(alpha, grow.cpu())

            return {
                "alpha": alpha,
                "info": {"backend": self.KEY, "frames": int(B),
                         "delegate": self._SEG_CLS_PATH},
            }
        finally:
            free_vram(unload_models=False)


@register
class BiRefNetMatter(_SalientMatterBase):
    KEY = "birefnet"
    DISPLAY = "BiRefNet (matter / soft alpha)"
    MODELS_KEY = "birefnet"
    _SEG_CLS_PATH = "BiRefNetSegmenter"


@register
class RMBG2Matter(_SalientMatterBase):
    KEY = "rmbg"
    DISPLAY = "RMBG-2.0 (matter / soft alpha)"
    MODELS_KEY = "rmbg"
    _SEG_CLS_PATH = "RMBG2Segmenter"
