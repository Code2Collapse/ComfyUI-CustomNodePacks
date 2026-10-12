"""Model-free matter (L7.91): a guided filter in the trimap's unknown band.

Always available - no weights, no download. What MaskOps falls back to when the chosen matter (ViTMatte, RVM,
MatAnyone, ...) is not installed; it can be picked directly. Approximate and labelled so (info["approximate"]).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from . import BaseMatter, register
from .. import _classic


@register
class GuidedFilterMatter(BaseMatter):
    KEY = "guided_filter"
    DISPLAY = "Guided filter (model-free, approximate)"
    STATUS = "ready"
    MODELS_KEY = ""
    NEEDS_TRIMAP = False
    PARAMS = ()

    def load(self) -> None:
        self._model = True

    def matte(
        self,
        image_bhwc: torch.Tensor,
        coarse_mask: torch.Tensor,
        *,
        trimap: Optional[torch.Tensor] = None,
        edge_radius: int = 4,
        memory_size: int = 8,
        **_kw,
    ) -> Dict[str, Any]:
        img = image_bhwc.detach().float().cpu().clamp(0, 1)
        if trimap is None:
            # ONYX hands a trimap and a zero coarse mask; the cascade may hand only the coarse mask.
            from ..utils import mask_to_trimap
            r = max(1, int(edge_radius))
            trimap = mask_to_trimap(coarse_mask.detach().float().cpu(), dilate=r, erode=r)
        tri = trimap.detach().float().cpu()
        if tri.ndim == 2:
            tri = tri.unsqueeze(0)
        alpha = _classic.guided_matte(img, tri, radius=max(2, int(edge_radius) * 2))
        return {"alpha": alpha, "info": {"backend": _classic.LABEL_MATTE, "approximate": True}}
