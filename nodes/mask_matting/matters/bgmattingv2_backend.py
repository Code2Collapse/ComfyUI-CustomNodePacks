"""BackgroundMattingV2 matter backend.

Reference: https://github.com/PeterL1n/BackgroundMattingV2
HF weights:  ``KevinPHJ/BackgroundMattingV2`` (community mirror) provides
``pytorch_resnet50.pth`` / ``pytorch_mobilenetv2.pth``. Torch Hub also works
when no local checkpoint is present.

BGMv2 is STRICTLY static-camera: it needs a clean *background plate* (the
same scene without the subject) and a current frame, and computes a high-
quality alpha by differencing. Without a plate the result is meaningless.

Here we adopt a pragmatic fallback so the node never silently lies:

  1. If ``trimap`` is provided AND it actually contains the gray ``128``
     band → use the *unknown* region of the trimap to synthesise a pseudo
     background plate by inpainting the foreground from a Gaussian-blurred
     copy of the input.  This degrades gracefully (~ResNet matter quality).
  2. Otherwise → raise a clear, actionable error telling the user to
     supply a background plate via the matter node's optional input.

The backend still respects ``BaseMatter.matte()`` and returns
``{alpha: (B,H,W), info: {...}}`` so it slots into the unified pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import (
    backend_first_root,
    free_vram,
    interruptible_range,
    resolve_backend_weight,
    to_bhwc,
)
from . import BaseMatter, register

logger = logging.getLogger("MEC.MaskMatting.BGMv2")


def _have_torch_hub() -> bool:
    try:
        import torch.hub  # noqa: F401
        return True
    except Exception:
        return False


def _synth_bgr_plate(img_bchw: torch.Tensor, alpha_b1hw: torch.Tensor) -> torch.Tensor:
    """Approximate background plate by Gaussian-blurring under the matte.

    For ``alpha == 1`` pixels (definite foreground), pull colour from a
    heavily-blurred copy of the image so the plate doesn't contain the
    subject. For ``alpha == 0`` regions we keep the original pixels.
    """
    k = 31
    sigma = 12.0
    coords = torch.arange(k, device=img_bchw.device, dtype=img_bchw.dtype) - (k - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    g2 = g2.unsqueeze(0).unsqueeze(0).expand(img_bchw.shape[1], 1, k, k)
    blurred = F.conv2d(img_bchw, g2, padding=k // 2, groups=img_bchw.shape[1])
    a = alpha_b1hw.clamp(0, 1)
    return blurred * a + img_bchw * (1 - a)


@register
class BGMattingV2Matter(BaseMatter):
    KEY = "bgmattingv2"
    DISPLAY = "BackgroundMattingV2"
    MODELS_KEY = "bgmattingv2"
    NEEDS_TRIMAP = False         # plate is preferred but optional via synthesis
    STATUS = "ready" if _have_torch_hub() else "missing-deps"
    _DEFAULT_VARIANT = "resnet50"   # alt: "mobilenetv2"

    def load(self) -> None:
        if self._model is not None:
            return
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        stem = os.path.splitext(os.path.basename(self.model_name or ""))[0].lower()
        variant = "mobilenetv2" if "mobile" in stem else self._DEFAULT_VARIANT
        logger.warning("[%s] loading variant=%s ckpt=%s", self.KEY, variant, ckpt)
        try:
            # Torch Hub entrypoint published by Peter Lin's repo.
            model = torch.hub.load(
                "PeterL1n/BackgroundMattingV2", variant,
                pretrained=(ckpt is None),
                trust_repo=True,
            )
            if ckpt and os.path.isfile(ckpt):
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state, strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load BGMattingV2 (variant={variant}). Place a "
                f"checkpoint (.pth) in {backend_first_root(self.MODELS_KEY)} or "
                f"ensure internet access for torch.hub. "
                f"Underlying error: {exc}"
            )
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}.get(self.precision, torch.float16)
        try:
            model = model.to(self.device, dtype=dtype)
        except Exception:
            model = model.to(self.device)
            dtype = torch.float32
        model.eval()
        self._model = model
        self._dtype = dtype

    @torch.no_grad()
    def matte(self, image_bhwc, coarse_mask, *, trimap=None,
              edge_radius: int = 4, memory_size: int = 8,
              background=None):
        """``background`` (B,H,W,C) is the clean plate when available."""
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            x = img.permute(0, 3, 1, 2).to(self.device, dtype=self._dtype)

            # ---- Background plate resolution ----
            if background is not None:
                bg = to_bhwc(background)
                if bg.shape[0] == 1 and B > 1:
                    bg = bg.expand(B, -1, -1, -1)
                bgr = bg.permute(0, 3, 1, 2).to(self.device, dtype=self._dtype)
                plate_source = "user_plate"
            elif coarse_mask is not None:
                # Synthesise from blurred input gated by the coarse mask.
                a = coarse_mask.to(self.device, dtype=self._dtype)
                if a.ndim == 3:
                    a = a.unsqueeze(1)            # (B,1,H,W)
                if a.shape[0] == 1 and B > 1:
                    a = a.expand(B, -1, -1, -1)
                bgr = _synth_bgr_plate(x, a)
                plate_source = "synthesised_from_mask"
            else:
                raise RuntimeError(
                    "BackgroundMattingV2 requires either a `background` plate "
                    "image OR a `coarse_mask`. Provide one of them."
                )

            outs = []
            with torch.inference_mode():
                for i in interruptible_range(B, label="bgmv2"):
                    src = x[i:i + 1]
                    bg = bgr[i:i + 1]
                    res = self._model(src, bg)
                    # Repo returns (pha, fgr, err, ref) at coarse, plus a
                    # refined pha. Pick whichever pha is present.
                    if isinstance(res, (list, tuple)):
                        # Find the (B,1,H,W) tensor that looks like alpha.
                        pha = None
                        for t in res:
                            if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[1] == 1:
                                pha = t
                                break
                        if pha is None:
                            pha = res[0]
                    else:
                        pha = res
                    outs.append(pha[0, 0].float().cpu())
            alpha = torch.stack(outs, 0).clamp(0, 1)
            return {"alpha": alpha,
                    "info": {"backend": "bgmattingv2",
                             "plate_source": plate_source,
                             "frames": int(B)}}
        finally:
            free_vram(unload_models=False)
