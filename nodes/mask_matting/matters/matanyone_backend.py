"""MatAnyone — video matting backend (pq-yang/MatAnyone).

Reference:
  Paper:  "MatAnyone: Stable Video Matting with Consistent Memory Propagation"
  Repo:   https://github.com/pq-yang/MatAnyone
  PyPI:   `pip install matanyone` (community wheels) — preferred.

The model takes a clip + a first-frame coarse mask and returns a temporally
consistent alpha. We honor the unified ``BaseMatter`` contract:

  - ``image_bhwc``:    (B,H,W,C) float in [0,1]
  - ``coarse_mask``:  (B,H,W) float in [0,1] — only the first frame is read
                       as the seed; remaining frames are ignored if all zero
  - ``memory_size``: passed straight to MatAnyone's memory bank.

If the upstream package is missing we report ``STATUS = "missing-deps"`` and
``load()`` raises a clear install hint.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

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

logger = logging.getLogger("MEC.MaskMatting.MatAnyone")


def _have_matanyone() -> bool:
    try:
        import importlib.util as _iu
        return _iu.find_spec("matanyone") is not None
    except Exception:
        return False


@register
class MatAnyoneMatter(BaseMatter):
    KEY = "matanyone"
    DISPLAY = "MatAnyone (video matting)"
    MODELS_KEY = "matanyone"
    NEEDS_TRIMAP = False
    STATUS = "ready" if _have_matanyone() else "missing-deps"

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "MatAnyone is not installed. Run "
                "`pip install matanyone` "
                "or clone https://github.com/pq-yang/MatAnyone and "
                "`pip install -e .` inside the ComfyUI Python env."
            )
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"MatAnyone checkpoint '{self.model_name}' not under "
                f"{backend_first_root(self.MODELS_KEY)}."
            )
        # The upstream package exposes a few subtly different entry
        # points across versions. Probe them in order.
        try:
            from matanyone.utils.get_default_model import get_matanyone_model  # type: ignore
            from matanyone.inference.inference_core import InferenceCore        # type: ignore
            net = get_matanyone_model(ckpt)
            net = net.to(self.device).eval()
            processor = InferenceCore(net, cfg=None)
        except Exception:
            try:
                from matanyone import InferenceCore                              # type: ignore
                processor = InferenceCore(ckpt) if os.path.isfile(ckpt) \
                            else InferenceCore("PeterL1n/MatAnyone")
            except Exception as exc:
                raise RuntimeError(
                    f"Installed MatAnyone build does not expose a known entry "
                    f"point. Tried `matanyone.utils.get_default_model` and "
                    f"`matanyone.InferenceCore`. Underlying error: {exc}"
                )
        self._processor = processor
        self._model = processor

    @torch.inference_mode()
    def matte(self, image_bhwc, coarse_mask,
              *, trimap=None, edge_radius: int = 4, memory_size: int = 8):
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape

            if coarse_mask is None:
                raise RuntimeError(
                    "MatAnyone needs a first-frame `coarse_mask` to seed memory."
                )
            cm = coarse_mask
            if cm.ndim == 4:
                cm = cm[..., 0]                          # (B,H,W,1) → (B,H,W)
            if cm.shape[-2:] != (H, W):
                cm = F.interpolate(cm[:, None].float(), size=(H, W),
                                   mode="bilinear", align_corners=False)[:, 0]
            seed = cm[0].to(self.device, dtype=torch.float32).clamp(0, 1)

            proc = self._processor

            # ---- Single-shot batched path (preferred) ----
            if hasattr(proc, "process_video"):
                vid = img.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
                try:
                    out = proc.process_video(vid, seed, memory_size=memory_size)
                except TypeError:
                    out = proc.process_video(vid, seed)
                if torch.is_tensor(out):
                    alpha = out.detach().cpu().float()
                    if alpha.ndim == 4 and alpha.shape[1] == 1:
                        alpha = alpha[:, 0]
                elif isinstance(out, (list, tuple)):
                    alpha = torch.stack([t.detach().cpu().float() for t in out], 0)
                else:
                    raise RuntimeError(
                        f"MatAnyone.process_video returned unexpected type {type(out)}"
                    )
                return {"alpha": alpha.clamp(0, 1),
                        "info": {"backend": "matanyone",
                                 "memory_size": memory_size,
                                 "frames": int(B),
                                 "path": "process_video"}}

            # ---- Step-by-step fallback ----
            if hasattr(proc, "step"):
                outs = []
                for i in interruptible_range(B, label="matanyone"):
                    frame = img[i].permute(2, 0, 1).to(self.device,
                                                       dtype=torch.float32)
                    if i == 0:
                        pha = proc.step(frame, mask=seed, first_frame_pred=True)
                    else:
                        pha = proc.step(frame)
                    if torch.is_tensor(pha):
                        if pha.ndim == 4:
                            pha = pha[0, 0]
                        elif pha.ndim == 3:
                            pha = pha[0]
                    outs.append(pha.detach().cpu().float())
                alpha = torch.stack(outs, 0).clamp(0, 1)
                return {"alpha": alpha,
                        "info": {"backend": "matanyone",
                                 "memory_size": memory_size,
                                 "frames": int(B),
                                 "path": "step"}}

            raise RuntimeError(
                "Loaded MatAnyone processor exposes neither `process_video` "
                "nor `step()`. The installed build is unsupported."
            )
        finally:
            free_vram(unload_models=False)
