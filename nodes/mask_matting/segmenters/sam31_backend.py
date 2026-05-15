"""SAM 3.1 segmenter backend — fully independent.

Uses MEC's own vendored copy of Meta's SAM3 library at
``ComfyUI-CustomNodePacks/third_party/sam3_lib`` — no dependency on the
``comfyui-sam3`` pack, no dependency on the upstream ``sam3`` Python
package.

Weights live under ``ComfyUI/models/sam3.1/``.

Features
--------
* Points (positive + negative) — coords normalised to [0,1].
* Bounding boxes (positive only — model treats label=True as foreground).
* Open-vocabulary text prompts (e.g. "person in red").
* Mask-input refine pass — re-runs detection with the previous mask as
  input. Controlled by ``MEC_SAM31_REFINE`` env var (default off).
* Output is the union of all retained masks (>= confidence_threshold).

License
-------
The vendored library is Meta's SAM3 reference implementation. See
``third_party/sam3_lib`` and ``THIRD_PARTY_LICENSES/SAM3_LICENSE``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ..utils import (
    backend_first_root,
    free_vram,
    interruptible_range,
    resolve_backend_weight,
)
from . import BaseSegmenter, register

logger = logging.getLogger("MEC.MaskMatting.SAM31")


# ──────────────────────────────────────────────────────────────────────
# Vendored library availability check.
# ──────────────────────────────────────────────────────────────────────
def _vendor_root() -> Path:
    # nodes/mask_matting/segmenters/sam31_backend.py -> pack/third_party/sam3_lib
    here = Path(__file__).resolve()
    return here.parents[3] / "third_party" / "sam3_lib"


def _have_vendor() -> bool:
    root = _vendor_root()
    return (root / "__init__.py").is_file() and (root / "model" / "sam3_image_processor.py").is_file()


def _import_vendor():
    """Import vendored sam3_lib via the pack's package namespace."""
    import importlib
    import sys

    pkg = __package__ or ""
    pack_pkg = ".".join(pkg.split(".")[:-3]) if pkg else ""
    if pack_pkg:
        try:
            mod = importlib.import_module(f"{pack_pkg}.third_party.sam3_lib")
            proc = importlib.import_module(f"{pack_pkg}.third_party.sam3_lib.model.sam3_image_processor")
            return mod, proc
        except ImportError:
            pass

    vendor_parent = str(_vendor_root().parent)
    if vendor_parent not in sys.path:
        sys.path.insert(0, vendor_parent)
    mod = importlib.import_module("sam3_lib")
    proc = importlib.import_module("sam3_lib.model.sam3_image_processor")
    return mod, proc


# ──────────────────────────────────────────────────────────────────────
# Backend
# ──────────────────────────────────────────────────────────────────────
@register
class SAM31Segmenter(BaseSegmenter):
    """SAM 3.1 — points + bbox + text + mask-refine, fully independent."""

    KEY = "sam3.1"
    DISPLAY = "SAM 3.1 (independent: text + points + bbox)"
    MODELS_KEY = "sam3.1"
    SUPPORTS_MODES = {"points", "bbox", "text", "auto"}
    STATUS = "ready" if _have_vendor() else "missing-deps"

    DEFAULT_CONF = 0.20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._processor = None
        self._video_predictor = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not _have_vendor():
            raise RuntimeError(
                "SAM 3.1 vendored library not present. Expected at "
                f"{_vendor_root()}. Reinstall ComfyUI-CustomNodePacks."
            )
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"SAM 3.1 checkpoint '{self.model_name}' not under "
                f"{backend_first_root(self.MODELS_KEY)}. Pick a "
                "[preset:sam3.1] entry and tick auto_download."
            )

        mod, proc_mod = _import_vendor()
        Sam3VideoPredictor = mod.Sam3VideoPredictor  # type: ignore[attr-defined]
        Sam3Processor = proc_mod.Sam3Processor  # type: ignore[attr-defined]

        bpe_path = _vendor_root() / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.is_file():
            raise FileNotFoundError(f"BPE tokenizer missing: {bpe_path}")

        logger.warning(
            "[SAM3.1] building model from %s (device=%s precision=%s)",
            ckpt, self.device, self.precision,
        )

        self._video_predictor = Sam3VideoPredictor(
            checkpoint_path=str(ckpt),
            bpe_path=str(bpe_path),
            enable_inst_interactivity=True,
        )
        detector = self._video_predictor.model.detector
        self._processor = Sam3Processor(
            model=detector,
            resolution=1008,
            device=str(self.device),
            confidence_threshold=self.DEFAULT_CONF,
        )
        self._model = self._processor

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(frame_hwc: np.ndarray):
        from PIL import Image
        u8 = (frame_hwc * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(u8)

    @staticmethod
    def _norm_points(points, W: int, H: int) -> List[List[float]]:
        out: List[List[float]] = []
        for p in points or []:
            x = float(p[0]) / max(W, 1)
            y = float(p[1]) / max(H, 1)
            out.append([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])
        return out

    @staticmethod
    def _xyxy_to_cxcywh_norm(box: Tuple[int, int, int, int], W: int, H: int) -> List[float]:
        """Convert pixel xyxy to normalized cxcywh (SAM 3.1 input format)."""
        x0, y0, x1, y1 = (float(v) for v in box)
        # Clamp + ensure x1>x0, y1>y0
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0 = max(0.0, min(float(W), x0))
        x1 = max(0.0, min(float(W), x1))
        y0 = max(0.0, min(float(H), y0))
        y1 = max(0.0, min(float(H), y1))
        cx = (x0 + x1) * 0.5 / max(W, 1)
        cy = (y0 + y1) * 0.5 / max(H, 1)
        w = max(x1 - x0, 1.0) / max(W, 1)
        h = max(y1 - y0, 1.0) / max(H, 1)
        return [cx, cy, w, h]

    def _seg_one(self, frame_hwc: np.ndarray, pos, neg, bbox, neg_bbox, text):
        H, W = frame_hwc.shape[:2]
        pil = self._to_pil(frame_hwc)
        proc = self._processor
        proc.set_confidence_threshold(self.DEFAULT_CONF)
        state = proc.set_image(pil)

        if text and text.strip():
            state = proc.set_text_prompt(text.strip(), state)

        # Combined pos+neg bbox prompts in a single forward pass.
        # SAM 3.1 vendored API takes [cx,cy,w,h] normalized boxes + bool labels
        # (True=include, False=exclude). Native support — no second pass needed.
        boxes_norm: List[List[float]] = []
        box_labels: List[bool] = []
        if bbox is not None:
            boxes_norm.append(self._xyxy_to_cxcywh_norm(bbox, W, H))
            box_labels.append(True)
        if neg_bbox is not None:
            boxes_norm.append(self._xyxy_to_cxcywh_norm(neg_bbox, W, H))
            box_labels.append(False)
        if boxes_norm:
            state = proc.add_multiple_box_prompts(boxes_norm, box_labels, state)

        pos_pts = self._norm_points(pos, W, H)
        neg_pts = self._norm_points(neg, W, H)
        if pos_pts or neg_pts:
            state = proc.add_point_prompt(
                pos_pts + neg_pts,
                [1] * len(pos_pts) + [0] * len(neg_pts),
                state,
            )

        # Optional refine pass.
        if os.environ.get("MEC_SAM31_REFINE", "0") == "1":
            try:
                m_logits = state.get("masks_logits")
                if m_logits is not None and m_logits.numel() > 0:
                    state = proc.add_mask_prompt(m_logits[0].squeeze(0), state)
            except Exception as exc:
                logger.warning("[SAM3.1] refine skipped: %s", exc)

        masks = state.get("masks", None)
        scores = state.get("scores", None)

        if masks is None or len(masks) == 0:
            return np.zeros((H, W), dtype=np.float32), 0.0

        m = masks.to(torch.float32).squeeze(1).max(dim=0).values
        m_np = m.cpu().numpy().astype(np.float32)
        score = float(scores.max().item()) if scores is not None and len(scores) else 0.0
        return m_np, score

    # ── public ──────────────────────────────────────────────────────────
    def segment(self, image_bhwc, *, mode="auto",
                positive_points=None, negative_points=None,
                bbox=None, neg_bbox=None, text_prompt="", frame_annotation=0,
                object_id=0, max_frames=0, memory_size=8,
                start_frame=0, end_frame=-1, individual_objects=False,
                tracking_direction="forward", seed=0):
        try:
            self.load()
            B = image_bhwc.shape[0]
            outs: List[np.ndarray] = []
            score = 0.0
            for i in interruptible_range(B, label="sam3.1"):
                m, s = self._seg_one(
                    image_bhwc[i].cpu().numpy(),
                    positive_points or [], negative_points or [],
                    bbox, neg_bbox, text_prompt,
                )
                outs.append(m)
                score = max(score, s)
            mask_t = torch.from_numpy(np.stack(outs, 0))
            logger.warning(
                "[SAM3.1] segment done — B=%d sum=%.1f score=%.3f text=%r pts=+%d/-%d bbox=%s neg_bbox=%s",
                B, float(mask_t.sum()), score, (text_prompt or "")[:32],
                len(positive_points or []), len(negative_points or []),
                bbox, neg_bbox,
            )
            return {"mask": mask_t.float(), "score": float(score),
                    "info": {"backend": self.KEY}}
        except Exception:
            free_vram()
            raise

    def unload(self) -> None:
        self._processor = None
        self._video_predictor = None
        self._model = None
        free_vram()
