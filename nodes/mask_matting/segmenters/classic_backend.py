"""Model-free segmenter (L7.91): GrabCut on the annotated frame, DIS optical flow + GrabCut for the rest of a clip.

Always available - OpenCV only, no weights, no download. It is what MaskOps falls back to when the chosen model is
not installed, and it can be picked directly. Results are approximate and labelled so (info["approximate"]).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import BaseSegmenter, register
from .. import _classic


@register
class ClassicSegmenter(BaseSegmenter):
    KEY = "classic"
    DISPLAY = "Model-free (GrabCut, approximate)"
    STATUS = "ready"
    MODELS_KEY = ""
    SUPPORTS_MODES = {"points", "bbox", "video", "auto"}
    PARAMS = ()

    def load(self) -> None:
        self._model = True

    def segment(
        self,
        image_bhwc: torch.Tensor,
        *,
        mode: str = "auto",
        positive_points: Optional[List[Tuple[float, float]]] = None,
        negative_points: Optional[List[Tuple[float, float]]] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        neg_bbox: Optional[Tuple[int, int, int, int]] = None,
        text_prompt: str = "",
        frame_annotation: int = 0,
        object_id: int = 0,
        max_frames: int = 0,
        memory_size: int = 8,
        start_frame: int = 0,
        end_frame: int = -1,
        individual_objects: bool = False,
        tracking_direction: str = "forward",
        seed: int = 0,
    ) -> Dict[str, Any]:
        pos = list(positive_points or [])
        neg = list(negative_points or [])
        if not pos and not neg and bbox is None:
            if text_prompt and text_prompt.strip():
                raise ValueError(_classic.TEXT_NEEDS_MODEL)
            raise ValueError("The model-free segmenter needs a point or a box.")
        x = image_bhwc.detach().float().cpu().clamp(0, 1)
        B, H, W = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        frames = [(x[i, ..., :3].numpy() * 255.0).astype(np.uint8) for i in range(B)]
        key = min(B - 1, max(0, int(frame_annotation)))
        if B == 1:
            m, score = _classic.grabcut_segment(frames[0], pos, neg, bbox)
            masks = m[None]
            scores = [score]
        else:
            tr = _classic.ClassicVideoTracker()
            state = tr.init_state(frames)
            pts, labels = [], []
            if bbox is not None:
                pts += [[bbox[0] / W, bbox[1] / H], [bbox[2] / W, bbox[3] / H]]
                labels += [2, 3]
            pts += [[px / W, py / H] for px, py in pos] + [[px / W, py / H] for px, py in neg]
            labels += [1] * len(pos) + [0] * len(neg)
            tr.add_prompt(state, key, points=pts, point_labels=labels, obj_id=1)
            masks = np.zeros((B, H, W), np.float32)
            scores = []
            d = str(tracking_direction or "forward").lower()
            n = int(max_frames) if int(max_frames or 0) > 0 else None
            runs = [False, True] if d.startswith("bi") else [d.startswith("back")]
            for reverse in runs:
                for fi, out in tr.propagate_in_video(state, key, n, reverse):
                    masks[fi] = out["out_binary_masks"][0].astype(np.float32)
                    scores.append(float(out["out_probs"][0]))
            tr.reset_state(state)
        return {
            "mask": torch.from_numpy(masks).float(),
            "score": float(np.mean(scores)) if scores else 0.0,
            "info": {"backend": _classic.LABEL_SEGMENT, "approximate": True, "frames": B},
        }
