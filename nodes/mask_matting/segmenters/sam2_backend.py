"""SAM 2.1 segmenter backend (image + video).

Honors the user's permanent rules:
  * folder_paths.get_filename_list("sam2") for weights
  * try/finally + free_vram on error
  * interrupt-aware iteration on video
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils import (
    backend_first_root,
    free_vram,
    interruptible_range,
    list_backend_files,
    resolve_backend_weight,
)
from . import BaseSegmenter, register

logger = logging.getLogger("MEC.MaskMatting.SAM2")


def _have_sam2() -> bool:
    try:
        import sam2  # noqa: F401
        return True
    except ImportError:
        return False


@register
class SAM2Segmenter(BaseSegmenter):
    KEY = "sam2.1"
    DISPLAY = "SAM 2.1 (image + video)"
    MODELS_KEY = "sam2"
    SUPPORTS_MODES = {"points", "bbox", "auto", "video"}
    STATUS = "ready" if _have_sam2() else "missing-deps"

    # Default config files shipped with the sam2 package.
    _CFG_BY_NAME = {
        "sam2.1_hiera_tiny":      "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small":     "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large":     "configs/sam2.1/sam2.1_hiera_l.yaml",
        # SAM 2.0 fallbacks
        "sam2_hiera_tiny":        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_small":       "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_base_plus":   "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_large":       "configs/sam2/sam2_hiera_l.yaml",
    }

    @staticmethod
    def _cfg_for(name: str) -> str:
        stem = os.path.splitext(os.path.basename(name))[0]
        return SAM2Segmenter._CFG_BY_NAME.get(stem, "configs/sam2.1/sam2.1_hiera_l.yaml")

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError("SAM2 not installed. `pip install sam2` (or `git+https://github.com/facebookresearch/sam2`).")
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"SAM2 checkpoint '{self.model_name}' not found under {backend_first_root(self.MODELS_KEY)}."
            )
        cfg = self._cfg_for(self.model_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(self.precision, torch.float16)
        # Bypass SAM2's _load_checkpoint entirely — it hard-codes
        # `torch.load(..., weights_only=True)` and assumes a `.pt` pickle. The
        # KJ-distributed weights are `.safetensors`, which makes torch.load
        # mis-read header bytes as a legacy pickle protocol (e.g. "27"). We
        # build the model with no checkpoint and load the state dict ourselves,
        # supporting both formats.
        with torch.inference_mode():
            self._image_model = build_sam2(cfg, ckpt_path=None, device=self.device)
            sd = self._read_state_dict(ckpt)
            self._load_state_dict_lenient(self._image_model, sd)
            self._predictor = SAM2ImagePredictor(self._image_model)
            self._video_predictor = None  # lazy-build on video call

            def _build_video():
                vp = build_sam2_video_predictor(cfg, ckpt_path=None, device=self.device)
                self._load_state_dict_lenient(vp, self._read_state_dict(ckpt))
                return vp
            self._video_builder = _build_video
        self._dtype = dtype
        self._model = self._predictor

    @staticmethod
    def _read_state_dict(path: str) -> Dict[str, torch.Tensor]:
        """Load a SAM2 checkpoint as a flat ``{name: tensor}`` dict.

        Supports ``.safetensors`` (preferred — what KJ ships) and ``.pt``
        / ``.pth`` torch pickles. Strips a leading ``model.`` prefix that
        appears in some redistributions.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(path, device="cpu")
        else:
            try:
                obj = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            sd = obj.get("model", obj) if isinstance(obj, dict) else obj
        # Strip "model." prefix if every key has it
        if sd and all(k.startswith("model.") for k in sd.keys()):
            sd = {k[len("model."):]: v for k, v in sd.items()}
        return sd

    @staticmethod
    def _load_state_dict_lenient(model: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # SAM2 has a few buffers reinitialized from cfg; tolerate up to a
        # handful of mismatched keys silently, log the rest.
        if len(missing) > 8 or len(unexpected) > 8:
            import logging as _lg
            _lg.getLogger("MEC.MaskMatting").warning(
                "[SAM2] state_dict mismatch: missing=%d unexpected=%d (first missing=%s)",
                len(missing), len(unexpected), missing[:3]
            )

    def _segment_image(self, image_hwc: np.ndarray, pos, neg, bbox) -> Tuple[np.ndarray, float]:
        img_u8 = (image_hwc * 255).clip(0, 255).astype(np.uint8)
        with torch.inference_mode(), torch.autocast(self.device, dtype=self._dtype, enabled=(self.device == "cuda")):
            self._predictor.set_image(img_u8)
            point_coords = None
            point_labels = None
            if pos or neg:
                pts = list(pos) + list(neg)
                lbls = [1] * len(pos) + [0] * len(neg)
                point_coords = np.array(pts, dtype=np.float32)
                point_labels = np.array(lbls, dtype=np.int64)
            box = None
            if bbox is not None:
                box = np.array(bbox, dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
        # Smart pick — when neg points are supplied AND the user enabled
        # auto_disambiguate (default True), favour the candidate that
        # excludes them; otherwise fall back to highest score.
        use_smart = bool(getattr(self, "auto_disambiguate", True))
        if neg and use_smart:
            try:
                from .._auto_quality import smart_pick_sam_mask
                H, W = image_hwc.shape[:2]
                best, _details = smart_pick_sam_mask(
                    masks, scores, list(pos), list(neg), H, W,
                )
            except Exception:
                best = int(np.argmax(scores))
        else:
            best = int(np.argmax(scores))
        return masks[best].astype(np.float32), float(scores[best])

    def _segment_video(self, frames_bhwc: torch.Tensor, pos, neg, bbox,
                       frame_annotation: int, object_id: int,
                       max_frames: int, start_frame: int, end_frame: int,
                       tracking_direction: str) -> Tuple[np.ndarray, float]:
        import tempfile
        from PIL import Image

        if self._video_predictor is None:
            self._video_predictor = self._video_builder()
        B = frames_bhwc.shape[0]
        end = B - 1 if end_frame < 0 else min(end_frame, B - 1)
        start = max(0, min(start_frame, end))
        sub = frames_bhwc[start:end + 1]
        if max_frames > 0 and sub.shape[0] > max_frames:
            sub = sub[:max_frames]
        N = sub.shape[0]

        # SAM2 video predictor wants a frame folder of JPEGs.
        with tempfile.TemporaryDirectory(prefix="mec_sam2_") as tmp:
            for i in interruptible_range(N, label="export-frame"):
                Image.fromarray((sub[i].cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(tmp, f"{i:06d}.jpg"), quality=95,
                )
            with torch.inference_mode(), torch.autocast(self.device, dtype=self._dtype, enabled=(self.device == "cuda")):
                state = self._video_predictor.init_state(video_path=tmp)
                ann_frame = max(0, min(frame_annotation - start, N - 1))
                if pos or neg:
                    pts = np.array(list(pos) + list(neg), dtype=np.float32)
                    lbl = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
                    self._video_predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=ann_frame,
                        obj_id=int(object_id), points=pts, labels=lbl,
                    )
                if bbox is not None:
                    self._video_predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=ann_frame,
                        obj_id=int(object_id), box=np.array(bbox, dtype=np.float32),
                    )
                reverse = (tracking_direction == "backward")
                masks_out: List[np.ndarray] = [None] * N  # type: ignore
                for f_idx, _obj_ids, mask_logits in self._video_predictor.propagate_in_video(
                    state, reverse=reverse,
                ):
                    m = (mask_logits[0] > 0).cpu().numpy().astype(np.float32)
                    if m.ndim == 3:
                        m = m[0]
                    masks_out[f_idx] = m
                if tracking_direction == "bidirectional":
                    for f_idx, _obj_ids, mask_logits in self._video_predictor.propagate_in_video(
                        state, reverse=True,
                    ):
                        m = (mask_logits[0] > 0).cpu().numpy().astype(np.float32)
                        if m.ndim == 3:
                            m = m[0]
                        if masks_out[f_idx] is None:
                            masks_out[f_idx] = m
        # pad missing
        H, W = sub.shape[1], sub.shape[2]
        full = np.zeros((B, H, W), dtype=np.float32)
        for i, m in enumerate(masks_out):
            if m is not None:
                full[start + i] = m
        return full, 1.0

    def segment(self, image_bhwc, *, mode="auto",
                positive_points=None, negative_points=None,
                bbox=None, neg_bbox=None, text_prompt="", frame_annotation=0,
                object_id=0, max_frames=0, memory_size=8,
                start_frame=0, end_frame=-1, individual_objects=False,
                tracking_direction="forward", seed=0):
        try:
            self.load()
            B = image_bhwc.shape[0]
            pos = positive_points or []
            neg = negative_points or []
            if mode == "auto":
                mode = "video" if B > 1 else ("bbox" if bbox is not None else "points")

            if mode == "video" and B > 1:
                masks, score = self._segment_video(
                    image_bhwc, pos, neg, bbox, frame_annotation, object_id,
                    max_frames, start_frame, end_frame, tracking_direction,
                )
                mask_t = torch.from_numpy(masks)
            else:
                # per-frame image segmentation
                outs = []
                for i in interruptible_range(B, label="sam2"):
                    m, s = self._segment_image(image_bhwc[i].cpu().numpy(), pos, neg, bbox)
                    outs.append(m)
                    score = s
                mask_t = torch.from_numpy(np.stack(outs, axis=0))
            return {"mask": mask_t.float(), "score": float(score), "info": {"backend": self.KEY}}
        except Exception:
            free_vram()
            raise
