"""Video-object-segmentation (VOS) backends — Cutie, XMem, SeC, VideoMaMa.

These four backends share a common shape:

* They consume a video clip ``(B,H,W,C)`` plus a *seed* mask on a single
  reference frame (``frame_annotation``, default 0) and propagate the
  mask forward (and optionally backward) through the rest of the clip.
* They keep a small memory bank of past frames/features for temporal
  consistency. ``memory_size`` from the node UI feeds straight through.
* They never accept text prompts. Modes supported: ``video``, ``bbox``,
  ``points``, ``auto``.

The reference implementations are NOT pip-installable as a single name:

  - Cutie:    https://github.com/hkchengrex/Cutie       (``pip install -e .``)
  - XMem:     https://github.com/hkchengrex/XMem         (clone + add to PYTHONPATH)
  - SeC:      https://github.com/OpenIXCLab/SeC          (clone)
  - VideoMaMa: https://github.com/yyk-wew/VideoMamba     (clone)

To stay honest we report ``STATUS = "missing-deps"`` until the upstream
module imports cleanly. ``load()`` raises a clear, actionable error so
the user immediately knows what to install.

The seeding strategy if no explicit mask is supplied:

  * ``bbox`` → filled rectangle on the reference frame.
  * ``positive_points`` → small disc (radius = 16 px) per point.
  * neither → ValueError; we will not silently propagate an empty mask.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

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
from . import BaseSegmenter, register

logger = logging.getLogger("MEC.MaskMatting.Video")


# ──────────────────────────────────────────────────────────────────────
# Seeding helpers
# ──────────────────────────────────────────────────────────────────────
def _seed_mask_from_prompt(
    H: int, W: int,
    *,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    positive_points: Optional[List[Tuple[float, float]]] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build an (H,W) binary float mask from user prompts."""
    m = torch.zeros((H, W), dtype=torch.float32, device=device)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(W - 1, int(x1)))
        x2 = max(0, min(W,     int(x2)))
        y1 = max(0, min(H - 1, int(y1)))
        y2 = max(0, min(H,     int(y2)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1.0
        return m
    if positive_points:
        ys = torch.arange(H, device=device).view(-1, 1)
        xs = torch.arange(W, device=device).view(1, -1)
        r2 = 16 * 16
        for (px, py) in positive_points:
            d2 = (xs - float(px)) ** 2 + (ys - float(py)) ** 2
            m = torch.maximum(m, (d2 <= r2).float())
        return m
    raise ValueError(
        "Video segmenter needs a seed: provide either `bbox` or "
        "`positive_points` on `frame_annotation`."
    )


def _propagate_indices(B: int, ref: int, direction: str) -> List[int]:
    """Frame indices to visit, in order. Always start with ref."""
    ref = max(0, min(B - 1, int(ref)))
    forward  = list(range(ref, B))
    backward = list(range(ref - 1, -1, -1)) if direction != "forward" else []
    return forward + backward if direction == "bidirectional" else (
        list(reversed(range(0, ref + 1))) if direction == "backward" else forward
    )


def _pack_output(
    masks_by_frame: Dict[int, torch.Tensor], B: int, H: int, W: int,
    score: float, info: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble final (B,H,W) tensor from per-frame dict; fill blanks with zeros."""
    out = torch.zeros((B, H, W), dtype=torch.float32)
    for i, m in masks_by_frame.items():
        if 0 <= i < B:
            out[i] = m.detach().to("cpu", torch.float32).clamp(0, 1)
    return {"mask": out, "score": float(score), "info": info}


# ──────────────────────────────────────────────────────────────────────
# Cutie (hkchengrex/Cutie)
# ──────────────────────────────────────────────────────────────────────
def _have_cutie() -> bool:
    try:
        import importlib.util as _iu
        for mod in ("cutie", "cutie.model.cutie", "cutie.inference.inference_core"):
            if _iu.find_spec(mod) is None:
                return False
        return True
    except Exception:
        return False


@register
class CutieSegmenter(BaseSegmenter):
    KEY = "cutie"
    DISPLAY = "Cutie (video object segmentation)"
    MODELS_KEY = "cutie"
    SUPPORTS_MODES = {"video", "bbox", "points", "auto"}
    STATUS = "ready" if _have_cutie() else "missing-deps"

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "Cutie is not installed. Clone https://github.com/hkchengrex/Cutie "
                "and `pip install -e .` inside the ComfyUI python env."
            )
        from cutie.model.cutie import CUTIE                # type: ignore
        from cutie.inference.inference_core import InferenceCore  # type: ignore
        try:
            from cutie.config.config import get_config     # type: ignore
        except Exception:
            from hydra import compose, initialize          # type: ignore
            get_config = None
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Cutie checkpoint '{self.model_name}' not under "
                f"{backend_first_root(self.MODELS_KEY)}."
            )
        cfg = get_config("eval_config.yaml") if get_config else None
        net = CUTIE(cfg).to(self.device).eval()
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        net.load_weights(sd) if hasattr(net, "load_weights") else net.load_state_dict(sd, strict=False)
        self._net = net
        self._cfg = cfg
        self._InferenceCore = InferenceCore
        self._model = net  # sentinel for is-loaded

    @torch.inference_mode()
    def segment(self, image_bhwc, **kw):
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            ref = int(kw.get("frame_annotation", 0))
            mem_size = int(kw.get("memory_size", 8))
            direction = str(kw.get("tracking_direction", "forward"))

            seed = _seed_mask_from_prompt(
                H, W,
                bbox=kw.get("bbox"),
                positive_points=kw.get("positive_points"),
                device=torch.device(self.device),
            )

            processor = self._InferenceCore(self._net, cfg=self._cfg)
            if hasattr(processor, "max_internal_size"):
                processor.max_internal_size = max(480, min(H, W))
            # Memory configs that exist in Cutie's InferenceCore
            for attr, val in (("max_mem_frames", mem_size),
                              ("mem_every", max(1, mem_size // 2))):
                if hasattr(processor, attr):
                    setattr(processor, attr, val)

            order = _propagate_indices(B, ref, direction)
            out: Dict[int, torch.Tensor] = {}
            objects = [1]
            ref_passed = False
            for idx in interruptible_range(len(order), label="cutie"):
                t = order[idx]
                frame = img[t].permute(2, 0, 1).to(self.device, dtype=torch.float32)
                if t == ref and not ref_passed:
                    prob = processor.step(frame, mask=seed.to(self.device).long(),
                                          objects=objects)
                    ref_passed = True
                else:
                    prob = processor.step(frame)
                # prob: (num_obj+1, H, W) — channel 1 = our object
                if prob.ndim == 3 and prob.shape[0] >= 2:
                    out[t] = prob[1].detach().cpu()
                else:
                    out[t] = (prob.argmax(0) == 1).float().detach().cpu()
            return _pack_output(out, B, H, W, score=0.85,
                                info={"backend": "cutie",
                                      "ref_frame": ref,
                                      "memory_size": mem_size,
                                      "direction": direction,
                                      "frames": B})
        finally:
            free_vram(unload_models=False)


# ──────────────────────────────────────────────────────────────────────
# XMem (hkchengrex/XMem)
# ──────────────────────────────────────────────────────────────────────
def _have_xmem() -> bool:
    try:
        import importlib.util as _iu
        # XMem isn't a packaged module; users clone the repo and add it
        # to PYTHONPATH so ``model.network`` resolves.
        for mod in ("model.network", "inference.inference_core"):
            if _iu.find_spec(mod) is None:
                return False
        return True
    except Exception:
        return False


@register
class XMemSegmenter(BaseSegmenter):
    KEY = "xmem"
    DISPLAY = "XMem (video segmentation, long-term memory)"
    MODELS_KEY = "xmem"
    SUPPORTS_MODES = {"video", "bbox", "points", "auto"}
    STATUS = "ready" if _have_xmem() else "missing-deps"

    _DEFAULT_CFG = {
        "top_k": 30,
        "mem_every": 5,
        "deep_update_every": -1,
        "enable_long_term": True,
        "enable_long_term_count_usage": True,
        "num_prototypes": 128,
        "min_mid_term_frames": 5,
        "max_mid_term_frames": 10,
        "max_long_term_elements": 10000,
        "save_scores": False,
    }

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "XMem not on PYTHONPATH. Clone https://github.com/hkchengrex/XMem "
                "into a folder on PYTHONPATH (the repo is not a pip package — "
                "you need its top-level `model/` and `inference/` dirs importable)."
            )
        from model.network import XMem                          # type: ignore
        from inference.inference_core import InferenceCore       # type: ignore
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"XMem checkpoint '{self.model_name}' not under "
                f"{backend_first_root(self.MODELS_KEY)}."
            )
        cfg = dict(self._DEFAULT_CFG)
        net = XMem(cfg, ckpt).to(self.device).eval()
        self._net = net
        self._cfg = cfg
        self._InferenceCore = InferenceCore
        self._model = net

    @torch.inference_mode()
    def segment(self, image_bhwc, **kw):
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            ref = int(kw.get("frame_annotation", 0))
            mem_size = int(kw.get("memory_size", 8))
            direction = str(kw.get("tracking_direction", "forward"))

            seed = _seed_mask_from_prompt(
                H, W,
                bbox=kw.get("bbox"),
                positive_points=kw.get("positive_points"),
                device=torch.device(self.device),
            )

            cfg = dict(self._cfg, mem_every=max(1, mem_size // 2))
            processor = self._InferenceCore(self._net, config=cfg)
            processor.set_all_labels([1])

            order = _propagate_indices(B, ref, direction)
            out: Dict[int, torch.Tensor] = {}
            ref_passed = False
            for k in interruptible_range(len(order), label="xmem"):
                t = order[k]
                frame = img[t].permute(2, 0, 1).to(self.device, dtype=torch.float32)
                if t == ref and not ref_passed:
                    prob = processor.step(frame, mask=seed.to(self.device).long(),
                                          valid_labels=[1])
                    ref_passed = True
                else:
                    prob = processor.step(frame)
                # XMem prob shape: (num_obj+1, H, W)
                if prob.ndim == 3 and prob.shape[0] >= 2:
                    out[t] = prob[1].detach().cpu()
                else:
                    out[t] = (prob.argmax(0) == 1).float().detach().cpu()
            return _pack_output(out, B, H, W, score=0.83,
                                info={"backend": "xmem",
                                      "ref_frame": ref,
                                      "memory_size": mem_size,
                                      "direction": direction,
                                      "frames": B})
        finally:
            free_vram(unload_models=False)


# ──────────────────────────────────────────────────────────────────────
# SeC (OpenIXCLab/SeC) — Segment Concept
# ──────────────────────────────────────────────────────────────────────
def _have_sec() -> bool:
    try:
        import importlib.util as _iu
        return _iu.find_spec("sec") is not None or _iu.find_spec("SeC") is not None
    except Exception:
        return False


@register
class SeCSegmenter(BaseSegmenter):
    KEY = "sec"
    DISPLAY = "SeC (Segment Concept, video)"
    MODELS_KEY = "sec"
    SUPPORTS_MODES = {"video", "bbox", "points", "auto"}
    STATUS = "ready" if _have_sec() else "missing-deps"

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "SeC not installed. Clone https://github.com/OpenIXCLab/SeC and "
                "add it to PYTHONPATH (or `pip install -e .` if a setup.py is present)."
            )
        # SeC ships its own builder; try the documented public entry points.
        try:
            from sec.build import build_sec_predictor          # type: ignore
        except Exception:
            from SeC.build import build_sec_predictor          # type: ignore
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"SeC checkpoint '{self.model_name}' not under "
                f"{backend_first_root(self.MODELS_KEY)}."
            )
        self._predictor = build_sec_predictor(ckpt_path=ckpt, device=self.device)
        self._model = self._predictor

    @torch.inference_mode()
    def segment(self, image_bhwc, **kw):
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            ref = int(kw.get("frame_annotation", 0))
            mem_size = int(kw.get("memory_size", 8))
            direction = str(kw.get("tracking_direction", "forward"))

            seed = _seed_mask_from_prompt(
                H, W,
                bbox=kw.get("bbox"),
                positive_points=kw.get("positive_points"),
                device=torch.device(self.device),
            )

            # SeC public API has two flavours; support both.
            pred = self._predictor
            if hasattr(pred, "init_state") and hasattr(pred, "propagate_in_video"):
                state = pred.init_state(images=img.permute(0, 3, 1, 2)
                                         .to(self.device),
                                         memory_size=mem_size)
                pred.add_new_mask(state, frame_idx=ref, obj_id=1,
                                  mask=seed.to(self.device))
                out: Dict[int, torch.Tensor] = {}
                for fi, ids, probs in pred.propagate_in_video(state,
                                                              direction=direction):
                    if probs.ndim == 3 and probs.shape[0] >= 1:
                        out[int(fi)] = probs[0].detach().cpu()
            elif hasattr(pred, "step"):
                order = _propagate_indices(B, ref, direction)
                out = {}
                ref_passed = False
                for k in interruptible_range(len(order), label="sec"):
                    t = order[k]
                    frame = img[t].permute(2, 0, 1).to(self.device)
                    if t == ref and not ref_passed:
                        prob = pred.step(frame, mask=seed.to(self.device))
                        ref_passed = True
                    else:
                        prob = pred.step(frame)
                    out[t] = (prob if prob.ndim == 2 else prob[0]).float().detach().cpu()
            else:
                raise RuntimeError(
                    "Loaded SeC predictor exposes neither `init_state`/`propagate_in_video` "
                    "nor `step()`. The installed SeC fork is unsupported."
                )
            return _pack_output(out, B, H, W, score=0.80,
                                info={"backend": "sec",
                                      "ref_frame": ref,
                                      "memory_size": mem_size,
                                      "direction": direction,
                                      "frames": B})
        finally:
            free_vram(unload_models=False)


# ──────────────────────────────────────────────────────────────────────
# VideoMaMa (Mamba-based VOS, yyk-wew/VideoMamba)
# ──────────────────────────────────────────────────────────────────────
def _have_videomama() -> bool:
    try:
        import importlib.util as _iu
        for name in ("videomamba", "VideoMamba", "video_mamba"):
            if _iu.find_spec(name) is not None:
                return True
        return False
    except Exception:
        return False


@register
class VideoMaMaSegmenter(BaseSegmenter):
    KEY = "videomama"
    DISPLAY = "VideoMaMa (Mamba-based VOS)"
    MODELS_KEY = "videomama"
    SUPPORTS_MODES = {"video", "bbox", "points", "auto"}
    STATUS = "ready" if _have_videomama() else "missing-deps"

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "VideoMaMa not installed. Clone the Mamba VOS reference "
                "implementation (e.g. https://github.com/yyk-wew/VideoMamba "
                "or https://github.com/OpenGVLab/VideoMamba) and add it to "
                "PYTHONPATH. Mamba also needs `pip install mamba-ssm causal-conv1d`."
            )
        try:
            from videomamba.inference import build_predictor       # type: ignore
        except Exception:
            try:
                from VideoMamba.inference import build_predictor   # type: ignore
            except Exception:
                from video_mamba.inference import build_predictor  # type: ignore
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"VideoMaMa checkpoint '{self.model_name}' not under "
                f"{backend_first_root(self.MODELS_KEY)}."
            )
        self._predictor = build_predictor(ckpt_path=ckpt, device=self.device)
        self._model = self._predictor

    @torch.inference_mode()
    def segment(self, image_bhwc, **kw):
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            ref = int(kw.get("frame_annotation", 0))
            mem_size = int(kw.get("memory_size", 8))
            direction = str(kw.get("tracking_direction", "forward"))

            seed = _seed_mask_from_prompt(
                H, W,
                bbox=kw.get("bbox"),
                positive_points=kw.get("positive_points"),
                device=torch.device(self.device),
            )

            pred = self._predictor
            order = _propagate_indices(B, ref, direction)
            out: Dict[int, torch.Tensor] = {}
            if hasattr(pred, "predict_video"):
                # Some impls expose a one-shot batched API.
                vid = img.permute(0, 3, 1, 2).to(self.device)
                seeds = {ref: seed.to(self.device)}
                result = pred.predict_video(vid, seeds=seeds,
                                            memory_size=mem_size,
                                            direction=direction)
                # Expect (B,H,W) probability tensor in [0,1]
                if torch.is_tensor(result):
                    for i in range(B):
                        out[i] = result[i].detach().cpu()
                elif isinstance(result, dict):
                    for i, m in result.items():
                        out[int(i)] = m.detach().cpu()
            else:
                ref_passed = False
                for k in interruptible_range(len(order), label="videomama"):
                    t = order[k]
                    frame = img[t].permute(2, 0, 1).to(self.device)
                    if t == ref and not ref_passed:
                        prob = pred.step(frame, mask=seed.to(self.device))
                        ref_passed = True
                    else:
                        prob = pred.step(frame)
                    out[t] = (prob if prob.ndim == 2 else prob[0]).float().detach().cpu()
            return _pack_output(out, B, H, W, score=0.78,
                                info={"backend": "videomama",
                                      "ref_frame": ref,
                                      "memory_size": mem_size,
                                      "direction": direction,
                                      "frames": B})
        finally:
            free_vram(unload_models=False)
