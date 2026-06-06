"""Base classes and registry for segmenter backends."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("MEC.MaskMatting.Segmenters")


class BaseSegmenter:
    """Abstract segmenter. Subclasses implement ``segment``."""

    KEY: str = "base"
    DISPLAY: str = "Base"
    STATUS: str = "experimental"          # "ready" | "experimental" | "missing-deps"
    MODELS_KEY: str = ""
    SUPPORTS_MODES: set = set()           # {"points", "bbox", "text", "video", "auto"}

    def __init__(self, model_name: str = "", device: str = "cuda",
                 precision: str = "fp16", attention: str = "auto",
                 offload: str = "none"):
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.attention = attention
        self.offload = offload
        self._model: Any = None

    def is_available(self) -> bool:
        return self.STATUS == "ready"

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self._model = None

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
        """Return dict with at least:
            ``mask`` (B,H,W) torch.Tensor in [0,1]
            ``score`` float in [0,1]
            ``info`` dict
        """
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, type] = {}


def register(cls: type) -> type:
    _REGISTRY[cls.KEY] = cls
    return cls


def all_segmenters() -> Dict[str, type]:
    return dict(_REGISTRY)


def get_segmenter_cls(key: str) -> Optional[type]:
    return _REGISTRY.get(key)


def list_keys(installed_only: bool = False) -> List[str]:
    keys: List[str] = []
    for k, cls in _REGISTRY.items():
        if installed_only and cls.STATUS != "ready":
            continue
        keys.append(k)
    return keys


# Eager import each backend so they self-register. Wrapped in try so a
# broken optional dep can't take down the whole pack, but every failure
# is forwarded to the C2C registry so the user can see exactly what is
# missing (instead of nodes silently disappearing — the #1 cause of the
# "everything is a stub" perception per ideas_summary.md §2.1).
_HINT_BY_MODULE = {
    "sam2_backend":         "Install `sam2` (Meta SAM 2) and place sam2 weights under models/sam2/.",
    "sam3_backend":         "Install `sam-3` and place SAM 3 weights under models/sam3/.",
    "sam31_backend":        "Install `sam-3.1` and place SAM 3.1 weights under models/sam3/ (or sam31/).",
    "salient_backend":      "Install `transformers` (for BiRefNet/U2-Net) and download salient-object weights to models/saliency/.",
    "experimental_backend": "Experimental backend — enable explicitly via the node widget; safe to ignore otherwise.",
    "video_backend":        "Install SAM2 video-predictor support (sam2 ≥ 1.0) for video segmentation.",
    "locate_anything_backend": "Install `transformers` (for LocateAnything-3B) — open-vocabulary grounding segmenter.",
}


def _import_all() -> None:
    from importlib import import_module
    try:
        from ..._c2c_registry import record_failure
    except Exception:
        record_failure = None  # type: ignore
    for mod in ("sam2_backend", "sam3_backend", "sam31_backend",
                "salient_backend", "experimental_backend",
                "video_backend", "locate_anything_backend"):
        try:
            import_module(f".{mod}", package=__name__)
        except Exception as exc:
            if record_failure is not None:
                record_failure(
                    f"segmenter:{mod}",
                    exc,
                    hint=_HINT_BY_MODULE.get(mod),
                    group="mask_matting/segmenters",
                )
            else:
                logger.warning("[MaskMatting] segmenter %s import skipped: %s", mod, exc)


_import_all()
