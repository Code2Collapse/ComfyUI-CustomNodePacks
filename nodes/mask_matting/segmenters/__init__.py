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
# broken optional dep can't take down the whole pack.
def _import_all() -> None:
    from importlib import import_module
    for mod in ("sam2_backend", "sam3_backend", "sam31_backend",
                "salient_backend", "experimental_backend", "stubs"):
        try:
            import_module(f".{mod}", package=__name__)
        except Exception as exc:
            logger.debug("[MaskMatting] segmenter %s import skipped: %s", mod, exc)


_import_all()
