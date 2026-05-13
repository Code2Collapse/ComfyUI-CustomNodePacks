"""Base class + registry for matter (alpha-refinement) backends."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("MEC.MaskMatting.Matters")


class BaseMatter:
    KEY: str = "base"
    DISPLAY: str = "Base"
    STATUS: str = "experimental"
    MODELS_KEY: str = ""
    NEEDS_TRIMAP: bool = False

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

    def matte(
        self,
        image_bhwc: torch.Tensor,
        coarse_mask: torch.Tensor,
        *,
        trimap: Optional[torch.Tensor] = None,
        edge_radius: int = 4,
        memory_size: int = 8,
    ) -> Dict[str, Any]:
        """Returns dict with ``alpha`` (B,H,W) and ``info``."""
        raise NotImplementedError


_REGISTRY: Dict[str, type] = {}


def register(cls):
    _REGISTRY[cls.KEY] = cls
    return cls


def all_matters() -> Dict[str, type]:
    return dict(_REGISTRY)


def get_matter_cls(key: str):
    return _REGISTRY.get(key)


def list_keys(installed_only: bool = False):
    out = []
    for k, cls in _REGISTRY.items():
        if installed_only and cls.STATUS != "ready":
            continue
        out.append(k)
    return out


def _import_all() -> None:
    from importlib import import_module
    for mod in ("vitmatte_backend", "rvm_backend",
                "bgmattingv2_backend", "matanyone_backend",
                "salient_matter_backend"):
        try:
            import_module(f".{mod}", package=__name__)
        except Exception as exc:
            logger.debug("[MaskMatting] matter %s import skipped: %s", mod, exc)


_import_all()
