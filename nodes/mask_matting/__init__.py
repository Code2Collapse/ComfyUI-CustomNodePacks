"""
MaskMattingMEC — single-node multi-backend segmentation + matting pipeline.

Architecture
------------
This package provides ONE ComfyUI node ``MaskMattingMEC`` that wraps a
catalog of segmenter and matter backends behind a uniform interface.

Sub-packages:
    * ``segmenters`` – classes that turn (image, prompts) → coarse mask.
    * ``matters``    – classes that turn (image, mask/trimap) → alpha.

Each backend module declares:
    KEY            (str)  unique stable id used in widgets / JSON
    DISPLAY        (str)  human-readable label
    STATUS         (str)  "ready" | "experimental" | "missing-deps"
    MODELS_KEY     (str)  folder_paths key for its weights
    SUPPORTS_MODES (set)  any of {"points","bbox","text","video","auto"}

The main node is intentionally registered in ``node.py`` and exported via
``NODE_CLASS_MAPPINGS`` from this ``__init__.py``.

All weights are looked up through ``folder_paths`` (per project rules) and
auto-downloaded only when the user opts in (``auto_download=True``).
This pack contains ZERO Forbidden-Vision (AGPL) code.
"""
from __future__ import annotations

from .node import MaskOpsMEC
from .node import NODE_CLASS_MAPPINGS as _MO_MAPPINGS
from .node import NODE_DISPLAY_NAME_MAPPINGS as _MO_DISPLAY

try:
    from .temporal_node import (
        MaskTemporalMEC,
        NODE_CLASS_MAPPINGS as _MT_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _MT_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    import logging
    logging.getLogger("MEC.MaskMatting").warning(
        "MaskTemporalMEC unavailable: %s", _exc
    )
    MaskTemporalMEC = None  # type: ignore
    _MT_MAPPINGS, _MT_DISPLAY = {}, {}

try:
    from .refine_node import (
        MaskRefineMEC,
        NODE_CLASS_MAPPINGS as _MR_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _MR_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    import logging
    logging.getLogger("MEC.MaskMatting").warning(
        "MaskRefineMEC unavailable: %s", _exc
    )
    MaskRefineMEC = None  # type: ignore
    _MR_MAPPINGS, _MR_DISPLAY = {}, {}

NODE_CLASS_MAPPINGS = {**_MO_MAPPINGS, **_MT_MAPPINGS, **_MR_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_MO_DISPLAY, **_MT_DISPLAY, **_MR_DISPLAY}

__all__ = [
    "MaskOpsMEC", "MaskTemporalMEC", "MaskRefineMEC",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
]
