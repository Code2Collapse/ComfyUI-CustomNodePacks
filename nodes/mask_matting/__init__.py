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

NODE_CLASS_MAPPINGS = {**_MO_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_MO_DISPLAY}

__all__ = [
    "MaskOpsMEC",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
]
