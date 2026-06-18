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

try:
    from .._c2c_registry import record_failure as _c2c_rec
except Exception:  # pragma: no cover
    _c2c_rec = None  # type: ignore

# The base MaskOpsMEC node — guarded so a failure here (e.g. a missing helper
# module) records a clear diagnostic and lets the rest of the pack load,
# instead of raising out of this package's __init__ and killing every node
# in ComfyUI-CustomNodePacks.
try:
    from .node import MaskOpsMEC
    from .node import NODE_CLASS_MAPPINGS as _MO_MAPPINGS
    from .node import NODE_DISPLAY_NAME_MAPPINGS as _MO_DISPLAY
except Exception as _exc:  # pragma: no cover
    if _c2c_rec is not None:
        _c2c_rec(
            "MaskOpsMEC", _exc,
            hint="A mask_matting helper failed to import. Check that "
                 "nodes/mask_matting/_reanchor.py exists and opencv-python / "
                 "transformers are installed.",
            group="mask_matting",
        )
    else:
        import logging
        logging.getLogger("MEC.MaskMatting").warning(
            "MaskOpsMEC unavailable: %s", _exc
        )
    MaskOpsMEC = None  # type: ignore
    _MO_MAPPINGS, _MO_DISPLAY = {}, {}

try:
    from .temporal_node import (
        MaskTemporalMEC,
        NODE_CLASS_MAPPINGS as _MT_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _MT_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    if _c2c_rec is not None:
        _c2c_rec(
            "MaskTemporalMEC", _exc,
            hint="Install `opencv-python` and `scipy` for temporal mask consistency.",
            group="mask_matting",
        )
    else:
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
    if _c2c_rec is not None:
        _c2c_rec(
            "MaskRefineMEC", _exc,
            hint="Install `pydensecrf` (or fallback path) and `scipy` for full mask refine stages.",
            group="mask_matting",
        )
    else:
        import logging
        logging.getLogger("MEC.MaskMatting").warning(
            "MaskRefineMEC unavailable: %s", _exc
        )
    MaskRefineMEC = None  # type: ignore
    _MR_MAPPINGS, _MR_DISPLAY = {}, {}

NODE_CLASS_MAPPINGS = {
    **_MO_MAPPINGS, **_MT_MAPPINGS, **_MR_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_MO_DISPLAY, **_MT_DISPLAY, **_MR_DISPLAY,
}

__all__ = [
    "MaskOpsMEC", "MaskTemporalMEC", "MaskRefineMEC",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "morph_erode", "morph_dilate", "morph_open", "morph_close", "morph_gradient",
]

from .utils import morph_erode, morph_dilate, morph_open, morph_close, morph_gradient
