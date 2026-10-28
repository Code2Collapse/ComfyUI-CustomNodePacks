"""Kill switch for L7.65 P01: C2C_LEGACY_MASK_EDITORS=1 registers the five editors the Image Mask Editor took over
again, from their unchanged code (design docs/design/image_mask_editor_modes.md section 6). The workflow migration only
rewrites node types that are not registered, so old workflows then load the original nodes."""
from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import Dict, Tuple

logger = logging.getLogger("MEC")

_LEGACY = (
    ("MaskEditMEC", "mask_edit_mec", "Mask Edit — Transform/Draw/Points/BBox (legacy)"),
    ("SplineMaskMEC", "spline_mask_mec", "Spline Mask — Edit/Track/Flow-Path (legacy)"),
    ("VectorRotoMEC", "roto", "Vector Roto — Bezier (legacy)"),
    ("VideoMaskEditorMEC", "video_mask_editor", "Video Mask Editor (legacy)"),
    ("MECAdvancedPaintCanvas", "mec_paint_suite", "Advanced Paint Canvas (legacy)"),
)


def legacy_editor_mappings() -> Tuple[Dict[str, type], Dict[str, str]]:
    if os.environ.get("C2C_LEGACY_MASK_EDITORS") != "1":
        return {}, {}
    mappings: Dict[str, type] = {}
    display: Dict[str, str] = {}
    for node_id, module, label in _LEGACY:
        try:
            mappings[node_id] = getattr(import_module(f"..{module}", package=__package__), node_id)
            display[node_id] = label
        except Exception as exc:  # noqa: BLE001 - one missing editor must not cost the others
            logger.warning("[MEC] C2C_LEGACY_MASK_EDITORS: %s not registered (%s)", node_id, exc)
    return mappings, display
