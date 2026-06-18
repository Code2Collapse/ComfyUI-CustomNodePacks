"""
C2C Helpers — 12 small utility nodes that compose with the rest of the
ComfyUI-CustomNodePacks ecosystem (Wan Director, Inpaint, Mask, AI spine).

These are intentionally tiny, single-purpose, dependency-free and 100%
covered by smoke tests. They fill the "missing 5%" between large nodes:
seed lists, aspect presets, batch slicing, conditional passthrough,
template strings, lerp ramps, dimension snapping, image/mask stats,
execution timer, schedule split.

Author: Code2Collapse, May 2026.
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from .helpers import (
    ImageBatchSliceMEC,
    ImageBatchSplitMEC,
    MaskBatchCombineMEC,
    SeedListMEC,
    ConditionalSwitchMEC,
    TextTemplateMEC,
    NumberLerpMEC,
    DimensionsSnapMEC,
    AspectPresetMEC,
    ImageStatsProbeMEC,
    MaskAreaProbeMEC,
    ExecutionTimerMEC,
)

NODE_CLASS_MAPPINGS = {
    "ImageBatchSliceMEC":   ImageBatchSliceMEC,
    "ImageBatchSplitMEC":   ImageBatchSplitMEC,
    "MaskBatchCombineMEC":  MaskBatchCombineMEC,
    "SeedListMEC":          SeedListMEC,
    "ConditionalSwitchMEC": ConditionalSwitchMEC,
    "TextTemplateMEC":      TextTemplateMEC,
    "NumberLerpMEC":        NumberLerpMEC,
    "DimensionsSnapMEC":    DimensionsSnapMEC,
    "AspectPresetMEC":      AspectPresetMEC,
    "ImageStatsProbeMEC":   ImageStatsProbeMEC,
    "MaskAreaProbeMEC":     MaskAreaProbeMEC,
    "ExecutionTimerMEC":    ExecutionTimerMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchSliceMEC":   "Image Batch Slice (C2C)",
    "ImageBatchSplitMEC":   "Image Batch Split (C2C)",
    "MaskBatchCombineMEC":  "Mask Batch Combine (C2C)",
    "SeedListMEC":          "Seed List Generator (C2C)",
    "ConditionalSwitchMEC": "Conditional Switch",
    "TextTemplateMEC":      "Text Template (C2C)",
    "NumberLerpMEC":        "Number Lerp (C2C)",
    "DimensionsSnapMEC":    "Dimensions Snap (C2C)",
    "AspectPresetMEC":      "Aspect Ratio Preset (C2C)",
    "ImageStatsProbeMEC":   "Image Stats Probe (C2C)",
    "MaskAreaProbeMEC":     "Mask Area Probe (C2C)",
    "ExecutionTimerMEC":    "Execution Timer (C2C)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
