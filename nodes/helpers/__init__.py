"""
C2C Helpers — 9 small utility nodes that compose with the rest of the
ComfyUI-CustomNodePacks ecosystem (Wan Director, Inpaint, Mask, AI spine).

These are intentionally tiny, single-purpose, dependency-free and 100%
covered by smoke tests. They fill the "missing 5%" between large nodes:
seed lists, sizes (aspect presets / snapping), batch ranges, conditional
passthrough, template strings, lerp ramps, an image / mask / latent probe,
execution timer. L7.65 merged Batch Split + Video Frame Extractor into Batch
Range, Dimensions Snap into Size, Mask Area Probe + Latent Inspector into Probe.

Author: Code2Collapse, May 2026.
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from .helpers import (
    ImageBatchSliceMEC,
    MaskBatchCombineMEC,
    SeedListMEC,
    ConditionalSwitchMEC,
    TextTemplateMEC,
    NumberLerpMEC,
    AspectPresetMEC,
    ImageStatsProbeMEC,
    ExecutionTimerMEC,
)

NODE_CLASS_MAPPINGS = {
    "ImageBatchSliceMEC":   ImageBatchSliceMEC,
    "MaskBatchCombineMEC":  MaskBatchCombineMEC,
    "SeedListMEC":          SeedListMEC,
    "ConditionalSwitchMEC": ConditionalSwitchMEC,
    "TextTemplateMEC":      TextTemplateMEC,
    "NumberLerpMEC":        NumberLerpMEC,
    "AspectPresetMEC":      AspectPresetMEC,
    "ImageStatsProbeMEC":   ImageStatsProbeMEC,
    "ExecutionTimerMEC":    ExecutionTimerMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchSliceMEC":   "Batch Range (C2C)",
    "MaskBatchCombineMEC":  "Mask Batch Combine (C2C)",
    "SeedListMEC":          "Seed List Generator (C2C)",
    "ConditionalSwitchMEC": "Conditional Switch",
    "TextTemplateMEC":      "Text Template (C2C)",
    "NumberLerpMEC":        "Number Lerp (C2C)",
    "AspectPresetMEC":      "Size (C2C)",
    "ImageStatsProbeMEC":   "Probe (C2C)",
    "ExecutionTimerMEC":    "Execution Timer (C2C)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
