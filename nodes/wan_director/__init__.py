"""
Wan Director v1 — multi-shot orchestration toolkit (C2C).

A pragmatic, *model-agnostic* multi-shot director for video generation
pipelines (Wan 2.x, AnimateDiff, HunyuanVideo, LTX, etc).  Rather than
monkey-patching diffusion internals (which is fragile and breaks on
every ComfyUI update), this pack provides composable graph utilities:

  - WanShotListMEC     : JSON shot list -> SHOTLIST socket
  - WanShotPickerMEC   : SHOTLIST + index -> per-shot parameters
  - WanShotCountMEC    : SHOTLIST -> shot count (for batch iteration)
  - WanFrameBridgeMEC  : IMAGE clip -> last frame (start-image of next shot)
  - WanShotConcatMEC   : 1..6 IMAGE clips -> one IMAGE timeline (optional crossfade)
  - WanPromptScheduleMEC: SHOTLIST -> frame-indexed prompt schedule string

These nodes share a single ``SHOTLIST`` custom socket type so they connect
cleanly without stringly-typed data.

Author: Code2Collapse, May 2026.
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from .shotlist import WanShotListMEC
from .picker import WanShotPickerMEC, WanShotCountMEC
from .bridge import WanFrameBridgeMEC
from .concat import WanShotConcatMEC
from .schedule import WanPromptScheduleMEC

NODE_CLASS_MAPPINGS = {
    "WanShotListMEC":        WanShotListMEC,
    "WanShotPickerMEC":      WanShotPickerMEC,
    "WanShotCountMEC":       WanShotCountMEC,
    "WanFrameBridgeMEC":     WanFrameBridgeMEC,
    "WanShotConcatMEC":      WanShotConcatMEC,
    "WanPromptScheduleMEC":  WanPromptScheduleMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanShotListMEC":        "Wan Director — Shot List (C2C)",
    "WanShotPickerMEC":      "Wan Director — Shot Picker (C2C)",
    "WanShotCountMEC":       "Wan Director — Shot Count (C2C)",
    "WanFrameBridgeMEC":     "Wan Director — Frame Bridge (C2C)",
    "WanShotConcatMEC":      "Wan Director — Shot Concat (C2C)",
    "WanPromptScheduleMEC":  "Wan Director — Prompt Schedule (C2C)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
