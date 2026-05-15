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

from .director_node import WanDirectorC2C

# Wan Director is intentionally ONE node — the visual timeline holds
# shot-list, picker, frame-bridge, concat and schedule semantics inline
# (matching WhatDreamsCost LTX Director's single-node UX). The helper
# modules on disk are kept as importable utilities only.

NODE_CLASS_MAPPINGS = {
    "WanDirectorC2C": WanDirectorC2C,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanDirectorC2C": "Wan Director",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
