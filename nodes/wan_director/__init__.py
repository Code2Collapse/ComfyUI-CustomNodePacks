"""
Wan Director v2 — single-node visual timeline (C2C).

ONE node, ALL Wan variants. The visual timeline (drag image / text /
audio clips, scrub, ghost-preview), dual-backend dispatch (native
ComfyUI + Kijai WanVideoWrapper), prompt-relay temporal cross-attention
bias, and per-variant latent + conditioning assembly all live inside
``WanDirectorC2C`` — there are NO helper sub-nodes to wire.

  - Variants: wan2.1_t2v / i2v, wan2.2_t2v / i2v (dual-CFG),
              wan_fun_inp, wan_animate.
  - Backends: native (MODEL + CLIP) or kijai (WANVIDEOMODEL + WANTEXTENCODER).
  - PromptRelay: optional, works on both backbones + any third-party
    diffusion model via the generic-introspection fallback patcher.

Companion frontend extension: ``js/wan_director_timeline.js``.

Forked-from inspiration: WhatDreamsCost-ComfyUI / LTX Director (MIT).
See NOTICE.

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
