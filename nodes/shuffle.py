# FILE: nodes/shuffle.py
# FEATURE: F4 — ShuffleMEC (single-node 4-channel matrix; no external math nodes)
# INTEGRATES WITH: any IMAGE consumer
"""
Single-node Nuke-style Shuffle. Each output channel can pull from any input
channel (R/G/B/A) or a constant (zero/one). The whole 4-channel assignment is
done inside one forward pass — no DUMB_BLUR / MULTI_NODE_SHUFFLE diagnoses.
"""
from __future__ import annotations

from typing import Tuple

import torch


_SOURCES = ("R", "G", "B", "A", "Lum", "InvR", "InvG", "InvB", "InvA",
            "zero", "one")


def _channel(image: torch.Tensor, src: str) -> torch.Tensor:
    """image: (B,H,W,C); returns (B,H,W,1) per the source code."""
    B, H, W, C = image.shape
    one = torch.ones(B, H, W, 1, device=image.device, dtype=image.dtype)
    zero = torch.zeros_like(one)
    if src == "zero":
        return zero
    if src == "one":
        return one
    if src == "R":
        return image[..., 0:1]
    if src == "G":
        return image[..., 1:2]
    if src == "B":
        return image[..., 2:3]
    if src == "A":
        return image[..., 3:4] if C >= 4 else one
    if src == "Lum":
        r = image[..., 0:1]
        g = image[..., 1:2]
        b = image[..., 2:3]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    if src == "InvR":
        return 1.0 - image[..., 0:1]
    if src == "InvG":
        return 1.0 - image[..., 1:2]
    if src == "InvB":
        return 1.0 - image[..., 2:3]
    if src == "InvA":
        return 1.0 - (image[..., 3:4] if C >= 4 else one)
    raise ValueError(f"unknown shuffle source '{src}'")


class ShuffleMEC:
    DESCRIPTION = "Nuke-style 4-channel shuffle in a single node."
    CATEGORY = "MaskEditControl/Channels"
    FUNCTION = "shuffle"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")

    @classmethod
    def INPUT_TYPES(cls):
        opt = {"default": "R"}
        return {
            "required": {
                "image":  ("IMAGE",),
                "out_R":  (list(_SOURCES), {"default": "R"}),
                "out_G":  (list(_SOURCES), {"default": "G"}),
                "out_B":  (list(_SOURCES), {"default": "B"}),
                "out_A":  (list(_SOURCES), {"default": "A"}),
                "premultiply_output": ("BOOLEAN", {"default": False}),
            },
        }

    def shuffle(self, image: torch.Tensor, out_R: str, out_G: str,
                out_B: str, out_A: str, premultiply_output: bool):
        r = _channel(image, out_R)
        g = _channel(image, out_G)
        b = _channel(image, out_B)
        a = _channel(image, out_A).clamp(0.0, 1.0)
        rgb = torch.cat((r, g, b), dim=-1)
        if premultiply_output:
            rgb = rgb * a
        out = torch.cat((rgb, a), dim=-1).clamp(0.0, 1.0)
        return (out, a.squeeze(-1))


NODE_CLASS_MAPPINGS = {"ShuffleMEC": ShuffleMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"ShuffleMEC": "Shuffle — Channels (MEC)"}
