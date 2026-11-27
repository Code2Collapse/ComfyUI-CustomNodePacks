"""Mask Combine (C2C): any number of masks into one (owner request 2026-11-27).

The mask inputs grow: connect the last empty socket and a new one appears (core Autogrow, so it works in the classic
canvas, Nodes 2.0 and inside subgraphs, and saved workflows keep their links). Core caps one Autogrow group at 100
inputs; chain a second Mask Combine for more.

Every operation keeps soft alpha (no thresholding) and stays in 0..1. Inputs are combined in socket order, which matters
only for subtract and difference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from comfy_api.latest import io

MAX_MASKS = 100          # core Autogrow limit (comfy_api io.Autogrow._MaxNames)
PREFIX = "mask_"

OPERATIONS = {
    "union (max)": "Keep the strongest value at each pixel. The usual way to merge selections.",
    "add (clamped)": "Sum all masks, clipped at 1. Soft edges build up where masks overlap.",
    "subtract (first minus rest)": "Start from the first mask and cut every other mask out of it: "
                                   "first x (1 - rest) for each, so soft holdouts give soft holes.",
    "intersect (min)": "Keep only what every mask covers (the weakest value at each pixel).",
    "multiply": "Product of all masks. Like intersect, but soft values darken each other.",
    "screen": "1 - (1 - a)(1 - b)...: a soft union that brightens overlaps without hard clipping.",
    "difference": "Absolute difference, chained in socket order (|a - b|, then against c...). Shows where masks disagree.",
    "average": "Mean of all masks.",
}
SIZE_MODES = ("resize to first mask", "error")


def _as_bhw(m: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(m, torch.Tensor):
        raise ValueError(f"Mask Combine: {name} is not a mask.")
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4 and m.shape[1] == 1:
        m = m[:, 0]
    elif m.ndim != 3:
        raise ValueError(f"Mask Combine: {name} has shape {tuple(m.shape)}; a mask is [frames, height, width].")
    return m.float()


def ordered_masks(masks: dict | None) -> list[tuple[str, torch.Tensor]]:
    """Connected masks in socket order (mask_0, mask_1, ... mask_10: numeric, not text order)."""
    def idx(name: str) -> int:
        try:
            return int(name.removeprefix(PREFIX))
        except ValueError:
            return 1 << 30
    return [(n, masks[n]) for n in sorted(masks or {}, key=idx) if masks[n] is not None]


def conform(items: list[tuple[str, torch.Tensor]], size_mode: str) -> tuple[list[torch.Tensor], list[str]]:
    """Bring every mask to the first mask's size, device and frame count. Single-frame masks repeat over the clip."""
    notes: list[str] = []
    first = _as_bhw(items[0][1], items[0][0])
    dev, (h, w) = first.device, first.shape[-2:]
    out = [first]
    frames = max(_as_bhw(m, n).shape[0] for n, m in items)
    for name, m in items[1:]:
        m = _as_bhw(m, name).to(dev)
        if m.shape[-2:] != (h, w):
            if size_mode == "error":
                raise ValueError(f"Mask Combine: {name} is {m.shape[-1]}x{m.shape[-2]} but {items[0][0]} is {w}x{h}. "
                                 f"Resize it first, or set size_mismatch to 'resize to first mask'.")
            notes.append(f"{name} resized {m.shape[-1]}x{m.shape[-2]} -> {w}x{h}")
            m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)[:, 0].clamp(0, 1)
        out.append(m)
    for i, (m, (name, _)) in enumerate(zip(out, items)):
        if m.shape[0] == frames:
            continue
        if m.shape[0] != 1:
            sizes = ", ".join(f"{n}={_as_bhw(t, n).shape[0]}" for n, t in items)
            raise ValueError(f"Mask Combine: the masks have different frame counts ({sizes}). Masks with one frame "
                             f"are repeated over the clip; any other count must match.")
        out[i] = m.expand(frames, -1, -1)
    return out, notes


def combine(ms: list[torch.Tensor], operation: str) -> torch.Tensor:
    """Running in-place reduction: peak memory is the result plus one input, whatever the mask count (a torch.stack
    of N masks held N extra copies - 124 frames of 2K is ~1 GB per mask)."""
    if operation not in OPERATIONS:
        raise ValueError(f"Mask Combine: unknown operation {operation!r}; choose one of {', '.join(OPERATIONS)}.")
    r = ms[0].float().clone()
    if r.shape[0] != max(m.shape[0] for m in ms):                   # a 1-frame first mask expanded over the clip
        r = r.expand(max(m.shape[0] for m in ms), -1, -1).clone()
    if operation == "screen":
        r = 1.0 - r.clamp_(0, 1)                                     # r holds prod(1 - m)
    for m in ms[1:]:
        if operation == "union (max)":
            torch.maximum(r, m, out=r)
        elif operation in ("add (clamped)", "average"):
            r.add_(m)
        elif operation == "subtract (first minus rest)":
            r.mul_(1.0 - m.clamp(0, 1))
        elif operation == "intersect (min)":
            torch.minimum(r, m, out=r)
        elif operation == "multiply":
            r.mul_(m)
        elif operation == "screen":
            r.mul_(1.0 - m.clamp(0, 1))
        elif operation == "difference":
            r.sub_(m).abs_()
    if operation == "average":
        r.div_(len(ms))
    elif operation == "screen":
        r = 1.0 - r
    return r.clamp_(0.0, 1.0).contiguous()


class MaskCombineMEC(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MaskCombineMEC",
            display_name="Mask Combine (C2C)",
            category="MEC/Mask",
            search_aliases=["combine masks", "merge masks", "mask union", "add masks", "mask math", "multi mask"],
            description="Combine any number of masks into one. Connect a mask to the last empty socket and another "
                        "socket appears (up to 100; chain a second node for more). Soft edges are kept.",
            inputs=[
                io.Autogrow.Input("masks", template=io.Autogrow.TemplatePrefix(
                    io.Mask.Input("mask", optional=True,
                                  tooltip="A mask to combine. Masks are used in socket order (it matters for subtract "
                                          "and difference). Empty sockets are skipped."),
                    prefix=PREFIX, min=2, max=MAX_MASKS)),
                io.Combo.Input("operation", options=list(OPERATIONS), default="union (max)",
                               tooltip="\n".join(f"{k}: {v}" for k, v in OPERATIONS.items())),
                io.Combo.Input("size_mismatch", options=list(SIZE_MODES), default=SIZE_MODES[0],
                               tooltip="Masks whose size differs from the first connected mask: resize them to it "
                                       "(bilinear, soft) or stop with an error."),
                io.Boolean.Input("invert_result", default=False, tooltip="Output 1 - result."),
            ],
            outputs=[
                io.Mask.Output("mask", tooltip="The combined mask, 0..1, the size of the first connected mask."),
                io.Int.Output("mask_count", tooltip="How many masks were connected and combined."),
            ],
        )

    @classmethod
    def execute(cls, masks: io.Autogrow.Type = None, operation: str = "union (max)",
                size_mismatch: str = SIZE_MODES[0], invert_result: bool = False) -> io.NodeOutput:
        items = ordered_masks(masks)
        if not items:
            raise ValueError("Mask Combine: no mask is connected. Connect at least one mask.")
        ms, notes = conform(items, size_mismatch)
        out = combine(ms, operation)
        if invert_result:
            out = 1.0 - out
        if notes:
            import logging
            logging.getLogger("MEC").info("[Mask Combine] %s", "; ".join(notes))
        return io.NodeOutput(out, len(ms))


NODE_CLASS_MAPPINGS = {"MaskCombineMEC": MaskCombineMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskCombineMEC": "Mask Combine (C2C)"}
