"""Tiled video refinement, as nodes.

THE SHAPE OF THIS, and why it is not one big node.

The obvious design is a single "tiled upscale" node that takes a MODEL and a
sampler and loops over tiles internally. Every pack that does this is welded
to one sampler, one ComfyUI version, and one model family, and "works with any
model set" becomes a list of the four it was tested against.

So the loop is turned inside out. `TileSplitMEC` emits the tiles AS A BATCH,
the user wires whatever refiner they like - any sampler, any model, an upscale
model, a LUT, three nodes in a row - and `TileMergeMEC` puts the batch back.
ComfyUI already knows how to run a batch through anything, so there is no
sampler code here at all and nothing to break when the sampler API moves.

    IMAGE ──> TileSplit ──> [ anything that takes and returns IMAGE ] ──┐
      │           │                                                     │
      │        TILE_PLAN ────────────────────────────────────────────> TileMerge
      │                                                                 │
      └────────────────> IdentityLock <─────────────────────────────────┘

BATCH ORDER is tile-major: every frame of tile 0, then every frame of tile 1.
Stated because it matters - a video refiner reading the batch as a sequence
sees one tile's whole time range contiguously, which is what lets it be
temporally consistent. Frame-major would hand it a batch that jumps around the
frame, and it would deflicker across tiles rather than across time.
"""

from __future__ import annotations

import colorsys
import dataclasses
import json
import math

import torch
import torch.nn.functional as F

from ._core import (
    TilePlan,
    TilingError,
    _is_bhwc,
    build_plan,
    build_plan_from_starts,
    crop,
    describe_plan,
    merge,
    placements,
    plan_for_tile_size,
)
from ._identity import (
    IdentityLockError,
    describe as describe_lock,
    describe_bands,
    drift,
    drift_map,
    drift_report,
    identity_lock,
    low_pass_cache,
    multiband_lock,
    parse_bands,
    track_protect_mask,
)
from ._motion import box_mean, gradient_magnitude, luma, max_global_shift

try:  # optional: tile numbers on the tile map (outlines are drawn without it)
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is a declared dependency, but the node must not need it
    cv2 = None
try:                                                       # A28 layout declarations (display only)
    from .._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '..', '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

CATEGORY = "C2C/Tiling"


def _spatial(x: torch.Tensor) -> tuple[int, int]:
    from ._core import _spatial_dims
    return _spatial_dims(x)


def _fingerprint(*tensors, **widgets) -> str:
    """IS_CHANGED value that works for core's real call: linked inputs arrive as None (core passes constants only,
    execution.py get_input_data(..., None)); core's own cache key already covers what flows in through links. A
    tensor passed directly (tests, scripts) is still hashed. A22 L9.135: raising here made the result NaN, so these
    nodes never cached and the refiner between them re-ran on every queue."""
    import hashlib
    h = hashlib.md5()
    for t in tensors:
        if isinstance(t, torch.Tensor):
            h.update(str(tuple(t.shape)).encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(t, TilePlan):
            h.update(repr(t.tiles).encode())
            if t.halo:                                    # L9.172: only a non-default halo changes the value
                h.update(f"halo={t.halo}".encode())
        elif t is not None:
            h.update(repr(t).encode())
    h.update(repr(sorted((k, str(v)) for k, v in widgets.items())).encode())
    return h.hexdigest()


# A28 (docs/audit/A28/PARAM_CNP_B.md, Tiling): each plan mode's own sizes; the lock's tuning in a section.
# Display only.
_PLAN_LAYOUT = {"tile_size": {"c2c_show_if": {"mode": ["tile size"]}},
                "rows": {"c2c_show_if": {"mode": ["tile count"]}},
                "cols": {"c2c_show_if": {"mode": ["tile count"]}},
                # L9.172 (A25): the memory budget and the content-aware seams, folded in their own section
                "vram_budget_mb": {"c2c_section": "Memory / content"},
                "mb_per_megapixel": {"c2c_section": "Memory / content",
                                     "c2c_show_if": {"vram_budget_mb": {"truthy": True}}},
                "content_aware": {"c2c_section": "Memory / content"},
                "subject_weight": {"c2c_section": "Memory / content", "c2c_show_if": {"content_aware": [True]}}}
_LOCK_LAYOUT = {"corrections": {"c2c_section": "Lock"}, "want_drift_map": {"c2c_section": "Lock"},
                # L9.172 (A25): multi-band preservation and the tracked protect mask
                "band_radii": {"c2c_section": "Bands / video"}, "band_strengths": {"c2c_section": "Bands / video"},
                "track_protect": {"c2c_section": "Bands / video"}}

# L9.172 (A25 spec 57-60): inputs appended after A28, in order, per node (the tests pin that they are LAST)
APPENDED_L9172 = {
    "TilePlanMEC": ("vram_budget_mb", "mb_per_megapixel", "content_aware", "subject_weight", "subject_mask"),
    "TileSplitMEC": ("halo_px", "motion_halo"),
    "TileMergeMEC": ("adaptive_blend", "reblend_threshold"),
    "IdentityLockMEC": ("band_radii", "band_strengths", "track_protect"),
}


# ── L9.172: plan helpers (memory budget, content-aware seams) ──────────────────────────────────────────────────

def _round8(x: float) -> int:
    return int(round(x / 8.0)) * 8


def _frames_of(image: torch.Tensor) -> int:
    """How many frames a tile carries through a tile-major refiner run (a 5D video latent: batch x time)."""
    if image.ndim == 5:
        return int(image.shape[0]) * int(image.shape[2])
    return int(image.shape[0])


def _even_starts(total: int, n: int, size: int) -> list:
    if n <= 1:
        return [0]
    stride = (total - size) / float(n - 1)
    s = [int(round(i * stride)) for i in range(n)]
    s[-1] = total - size
    return s


def _luma_clip(image: torch.Tensor) -> torch.Tensor:
    """[B, H, W] luma of an IMAGE (BHWC) or a latent (BCHW: its first channels; BCTHW: frames folded in)."""
    if _is_bhwc(image):
        return luma(image)
    x = image.float()
    if x.ndim == 5:
        x = x.movedim(2, 1).reshape(-1, x.shape[1], x.shape[-2], x.shape[-1])
    return luma(x.movedim(1, -1))


def _seam_profiles(image, subject_mask, subject_weight: float, h: int, w: int):
    """1D cost profiles (per row, per column) for where a blend band should NOT sit: detail energy averaged over the
    WHOLE clip (one plan per clip) + subject_weight x the subject mask averaged over frames."""
    g = gradient_magnitude(_luma_clip(image).cpu()).mean(dim=0)                    # [H, W]
    if subject_mask is not None and float(subject_weight) > 0:
        m = subject_mask.float().cpu()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if tuple(m.shape[-2:]) != (h, w):
            m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        g = g + float(subject_weight) * m.clamp(0.0, 1.0).mean(dim=0)
    return g.mean(dim=1).double(), g.mean(dim=0).double()


def _choose_starts(profile: torch.Tensor, total: int, n: int, size: int, overlap: int, search: int):
    """Interior tile starts inside +-search of the evenly spaced ones that put the overlap bands on the cheapest
    stretch of `profile`, every overlap >= `overlap` (dynamic programming over all seams of the axis). Ties go to the
    evenly spaced start. Returns (starts, nominal, cost_nominal, cost_chosen).

    The cost of a band is the SUM of the profile inside it: the tiles keep one size, so the total overlap of an axis
    is fixed, and what can be chosen is where that overlap lies - on as little detail / subject as possible (a mean
    would instead reward widening a band over flat ground while it still covers the detail)."""
    nominal = _even_starts(total, n, size)
    if n <= 1:
        return nominal, nominal, 0.0, 0.0
    cum = torch.zeros(total + 1, dtype=torch.float64)
    cum[1:] = torch.cumsum(profile, dim=0)

    def band(prev, cur):                     # summed cost of the overlap band [cur, prev + size)
        return cum[prev + size] - cum[cur]

    cands = []
    for i, nom in enumerate(nominal):
        if i == 0:
            c = [0]
        elif i == n - 1:
            c = [total - size]
        else:
            c = list(range(max(1, nom - search), min(total - size - 1, nom + search) + 1)) or [nom]
        cands.append(torch.tensor(c, dtype=torch.long))
    max_stride = size - overlap
    cost = torch.zeros(1, dtype=torch.float64)
    back = []
    for i in range(1, n):
        p, c = cands[i - 1][:, None], cands[i][None, :]
        stride = c - p
        valid = (stride >= 1) & (stride <= max_stride)
        step = band(p, c) + 1e-9 * (cands[i] - nominal[i]).abs().double()[None, :]
        total_cost = torch.where(valid, cost[:, None] + step, torch.full_like(step, float("inf")))
        cost, arg = total_cost.min(dim=0)
        back.append(arg)
    if not torch.isfinite(cost).any():
        return nominal, nominal, 0.0, 0.0
    k = int(torch.argmin(cost))
    starts = [int(cands[-1][k])]
    for i in range(n - 1, 0, -1):
        k = int(back[i - 1][k])
        starts.append(int(cands[i - 1][k]))
    starts.reverse()

    def total_band(st):
        return float(sum(band(torch.tensor(a), torch.tensor(b)) for a, b in zip(st, st[1:])))
    return starts, nominal, total_band(nominal), total_band(starts)


def _content_plan(h, w, mode, tile_size, rows, cols, overlap, blend, image, subject_mask, subject_weight):
    """The content-aware plan and its report lines (one plan for the whole clip, equal tile sizes)."""
    search = max(8, _round8(overlap // 2))
    if mode == "tile size":
        if tile_size - 2 * search <= overlap:
            raise TilingError(
                f"content_aware needs room to move the seams: a {tile_size}px tile minus 2 x {search}px search "
                f"leaves {tile_size - 2 * search}px, not more than the {overlap}px overlap. Raise tile_size or "
                "lower the overlap.")
        base = plan_for_tile_size(h, w, tile=int(tile_size) - 2 * search, overlap=int(overlap), blend=blend)
    else:
        base = build_plan(h, w, rows=int(rows), cols=int(cols), overlap=int(overlap), blend=blend)
    t0 = base.tiles[0]
    # an axis with 1 or 2 tiles has no seam that can move (both tiles are pinned to the frame edges): no growth there
    th = min(h, t0.height + 2 * search) if base.rows >= 3 else t0.height
    tw = min(w, t0.width + 2 * search) if base.cols >= 3 else t0.width
    prow, pcol = _seam_profiles(image, subject_mask, subject_weight, h, w)
    rs, rnom, rb, ra = _choose_starts(prow, h, base.rows, th, overlap, search)
    cs, cnom, cb, ca = _choose_starts(pcol, w, base.cols, tw, overlap, search)
    plan = build_plan_from_starts(h, w, rs, cs, th, tw, overlap=int(overlap), blend=blend)
    moved = [f"row seam {i}: {b - a:+d}px" for i, (a, b) in enumerate(zip(rnom[1:], rs[1:]), 1) if a != b]
    moved += [f"column seam {i}: {b - a:+d}px" for i, (a, b) in enumerate(zip(cnom[1:], cs[1:]), 1) if a != b]
    pinned = [ax for ax, n in (("rows", base.rows), ("columns", base.cols)) if n == 2]
    lines = [
        f"Content-aware: tiles grown by 2 x {search}px on axes with 3+ tiles so each seam may move up to "
        f"{search}px; overlaps stay >= {overlap}px; one plan for all {_frames_of(image)} frame(s)."
        + (f" With 2 {' / '.join(pinned)} both tiles touch the frame edges, so those seams cannot move." if pinned
           else ""),
        ("Seams moved: " + ", ".join(moved)) if moved else "No seam moved (the evenly spaced bands were cheapest).",
        f"Detail + subject inside the overlap bands, evenly spaced -> chosen: rows {rb:.3f} -> {ra:.3f}, "
        f"columns {cb:.3f} -> {ca:.3f}.",
    ]
    return plan, lines


def _auto_budget_mb():
    """80 % of the free memory of ComfyUI's torch device, in MB; (None, reason) when it cannot be read."""
    try:
        import comfy.model_management as mm  # type: ignore
        free = float(mm.get_free_memory(mm.get_torch_device()))
        return 0.8 * free / (1024.0 * 1024.0), None
    except Exception as exc:  # noqa: BLE001 - outside ComfyUI, or a backend without the query
        return None, f"{type(exc).__name__}: {exc}"


# ── L9.172: split helpers (halo, tile map) ──────────────────────────────────────────────────────────────────────

def _pad_replicate(x: torch.Tensor, pad: int) -> torch.Tensor:
    """x with `pad` border pixels repeated on every side of its two spatial axes (IMAGE BHWC, LATENT BCHW / BCTHW)."""
    if _is_bhwc(x):
        y = F.pad(x.movedim(-1, 1).float(), (pad, pad, pad, pad), mode="replicate")
        return y.movedim(1, -1).to(x.dtype)
    lead, (h, w) = x.shape[:-2], x.shape[-2:]
    y = F.pad(x.reshape(1, -1, h, w).float(), (pad, pad, pad, pad), mode="replicate")
    return y.reshape(*lead, h + 2 * pad, w + 2 * pad).to(x.dtype)


def _crop_padded(padded: torch.Tensor, tile, halo: int) -> torch.Tensor:
    """The tile plus `halo` on every side, out of a tensor padded by `halo` (so the tile starts at its own top/left)."""
    hh, ww = tile.height + 2 * halo, tile.width + 2 * halo
    if _is_bhwc(padded):
        return padded[:, tile.top:tile.top + hh, tile.left:tile.left + ww, :]
    return padded[..., tile.top:tile.top + hh, tile.left:tile.left + ww]


def _tile_colour(i: int):
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _tile_map(image: torch.Tensor, plan: TilePlan) -> torch.Tensor:
    """IMAGE [1, H, W, 3]: frame 0 (grey for a latent), overlaps darkened, each tile outlined in its own colour with its
    id (OpenCV) - what the plan does to this footage, at a glance."""
    import numpy as np
    H, W = plan.height, plan.width
    if _is_bhwc(image):
        base = image[0].detach().float().cpu()
        base = base[..., :3] if base.shape[-1] >= 3 else base[..., :1].expand(H, W, 3)
    else:
        base = torch.full((H, W, 3), 0.5)
    cover = torch.zeros((H, W))
    for t in plan.tiles:
        cover[t.top:t.bottom, t.left:t.right] += 1
    base = base.clamp(0.0, 1.0) * torch.where(cover >= 2, 0.55, 1.0).unsqueeze(-1)
    arr = np.ascontiguousarray((base.numpy() * 255.0).round().astype(np.uint8))
    lw = max(1, min(H, W) // 300 + 1)
    for i, t in enumerate(plan.tiles):
        col = _tile_colour(i)
        y0, y1, x0, x1 = t.top, t.bottom, t.left, t.right
        arr[y0:min(y1, y0 + lw), x0:x1] = col
        arr[max(y0, y1 - lw):y1, x0:x1] = col
        arr[y0:y1, x0:min(x1, x0 + lw)] = col
        arr[y0:y1, max(x0, x1 - lw):x1] = col
        if cv2 is not None:
            scale = max(0.4, min(H, W) / 700.0)
            org = (x0 + 4 * lw + 2, y0 + int(22 * scale) + 2 * lw)
            cv2.putText(arr, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3 * lw, cv2.LINE_AA)
            cv2.putText(arr, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, lw, cv2.LINE_AA)
    return torch.from_numpy(arr).float().div(255.0).unsqueeze(0)


# ── L9.172: merge helpers (seam inspection, adaptive weights) ──────────────────────────────────────────────────

def _seam_stats(per_tile, pl, frames: int, out_h: int, out_w: int):
    """(heat [B,H,W] = weighted std-dev of the overlapping tiles' luma, 0 where one tile covers; coverage count
    [H,W]; the tiles' luma)."""
    lums = [luma(p) for p in per_tile]
    dev = lums[0].device
    s0 = torch.zeros((frames, out_h, out_w), dtype=torch.float32, device=dev)
    s1 = torch.zeros_like(s0)
    count = torch.zeros((out_h, out_w), dtype=torch.float32, device=dev)
    for (top, left, ph, pw, w2), l in zip(pl, lums):
        s0[:, top:top + ph, left:left + pw] += w2
        s1[:, top:top + ph, left:left + pw] += w2 * l
        count[top:top + ph, left:left + pw] += 1
    mean = s1 / s0.clamp(min=1e-8)
    var = torch.zeros_like(s0)
    for (top, left, ph, pw, w2), l in zip(pl, lums):     # two-pass: no cancellation noise where tiles agree
        d = l - mean[:, top:top + ph, left:left + pw]
        var[:, top:top + ph, left:left + pw] += w2 * d * d
    heat = torch.sqrt((var / s0.clamp(min=1e-8)).clamp(min=0.0)) * (count >= 2).float()
    return heat, count, lums


def _adaptive_merge(per_tile, pl, merged: torch.Tensor, lums, count, exponent: float) -> torch.Tensor:
    """Merge with each tile's window weight times q ** exponent, q = (local detail + 0.01) / (local disagreement with
    the plain merge + 0.01) (7x7 box means of luma gradient / of |tile - merge|, clamped to [0.1, 10]). Only the ratio
    between the tiles at a pixel matters (the weights are renormalised there). Pixels one tile covers keep the plain
    merge exactly.

    Deviation from the first design (a per-tile mean normalisation): measured, the single-tile area - where the
    disagreement is 0 - dominated that mean and clamped every overlap pixel to the floor, so the softer tile could
    win. The 0.01 floor (image units) keeps q finite without it."""
    m = merged.float()
    lm = luma(m)
    acc = torch.zeros_like(m)
    wacc = torch.zeros((*m.shape[:3], 1), dtype=torch.float32, device=m.device)
    for (top, left, ph, pw, w2), p, l in zip(pl, per_tile, lums):
        d = box_mean(gradient_magnitude(l))
        e = box_mean((l - lm[:, top:top + ph, left:left + pw]).abs())
        q = ((d + 0.01) / (e + 0.01)).clamp(0.1, 10.0)
        w = (w2.unsqueeze(0) * q.pow(float(exponent))).unsqueeze(-1)
        acc[:, top:top + ph, left:left + pw, :] += p.float() * w
        wacc[:, top:top + ph, left:left + pw, :] += w
    out = acc / wacc.clamp(min=1e-8)
    out = torch.where((count == 1).view(1, *count.shape, 1), m, out)
    return out.to(merged.dtype)


def _seam_report(plan: TilePlan, pl, lums, heat, threshold: float) -> list:
    index = {(t.row, t.col): i for i, t in enumerate(plan.tiles)}
    rows = []
    for i, t in enumerate(plan.tiles):
        for dr, dc, axis in ((0, 1, "x"), (1, 0, "y")):
            j = index.get((t.row + dr, t.col + dc))
            if j is None:
                continue
            ti, li, hi, wi, _ = pl[i]
            tj, lj, hj, wj, _ = pl[j]
            y0, y1 = max(ti, tj), min(ti + hi, tj + hj)
            x0, x1 = max(li, lj), min(li + wi, lj + wj)
            if y1 <= y0 or x1 <= x0:
                continue
            a = lums[i][:, y0 - ti:y1 - ti, x0 - li:x1 - li]
            b = lums[j][:, y0 - tj:y1 - tj, x0 - lj:x1 - lj]
            diff = (a - b).abs()
            flagged = threshold > 0 and bool((heat[:, y0:y1, x0:x1] > threshold).any())
            rows.append({"tiles": [i, j], "axis": axis, "band": [x0, y0, x1 - x0, y1 - y0],
                         "mean_disagreement": round(float(diff.mean()), 6),
                         "max_disagreement": round(float(diff.max()), 6), "flagged": flagged})
    return rows


class TilePlanMEC:
    """Work out the tiles once, for the whole clip."""

    VRAM_TIER = 0

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "image": ("IMAGE", {"tooltip": "Any frame at the size you will "
                                               "be tiling. Only its dimensions "
                                               "are read."}),
                "mode": (["tile size", "tile count"], {"default": "tile size",
                         "tooltip": "tile size: the largest equal tiles up to tile_size. tile count: a fixed rows x "
                                    "cols grid."}),
                "tile_size": ("INT", {
                    "default": 1024, "min": 128, "max": 8192, "step": 64,
                    "tooltip": "A MAXIMUM, not a target. Tiles are equal-sized "
                               "and must cover the frame exactly, so the size "
                               "chosen is the largest at or below this that "
                               "does. The report says what it picked."}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 16, "tooltip": "tile count: tiles down the frame."}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 16, "tooltip": "tile count: tiles across the frame."}),
                "overlap": ("INT", {
                    "default": 96, "min": 0, "max": 512, "step": 8,
                    "tooltip": "How far neighbouring tiles share pixels. The "
                               "blend has to hide any difference between two "
                               "tiles inside this band, so too narrow shows a "
                               "seam. 64-128px is usual."}),
                "blend": (["cosine", "linear", "none"], {
                    "default": "cosine",
                    "tooltip": "cosine is the right answer: the fade-in and "
                               "fade-out sum to exactly 1 across the overlap, "
                               "so there is no seam and no brightness dip. "
                               "'none' is for checking where the tiles are."}),
            },
            # L9.172 (A25 spec 59): appended last, defaults = the plan exactly as before
            "optional": {
                "vram_budget_mb": ("INT", {
                    "default": 0, "min": -1, "max": 1048576, "step": 256,
                    "tooltip": "Memory budget for ONE tile run of the refiner, in MB. 0 = off (the plan as before). "
                               "-1 = auto: 80% of the free memory ComfyUI reports for its device. Above 0 = this "
                               "budget. While the estimate (see mb_per_megapixel) is over it, tile size mode lowers "
                               "the tile size in 64px steps and tile count mode adds rows / columns."}),
                "mb_per_megapixel": ("FLOAT", {
                    "default": 1500.0, "min": 1.0, "max": 100000.0, "step": 50.0,
                    "tooltip": "Your calibration of the refiner: MB it needs per megapixel per frame. The estimate "
                               "per tile = tile width x height / 1e6 x frames x this (tile-major batches send a "
                               "tile's whole time range in one run). Only used when vram_budget_mb is not 0."}),
                "content_aware": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move the seams off detail and off the subject: tiles grow by 2 x (overlap / 2, at "
                               "least 8px) so every seam may shift that far, overlaps never shrink below overlap, "
                               "and the blend bands go where the image (averaged over ALL frames - one plan per "
                               "clip) has the least edge detail and subject_mask. Reads the pixels of image. Off = "
                               "evenly spaced tiles as before."}),
                "subject_weight": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "content_aware: how strongly subject_mask repels a seam, relative to edge detail "
                               "(0 = ignore the mask)."}),
                "subject_mask": ("MASK", {
                    "tooltip": "content_aware: where the subject is (white). Seams avoid it, averaged over the "
                               "clip's frames."}),
            },
        }, _PLAN_LAYOUT)

    RETURN_TYPES = ("TILE_PLAN", "STRING")
    RETURN_NAMES = ("tile_plan", "info")
    OUTPUT_TOOLTIPS = ("The plan: connect it to Tile Split and Tile Merge (the same plan for both).",
                       "What it chose: tile size, grid, overlap.")
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Decide the tiles once for a whole clip. Reusing one plan "
                   "for every frame is what stops seams crawling.")

    @classmethod
    def IS_CHANGED(cls, image=None, **kw):
        # the subject mask is a tensor: hashed by content (a None one adds nothing, so old prompts keep their value)
        return _fingerprint(image, kw.pop("subject_mask", None), **kw)

    def plan(self, image, mode, tile_size, rows, cols, overlap, blend,
             vram_budget_mb=0, mb_per_megapixel=1500.0, content_aware=False, subject_weight=4.0,
             subject_mask=None):
        h, w = _spatial(image)
        with torch.no_grad():
            content = bool(content_aware) and int(overlap) > 0
            extra = []
            if bool(content_aware) and int(overlap) <= 0:
                extra.append("content_aware is on but the overlap is 0: there is no blend band to move, so the "
                             "tiles are evenly spaced.")

            def make(tile, r, c):
                if content:
                    return _content_plan(h, w, mode, tile, r, c, int(overlap), blend, image, subject_mask,
                                         float(subject_weight))
                if mode == "tile size":
                    return plan_for_tile_size(h, w, tile=int(tile), overlap=int(overlap), blend=blend), []
                return build_plan(h, w, rows=int(r), cols=int(c), overlap=int(overlap), blend=blend), []

            budget = None
            budget_int = int(vram_budget_mb)
            if budget_int == -1:
                budget, why = _auto_budget_mb()
                if budget is None:
                    extra.append(f"vram_budget_mb = auto, but the free memory could not be read ({why}); planned "
                                 "without a budget.")
            elif budget_int > 0:
                budget = float(budget_int)
            elif budget_int < -1:
                raise TilingError(f"vram_budget_mb must be -1 (auto), 0 (off) or a budget in MB, got {budget_int}.")

            frames = _frames_of(image)

            def estimate(p):
                t = p.tiles[0]
                return t.width * t.height / 1e6 * frames * float(mb_per_megapixel)

            tile, r, c = int(tile_size), int(rows), int(cols)
            plan, lines = make(tile, r, c)
            if budget is not None:
                floor = 128
                while estimate(plan) > budget:
                    if mode == "tile size":
                        tile -= 64
                        if tile < floor or tile <= int(overlap) + 64:
                            raise TilingError(
                                f"No tile size fits the {budget:.0f} MB budget: even {tile + 64}px tiles need "
                                f"{estimate(plan):.0f} MB ({frames} frame(s) x {float(mb_per_megapixel):g} MB per "
                                "megapixel). Raise the budget, lower mb_per_megapixel if it is pessimistic, or "
                                "send fewer frames per run.")
                    else:
                        t0 = plan.tiles[0]
                        if t0.height >= t0.width and r < 16:
                            r += 1
                        elif c < 16:
                            c += 1
                        elif r < 16:
                            r += 1
                        else:
                            raise TilingError(
                                f"No tile count up to 16 x 16 fits the {budget:.0f} MB budget: 16 x 16 tiles still "
                                f"need {estimate(plan):.0f} MB each. Raise the budget or lower mb_per_megapixel.")
                    try:
                        plan, lines = make(tile, r, c)
                    except TilingError as exc:
                        raise TilingError(
                            f"Fitting the {budget:.0f} MB budget needed tiles this plan cannot make "
                            f"({exc}). Raise the budget or lower the overlap.") from None
                t0 = plan.tiles[0]
                extra.append(
                    f"Memory: about {estimate(plan):.0f} MB per tile run ({t0.width}x{t0.height} x {frames} "
                    f"frame(s) x {float(mb_per_megapixel):g} MB/MP) within the {budget:.0f} MB budget"
                    + (" (auto)" if budget_int == -1 else "") + f"; {len(plan.tiles)} tiles.")
            t0 = plan.tiles[0]
            if t0.width % 16 or t0.height % 16:
                # audit 2026-11-27: equal tiles that cover the frame exactly can come out at any size (588, 747...).
                extra.append(
                    f"NOTE: the tiles are {t0.width}x{t0.height}, not a multiple of 16"
                    + (" (nor of 8)" if t0.width % 8 or t0.height % 8 else "") + ". A VAE / video model crops or pads "
                    "such tiles (8x VAEs need /8, Wan / H3-class video VAEs /16, LTX /32), and tiles that come back "
                    "a different size cannot be merged. Pick a tile_size / rows / cols / overlap that lands on a "
                    "multiple, or resize the tiles to one before the refiner and back after it.")
            info = describe_plan(plan)
            if lines or extra:
                info = "\n".join([info, *lines, *extra])
        return (plan, info)


class TileSplitMEC:
    """Every tile of every frame, as one batch."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The clip to cut up (the size the plan was made for)."}),
                "tile_plan": ("TILE_PLAN", {"tooltip": "Tile Plan's output."}),
            },
            # L9.172 (A25 spec 60): appended last, defaults = the split exactly as before
            "optional": {
                "halo_px": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Extra context around every tile, in pixels on each side, so the refiner sees what is "
                               "beyond the tile's edge (frame edges are padded by repeating the border, so every "
                               "tile keeps one size). Tile Merge cuts it off again - wire THIS node's tile_plan "
                               "output to Tile Merge. 0 = off."}),
                "motion_halo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add the clip's largest frame-to-frame global movement (measured by phase "
                               "correlation, in pixels, rounded up) to halo_px, so moving content near a tile edge "
                               "is inside the context in every frame. Needs 2+ frames."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "IMAGE", "STRING", "TILE_PLAN")
    OUTPUT_TOOLTIPS = ("Every tile of every frame as one batch, TILE-MAJOR: all frames of tile 1, then all frames of tile 2, in plan order (tile_coords gives each tile's batch_start / batch_end).",
                       "How many tiles came out (frames x tiles per frame).", "What was split, as text.",
                       "Frame 0 with every tile outlined in its own colour, overlaps darkened and the tile id "
                       "written in (ids need OpenCV; outlines always).",
                       "JSON, one entry per tile: id, row, col, x, y, w, h, the four fades, halo, and batch_start / "
                       "batch_end (end exclusive) - where that tile's frames sit in the tiles batch.",
                       "The plan with this split's halo recorded. Wire it to Tile Merge when halo_px or "
                       "motion_halo is used (without a halo it is the input plan).")
    RETURN_NAMES = ("tiles", "tile_count", "info", "tile_map", "tile_coords", "tile_plan")
    FUNCTION = "split"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Split a clip into overlapping tiles, emitted as a batch so "
                   "any refiner can process them. Wire the result through "
                   "whatever you like, then Tile Merge.")

    @classmethod
    def IS_CHANGED(cls, image=None, tile_plan=None, **kw):
        # core passes every input (L9.172 appended halo_px / motion_halo); none given = the old value
        return _fingerprint(image, tile_plan, **kw)

    def split(self, image, tile_plan, halo_px=0, motion_halo=False):
        if not isinstance(tile_plan, TilePlan):
            raise TilingError(
                "The tile_plan input is not a tile plan. Wire it from Tile "
                "Plan's own output.")
        h, w = _spatial(image)
        if (h, w) != (tile_plan.height, tile_plan.width):
            raise TilingError(
                f"This plan is for {tile_plan.width}x{tile_plan.height} but the "
                f"image is {w}x{h}. Build the plan from the same footage you "
                "are tiling, or every tile lands on the wrong part of the frame.")

        frames = image.shape[0]
        # Tile-major: all frames of tile 0, then all frames of tile 1. A video
        # refiner reading this batch as a sequence sees one tile's whole time
        # range contiguously, which is what lets it be temporally consistent.
        halo = int(halo_px)
        if halo < 0:
            raise TilingError(f"halo_px cannot be negative, got {halo}.")
        motion_note = None
        if motion_halo:
            if image.ndim == 5 or (image.shape[0] if image.ndim == 4 else 0) < 2:
                motion_note = ("motion_halo: needs a clip of 2+ frames as an IMAGE batch; nothing measured, "
                               "no motion margin added.")
            else:
                with torch.no_grad():
                    shift = max_global_shift(_luma_clip(image).cpu())
                add = int(math.ceil(shift - 1e-6)) if shift > 0 else 0
                halo += add
                motion_note = (f"motion_halo: largest frame-to-frame movement {shift:.2f}px, so {add}px added to "
                               "the halo.")
        if halo > 0:
            with torch.no_grad():
                padded = _pad_replicate(image, halo)
                pieces = [_crop_padded(padded, t, halo) for t in tile_plan.tiles]
        else:
            pieces = [crop(image, t) for t in tile_plan.tiles]
        batch = torch.cat(pieces, dim=0)
        out_plan = dataclasses.replace(tile_plan, halo=halo) if halo != tile_plan.halo else tile_plan

        info = [
            f"{len(tile_plan.tiles)} tiles x {frames} frame(s) = "
            f"{batch.shape[0]} images out, each "
            f"{batch.shape[2]}x{batch.shape[1]}.",
            "Batch order is TILE-MAJOR: every frame of tile 1, then every "
            "frame of tile 2, and so on. A video refiner therefore sees each "
            "tile's whole time range in one run, which is what it needs to be "
            "temporally consistent.",
            "Whatever you wire next must return the SAME number of images in "
            "the SAME order. It may change their size, as long as every tile "
            "changes by the same factor.",
        ]
        if batch.shape[0] > 64:
            info.append(
                f"WARNING: {batch.shape[0]} images is a large batch. Most "
                "samplers will try to do it in one go and run out of memory - "
                "consider fewer tiles, or a node that batches internally.")
        if frames > 1 and len(tile_plan.tiles) > 1:
            info.append(
                f"VIDEO MODELS: this is ONE flat batch of {len(tile_plan.tiles)} x {frames} frames. A video VAE / "
                "video model reading it as one clip would see every tile's frames end to end (a cut at each tile "
                "boundary, and a frame count off the model's grid, e.g. 4n+1 / 17n+5). Run each tile's range on its "
                "own (tile_coords batch_start / batch_end, e.g. with Image Batch Slice), then concatenate in the same "
                "order for Tile Merge.")
        if halo > 0:
            info.append(
                f"Halo: every tile carries {halo}px of context on each side (frame edges repeat the border). Wire "
                "this node's tile_plan output to Tile Merge - it cuts the halo off before blending.")
        if motion_note:
            info.append(motion_note)
        with torch.no_grad():
            tmap = _tile_map(image, tile_plan)
        coords = json.dumps([{
            "id": i, "row": t.row, "col": t.col, "x": t.left, "y": t.top, "w": t.width, "h": t.height,
            "fade_top": t.fade_top, "fade_bottom": t.fade_bottom, "fade_left": t.fade_left,
            "fade_right": t.fade_right, "halo": halo, "batch_start": i * frames, "batch_end": (i + 1) * frames,
        } for i, t in enumerate(tile_plan.tiles)])
        return (batch, len(tile_plan.tiles), "\n".join(info), tmap, coords, out_plan)


class TileMergeMEC:
    """Put the tiles back, weighted, with no seam."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {"tooltip": "The tiles after processing, in the order Tile Split gave them "
                                               "(they may be enlarged)."}),
                "tile_plan": ("TILE_PLAN", {"tooltip": "The same plan Tile Split used."}),
            },
            "optional": {
                "scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "How much the refiner enlarged each tile. 0 "
                               "means work it out from the tiles themselves, "
                               "which is right when every tile scaled by the "
                               "same clean factor. Set it explicitly if the "
                               "ratio is not a round number."}),
                # L9.172 (A25 spec 58): appended last, 0 = the merge exactly as before
                "adaptive_blend": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Where tiles overlap, give more weight to the tile that has more local detail and "
                               "agrees better with the blended result (7x7 neighbourhood): weight x (detail / "
                               "disagreement) ^ this. 0 = the plain cosine / linear blend as before. Pixels only one "
                               "tile covers never change."}),
                "reblend_threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Selective repair: overlap pixels where the tiles disagree by more than this (standard "
                               "deviation of their luma, 0-1 image units, as in seam_heatmap before scaling) are "
                               "re-blended with the adaptive weights at full strength; the rest is left as it is. "
                               "0 = off. seam_report marks the seams it touched."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "STRING")
    RETURN_NAMES = ("image", "info", "seam_heatmap", "seam_report")
    OUTPUT_TOOLTIPS = ("The clip put back together, seams blended.", "What was merged and at what scale.",
                       "Where the tiles disagree inside their overlaps (weighted standard deviation of their luma), "
                       "scaled so the clip's worst pixel is 1; black where one tile covers or all tiles agree.",
                       "JSON, one entry per pair of neighbouring tiles: tiles, axis (x = side by side, y = above / "
                       "below), the overlap band [x, y, w, h] in output pixels, mean and max luma disagreement, and "
                       "whether reblend_threshold repaired it.")
    FUNCTION = "merge"
    CATEGORY = CATEGORY
    DESCRIPTION = "Blend the tiles back into one clip, in float32, with no seam."

    @classmethod
    def IS_CHANGED(cls, tiles=None, tile_plan=None, scale=0.0, **kw):
        # core passes every input (L9.172 appended adaptive_blend / reblend_threshold); none given = the old value
        return _fingerprint(tiles, tile_plan, scale=scale, **kw)

    def merge(self, tiles, tile_plan, scale=0.0, adaptive_blend=0.0, reblend_threshold=0.0):
        if not isinstance(tile_plan, TilePlan):
            raise TilingError(
                "The tile_plan input is not a tile plan. It must be the SAME "
                "plan Tile Split used - a different one puts the tiles back in "
                "the wrong places.")

        n_tiles = len(tile_plan.tiles)
        total = tiles.shape[0]
        if total % n_tiles:
            raise TilingError(
                f"{total} images came back but the plan has {n_tiles} tiles, "
                f"which does not divide evenly. Whatever refined the tiles "
                "changed how many there are - it must return the same count, "
                "in the same order.")
        frames = total // n_tiles

        th, tw = _spatial(tiles)
        src = tile_plan.tiles[0]
        halo = int(getattr(tile_plan, "halo", 0) or 0)          # L9.172: context Tile Split added around each tile
        if scale <= 0:
            ratio_h = th / float(src.height + 2 * halo)
            ratio_w = tw / float(src.width + 2 * halo)
            if abs(ratio_h - ratio_w) > 0.01:
                raise TilingError(
                    f"The tiles came back {tw}x{th} from {src.width}x"
                    f"{src.height} - that is {ratio_w:.3f}x wide by "
                    f"{ratio_h:.3f}x tall, which is not a uniform scale. Set "
                    "the scale input explicitly, or make the refiner preserve "
                    "the aspect."
                    + (f" (The plan says each tile carries a {halo}px halo.)" if halo else ""))
            scale = ratio_h
        adaptive = float(adaptive_blend)
        threshold = float(reblend_threshold)
        if not 0.0 <= adaptive <= 1.0:
            raise TilingError(f"adaptive_blend must be between 0 and 1, got {adaptive}.")
        if threshold < 0.0:
            raise TilingError(f"reblend_threshold cannot be negative, got {threshold}.")
        if halo:
            cut = int(round(halo * float(scale)))
            if th - 2 * cut < 1 or tw - 2 * cut < 1:
                raise TilingError(
                    f"The tiles are {tw}x{th} but the plan's {halo}px halo at {float(scale):.3f}x is {cut}px per "
                    "side - nothing would be left. Wire Tile Split's own tile_plan output here.")
            tiles = (tiles[:, cut:th - cut, cut:tw - cut, :] if _is_bhwc(tiles)
                     else tiles[..., cut:th - cut, cut:tw - cut])

        # Regroup from tile-major back to per-tile clips.
        per_tile = [tiles[i * frames:(i + 1) * frames] for i in range(n_tiles)]
        out = merge(per_tile, tile_plan, scale=float(scale))

        info = [
            f"Merged {n_tiles} tiles x {frames} frame(s) into "
            f"{out.shape[2]}x{out.shape[1]}, scale {scale:.3f}x.",
            "Blended in float32 and divided by the accumulated weight, so the "
            "frame edges - where the windows cannot sum to 1 because there is "
            "no neighbour - come out at the right level rather than dark.",
        ]
        if halo:
            info.append(f"Cut the {halo}px halo ({cut}px at this scale) off every tile before blending.")

        out_h, out_w = _spatial(out)
        if _is_bhwc(out) and n_tiles > 1:
            with torch.no_grad():
                pl = placements(per_tile, tile_plan, scale=float(scale))
                heat, count, lums = _seam_stats(per_tile, pl, frames, out_h, out_w)
                final = out
                if adaptive > 0:
                    final = _adaptive_merge(per_tile, pl, out, lums, count, adaptive)
                    info.append(f"Adaptive seam blend at {adaptive:.2f}: overlap weights follow local detail and "
                                "agreement; single-tile pixels unchanged.")
                if threshold > 0:
                    sel = heat > threshold
                    n_sel = int(sel.sum())
                    if n_sel:
                        full = final if adaptive == 1.0 else _adaptive_merge(per_tile, pl, out, lums, count, 1.0)
                        final = torch.where(sel.unsqueeze(-1), full, final)
                    info.append(f"Selective reblend: {n_sel} overlap pixel(s) over the {threshold:g} disagreement "
                                "threshold re-blended with full adaptive weights.")
                report = _seam_report(tile_plan, pl, lums, heat, threshold)
                peak = float(heat.max()) if heat.numel() else 0.0
                heatmap = heat / peak if peak > 1e-6 else torch.zeros_like(heat)
                out = final
        else:
            if adaptive > 0 or threshold > 0:
                if not _is_bhwc(out):
                    raise TilingError("adaptive_blend / reblend_threshold work on IMAGE tiles [B,H,W,C].")
                info.append("adaptive_blend / reblend_threshold: one tile, no seams to work on.")
            heatmap = torch.zeros((frames, out_h, out_w), dtype=torch.float32)
            report = []
        return (out, "\n".join(info), heatmap.float().cpu(), json.dumps(report))


class IdentityLockMEC:
    """Take the refinement's detail; keep the plate's structure."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "original": ("IMAGE", {
                    "tooltip": "The plate, at the SAME size as the refined "
                               "image. If the refinement upscaled, upscale "
                               "this to match first."}),
                "refined": ("IMAGE", {"tooltip": "What came back from the model."}),
                "radius": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 128.0, "step": 0.5,
                    "tooltip": "The split frequency. SMALLER locks harder and "
                               "lets less refinement through; LARGER passes "
                               "more detail and more movement with it. If a "
                               "face is still moving, go SMALLER."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = the raw refinement, 1 = fully locked. Set "
                               "it to 0 WITH a protect mask to lock only the "
                               "face and leave the background completely free."}),
                "corrections": ("INT", {
                    "default": 2, "min": 0, "max": 8,
                    "tooltip": "Extra passes that push the result's structure "
                               "back onto the plate's. A Gaussian is not a "
                               "brick-wall filter, so one pass alone holds "
                               "only about 70% of a face shift; two hold about "
                               "94%, at a 2% cost in detail."}),
            },
            "optional": {
                "protect": ("MASK", {
                    "tooltip": "Where the lock is at FULL strength regardless "
                               "of the strength above - a face. Everywhere "
                               "else gets whatever strength says."}),
                "want_drift_map": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also output a map of where structure moved. "
                               "Bright means it moved. Use it to find what the "
                               "drift number is telling you about."}),
                # L9.172 (A25 spec 57): appended last, defaults = the lock exactly as before
                "band_radii": ("STRING", {
                    "default": "",
                    "tooltip": "Multi-band lock: comma-separated blur radii in px, fine to coarse (e.g. '2, 8, 32'). "
                               "They cut the picture into bands: detail finer than the first, one band between each "
                               "pair, and the structure coarser than the last. Empty = the single split at radius "
                               "(as before). Needs band_strengths."}),
                "band_strengths": ("STRING", {
                    "default": "",
                    "tooltip": "Multi-band lock: one number 0-1 per band (radii + 1 of them, fine to coarse) = how "
                               "much of that band comes from the PLATE (1 = locked, 0 = the refinement's). Example "
                               "'0, 0.5, 1, 1' keeps the model's fine detail, half the mid band and the plate's "
                               "structure. strength fades them all; inside protect they apply in full."}),
                "track_protect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When protect has fewer frames than the clip (e.g. one mask drawn on frame 0), follow "
                               "the protected region through the original clip (phase correlation, frame to frame) "
                               "and move the mask with it. The offsets are in drift_report. Off = the mask as given."}),
            },
        }, _LOCK_LAYOUT)

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("image", "drift_map", "drift", "info", "drift_report")
    OUTPUT_TOOLTIPS = ("The refined detail on the plate's structure.",
                       "Where structure moved (bright = moved); all black unless want_drift_map is on.",
                       "How much structure the refinement moved (lower = closer to the plate).",
                       "Settings used and the drift numbers, as text.",
                       "JSON per frame: structure drift inside the protect region (the whole frame without one), "
                       "how far that region moved from the original to the result in pixels [dy, dx] (phase "
                       "correlation), and the tracked mask offset when track_protect is on.")
    FUNCTION = "lock"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Refine every pixel without letting anything move. Keeps "
                   "the plate's low frequencies - where composition and face "
                   "geometry live - and takes only the model's high "
                   "frequencies. The face cannot move, because the only thing "
                   "taken from the model is the band that does not encode "
                   "where anything is.")

    @classmethod
    def IS_CHANGED(cls, original=None, refined=None, **kw):
        return _fingerprint(original, refined, **kw)

    def lock(self, original, refined, radius, strength, corrections,
             protect=None, want_drift_map=False, band_radii="", band_strengths="", track_protect=False):
        if original.shape != refined.shape:
            raise IdentityLockError(
                f"The plate is {original.shape[2]}x{original.shape[1]} and the "
                f"refined image is {refined.shape[2]}x{refined.shape[1]}. They "
                "must match - upscale the plate to the refined size first, or "
                "the lock would compare different pixels.")

        bands = parse_bands(band_radii, band_strengths)          # None = the single split (as before)
        notes = []
        tracked = None
        if track_protect:
            if protect is None:
                notes.append("track_protect is on but no protect mask is connected: nothing to track.")
            else:
                pf = protect if protect.ndim == 3 else protect.unsqueeze(0)
                if int(pf.shape[0]) < int(original.shape[0]):
                    with torch.no_grad():
                        protect, tracked = track_protect_mask(original, pf[0])
                    last = tracked[-1]
                    notes.append(
                        f"Protect mask tracked through {original.shape[0]} frames from its frame 0 "
                        f"(last offset dy={last[0]}, dx={last[1]} px)."
                        + (f" The mask had {pf.shape[0]} frames; only the first was used." if pf.shape[0] > 1
                           else ""))
                else:
                    notes.append("track_protect: the protect mask already has a frame for every frame; used as "
                                 "given.")

        with low_pass_cache():                  # audit 2026-11-27: each blur of the clip once per run
            if bands is None:
                out = identity_lock(original, refined, radius=float(radius),
                                    strength=float(strength), protect=protect,
                                    corrections=int(corrections))
            else:
                with torch.no_grad():
                    out = multiband_lock(original, refined, radii=bands[0], strengths=bands[1],
                                         strength=float(strength), protect=protect)
            moved = drift(original, out, radius=float(radius))
            info = describe_lock(original, out, radius=float(radius),
                                 strength=float(strength),
                                 protected=protect is not None,
                                 corrections=int(corrections))
            if bands is not None:
                info = info + "\n" + describe_bands(*bands)
            if notes:
                info = "\n".join([info, *notes])
            with torch.no_grad():
                report = drift_report(original, out, radius=float(radius), protect=protect, tracked_offsets=tracked)

            if want_drift_map:
                dmap = drift_map(original, out, radius=float(radius))
            else:
                # An all-zero map rather than None: a MASK output that is sometimes
                # absent breaks any graph that wires it.
                dmap = torch.zeros(original.shape[0], original.shape[1],
                                   original.shape[2], dtype=torch.float32)
        return (out, dmap, float(moved), info, json.dumps(report))


NODE_CLASS_MAPPINGS = {
    "TilePlanMEC": TilePlanMEC,
    "TileSplitMEC": TileSplitMEC,
    "TileMergeMEC": TileMergeMEC,
    "IdentityLockMEC": IdentityLockMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TilePlanMEC": "Tile Plan",
    "TileSplitMEC": "Tile Split",
    "TileMergeMEC": "Tile Merge",
    "IdentityLockMEC": "Identity Lock — refine without moving",
}
