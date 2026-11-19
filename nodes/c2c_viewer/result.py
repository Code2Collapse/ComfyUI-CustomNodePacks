"""Viewer helper "Viewer Result" (L7.114): the matte a viewer render produced -> proxy frames + QC numbers.

The viewer's Render posts: the nodes feeding Mask + Matting, Mask + Matting itself (with the selection, and for the
preview chunk a frame range), and this node wired to its `mask` (coarse) and `alpha` outputs and to the same plate.
It writes the plate and the alpha of every rendered frame (every k-th on a long render) as proxy files, and measures
coverage / edge detail / flicker (qc.py) on the full rendered range. A SeC + MatAnyone2 run picked mid-clip starts
on the pick frame and leaves the frames before it empty (owner answer 2026-10-28, Q3): the Matte node's info output
names them (viewer.empty_frames), QC skips them and the payload passes them on so the viewer can say so.
"""
from __future__ import annotations

import hashlib
import json
import os

import torch

from . import _common as C
from . import qc


class C2CViewerResult:
    DEV_ONLY = True
    CATEGORY = C.HIDDEN_CATEGORY
    DESCRIPTION = ("Viewer helper (hidden): writes the rendered alpha and the plate as proxy frames and measures QC "
                   "for the Mask + Matting viewer. Not meant to be added by hand.")
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The whole plate (the rendered frames are cut from it)."}),
                "alpha": ("MASK", {"tooltip": "Mask + Matting's alpha."}),
                "start": ("INT", {"default": 0, "min": 0, "max": 1_000_000,
                                  "tooltip": "Plate frame of the first rendered frame (the preview chunk's start)."}),
            },
            "optional": {
                "coarse": ("MASK", {"tooltip": "Mask + Matting's mask (the segmenter's coarse selection)."}),
                "info": ("STRING", {"forceInput": True,
                                    "tooltip": "Mask + Matting's info output (names the frames a run left empty)."}),
                "max_side": ("INT", {"default": C.MAX_SIDE, "min": 64, "max": 2048, "step": 16, "tooltip": "Longest side of the preview images the viewer gets, in pixels (display only - the matte itself is untouched)."}),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 2000, "tooltip": "Most frames sent to the viewer; longer renders are sampled evenly."}),
            },
        }

    def execute(self, image, alpha, start=0, coarse=None, max_side=C.MAX_SIDE, max_frames=120, info=""):
        from ..mask_matting._stage_cache import fingerprint
        a = alpha if alpha.ndim == 3 else alpha.reshape(-1, *alpha.shape[-2:])
        n = int(a.shape[0])
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        s = int(start)
        if s + n > B:
            raise ValueError(f"The render has {n} frames from frame {s}, but the plate has {B}. Render again.")
        if tuple(a.shape[-2:]) != (H, W):
            raise ValueError(f"The alpha is {a.shape[-1]}x{a.shape[-2]} but the plate is {W}x{H}. Render again.")
        plate = image[s:s + n]
        c = None
        if coarse is not None:
            c = coarse if coarse.ndim == 3 else coarse.reshape(-1, *coarse.shape[-2:])
            c = c[:n] if c.shape[0] >= n else c[:1]
        empty = empty_frames(info, s, n)
        q0 = (empty[1] - s) if empty else 0                  # QC on the rendered frames only
        if q0 >= n:
            q0 = 0
        with torch.no_grad():
            report = qc.measure(plate[q0:], a[q0:], c[q0:] if c is not None and c.shape[0] == n else c)
        report["thresholds"] = qc.THRESHOLDS
        report["qc_side"] = qc.QC_SIDE

        pw, ph = C.proxy_size(W, H, int(max_side))
        idx, step = C.sample_indices(n, int(max_frames))
        key = fingerprint(a)[:12] + hashlib.sha1(f"{fingerprint(plate)}|{s}|{pw}".encode()).hexdigest()[:8]
        tag = f"result_{key}"
        d = C.out_dir(tag)
        frames = []
        for i in idx:
            pn, an = f"p{i:06d}.jpg", f"a{i:06d}.png"
            if not os.path.isfile(os.path.join(d, pn)):
                C.save_jpeg(C.resize_image(plate[i], (pw, ph)), os.path.join(d, pn))
            if not os.path.isfile(os.path.join(d, an)):
                C.save_alpha_png(C.resize_mask(a[i], (pw, ph)), os.path.join(d, an))
            frames.append({"index": s + i, "plate": C.file_ref(tag, pn), "alpha": C.file_ref(tag, an)})
        return C.ui({"kind": "result", "start": s, "count": n, "width": W, "height": H, "proxy": [pw, ph],
                     "step": step, "frames": frames, "qc": report, "empty": list(empty) if empty else None})


def empty_frames(info, start: int, count: int):
    """[first, stop) plate frames the Matte run left empty, cut to this render; None when there are none."""
    try:
        e = (json.loads(info) if isinstance(info, str) and info.strip() else {}).get("viewer", {}).get("empty_frames")
        a, b = max(int(e[0]), start), min(int(e[1]), start + count)
    except Exception:  # noqa: BLE001 - no info, another node's info, or no empty frames
        return None
    return (a, b) if b > a else None


# ── L7.124: the same Result payload from a node's own run (Mask Refiner, Mask Track: viewer QC mode) ───────────────

def _as_bhw(m):
    if m is None or not isinstance(m, torch.Tensor) or m.numel() == 0:
        return None
    if m.ndim == 2:
        return m.unsqueeze(0)
    if m.ndim == 3:
        return m
    if m.ndim == 4 and m.shape[-1] == 1:
        return m[..., 0]
    return None


def _frame(t, i):
    """Frame i of a batch, frame 0 of a single-frame batch (broadcast)."""
    return t[i] if t.shape[0] > 1 else t[0]


def node_payload(image, alpha, coarse=None, *, source: str, max_side: int = C.NODE_MAX_SIDE,
                 max_frames: int = C.NODE_MAX_FRAMES, qc_pairs: int = C.NODE_QC_PAIRS):
    """The viewer's Result payload ("kind": "result", the keys C2CViewerResult sends) for a node's own run, or None
    when it cannot be built (no plate, sizes differ, an empty batch).

    Bounded work, whatever the clip length: at most ``max_frames`` proxy frames (every k-th of a longer clip, at most
    ``max_side`` px on the long side) are written, and the QC numbers are measured on at most ``qc_pairs`` pairs of
    consecutive frames, each frame first scaled to qc.QC_SIDE (never the whole clip at full size). ``coarse`` (the
    node's input mask) gives coverage / edge detail; without it they are n/a and flicker uses the alpha's own edge.
    """
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] == 0:
        return None
    a = _as_bhw(alpha)
    if a is None:
        return None
    B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    if tuple(a.shape[-2:]) != (H, W):
        return None
    n = max(B, int(a.shape[0])) if 1 in (B, int(a.shape[0])) else min(B, int(a.shape[0]))
    c = _as_bhw(coarse)
    if c is not None and (tuple(c.shape[-2:]) != (H, W) or int(c.shape[0]) not in (1, n)):
        c = None
    with torch.no_grad():
        # QC on sampled consecutive pairs, each frame scaled down first
        ts = qc.pairs(n, max(1, int(qc_pairs)))
        need = sorted({t - 1 for t in ts} | set(ts)) or [0]
        hw = qc._scale(H, W)
        p_s = torch.cat([qc._resize_image(_frame(image, i)[None], hw) for i in need])
        a_s = torch.cat([qc._resize_bhw(_frame(a, i)[None].float().cpu().clamp(0, 1), hw) for i in need])
        c_s = (torch.cat([qc._resize_bhw(_frame(c, i)[None].float().cpu().clamp(0, 1), hw) for i in need])
               if c is not None else None)
        local_ts = [j for j in range(1, len(need)) if need[j] - need[j - 1] == 1]
        report = qc.measure(p_s, a_s, c_s, ts=local_ts)
        report["frames"] = n
        report["sampled_frames"] = len(need)
        report["thresholds"] = qc.THRESHOLDS
        report["qc_side"] = qc.QC_SIDE

        pw, ph = C.proxy_size(W, H, int(max_side))
        idx, step = C.sample_indices(n, int(max_frames))
        h = hashlib.sha1(f"{source}|{n}|{W}x{H}|{pw}x{ph}|{step}".encode())
        arrays = []
        for i in idx:
            pa = C.resize_image(_frame(image, i), (pw, ph))
            aa = C.resize_mask(_frame(a, i), (pw, ph))
            h.update(pa.tobytes())
            h.update(aa.tobytes())
            arrays.append((i, pa, aa))
    tag = f"node_{source}_{h.hexdigest()[:20]}"
    d = C.out_dir(tag)
    frames = []
    for i, pa, aa in arrays:
        pn, an = f"p{i:06d}.jpg", f"a{i:06d}.png"
        if not os.path.isfile(os.path.join(d, pn)):     # the same result again: the files are already there
            C.save_jpeg(pa, os.path.join(d, pn))
        if not os.path.isfile(os.path.join(d, an)):
            C.save_alpha_png(aa, os.path.join(d, an))
        frames.append({"index": i, "plate": C.file_ref(tag, pn), "alpha": C.file_ref(tag, an)})
    return {"kind": "result", "source": source, "start": 0, "count": n, "width": W, "height": H,
            "proxy": [pw, ph], "step": step, "frames": frames, "qc": report, "empty": None}


def with_node_viewer(result, image, alpha, coarse=None, *, source: str):
    """A node's return value with the viewer payload added to its ui (L7.124). ``result`` is the node's tuple or its
    {"ui", "result"} dict; the outputs are never touched. Unchanged when the viewer is off (C2C_VIEWER=0), when the
    node is called outside ComfyUI's executor (a plain tuple stays a plain tuple), or when anything goes wrong - the
    viewer must never fail the node."""
    try:
        if not C.viewer_enabled() or not C.in_executor():
            return result
        payload = node_payload(image, alpha, coarse, source=source)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger("C2C.Viewer").warning("[C2C] viewer payload for %s skipped: %s", source, exc)
        return result
    if payload is None:
        return result
    if isinstance(result, dict):
        ui = dict(result.get("ui") or {})
        ui[C.UI_KEY] = [payload]
        return {**result, "ui": ui}
    return {"ui": {C.UI_KEY: [payload]}, "result": tuple(result)}
