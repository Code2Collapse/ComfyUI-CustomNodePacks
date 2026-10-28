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
                "max_side": ("INT", {"default": C.MAX_SIDE, "min": 64, "max": 2048, "step": 16}),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 2000}),
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
