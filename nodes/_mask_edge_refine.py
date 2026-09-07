"""Shared, model-free mask edge-refinement + prompt parsing helpers.

Used by the LIVE mask generators (sam_mask_generator.py) to turn a binary SAM
mask into a soft, roto-grade alpha and to accept multi-bbox / spline prompts.
Pure torch (+ a lazy matte-backend delegate); never raises on the fallback path.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("MEC.mask_edge_refine")

EDGE_MODES = ["none", "guided", "guided_strong", "matte"]


# ── guided filter (He et al. 2010) — pure torch, no deps ──────────────
def _box(x, r):
    k = 2 * r + 1
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.avg_pool2d(x, k, stride=1)


def _guided_filter(guide, src, r, eps):
    g = guide[None, None]
    p = src[None, None]
    mean_g = _box(g, r)
    mean_p = _box(p, r)
    var_g = _box(g * g, r) - mean_g * mean_g
    cov_gp = _box(g * p, r) - mean_g * mean_p
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    return (_box(a, r) * g + _box(b, r))[0, 0]


def refine_edges(masks, image, mode="guided", radius=8):
    """Edge-snap a [B,H,W] or [H,W] mask against the IMAGE luma guide → soft
    0..1 alpha (matching input rank). ``mode`` ∈ EDGE_MODES. ``matte`` delegates
    to the mask_matting backends (ViTMatte/BiRefNet) and falls back to ``guided``
    when no model is installed. Never raises — returns the input mask on failure."""
    if masks is None or mode == "none":
        return masks
    if mode == "matte":
        try:
            return _matte_refine(masks, image, radius)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MEC] matte refine unavailable (%s) — using guided filter.", exc)
            mode = "guided"
    try:
        was2d = masks.ndim == 2
        m = masks.unsqueeze(0) if was2d else masks
        Bm, H, W = m.shape
        img = image
        if img.ndim == 4:
            guide_all = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        else:
            guide_all = img if img.ndim == 3 else img.unsqueeze(0)
        r = max(1, int(radius)) * (2 if mode == "guided_strong" else 1)
        out = torch.empty((Bm, H, W), dtype=torch.float32, device=m.device)
        for i in range(Bm):
            gi = min(i, guide_all.shape[0] - 1)
            guide = guide_all[gi].to(m.device, torch.float32)
            if guide.shape != (H, W):
                guide = F.interpolate(guide[None, None], size=(H, W),
                                      mode="bilinear", align_corners=False)[0, 0]
            src = m[i].to(torch.float32).clamp(0, 1)
            out[i] = _guided_filter(guide, src, r, 1e-4).clamp(0, 1)
        out = out.to(masks.dtype)
        return out[0] if was2d else out
    except Exception as exc:  # noqa: BLE001
        logger.warning("[MEC] edge_refine failed (%s) — using raw mask.", exc)
        return masks


def _matte_refine(masks, image, radius):
    """Roto-grade alpha matting via the existing mask_matting backends."""
    from .mask_matting.matters import get_matter_cls, list_keys

    ready = list_keys(installed_only=True)
    if not ready:
        raise RuntimeError("no matte backend installed (ViTMatte/BiRefNet/…)")
    order = [k for k in ("vitmatte", "rmbg2", "birefnet", "rvm", "bgmattingv2") if k in ready]
    key = order[0] if order else ready[0]
    Matter = get_matter_cls(key)
    if Matter is None:
        raise RuntimeError(f"matte backend '{key}' not resolvable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matter = Matter(device=device)
    was2d = masks.ndim == 2
    m = (masks.unsqueeze(0) if was2d else masks).float()
    img = (image if image.ndim == 4 else image.unsqueeze(0)).float()
    if img.shape[0] != m.shape[0]:
        img = img[:1].repeat(m.shape[0], 1, 1, 1) if img.shape[0] == 1 else img[: m.shape[0]]
    res = matter.matte(img, m, trimap=None, edge_radius=int(radius))
    alpha = res.get("alpha") if isinstance(res, dict) else None
    if alpha is None:
        raise RuntimeError(f"matte backend '{key}' returned no alpha")
    alpha = alpha.to(masks.device, masks.dtype).clamp(0, 1)
    return alpha[0] if was2d else alpha


# ── prompt parsing ────────────────────────────────────────────────────
def parse_bboxes(s):
    """JSON list of boxes → list of [x1,y1,x2,y2] floats. Accepts a single box
    too. Empty/invalid → []."""
    if not s or not str(s).strip():
        return []
    try:
        data = json.loads(s)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, list) or not data:
        return []
    if len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
        return [[float(v) for v in data]]
    out = []
    for b in data:
        if isinstance(b, (list, tuple)) and len(b) >= 4 and all(isinstance(v, (int, float)) for v in b[:4]):
            out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return out


def parse_spline(spline_json):
    """SplineMask spline_data → (coords Nx2, labels N ones, bbox [x1,y1,x2,y2]) or
    None. Uses the polygon vertices as positive points + their bounding box."""
    if not spline_json or not str(spline_json).strip():
        return None
    try:
        data = json.loads(spline_json)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    pts = []
    for shape in data:
        if isinstance(shape, dict):
            for p in shape.get("points", []):
                if isinstance(p, dict) and "x" in p and "y" in p:
                    pts.append([float(p["x"]), float(p["y"])])
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append([float(p[0]), float(p[1])])
        elif isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[0], (int, float)):
            pts.append([float(shape[0]), float(shape[1])])
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    coords = np.array(pts, dtype=float)
    labels = np.ones(len(pts), dtype=int)
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    return coords, labels, bbox
