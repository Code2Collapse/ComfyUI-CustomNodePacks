# FILE: nodes/roto.py
# FEATURE: F1 — VectorRotoMEC (cubic Bezier vector roto with Catmull-Rom keyframe interp)
# INTEGRATES WITH: any MASK consumer (composite, inpaint, keyer)
"""
Vector roto.

Data model (passed in as JSON string from the JS overlay widget):

    {
      "canvas": {"w": 1920, "h": 1080},
      "frames": [
        {
          "frame": 0,
          "splines": [
            [
              {"x":  100.0, "y":  200.0,
               "out": [120.0, 200.0],
               "in":  [ 80.0, 200.0]},
              ...
            ],
            ...next spline...
          ]
        },
        ...
      ]
    }

Each control point carries explicit `in` / `out` handle positions (cubic
Bezier). Curve evaluation is de Casteljau, NOT polyline.

Sparse keyframes are interpolated with Catmull-Rom on every numeric field
(point xy, in-handle xy, out-handle xy) so handles tween smoothly.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

import torch


# ---------- de Casteljau cubic Bezier ---------------------------------
def _bezier_segment(p0: Tuple[float, float], p1: Tuple[float, float],
                    p2: Tuple[float, float], p3: Tuple[float, float],
                    samples: int) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for i in range(samples + 1):
        t = i / samples
        u = 1.0 - t
        # Three lerps -> two -> one (de Casteljau).
        a0 = (u * p0[0] + t * p1[0], u * p0[1] + t * p1[1])
        a1 = (u * p1[0] + t * p2[0], u * p1[1] + t * p2[1])
        a2 = (u * p2[0] + t * p3[0], u * p2[1] + t * p3[1])
        b0 = (u * a0[0] + t * a1[0], u * a0[1] + t * a1[1])
        b1 = (u * a1[0] + t * a2[0], u * a1[1] + t * a2[1])
        c = (u * b0[0] + t * b1[0], u * b0[1] + t * b1[1])
        out.append(c)
    return out


def _evaluate_spline(spline: List[Dict], samples_per_seg: int) -> List[Tuple[float, float]]:
    """Closed cubic Bezier spline -> dense polyline."""
    n = len(spline)
    if n < 2:
        return [(p["x"], p["y"]) for p in spline]
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        a = spline[i]
        b = spline[(i + 1) % n]
        seg = _bezier_segment(
            (a["x"], a["y"]),
            (a["out"][0], a["out"][1]),
            (b["in"][0], b["in"][1]),
            (b["x"], b["y"]),
            samples_per_seg,
        )
        # Drop last sample so successive segments don't double-emit the joint.
        pts.extend(seg[:-1])
    return pts


# ---------- scanline polygon fill -------------------------------------
def _rasterize_polygon(polyline: List[Tuple[float, float]],
                       H: int, W: int, device: torch.device) -> torch.Tensor:
    """Even-odd scanline fill. Returns float mask (H, W) in [0, 1]."""
    if len(polyline) < 3:
        return torch.zeros(H, W, device=device)
    mask = torch.zeros(H, W, device=device, dtype=torch.float32)
    pts = polyline + [polyline[0]]  # close
    edges = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 == y1:
            continue
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        edges.append((y0, y1, x0, (x1 - x0) / (y1 - y0)))
    if not edges:
        return mask

    y_min = max(0, int(math.floor(min(e[0] for e in edges))))
    y_max = min(H - 1, int(math.ceil(max(e[1] for e in edges))))
    for y in range(y_min, y_max + 1):
        ys = y + 0.5
        xs = sorted(x0 + (ys - y0) * slope for y0, y1, x0, slope in edges if y0 <= ys < y1)
        for i in range(0, len(xs) - 1, 2):
            xa = max(0, int(math.ceil(xs[i] - 0.5)))
            xb = min(W - 1, int(math.floor(xs[i + 1] - 0.5)))
            if xb >= xa:
                mask[y, xa:xb + 1] = 1.0
    return mask


# ---------- Catmull-Rom keyframe interpolation ------------------------
def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _interp_point(kfs: List[Dict], frame: int, key: str, idx: int) -> Tuple[float, float]:
    """Catmull-Rom interpolate a single (x,y) field across sparse keyframes."""
    kfs = sorted(kfs, key=lambda k: k["frame"])
    frames = [k["frame"] for k in kfs]
    if frame <= frames[0]:
        p = kfs[0]["splines"][0][idx][key] if key in ("in", "out") else \
            (kfs[0]["splines"][0][idx]["x"], kfs[0]["splines"][0][idx]["y"])
        return tuple(p) if isinstance(p, list) else p
    if frame >= frames[-1]:
        p = kfs[-1]["splines"][0][idx][key] if key in ("in", "out") else \
            (kfs[-1]["splines"][0][idx]["x"], kfs[-1]["splines"][0][idx]["y"])
        return tuple(p) if isinstance(p, list) else p
    # Find bracketing keyframes.
    lo = max(i for i, f in enumerate(frames) if f <= frame)
    hi = lo + 1
    p1 = kfs[lo]["splines"][0][idx]
    p2 = kfs[hi]["splines"][0][idx]
    p0 = kfs[max(lo - 1, 0)]["splines"][0][idx]
    p3 = kfs[min(hi + 1, len(kfs) - 1)]["splines"][0][idx]
    t = (frame - frames[lo]) / max(frames[hi] - frames[lo], 1)
    if key in ("in", "out"):
        return (
            _catmull_rom(p0[key][0], p1[key][0], p2[key][0], p3[key][0], t),
            _catmull_rom(p0[key][1], p1[key][1], p2[key][1], p3[key][1], t),
        )
    return (
        _catmull_rom(p0["x"], p1["x"], p2["x"], p3["x"], t),
        _catmull_rom(p0["y"], p1["y"], p2["y"], p3["y"], t),
    )


def _evaluate_at_frame(roto: Dict, frame: int) -> List[List[Dict]]:
    """Return list-of-splines (each a list of point dicts) at the given frame."""
    frames_data = roto.get("frames", [])
    if not frames_data:
        return []
    if len(frames_data) == 1:
        return frames_data[0]["splines"]

    # Per-spline / per-point Catmull-Rom across the keyframes that contain it.
    n_splines = max(len(fr["splines"]) for fr in frames_data)
    out: List[List[Dict]] = []
    for s_idx in range(n_splines):
        # Filter keyframes that have this spline index.
        kfs = [{"frame": fr["frame"],
                "splines": [fr["splines"][s_idx] if s_idx < len(fr["splines"]) else []]}
               for fr in frames_data
               if s_idx < len(fr["splines"])]
        if not kfs:
            continue
        n_pts = len(kfs[0]["splines"][0])
        out_spline: List[Dict] = []
        for p_idx in range(n_pts):
            xy = _interp_point(kfs, frame, "xy", p_idx)
            ih = _interp_point(kfs, frame, "in", p_idx)
            oh = _interp_point(kfs, frame, "out", p_idx)
            out_spline.append({"x": xy[0], "y": xy[1],
                               "in": [ih[0], ih[1]], "out": [oh[0], oh[1]]})
        out.append(out_spline)
    return out


# ---------- Node ------------------------------------------------------
class VectorRotoMEC:
    DESCRIPTION = ("Cubic-Bezier vector roto with Catmull-Rom keyframe tweening. "
                   "Hands evaluated polylines off to a scanline rasteriser.")
    CATEGORY = "MaskEditControl/Roto"
    FUNCTION = "rasterize"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        empty = json.dumps({"canvas": {"w": 1024, "h": 1024},
                            "frames": [{"frame": 0, "splines": []}]})
        return {
            "required": {
                "roto_json":       ("STRING", {"multiline": True, "default": empty}),
                "frame_count":     ("INT", {"default": 1, "min": 1, "max": 4096}),
                "width":           ("INT", {"default": 1024, "min": 16, "max": 8192}),
                "height":          ("INT", {"default": 1024, "min": 16, "max": 8192}),
                "samples_per_seg": ("INT", {"default": 24, "min": 2, "max": 256}),
                "feather_px":      ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0,
                                              "step": 0.5}),
            },
        }

    def rasterize(self, roto_json: str, frame_count: int, width: int, height: int,
                  samples_per_seg: int, feather_px: float):
        roto = json.loads(roto_json)
        canvas = roto.get("canvas", {"w": width, "h": height})
        sx = width / max(canvas.get("w", width), 1)
        sy = height / max(canvas.get("h", height), 1)

        device = torch.device("cpu")  # rasterise on CPU; user composites later
        masks = torch.zeros(frame_count, height, width, dtype=torch.float32, device=device)

        n_splines_total = 0
        for f in range(frame_count):
            splines = _evaluate_at_frame(roto, f)
            for spline in splines:
                if len(spline) < 2:
                    continue
                scaled = [{"x": p["x"] * sx, "y": p["y"] * sy,
                           "in": [p["in"][0] * sx, p["in"][1] * sy],
                           "out": [p["out"][0] * sx, p["out"][1] * sy]}
                          for p in spline]
                poly = _evaluate_spline(scaled, samples_per_seg)
                masks[f] = torch.maximum(
                    masks[f], _rasterize_polygon(poly, height, width, device),
                )
                n_splines_total += 1

        if feather_px > 0:
            sigma = feather_px / 2.0
            radius = max(1, int(math.ceil(sigma * 3)))
            ks = 2 * radius + 1
            xs = torch.arange(ks, dtype=torch.float32) - radius
            g = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
            g = (g / g.sum()).view(1, 1, ks)
            blurred = masks.unsqueeze(1)
            blurred = torch.nn.functional.conv1d(
                blurred.view(-1, 1, width), g, padding=radius,
            ).view(frame_count, 1, height, width)
            blurred = torch.nn.functional.conv1d(
                blurred.permute(0, 1, 3, 2).contiguous().view(-1, 1, height),
                g, padding=radius,
            ).view(frame_count, 1, width, height).permute(0, 1, 3, 2).contiguous()
            masks = blurred.squeeze(1).clamp(0.0, 1.0)

        info = (f"frames={frame_count} splines_total={n_splines_total} "
                f"size={width}x{height} feather={feather_px}px")
        return (masks, info)


NODE_CLASS_MAPPINGS = {"VectorRotoMEC": VectorRotoMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"VectorRotoMEC": "Vector Roto — Bezier (MEC)"}
