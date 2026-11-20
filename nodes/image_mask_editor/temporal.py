"""Keyframed transforms of a drawn shape over frames (L9.172, A25 spec 5.8 item 43.2: temporal vector editing).

`spline_keyframes` is JSON: a list of keys, or {"keys": [...], "frames": N, "pivot": [x, y]}. A key is
{"frame": int, "x": px, "y": px, "scale": s, "rotation": degrees}; missing fields are 0 (x, y, rotation) and 1 (scale).
Between two keys every field is interpolated (linear, or smoothstep ease in / out); before the first key the first key
holds, after the last the last. The transform is applied to the rasterised shape about the pivot (default: the
shape's own centroid on frame 0): out(p) = shape(R(-rot) (p - pivot - (x, y)) / scale + pivot).
"""
from __future__ import annotations

import json
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

EASINGS = ["linear", "smooth"]
_FIELDS = (("x", 0.0), ("y", 0.0), ("scale", 1.0), ("rotation", 0.0))
_MAX_FRAMES = 100_000


def _err(msg: str) -> ValueError:
    return ValueError(f"Image Mask Editor: spline_keyframes {msg}")


def parse(text: str) -> Tuple[List[dict], Optional[int], Optional[Tuple[float, float]]]:
    """(keys sorted by frame, frame count or None, pivot or None). Raises ValueError with a plain message."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise _err(f"is not valid JSON ({exc}).") from None
    frames = pivot = None
    if isinstance(data, dict):
        frames = data.get("frames")
        pivot = data.get("pivot")
        data = data.get("keys")
    if not isinstance(data, list) or not data:
        raise _err("needs at least one key: [{\"frame\": 0, \"x\": 0, \"y\": 0, \"scale\": 1, \"rotation\": 0}, ...].")
    keys, seen = [], set()
    for k in data:
        if not isinstance(k, dict) or "frame" not in k:
            raise _err("keys must be objects with a \"frame\".")
        try:
            f = int(k["frame"])
            key = {"frame": f, **{n: float(k.get(n, d)) for n, d in _FIELDS}}
        except (TypeError, ValueError):
            raise _err(f"key {k!r}: frame must be an integer and x / y / scale / rotation numbers.") from None
        if f < 0 or f > _MAX_FRAMES:
            raise _err(f"frame {f} is out of range (0..{_MAX_FRAMES}).")
        if f in seen:
            raise _err(f"has two keys on frame {f}.")
        if not (key["scale"] > 0) or not all(math.isfinite(key[n]) for n, _ in _FIELDS):
            raise _err(f"key on frame {f}: scale must be > 0 and every value finite.")
        seen.add(f)
        keys.append(key)
    keys.sort(key=lambda k: k["frame"])
    if frames is not None:
        try:
            frames = int(frames)
        except (TypeError, ValueError):
            raise _err("\"frames\" must be an integer.") from None
        if frames < 1 or frames > _MAX_FRAMES:
            raise _err(f"\"frames\" must be 1..{_MAX_FRAMES}.")
    if pivot is not None:
        if not (isinstance(pivot, (list, tuple)) and len(pivot) == 2):
            raise _err("\"pivot\" must be [x, y].")
        pivot = (float(pivot[0]), float(pivot[1]))
    return keys, frames, pivot


def value_at(keys: List[dict], frame: int, easing: str) -> dict:
    """The interpolated transform on ``frame`` (held outside the keys)."""
    if frame <= keys[0]["frame"]:
        return dict(keys[0])
    if frame >= keys[-1]["frame"]:
        return dict(keys[-1])
    for a, b in zip(keys, keys[1:]):
        if a["frame"] <= frame <= b["frame"]:
            t = (frame - a["frame"]) / float(b["frame"] - a["frame"])
            if easing == "smooth":
                t = t * t * (3.0 - 2.0 * t)
            return {"frame": frame, **{n: a[n] + (b[n] - a[n]) * t for n, _ in _FIELDS}}
    return dict(keys[-1])  # pragma: no cover


def centroid(mask: torch.Tensor) -> Tuple[float, float]:
    h, w = mask.shape[-2:]
    s = float(mask.sum())
    if s <= 1e-8:
        return (w - 1) / 2.0, (h - 1) / 2.0
    ys = torch.arange(h, dtype=torch.float32, device=mask.device)
    xs = torch.arange(w, dtype=torch.float32, device=mask.device)
    return float((mask.sum(0) * xs).sum()) / s, float((mask.sum(1) * ys).sum()) / s


def transform(mask: torch.Tensor, t: dict, pivot: Tuple[float, float]) -> torch.Tensor:
    """mask [H,W] moved / scaled / rotated by key ``t`` about ``pivot`` (bilinear, zero outside)."""
    if t["x"] == 0.0 and t["y"] == 0.0 and t["scale"] == 1.0 and t["rotation"] == 0.0:
        return mask.clone()
    h, w = mask.shape
    dev = mask.device
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=dev),
                            torch.arange(w, dtype=torch.float32, device=dev), indexing="ij")
    px, py = pivot
    dx, dy = xs - px - t["x"], ys - py - t["y"]
    a = math.radians(t["rotation"])
    ca, sa = math.cos(a), math.sin(a)
    sx = (ca * dx + sa * dy) / t["scale"] + px            # inverse rotation, then inverse scale
    sy = (-sa * dx + ca * dy) / t["scale"] + py
    grid = torch.stack([2.0 * sx / max(w - 1, 1) - 1.0, 2.0 * sy / max(h - 1, 1) - 1.0], dim=-1).unsqueeze(0)
    out = F.grid_sample(mask.float()[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0, 0].clamp(0.0, 1.0).to(mask.dtype)


def animate(mask: torch.Tensor, text: str, easing: str, batch: Optional[int]) -> Tuple[torch.Tensor, str]:
    """The shape (frame 0 of ``mask`` [B,H,W]) keyed over the clip: [N,H,W] and an info note. N = the image batch
    when one is connected, else "frames" from the JSON, else the last key + 1."""
    if easing not in EASINGS:
        raise _err(f"easing {easing!r} is unknown (use one of {EASINGS}).")
    keys, frames, pivot = parse(text)
    n = int(batch) if batch else (frames or keys[-1]["frame"] + 1)
    src = mask[0]
    pv = pivot or centroid(src)
    out = torch.stack([transform(src, value_at(keys, i, easing), pv) for i in range(n)], dim=0)
    note = (f"keyframes: {len(keys)} key(s) over {n} frame(s), {easing}, pivot ({pv[0]:.1f}, {pv[1]:.1f})")
    return out, note
