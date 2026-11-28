"""L9.172 (A25) part B: the new Inpaint Crop Pro / Inpaint Stitch Pro features, kept out of inpaint_suite.py.

Inpaint Crop Pro
  * classic fills ``telea`` / ``navier_stokes`` for fill_masked_area (OpenCV cv2.inpaint; audit L7.92 fix 4, G5)
  * multi-region crop planning: independent mask regions with stable region ids (A25 spec 5.5 item 18.1)
  * context-aware crop sizing: crop edges moved off nearby detail, and a megapixel cap on the target size
    (A25 spec 5.5 item 18.2)
Inpaint Stitch Pro
  * registration-assisted reconstruction: shift (and scale) of the generated crop measured on the context ring and
    undone before compositing (A25 spec 5.5 item 19.1)
  * temporal seam QC: per-frame seam error along the blend ring, frames that jump flagged, optional local repair
    (A25 spec 5.5 item 19.2)

Written for this pack (Apache-2.0); nothing ported. CPU / GPU agnostic torch, OpenCV only where named.
No image-quality claim is made for any of these: the tests are synthetic unit tests.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:  # pragma: no cover - opencv is a declared dependency of the pack
    cv2 = None
    np = None
    HAS_CV2 = False

REGION_MODES = ["single (as before)", "per region"]
REGISTRATION_MODES = ["off", "translation", "translation + scale"]
CLASSIC_FILLS = ("telea", "navier_stokes")


def _luma(x: torch.Tensor) -> torch.Tensor:
    """[..., C] image -> [...] luma (Rec.709 weights on the first three channels; one channel passes through)."""
    if x.shape[-1] >= 3:
        return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    return x[..., 0]


# ── Crop: classic fills ────────────────────────────────────────────────────────────────────────────────────────────

def classic_fill(images: torch.Tensor, masks: torch.Tensor, method: str, radius: int = 3) -> torch.Tensor:
    """Fill the masked pixels of each crop with OpenCV's Telea (fast marching) or Navier-Stokes inpaint.

    images [B,H,W,C] 0..1, masks [B,H,W]; pixels with mask > 0.5 are filled, every other pixel is returned unchanged
    (bit-exact). An alpha channel is kept as it was. Raises when OpenCV is missing (never a silent no-op)."""
    if method not in CLASSIC_FILLS:
        raise ValueError(f"Inpaint Crop Pro: unknown classic fill {method!r} (use one of {CLASSIC_FILLS}).")
    if not HAS_CV2:
        raise RuntimeError("Inpaint Crop Pro: fill_masked_area 'telea' / 'navier_stokes' needs OpenCV "
                           "(opencv-python, a dependency of this pack). Install it or pick another fill.")
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    out = images.clone()
    hole = masks > 0.5
    for i in range(int(images.shape[0])):
        h = hole[min(i, hole.shape[0] - 1)]
        if not bool(h.any()):
            continue
        rgb = images[i, ..., :3].detach().float().clamp(0, 1).cpu().numpy()
        u8 = np.ascontiguousarray((rgb * 255.0 + 0.5).astype(np.uint8))
        hm = np.ascontiguousarray(h.cpu().numpy().astype(np.uint8) * 255)
        filled = cv2.inpaint(u8, hm, int(radius), flag)
        if filled.ndim == 2:
            filled = filled[..., None]
        f = torch.from_numpy(filled.astype(np.float32) / 255.0).to(out.device, out.dtype)
        sel = h.to(out.device).unsqueeze(-1)
        out[i, ..., :3] = torch.where(sel, f[..., :3], out[i, ..., :3])
    return out


# ── Crop: multi-region planning ────────────────────────────────────────────────────────────────────────────────────

def _fit_bhw(mask: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] == 1 and B > 1:
        m = m.expand(B, -1, -1)
    elif m.shape[0] != B:
        m = m[:1].expand(B, -1, -1) if m.shape[0] < B else m[:B]
    if tuple(m.shape[-2:]) != (H, W):
        m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
    return m.clamp(0.0, 1.0)


def split_regions(mask: torch.Tensor, B: int, H: int, W: int, *, preserve_polarity: bool, invert: bool,
                  min_area: int, max_regions: int) -> Tuple[List[torch.Tensor], List[dict]]:
    """Independent regions of the area to regenerate, with stable ids.

    The regions are the 8-connected components of the regenerate area over the WHOLE clip (union over frames), so a
    region keeps its id on every frame; components smaller than ``min_area`` pixels are dropped, the ``max_regions``
    largest are kept and numbered top-to-bottom, left-to-right. Returns per region the mask to hand to the crop - in
    the node's own mask convention (polarity), with mask_invert already applied - and a record {id, bbox, area}.
    """
    if not HAS_CV2:
        raise RuntimeError("Inpaint Crop Pro: region_mode 'per region' needs OpenCV (opencv-python) to find the "
                           "mask regions.")
    m = _fit_bhw(mask, B, H, W)
    regen = 1.0 - m if preserve_polarity else m
    if invert:
        regen = 1.0 - regen
    union = (regen > 0.5).any(dim=0).cpu().numpy().astype(np.uint8)
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(union, connectivity=8)
    comps = [(int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n)
             if int(stats[k, cv2.CC_STAT_AREA]) >= max(1, int(min_area))]
    comps.sort(key=lambda t: -t[0])
    comps = comps[:max(1, int(max_regions))]
    comps.sort(key=lambda t: (int(stats[t[1], cv2.CC_STAT_TOP]), int(stats[t[1], cv2.CC_STAT_LEFT])))
    lab = torch.from_numpy(labels.astype(np.int64)).to(m.device)
    masks, records = [], []
    for rid, (area, k) in enumerate(comps):
        foot = (lab == k).float()[None, None]
        # 3 px of slack so the soft edge just outside the binary footprint stays with its region
        foot = F.max_pool2d(foot, kernel_size=7, stride=1, padding=3)[0, 0]
        r = regen * foot.unsqueeze(0)
        if invert:
            r = 1.0 - r
        if preserve_polarity:
            r = 1.0 - r
        masks.append(r.clamp(0.0, 1.0))
        records.append({"id": rid, "bbox": [int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP]),
                                            int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])],
                        "area": area})
    return masks, records


def merge_region_stitchers(stitchers: List[dict], frames: int) -> dict:
    """One stitcher for every (region, frame) crop, region-major, with the ids Inpaint Stitch Pro groups by."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in stitchers[0].items()}
    for st in stitchers[1:]:
        for k, v in st.items():
            if isinstance(v, list):
                out[k].extend(v)
    out["region_ids"] = [r for r in range(len(stitchers)) for _ in range(frames)]
    out["region_frame"] = [f for _ in range(len(stitchers)) for f in range(frames)]
    out["region_count"] = len(stitchers)
    out["region_frames"] = int(frames)
    return out


# ── Crop: context-aware sizing ─────────────────────────────────────────────────────────────────────────────────────

_DETAIL_MAX_SIDE = 512


def detail_map(image: torch.Tensor) -> torch.Tensor:
    """Mean luma gradient magnitude over the clip [h,w] (cpu float32, at most 512 px on the long side).

    Averaged over every frame, so one map serves a stable crop; the crop reads it in normalised coordinates."""
    img = image.detach().float()
    if img.ndim == 3:
        img = img.unsqueeze(0)
    y = _luma(img).unsqueeze(1)                                   # [B,1,H,W]
    H, W = int(y.shape[-2]), int(y.shape[-1])
    s = min(1.0, _DETAIL_MAX_SIDE / float(max(H, W)))
    if s < 1.0:
        y = F.interpolate(y, size=(max(1, round(H * s)), max(1, round(W * s))), mode="area")
    gx = torch.zeros_like(y)
    gy = torch.zeros_like(y)
    gx[..., :, 1:-1] = (y[..., :, 2:] - y[..., :, :-2]) * 0.5
    gy[..., 1:-1, :] = (y[..., 2:, :] - y[..., :-2, :]) * 0.5
    return torch.sqrt(gx * gx + gy * gy).mean(dim=0)[0].cpu()


def detail_in_processed_space(dmap: torch.Tensor, pre_hw: Tuple[int, int],
                              pads: Optional[Tuple[int, int, int, int]]) -> torch.Tensor:
    """The detail map at the crop's working size: resized to the (pre-resized) image, zero over outpaint padding."""
    Hp, Wp = int(pre_hw[0]), int(pre_hw[1])
    d = F.interpolate(dmap.float()[None, None], size=(Hp, Wp), mode="bilinear", align_corners=False)[0, 0]
    if pads:
        up, down, left, right = (int(v) for v in pads)
        d = F.pad(d[None, None], (left, right, up, down), mode="constant", value=0.0)[0, 0]
    return d


def _best_edge(costs: torch.Tensor, thr: float) -> int:
    """Index into ``costs`` (index 0 = the current edge, rising = further out) of the edge to use."""
    if costs.numel() == 0 or float(costs[0]) <= thr:
        return 0
    n = costs.numel()
    pen = 0.1 * thr * torch.arange(n, dtype=costs.dtype) / max(1, n - 1)
    return int(torch.argmin(costs + pen))


def detail_aware_box(x: int, y: int, w: int, h: int, d: torch.Tensor, max_grow: float) -> Tuple[int, int, int, int]:
    """Move each crop edge OUTWARD (never inward, so the mask stays inside) by up to ``max_grow`` x the box size to
    where the edge line crosses the least detail, when the current edge cuts through more detail than the image's
    median. ``d`` is the detail map in the crop's working coordinates [H,W]."""
    H, W = int(d.shape[0]), int(d.shape[1])
    if w <= 0 or h <= 0 or max_grow <= 0:
        return x, y, w, h
    thr = float(d.median())
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return x, y, w, h
    gw, gh = int(round(max_grow * w)), int(round(max_grow * h))
    rows, cols = d[y0:y1], d[:, x0:x1]
    # left: columns x0, x0-1, ...; right: x1-1, x1, ...; top / bottom alike
    lc = torch.stack([rows[:, c].mean() for c in range(x0, max(-1, x0 - gw - 1), -1)]) if x0 >= 0 else torch.zeros(0)
    rc = torch.stack([rows[:, c].mean() for c in range(x1 - 1, min(W, x1 + gw))])
    tc = torch.stack([cols[r].mean() for r in range(y0, max(-1, y0 - gh - 1), -1)])
    bc = torch.stack([cols[r].mean() for r in range(y1 - 1, min(H, y1 + gh))])
    nx0 = x0 - _best_edge(lc, thr)
    nx1 = x1 + _best_edge(rc, thr)
    ny0 = y0 - _best_edge(tc, thr)
    ny1 = y1 + _best_edge(bc, thr)
    # keep any part of the box that lay outside the image (outpaint) as it was
    nx0 = min(nx0, x) if x < 0 else nx0
    ny0 = min(ny0, y) if y < 0 else ny0
    nx1 = max(nx1, x + w) if x + w > W else nx1
    ny1 = max(ny1, y + h) if y + h > H else ny1
    return nx0, ny0, nx1 - nx0, ny1 - ny0


def cap_megapixels(tw: int, th: int, cap_mp: float, multiple: int = 8) -> Tuple[int, int]:
    """The target size scaled down (aspect kept) so that, once the crop rounds it UP to a multiple of ``multiple``
    (output_padding / wan_align_multiple), w*h <= cap_mp * 1e6. Never below one multiple (and 64 px)."""
    m = max(1, int(multiple))
    up = lambda v: -(-int(v) // m) * m  # noqa: E731 - the crop's own rounding (_pad_to_multiple)
    if cap_mp <= 0 or up(tw) * up(th) <= cap_mp * 1e6:
        return int(tw), int(th)
    s = math.sqrt(cap_mp * 1e6 / float(up(tw) * up(th)))
    floor = up(64)
    nw = max(floor, int(tw * s) // m * m)
    nh = max(floor, int(th * s) // m * m)
    while nw * nh > cap_mp * 1e6 and (nw > floor or nh > floor):   # rounding can still overshoot by a step
        if nw >= nh and nw > floor:
            nw -= m
        else:
            nh -= m
    return nw, nh


# ── Stitch: registration ───────────────────────────────────────────────────────────────────────────────────────────

def _hann2d(h: int, w: int, device, dtype) -> torch.Tensor:
    return torch.hann_window(h, periodic=False, device=device, dtype=dtype)[:, None] * \
        torch.hann_window(w, periodic=False, device=device, dtype=dtype)[None, :]


def _phase_shift(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    """(dy, dx, peak): the shift that, applied to ``b`` (out(p) = b(p - d)), lines it up with ``a``. Sub-pixel by a
    parabola through the peak and its neighbours. a, b: [h,w] float."""
    h, w = a.shape
    win = _hann2d(h, w, a.device, a.dtype)
    fa = torch.fft.rfft2((a - a.mean()) * win)
    fb = torch.fft.rfft2((b - b.mean()) * win)
    cps = fa * torch.conj(fb)
    cps = cps / cps.abs().clamp(min=1e-9)
    r = torch.fft.irfft2(cps, s=(h, w))
    idx = int(torch.argmax(r))
    py, px = divmod(idx, w)
    peak = float(r[py, px])

    def sub(cm, c0, cp):
        den = cm - 2 * c0 + cp
        return 0.0 if abs(den) < 1e-12 else 0.5 * (cm - cp) / den

    oy = sub(float(r[(py - 1) % h, px]), peak, float(r[(py + 1) % h, px]))
    ox = sub(float(r[py, (px - 1) % w]), peak, float(r[py, (px + 1) % w]))
    dy = py + oy
    dx = px + ox
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dy, dx, peak


def _warp(img: torch.Tensor, scale: float, dy: float, dx: float) -> torch.Tensor:
    """img [1,H,W,C] -> out(p) = img(c + (p - d - c) / scale): scaled about the centre, then shifted by d (pixels).
    Border pixels repeat."""
    _, H, W, _ = img.shape
    dev, dt = img.device, torch.float32
    ys = torch.arange(H, device=dev, dtype=dt)
    xs = torch.arange(W, device=dev, dtype=dt)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    sy = cy + (ys - dy - cy) / scale
    sx = cx + (xs - dx - cx) / scale
    gy = (2.0 * sy / max(H - 1, 1)) - 1.0
    gx = (2.0 * sx / max(W - 1, 1)) - 1.0
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1).unsqueeze(0)
    out = F.grid_sample(img.permute(0, 3, 1, 2).float(), grid, mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out.permute(0, 2, 3, 1).to(img.dtype)


def _ncc(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> float:
    s = float(w.sum())
    if s < 1.0:
        return -1.0
    ma, mb = float((a * w).sum()) / s, float((b * w).sum()) / s
    da, db = (a - ma) * w, (b - mb) * w
    den = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    return float((da * db).sum()) / den if den > 1e-12 else -1.0


def register_crop(inp: torch.Tensor, ref: torch.Tensor, msk: torch.Tensor, mode: str,
                  max_shift: float) -> Tuple[torch.Tensor, dict]:
    """Line the generated crop ``inp`` up with the plate crop ``ref`` before compositing.

    Measured only on the CONTEXT ring (blend weight < 0.05: pixels the model was asked to keep), where the two should
    agree; the regenerated area is flattened out of the measurement. ``mode`` "translation" or "translation +
    scale" (0.94-1.06 in 0.01 steps). The correction is applied only when it is within ``max_shift`` pixels and
    raises the context correlation; otherwise ``inp`` comes back unchanged and the record says why.
    inp, ref [1,H,W,C]; msk [1,H,W]."""
    rec: Dict[str, Any] = {"applied": False}
    ctx = (msk[0] < 0.05).float()
    frac = float(ctx.mean())
    if frac < 0.05:
        rec["reason"] = "no context ring to register on (mask covers the crop)"
        return inp, rec
    a = _luma(ref[0].float())
    b = _luma(inp[0].float())
    fill = float((a * ctx).sum() / ctx.sum())
    a_m = a * ctx + fill * (1 - ctx)
    before = _ncc(a, b, ctx)
    scales = [1.0] if mode == "translation" else [round(0.94 + 0.01 * k, 2) for k in range(13)]
    best = (before, 1.0, 0.0, 0.0)
    for s in scales:
        bs = b if s == 1.0 else _warp(inp[:1], s, 0.0, 0.0)[0].float()
        bs = _luma(bs) if bs.ndim == 3 else bs
        b_m = bs * ctx + fill * (1 - ctx)
        dy, dx, _peak = _phase_shift(a_m, b_m)
        if abs(dy) > max_shift or abs(dx) > max_shift:
            continue
        cand = _luma(_warp(inp[:1], s, dy, dx)[0].float())
        score = _ncc(a, cand, ctx)
        if score > best[0] + 1e-4:
            best = (score, s, dy, dx)
    rec.update({"ncc_before": round(before, 5), "ncc_after": round(best[0], 5), "scale": best[1],
                "shift_y": round(best[2], 3), "shift_x": round(best[3], 3)})
    if best[1] == 1.0 and best[2] == 0.0 and best[3] == 0.0:
        rec["reason"] = "no shift / scale within the limit improved the context match"
        return inp, rec
    rec["applied"] = True
    return _warp(inp, best[1], best[2], best[3]), rec


# ── Stitch: temporal seam QC ───────────────────────────────────────────────────────────────────────────────────────

def seam_ring(blend: torch.Tensor, band: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """(ring [B,H,W] bool, repair weight [B,H,W]) of a blend mask: the soft transition (0.02 < w < 0.98) plus
    ``band`` px on both sides of the 0.5 contour, so a hard-edged blend still has a seam to measure. The weight peaks
    at w = 0.5 and is 1 on the contour band."""
    b = blend.float()
    hard = (b > 0.5).float().unsqueeze(1)
    k = 2 * band + 1
    dil = F.max_pool2d(hard, k, 1, band)
    ero = 1.0 - F.max_pool2d(1.0 - hard, k, 1, band)
    contour = ((dil - ero) > 0)[:, 0]
    soft = (b > 0.02) & (b < 0.98)
    ring = soft | contour
    wgt = torch.maximum((1.0 - (2.0 * b - 1.0).abs()).clamp(0, 1), contour.float()) * ring.float()
    return ring, wgt


def seam_qc(out: torch.Tensor, orig: torch.Tensor, blend: torch.Tensor, threshold: float
            ) -> Tuple[torch.Tensor, dict]:
    """Per-frame seam error along the blend ring and frame-to-frame jumps, with optional local repair.

    out, orig [B,H,W,C] (stitched result and the plate, same size); blend [B,H,W] the blend weight in the plate's
    frame (regenerate = 1). The ring is seam_ring() (soft transition + 2 px around the 0.5
    contour). Seam error = mean |luma(out) - luma(orig)| on the ring.
    A frame is flagged when its error differs from the clip's median by more than ``threshold`` (image units; 0 =
    report only). A flagged frame is repaired on its ring with a per-channel OFFSET: the frame's mean seam residual
    (stitched - plate, on the ring) is replaced by the mean residual of the nearest unflagged frame before and after
    (weighted by how deep in the ring a pixel is), so a seam that pops for one frame follows its neighbours' level.
    Only this frame's own pixels are kept - the neighbours' PIXELS are never copied in (audit 2026-11-27: copying
    them left the neighbours' moving content as a double exposure on the ring). Nothing outside the ring is
    touched."""
    B = int(out.shape[0])
    ring, wgt_all = seam_ring(blend)
    lo, lp = _luma(out.float()), _luma(orig.float())
    errs = []
    for i in range(B):
        r = ring[i]
        errs.append(float((lo[i] - lp[i]).abs()[r].mean()) if bool(r.any()) else 0.0)
    e = torch.tensor(errs)
    med = float(e.median()) if B else 0.0
    frames = [{"frame": i, "seam_error": round(errs[i], 6), "deviation": round(abs(errs[i] - med), 6),
               "ring_pixels": int(ring[i].sum()), "flagged": False, "repaired": False} for i in range(B)]
    rep = {"frames": frames, "median_seam_error": round(med, 6), "threshold": float(threshold), "flagged": [],
           "repaired": []}
    if threshold <= 0 or B < 2:
        if threshold > 0:
            rep["note"] = "one frame: nothing to compare against"
        return out, rep
    flagged = [i for i in range(B) if abs(errs[i] - med) > threshold and frames[i]["ring_pixels"] > 0]
    for i in flagged:
        frames[i]["flagged"] = True
    rep["flagged"] = flagged
    good = [i for i in range(B) if i not in flagged]
    if not good:
        rep["note"] = "every frame flagged: no reference frame to repair from"
        return out, rep
    res = out.clone()
    for i in flagged:
        prev = [g for g in good if g < i]
        nxt = [g for g in good if g > i]
        refs = ([prev[-1]] if prev else []) + ([nxt[0]] if nxt else [])
        r = ring[i]

        bw = blend[i].float().clamp(0, 1)

        def _level(f):
            # the stitched-minus-plate residual on the ring grows with the blend weight (0 outside, full inside), so
            # it is fitted as blend * level per channel (least squares), not as one flat offset
            d = out[f].float() - orig[f].float()
            if not bool(r.any()):
                return torch.zeros(d.shape[-1])
            w = blend[f].float().clamp(0, 1)[r].unsqueeze(-1)
            return (d[r] * w).sum(0) / (w * w).sum(0).clamp(min=1e-6)
        shift = torch.stack([_level(g) for g in refs]).mean(0) - _level(i)       # [C]
        wgt = (bw * r.float()).unsqueeze(-1)
        res[i] = (out[i].float() + wgt * shift.view(1, 1, -1)).to(out.dtype)
        frames[i]["repaired"] = True
        frames[i]["repaired_from"] = refs
        frames[i]["seam_error_after"] = round(float((_luma(res[i].float()) - lp[i]).abs()[ring[i]].mean()), 6)
    rep["repaired"] = flagged
    return res, rep


def report_text(rep: dict) -> str:
    return json.dumps(rep, separators=(",", ":"))
