"""QC numbers for the Matte viewer's Result tab (L7.114, owner A14).

Three numbers, each computed on frames scaled to at most QC_SIDE px on the long side (the thresholds were measured
at that scale):

* coverage    - share of the selection (the segmenter's coarse mask, > 0.5) that the final alpha covers (> 0.5).
                Low = the matte lost part of the subject.
* edge detail - gradient energy of the alpha inside the edge band, divided by the coarse mask's. The band is the
                coarse mask dilated minus eroded by BAND_FRACTION of the long side. A binary coarse mask has the
                sharpest possible edge, so a clean soft matte of a hard edge lands below 1; strands of hair (edges the
                coarse mask never had) push it up; a smeared matte drops toward 0.
* flicker     - motion-compensated alpha change inside the edge band: the previous frame's alpha is warped onto the
                current frame with optical flow measured on the plate (OpenCV Farneback), then the mean absolute
                difference is taken in the band. Units: alpha (0..1). Needs 2+ consecutive frames.

THRESHOLDS come from measurements on synthetic clips with a known true alpha
(tests/test_viewer_qc.py rebuilds the clips and checks every case still lands in the level written here; the table of
measured values is in docs/design/matte_viewer.md, "QC thresholds"). They are not claims about real footage.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

QC_SIDE = 640
BAND_FRACTION = 0.01
#: Pairs of consecutive frames the flicker number is averaged over on a long clip (spread evenly).
MAX_PAIRS = 48

#: level boundaries: "higher" metrics are green at >= green, amber at >= amber, else red; "lower" the reverse.
#: Chosen between the measured synthetic cases (docs/design/matte_viewer.md section 4).
THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "coverage": {"better": "higher", "green": 0.97, "amber": 0.90, "unit": "share"},
    "edge_detail": {"better": "higher", "green": 0.30, "amber": 0.10, "unit": "ratio"},
    "flicker": {"better": "lower", "green": 0.025, "amber": 0.040, "unit": "alpha"},
}


def level(name: str, value: Optional[float]) -> str:
    """green / amber / red for a number, n/a when there is none."""
    if value is None or not np.isfinite(value):
        return "n/a"
    t = THRESHOLDS[name]
    if t["better"] == "higher":
        return "green" if value >= t["green"] else "amber" if value >= t["amber"] else "red"
    return "green" if value <= t["green"] else "amber" if value <= t["amber"] else "red"


def _to_bhw(m: torch.Tensor) -> torch.Tensor:
    m = m.detach().float().cpu()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim == 4:
        m = m[..., 0] if m.shape[-1] in (1, 3, 4) else m[:, 0]
    return m.clamp(0, 1)


def _scale(h: int, w: int, side: int = QC_SIDE):
    s = min(1.0, side / float(max(h, w)))
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


def _resize_bhw(m: torch.Tensor, hw) -> torch.Tensor:
    if tuple(m.shape[-2:]) == tuple(hw):
        return m
    return F.interpolate(m.unsqueeze(1), size=hw, mode="area")[:, 0]


def _resize_image(img: torch.Tensor, hw) -> torch.Tensor:
    x = img.detach().float().cpu()[..., :3].permute(0, 3, 1, 2)
    if tuple(x.shape[-2:]) != tuple(hw):
        x = F.interpolate(x, size=hw, mode="area")
    return x.permute(0, 2, 3, 1).clamp(0, 1)


def edge_band(coarse_bhw: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilated minus eroded binary coarse mask (bool, B,H,W)."""
    b = (coarse_bhw > 0.5).float().unsqueeze(1)
    k = 2 * radius + 1
    dil = F.max_pool2d(b, k, stride=1, padding=radius)
    ero = 1.0 - F.max_pool2d(1.0 - b, k, stride=1, padding=radius)
    return (dil - ero)[:, 0] > 0.5


def _grad_energy(m: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(m)
    gy = torch.zeros_like(m)
    gx[..., :, :-1] = m[..., :, 1:] - m[..., :, :-1]
    gy[..., :-1, :] = m[..., 1:, :] - m[..., :-1, :]
    return gx * gx + gy * gy


def coverage(alpha: torch.Tensor, coarse: torch.Tensor) -> Optional[float]:
    sel = coarse > 0.5
    n = int(sel.sum())
    if n == 0:
        return None
    return float(((alpha > 0.5) & sel).sum()) / n


def edge_detail(alpha: torch.Tensor, coarse: torch.Tensor, band: torch.Tensor) -> Optional[float]:
    if not bool(band.any()):
        return None
    ec = float(_grad_energy((coarse > 0.5).float())[band].mean())
    if ec <= 1e-9:
        return None
    return float(_grad_energy(alpha)[band].mean()) / ec


def _flow(prev_gray: np.ndarray, cur_gray: np.ndarray):
    import cv2
    # flow from the current frame back to the previous one: where each current pixel came from
    return cv2.calcOpticalFlowFarneback(cur_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def pairs(n: int, max_pairs: int = MAX_PAIRS) -> List[int]:
    """Indices t (1..n-1) of the consecutive pairs (t-1, t) that are measured: all, or spread evenly."""
    if n < 2:
        return []
    ts = list(range(1, n))
    if len(ts) <= max_pairs:
        return ts
    return sorted({ts[int(round(i * (len(ts) - 1) / (max_pairs - 1)))] for i in range(max_pairs)})


def flicker(plate: torch.Tensor, alpha: torch.Tensor, band: torch.Tensor,
            ts: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Mean motion-compensated alpha change in the edge band, and the worst pair."""
    n = int(alpha.shape[0])
    ts = list(ts) if ts is not None else pairs(n)
    if not ts:
        return {"value": None, "worst": None, "pairs": 0, "flow": "none"}
    try:
        import cv2  # noqa: F401
        have_flow = True
    except Exception:  # noqa: BLE001
        have_flow = False
    gray = (plate[..., :3] * torch.tensor([0.299, 0.587, 0.114])).sum(-1).mul(255).clamp(0, 255)
    gray = gray.to(torch.uint8).numpy()
    a = alpha.numpy().astype(np.float32)
    vals, worst = [], (0.0, None)
    for t in ts:
        bt = band[t].numpy()
        if not bt.any():
            continue
        prev = _warp(a[t - 1], _flow(gray[t - 1], gray[t])) if have_flow else a[t - 1]
        d = float(np.abs(a[t] - prev)[bt].mean())
        vals.append(d)
        if d > worst[0]:
            worst = (d, t)
    if not vals:
        return {"value": None, "worst": None, "pairs": 0, "flow": "farneback" if have_flow else "none"}
    return {"value": float(np.mean(vals)), "worst": {"value": worst[0], "frame": worst[1]}, "pairs": len(vals),
            "flow": "farneback" if have_flow else "none (OpenCV missing: not motion-compensated)"}


def measure(plate: torch.Tensor, alpha: torch.Tensor, coarse: Optional[torch.Tensor] = None,
            ts: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """The three QC numbers with their levels. ``plate`` (B,H,W,3), ``alpha`` / ``coarse`` (B,H,W) of the same B."""
    a = _to_bhw(alpha)
    B, H, W = (int(s) for s in a.shape)
    hw = _scale(H, W)
    a = _resize_bhw(a, hw)
    p = _resize_image(plate[:B], hw)
    radius = max(2, int(round(BAND_FRACTION * max(hw))))
    out: Dict[str, Any] = {"scale": list(hw)[::-1], "band_px": radius, "frames": B}
    if coarse is None or coarse.numel() == 0:
        out["coverage"] = {"value": None, "level": "n/a", "why": "no coarse mask"}
        out["edge_detail"] = {"value": None, "level": "n/a", "why": "no coarse mask"}
        band = edge_band((a > 0.5).float(), radius)
    else:
        c = _resize_bhw(_to_bhw(coarse)[:B], hw)
        if c.shape[0] == 1 and B > 1:
            c = c.expand(B, -1, -1)
        band = edge_band(c, radius)
        cv = coverage(a, c)
        ed = edge_detail(a, c, band)
        out["coverage"] = {"value": cv, "level": level("coverage", cv)}
        out["edge_detail"] = {"value": ed, "level": level("edge_detail", ed)}
    fl = flicker(p, a, band, ts)
    out["flicker"] = {**fl, "level": level("flicker", fl["value"])}
    if fl["value"] is None:
        out["flicker"]["why"] = "one frame" if B < 2 else "no edge in the measured frames"
    return out
