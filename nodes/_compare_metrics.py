"""Per-frame comparison metrics for Video Comparer (C2C) v2 (L9.40, docs/save-video-v2.md): PSNR (RGB), SSIM (on
BT.709 luma, 11-tap Gaussian window, the Wang et al. 2004 constants), and the mean CIEDE2000 colour difference on a
fixed pixel subsample. Float32 throughout; frames above 960 px on the long side are area-downscaled for SSIM only
(PSNR and ΔE use the full frame / the subsample of the full frame), as the design budget says.

Original code (Apache-2.0). CIEDE2000 follows Sharma, Wu & Dalal, "The CIEDE2000 Color-Difference Formula:
Implementation Notes, Supplementary Test Data, and Mathematical Observations" (2005); the test checks their table.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SSIM_MAX_LONG = 960
DE_SAMPLES = 65536
_K1, _K2 = 0.01, 0.03


def _bchw(x: torch.Tensor) -> torch.Tensor:
    return x[..., :3].permute(0, 3, 1, 2).float()


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[N] PSNR in dB of [N, H, W, C] 0-1 frames (inf -> 99 for identical frames)."""
    mse = ((a[..., :3].float() - b[..., :3].float()) ** 2).flatten(1).mean(1)
    out = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
    return torch.where(mse <= 1e-12, torch.full_like(out, 99.0), out)


def _gauss(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def _luma(x: torch.Tensor) -> torch.Tensor:
    return (0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3])


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[N] SSIM on luma of [N, H, W, C] 0-1 frames."""
    ya, yb = _luma(_bchw(a)), _luma(_bchw(b))
    h, w = ya.shape[-2:]
    s = min(1.0, SSIM_MAX_LONG / float(max(h, w)))
    if s < 1.0:
        size = (max(11, int(round(h * s))), max(11, int(round(w * s))))
        ya, yb = F.interpolate(ya, size=size, mode="area"), F.interpolate(yb, size=size, mode="area")
    win = _gauss().to(ya.device)
    mu_a, mu_b = F.conv2d(ya, win), F.conv2d(yb, win)
    sa = F.conv2d(ya * ya, win) - mu_a ** 2
    sb = F.conv2d(yb * yb, win) - mu_b ** 2
    sab = F.conv2d(ya * yb, win) - mu_a * mu_b
    c1, c2 = _K1 ** 2, _K2 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return m.flatten(1).mean(1)


def _srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """[..., 3] display sRGB 0-1 -> CIE L*a*b* (D65)."""
    c = rgb.clamp(0, 1)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=lin.dtype, device=lin.device)
    xyz = lin @ m.T / torch.tensor([0.95047, 1.0, 1.08883], dtype=lin.dtype, device=lin.device)
    d = 6.0 / 29.0
    f = torch.where(xyz > d ** 3, xyz.clamp(min=1e-12) ** (1.0 / 3.0), xyz / (3 * d * d) + 4.0 / 29.0)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 between [..., 3] Lab arrays (kL = kC = kH = 1)."""
    L1, a1, b1 = lab1.unbind(-1)
    L2, a2, b2 = lab2.unbind(-1)
    C1, C2 = torch.hypot(a1, b1), torch.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - torch.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = torch.hypot(a1p, b1), torch.hypot(a2p, b2)
    h1p = torch.where((a1p == 0) & (b1 == 0), torch.zeros_like(a1p), torch.atan2(b1, a1p) % (2 * math.pi))
    h2p = torch.where((a2p == 0) & (b2 == 0), torch.zeros_like(a2p), torch.atan2(b2, a2p) % (2 * math.pi))
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = torch.where(dh > math.pi, dh - 2 * math.pi, torch.where(dh < -math.pi, dh + 2 * math.pi, dh))
    dh = torch.where(C1p * C2p == 0, torch.zeros_like(dh), dh)
    dHp = 2 * torch.sqrt(C1p * C2p) * torch.sin(dh / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = torch.where(C1p * C2p == 0, hsum,
                      torch.where((h1p - h2p).abs() <= math.pi, hsum / 2,
                                  torch.where(hsum < 2 * math.pi, (hsum + 2 * math.pi) / 2, (hsum - 2 * math.pi) / 2)))
    T = (1 - 0.17 * torch.cos(hbp - math.radians(30)) + 0.24 * torch.cos(2 * hbp)
         + 0.32 * torch.cos(3 * hbp + math.radians(6)) - 0.20 * torch.cos(4 * hbp - math.radians(63)))
    dtheta = math.radians(30) * torch.exp(-(((torch.rad2deg(hbp) - 275) / 25) ** 2))
    Rc = 2 * torch.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / torch.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -torch.sin(2 * dtheta) * Rc
    return torch.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def delta_e(a: torch.Tensor, b: torch.Tensor, samples: int = DE_SAMPLES, seed: int = 0) -> torch.Tensor:
    """[N] mean CIEDE2000 over the same fixed pixel subsample of every frame (display sRGB)."""
    n, h, w = a.shape[:3]
    k = min(samples, h * w)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(h * w, generator=gen)[:k].to(a.device)
    pa = a[..., :3].reshape(n, h * w, 3)[:, idx].double()
    pb = b[..., :3].reshape(n, h * w, 3)[:, idx].double()
    return ciede2000(_srgb_to_lab(pa), _srgb_to_lab(pb)).mean(1).float()


def frame_metrics(a: torch.Tensor, b: torch.Tensor, chunk: int = 16) -> dict:
    """Per-frame PSNR / SSIM / ΔE of two [N, H, W, C] batches (B resized to A when the sizes differ, the shorter
    length wins), plus the worst frame of each."""
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n == 0:
        return {"frames": 0, "psnr": [], "ssim": [], "delta_e": [], "worst": {}}
    if tuple(b.shape[1:3]) != tuple(a.shape[1:3]):
        b = F.interpolate(_bchw(b[:n]), size=tuple(a.shape[1:3]), mode="bilinear",
                          align_corners=False).permute(0, 2, 3, 1)
    out = {"psnr": [], "ssim": [], "delta_e": []}
    for s in range(0, n, chunk):
        ca, cb = a[s:s + chunk].float(), b[s:s + chunk].float()
        out["psnr"] += psnr(ca, cb).tolist()
        out["ssim"] += ssim(ca, cb).tolist()
        out["delta_e"] += delta_e(ca, cb).tolist()
    worst = {"psnr": int(min(range(n), key=lambda i: out["psnr"][i])),
             "ssim": int(min(range(n), key=lambda i: out["ssim"][i])),
             "delta_e": int(max(range(n), key=lambda i: out["delta_e"][i]))}
    summary = {k: {"mean": float(sum(v) / n), "min": float(min(v)), "max": float(max(v))} for k, v in out.items()}
    return {"frames": n, **out, "worst": worst, "summary": summary}
