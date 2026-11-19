"""L9.174 (A25): repeatable VAE reconstruction benchmark + small chart rendering.

Shared by VAE Inspect (VAEBlockInspectorMEC, benchmark) and VAE Merge (VAEMergeMEC, validation). Deterministic synthetic
images (fixed seeds), encode -> decode through ComfyUI's VAE wrapper API (encode takes IMAGE (B,H,W,3) in [0,1], decode
returns (B,H,W,C)), error statistics per image type and resolution, and a decode of random latents. Bounded work: at most
4 resolutions of 32..1024 px, 5 image types, 6 latent decodes. Precision modes are not covered (that would need a
dtype change on the caller's VAE).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

IMAGE_TYPES = ("flat_grey", "gradient", "checker", "colour_bars", "noise")


def parse_resolutions(text) -> List[int]:
    """'64,128' -> [64, 128]. Multiples of 8, 32..1024, at most 4 entries."""
    if isinstance(text, (list, tuple)):
        parts = [str(x) for x in text]
    else:
        parts = [p.strip() for p in str(text or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("benchmark resolutions: give at least one size, e.g. '64,128'")
    if len(parts) > 4:
        raise ValueError(f"benchmark resolutions: at most 4 sizes, got {len(parts)}")
    out = []
    for p in parts:
        try:
            r = int(p)
        except ValueError:
            raise ValueError(f"benchmark resolutions: {p!r} is not a whole number") from None
        if r < 32 or r > 1024 or r % 8:
            raise ValueError(f"benchmark resolutions: {r} must be a multiple of 8 between 32 and 1024")
        out.append(r)
    return out


def synthetic_images(resolution: int, types: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """Deterministic test images (1,H,W,3) float32 in [0,1]."""
    r = int(resolution)
    types = list(types or IMAGE_TYPES)
    g = torch.Generator().manual_seed(1234 + r)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, r), torch.linspace(0, 1, r), indexing="ij")
    out = {}
    for t in types:
        if t == "flat_grey":
            img = torch.full((r, r, 3), 0.5)
        elif t == "gradient":
            img = torch.stack([xx, yy, 1.0 - 0.5 * (xx + yy)], dim=-1)
        elif t == "checker":
            ii = torch.arange(r)
            c = (((ii.view(-1, 1) // 2) + (ii.view(1, -1) // 2)) % 2).float()
            img = c.unsqueeze(-1).expand(r, r, 3).clone()
        elif t == "colour_bars":
            bars = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1],
                                 [0, 0, 0]], dtype=torch.float32)
            idx = (torch.arange(r) * len(bars) // r).clamp(max=len(bars) - 1)
            img = bars[idx].unsqueeze(0).expand(r, r, 3).clone()
        elif t == "noise":
            img = torch.rand(r, r, 3, generator=g)
        else:
            raise ValueError(f"unknown benchmark image type {t!r}; choose from {', '.join(IMAGE_TYPES)}")
        out[t] = img.unsqueeze(0).to(torch.float32).contiguous()
    return out


def _roundtrip(vae, img: torch.Tensor) -> torch.Tensor:
    lat = vae.encode(img)
    rec = vae.decode(lat)
    if not isinstance(rec, torch.Tensor):
        raise TypeError(f"decode returned {type(rec).__name__}, not a tensor")
    rec = rec.to(torch.float32).cpu()
    while rec.ndim > 4:                       # video VAEs: (B,T,H,W,C) -> first frame
        rec = rec[:, 0]
    rec = rec[..., :3]
    if tuple(rec.shape[1:3]) != tuple(img.shape[1:3]):
        rec = F.interpolate(rec.permute(0, 3, 1, 2), size=tuple(img.shape[1:3]), mode="bilinear",
                            align_corners=False).permute(0, 2, 3, 1)
    return rec[:1]


def _stats(img: torch.Tensor, rec: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(rec).all())
    d = rec - img
    mse = float((d ** 2).mean()) if finite else float("nan")
    return {"mse": round(mse, 8) if finite else None,
            "psnr_db": round(10.0 * math.log10(1.0 / max(mse, 1e-12)), 3) if finite else None,
            "max_abs_err": round(float(d.abs().max()), 6) if finite else None,
            "mean_shift_rgb": [round(float(x), 6) for x in d.mean(dim=(0, 1, 2))] if finite else None,
            "finite": finite}


def reconstruction_suite(vae, resolutions=(64, 128), types: Optional[Iterable[str]] = None) -> dict:
    """Encode -> decode every synthetic image at every resolution; error statistics per case and as means."""
    res = parse_resolutions(resolutions)
    t0 = time.perf_counter()
    cases, errors = [], []
    with torch.no_grad():
        for r in res:
            for name, img in synthetic_images(r, types).items():
                try:
                    st = _stats(img, _roundtrip(vae, img))
                except Exception as exc:  # noqa: BLE001 - reported per case, never a fake number
                    st = {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "finite": False}
                    errors.append(f"{name}@{r}")
                cases.append({"type": name, "resolution": r, **st})
    ok = [c for c in cases if c.get("psnr_db") is not None]
    per_res = {}
    for r in res:
        v = [c["psnr_db"] for c in ok if c["resolution"] == r]
        per_res[str(r)] = round(sum(v) / len(v), 3) if v else None
    return {"cases": cases, "mean_psnr_db_by_resolution": per_res,
            "mean_psnr_db": round(sum(c["psnr_db"] for c in ok) / len(ok), 3) if ok else None,
            "all_finite": all(c.get("finite") for c in cases), "failed_cases": errors,
            "seconds": round(time.perf_counter() - t0, 3)}


def latent_suite(vae, resolution: int = 64) -> dict:
    """Decode N(0,1) latents (seeds 0..2, scales 0.5 / 1.0) of the shape this VAE encodes to."""
    r = parse_resolutions([resolution])[0]
    with torch.no_grad():
        try:
            lat = vae.encode(synthetic_images(r, ["gradient"])["gradient"])
        except Exception as exc:  # noqa: BLE001
            return {"error": f"encode failed: {type(exc).__name__}: {str(exc)[:200]}"}
        rows = []
        for seed in (0, 1, 2):
            for scale in (0.5, 1.0):
                g = torch.Generator().manual_seed(seed)
                z = (torch.randn(tuple(lat.shape), generator=g) * scale).to(dtype=lat.dtype, device=lat.device)
                try:
                    out = vae.decode(z).to(torch.float32)
                    fin = bool(torch.isfinite(out).all())
                    rows.append({"seed": seed, "scale": scale, "finite": fin,
                                 "out_min": round(float(out.min()), 5) if fin else None,
                                 "out_max": round(float(out.max()), 5) if fin else None,
                                 "clip_fraction": round(float(((out < 0) | (out > 1)).float().mean()), 6)
                                 if fin else None})
                except Exception as exc:  # noqa: BLE001
                    rows.append({"seed": seed, "scale": scale, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    return {"latent_shape": list(lat.shape), "decodes": rows,
            "all_finite": all(x.get("finite") for x in rows)}


# ── charts (IMAGE tensors) ───────────────────────────────────────────────────────────────────────────────────────────

def _ramp(v: float):
    """0 -> dark blue, 1 -> bright yellow (perceptually ordered)."""
    v = max(0.0, min(1.0, float(v)))
    return (0.08 + 0.92 * v, 0.10 + 0.80 * v, 0.35 * (1.0 - v) + 0.10)


def render_grid(rows: List[str], cols: List[str], values: List[List[Optional[float]]],
                marks: Optional[List[bool]] = None, cell: int = 22, label_w: int = 180,
                header_h: int = 18, colour=None) -> torch.Tensor:
    """A labelled heatmap as IMAGE (1,H,W,3). values in [0,1] (None = grey); marks = red marker column."""
    colour = colour or _ramp
    rows = rows[:256]
    nmark = 1 if marks is not None else 0
    H = header_h + cell * max(1, len(rows))
    W = label_w + cell * (len(cols) + nmark)
    img = torch.full((H, W, 3), 0.12)
    for i in range(len(rows)):
        y0 = header_h + i * cell
        for j, v in enumerate(values[i][:len(cols)]):
            x0 = label_w + j * cell
            c = (0.35, 0.35, 0.35) if v is None else colour(v)
            img[y0 + 1:y0 + cell - 1, x0 + 1:x0 + cell - 1] = torch.tensor(c)
        if marks is not None and marks[i]:
            x0 = label_w + len(cols) * cell
            img[y0 + 1:y0 + cell - 1, x0 + 1:x0 + cell - 1] = torch.tensor((0.90, 0.15, 0.15))
    try:                                            # labels: PIL is a ComfyUI dependency; the grid works without it
        from PIL import Image, ImageDraw
        pil = Image.fromarray((img.numpy() * 255).astype("uint8"))
        d = ImageDraw.Draw(pil)
        for i, name in enumerate(rows):
            d.text((4, header_h + i * cell + 4), str(name)[:28], fill=(225, 225, 225))
        for j, name in enumerate(list(cols) + (["!"] if nmark else [])):
            d.text((label_w + j * cell + 2, 3), str(name)[:3], fill=(225, 225, 225))
        import numpy as np
        img = torch.from_numpy(np.asarray(pil).astype("float32") / 255.0)
    except Exception:  # noqa: BLE001
        pass
    return img.unsqueeze(0).contiguous()


def render_bars(names: List[str], series: Dict[str, List[Optional[float]]], width: int = 360, row_h: int = 18,
                label_w: int = 140) -> torch.Tensor:
    """Horizontal bars per name for each series (values in [0,1]) as IMAGE (1,H,W,3)."""
    palette = [(0.30, 0.65, 0.95), (0.95, 0.65, 0.25), (0.55, 0.85, 0.45)]
    keys = list(series)
    band = row_h * max(1, len(keys)) + 6
    H = 20 + band * max(1, len(names))
    W = label_w + width + 10
    img = torch.full((H, W, 3), 0.12)
    for i, _n in enumerate(names):
        for k, key in enumerate(keys):
            v = series[key][i] if i < len(series[key]) else None
            if v is None:
                continue
            y0 = 20 + i * band + k * row_h
            x1 = label_w + int(max(0.0, min(1.0, float(v))) * width)
            img[y0 + 2:y0 + row_h - 2, label_w:max(label_w + 1, x1)] = torch.tensor(palette[k % len(palette)])
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray((img.numpy() * 255).astype("uint8"))
        d = ImageDraw.Draw(pil)
        d.text((4, 4), "  ".join(f"[{k}]" for k in keys), fill=(225, 225, 225))
        for i, n in enumerate(names):
            d.text((4, 20 + i * band + 4), str(n)[:22], fill=(225, 225, 225))
        import numpy as np
        img = torch.from_numpy(np.asarray(pil).astype("float32") / 255.0)
    except Exception:  # noqa: BLE001
        pass
    return img.unsqueeze(0).contiguous()
