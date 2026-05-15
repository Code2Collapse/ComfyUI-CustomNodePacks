"""MaskRefineMEC — SINGLE training-free mask refinement node.

Consolidates all post-segmentation mask cleanup techniques into ONE node.
Zero model weights, zero fine-tuning. Every step is independently toggleable
so the same node can be used as a cheap edge-smoother, a CascadePSP-style
boundary snapper, or a full hair/fur recovery pipeline.

Pipeline (executed in this order, each step gated by its `enable_*` flag):

    1.  hole_fill          – scipy.ndimage.binary_fill_holes on a binarized
                              copy. Plugs gaps that the segmenter punched
                              through interior textures (eyes, mouths, ...).
    2.  morphology         – open / close / erode / dilate with circular SE.
                              Removes salt+pepper noise or fills slivers.
    3.  thin_recover       – skeletonize → keep branches ≥ N px → dilate.
                              Re-injects hair / wire / grass that ViT
                              segmenters routinely cut off.
    4.  joint_bilateral    – cv2.ximgproc.jointBilateralFilter (RGB guide).
                              Cheap edge-preserving denoise.
    5.  guided_filter      – He et al., PAMI 2013. Snaps alpha to color
                              gradients. The backbone of "free" matting.
    6.  dense_crf          – Krähenbühl & Koltun, NeurIPS 2011. Gold-standard
                              boundary refinement when pydensecrf is present.
    7.  edge_snap          – Multiply mask boundary band by RGB gradient
                              magnitude (with `strength` blend). Cheap
                              alternative to CRF.
    8.  cascade_passes     – Repeat steps 4-7 with a shrinking uncertainty
                              band (CascadePSP-style without the network).
    9.  feather            – Gaussian blur on the soft alpha.
   10.  gamma              – Pow curve on soft alpha.
   11.  threshold          – If > 0, hard-binarize at this value (else soft).

References (all training-free):
- Krähenbühl & Koltun, "Efficient Inference in Fully Connected CRFs", NeurIPS 2011.
- He, Sun, Tang, "Guided Image Filtering", PAMI 2013.
- Cheng et al., "CascadePSP: Toward Class-Agnostic and Very High-Resolution
  Segmentation via Global and Local Refinement", CVPR 2020 (cascade pattern only).
- Tomasi & Manduchi, "Bilateral Filtering for Gray and Color Images", ICCV 1998.
- Zhang et al., "Rolling Guidance Filter", ECCV 2014 (cascade idea).
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("MEC.MaskRefineMEC")

# ── Optional deps (degrade gracefully) ───────────────────────────────
try:
    import cv2  # type: ignore
    _CV2 = True
    try:
        _ = cv2.ximgproc.jointBilateralFilter  # type: ignore[attr-defined]
        _HAS_XIMG = True
    except Exception:
        _HAS_XIMG = False
except Exception:
    cv2 = None  # type: ignore
    _CV2 = False
    _HAS_XIMG = False

try:
    from scipy import ndimage as _ndi  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _ndi = None  # type: ignore
    _HAS_SCIPY = False

try:
    from skimage.morphology import skeletonize as _skeletonize  # type: ignore
    _HAS_SKIMAGE = True
except Exception:
    _skeletonize = None  # type: ignore
    _HAS_SKIMAGE = False

try:
    import pydensecrf.densecrf as _dcrf  # type: ignore
    from pydensecrf.utils import unary_from_softmax as _unary_from_softmax  # type: ignore
    _HAS_CRF = True
except Exception:
    _dcrf = None  # type: ignore
    _unary_from_softmax = None  # type: ignore
    _HAS_CRF = False


# ── Tensor helpers ────────────────────────────────────────────────────
def _to_mask_bhw(m: torch.Tensor) -> torch.Tensor:
    if m is None:
        raise ValueError("mask is None")
    if m.ndim == 4:
        m = m.mean(dim=1) if (m.shape[1] in (1, 3) and m.shape[-1] not in (1, 3)) else m.mean(dim=-1)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    return m.float().clamp(0.0, 1.0)


def _to_image_bhwc(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.shape[1] == 3 and img.shape[-1] != 3:
        img = img.permute(0, 2, 3, 1).contiguous()
    return img.float().clamp(0.0, 1.0)


def _match_batch(img: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if img.shape[0] == m.shape[0]:
        return img, m
    if img.shape[0] == 1 and m.shape[0] > 1:
        img = img.expand(m.shape[0], -1, -1, -1)
    elif m.shape[0] == 1 and img.shape[0] > 1:
        m = m.expand(img.shape[0], -1, -1)
    return img, m


# ── Core training-free primitives ────────────────────────────────────
def _morph_disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def _hole_fill(m: np.ndarray, bin_thresh: float) -> np.ndarray:
    if not _HAS_SCIPY:
        return m
    binary = (m > bin_thresh).astype(np.uint8)
    filled = _ndi.binary_fill_holes(binary).astype(np.float32)
    # Re-inject soft values where the original was already high; only fill
    # places that were < bin_thresh AND ended up filled.
    out = np.maximum(m, filled * (m <= bin_thresh).astype(np.float32))
    return out.clip(0.0, 1.0)


def _morphology(m: np.ndarray, op: str, radius: int) -> np.ndarray:
    if op == "none" or radius <= 0 or not _CV2:
        return m
    se = _morph_disk(radius)
    src = (m * 255.0).clip(0, 255).astype(np.uint8)
    if op == "close":
        dst = cv2.morphologyEx(src, cv2.MORPH_CLOSE, se)
    elif op == "open":
        dst = cv2.morphologyEx(src, cv2.MORPH_OPEN, se)
    elif op == "dilate":
        dst = cv2.dilate(src, se)
    elif op == "erode":
        dst = cv2.erode(src, se)
    else:
        return m
    return dst.astype(np.float32) / 255.0


def _thin_structure_recover(m: np.ndarray, thresh: float,
                            min_len: int, dilate: int) -> np.ndarray:
    if not (_HAS_SKIMAGE and _HAS_SCIPY):
        return m
    binary = (m > thresh).astype(np.uint8)
    if binary.sum() < 5:
        return m
    skel = _skeletonize(binary.astype(bool)).astype(np.uint8)
    interior = _ndi.binary_erosion(binary, iterations=max(1, dilate)).astype(np.uint8)
    thin = skel * (1 - interior)
    labels, n = _ndi.label(thin)
    keep = np.zeros_like(thin)
    for lbl in range(1, n + 1):
        comp = labels == lbl
        if comp.sum() >= int(min_len):
            keep[comp] = 1
    recovered = _ndi.binary_dilation(keep, iterations=int(dilate)).astype(np.float32)
    return np.maximum(m, recovered).clip(0.0, 1.0)


def _joint_bilateral(m: np.ndarray, rgb: np.ndarray,
                     d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    if not _CV2:
        return m
    guide = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    src = (m * 255.0).clip(0, 255).astype(np.uint8)
    if _HAS_XIMG:
        dst = cv2.ximgproc.jointBilateralFilter(
            guide, src, int(d), float(sigma_color), float(sigma_space)
        )
    else:
        # Fallback: plain bilateral on mask alone.
        dst = cv2.bilateralFilter(src, int(d), float(sigma_color), float(sigma_space))
    return dst.astype(np.float32) / 255.0


def _guided_filter_torch(I: torch.Tensor, p: torch.Tensor,
                         r: int, eps: float) -> torch.Tensor:
    """He et al. guided filter. I: (B,3,H,W) or (B,1,H,W). p: (B,1,H,W)."""
    if I.shape[1] == 3:
        I = 0.2126 * I[:, 0:1] + 0.7152 * I[:, 1:2] + 0.0722 * I[:, 2:3]
    k = 2 * r + 1
    box = torch.ones((1, 1, k, k), device=I.device, dtype=I.dtype) / float(k * k)
    pad = (r, r, r, r)

    def _b(x):
        return F.conv2d(F.pad(x, pad, mode="reflect"), box)

    mI = _b(I)
    mp = _b(p)
    cI = _b(I * I)
    cIp = _b(I * p)
    vI = cI - mI * mI
    covIp = cIp - mI * mp
    a = covIp / (vI + eps)
    b = mp - a * mI
    return _b(a) * I + _b(b)


def _dense_crf(rgb: np.ndarray, soft: np.ndarray,
               iters: int, gauss_sxy: float,
               bilat_sxy: float, bilat_srgb: float) -> np.ndarray:
    if not _HAS_CRF:
        return soft
    H, W = soft.shape
    p = np.clip(soft, 1e-5, 1.0 - 1e-5)
    probs = np.stack([1.0 - p, p], axis=0).astype(np.float32)
    d = _dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(_unary_from_softmax(probs))
    d.addPairwiseGaussian(sxy=(gauss_sxy, gauss_sxy), compat=3)
    rgb_u8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    d.addPairwiseBilateral(
        sxy=(bilat_sxy, bilat_sxy),
        srgb=(bilat_srgb, bilat_srgb, bilat_srgb),
        rgbim=np.ascontiguousarray(rgb_u8),
        compat=10,
    )
    Q = d.inference(int(iters))
    return np.array(Q).reshape(2, H, W)[1].astype(np.float32)


def _edge_snap(m: np.ndarray, rgb: np.ndarray, strength: float,
               band_radius: int = 6) -> np.ndarray:
    """Multiply mask boundary band by image gradient magnitude."""
    if strength <= 0.0:
        return m
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    g = np.sqrt(gx * gx + gy * gy)
    g = g / max(g.max(), 1e-6)
    # Boundary band of the mask.
    mgx = np.gradient(m, axis=1)
    mgy = np.gradient(m, axis=0)
    mg = np.sqrt(mgx * mgx + mgy * mgy)
    if _HAS_SCIPY:
        band = _ndi.binary_dilation(mg > 0.05, iterations=int(band_radius)).astype(np.float32)
    else:
        band = (mg > 0.05).astype(np.float32)
    # Inside the band, blend mask towards (mask * (alpha + (1-alpha)*image_grad)).
    blend = (1.0 - strength) + strength * g
    new = m.copy()
    new = m * (1.0 - band) + (m * blend) * band
    return new.clip(0.0, 1.0)


# ── NEW power-user primitives ────────────────────────────────────────
def _domain_transform(m: np.ndarray, rgb: np.ndarray,
                       sigma_s: float, sigma_r: float) -> np.ndarray:
    """Gastal & Oliveira (2011) Domain Transform — RGB-edge-preserving
    filter. Sharper than guided filter on hair-like fine detail, much
    faster than DenseCRF. Uses cv2.ximgproc.dtFilter when available."""
    if not (_CV2 and _HAS_XIMG):
        return m
    try:
        dt = cv2.ximgproc.dtFilter  # type: ignore[attr-defined]
    except Exception:
        return m
    guide = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    src = m.astype(np.float32)
    try:
        return np.clip(dt(guide, src, float(sigma_s), float(sigma_r),
                          mode=cv2.ximgproc.DTF_RF), 0.0, 1.0)
    except Exception:
        return m


def _color_decontaminate(m: np.ndarray, rgb: np.ndarray,
                          band_radius: int = 3,
                          strength: float = 0.5) -> np.ndarray:
    """Push the alpha in the thin boundary band toward 0 or 1 based on
    LAB-distance to the local fg/bg means. Boundary colour cleanup
    without modifying RGB (since this node returns mask not RGBA)."""
    if not (_CV2 and _HAS_SCIPY):
        return m
    binary = (m > 0.5).astype(np.uint8)
    if binary.sum() < 5 or (1 - binary).sum() < 5:
        return m
    # build band
    eroded = _ndi.binary_erosion(binary, iterations=band_radius).astype(np.uint8)
    dilated = _ndi.binary_dilation(binary, iterations=band_radius).astype(np.uint8)
    band = (dilated & (1 - eroded)).astype(bool)
    if not band.any():
        return m
    lab = cv2.cvtColor((rgb * 255).clip(0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_pixels = lab[eroded.astype(bool)]
    bg_pixels = lab[(1 - dilated).astype(bool)]
    if fg_pixels.size < 9 or bg_pixels.size < 9:
        return m
    fg_mu = fg_pixels.mean(axis=0)
    bg_mu = bg_pixels.mean(axis=0)
    band_pix = lab[band]
    d_fg = np.linalg.norm(band_pix - fg_mu[None], axis=1)
    d_bg = np.linalg.norm(band_pix - bg_mu[None], axis=1)
    target = (d_bg / (d_fg + d_bg + 1e-6)).astype(np.float32)
    out = m.copy()
    out[band] = (1.0 - strength) * m[band] + strength * target
    return out.clip(0.0, 1.0)


def _unsharp_alpha(m: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Alpha sharpening via unsharp mask: α + amount·(α − gauss(α))."""
    if not _CV2 or sigma <= 0 or amount <= 0:
        return m
    blur = cv2.GaussianBlur(m.astype(np.float32), (0, 0), float(sigma))
    sharp = m + float(amount) * (m - blur)
    return np.clip(sharp, 0.0, 1.0)


def _anti_alias(m: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Subpixel boundary smoothing: bilinear up 2× → soft thresh → down."""
    if not _CV2 or strength <= 0:
        return m
    h, w = m.shape
    big = cv2.resize(m, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    # soft threshold-like contrast bend that smooths the rampy region
    big = np.clip((big - 0.5) * (1.0 + strength) + 0.5, 0.0, 1.0)
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def _chroma_lock(m: np.ndarray, rgb: np.ndarray,
                  strength: float = 0.5, band_radius: int = 5) -> np.ndarray:
    """When the LAB-luma gradient is flat (similar fg/bg lightness) but
    LAB-chroma changes, weight the boundary by chroma gradient instead.
    Helps when fg/bg luminance is identical (e.g. red lipstick on similar
    red background)."""
    if not _CV2:
        return m
    lab = cv2.cvtColor((rgb * 255).clip(0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    gL = np.hypot(*np.gradient(L))
    gC = np.hypot(*np.gradient(A)) + np.hypot(*np.gradient(B))
    gL_n = gL / max(gL.max(), 1e-6)
    gC_n = gC / max(gC.max(), 1e-6)
    # weight = strong where chroma > luma gradient
    chroma_dom = np.clip(gC_n - gL_n, 0.0, 1.0)
    mgx = np.gradient(m, axis=1)
    mgy = np.gradient(m, axis=0)
    mg = np.hypot(mgx, mgy)
    if _HAS_SCIPY:
        band = _ndi.binary_dilation(mg > 0.05,
                                     iterations=int(band_radius)).astype(np.float32)
    else:
        band = (mg > 0.05).astype(np.float32)
    blend = (1.0 - strength) + strength * chroma_dom
    out = m * (1.0 - band) + (m * blend) * band
    return out.clip(0.0, 1.0)


def _speck_removal(m: np.ndarray, min_area: int = 32,
                    fill_holes_below: int = 32) -> np.ndarray:
    """Drop foreground specks below min_area px and fill background holes
    below fill_holes_below px (small isolated negatives inside the
    subject)."""
    if not _HAS_SCIPY:
        return m
    binary = (m > 0.5).astype(np.uint8)
    if binary.sum() < 1:
        return m
    labels, n = _ndi.label(binary)
    keep = np.zeros_like(binary, dtype=bool)
    for lbl in range(1, n + 1):
        comp = labels == lbl
        if comp.sum() >= int(min_area):
            keep |= comp
    # holes inside foreground
    inv = 1 - keep.astype(np.uint8)
    h_labels, h_n = _ndi.label(inv)
    for lbl in range(1, h_n + 1):
        comp = h_labels == lbl
        # exclude background that touches the image border
        if (comp[0, :].any() or comp[-1, :].any()
                or comp[:, 0].any() or comp[:, -1].any()):
            continue
        if comp.sum() < int(fill_holes_below):
            keep |= comp
    out = m * keep.astype(np.float32)
    return out.clip(0.0, 1.0)


def _temporal_smooth(stack: torch.Tensor, alpha_ema: float = 0.7,
                      bidirectional: bool = True) -> torch.Tensor:
    """Per-pixel EMA across the batch dimension. Use only for video.
    stack: (B,H,W). When bidirectional, runs forward then backward and
    averages — kills high-freq flicker without lagging the mask."""
    if stack.shape[0] < 2 or alpha_ema <= 0:
        return stack
    a = float(np.clip(alpha_ema, 0.0, 0.99))
    fwd = stack.clone()
    for t in range(1, stack.shape[0]):
        fwd[t] = a * fwd[t - 1] + (1.0 - a) * stack[t]
    if not bidirectional:
        return fwd.clamp(0.0, 1.0)
    bwd = stack.clone()
    for t in range(stack.shape[0] - 2, -1, -1):
        bwd[t] = a * bwd[t + 1] + (1.0 - a) * stack[t]
    return ((fwd + bwd) * 0.5).clamp(0.0, 1.0)


def _feather(m: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return m
    # 1-D separable Gaussian
    radius = max(1, int(round(sigma * 3)))
    k = 2 * radius + 1
    x = torch.arange(k, dtype=m.dtype, device=m.device) - radius
    w = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    w = w / w.sum()
    wx = w.view(1, 1, 1, k)
    wy = w.view(1, 1, k, 1)
    p = m.unsqueeze(1)
    p = F.conv2d(F.pad(p, (radius, radius, 0, 0), mode="reflect"), wx)
    p = F.conv2d(F.pad(p, (0, 0, radius, radius), mode="reflect"), wy)
    return p.squeeze(1).clamp(0.0, 1.0)


# ── The node ─────────────────────────────────────────────────────────
class MaskRefineMEC:
    """Single training-free mask refinement node — all SOTA primitives."""

    DESCRIPTION = (
        "Unified, training-free mask refinement. Toggle each stage on or "
        "off and chain them in fixed order: hole-fill → morphology → "
        "thin-structure recover → joint bilateral → guided filter → "
        "DenseCRF → edge-snap → optional CascadePSP-style multi-pass → "
        "feather → gamma → threshold. No model weights. Optional deps: "
        "opencv-contrib-python (ximgproc joint bilateral), scikit-image "
        "(thin recovery), scipy (hole fill / morphology helpers), "
        "pydensecrf (DenseCRF). Each missing dep silently disables only "
        "its stage; the rest still run."
    )
    CATEGORY = "C2C/Pipeline"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "alpha", "preview", "info")
    OUTPUT_TOOLTIPS = (
        "Refined mask, same (B,H,W) as input.",
        "Soft alpha (same as mask before threshold).",
        "RGB×alpha preview for quick visual diff.",
        "JSON describing which stages ran / were skipped.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        # Compact UI: stage toggles + one "preset" combo that fills in the
        # numerics. Power users can override any default by wiring a JSON
        # blob into `advanced_overrides_json` (optional, defaults to "").
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "RGB guide. Required for all edge-aware stages."}),
                "mask":  ("MASK",  {"tooltip": "Mask to refine (soft or hard)."}),
                "preset": (
                    ["balanced", "fast", "hair", "aggressive", "crf_heavy"],
                    {"default": "balanced",
                     "tooltip": "Picks sensible numeric defaults for every enabled stage. "
                                "Override with `advanced_overrides_json` if needed."},
                ),
                "auto_edge_lock": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Pin the mask to real image edges to stop bleeding into "
                        "the background. Force-enables guided_filter + edge_snap "
                        "with parameters tuned by `subject_class`. Recommended "
                        "for any face / garment / product / hair workflow."
                    ),
                }),
                "subject_class": (
                    ["general", "face", "hair", "garment", "object", "hard_surface"],
                    {"default": "general",
                     "tooltip": (
                        "Tunes auto_edge_lock for the dominant subject:\n"
                        "  general     — balanced edge protection\n"
                        "  face        — tight 2-3 px band, high snap, no morph dilate\n"
                        "  hair        — thin-recover ON, soft band, low snap\n"
                        "  garment     — medium band, medium snap, close holes\n"
                        "  object      — medium band, high snap, fill holes\n"
                        "  hard_surface— thin band, max snap, threshold > 0.5"),
                    },
                ),

                # 11 stage toggles (one widget each — these are the only knobs
                # most users touch).
                "enable_hole_fill":       ("BOOLEAN",                                      {"default": False}),
                "morph_op":               (["none", "close", "open", "dilate", "erode"],   {"default": "none"}),
                "enable_thin_recover":    ("BOOLEAN",                                      {"default": False}),
                "enable_joint_bilateral": ("BOOLEAN",                                      {"default": False}),
                "enable_guided_filter":   ("BOOLEAN",                                      {"default": True}),
                "enable_dense_crf":       ("BOOLEAN",                                      {"default": False}),
                "enable_edge_snap":       ("BOOLEAN",                                      {"default": False}),
                "cascade_passes":         ("INT",   {"default": 0, "min": 0, "max": 5}),
                "feather_sigma":          ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "gamma":                  ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0,  "step": 0.05}),
                "threshold":              ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,  "step": 0.01}),

                # ── NEW power-user stages ──────────────────────────────
                "enable_domain_transform": ("BOOLEAN", {"default": False, "tooltip":
                    "Gastal & Oliveira (2011) Domain Transform RGB-edge "
                    "filter. Often sharper than guided filter on hair / "
                    "fine detail; much faster than DenseCRF. Requires "
                    "opencv-contrib-python (ximgproc)."}),
                "enable_color_decontaminate": ("BOOLEAN", {"default": False, "tooltip":
                    "Push alpha in the boundary band toward 0/1 using "
                    "local LAB-distance to fg/bg means. Fixes 'halo' "
                    "alpha bleed when bg has similar luminance."}),
                "enable_unsharp_alpha": ("BOOLEAN", {"default": False, "tooltip":
                    "Sharpen the soft alpha (α + amount·(α − gauss(α)))."}),
                "enable_anti_alias": ("BOOLEAN", {"default": False, "tooltip":
                    "Sub-pixel boundary smoothing (bilinear up 2× → soft "
                    "contrast → down)."}),
                "enable_chroma_lock": ("BOOLEAN", {"default": False, "tooltip":
                    "When fg/bg luminance is similar but chroma differs, "
                    "weight the boundary by LAB chroma gradient instead "
                    "of luma. Helps red-on-red, green-on-green, etc."}),
                "enable_speck_removal": ("BOOLEAN", {"default": False, "tooltip":
                    "Drop foreground components below `speck_min_area` "
                    "and fill background holes inside the subject."}),
                "enable_temporal_smooth": ("BOOLEAN", {"default": False, "tooltip":
                    "Bidirectional alpha EMA across the batch dim (for "
                    "VIDEO masks only). Removes flicker without lag."}),
            },
            "optional": {
                "advanced_overrides_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional JSON overriding any preset numeric. "
                        'Example: {"gf_radius":12, "jb_sigma_color":40, "crf_iterations":8}. '
                        "Recognised keys: hole_fill_threshold, morph_radius, "
                        "thin_threshold, thin_min_branch_len, thin_branch_dilate, "
                        "jb_diameter, jb_sigma_color, jb_sigma_space, gf_radius, "
                        "gf_epsilon, crf_iterations, crf_gauss_sxy, crf_bilateral_sxy, "
                        "crf_bilateral_srgb, edge_snap_strength, edge_snap_band, "
                        "dt_sigma_s, dt_sigma_r, decontam_band, decontam_strength, "
                        "unsharp_sigma, unsharp_amount, anti_alias_strength, "
                        "chroma_lock_strength, chroma_lock_band, speck_min_area, "
                        "speck_fill_holes_below, temporal_alpha, temporal_bidi."
                    ),
                }),
                "enable_integrity_check": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Compute per-frame integrity stats on the refined mask "
                        "(coverage, abrupt drops, frame-to-frame jumps) and "
                        "append them to the `info` JSON. Adds negligible cost "
                        "for single frames; cheap for short clips."
                    ),
                }),
                "integrity_drop_threshold": ("FLOAT", {
                    "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Relative coverage drop that flags a frame."}),
                "integrity_jump_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Relative frame-to-frame coverage jump that flags a frame."}),
            },
        }

    # ── Preset table (used to populate stage numerics) ────────────────
    # Defaults for the new power-user stages — same for every preset
    # unless a preset overrides them below.
    _NEW_STAGE_DEFAULTS = {
        "dt_sigma_s": 30.0, "dt_sigma_r": 0.10,
        "decontam_band": 3, "decontam_strength": 0.5,
        "unsharp_sigma": 1.0, "unsharp_amount": 0.5,
        "anti_alias_strength": 0.5,
        "chroma_lock_strength": 0.5, "chroma_lock_band": 5,
        "speck_min_area": 32, "speck_fill_holes_below": 32,
        "temporal_alpha": 0.7, "temporal_bidi": True,
    }
    _PRESETS = {
        "balanced": {
            "hole_fill_threshold": 0.5,
            "morph_radius": 3,
            "thin_threshold": 0.5, "thin_min_branch_len": 8, "thin_branch_dilate": 2,
            "jb_diameter": 9, "jb_sigma_color": 25.0, "jb_sigma_space": 7.0,
            "gf_radius": 8, "gf_epsilon": 1e-4,
            "crf_iterations": 5, "crf_gauss_sxy": 3.0,
            "crf_bilateral_sxy": 50.0, "crf_bilateral_srgb": 13.0,
            "edge_snap_strength": 0.5, "edge_snap_band": 6,
        },
        "fast": {
            "hole_fill_threshold": 0.5,
            "morph_radius": 2,
            "thin_threshold": 0.5, "thin_min_branch_len": 12, "thin_branch_dilate": 1,
            "jb_diameter": 5, "jb_sigma_color": 15.0, "jb_sigma_space": 5.0,
            "gf_radius": 4, "gf_epsilon": 1e-3,
            "crf_iterations": 3, "crf_gauss_sxy": 2.0,
            "crf_bilateral_sxy": 30.0, "crf_bilateral_srgb": 10.0,
            "edge_snap_strength": 0.35, "edge_snap_band": 4,
        },
        "hair": {
            "hole_fill_threshold": 0.4,
            "morph_radius": 2,
            "thin_threshold": 0.35, "thin_min_branch_len": 4, "thin_branch_dilate": 1,
            "jb_diameter": 7, "jb_sigma_color": 20.0, "jb_sigma_space": 4.0,
            "gf_radius": 4, "gf_epsilon": 1e-5,
            "crf_iterations": 5, "crf_gauss_sxy": 2.0,
            "crf_bilateral_sxy": 40.0, "crf_bilateral_srgb": 8.0,
            "edge_snap_strength": 0.65, "edge_snap_band": 4,
        },
        "aggressive": {
            "hole_fill_threshold": 0.5,
            "morph_radius": 5,
            "thin_threshold": 0.5, "thin_min_branch_len": 6, "thin_branch_dilate": 3,
            "jb_diameter": 13, "jb_sigma_color": 40.0, "jb_sigma_space": 10.0,
            "gf_radius": 16, "gf_epsilon": 1e-4,
            "crf_iterations": 10, "crf_gauss_sxy": 4.0,
            "crf_bilateral_sxy": 70.0, "crf_bilateral_srgb": 15.0,
            "edge_snap_strength": 0.8, "edge_snap_band": 10,
        },
        "crf_heavy": {
            "hole_fill_threshold": 0.5,
            "morph_radius": 3,
            "thin_threshold": 0.5, "thin_min_branch_len": 8, "thin_branch_dilate": 2,
            "jb_diameter": 9, "jb_sigma_color": 25.0, "jb_sigma_space": 7.0,
            "gf_radius": 8, "gf_epsilon": 1e-4,
            "crf_iterations": 15, "crf_gauss_sxy": 3.0,
            "crf_bilateral_sxy": 60.0, "crf_bilateral_srgb": 13.0,
            "edge_snap_strength": 0.5, "edge_snap_band": 6,
        },
    }

    # -----------------------------------------------------------------
    # Subject-aware edge-lock overlays (only applied when auto_edge_lock=True)
    # Each entry layers ON TOP of the numeric preset and ALSO force-enables
    # the stages needed to physically pin the boundary to the image gradient.
    _SUBJECT_PROFILES = {
        "general": {
            "force": {"enable_guided_filter": True, "enable_edge_snap": True},
            "cfg":   {"gf_radius": 8,  "gf_epsilon": 1e-4,
                      "edge_snap_strength": 0.55, "edge_snap_band": 6},
            "morph": "keep",
        },
        "face": {
            "force": {"enable_guided_filter": True, "enable_edge_snap": True,
                      "enable_joint_bilateral": True},
            "cfg":   {"gf_radius": 6,  "gf_epsilon": 1e-5,
                      "jb_diameter": 7, "jb_sigma_color": 18.0, "jb_sigma_space": 5.0,
                      "edge_snap_strength": 0.75, "edge_snap_band": 3,
                      "feather_sigma_override": 0.6},
            "morph": "none",   # never dilate a face mask
        },
        "hair": {
            "force": {"enable_guided_filter": True, "enable_edge_snap": True,
                      "enable_thin_recover": True},
            "cfg":   {"gf_radius": 4,  "gf_epsilon": 1e-5,
                      "thin_threshold": 0.35, "thin_min_branch_len": 4,
                      "thin_branch_dilate": 1,
                      "edge_snap_strength": 0.5, "edge_snap_band": 5},
            "morph": "none",   # preserve thin strands
        },
        "garment": {
            "force": {"enable_guided_filter": True, "enable_edge_snap": True,
                      "enable_hole_fill": True},
            "cfg":   {"gf_radius": 10, "gf_epsilon": 1e-4,
                      "hole_fill_threshold": 0.5,
                      "edge_snap_strength": 0.6, "edge_snap_band": 5},
            "morph": "keep",
        },
        "object": {
            "force": {"enable_guided_filter": True, "enable_edge_snap": True,
                      "enable_hole_fill": True},
            "cfg":   {"gf_radius": 12, "gf_epsilon": 1e-4,
                      "hole_fill_threshold": 0.5,
                      "edge_snap_strength": 0.7, "edge_snap_band": 5},
            "morph": "close_if_none",
        },
        "hard_surface": {
            "force": {"enable_guided_filter": True, "enable_edge_snap": True},
            "cfg":   {"gf_radius": 6,  "gf_epsilon": 1e-5,
                      "edge_snap_strength": 0.9, "edge_snap_band": 2,
                      "threshold_override": 0.5},
            "morph": "none",
        },
    }

    # -----------------------------------------------------------------
    def execute(
        self,
        image, mask,
        preset,
        auto_edge_lock,
        subject_class,
        enable_hole_fill,
        morph_op,
        enable_thin_recover,
        enable_joint_bilateral,
        enable_guided_filter,
        enable_dense_crf,
        enable_edge_snap,
        cascade_passes,
        feather_sigma, gamma, threshold,
        enable_domain_transform=False,
        enable_color_decontaminate=False,
        enable_unsharp_alpha=False,
        enable_anti_alias=False,
        enable_chroma_lock=False,
        enable_speck_removal=False,
        enable_temporal_smooth=False,
        advanced_overrides_json="",
        enable_integrity_check: bool = False,
        integrity_drop_threshold: float = 0.40,
        integrity_jump_threshold: float = 0.15,
    ):
        # Resolve preset → numerics, then layer JSON overrides on top.
        cfg = dict(self._PRESETS.get(preset, self._PRESETS["balanced"]))
        # New power-user stage defaults (presets can override per-key below)
        for _k, _v in self._NEW_STAGE_DEFAULTS.items():
            cfg.setdefault(_k, _v)
        if advanced_overrides_json and advanced_overrides_json.strip():
            try:
                import json as _json
                user = _json.loads(advanced_overrides_json)
                if isinstance(user, dict):
                    cfg.update({k: v for k, v in user.items() if k in cfg})
            except Exception as e:
                log.warning("[MaskRefineMEC] bad advanced_overrides_json: %s", e)

        # Subject-aware edge-lock: force-enable edge-pinning stages and
        # overlay subject-specific numerics on top of the preset.
        edge_lock_applied = None
        if auto_edge_lock:
            prof = self._SUBJECT_PROFILES.get(subject_class,
                                              self._SUBJECT_PROFILES["general"])
            cfg.update(prof["cfg"])
            for flag, val in prof["force"].items():
                # force-enable stages
                if flag == "enable_hole_fill":
                    enable_hole_fill = bool(val) or enable_hole_fill
                elif flag == "enable_thin_recover":
                    enable_thin_recover = bool(val) or enable_thin_recover
                elif flag == "enable_joint_bilateral":
                    enable_joint_bilateral = bool(val) or enable_joint_bilateral
                elif flag == "enable_guided_filter":
                    enable_guided_filter = bool(val) or enable_guided_filter
                elif flag == "enable_edge_snap":
                    enable_edge_snap = bool(val) or enable_edge_snap
            # Morphology override policy
            if prof["morph"] == "none":
                morph_op = "none"
            elif prof["morph"] == "close_if_none" and morph_op == "none":
                morph_op = "close"
            # Optional auto-overrides for feather / threshold
            if "feather_sigma_override" in cfg and feather_sigma == 0.0:
                feather_sigma = float(cfg["feather_sigma_override"])
            if "threshold_override" in cfg and threshold == 0.0:
                threshold = float(cfg["threshold_override"])
            edge_lock_applied = subject_class

        # Bind locals so the original pipeline body works unchanged.
        hole_fill_threshold = float(cfg["hole_fill_threshold"])
        morph_radius        = int(cfg["morph_radius"])
        thin_threshold      = float(cfg["thin_threshold"])
        thin_min_branch_len = int(cfg["thin_min_branch_len"])
        thin_branch_dilate  = int(cfg["thin_branch_dilate"])
        jb_diameter         = int(cfg["jb_diameter"])
        jb_sigma_color      = float(cfg["jb_sigma_color"])
        jb_sigma_space      = float(cfg["jb_sigma_space"])
        gf_radius           = int(cfg["gf_radius"])
        gf_epsilon          = float(cfg["gf_epsilon"])
        crf_iterations      = int(cfg["crf_iterations"])
        crf_gauss_sxy       = float(cfg["crf_gauss_sxy"])
        crf_bilateral_sxy   = float(cfg["crf_bilateral_sxy"])
        crf_bilateral_srgb  = float(cfg["crf_bilateral_srgb"])
        edge_snap_strength  = float(cfg["edge_snap_strength"])
        edge_snap_band      = int(cfg["edge_snap_band"])

        img_t = _to_image_bhwc(image)
        m_t = _to_mask_bhw(mask)
        img_t, m_t = _match_batch(img_t, m_t)

        ran: list[str] = []
        skipped: list[str] = []

        B = img_t.shape[0]
        out_np = np.zeros_like(m_t.cpu().numpy())

        for b in range(B):
            rgb = img_t[b].cpu().numpy()
            m = m_t[b].cpu().numpy()

            # 1. hole fill
            if enable_hole_fill:
                if _HAS_SCIPY:
                    m = _hole_fill(m, float(hole_fill_threshold))
                    if b == 0:
                        ran.append("hole_fill")
                elif b == 0:
                    skipped.append("hole_fill(missing:scipy)")

            # 2. morphology
            if morph_op != "none":
                if _CV2:
                    m = _morphology(m, morph_op, int(morph_radius))
                    if b == 0:
                        ran.append(f"morph_{morph_op}_r{int(morph_radius)}")
                elif b == 0:
                    skipped.append("morphology(missing:cv2)")

            # 3. thin recover
            if enable_thin_recover:
                if _HAS_SKIMAGE and _HAS_SCIPY:
                    m = _thin_structure_recover(m, float(thin_threshold),
                                                int(thin_min_branch_len),
                                                int(thin_branch_dilate))
                    if b == 0:
                        ran.append("thin_recover")
                elif b == 0:
                    skipped.append("thin_recover(missing:skimage/scipy)")

            # 4. joint bilateral
            if enable_joint_bilateral:
                if _CV2:
                    m = _joint_bilateral(m, rgb, int(jb_diameter),
                                         float(jb_sigma_color), float(jb_sigma_space))
                    if b == 0:
                        ran.append("joint_bilateral" + ("(ximg)" if _HAS_XIMG else "(fallback)"))
                elif b == 0:
                    skipped.append("joint_bilateral(missing:cv2)")

            # 5. guided filter (torch — always available)
            if enable_guided_filter:
                It = img_t[b:b + 1].permute(0, 3, 1, 2).contiguous()
                mt = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(It.device, dtype=It.dtype)
                gf = _guided_filter_torch(It, mt, int(gf_radius), float(gf_epsilon))
                m = gf.squeeze().cpu().numpy().clip(0.0, 1.0)
                if b == 0:
                    ran.append("guided_filter")

            # 6. dense CRF
            if enable_dense_crf:
                if _HAS_CRF:
                    m = _dense_crf(rgb, m, int(crf_iterations),
                                   float(crf_gauss_sxy),
                                   float(crf_bilateral_sxy),
                                   float(crf_bilateral_srgb))
                    if b == 0:
                        ran.append("dense_crf")
                elif b == 0:
                    skipped.append("dense_crf(missing:pydensecrf)")

            # 7. edge snap
            if enable_edge_snap:
                m = _edge_snap(m, rgb, float(edge_snap_strength), int(edge_snap_band))
                if b == 0:
                    ran.append("edge_snap")

            # 8. cascade passes (shrinking radii)
            for cp in range(int(cascade_passes)):
                # Radii shrink each pass; eps and band scale similarly.
                shrink = 0.5 ** cp
                r_g = max(2, int(round(int(gf_radius) * shrink)))
                if enable_joint_bilateral and _CV2:
                    m = _joint_bilateral(m, rgb,
                                         max(3, int(jb_diameter * shrink) | 1),
                                         float(jb_sigma_color),
                                         float(jb_sigma_space) * shrink)
                It = img_t[b:b + 1].permute(0, 3, 1, 2).contiguous()
                mt = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(It.device, dtype=It.dtype)
                gf = _guided_filter_torch(It, mt, r_g, float(gf_epsilon))
                m = gf.squeeze().cpu().numpy().clip(0.0, 1.0)
                if enable_edge_snap:
                    m = _edge_snap(m, rgb,
                                   float(edge_snap_strength) * shrink,
                                   max(2, int(edge_snap_band * shrink)))
                if b == 0 and cp == 0:
                    ran.append(f"cascade_x{int(cascade_passes)}")

            # 8a. NEW: Domain Transform (sharper than guided on hair)
            if enable_domain_transform:
                if _CV2 and _HAS_XIMG:
                    m = _domain_transform(m, rgb,
                                           float(cfg["dt_sigma_s"]),
                                           float(cfg["dt_sigma_r"]))
                    if b == 0:
                        ran.append("domain_transform")
                elif b == 0:
                    skipped.append("domain_transform(missing:cv2.ximgproc)")

            # 8b. NEW: colour decontamination at boundary
            if enable_color_decontaminate:
                if _CV2 and _HAS_SCIPY:
                    m = _color_decontaminate(m, rgb,
                                              int(cfg["decontam_band"]),
                                              float(cfg["decontam_strength"]))
                    if b == 0:
                        ran.append("color_decontaminate")
                elif b == 0:
                    skipped.append("color_decontaminate(missing:cv2/scipy)")

            # 8c. NEW: speck / hole cleanup
            if enable_speck_removal:
                if _HAS_SCIPY:
                    m = _speck_removal(m,
                                        int(cfg["speck_min_area"]),
                                        int(cfg["speck_fill_holes_below"]))
                    if b == 0:
                        ran.append("speck_removal")
                elif b == 0:
                    skipped.append("speck_removal(missing:scipy)")

            # 8d. NEW: chroma-gradient edge lock
            if enable_chroma_lock:
                if _CV2:
                    m = _chroma_lock(m, rgb,
                                      float(cfg["chroma_lock_strength"]),
                                      int(cfg["chroma_lock_band"]))
                    if b == 0:
                        ran.append("chroma_lock")
                elif b == 0:
                    skipped.append("chroma_lock(missing:cv2)")

            # 8e. NEW: alpha unsharp
            if enable_unsharp_alpha:
                if _CV2:
                    m = _unsharp_alpha(m,
                                        float(cfg["unsharp_sigma"]),
                                        float(cfg["unsharp_amount"]))
                    if b == 0:
                        ran.append("unsharp_alpha")
                elif b == 0:
                    skipped.append("unsharp_alpha(missing:cv2)")

            # 8f. NEW: subpixel anti-alias
            if enable_anti_alias:
                if _CV2:
                    m = _anti_alias(m, float(cfg["anti_alias_strength"]))
                    if b == 0:
                        ran.append("anti_alias")
                elif b == 0:
                    skipped.append("anti_alias(missing:cv2)")

            out_np[b] = m

        out_t = torch.from_numpy(out_np).to(m_t.device, dtype=torch.float32)

        # 8g. NEW: temporal EMA across batch (video flicker)
        if enable_temporal_smooth and out_t.shape[0] > 1:
            out_t = _temporal_smooth(out_t,
                                      float(cfg["temporal_alpha"]),
                                      bool(cfg["temporal_bidi"]))
            ran.append(f"temporal_smooth(a={float(cfg['temporal_alpha']):.2f})")

        # 9. feather (torch, batched)
        if feather_sigma > 0:
            out_t = _feather(out_t, float(feather_sigma))
            ran.append(f"feather_s{float(feather_sigma):.2f}")

        # 10. gamma
        if abs(float(gamma) - 1.0) > 1e-6:
            out_t = out_t.clamp(0.0, 1.0).pow(float(gamma))
            ran.append(f"gamma_{float(gamma):.2f}")

        alpha = out_t.clone()

        # 11. threshold
        if threshold > 0:
            out_t = (out_t >= float(threshold)).float()
            ran.append(f"threshold@{float(threshold):.2f}")

        # Preview = image * alpha (B,H,W,3).
        preview = (img_t * alpha.unsqueeze(-1)).clamp(0.0, 1.0)

        import json as _json
        info_payload = {
            "ran": ran,
            "skipped": skipped,
            "auto_edge_lock": edge_lock_applied,
            "deps": {
                "cv2": _CV2,
                "cv2.ximgproc": _HAS_XIMG,
                "scipy": _HAS_SCIPY,
                "skimage": _HAS_SKIMAGE,
                "pydensecrf": _HAS_CRF,
            },
            "input_shape": list(m_t.shape),
            "output_shape": list(out_t.shape),
            "input_sum": float(m_t.sum().item()),
            "output_sum": float(out_t.sum().item()),
        }

        if enable_integrity_check:
            try:
                from .temporal_node import compute_integrity
                _mask_for_integrity = out_t if out_t.dim() == 3 else out_t.unsqueeze(0)
                integ = compute_integrity(
                    _mask_for_integrity,
                    drop_threshold=float(integrity_drop_threshold),
                    jump_threshold=float(integrity_jump_threshold),
                )
                info_payload["integrity"] = integ
                try:
                    from .integrity_bridge import publish as _integrity_publish
                    extra = {}
                    frames_arr = integ.get("frames", [])
                    if isinstance(frames_arr, list) and frames_arr:
                        extra["area"] = [float(f.get("area", 0.0)) for f in frames_arr]
                        extra["centroid_dx"] = [float(f.get("centroid_delta", 0.0)) for f in frames_arr]
                        extra["iou_prev"] = [float(f.get("iou_prev", 1.0)) for f in frames_arr]
                    _integrity_publish("MaskRefineMEC", integ, extra)
                except Exception:
                    pass
            except Exception as _e:  # pragma: no cover - defensive
                info_payload["integrity_error"] = str(_e)

        info = _json.dumps(info_payload, indent=2)

        return (out_t, alpha, preview, info)


NODE_CLASS_MAPPINGS = {"MaskRefineMEC": MaskRefineMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskRefineMEC": "Mask Refiner (C2C)"}
