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
    CATEGORY = "MaskEditControl/Pipeline"
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
                        "crf_bilateral_srgb, edge_snap_strength, edge_snap_band."
                    ),
                }),
            },
        }

    # ── Preset table (used to populate stage numerics) ────────────────
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
    def execute(
        self,
        image, mask,
        preset,
        enable_hole_fill,
        morph_op,
        enable_thin_recover,
        enable_joint_bilateral,
        enable_guided_filter,
        enable_dense_crf,
        enable_edge_snap,
        cascade_passes,
        feather_sigma, gamma, threshold,
        advanced_overrides_json="",
    ):
        # Resolve preset → numerics, then layer JSON overrides on top.
        cfg = dict(self._PRESETS.get(preset, self._PRESETS["balanced"]))
        if advanced_overrides_json and advanced_overrides_json.strip():
            try:
                import json as _json
                user = _json.loads(advanced_overrides_json)
                if isinstance(user, dict):
                    cfg.update({k: v for k, v in user.items() if k in cfg})
            except Exception as e:
                log.warning("[MaskRefineMEC] bad advanced_overrides_json: %s", e)

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

            out_np[b] = m

        out_t = torch.from_numpy(out_np).to(m_t.device, dtype=torch.float32)

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
        info = _json.dumps({
            "ran": ran,
            "skipped": skipped,
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
        }, indent=2)

        return (out_t, alpha, preview, info)


NODE_CLASS_MAPPINGS = {"MaskRefineMEC": MaskRefineMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskRefineMEC": "Mask Refine — Unified (MEC)"}
