"""
_auto_quality — internal helper for MaskOpsMEC.

Three responsibilities:

1. ``analyze_image(img_bhwc)`` — fast probe that returns a dict of issues
   detected per-frame (motion blur, low contrast, low light, speck noise,
   low boundary contrast).  No models, no heavy ops.

2. ``preprocess_for_segmentation(img_bhwc, issues, mode)`` — returns a
   cleaned image suitable for handing to the segmenter.  The original
   image is preserved for matting; only the SAM/BiRefNet/RMBG input is
   touched.  Operations:
     * CLAHE on the L channel of LAB when contrast is low
     * adaptive unsharp mask when blur is detected
     * NL-means / bilateral denoise when speck noise is detected
     * mild gamma lift when median brightness is too low

3. ``smart_pick_sam_mask(masks, scores, pos_points, neg_points, w, h)``
   — given the 3 multimask candidates returned by SAM, pick the one that
   maximises (pos_covered − neg_covered) × score.  This is what fixes the
   "click face + neg-click neck → grab whole person" problem.

4. ``polish_alpha(img, alpha, strength)`` — light guided-filter + edge-snap
   on the final alpha.  Run after matting, never replaces MaskRefineMEC.

Everything degrades gracefully when cv2 / scipy aren't available.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("MEC.MaskOps.AutoQ")

try:
    import cv2  # type: ignore
    _CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _CV2 = False


# ──────────────────────────────────────────────────────────────────────
# 1. ANALYSE
# ──────────────────────────────────────────────────────────────────────
def analyze_image(img_bhwc: torch.Tensor) -> Dict[str, float]:
    """
    Single-pass image probe.  Uses the FIRST frame as a representative
    sample (cheap; the same lighting/blur usually applies to a clip).

    Returns:
        dict with floats in [0,1] indicating severity of each issue:
            blur_score          (higher = more blur)
            contrast_score      (higher = lower contrast)
            lowlight_score      (higher = darker)
            speckle_score       (higher = more speck noise)
            boundary_ambig      (higher = bg/fg colours similar)
        plus the raw measurements for the explainer.
    """
    img = img_bhwc[0] if img_bhwc.ndim == 4 else img_bhwc
    rgb = img.detach().float().cpu().clamp(0, 1).numpy()
    H, W, _ = rgb.shape
    gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]).astype(np.float32)

    # blur — Laplacian variance (Pech-Pacheco 2000).  Higher var = sharper.
    if _CV2:
        lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F)
        lap_var = float(lap.var())
    else:
        gy = np.diff(gray, axis=0, prepend=gray[:1])
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        lap_var = float((gx * gx + gy * gy).var() * 255.0 * 255.0)
    # below 50 = obvious blur, above 500 = sharp.  Map to 0..1 inverse.
    blur_score = float(np.clip((500.0 - lap_var) / 500.0, 0.0, 1.0))

    # contrast — std of L channel (LAB) when cv2 is available
    if _CV2:
        lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        L = lab[..., 0].astype(np.float32) / 255.0
    else:
        L = gray
    contrast_std = float(L.std())
    # std < 0.10 = flat, > 0.25 = punchy
    contrast_score = float(np.clip((0.20 - contrast_std) / 0.20, 0.0, 1.0))

    # lowlight — mean luminance
    mean_lum = float(gray.mean())
    lowlight_score = float(np.clip((0.35 - mean_lum) / 0.35, 0.0, 1.0))

    # speckle — local stddev minus global stddev (rough)
    if _CV2:
        blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residual = gray - blur
        local_noise = float(residual.std())
    else:
        local_noise = 0.0
    # 0.005 = clean, 0.04+ = heavy noise
    speckle_score = float(np.clip((local_noise - 0.005) / 0.04, 0.0, 1.0))

    # boundary ambiguity — corners (bg) vs center (likely fg) colour distance
    h2, w2 = H // 4, W // 4
    center = rgb[h2:H - h2, w2:W - w2].reshape(-1, 3).mean(0)
    corners = np.stack([
        rgb[:h2, :w2].reshape(-1, 3).mean(0),
        rgb[:h2, -w2:].reshape(-1, 3).mean(0),
        rgb[-h2:, :w2].reshape(-1, 3).mean(0),
        rgb[-h2:, -w2:].reshape(-1, 3).mean(0),
    ])
    dist = float(np.linalg.norm(corners - center[None], axis=1).mean())
    # dist < 0.05 = bg and fg very similar (low contrast boundary)
    boundary_ambig = float(np.clip((0.15 - dist) / 0.15, 0.0, 1.0))

    return {
        "blur_score": blur_score,
        "contrast_score": contrast_score,
        "lowlight_score": lowlight_score,
        "speckle_score": speckle_score,
        "boundary_ambig": boundary_ambig,
        "_lap_var": lap_var,
        "_contrast_std": contrast_std,
        "_mean_lum": mean_lum,
        "_local_noise": local_noise,
        "_bg_fg_dist": dist,
    }


# ──────────────────────────────────────────────────────────────────────
# 2. PRE-PROCESS (only the segmenter's input — original kept for matting)
# ──────────────────────────────────────────────────────────────────────
def preprocess_for_segmentation(
    img_bhwc: torch.Tensor,
    issues: Dict[str, float],
    quality_mode: str = "balanced",
) -> Tuple[torch.Tensor, List[str]]:
    """
    Return a copy of ``img_bhwc`` cleaned per the detected issues.
    quality_mode in {"fast","balanced","max_fidelity"} controls strength.

    Returns (clean_image, list_of_steps_run).
    """
    if not _CV2:
        return img_bhwc, ["preprocess_skipped:cv2_missing"]

    strength = {"fast": 0.5, "balanced": 1.0, "max_fidelity": 1.5}.get(
        quality_mode, 1.0
    )
    out = img_bhwc.detach().clone()
    steps: List[str] = []
    B = out.shape[0]

    for b in range(B):
        rgb = (out[b].cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        modified = False

        # 1. LOW CONTRAST or LOW LIGHT → CLAHE on L channel
        if issues["contrast_score"] > 0.35 or issues["lowlight_score"] > 0.30:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            clip = float(np.clip(2.0 + 4.0 * issues["contrast_score"]
                                  + 3.0 * issues["lowlight_score"],
                                  2.0, 10.0) * strength)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            lab[..., 0] = clahe.apply(lab[..., 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            if b == 0:
                steps.append(f"clahe(clip={clip:.1f})")
            modified = True

        # 2. BLUR / MOTION BLUR → adaptive unsharp mask
        if issues["blur_score"] > 0.50:
            sigma = float(np.clip(1.0 + 1.5 * issues["blur_score"], 1.0, 3.0))
            amount = float(np.clip(0.8 + 1.2 * issues["blur_score"], 0.8, 2.5)
                           * strength)
            blur = cv2.GaussianBlur(rgb, (0, 0), sigma)
            sharp = cv2.addWeighted(rgb, 1.0 + amount, blur, -amount, 0)
            rgb = np.clip(sharp, 0, 255).astype(np.uint8)
            if b == 0:
                steps.append(f"unsharp(sigma={sigma:.1f},amt={amount:.1f})")
            modified = True

        # 3. SPECKLE NOISE → non-local-means denoise (slow but excellent)
        if issues["speckle_score"] > 0.40 and quality_mode != "fast":
            h = float(np.clip(5.0 + 15.0 * issues["speckle_score"],
                               5.0, 25.0) * strength)
            rgb = cv2.fastNlMeansDenoisingColored(rgb, None, h, h, 7, 21)
            if b == 0:
                steps.append(f"nlmeans(h={h:.1f})")
            modified = True
        elif issues["speckle_score"] > 0.20:
            # cheap bilateral for mild noise / fast mode
            d = 7 if quality_mode == "fast" else 9
            rgb = cv2.bilateralFilter(rgb, d, 35, 7)
            if b == 0:
                steps.append("bilateral")
            modified = True

        # 4. BOUNDARY AMBIGUITY (bg/fg colour similar) → edge enhancement
        # via guided unsharp on the chroma channels to push fg/bg apart.
        if issues["boundary_ambig"] > 0.50:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            # Mild chroma stretch around the mean.
            for c in (1, 2):
                ch = lab[..., c]
                mu = float(ch.mean())
                spread = float(np.clip(1.10 + 0.20 * issues["boundary_ambig"]
                                        * strength, 1.05, 1.5))
                ch = (ch - mu) * spread + mu
                lab[..., c] = np.clip(ch, 0, 255)
            rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            if b == 0:
                steps.append(f"chroma_stretch(x{spread:.2f})")
            modified = True

        if modified:
            out[b] = torch.from_numpy(rgb.astype(np.float32) / 255.0).to(out.device)

    if not steps:
        steps.append("no_preprocess_needed")
    return out, steps


# ──────────────────────────────────────────────────────────────────────
# 3. SMART SAM MASK PICKER (the face-vs-person fix)
# ──────────────────────────────────────────────────────────────────────
def smart_pick_sam_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    img_h: int,
    img_w: int,
) -> Tuple[int, Dict[str, float]]:
    """
    Pick the SAM multimask candidate that best satisfies the user's
    point constraints.  Returns ``(best_idx, score_dict)``.

    Score = w_score * model_score
          + w_pos   * (fraction of pos points inside mask)
          - w_neg   * (fraction of neg points inside mask)
          - w_size  * size_penalty_when_neg_present

    When neg_points are provided we strongly prefer the SMALLEST mask
    among those that exclude all neg points — that's how you stop SAM
    grabbing the whole torso when the user pos-clicked face and
    neg-clicked neck.
    """
    if masks.ndim == 2:
        masks = masks[None]
        scores = np.array([float(scores)])
    K = masks.shape[0]
    if K == 0:
        return 0, {"reason": "no_masks"}

    # Convert points to integer pixel indices, clipped to image bounds.
    def _idx(pts):
        return [(int(np.clip(round(x), 0, img_w - 1)),
                 int(np.clip(round(y), 0, img_h - 1)))
                for (x, y) in pts]
    pos_idx = _idx(pos_points)
    neg_idx = _idx(neg_points)

    have_neg = len(neg_idx) > 0
    have_pos = len(pos_idx) > 0

    best_idx = 0
    best_combined = -1e9
    details: List[Dict] = []
    for k in range(K):
        m = (masks[k] > 0.5).astype(np.float32)
        area = float(m.sum()) / max(1.0, float(img_h * img_w))

        pos_cov = 1.0
        if have_pos:
            pos_cov = float(np.mean([m[y, x] for (x, y) in pos_idx]))
        neg_cov = 0.0
        if have_neg:
            neg_cov = float(np.mean([m[y, x] for (x, y) in neg_idx]))

        # weights — when neg present, heavily punish neg coverage and
        # also lightly punish huge masks (whole-person sweep).
        if have_neg:
            combined = (0.30 * float(scores[k])
                        + 0.50 * pos_cov
                        - 1.20 * neg_cov
                        - 0.20 * area)
        else:
            combined = (0.70 * float(scores[k])
                        + 0.30 * pos_cov)

        details.append({
            "idx": k,
            "model_score": float(scores[k]),
            "pos_cov": pos_cov,
            "neg_cov": neg_cov,
            "area": area,
            "combined": combined,
        })
        if combined > best_combined:
            best_combined = combined
            best_idx = k

    return best_idx, {
        "picked": best_idx,
        "have_neg": have_neg,
        "have_pos": have_pos,
        "candidates": details,
    }


# ──────────────────────────────────────────────────────────────────────
# 4. POST-MATTING ALPHA POLISH
# ──────────────────────────────────────────────────────────────────────
def _guided_filter_torch(I: torch.Tensor, p: torch.Tensor,
                          r: int, eps: float) -> torch.Tensor:
    """He et al. guided filter — same impl as MaskRefineMEC, inlined to
    avoid a circular import."""
    if I.shape[1] == 3:
        I = 0.2126 * I[:, 0:1] + 0.7152 * I[:, 1:2] + 0.0722 * I[:, 2:3]
    k = 2 * r + 1
    box = torch.ones((1, 1, k, k), device=I.device, dtype=I.dtype) / float(k * k)
    pad = (r, r, r, r)

    def _b(x):
        return F.conv2d(F.pad(x, pad, mode="reflect"), box)

    mI = _b(I); mp = _b(p)
    cI = _b(I * I); cIp = _b(I * p)
    vI = cI - mI * mI
    covIp = cIp - mI * mp
    a = covIp / (vI + eps)
    b = mp - a * mI
    return _b(a) * I + _b(b)


def polish_alpha(
    img_bhwc: torch.Tensor,
    alpha_bhw: torch.Tensor,
    quality_mode: str = "balanced",
) -> torch.Tensor:
    """
    Light post-matting polish: guided filter (snap to RGB gradients) +
    soft edge snap.  Strengths chosen so the alpha doesn't change much
    on already-clean inputs.  This is NOT a substitute for MaskRefineMEC
    — that node still wins for power users.
    """
    cfg = {
        "fast":          dict(gf_r=4,  gf_eps=1e-4, snap=0.25, band=3),
        "balanced":      dict(gf_r=6,  gf_eps=5e-5, snap=0.40, band=4),
        "max_fidelity": dict(gf_r=10, gf_eps=1e-5, snap=0.55, band=6),
    }.get(quality_mode, {"gf_r": 6, "gf_eps": 5e-5, "snap": 0.4, "band": 4})

    dev = alpha_bhw.device
    img = img_bhwc.to(dev).float().clamp(0, 1)
    if img.shape[1] != 3 and img.shape[-1] == 3:
        img_chw = img.permute(0, 3, 1, 2).contiguous()
    else:
        img_chw = img
    a = alpha_bhw.float().clamp(0, 1).unsqueeze(1)  # (B,1,H,W)
    a_gf = _guided_filter_torch(img_chw, a,
                                 int(cfg["gf_r"]), float(cfg["gf_eps"]))
    a_gf = a_gf.clamp(0, 1)

    # cheap edge snap via image-gradient-weighted blend in a thin band
    snap = float(cfg["snap"])
    band = int(cfg["band"])
    if snap > 0 and band > 0:
        # gray gradient magnitude
        gray = (0.2126 * img_chw[:, 0:1] + 0.7152 * img_chw[:, 1:2]
                + 0.0722 * img_chw[:, 2:3])
        gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        gy = F.pad(gy, (0, 0, 0, 1))
        gx = F.pad(gx, (0, 1, 0, 0))
        gmag = torch.sqrt(gx * gx + gy * gy).clamp(0, 1)
        gmag = gmag / (gmag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        # mask boundary band
        ay = a_gf[:, :, 1:, :] - a_gf[:, :, :-1, :]
        ax = a_gf[:, :, :, 1:] - a_gf[:, :, :, :-1]
        ay = F.pad(ay, (0, 0, 0, 1))
        ax = F.pad(ax, (0, 1, 0, 0))
        edge = (ax.abs() + ay.abs()).clamp(0, 1)
        edge_band = F.max_pool2d(edge, kernel_size=band * 2 + 1,
                                  stride=1, padding=band)
        edge_band = (edge_band > 0.05).float()
        blend = (1.0 - snap) + snap * gmag
        a_gf = a_gf * (1.0 - edge_band) + (a_gf * blend) * edge_band
        a_gf = a_gf.clamp(0, 1)

    return a_gf.squeeze(1)
