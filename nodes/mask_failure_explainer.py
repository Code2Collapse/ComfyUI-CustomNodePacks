"""
MaskFailureExplainerMEC – Diagnose why a mask failed and suggest fixes.

Input: image (B,H,W,C), mask (B,H,W) of unknown quality.
Runs a pure-tensor analysis pipeline:
  1. Brightness: mean luminance per frame. <0.15 = dark scene.
  2. Blur: Laplacian variance of image. <50 = blurry.
  3. Contrast at boundary: std of image pixels at mask edge ring. <0.05 = similar.
  4. Boundary color confusion: mean color distance inside vs outside mask boundary. <0.1 = too similar.
  5. Background complexity: edge density outside mask region. >0.3 = busy background.

Outputs:
  - explanation STRING: real computed values per-frame, specific actionable advice
  - problem_regions_mask MASK: heatmap of detected issues (not zeros)
  - severity_score FLOAT: 0-100 computed from metrics
  - suggested_method STRING: based on which conditions triggered

No models. Pure tensor math. VRAM Tier 1.
"""


from __future__ import annotations

from . import _interrupt_check as _IC

import gc
import torch
import torch.nn.functional as F

# ── Optional cv2 with torch fallback ──────────────────────────────────
from . import _progress as _PB
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Device helper ─────────────────────────────────────────────────────

def _get_device(tensor: torch.Tensor) -> torch.device:
    """Return the device of the tensor — never hardcode 'cuda'."""
    return tensor.device


# ── Laplacian kernel (3x3 standard) ──────────────────────────────────

_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)

# ── Sobel kernels ────────────────────────────────────────────────────

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [0.0,  0.0,  0.0],
     [1.0,  2.0,  1.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════
#  Analysis functions — pure torch, batch-aware
# ══════════════════════════════════════════════════════════════════════

def _compute_luminance(image: torch.Tensor) -> torch.Tensor:
    """BT.709 luminance from (B,H,W,C) image → (B,H,W)."""
    return 0.2126 * image[:, :, :, 0] + 0.7152 * image[:, :, :, 1] + 0.0722 * image[:, :, :, 2]


def _compute_brightness(image: torch.Tensor) -> torch.Tensor:
    """Per-frame mean brightness. Returns (B,) tensor."""
    luma = _compute_luminance(image)  # (B,H,W)
    return luma.mean(dim=(-2, -1))  # (B,)


def _compute_blur_score_torch(gray: torch.Tensor) -> torch.Tensor:
    """Laplacian variance per frame via conv2d. gray: (B,H,W) → (B,) scores.

    Convention: higher = sharper. Multiply by 1000 so threshold ~50 is meaningful.
    """
    device = _get_device(gray)
    B, H, W = gray.shape
    kernel = _LAPLACIAN_KERNEL.to(device=device, dtype=gray.dtype)
    # (B,1,H,W) for conv2d
    inp = gray.unsqueeze(1)
    lap = F.conv2d(inp, kernel, padding=1)  # (B,1,H,W)
    lap = lap.squeeze(1)  # (B,H,W)
    # Variance of Laplacian per frame
    var_per_frame = lap.var(dim=(-2, -1))  # (B,)
    return var_per_frame * 1000.0


def _compute_blur_score_cv2(gray_np):
    """Laplacian variance via cv2 for a single HxW numpy array. Returns float."""
    lap = cv2.Laplacian(gray_np, cv2.CV_64F)
    return float(lap.var()) * 1000.0


def _compute_blur_score(image: torch.Tensor) -> torch.Tensor:
    """Blur score per frame. Returns (B,) tensor. Uses cv2 if available, else torch."""
    luma = _compute_luminance(image)  # (B,H,W)
    if HAS_CV2:
        import numpy as np
        scores = []
        for i in _PB.track(range(luma.shape[0]), luma.shape[0], "MaskFailure"):
            _IC.check()
            gray_np = luma[i].cpu().numpy().astype(np.float64)
            scores.append(_compute_blur_score_cv2(gray_np))
        return torch.tensor(scores, device=_get_device(image), dtype=image.dtype)
    else:
        return _compute_blur_score_torch(luma)


def _get_mask_edge_ring(mask: torch.Tensor, ring_width: int = 5) -> torch.Tensor:
    """Compute a binary edge ring around the mask boundary.

    mask: (B,H,W) → returns (B,H,W) binary ring.
    Uses morphological dilation minus erosion via max_pool2d.
    """
    B, H, W = mask.shape
    binary = (mask > 0.5).float().unsqueeze(1)  # (B,1,H,W)

    pad = ring_width
    # Dilation via max_pool
    dilated = F.max_pool2d(
        binary, kernel_size=2 * pad + 1, stride=1, padding=pad
    )
    # Erosion via -max_pool(-x)
    eroded = -F.max_pool2d(
        -binary, kernel_size=2 * pad + 1, stride=1, padding=pad
    )
    ring = (dilated - eroded).squeeze(1).clamp(0.0, 1.0)  # (B,H,W)
    return ring


def _compute_boundary_contrast(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Std of image pixels at mask edge ring, per frame. Returns (B,)."""
    ring = _get_mask_edge_ring(mask)  # (B,H,W)
    luma = _compute_luminance(image)  # (B,H,W)
    B = image.shape[0]
    results = []
    for i in _PB.track(range(B), B, "MaskFailure"):
        _IC.check()
        ring_pixels = luma[i][ring[i] > 0.5]
        if ring_pixels.numel() < 2:
            results.append(0.0)
        else:
            results.append(ring_pixels.std().item())
    return torch.tensor(results, device=_get_device(image), dtype=image.dtype)


def _compute_boundary_color_confusion(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean color distance between inside and outside mask at boundary. Returns (B,).

    At the mask boundary ring, compare mean color on the mask side vs bg side.
    """
    ring = _get_mask_edge_ring(mask)  # (B,H,W)
    binary = (mask > 0.5).float()
    B = image.shape[0]
    results = []
    for i in _PB.track(range(B), B, "MaskFailure"):
        _IC.check()
        ring_mask = ring[i] > 0.5
        inside = ring_mask & (binary[i] > 0.5)
        outside = ring_mask & (binary[i] <= 0.5)
        if inside.sum() < 1 or outside.sum() < 1:
            results.append(0.0)
            continue
        # Mean color inside and outside the ring
        color_inside = image[i][inside].mean(dim=0)   # (C,)
        color_outside = image[i][outside].mean(dim=0)  # (C,)
        dist = (color_inside - color_outside).abs().mean().item()
        results.append(dist)
    return torch.tensor(results, device=_get_device(image), dtype=image.dtype)


def _compute_bg_complexity_torch(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Edge density in background region via Sobel. Returns (B,) in [0,1]."""
    device = _get_device(image)
    luma = _compute_luminance(image)  # (B,H,W)
    B, H, W = luma.shape
    sx = _SOBEL_X.to(device=device, dtype=luma.dtype)
    sy = _SOBEL_Y.to(device=device, dtype=luma.dtype)
    inp = luma.unsqueeze(1)  # (B,1,H,W)
    gx = F.conv2d(inp, sx, padding=1).squeeze(1)  # (B,H,W)
    gy = F.conv2d(inp, sy, padding=1).squeeze(1)  # (B,H,W)
    edges = (gx.pow(2) + gy.pow(2)).sqrt()  # (B,H,W)
    # Threshold edges at 0.1 to get binary edge map
    edge_binary = (edges > 0.1).float()

    bg_mask = (mask <= 0.5).float()  # (B,H,W)
    results = []
    for i in _PB.track(range(B), B, "MaskFailure"):
        _IC.check()
        bg_pixels = bg_mask[i].sum().item()
        if bg_pixels < 1:
            results.append(0.0)
        else:
            edge_in_bg = (edge_binary[i] * bg_mask[i]).sum().item()
            results.append(edge_in_bg / bg_pixels)
    return torch.tensor(results, device=device, dtype=image.dtype)


def _compute_bg_complexity_cv2(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Edge density in background region via cv2 Canny. Returns (B,)."""
    import numpy as np
    luma = _compute_luminance(image)  # (B,H,W)
    bg_mask = (mask <= 0.5).float()
    B = image.shape[0]
    results = []
    for i in _PB.track(range(B), B, "MaskFailure"):
        _IC.check()
        gray_np = (luma[i].cpu().numpy() * 255).astype(np.uint8)
        edges = cv2.Canny(gray_np, 50, 150)
        edge_binary = (edges > 0).astype(np.float32)
        bg_np = bg_mask[i].cpu().numpy()
        bg_pixels = bg_np.sum()
        if bg_pixels < 1:
            results.append(0.0)
        else:
            results.append(float((edge_binary * bg_np).sum() / bg_pixels))
    return torch.tensor(results, device=_get_device(image), dtype=image.dtype)


def _compute_bg_complexity(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Background complexity (edge density outside mask). Returns (B,)."""
    if HAS_CV2:
        return _compute_bg_complexity_cv2(image, mask)
    else:
        return _compute_bg_complexity_torch(image, mask)


# ══════════════════════════════════════════════════════════════════════
#  Problem regions heatmap
# ══════════════════════════════════════════════════════════════════════

def _build_problem_heatmap(
    image: torch.Tensor,
    mask: torch.Tensor,
    brightness: torch.Tensor,
    blur: torch.Tensor,
    boundary_contrast: torch.Tensor,
    color_confusion: torch.Tensor,
    bg_complexity: torch.Tensor,
) -> torch.Tensor:
    """Build a (B,H,W) heatmap highlighting problematic regions.

    Combines:
      - Low brightness regions → heatmap where image is dark
      - Blurry regions → high-frequency deficit areas
      - Boundary zone → where contrast/color confusion is bad
      - Complex bg → edge-dense background areas
    """
    B, H, W, C = image.shape
    device = _get_device(image)
    heatmap = torch.zeros(B, H, W, device=device, dtype=image.dtype)

    luma = _compute_luminance(image)  # (B,H,W)
    ring = _get_mask_edge_ring(mask)  # (B,H,W)
    bg_mask = (mask <= 0.5).float()

    for i in _PB.track(range(B), B, "MaskFailure"):
        _IC.check()
        frame_heat = torch.zeros(H, W, device=device, dtype=image.dtype)

        # Dark regions contribute where brightness is low
        if brightness[i].item() < 0.15:
            dark_map = (1.0 - luma[i]).clamp(0.0, 1.0)
            frame_heat = frame_heat + dark_map * 0.3

        # Boundary problems: highlight ring where contrast is low
        if boundary_contrast[i].item() < 0.05 or color_confusion[i].item() < 0.1:
            frame_heat = frame_heat + ring[i] * 0.4

        # Background complexity: highlight edges in bg
        if bg_complexity[i].item() > 0.3:
            # Compute edge map for this frame
            sx = _SOBEL_X.to(device=device, dtype=image.dtype)
            sy = _SOBEL_Y.to(device=device, dtype=image.dtype)
            inp = luma[i].unsqueeze(0).unsqueeze(0)
            gx = F.conv2d(inp, sx, padding=1).squeeze()
            gy = F.conv2d(inp, sy, padding=1).squeeze()
            edges = (gx.pow(2) + gy.pow(2)).sqrt()
            frame_heat = frame_heat + edges * bg_mask[i] * 0.3

        # If blur is bad, add a uniform low-level heat (blur is global)
        if blur[i].item() < 50.0:
            frame_heat = frame_heat + 0.15

        # Ensure some signal even if no issues detected (baseline from mask edge)
        frame_heat = frame_heat + ring[i] * 0.05

        heatmap[i] = frame_heat

    return heatmap.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Severity scoring
# ══════════════════════════════════════════════════════════════════════

# MANUAL bug-fix (Apr 2026): severity-score component weights extracted as
# named constants so they can be calibrated against a labeled set without
# editing function bodies. Each weight is the maximum penalty contribution
# for its metric (full 100 = bad image+bad mask combination).
_SEV_W_DARK     = 20.0
_SEV_W_BLUR     = 20.0
_SEV_W_CONTRAST = 20.0
_SEV_W_COLOR    = 20.0
_SEV_W_BG       = 20.0


# ╔══════════════════════════════════════════════════════════════════════
#  Mask-centric quality metrics (v2 — Nov 2026)
#  The image-side metrics (brightness/blur/contrast/color/bg) were tripping
#  the same "blurry → sharpen" branch every time. The metrics below score
#  the MASK ITSELF so the explainer can name the actual defect.
# ╚══════════════════════════════════════════════════════════════════════

# Thresholds — calibrated against the v2-ai-spine smoke fixtures.
_THR_COVERAGE_LO      = 0.001   # below this = effectively empty
_THR_COVERAGE_HI      = 0.95    # above this = over-segmented (mask whole frame)
_THR_FRAGMENTATION    = 3       # > N disconnected components = fragmented
_THR_HOLES_FRACTION   = 0.005   # holes >0.5 % of mask area = holey
_THR_JAGGEDNESS       = 2.5     # perimeter² / (4π·area) > 2.5 = very irregular
_THR_BIMODAL_SOFT     = 0.02    # < 2 % of pixels in mid-alpha = fully binary
_THR_EDGE_IOU         = 0.20    # boundary↔image-gradient IoU below this = misaligned
_THR_TRUNCATION       = 0.20    # >20 % of mask perimeter on frame border = cut off

_SEV_W_COVERAGE       = 14.0
_SEV_W_FRAGMENT       = 12.0
_SEV_W_HOLES          = 10.0
_SEV_W_JAGGED         =  8.0
_SEV_W_BIMODAL        =  8.0
_SEV_W_EDGE_IOU       = 14.0
_SEV_W_TRUNCATION     =  6.0


def _connected_components_count(binary_np) -> int:
    """Count 4-connected components in a HxW uint8 binary mask."""
    if HAS_CV2:
        n, _ = cv2.connectedComponents(binary_np, connectivity=4)
        return max(0, int(n) - 1)  # subtract background label
    try:
        from scipy.ndimage import label as _label  # type: ignore
        _, n = _label(binary_np > 0)
        return int(n)
    except Exception:
        return 1 if binary_np.any() else 0


def _holes_fraction(binary_np) -> float:
    """Return (interior-hole area) / (mask area), clamped to [0,1].

    Mask must be HxW uint8 in {0,1}. Critically, the floodFill seed is chosen
    from a *background* pixel on the frame border — seeding at (0,0)
    silently fails whenever the corner happens to lie inside the mask.
    """
    if not HAS_CV2:
        return 0.0
    np = __import__("numpy")
    inv = (binary_np == 0).astype("uint8")
    h, w = inv.shape
    # Locate any background border pixel (mask==0 on row 0 / row h-1 / col 0 / col w-1).
    seed = None
    for x in range(w):
        if inv[0, x]:
            seed = (x, 0); break
        if inv[h - 1, x]:
            seed = (x, h - 1); break
    if seed is None:
        for y in range(h):
            if inv[y, 0]:
                seed = (0, y); break
            if inv[y, w - 1]:
                seed = (w - 1, y); break
    if seed is None:
        # No background on the border at all → the mask either fills the
        # frame or wraps every edge; there is no exterior to flood from, so
        # the concept of "interior holes" is undefined here.
        return 0.0
    ff = inv.copy()
    mask_pad = np.zeros((h + 2, w + 2), dtype="uint8")
    cv2.floodFill(ff, mask_pad, seed, 2)
    holes = (ff == 1)
    mask_area = float(binary_np.sum())
    if mask_area < 1.0:
        return 0.0
    return min(float(holes.sum()) / mask_area, 1.0)


def _perimeter_pixels(binary_np) -> float:
    """4-connected perimeter (# of mask pixels with at least one bg neighbour)."""
    np = __import__("numpy")
    m = binary_np > 0
    if not m.any():
        return 0.0
    # Shift each direction and find pixels whose neighbour is bg
    up    = np.pad(m, ((1, 0), (0, 0)), mode="constant")[:-1, :]
    down  = np.pad(m, ((0, 1), (0, 0)), mode="constant")[1:, :]
    left  = np.pad(m, ((0, 0), (1, 0)), mode="constant")[:, :-1]
    right = np.pad(m, ((0, 0), (0, 1)), mode="constant")[:, 1:]
    boundary = m & ~(up & down & left & right)
    return float(boundary.sum())


def _compute_mask_quality(image: torch.Tensor, mask: torch.Tensor) -> dict:
    """Compute the mask-centric metric bank. Returns a dict of (B,) tensors."""
    import math
    import numpy as np
    device = _get_device(image)
    dtype = image.dtype
    B, H, W = mask.shape

    coverage      = torch.zeros(B, device=device, dtype=dtype)
    fragmentation = torch.zeros(B, device=device, dtype=dtype)
    holes_frac    = torch.zeros(B, device=device, dtype=dtype)
    jaggedness    = torch.zeros(B, device=device, dtype=dtype)
    bimodality    = torch.zeros(B, device=device, dtype=dtype)
    edge_iou      = torch.zeros(B, device=device, dtype=dtype)
    truncation    = torch.zeros(B, device=device, dtype=dtype)

    # Gradient magnitude for edge-IoU.
    luma = _compute_luminance(image)
    sx = _SOBEL_X.to(device=device, dtype=dtype)
    sy = _SOBEL_Y.to(device=device, dtype=dtype)
    gx = F.conv2d(luma.unsqueeze(1), sx, padding=1).squeeze(1)
    gy = F.conv2d(luma.unsqueeze(1), sy, padding=1).squeeze(1)
    grad_mag = (gx.pow(2) + gy.pow(2)).sqrt()

    mask_ring = _get_mask_edge_ring(mask, ring_width=2)

    for i in _PB.track(range(B), B, "MaskFailure"):
        _IC.check()
        m = mask[i]
        m_np = (m.cpu().numpy() > 0.5).astype(np.uint8)
        area = float(m_np.sum())
        total = float(H * W)

        # 1. coverage
        coverage[i] = area / total if total > 0 else 0.0

        # 2. fragmentation — # of connected components
        fragmentation[i] = float(_connected_components_count(m_np))

        # 3. interior holes
        holes_frac[i] = _holes_fraction(m_np)

        # 4. boundary jaggedness — isoperimetric quotient inverted
        if area > 4.0:
            P = _perimeter_pixels(m_np)
            jaggedness[i] = (P * P) / (4.0 * math.pi * area) if area > 0 else 0.0
        else:
            P = 0.0

        # 5. alpha bimodality — fraction of soft (mid-alpha) pixels
        soft = ((m > 0.05) & (m < 0.95)).float().mean().item()
        bimodality[i] = soft

        # 6. edge-alignment IoU between mask ring and top-30% gradient pixels
        ring = mask_ring[i] > 0.5
        if ring.sum() > 1:
            g = grad_mag[i]
            # threshold at 70th percentile of gradient
            try:
                thr = torch.quantile(g.flatten(), 0.7).item()
            except Exception:
                thr = float(g.mean().item())
            high_grad = g >= thr
            inter = (ring & high_grad).sum().item()
            union = (ring | high_grad).sum().item()
            edge_iou[i] = (inter / union) if union > 0 else 0.0

        # 7. truncation — fraction of mask *perimeter* lying on the frame border.
        # Dividing by mask area (the v1 formulation) made thin shapes score
        # tiny and half-frame masks score ~0, missing every real truncation.
        if area > 0 and P > 0.0:
            border = (m_np[0, :].sum() + m_np[-1, :].sum()
                      + m_np[:, 0].sum() + m_np[:, -1].sum())
            truncation[i] = min(float(border) / P, 1.0)

    return {
        "coverage": coverage,
        "fragmentation": fragmentation,
        "holes_frac": holes_frac,
        "jaggedness": jaggedness,
        "bimodality": bimodality,
        "edge_iou": edge_iou,
        "truncation": truncation,
    }


def _mask_quality_findings(mq: dict) -> list[tuple[str, str]]:
    """Translate metric values into (issue_key, advice) pairs (deduped, mean across batch)."""
    out: list[tuple[str, str]] = []
    cov = mq["coverage"].mean().item()
    frag = mq["fragmentation"].mean().item()
    holes = mq["holes_frac"].mean().item()
    jag = mq["jaggedness"].mean().item()
    bim = mq["bimodality"].mean().item()
    eiou = mq["edge_iou"].mean().item()
    trunc = mq["truncation"].mean().item()

    if cov < _THR_COVERAGE_LO:
        out.append((
            "empty_mask",
            "Mask is effectively empty (<0.1% coverage). The segmenter "
            "did not find the subject. Re-prompt with a different point/box "
            "or switch to a text-prompt model (GroundingDINO/Florence2).",
        ))
    elif cov > _THR_COVERAGE_HI:
        out.append((
            "over_segmented",
            "Mask covers >95% of the frame — the segmenter selected the "
            "background. Invert the mask or supply a negative prompt / "
            "background point.",
        ))
    if frag > _THR_FRAGMENTATION:
        out.append((
            "fragmented",
            f"Mask is split into {int(frag)} disconnected components. "
            "Run hole-fill + morphological close in Mask Refiner, or filter "
            "to the largest blob.",
        ))
    if holes > _THR_HOLES_FRACTION and cov > 0.02 and frag < 100:
        out.append((
            "interior_holes",
            f"Mask has interior holes covering {holes*100:.1f}% of its area. "
            "Enable hole_fill in Mask Refiner (subject_class=object) before "
            "matting.",
        ))
    if jag > _THR_JAGGEDNESS:
        out.append((
            "jagged_boundary",
            f"Boundary is highly irregular (isoperimetric quotient "
            f"{jag:.2f} — round shape = 1.0). Apply guided filter + light "
            "feather, or re-run matter with a wider trimap unknown band.",
        ))
    if bim < _THR_BIMODAL_SOFT:
        out.append((
            "fully_binary",
            "Mask is fully binary (no soft alpha). Hair, motion blur and "
            "translucency cannot be preserved — re-run with a matter "
            "(ViTMatte) instead of a hard segmenter.",
        ))
    if eiou < _THR_EDGE_IOU:
        out.append((
            "edge_misalignment",
            f"Mask boundary aligns poorly with image edges (IoU "
            f"{eiou:.2f}). The mask is bleeding into the background — enable "
            "auto_edge_lock in Mask Refiner with the correct subject_class "
            "(face/garment/object).",
        ))
    if trunc > _THR_TRUNCATION:
        out.append((
            "subject_truncated",
            f"{trunc*100:.1f}% of the mask sits on the frame border — the "
            "subject is cut off. Crop / pad the input or accept truncation "
            "in your composite.",
        ))
    return out


def _compute_severity(
    brightness: torch.Tensor,
    blur: torch.Tensor,
    boundary_contrast: torch.Tensor,
    color_confusion: torch.Tensor,
    bg_complexity: torch.Tensor,
    mq: dict | None = None,
) -> float:
    """Compute a 0-100 severity score (mean across batch).

    Each metric contributes up to its named weight (see ``_SEV_W_*`` above).
    The thresholds (0.15, 50.0, 0.05, 0.1, 0.3) match ``_THRESHOLD_*``
    used in the explanation builder, keeping numerics consistent.
    """
    # Average across batch
    b = brightness.mean().item()
    bl = blur.mean().item()
    bc = boundary_contrast.mean().item()
    cc = color_confusion.mean().item()
    bg = bg_complexity.mean().item()

    score = 0.0
    # Dark scene penalty (only if dark)
    score += _SEV_W_DARK     * max(0.0, 1.0 - min(b / 0.15, 1.0))
    # Blur penalty
    score += _SEV_W_BLUR     * max(0.0, 1.0 - min(bl / 50.0, 1.0))
    # Low boundary contrast penalty
    score += _SEV_W_CONTRAST * max(0.0, 1.0 - min(bc / 0.05, 1.0))
    # Color confusion penalty
    score += _SEV_W_COLOR    * max(0.0, 1.0 - min(cc / 0.1, 1.0))
    # Background complexity penalty
    score += _SEV_W_BG       * min(bg / 0.3, 1.0)

    # Mask-centric penalties (v2 additions)
    if mq is not None:
        cov = mq["coverage"].mean().item()
        frag = mq["fragmentation"].mean().item()
        holes = mq["holes_frac"].mean().item()
        jag = mq["jaggedness"].mean().item()
        bim = mq["bimodality"].mean().item()
        eiou = mq["edge_iou"].mean().item()
        trunc = mq["truncation"].mean().item()

        # Coverage extremes (empty OR full)
        if cov < _THR_COVERAGE_LO:
            score += _SEV_W_COVERAGE
        elif cov > _THR_COVERAGE_HI:
            score += _SEV_W_COVERAGE * 0.7
        # Fragmentation — saturating ramp
        if frag > _THR_FRAGMENTATION:
            score += _SEV_W_FRAGMENT * min((frag - _THR_FRAGMENTATION) / 6.0, 1.0)
        # Holes — gated by coverage + non-fragmented (noise spam guard)
        if holes > _THR_HOLES_FRACTION and cov > 0.02 and frag < 100:
            score += _SEV_W_HOLES * min(holes / 0.05, 1.0)
        # Jagged boundary
        if jag > _THR_JAGGEDNESS:
            score += _SEV_W_JAGGED * min((jag - _THR_JAGGEDNESS) / 4.0, 1.0)
        # Bimodality (only flag if mask exists)
        if cov > _THR_COVERAGE_LO and bim < _THR_BIMODAL_SOFT:
            score += _SEV_W_BIMODAL
        # Edge alignment (only flag if mask exists)
        if cov > _THR_COVERAGE_LO and eiou < _THR_EDGE_IOU:
            score += _SEV_W_EDGE_IOU * (1.0 - eiou / _THR_EDGE_IOU)
        # Truncation
        if trunc > _THR_TRUNCATION:
            score += _SEV_W_TRUNCATION * min((trunc - _THR_TRUNCATION) / 0.5, 1.0)

    return round(min(max(score, 0.0), 100.0), 2)


# ══════════════════════════════════════════════════════════════════════
#  Explanation + suggested method
# ══════════════════════════════════════════════════════════════════════

_THRESHOLD_DARK = 0.15
_THRESHOLD_BLUR = 50.0
_THRESHOLD_CONTRAST = 0.05
_THRESHOLD_COLOR = 0.1
_THRESHOLD_BG = 0.3


def _build_explanation(
    brightness: torch.Tensor,
    blur: torch.Tensor,
    boundary_contrast: torch.Tensor,
    color_confusion: torch.Tensor,
    bg_complexity: torch.Tensor,
    severity: float,
    B: int, H: int, W: int,
    mq: dict | None = None,
) -> str:
    """Build a detailed, per-frame explanation string with actionable advice."""
    lines = []
    lines.append(f"[MEC] Mask Failure Analysis — {B} frame(s), {H}x{W}")
    lines.append(f"Overall severity: {severity:.1f}/100")
    lines.append("")

    issues_found = []

    for i in _PB.track(range(B), B, "MaskFailure"):
        _IC.check()
        frame_prefix = f"Frame {i}" if B > 1 else "Image"
        frame_issues = []

        b = brightness[i].item()
        bl = blur[i].item()
        bc = boundary_contrast[i].item()
        cc = color_confusion[i].item()
        bg = bg_complexity[i].item()

        lines.append(f"--- {frame_prefix} ---")
        lines.append(f"  Brightness:         {b:.4f}" + (" âš  DARK SCENE" if b < _THRESHOLD_DARK else " ✓"))
        lines.append(f"  Blur score:         {bl:.2f}" + (" âš  BLURRY" if bl < _THRESHOLD_BLUR else " ✓"))
        lines.append(f"  Boundary contrast:  {bc:.4f}" + (" âš  LOW CONTRAST" if bc < _THRESHOLD_CONTRAST else " ✓"))
        lines.append(f"  Color confusion:    {cc:.4f}" + (" âš  COLORS TOO SIMILAR" if cc < _THRESHOLD_COLOR else " ✓"))
        lines.append(f"  BG complexity:      {bg:.4f}" + (" âš  BUSY BACKGROUND" if bg > _THRESHOLD_BG else " ✓"))

        if b < _THRESHOLD_DARK:
            frame_issues.append("dark_scene")
        if bl < _THRESHOLD_BLUR:
            frame_issues.append("blurry")
        if bc < _THRESHOLD_CONTRAST:
            frame_issues.append("low_boundary_contrast")
        if cc < _THRESHOLD_COLOR:
            frame_issues.append("color_confusion")
        if bg > _THRESHOLD_BG:
            frame_issues.append("busy_background")

        if frame_issues:
            lines.append(f"  Issues: {', '.join(frame_issues)}")
        else:
            lines.append("  No significant issues detected.")

        issues_found.extend(frame_issues)
        lines.append("")

    # Actionable advice
    unique_issues = list(dict.fromkeys(issues_found))

    # ----- v2: mask-centric metrics block -----
    mq_findings: list[tuple[str, str]] = []
    if mq is not None:
        lines.append("--- Mask quality (batch mean) ---")
        lines.append(f"  Coverage:           {mq['coverage'].mean().item()*100:.2f}%")
        lines.append(f"  Components:         {mq['fragmentation'].mean().item():.0f}")
        lines.append(f"  Interior holes:     {mq['holes_frac'].mean().item()*100:.2f}% of mask area")
        lines.append(f"  Boundary jagged:    {mq['jaggedness'].mean().item():.2f} (round=1.00)")
        lines.append(f"  Soft-alpha pixels:  {mq['bimodality'].mean().item()*100:.2f}%")
        lines.append(f"  Edge-grad IoU:      {mq['edge_iou'].mean().item():.2f}")
        lines.append(f"  Frame-border share: {mq['truncation'].mean().item()*100:.2f}% of mask")
        lines.append("")
        mq_findings = _mask_quality_findings(mq)

    if unique_issues or mq_findings:
        lines.append("=== Recommendations ===")
        if "dark_scene" in unique_issues:
            lines.append("• Dark scene: Try boosting image brightness/gamma before masking, or use a model with low-light capability (e.g., SAM2 with auto-point prompts).")
        if "blurry" in unique_issues:
            lines.append("• Blurry image: Apply sharpening before mask generation, or use a matting model (ViTMatte) that handles soft edges.")
        if "low_boundary_contrast" in unique_issues:
            lines.append("• Low boundary contrast: The subject blends with background at the edge. Use trimap-based matting (ViTMatte) or manual boundary refinement.")
        if "color_confusion" in unique_issues:
            lines.append("• Color confusion at boundary: Subject and background have similar colors. Use text-prompt segmentation (GroundingDINO/Florence2) or manual point prompts.")
        if "busy_background" in unique_issues:
            lines.append("• Busy background: High edge density behind subject. Use a model with strong figure-ground separation (RMBG, BiRefNet) or hierarchical SAM2 segmenter.")
        for _, advice in mq_findings:
            lines.append(f"• {advice}")
    else:
        lines.append("=== No significant issues detected ===")
        lines.append("The image+mask combination appears healthy. If masking still fails, consider increasing model resolution or using manual prompts.")

    return "\n".join(lines)


def _suggest_method(
    brightness: torch.Tensor,
    blur: torch.Tensor,
    boundary_contrast: torch.Tensor,
    color_confusion: torch.Tensor,
    bg_complexity: torch.Tensor,
    mq: dict | None = None,
) -> str:
    """Suggest the best masking method based on which conditions triggered."""
    # Average across batch
    b = brightness.mean().item()
    bl = blur.mean().item()
    bc = boundary_contrast.mean().item()
    cc = color_confusion.mean().item()
    bg = bg_complexity.mean().item()

    suggestions: list[str] = []

    # Mask-centric routing wins over image-centric routing because a defect
    # in the mask itself dictates the next tool, regardless of the image.
    if mq is not None:
        cov   = mq["coverage"].mean().item()
        frag  = mq["fragmentation"].mean().item()
        holes = mq["holes_frac"].mean().item()
        jag   = mq["jaggedness"].mean().item()
        bim   = mq["bimodality"].mean().item()
        eiou  = mq["edge_iou"].mean().item()
        trunc = mq["truncation"].mean().item()

        if cov < _THR_COVERAGE_LO:
            suggestions.append("Re-segment with GroundingDINO + SAM2 text prompt (mask is empty)")
        elif cov > _THR_COVERAGE_HI:
            suggestions.append("Invert mask or supply negative background point (mask covers whole frame)")
        if eiou < _THR_EDGE_IOU and cov > _THR_COVERAGE_LO:
            suggestions.append("Mask Refiner with auto_edge_lock=True, subject_class=face/garment/object (mask is bleeding)")
        if bim < _THR_BIMODAL_SOFT and cov > _THR_COVERAGE_LO:
            suggestions.append("ViTMatte / Matte-Anything for soft alpha (hard mask cannot hold hair / motion blur)")
        if holes > _THR_HOLES_FRACTION:
            suggestions.append("Mask Refiner: enable_hole_fill + morph_close (interior holes detected)")
        if frag > _THR_FRAGMENTATION:
            suggestions.append("Largest-blob filter or morph_close in Mask Refiner (mask is fragmented)")
        if jag > _THR_JAGGEDNESS:
            suggestions.append("Guided filter + light feather in Mask Refiner (jagged boundary)")
        if trunc > _THR_TRUNCATION:
            suggestions.append("Crop/pad input before masking (subject is cut off by frame border)")

    has_dark = b < _THRESHOLD_DARK
    has_blur = bl < _THRESHOLD_BLUR
    has_low_contrast = bc < _THRESHOLD_CONTRAST
    has_color_confusion = cc < _THRESHOLD_COLOR
    has_busy_bg = bg > _THRESHOLD_BG

    if has_color_confusion or has_low_contrast:
        suggestions.append("ViTMatte (trimap-based matting handles boundary ambiguity)")
    if has_dark:
        suggestions.append("SAM2 with auto-point prompts (robust in low light)")
    if has_blur:
        suggestions.append("ViTMatte (handles soft/blurry edges via alpha matting)")
    if has_busy_bg:
        suggestions.append("RMBG or BiRefNet (strong figure-ground separation)")

    if not suggestions:
        return "auto (no significant issues — any segmentation method should work)"

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in suggestions:
        key = s.split("(")[0].strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return " → ".join(unique[:4]) if unique else "auto"


# ══════════════════════════════════════════════════════════════════════
#  Node class
# ══════════════════════════════════════════════════════════════════════

class MaskFailureExplainerMEC:
    """Diagnose why a mask failed and suggest fixes.

    Runs five pure-tensor analysis metrics on the image+mask pair and
    produces an explanation, a problem-regions heatmap, a severity score,
    and a suggested masking method.
    """

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image(s) — (B,H,W,C) float32 [0,1].",
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask to diagnose — (B,H,W) float32 [0,1]. Can be from any segmentation method.",
                }),
            },
            "optional": {
                "ring_width": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Width in pixels of the boundary ring used for contrast/color analysis.",
                }),
                "blur_threshold": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "Laplacian variance threshold below which the image is considered blurry.",
                }),
                "brightness_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mean brightness threshold below which the scene is considered dark.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("explanation", "problem_regions_mask", "severity_score", "suggested_method")
    OUTPUT_TOOLTIPS = (
        "Human-readable diagnosis explaining likely failure causes.",
        "Heatmap mask highlighting regions most likely to be problematic.",
        "Overall severity score in [0, 100] (higher means more issues).",
        "Suggested masking method or refinement to try next.",
    )
    FUNCTION = "analyze"
    CATEGORY = "C2C/Diagnostics"
    DESCRIPTION = (
        "Diagnose why a mask might be failing. Analyzes brightness, blur, "
        "boundary contrast, color confusion, and background complexity. "
        "Outputs a detailed explanation, problem heatmap, severity score, "
        "and suggested masking method."
    )

    def analyze(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        ring_width: int = 5,
        blur_threshold: float = 50.0,
        brightness_threshold: float = 0.15,
    ) -> tuple[str, torch.Tensor, float, str]:
        with _PB.session("MaskFailure"):
            return self._analyze_impl(image, mask, ring_width, blur_threshold,
                                      brightness_threshold)

    def _analyze_impl(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        ring_width: int = 5,
        blur_threshold: float = 50.0,
        brightness_threshold: float = 0.15,
    ) -> tuple[str, torch.Tensor, float, str]:
        try:
            B, H, W, C = image.shape

            # Ensure mask matches image spatial dims
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != B:
                # Broadcast single mask to batch
                if mask.shape[0] == 1:
                    mask = mask.expand(B, -1, -1)
                else:
                    raise ValueError(
                        f"[MEC] Mask batch size {mask.shape[0]} does not match "
                        f"image batch size {B}."
                    )
            if mask.shape[1] != H or mask.shape[2] != W:
                mask = F.interpolate(
                    mask.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(1)

            # Move kernels to same device as image
            device = _get_device(image)
            mask = mask.to(device=device, dtype=image.dtype)

            # Downsample very large inputs for the heavy analytical kernels.
            # The metrics are scale-invariant in spirit; the heatmap is upsampled back.
            ANALYZE_MAX_EDGE = 2048
            long_edge = max(H, W)
            if long_edge > ANALYZE_MAX_EDGE:
                scale = ANALYZE_MAX_EDGE / float(long_edge)
                aH = max(1, int(round(H * scale)))
                aW = max(1, int(round(W * scale)))
                image_a = F.interpolate(
                    image.permute(0, 3, 1, 2), size=(aH, aW),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()
                mask_a = F.interpolate(
                    mask.unsqueeze(1), size=(aH, aW),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            else:
                image_a, mask_a = image, mask

            # ── Run all 5 analysis metrics ────────────────────────────
            brightness = _compute_brightness(image_a)                          # (B,)
            blur = _compute_blur_score(image_a)                                # (B,)
            boundary_contrast = _compute_boundary_contrast(image_a, mask_a)    # (B,)
            color_confusion = _compute_boundary_color_confusion(image_a, mask_a) # (B,)
            bg_complexity = _compute_bg_complexity(image_a, mask_a)             # (B,)

            # ── Mask-centric quality metrics (v2) ─────────────────────
            mq = _compute_mask_quality(image_a, mask_a)

            # ── Severity score ────────────────────────────────────────
            severity = _compute_severity(
                brightness, blur, boundary_contrast, color_confusion, bg_complexity,
                mq=mq,
            )

            # ── Explanation string ────────────────────────────────────
            explanation = _build_explanation(
                brightness, blur, boundary_contrast, color_confusion,
                bg_complexity, severity, B, H, W, mq=mq,
            )

            # ── Problem regions heatmap ───────────────────────────────
            heatmap = _build_problem_heatmap(
                image_a, mask_a, brightness, blur,
                boundary_contrast, color_confusion, bg_complexity,
            )
            # Upsample heatmap to match the original image size if we downsampled.
            if heatmap.shape[-2:] != (H, W):
                heatmap = F.interpolate(
                    heatmap.unsqueeze(1), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)

            # ── Suggested method ──────────────────────────────────────
            method = _suggest_method(
                brightness, blur, boundary_contrast, color_confusion, bg_complexity,
                mq=mq,
            )

            return (explanation, heatmap, severity, method)

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
