# FILE: nodes/propainter_temporal_inpaint.py
# FEATURE: P1 — ProPainterTemporalMEC (temporal video inpaint between InpaintCropProMEC and InpaintCompositeMEC)
# INTEGRATES WITH: nodes/inpaint_suite.py (consumes stitch_data, feeds InpaintCompositeMEC)
"""
ProPainterTemporalMEC — drop-in temporal inpainter.

Pipeline inside `inpaint_temporal`:
    1. RAFT bidirectional flow on the cropped clip.
    2. Bidirectional consistency check (|fwd + bwd| > thr -> drop).
    3. RecurrentFlowCompleteNet to fill flow holes (no Gaussian blur).
    4. Sliding-window InpaintGenerator pass (subvideo_length).
    5. Per-frame Reinhard / LAB color match (masked).
    6. Boundary blend with stitch_blend_mask_crop (BOUNDARY_SEAM defense).
    7. Inter-frame SSIM check on the filled region (warning only).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .propainter_bridge import (
    HAS_PROPAINTER,
    ProPainterMissingError,
    from_propainter_video,
    get_device,
    load_models,
    require_propainter,
    to_propainter_mask,
    to_propainter_video,
)

log = logging.getLogger("MEC.propainter_temporal")


# =====================================================================
# RAFT bidirectional flow with consistency check
# =====================================================================
@torch.no_grad()
def _compute_bidirectional_flow(raft, video_bchw: torch.Tensor,
                                iters: int = 20,
                                consistency_thr: float = 1.5,
                                chunk: int = 16,
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    video_bchw: (B, C, H, W) in [-1, 1] — RAFT expects [0, 255], we rescale.
    Returns:
        fwd:   (B-1, 2, H, W)
        bwd:   (B-1, 2, H, W)
        valid: (B-1, 1, H, W)  1 where bidirectional flow agrees

    RAFT keeps a 4-level correlation pyramid resident per pair so peak VRAM
    grows linearly with the number of pairs processed in one forward pass.
    On 8 GB cards a 720p video with B=80 frames OOMs immediately. We chunk
    the consecutive pairs in groups of `chunk` and concatenate the results.
    """
    from .propainter_bridge import InputPadder  # vendored
    if InputPadder is None:
        raise RuntimeError("InputPadder unavailable — vendored ProPainter missing at third_party/ProPainter/")

    # RAFT internally normalises by /255; it expects [0,255] floats.
    rgb = ((video_bchw.float() + 1.0) * 127.5).clamp(0.0, 255.0)
    padder = InputPadder(rgb.shape)
    rgb_p = padder.pad(rgb)[0]
    a_all = rgb_p[:-1]
    b_all = rgb_p[1:]

    fwd_chunks: List[torch.Tensor] = []
    bwd_chunks: List[torch.Tensor] = []
    n_pairs = a_all.shape[0]
    chunk = max(1, int(chunk))
    for s in range(0, n_pairs, chunk):
        e = min(s + chunk, n_pairs)
        _, fwd_c = raft(a_all[s:e], b_all[s:e], iters=iters, test_mode=True)
        _, bwd_c = raft(b_all[s:e], a_all[s:e], iters=iters, test_mode=True)
        fwd_chunks.append(fwd_c)
        bwd_chunks.append(bwd_c)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fwd = padder.unpad(torch.cat(fwd_chunks, dim=0))
    bwd = padder.unpad(torch.cat(bwd_chunks, dim=0))

    # Consistency: warp bwd by fwd; |fwd + bwd_warped| should be ~0 if vectors agree.
    H, W = video_bchw.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=fwd.device, dtype=fwd.dtype),
        torch.arange(W, device=fwd.device, dtype=fwd.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(fwd.shape[0], -1, -1, -1)
    sample = grid + fwd
    sample_norm = torch.stack(
        (2.0 * sample[:, 0] / max(W - 1, 1) - 1.0,
         2.0 * sample[:, 1] / max(H - 1, 1) - 1.0), dim=-1,
    )
    bwd_warped = F.grid_sample(bwd, sample_norm, mode="bilinear",
                               padding_mode="border", align_corners=True)
    diff = (fwd + bwd_warped).pow(2).sum(dim=1, keepdim=True).sqrt()
    valid = (diff < consistency_thr).to(fwd.dtype)
    return fwd, bwd, valid


# =====================================================================
# RFC-based flow hole filling (NO blur)
# =====================================================================
@torch.no_grad()
def _complete_flow(flow_complete_net, fwd: torch.Tensor, bwd: torch.Tensor,
                   masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    fwd, bwd: (B-1, 2, H, W)
    masks:    (B,   1, H, W) — 1 = inpaint region
    Returns completed (1, B-1, 2, H, W) for fwd & bwd.

    RAFT runs in fp32 even when the rest of the bundle is fp16, so cast the
    flow tensors to whatever the RFC weights are before forwarding.
    """
    target_dtype = next(flow_complete_net.parameters()).dtype
    fwd_5 = fwd.to(target_dtype).unsqueeze(0)  # (1, B-1, 2, H, W)
    bwd_5 = bwd.to(target_dtype).unsqueeze(0)
    flows_bi = (fwd_5, bwd_5)
    pred_bi, _ = flow_complete_net.forward_bidirect_flow(
        flows_bi, masks.to(target_dtype).unsqueeze(0)
    )
    return pred_bi[0], pred_bi[1]


# =====================================================================
# InpaintGenerator sliding window
# =====================================================================
@torch.no_grad()
def _inpaint_window(model, frames_in: torch.Tensor, masks_in: torch.Tensor,
                    fwd: torch.Tensor, bwd: torch.Tensor,
                    neighbor_stride: int, ref_stride: int,
                    subvideo_length: int) -> torch.Tensor:
    """
    frames_in: (1, T, C, H, W) [-1, 1]
    masks_in:  (1, T, 1, H, W) {0, 1}
    fwd / bwd: (1, T-1, 2, H, W)
    Returns:    (1, T, C, H, W) [-1, 1]
    """
    _, T, _, H, W = frames_in.shape
    out = frames_in.clone()
    masked = frames_in * (1.0 - masks_in)

    pos = 0
    while pos < T:
        end = min(pos + subvideo_length, T)
        # Neighbour indices: every neighbor_stride frames inside [pos, end).
        local = list(range(pos, end))
        # Reference frames: globally spaced by ref_stride for long-range context.
        # Exclude any ref that overlaps with the local window — upstream rule.
        refs = [r for r in range(0, T, ref_stride) if r not in local]
        # Upstream ProPainter: forward returns ONLY the first len(local) frames.
        # So indices MUST be local-first, then refs.
        idx = local + refs
        if not idx:
            pos = end
            continue

        idx_t = torch.tensor(idx, device=frames_in.device)
        sub_frames = masked.index_select(1, idx_t)
        sub_masks = masks_in.index_select(1, idx_t)

        # Sub-flows: ProPainter expects (T_local - 1) flows aligned with the
        # consecutive LOCAL frame pairs only.
        if len(local) >= 2:
            local_pair_idx = torch.tensor(local[:-1], device=fwd.device)
            sub_fwd = fwd.index_select(1, local_pair_idx)
            sub_bwd = bwd.index_select(1, local_pair_idx)
        else:
            # Single-frame edge case: fabricate a 1-pair zero flow.
            sub_fwd = fwd[:, :1] * 0
            sub_bwd = bwd[:, :1] * 0

        try:
            # Upstream ProPainter forward signature:
            #   forward(masked_frames, completed_flows, masks_in,
            #           masks_updated, num_local_frames, ...)
            # `completed_flows` is a tuple (fwd_5d, bwd_5d). We pass the same
            # mask for `masks_in` and `masks_updated` (skipping the
            # img_propagation pre-pass; acceptable for short clips).
            target_dtype = next(model.parameters()).dtype
            sub_frames_t = sub_frames.to(target_dtype)
            sub_masks_t = sub_masks.to(target_dtype)
            sub_fwd_t = sub_fwd.to(target_dtype)
            sub_bwd_t = sub_bwd.to(target_dtype)
            log.debug("[propainter] window pos=%d end=%d sub_frames=%s sub_fwd=%s sub_masks=%s len(local)=%d",
                      pos, end, tuple(sub_frames_t.shape), tuple(sub_fwd_t.shape),
                      tuple(sub_masks_t.shape), len(local))
            sub_out = model(
                sub_frames_t,
                (sub_fwd_t, sub_bwd_t),
                sub_masks_t,
                sub_masks_t,
                len(local),
            )
        except TypeError:
            # Some forks accept positional flows.
            sub_out = model(
                sub_frames, sub_fwd, sub_bwd, sub_masks, len(local)
            )

        # Place back only the LOCAL window slots. sub_out has shape
        # (1, len(local), 3, H, W) — its frame index `i` corresponds to local[i].
        for i, src_idx in enumerate(local):
            if pos <= src_idx < end and i < sub_out.shape[1]:
                out[:, src_idx] = sub_out[:, i]

        pos += max(neighbor_stride, 1) * subvideo_length // max(neighbor_stride, 1)
        # Advance by exactly subvideo_length minus a neighbor overlap to avoid drift.
        pos = end - (neighbor_stride if end < T else 0)
        if end >= T:
            break

    # Composite: keep original outside mask, ProPainter result inside.
    return frames_in * (1.0 - masks_in) + out * masks_in


# =====================================================================
# Reinhard / LAB-transfer color matching (masked)
# =====================================================================
def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb (..., 3) [0,1] -> LAB (..., 3) approximate, fast."""
    r, g, b = rgb.unbind(-1)
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    eps = 1e-6
    fx = torch.where(x > 0.008856, x.clamp_min(eps).pow(1.0 / 3.0), 7.787 * x + 16.0 / 116.0)
    fy = torch.where(y > 0.008856, y.clamp_min(eps).pow(1.0 / 3.0), 7.787 * y + 16.0 / 116.0)
    fz = torch.where(z > 0.008856, z.clamp_min(eps).pow(1.0 / 3.0), 7.787 * z + 16.0 / 116.0)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.stack((L, a, bb), dim=-1)


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    L, a, b = lab.unbind(-1)
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    x = torch.where(fx ** 3 > 0.008856, fx ** 3, (fx - 16.0 / 116.0) / 7.787)
    y = torch.where(fy ** 3 > 0.008856, fy ** 3, (fy - 16.0 / 116.0) / 7.787)
    z = torch.where(fz ** 3 > 0.008856, fz ** 3, (fz - 16.0 / 116.0) / 7.787)
    r = 3.240479 * x - 1.537150 * y - 0.498535 * z
    g = -0.969256 * x + 1.875991 * y + 0.041556 * z
    bb = 0.055648 * x - 0.204043 * y + 1.057311 * z
    return torch.stack((r, g, bb), dim=-1).clamp(0.0, 1.0)


def _color_match(filled: torch.Tensor, original: torch.Tensor,
                 mask: torch.Tensor, mode: str) -> torch.Tensor:
    """
    filled / original: (B, H, W, 3) float [0,1]
    mask:              (B, H, W)    float [0,1] (1 = inpaint region)
    """
    if mode == "none":
        return filled
    out = filled.clone()
    # Region statistics computed OUTSIDE the inpaint region (where original is trusted).
    ref_mask = (1.0 - mask).unsqueeze(-1).clamp(0.0, 1.0)
    if mode == "reinhard":
        for f in range(out.shape[0]):
            ref = original[f]
            tgt = out[f]
            w = ref_mask[f]
            denom = w.sum() + 1e-6
            ref_mean = (ref * w).sum(dim=(0, 1)) / denom
            ref_std = (((ref - ref_mean) ** 2 * w).sum(dim=(0, 1)) / denom).sqrt() + 1e-6
            tgt_mean = (tgt * w).sum(dim=(0, 1)) / denom
            tgt_std = (((tgt - tgt_mean) ** 2 * w).sum(dim=(0, 1)) / denom).sqrt() + 1e-6
            shifted = (tgt - tgt_mean) * (ref_std / tgt_std) + ref_mean
            inside = mask[f].unsqueeze(-1)
            out[f] = tgt * (1.0 - inside) + shifted.clamp(0.0, 1.0) * inside
        return out
    if mode == "lab_transfer":
        ref_lab = _rgb_to_lab(original)
        tgt_lab = _rgb_to_lab(out)
        for f in range(out.shape[0]):
            w = ref_mask[f]
            denom = w.sum() + 1e-6
            r_mean = (ref_lab[f] * w).sum(dim=(0, 1)) / denom
            r_std = (((ref_lab[f] - r_mean) ** 2 * w).sum(dim=(0, 1)) / denom).sqrt() + 1e-6
            t_mean = (tgt_lab[f] * w).sum(dim=(0, 1)) / denom
            t_std = (((tgt_lab[f] - t_mean) ** 2 * w).sum(dim=(0, 1)) / denom).sqrt() + 1e-6
            shifted_lab = (tgt_lab[f] - t_mean) * (r_std / t_std) + r_mean
            shifted = _lab_to_rgb(shifted_lab)
            inside = mask[f].unsqueeze(-1)
            out[f] = out[f] * (1.0 - inside) + shifted * inside
        return out
    return filled


# =====================================================================
# Inter-frame SSIM (filled region) — warning only
# =====================================================================
def _ssim_pair(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """Plain Wang-Bovik SSIM, single window 11x11, masked region only."""
    win = 11
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    k = torch.ones(1, 1, win, win, device=a.device, dtype=a.dtype) / (win * win)
    k = k.expand(3, 1, win, win)
    mu_a = F.conv2d(a, k, padding=win // 2, groups=3)
    mu_b = F.conv2d(b, k, padding=win // 2, groups=3)
    var_a = F.conv2d(a * a, k, padding=win // 2, groups=3) - mu_a ** 2
    var_b = F.conv2d(b * b, k, padding=win // 2, groups=3) - mu_b ** 2
    cov = F.conv2d(a * b, k, padding=win // 2, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
               ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    m = mask.unsqueeze(0).unsqueeze(0)
    if m.sum() < 1.0:
        return 1.0
    return float((ssim_map.mean(dim=1, keepdim=True) * m).sum() / (m.sum() * 1 + 1e-6))


def _temporal_ssim_check(filled: torch.Tensor, mask: torch.Tensor) -> List[Tuple[int, int, float]]:
    out: List[Tuple[int, int, float]] = []
    for i in range(filled.shape[0] - 1):
        joint = ((mask[i] + mask[i + 1]) > 0.5).to(filled.dtype)
        if joint.sum() < 16:
            continue
        s = _ssim_pair(filled[i], filled[i + 1], joint)
        if s < 0.85:
            out.append((i, i + 1, s))
    return out


# =====================================================================
# Boundary cleanup with stitch_blend_mask_crop
# =====================================================================
def _boundary_blend(filled: torch.Tensor, original_crop: torch.Tensor,
                    blend_mask: torch.Tensor) -> torch.Tensor:
    """
    filled / original_crop: (B, h, w, 3)
    blend_mask:             (B, h, w)  in [0, 1]; 0 outside, 1 deep inside fill
    """
    if blend_mask is None:
        return filled
    bm = blend_mask.to(filled.device).to(filled.dtype).clamp(0.0, 1.0).unsqueeze(-1)
    if bm.shape[1:3] != filled.shape[1:3]:
        bm = F.interpolate(bm.permute(0, 3, 1, 2), size=filled.shape[1:3],
                           mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return original_crop * (1.0 - bm) + filled * bm


# =====================================================================
# The node
# =====================================================================
class ProPainterTemporalMEC:
    DESCRIPTION = (
        "Temporal video inpaint via ProPainter (RAFT + RFC + InpaintGenerator). "
        "Drop in between InpaintCropProMEC and InpaintCompositeMEC."
    )
    CATEGORY = "MaskEditControl/Inpaint"
    FUNCTION = "inpaint_temporal"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("inpainted_image", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "masks":           ("MASK",),
                "stitch_data":     ("STITCH_DATA",),
                "neighbor_stride": ("INT",     {"default": 5,  "min": 1, "max": 20}),
                "ref_stride":      ("INT",     {"default": 10, "min": 1, "max": 50}),
                "raft_iter":       ("INT",     {"default": 20, "min": 1, "max": 100}),
                "subvideo_length": ("INT",     {"default": 30, "min": 8, "max": 300,
                                                "tooltip": "Frames per InpaintGenerator window. Lower = less VRAM. 8GB cards: 20-30. 12GB: 40-60. 24GB: 80+."}),
                "raft_chunk":      ("INT",     {"default": 16, "min": 1, "max": 64,
                                                "tooltip": "Frame-pairs per RAFT forward pass. Lower if you OOM on flow estimation. 8GB: 8-16. 24GB: 32+."}),
                "use_half":        ("BOOLEAN", {"default": True}),
                "blend_boundary":  ("BOOLEAN", {"default": True}),
                "color_match_mode": (["none", "reinhard", "lab_transfer"],
                                     {"default": "reinhard"}),
            },
        }

    # ---------- main entry ----------
    def inpaint_temporal(self, images: torch.Tensor, masks: torch.Tensor,
                         stitch_data: Dict, neighbor_stride: int, ref_stride: int,
                         raft_iter: int, subvideo_length: int,
                         use_half: bool,
                         blend_boundary: bool, color_match_mode: str,
                         raft_chunk: int = 16):
        if not HAS_PROPAINTER:
            require_propainter()  # raises ProPainterMissingError with install hint
        # Free any other models (SD checkpoint, VAE, encoders) sitting in VRAM
        # before we load the ProPainter bundle. ProPainter (RAFT + RFC +
        # InpaintGenerator) needs ~4-5GB on its own; on an 8GB card you OOM
        # immediately if a Wan/Flux checkpoint is still resident.
        try:
            import comfy.model_management as _mm  # type: ignore
            _mm.unload_all_models()
            _mm.soft_empty_cache()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        t0 = time.time()
        info_lines: List[str] = []

        B, H, W, C = images.shape
        if masks.dim() == 4:
            masks = masks.squeeze(-1)
        if masks.shape[0] != B:
            # Single-mask broadcast across video.
            masks = masks[:1].expand(B, -1, -1).contiguous()
        info_lines.append(f"frames={B} HxW={H}x{W} subvideo={subvideo_length}")

        models = load_models(half=use_half)
        dev = models.device

        try:
            video_pp = to_propainter_video(images, dev, models.half)
            mask_pp = to_propainter_mask(masks, dev, models.half)

            # Flatten the leading (1, B, ...) for RAFT consumption.
            video_bchw = video_pp.squeeze(0)  # (B, C, H, W)
            mask_b1hw = mask_pp.squeeze(0)    # (B, 1, H, W)

            # ---- RAFT bidirectional flow + consistency ----
            fwd, bwd, valid = _compute_bidirectional_flow(
                models.raft, video_bchw, iters=raft_iter, chunk=raft_chunk,
            )
            valid_ratio = float(valid.mean())
            info_lines.append(f"raft_valid={valid_ratio:.3f}")

            # Mask out inconsistent vectors so RFC re-fills them (no blur).
            inconsistent = (1.0 - valid)
            fwd = fwd * valid + 0.0 * inconsistent  # zero where invalid; RFC re-paints
            bwd = bwd * valid + 0.0 * inconsistent

            # ---- Recurrent flow completion ----
            # _complete_flow returns 5D tensors (1, B-1, 2, H, W) already.
            fwd_c, bwd_c = _complete_flow(models.flow_complete, fwd, bwd, mask_b1hw)

            # ---- InpaintGenerator sliding window ----
            filled_pp = _inpaint_window(
                models.inpaint, video_pp, mask_pp, fwd_c, bwd_c,
                neighbor_stride=neighbor_stride,
                ref_stride=ref_stride,
                subvideo_length=subvideo_length,
            )

            filled_bhwc = from_propainter_video(filled_pp)  # (B, H, W, C) cpu

            # ---- Color match (Reinhard / LAB) ----
            filled_bhwc = _color_match(filled_bhwc, images.cpu(),
                                       masks.cpu(), color_match_mode)
            info_lines.append(f"color_match={color_match_mode}")

            # ---- Boundary blend with stitch_blend_mask_crop ----
            if blend_boundary:
                bm = stitch_data.get("stitch_blend_mask_crop")
                if bm is not None:
                    filled_bhwc = _boundary_blend(filled_bhwc, images.cpu(), bm)
                    info_lines.append("boundary_blend=on")
                else:
                    info_lines.append("boundary_blend=requested_but_missing")

            # ---- Inter-frame SSIM diagnostic ----
            ssim_drops = _temporal_ssim_check(filled_bhwc, masks.cpu())
            if ssim_drops:
                info_lines.append(
                    "TEMPORAL_BLEED warnings: " + ", ".join(
                        f"f{a}->{b}={s:.3f}" for a, b, s in ssim_drops[:8]
                    )
                )
            else:
                info_lines.append("ssim_ok")

        finally:
            # Free intermediate GPU buffers (models stay cached for next run).
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        info_lines.append(f"elapsed={time.time() - t0:.2f}s")
        return (filled_bhwc.contiguous(), " | ".join(info_lines))


NODE_CLASS_MAPPINGS = {"ProPainterTemporalMEC": ProPainterTemporalMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"ProPainterTemporalMEC": "ProPainter Temporal Inpaint (MEC)"}
