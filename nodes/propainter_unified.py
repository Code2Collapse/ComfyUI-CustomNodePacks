"""
ProPainterMEC — unified ProPainter node.
========================================

Absorbs the five legacy ProPainter nodes into one mode-switched node:

    mode = "temporal"      -> ProPainterTemporalMEC      (video inpaint inside InpaintCrop/Stitch flow)
    mode = "remove"        -> ProPainterRemoveMEC        (object/wire/plate removal)
    mode = "stitch"        -> ProPainterStitchMEC        (flow-aware composite of inpainted crop)
    mode = "stitch_refine" -> ProPainterStitchRefineMEC  (seam ring refinement of already-stitched output)
    mode = "flow"          -> FlowRefineMEC              (RAFT optical flow viz / warp / consistency)

L7.92 (owner 2026-10-12): temporal / remove take an appended `backend` widget. "propainter" (the default and the
value every older save runs with) is the unchanged ProPainter-only path; "auto" uses ProPainter when the user installed
it (code + weights, S-Lab License 1.0 non-commercial), otherwise the model-free OpenCV fallback; "auto_no_propainter"
never uses ProPainter; "opencv_flow" forces the fallback (_opencv_flow_fill: Farneback flow + flow completion +
temporal propagation + Telea, labelled APPROXIMATE). Nothing is downloaded. Audit: docs/research/L7.92_inpaint_audit.md.

Helper classes are kept on disk as importable Python classes; only this
single node is registered with ComfyUI. Lazy imports keep load-time cost
near zero.

Outputs are a fixed superset so downstream wires never break across modes:

    (IMAGE image_out, MASK mask_out, IMAGE aux_image, MASK aux_mask, STRING info)

For each mode the slots carry:

    temporal      : (inpainted, inpaint_mask,  difference x4,           changed,      info)
    remove        : (filled,    fill_mask,     difference x4,           changed,      info)
    stitch        : (stitched,  repaint_mask,  difference x4,           changed,      info)
    stitch_refine : (refined,   ring_mask,     difference x4,           changed,      info)
    flow          : (warped,    consistency,   flow_field_rgb,          unreliable,   info)

L9.39: aux_image / aux_mask were zeros (aux_mask "reserved") and temporal's mask_out was zeros. Now they are QC:
"difference x4" = |output - input| amplified 4x, "changed" = pixels the node altered (more than 1.5/255 in any
channel), measured against the mode's input (images / the stitch canvas / stitched_image); "unreliable" = 1 -
consistency. The info line says how much changed and how much of that lies OUTSIDE mask_out (spill / seams). When
the output and its input differ in size, both stay zeros and info says why.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from ._is_changed_util import hash_args_and_kwargs

log = logging.getLogger("MEC.ProPainter")


# ────────────────────────────────────────────────────────────────────────
# Lazy helper getters (no import cost unless the mode is actually used).
# ────────────────────────────────────────────────────────────────────────
def _get_temporal_cls():
    from .propainter_temporal_inpaint import ProPainterTemporalMEC
    return ProPainterTemporalMEC


def _get_remove_cls():
    from .propainter_stitch_suite import ProPainterRemoveMEC
    return ProPainterRemoveMEC


def _get_stitch_cls():
    from .propainter_stitch_suite import ProPainterStitchMEC
    return ProPainterStitchMEC


def _get_stitch_refine_cls():
    from .propainter_stitch_suite import ProPainterStitchRefineMEC
    return ProPainterStitchRefineMEC


def _get_flow_cls():
    from .propainter_flow_refine import FlowRefineMEC
    return FlowRefineMEC


_MODES = ["temporal", "remove", "stitch", "stitch_refine", "flow"]

# L7.92: the `backend` widget (appended last; "propainter" = the behaviour of every save made before it).
# Mirrors propainter_bridge.INPAINT_BACKENDS / OPENCV_FALLBACK_LABEL (pinned equal by tests/test_l792_backend_select.py):
# not imported here, so registering the node never imports the bridge (which probes for a ProPainter checkout).
_BACKENDS = ["propainter", "auto", "auto_no_propainter", "opencv_flow"]
# Inputs appended after the A28 pin (tests/test_a28_cnp_a_layout.py excludes these, as for MaskToolsMEC's extras).
L792_APPENDED_INPUTS = ("backend",)
_CV_LABEL = ("APPROXIMATE model-free fallback (Farneback flow + temporal propagation + Telea; "
             "not ProPainter quality)")


def _select_backend(requested: str):
    from .propainter_bridge import select_inpaint_backend
    return select_inpaint_backend(requested)


def _check_interrupt() -> None:
    try:
        import comfy.model_management as _mm  # type: ignore
    except ImportError:
        return
    _mm.throw_exception_if_processing_interrupted()


def _complete_flow_nc(flow, invalid, sigmas=(4.0, 16.0, 64.0, 256.0)):
    """Fill `flow` (h,w,2) where `invalid` (h,w bool) by normalised convolution with growing Gaussians: the hole gets
    the smooth motion of its surroundings (background motion, not the masked object's). Model-free."""
    import cv2
    import numpy as np
    if not invalid.any():
        return flow
    valid = (~invalid).astype(np.float32)
    out = flow.copy()
    out[invalid] = 0.0
    todo = invalid.copy()
    for s in sigmas:
        if not todo.any():
            break
        w = cv2.GaussianBlur(valid, (0, 0), s)
        num = cv2.GaussianBlur(flow * valid[..., None], (0, 0), s)
        ok = todo & (w > 1e-4)
        out[ok] = num[ok] / w[ok][:, None]
        todo &= ~ok
    return out  # anything still in `todo` keeps zero motion


def _opencv_flow_fill(images: torch.Tensor, mask: torch.Tensor, max_flow_side: int = 512, telea_radius: int = 5):
    """L7.92 model-free video fill: Farneback flow (at <= max_flow_side) -> flow completion inside the hole ->
    forward and backward temporal propagation from the original frames (the source nearest in time wins) -> Telea
    for what no frame shows. Outside the hole the frames are returned bit-exact. APPROXIMATE: no learned prior, so
    it cannot invent structure the way ProPainter can.

    images (B,H,W,3) float 0-1, mask (B,H,W) float -> (filled (B,H,W,3) float32 cpu, stats dict)."""
    import cv2
    import numpy as np
    with torch.no_grad():
        img = images[..., :3].detach().float().cpu().clamp(0, 1).numpy()
        hole = (mask.detach().float().cpu() > 0.5).numpy()
    B, H, W, _ = img.shape
    stats = {"frames": int(B), "hole_px": int(hole.sum()), "propagated_px": 0, "telea_px": 0, "flow_res": "-"}
    if stats["hole_px"] == 0:
        return torch.from_numpy(img.copy()), stats

    known0 = ~hole
    if B > 1:
        sc = min(1.0, float(max_flow_side) / float(max(H, W)))
        fh, fw = max(8, int(round(H * sc))), max(8, int(round(W * sc)))
        stats["flow_res"] = f"{fw}x{fh}"
        gray = [cv2.resize((cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) * 255.0).astype(np.uint8), (fw, fh),
                           interpolation=cv2.INTER_AREA) for f in img]
        hole_s = [cv2.resize(h.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST) > 0 for h in hole]
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

        def flow_to(t, s):
            """Map of frame t's pixels into frame s: gray_t(x) ~ gray_s(x + flow)."""
            # The masked object is erased (Telea, at flow resolution) from BOTH frames first: otherwise its own motion
            # bleeds into the background flow around it (Farneback windows + pyramid), and the plate is pulled from
            # the wrong place (synthetic test clip, static plate: mean hole error 0.024 -> 0.0, tests/test_l792_backend_select.py).
            both = (hole_s[t] | hole_s[s]).astype(np.uint8)
            ga = cv2.inpaint(gray[t], both * 255, 3.0, cv2.INPAINT_TELEA) if both.any() else gray[t]
            gb = cv2.inpaint(gray[s], both * 255, 3.0, cv2.INPAINT_TELEA) if both.any() else gray[s]
            f = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 21, 3, 7, 1.5, 0)
            invalid = cv2.dilate(both, kern) > 0
            f = _complete_flow_nc(f, invalid)
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
            f[..., 0] *= W / float(fw)
            f[..., 1] *= H / float(fh)
            return gx + f[..., 0], gy + f[..., 1]

        def run_pass(order, step):
            fill = img.copy()
            known = known0.astype(np.float32)
            age = np.where(known0, 0.0, np.inf).astype(np.float32)
            for t in order:
                _check_interrupt()
                s = t - step
                mx, my = flow_to(t, s)
                src_k = cv2.remap(known[s], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                src_a = cv2.remap(np.where(np.isfinite(age[s]), age[s], 1e9).astype(np.float32), mx, my,
                                  cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1e9)
                src_c = cv2.remap(fill[s], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                take = hole[t] & (src_k >= 0.999)
                fill[t][take] = src_c[take]
                known[t][take] = 1.0
                age[t][take] = src_a[take] + 1.0
            return fill, age

        fill_f, age_f = run_pass(range(1, B), 1)
        fill_b, age_b = run_pass(range(B - 2, -1, -1), -1)
        use_b = age_b < age_f
        fill = np.where(use_b[..., None], fill_b, fill_f)
        age = np.minimum(age_f, age_b)
    else:
        fill = img.copy()
        age = np.where(known0, 0.0, np.inf).astype(np.float32)

    unknown = hole & ~np.isfinite(age)
    stats["propagated_px"] = int((hole & np.isfinite(age)).sum())
    stats["telea_px"] = int(unknown.sum())
    for t in range(B):
        if not unknown[t].any():
            continue
        _check_interrupt()
        u8 = (np.clip(fill[t], 0, 1) * 255.0 + 0.5).astype(np.uint8)
        tel = cv2.inpaint(u8, unknown[t].astype(np.uint8) * 255, float(telea_radius), cv2.INPAINT_TELEA)
        fill[t][unknown[t]] = tel[unknown[t]].astype(np.float32) / 255.0
    out = np.where(hole[..., None], np.clip(fill, 0, 1), img).astype(np.float32)
    return torch.from_numpy(out), stats


def _fallback_info(stats: dict) -> str:
    h = max(1, stats["hole_px"])
    return (f"{_CV_LABEL} | frames={stats['frames']} flow_res={stats['flow_res']} | hole filled from other frames "
            f"{100.0 * stats['propagated_px'] / h:.1f}%, Telea (spatial guess) {100.0 * stats['telea_px'] / h:.1f}%")


def _uniform(values, what):
    vals = [int(v) for v in values]
    if any(v != vals[0] for v in vals):
        raise ValueError(f"[ProPainterMEC] the STITCHER's {what} changes from frame to frame ({vals[:6]}...); "
                         "ProPainter stitch needs one crop size for the whole clip. Use a fixed crop size in "
                         "Inpaint Crop Pro.")
    return vals[0]


def _as_stitch_data(sd):
    """A22 L9.136: Inpaint Crop Pro outputs a STITCHER (per-frame lists); ProPainter's stitch modes read the
    STITCH_DATA v2/v3 contract, which nothing produced, so the input could never be wired. A STITCHER is converted
    here; anything else passes through unchanged.

    STITCHER: canvas_image[j] (H,W,C) = the frame inside its canvas; cropped_to_canvas_* = where the crop sits in the
    canvas; canvas_to_orig_* = where the original frame sits in the canvas; cropped_mask_for_blend[j] = the blend
    mask in crop space. STITCH_DATA: stacked canvas, one crop size (v3: per-frame crop offsets), original_mask in
    original-frame space (cto_w x cto_h), stitch_blend_mask_crop in crop space."""
    if not (isinstance(sd, dict) and "cropped_to_canvas_x" in sd and "canvas_image" in sd):
        return sd
    import torch.nn.functional as F
    n = len(sd["cropped_to_canvas_x"])
    if n == 0:
        raise ValueError("[ProPainterMEC] the STITCHER holds no frames.")
    ctc_w = _uniform(sd["cropped_to_canvas_w"], "crop size (width)")
    ctc_h = _uniform(sd["cropped_to_canvas_h"], "crop size (height)")
    cto = [_uniform(sd[f"canvas_to_orig_{k}"], f"original-frame placement ({k})") for k in ("x", "y", "w", "h")]
    frames = [c.squeeze(0) if c.dim() == 4 else c for c in sd["canvas_image"]]
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("[ProPainterMEC] the STITCHER's canvases differ in size from frame to frame; ProPainter "
                         "stitch needs one canvas size for the whole clip.")
    canvas = torch.stack([f.float() for f in frames])[..., :3].contiguous()
    _, Hc, Wc, _ = canvas.shape
    offsets = [(int(x), int(y)) for x, y in zip(sd["cropped_to_canvas_x"], sd["cropped_to_canvas_y"])]
    crop_masks = []
    canvas_mask = torch.zeros(n, Hc, Wc)
    raw_masks = sd.get("cropped_mask_for_blend") or []
    for j, (x, y) in enumerate(offsets):
        if raw_masks:
            m = raw_masks[min(j, len(raw_masks) - 1)].float()
            while m.dim() > 2:
                m = m.squeeze(0) if m.shape[0] == 1 else m[..., 0]
            if tuple(m.shape) != (ctc_h, ctc_w):
                m = F.interpolate(m[None, None], size=(ctc_h, ctc_w), mode="bilinear",
                                  align_corners=False)[0, 0]
        else:
            m = torch.ones(ctc_h, ctc_w)
        crop_masks.append(m.clamp(0, 1))
        y2, x2 = min(y + ctc_h, Hc), min(x + ctc_w, Wc)
        canvas_mask[j, y:y2, x:x2] = crop_masks[-1][:y2 - y, :x2 - x]
    cx, cy, cw, ch = cto
    same = all(o == offsets[0] for o in offsets)
    return {
        "version": 2 if same else 3,
        "canvas_image": canvas,
        "ctc_x": offsets[0][0], "ctc_y": offsets[0][1], "ctc_w": ctc_w, "ctc_h": ctc_h,
        "cto_x": cx, "cto_y": cy, "cto_w": cw, "cto_h": ch,
        "frame_offsets": None if same else offsets,
        "original_mask": canvas_mask[:, cy:cy + ch, cx:cx + cw].contiguous(),
        "stitch_blend_mask_crop": torch.stack(crop_masks),
        "source": "STITCHER",
    }


class ProPainterMEC:
    """Unified ProPainter dispatcher — one node, five modes."""

    CATEGORY = "C2C/Inpaint"
    FUNCTION = "execute"
    DESCRIPTION = (
        "Unified ProPainter node. Absorbs ProPainterTemporal / Remove / "
        "Stitch / StitchRefine / FlowRefine. Pick a mode and only the "
        "relevant widgets are read; others are ignored."
    )

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image_out", "mask_out", "aux_image", "aux_mask", "info")
    OUTPUT_TOOLTIPS = (
        "Primary image output (inpainted / filled / stitched / warped per mode).",
        "Primary mask output (fill mask / repaint mask / consistency per mode).",
        "QC image: |output - input| x4 (flow mode: the flow field as colour).",
        "QC mask: pixels this node changed (flow mode: where the flow is unreliable, 1 - consistency).",
        "Info string with timings, coverage, mode-specific stats.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (_MODES, {"default": "remove",
                    "tooltip": "Pick the ProPainter operation. Each mode reads a different subset of the optional inputs/widgets below."}),

                # Common knobs (used by all flow-based modes).
                "use_half":          ("BOOLEAN", {"default": True, "tooltip": "Run the flow-completion and inpainting networks in fp16 on CUDA (about half the VRAM). Ignored on CPU."}),
                "color_match_mode":  (["off", "reinhard", "lab", "lab_transfer", "none"],
                                      {"default": "reinhard",
                                       "tooltip": "Per-frame masked colour match between fill and surroundings (ignored in 'flow' mode)."}),

                # Shared ProPainter knobs.
                "raft_iter":         ("INT", {"default": 12, "min": 1, "max": 100, "tooltip": "RAFT optical-flow refinement steps. More = more accurate flow, slower."}),
                "neighbor_stride":   ("INT", {"default": 5,  "min": 1, "max": 32, "tooltip": "Step between local windows of neighbouring frames that are inpainted together. Smaller = more overlap, smoother, slower."}),
                "ref_stride":        ("INT", {"default": 10, "min": 1, "max": 64, "tooltip": "Spacing of the far-away reference frames each window also looks at. Smaller = more references (fills long occlusions better), more memory."}),
                "subvideo_length":   ("INT", {"default": 8,  "min": 2, "max": 300,
                    "tooltip": "Frames per InpaintGenerator window. 8GB cards: 8-30. 12GB: 40-60. 24GB: 80+."}),
                "raft_chunk":        ("INT", {"default": 16, "min": 1, "max": 64,
                    "tooltip": "Frame-pairs per RAFT forward pass."}),

                # ── 'temporal' specific ──────────────────────────────
                "blend_boundary":    ("BOOLEAN", {"default": True,
                    "tooltip": "[temporal] Blend the inpaint with original at the crop boundary using stitch_data."}),

                # ── 'remove' specific ────────────────────────────────
                "remove_quality":    (["fast", "balanced", "quality"],
                                     {"default": "balanced",
                                      "tooltip": "[remove] Preset that overrides raft_iter/neighbor_stride/ref_stride/subvideo_length."}),
                "remove_dilate_pixels": ("INT", {"default": 3, "min": 0, "max": 32,
                    "tooltip": "[remove] Dilate the mask by N px before filling to cover anti-aliasing."}),

                # ── 'stitch' specific ────────────────────────────────
                "boundary_band_pixels": ("INT", {"default": 12, "min": 0, "max": 96,
                    "tooltip": "[stitch] Width of the boundary band re-painted (0 = no boundary repaint)."}),
                "preserve_inpaint_center": ("BOOLEAN", {"default": True,
                    "tooltip": "[stitch] Keep generative inpaint untouched at the centre, only repaint the seam."}),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "nearest"],
                                   {"default": "lanczos",
                                    "tooltip": "[stitch] How to scale the inpainted crop to the canvas region."}),

                # ── 'stitch_refine' specific ─────────────────────────
                "ring_pixels": ("INT", {"default": 8, "min": 1, "max": 64,
                    "tooltip": "[stitch_refine] Half-width of the seam ring in pixels."}),

                # ── 'flow' specific ──────────────────────────────────
                "flow_consistency_thr": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.05,
                    "tooltip": "[flow] Forward/backward consistency threshold (pixels)."}),
            },
            "optional": {
                # Image / mask / data inputs — wired only as the mode needs.
                "images":           ("IMAGE",
                    {"tooltip": "[temporal/remove] Source frames."}),
                "masks":            ("MASK",
                    {"tooltip": "[temporal/remove] Region to inpaint."}),
                "stitch_data":      ("STITCHER,STITCH_DATA",
                    {"tooltip": "[temporal/stitch/stitch_refine] The stitcher output of Inpaint Crop Pro."}),
                "inpainted_image":  ("IMAGE",
                    {"tooltip": "[stitch] Crop-sized generative inpaint output."}),
                "stitched_image":   ("IMAGE",
                    {"tooltip": "[stitch_refine] Already-stitched canvas image (output of stitch)."}),
                "mask_override":    ("MASK",
                    {"tooltip": "[stitch_refine] Optional explicit canvas-space mask (overrides stitch_data)."}),
                "frame_a":          ("IMAGE",
                    {"tooltip": "[flow] First frame for optical-flow computation."}),
                "frame_b":          ("IMAGE",
                    {"tooltip": "[flow] Second frame for optical-flow computation."}),
                "flow_mask":        ("MASK",
                    {"tooltip": "[flow] Optional mask restricting the consistency visualisation."}),
                # L7.92: appended last so every earlier save keeps its values; the default is the old behaviour.
                "backend":          (list(_BACKENDS), {"default": "propainter",
                    "tooltip": "[temporal/remove] Which engine fills the hole.\n"
                               "propainter: ProPainter only (fails when you have not installed it) - the old "
                               "behaviour.\nauto: ProPainter when you installed it (code + weights; S-Lab License "
                               "1.0, non-commercial only), otherwise the OpenCV fallback.\nauto_no_propainter: "
                               "licence-safe for commercial work, never ProPainter.\nopencv_flow: model-free OpenCV "
                               "fallback (optical flow + temporal propagation + Telea) - APPROXIMATE, not ProPainter "
                               "quality. Nothing is ever downloaded. The info output says which engine ran and why."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, mode, use_half, color_match_mode, raft_iter, neighbor_stride,
                   ref_stride, subvideo_length, raft_chunk, blend_boundary,
                   remove_quality, remove_dilate_pixels, boundary_band_pixels,
                   preserve_inpaint_center, upscale_method, ring_pixels,
                   flow_consistency_thr,
                   images=None, masks=None, stitch_data=None, inpainted_image=None,
                   stitched_image=None, mask_override=None, frame_a=None, frame_b=None,
                   flow_mask=None, **kwargs):
        if str(kwargs.get("backend", "propainter")).startswith("auto"):
            # L7.92: the engine "auto" picks depends on what is installed: installing ProPainter re-runs the node.
            try:
                kwargs["_resolved_backend"] = _select_backend(kwargs["backend"])[0]
            except Exception:
                kwargs["_resolved_backend"] = "unavailable"
        return hash_args_and_kwargs(
            mode, use_half, color_match_mode, raft_iter, neighbor_stride,
            ref_stride, subvideo_length, raft_chunk, blend_boundary,
            remove_quality, remove_dilate_pixels, boundary_band_pixels,
            preserve_inpaint_center, upscale_method, ring_pixels,
            flow_consistency_thr,
            images, masks, stitch_data, inpainted_image,
            stitched_image, mask_override, frame_a, frame_b, flow_mask, **kwargs,
        )

    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalise_cm(cm: str) -> str:
        """Different sub-nodes accept slightly different colour-match labels."""
        if cm in ("off", "none"):
            return "off"
        if cm == "lab_transfer":
            return "lab"
        return cm

    @staticmethod
    def _zero_image_like(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            return torch.zeros_like(t)
        if t.dim() == 3:  # MASK
            return torch.zeros(t.shape[0], t.shape[1], t.shape[2], 3, dtype=t.dtype)
        return torch.zeros(1, 8, 8, 3, dtype=torch.float32)

    @staticmethod
    def _zero_mask_like(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            return torch.zeros(t.shape[0], t.shape[1], t.shape[2], dtype=t.dtype)
        if t.dim() == 3:
            return torch.zeros_like(t)
        return torch.zeros(1, 8, 8, dtype=torch.float32)

    @staticmethod
    def _qc(out: torch.Tensor, ref: Optional[torch.Tensor], region: torch.Tensor):
        """(difference x4 image, changed mask, info text) of `out` against `ref`; zeros when sizes differ."""
        if not (isinstance(ref, torch.Tensor) and ref.dim() == 4 and out.dim() == 4
                and tuple(ref.shape[1:3]) == tuple(out.shape[1:3])):
            why = "no input to compare" if ref is None else \
                f"input {tuple(ref.shape[1:3])} vs output {tuple(out.shape[1:3])}"
            return ProPainterMEC._zero_image_like(out), ProPainterMEC._zero_mask_like(out), f"qc: off ({why})"
        r = ref[..., :3].to(out.device, out.dtype)
        if r.shape[0] != out.shape[0]:
            r = r[torch.arange(out.shape[0]) % r.shape[0]]
        diff = (out[..., :3] - r).abs()
        changed = (diff.amax(-1) > 1.5 / 255.0).to(out.dtype)
        txt = f"qc: changed {100.0 * float(changed.mean()):.2f}% of pixels"
        if isinstance(region, torch.Tensor) and region.dim() == 3 and tuple(region.shape) == tuple(changed.shape):
            outside = changed * (region.to(changed.device, changed.dtype) <= 0.5)
            txt += f", {100.0 * float(outside.mean()):.2f}% outside mask_out"
        return (diff * 4.0).clamp(0, 1).contiguous(), changed.contiguous(), txt

    @staticmethod
    def _mask_like(masks: Optional[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
        """The input mask at the output's size and frame count (zeros when there is none)."""
        if not isinstance(masks, torch.Tensor):
            return ProPainterMEC._zero_mask_like(out)
        m = masks if masks.dim() == 3 else masks.reshape(-1, *masks.shape[-2:])
        m = m.to(out.device, torch.float32)
        if tuple(m.shape[1:]) != tuple(out.shape[1:3]):
            m = torch.nn.functional.interpolate(m.unsqueeze(1), size=tuple(out.shape[1:3]), mode="bilinear",
                                                align_corners=False).squeeze(1)
        if m.shape[0] != out.shape[0]:
            m = m[torch.arange(out.shape[0]) % m.shape[0]]
        return m.clamp(0, 1).contiguous()

    # ────────────────────────────────────────────────────────────────
    def execute(self, mode: str,
                use_half: bool, color_match_mode: str,
                raft_iter: int, neighbor_stride: int, ref_stride: int,
                subvideo_length: int, raft_chunk: int,
                blend_boundary: bool,
                remove_quality: str, remove_dilate_pixels: int,
                boundary_band_pixels: int, preserve_inpaint_center: bool,
                upscale_method: str,
                ring_pixels: int,
                flow_consistency_thr: float,
                # optional tensors
                images: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None,
                stitch_data: Optional[Dict[str, Any]] = None,
                inpainted_image: Optional[torch.Tensor] = None,
                stitched_image: Optional[torch.Tensor] = None,
                mask_override: Optional[torch.Tensor] = None,
                frame_a: Optional[torch.Tensor] = None,
                frame_b: Optional[torch.Tensor] = None,
                flow_mask: Optional[torch.Tensor] = None,
                backend: str = "propainter"):

        cm = self._normalise_cm(color_match_mode)
        stitch_data = _as_stitch_data(stitch_data)

        # L7.92: backend choice for the two filling modes. "propainter" (every pre-L7.92 save) skips this block and
        # runs the unchanged code below, info text included.
        backend = str(backend or "propainter")
        prefix = ""
        if mode in ("temporal", "remove") and backend != "propainter":
            if images is None or masks is None:
                raise ValueError(f"[ProPainterMEC] mode='{mode}' requires 'images' and 'masks'.")
            resolved, why = _select_backend(backend)
            prefix = f"backend={resolved} (requested {backend}: {why}) | "
            if resolved == "opencv_flow":
                return self._run_opencv(mode, images, masks, stitch_data, cm, bool(blend_boundary),
                                        int(remove_dilate_pixels), prefix)

        result = self._run_mode(mode, cm, stitch_data, use_half, raft_iter, neighbor_stride, ref_stride,
                                subvideo_length, raft_chunk, blend_boundary, remove_quality, remove_dilate_pixels,
                                boundary_band_pixels, preserve_inpaint_center, upscale_method, ring_pixels,
                                flow_consistency_thr, images, masks, inpainted_image, stitched_image,
                                mask_override, frame_a, frame_b, flow_mask)
        if prefix:
            info = result[4]
            head, sep, tail = info.partition(" | ")
            result = (*result[:4], f"{head} | {prefix}{tail}" if sep else f"{info} | {prefix.rstrip(' |')}")
        return result

    def _run_opencv(self, mode, images, masks, stitch_data, cm, blend_boundary, dilate_px, prefix):
        """L7.92 model-free fallback for temporal / remove (see _opencv_flow_fill)."""
        import time
        from .propainter_stitch_suite import _binary_dilate, _ensure_bhw_mask
        from .propainter_temporal_inpaint import _boundary_blend, _color_match
        t0 = time.time()
        if images.dim() != 4 or images.shape[-1] < 3:
            raise ValueError(f"[ProPainterMEC] IMAGE [B,H,W,3] expected, got {tuple(images.shape)}")
        images = images[..., :3]
        B, H, W, _ = images.shape
        m = _ensure_bhw_mask(masks, B, H, W)
        notes = []
        if mode == "remove" and dilate_px > 0:
            m = _binary_dilate(m, dilate_px)
            notes.append(f"dilate={dilate_px}")
        hole = (m > 0.5).float()
        if float(hole.mean()) < 1e-5:
            out = images.float().cpu()
            notes.append("NOOP_empty_mask")
        else:
            out, stats = _opencv_flow_fill(images, hole)
            notes.append(_fallback_info(stats))
            cm_t = {"off": "none", "lab": "lab_transfer"}.get(cm, cm)
            if cm_t not in ("none", "reinhard", "lab_transfer"):
                cm_t = "reinhard"
            if cm_t != "none":
                matched = _color_match(out, images.float().cpu(), hole, cm_t)
                out = torch.where(hole.unsqueeze(-1) > 0.5, matched, out)
            notes.append(f"color_match={cm_t}")
            if mode == "temporal" and blend_boundary:
                bm = (stitch_data or {}).get("stitch_blend_mask_crop") if isinstance(stitch_data, dict) else None
                if bm is not None:
                    out = _boundary_blend(out, images.float().cpu(), bm)
                    notes.append("boundary_blend=on")
                else:
                    notes.append("boundary_blend=requested_but_missing")
        out = out.clamp(0, 1).contiguous()
        notes.append(f"elapsed={time.time() - t0:.2f}s")
        region = hole if mode == "remove" else self._mask_like(masks, out)
        diff, changed, qc = self._qc(out, images, region)
        return (out, region, diff, changed, f"mode={mode} | {prefix}{' | '.join(notes)} | {qc}")

    def _run_mode(self, mode, cm, stitch_data, use_half, raft_iter, neighbor_stride, ref_stride, subvideo_length,
                  raft_chunk, blend_boundary, remove_quality, remove_dilate_pixels, boundary_band_pixels,
                  preserve_inpaint_center, upscale_method, ring_pixels, flow_consistency_thr, images, masks,
                  inpainted_image, stitched_image, mask_override, frame_a, frame_b, flow_mask):
        if mode == "temporal":
            if images is None or masks is None:
                raise ValueError("[ProPainterMEC] mode='temporal' requires 'images' and 'masks'.")
            node = _get_temporal_cls()()
            # A22 L9.136: "off" means no colour match (it was mapped to reinhard, so "off" still ran one)
            cm_t = "none" if cm == "off" else (
                "lab_transfer" if cm == "lab" else cm)
            if cm_t not in ("none", "reinhard", "lab_transfer"):
                cm_t = "reinhard"
            out_img, info = node.inpaint_temporal(
                images=images, masks=masks,
                stitch_data=(stitch_data or {}),
                neighbor_stride=int(neighbor_stride),
                ref_stride=int(ref_stride),
                raft_iter=int(raft_iter),
                subvideo_length=int(subvideo_length),
                use_half=bool(use_half),
                blend_boundary=bool(blend_boundary),
                color_match_mode=cm_t,
                raft_chunk=int(raft_chunk),
            )
            region = self._mask_like(masks, out_img)
            diff, changed, qc = self._qc(out_img, images, region)
            return (out_img, region, diff, changed, f"mode=temporal | {info} | {qc}")

        if mode == "remove":
            if images is None or masks is None:
                raise ValueError("[ProPainterMEC] mode='remove' requires 'images' and 'masks'.")
            node = _get_remove_cls()()
            out_img, used_mask, info = node.remove(
                images=images, masks=masks,
                quality=remove_quality,
                use_half=bool(use_half),
                dilate_mask_pixels=int(remove_dilate_pixels),
                color_match_mode=cm,
            )
            diff, changed, qc = self._qc(out_img, images, used_mask)
            return (out_img, used_mask, diff, changed, f"mode=remove | {info} | {qc}")

        if mode == "stitch":
            if stitch_data is None or inpainted_image is None:
                raise ValueError("[ProPainterMEC] mode='stitch' requires 'stitch_data' and 'inpainted_image'.")
            node = _get_stitch_cls()()
            out_img, used_mask, info = node.stitch(
                stitch_data=stitch_data,
                inpainted_image=inpainted_image,
                boundary_band_pixels=int(boundary_band_pixels),
                preserve_inpaint_center=bool(preserve_inpaint_center),
                raft_iter=int(raft_iter),
                neighbor_stride=int(neighbor_stride),
                ref_stride=int(ref_stride),
                subvideo_length=int(subvideo_length),
                use_half=bool(use_half),
                color_match_mode=cm,
                upscale_method=upscale_method,
            )
            canvas = stitch_data.get("canvas_image") if isinstance(stitch_data, dict) else None
            diff, changed, qc = self._qc(out_img, canvas, used_mask)
            return (out_img, used_mask, diff, changed, f"mode=stitch | {info} | {qc}")

        if mode == "stitch_refine":
            if stitched_image is None:
                raise ValueError("[ProPainterMEC] mode='stitch_refine' requires 'stitched_image'.")
            node = _get_stitch_refine_cls()()
            out_img, used_mask, info = node.refine(
                stitched_image=stitched_image,
                stitch_data=(stitch_data or {}),
                ring_pixels=int(ring_pixels),
                raft_iter=int(raft_iter),
                neighbor_stride=int(neighbor_stride),
                ref_stride=int(ref_stride),
                subvideo_length=int(subvideo_length),
                use_half=bool(use_half),
                color_match_mode=cm,
                mask_override=mask_override,
            )
            diff, changed, qc = self._qc(out_img, stitched_image, used_mask)
            return (out_img, used_mask, diff, changed, f"mode=stitch_refine | {info} | {qc}")

        if mode == "flow":
            if frame_a is None or frame_b is None:
                raise ValueError("[ProPainterMEC] mode='flow' requires 'frame_a' and 'frame_b'.")
            node = _get_flow_cls()()
            flow_rgb, warped, consistency = node.refine_flow(
                frame_a=frame_a, frame_b=frame_b,
                iters=int(raft_iter),
                consistency_thr=float(flow_consistency_thr),
                mask=flow_mask,
            )
            unreliable = (1.0 - consistency.float()).clamp(0, 1) if isinstance(consistency, torch.Tensor) \
                else self._zero_mask_like(warped)
            return (warped, consistency,
                    flow_rgb, unreliable,
                    f"mode=flow | iters={int(raft_iter)} consistency_thr={float(flow_consistency_thr):.3f}")

        raise ValueError(f"[ProPainterMEC] unknown mode {mode!r}; expected one of {_MODES}.")


NODE_CLASS_MAPPINGS = {"ProPainterMEC": ProPainterMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPainterMEC": "ProPainter — Temporal / Remove / Stitch / Refine / Flow",
}


# A28 (ORDERS 29, docs/audit/A28/PARAM_CNP_A.md "ProPainterMEC"): which controls show when, read by js/_c2c_node_layout.js.
# Display only: every input keeps its name, order, type and default, is saved and reaches the node's function as before.
# Each mode shows only what it reads (propainter_unified.py:339-436).
try:
    from ._layout_decl import declare_layout, merge, sections
except ImportError:  # pragma: no cover - loaded outside the pack
    declare_layout = None
if declare_layout is not None:
    _PP_NOT_FLOW = ["temporal", "remove", "stitch", "stitch_refine"]
    _PP_WINDOWED = ["temporal", "stitch", "stitch_refine"]
    declare_layout(ProPainterMEC, {
        "use_half": {"c2c_show_if": {"mode": _PP_NOT_FLOW}},
        "color_match_mode": {"c2c_show_if": {"mode": _PP_NOT_FLOW}},
        "raft_iter": {"c2c_section": "Flow", "c2c_show_if": {"mode": ["temporal", "stitch", "stitch_refine", "flow"]}},
        "neighbor_stride": {"c2c_section": "Windowing", "c2c_show_if": {"mode": _PP_WINDOWED}},
        "ref_stride": {"c2c_section": "Windowing", "c2c_show_if": {"mode": _PP_WINDOWED}},
        "subvideo_length": {"c2c_section": "Windowing", "c2c_show_if": {"mode": _PP_WINDOWED}},
        "raft_chunk": {"c2c_section": "Windowing", "c2c_show_if": {"mode": ["temporal"]}},
        "blend_boundary": {"c2c_show_if": {"mode": ["temporal"]}},
        "remove_quality": {"c2c_show_if": {"mode": ["remove"]}},
        "remove_dilate_pixels": {"c2c_show_if": {"mode": ["remove"]}},
        "boundary_band_pixels": {"c2c_show_if": {"mode": ["stitch"]}},
        "preserve_inpaint_center": {"c2c_show_if": {"mode": ["stitch"]}},
        "upscale_method": {"c2c_show_if": {"mode": ["stitch"]}},
        "ring_pixels": {"c2c_show_if": {"mode": ["stitch_refine"]}},
        "flow_consistency_thr": {"c2c_show_if": {"mode": ["flow"]}},
        "backend": {"c2c_show_if": {"mode": ["temporal", "remove"]}},
    })
