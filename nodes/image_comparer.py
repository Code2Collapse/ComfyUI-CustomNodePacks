"""ImageComparerMEC – Nuke-style A/B comparer with HDR / bit-depth-aware diff.

Modes (frontend):
  - Slider  : draggable wipe between A and B
  - Overlay : alpha-blended overlay
  - Diff    : pre-computed difference visualization (server-side, full float
              precision so 8-bit vs 16-bit differences are preserved)

Server pre-computes the diff in float32 at the *full* precision of the input
tensors (which ComfyUI delivers as float32 [0,1] regardless of the original
file bit depth). It applies user-controlled `diff_gain` and `diff_gamma`
*before* tone-mapping to 8-bit for transport, so subtle <1/255 differences
between an 8-bit image and a re-quantised 16-bit version remain visible.
"""

import os
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths


_DIFF_MODES = ["absolute", "signed", "luminance", "per-channel"]


class ImageComparerMEC:
    """Nuke-style A/B comparer with HDR-aware diff.

    The diff view is computed server-side at full float precision so that
    sub-quantization-step differences (e.g. 8-bit vs 16-bit re-encoded
    versions of the same source) are preserved and amplified by `diff_gain`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Left / base image (B,H,W,C) float32 [0,1]"}),
                "image_b": ("IMAGE", {"tooltip": "Right / compare image (B,H,W,C) float32 [0,1]"}),
            },
            "optional": {
                "label_a": ("STRING", {"default": "A", "tooltip": "Label for image A"}),
                "label_b": ("STRING", {"default": "B", "tooltip": "Label for image B"}),
                "diff_mode": (_DIFF_MODES, {
                    "default": "absolute",
                    "tooltip": "absolute: |A-B|; signed: 0.5 + (B-A)/2; luminance: |Y_A-Y_B|; per-channel: per-RGB |A-B|"}),
                "diff_gain": ("FLOAT", {
                    "default": 16.0, "min": 1.0, "max": 1024.0, "step": 1.0,
                    "tooltip": "Diff amplification (Nuke 'Multiply'). Applied in float space BEFORE 8-bit transport, so >1/255 differences survive."}),
                "diff_gamma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
                    "tooltip": "Gamma applied to the amplified diff (lower = lift mid-tones, higher = crush)"}),
                "diff_threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.001,
                    "tooltip": "Mask out diff values below this float threshold (in pre-gain space). 0=show all."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("diff_image", "diff_mask", "stats")
    OUTPUT_TOOLTIPS = (
        "Diff visualization (post gain/gamma) as IMAGE for downstream use.",
        "Single-channel absolute diff magnitude (pre-gain) as IMAGE for thresholding.",
        "Stats string: max/mean/PSNR/percent-of-pixels-different.",
    )
    FUNCTION = "compare"
    CATEGORY = "C2C/Preview"
    OUTPUT_NODE = True
    DESCRIPTION = "Nuke-style A/B comparer: wipe slider, overlay, and HDR-aware diff (gain + gamma) that preserves 8-vs-16-bit precision differences."

    def compare(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        label_a: str = "A",
        label_b: str = "B",
        diff_mode: str = "absolute",
        diff_gain: float = 16.0,
        diff_gamma: float = 1.0,
        diff_threshold: float = 0.0,
    ):
        B, H, W, C = image_a.shape

        # Match spatial dims if needed
        if image_b.shape[1] != H or image_b.shape[2] != W:
            image_b = F.interpolate(
                image_b.permute(0, 3, 1, 2), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
        if image_b.shape[3] != C:
            image_b = image_b[..., :C] if image_b.shape[3] > C else image_b.repeat(1, 1, 1, C // image_b.shape[3] + 1)[..., :C]

        # ── Compute float-precision diff (preserves <1/255 differences) ──
        a = image_a.float()
        b = image_b.float()
        delta = b - a  # signed, full precision

        if diff_mode == "signed":
            raw_diff_rgb = (0.5 + delta * 0.5).clamp(0.0, 1.0)
            mag = delta.abs().mean(dim=-1, keepdim=True)
        elif diff_mode == "luminance":
            ya = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
            yb = 0.2126 * b[..., 0] + 0.7152 * b[..., 1] + 0.0722 * b[..., 2]
            yd = (yb - ya).abs()
            raw_diff_rgb = yd.unsqueeze(-1).expand(-1, -1, -1, 3)
            mag = yd.unsqueeze(-1)
        elif diff_mode == "per-channel":
            raw_diff_rgb = delta.abs()
            mag = raw_diff_rgb.mean(dim=-1, keepdim=True)
        else:  # absolute
            mag_scalar = delta.abs().mean(dim=-1, keepdim=True)
            raw_diff_rgb = mag_scalar.expand(-1, -1, -1, 3)
            mag = mag_scalar

        if diff_threshold > 0.0:
            keep = (mag >= diff_threshold).float()
            raw_diff_rgb = raw_diff_rgb * keep

        # Apply gain + gamma (Nuke style)
        if diff_mode == "signed":
            amplified = ((raw_diff_rgb - 0.5) * diff_gain + 0.5).clamp(0.0, 1.0)
        else:
            amplified = (raw_diff_rgb * diff_gain).clamp(0.0, 1.0)

        if diff_gamma != 1.0:
            amplified = amplified.clamp(min=1e-8).pow(1.0 / diff_gamma)

        diff_image_out = amplified.clamp(0.0, 1.0)

        # Stats
        mse = (delta.pow(2)).mean().item()
        max_abs = delta.abs().max().item()
        mean_abs = delta.abs().mean().item()
        psnr = float("inf") if mse <= 1e-12 else 10.0 * float(np.log10(1.0 / mse))
        pct_diff = float((mag > (1.0 / 65535.0)).float().mean().item() * 100.0)
        stats = (
            f"diff_mode={diff_mode} gain={diff_gain:.1f} gamma={diff_gamma:.2f}\n"
            f"max|d|={max_abs:.6f}  mean|d|={mean_abs:.6f}\n"
            f"PSNR={psnr:.2f}dB  px>16bit_step={pct_diff:.4f}%"
        )

        ui_payload = {
            "image_a": [_save_temp(image_a[0], "cmp_a")],
            "image_b": [_save_temp(image_b[0], "cmp_b")],
            "image_diff": [_save_temp(diff_image_out[0], "cmp_d")],
            "label_a": [label_a],
            "label_b": [label_b],
            "stats": [stats],
            "diff_mode": [diff_mode],
            "diff_gain": [float(diff_gain)],
            "diff_gamma": [float(diff_gamma)],
        }

        return {
            "ui": ui_payload,
            "result": (diff_image_out, mag.squeeze(-1), stats),
        }


def _save_temp(tensor: torch.Tensor, prefix: str) -> dict:
    """Save a single (H,W,C) tensor as a temp PNG and return ComfyUI file info."""
    arr = (tensor.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    digest = hashlib.sha256(arr.tobytes()[:8192]).hexdigest()[:10]
    name = f"{prefix}_{digest}.png"
    Image.fromarray(arr).save(
        os.path.join(folder_paths.get_temp_directory(), name),
        compress_level=1,
    )
    return {"filename": name, "subfolder": "", "type": "temp"}
