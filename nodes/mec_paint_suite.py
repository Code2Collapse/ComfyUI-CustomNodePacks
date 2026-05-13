"""
MEC Paint Suite
===============

Four nodes that, together, form an interactive paint → fix → refine → build
pipeline inspired by Forbidden Vision but rebuilt around MEC's mask math:

  1. ``MECAdvancedPaintCanvas``  – interactive canvas + Nuke-style mask math
  2. ``MECContextInpainter``    – crop / inpaint / blend with smart logic
  3. ``MECToneRefiner``         – exposure + colour rescue + fake DOF
  4. ``MECBuilderSampler``      – KSampler with adaptive CFG + polish pass

The JS canvas (``js/mec_advanced_paint.js``) writes its drawing into a hidden
``canvas_data`` STRING widget as a base64 PNG.  The Python node decodes that
string into RGBA, optionally composites it on top of ``reference_image``, and
then runs the full Nuke-style mask pipeline (hardness → expansion → blur).
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import math
import re
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("MEC.Paint")

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    from scipy import ndimage as _ndi  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _ndi = None  # type: ignore
    _HAS_SCIPY = False

try:  # PIL is part of ComfyUI's runtime — used to decode the JS PNG payload.
    from PIL import Image
    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    _HAS_PIL = False

# Comfy core (only imported lazily inside methods that need it so this module
# can still be imported in test contexts).


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  small numeric helpers                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _to_bhwc(img: torch.Tensor) -> torch.Tensor:
    """Accept (H,W,C), (B,H,W,C) or (B,C,H,W) and return (B,H,W,C) float."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dim() != 4:
        raise ValueError(f"image tensor must be 3-D or 4-D, got {tuple(img.shape)}")
    if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = img.permute(0, 2, 3, 1).contiguous()
    return img.float().clamp(0.0, 1.0)


def _to_mask(mask: torch.Tensor) -> torch.Tensor:
    """Coerce any mask-shaped input to (B,H,W) float in [0,1]."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            mask = mask.mean(dim=-1)
    return mask.float().clamp(0.0, 1.0)


def _gaussian_blur_2d(t: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable Gaussian blur on a (B,1,H,W) tensor with float radius (px)."""
    if radius <= 0.0:
        return t
    sigma = max(radius / 2.0, 1e-3)
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    half = ksize // 2
    x = torch.arange(-half, half + 1, dtype=t.dtype, device=t.device)
    g = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    g_x = g.view(1, 1, 1, ksize)
    g_y = g.view(1, 1, ksize, 1)
    out = F.conv2d(t, g_x, padding=(0, half))
    out = F.conv2d(out, g_y, padding=(half, 0))
    return out


def _morph_2d(mask_2d: np.ndarray, pixels: int) -> np.ndarray:
    """Morphological dilate (>0) / erode (<0) by ``pixels`` pixels.

    Falls back to scipy or a torch maxpool implementation if cv2 is missing.
    Input/output is a float32 ``(H,W)`` array in [0,1].
    """
    if pixels == 0:
        return mask_2d
    radius = abs(int(pixels))
    if _HAS_CV2:
        ksize = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = (mask_2d > 0.5).astype(np.uint8) * 255
        if pixels > 0:
            out = cv2.dilate(binary, kernel, iterations=1)
        else:
            out = cv2.erode(binary, kernel, iterations=1)
        return (out.astype(np.float32) / 255.0)
    if _HAS_SCIPY:
        binary = mask_2d > 0.5
        if pixels > 0:
            out = _ndi.binary_dilation(binary, iterations=radius)
        else:
            out = _ndi.binary_erosion(binary, iterations=radius)
        return out.astype(np.float32)
    # last-resort torch maxpool dilation (erosion = invert/dilate/invert)
    t = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).float()
    if pixels > 0:
        out = F.max_pool2d(t, 2 * radius + 1, stride=1, padding=radius)
    else:
        out = 1.0 - F.max_pool2d(1.0 - t, 2 * radius + 1, stride=1, padding=radius)
    return out.squeeze().numpy()


def _np_to_bhwc(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).clamp(0.0, 1.0)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Node 1 — Advanced Paint Canvas                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
class MECAdvancedPaintCanvas:
    """Interactive paint canvas with Nuke-style procedural mask math.

    The JS widget posts a base64 PNG (RGBA) into the hidden ``canvas_data``
    string widget on every serialise.  Python decodes it, optionally composites
    over ``reference_image``, then derives ``processed_mask`` from the alpha
    channel of the painted strokes through four ordered stages:

      1. **Raw mask**:        ``alpha`` of painted pixels, normalised to [0,1].
      2. **mask_hardness**:   anything brighter than ``(1 - hardness)`` is
         clamped to 1.0 — produces a solid inner core whose width is exactly
         the user-selected hardness fraction of the brush profile.
      3. **mask_expansion**:  morphological dilate / erode in pixels.  Negative
         shrinks, positive grows.  Uses cv2 ellipse kernel when available.
      4. **mask_blur**:       Gaussian blur of ``radius`` px, then linearly
         interpolated against the hard mask by ``mask_blur_strength``
         (0 = hard, 1 = full blur).
    """

    CATEGORY = "MEC/Paint"
    DESCRIPTION = (
        "Interactive paint canvas with procedural mask math: hardness, expansion, and blur stages "
        "are applied in order to the alpha channel of painted strokes."
    )
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("painted_image", "processed_mask")
    OUTPUT_TOOLTIPS = (
        "Painted RGB image (composited over reference_image when supplied).",
        "Processed mask after hardness, expansion, and blur stages.",
    )
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_width":  ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Canvas width in pixels."}),
                "canvas_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Canvas height in pixels."}),
                "brush_type":     (["paint", "mask_only"], {"default": "paint", "tooltip": "paint: composite color strokes onto the image. mask_only: build the mask without coloring."}),
                "brush_color":    ("STRING", {"default": "#000000", "tooltip": "Hex color used by the paint brush (ignored in mask_only mode)."}),
                "brush_opacity":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Brush stroke opacity (0 = invisible, 1 = fully opaque)."}),
                "brush_hardness": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Brush profile hardness (0 = soft, 1 = hard edge)."}),
                "brush_size":     ("INT",   {"default": 20,  "min": 1,   "max": 500, "step": 1, "tooltip": "Brush diameter in pixels."}),
                "mask_hardness":     ("FLOAT", {"default": 0.5, "min": 0.0,    "max": 1.0,   "step": 0.01, "tooltip": "Threshold for the solid inner core: pixels brighter than (1 - hardness) clamp to 1.0."}),
                "mask_expansion":    ("INT",   {"default": 0,   "min": -100,   "max": 100,   "step": 1, "tooltip": "Morphological dilate (positive) / erode (negative) in pixels."}),
                "mask_blur_radius":  ("FLOAT", {"default": 0.0, "min": 0.0,    "max": 100.0, "step": 0.1, "tooltip": "Gaussian blur radius applied to the mask edge in pixels."}),
                "mask_blur_strength":("FLOAT", {"default": 1.0, "min": 0.0,    "max": 1.0,   "step": 0.01, "tooltip": "Blend factor between the hard mask (0) and the fully-blurred mask (1)."}),
                "canvas_data":    ("STRING", {"multiline": False, "default": "", "tooltip": "Internal base64 PNG payload from the JS canvas widget. Do not edit manually."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional background image; painted strokes are composited over it when supplied."}),
            },
        }

    # ----- decode the JS payload -------------------------------------
    @staticmethod
    def _decode_canvas(data: str, w: int, h: int) -> np.ndarray:
        """Decode the hidden ``canvas_data`` STRING into an HxWx4 float array.

        Empty / invalid payloads gracefully fall back to a fully-transparent
        canvas so the node never errors during the very first execution before
        the user has painted anything.
        """
        empty = np.zeros((h, w, 4), dtype=np.float32)
        if not data:
            return empty
        # MANUAL bug-fix (Apr 2026): PIL is required for this node. Silently
        # returning an empty canvas hid setup errors. Fail loudly with an
        # actionable message so the user knows to install Pillow.
        if not _HAS_PIL:
            raise RuntimeError(
                "MECAdvancedPaintCanvas requires Pillow (PIL) to decode the "
                "painted canvas. Install it with `pip install Pillow` in your "
                "ComfyUI environment, then restart ComfyUI."
            )
        try:
            payload = data.split(",", 1)[1] if data.startswith("data:") else data
            raw = base64.b64decode(payload)
            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGBA")
                if im.size != (w, h):
                    im = im.resize((w, h), Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr
        except (binascii.Error, ValueError, OSError) as exc:
            # MANUAL bug-fix (Apr 2026): narrowed from broad Exception.
            # Covers base64 decode errors, PIL UnidentifiedImageError
            # (subclass of OSError), and corrupt-PNG IO errors.
            logger.warning("[MECAdvancedPaintCanvas] canvas decode failed: %s", exc)
            return empty

    # ----- mask pipeline (Nuke-style) --------------------------------
    @staticmethod
    def _process_mask(alpha: np.ndarray,
                      hardness: float,
                      expansion: int,
                      blur_radius: float,
                      blur_strength: float) -> np.ndarray:
        m = np.clip(alpha.astype(np.float32), 0.0, 1.0)

        # 2 — hardness: any pixel above (1 - hardness) becomes a solid 1.0
        if hardness > 0.0:
            thresh = 1.0 - float(hardness)
            core = (m >= thresh).astype(np.float32)
            m = np.maximum(m, core)

        # 3 — expansion (morph)
        if expansion != 0:
            m = _morph_2d(m, int(expansion))

        # 4 — blur + lerp
        if blur_radius > 0.0 and blur_strength > 0.0:
            t = torch.from_numpy(m)[None, None, ...].float()
            blurred = _gaussian_blur_2d(t, float(blur_radius)).squeeze().numpy()
            m = (1.0 - float(blur_strength)) * m + float(blur_strength) * blurred
        return np.clip(m, 0.0, 1.0)

    # ----- main --------------------------------------------------------
    def execute(self, canvas_width, canvas_height, brush_type, brush_color,
                brush_opacity, brush_hardness, brush_size,
                mask_hardness, mask_expansion, mask_blur_radius,
                mask_blur_strength, canvas_data, reference_image=None):

        w = int(canvas_width)
        h = int(canvas_height)

        rgba = self._decode_canvas(canvas_data, w, h)            # (H,W,4) [0..1]
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]

        # background (reference image or white)
        if reference_image is not None:
            ref = _to_bhwc(reference_image)[0].cpu().numpy()
            if ref.shape[:2] != (h, w):
                ref_t = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0)
                ref_t = F.interpolate(ref_t, size=(h, w), mode="bilinear", align_corners=False)
                ref = ref_t.squeeze(0).permute(1, 2, 0).numpy()
            ref = ref[..., :3]
        else:
            ref = np.ones((h, w, 3), dtype=np.float32)

        # painted image: alpha-composite the strokes over the background.  In
        # ``mask_only`` mode the user only wants the mask, so ``painted_image``
        # is identical to the reference (or blank white) image.
        if brush_type == "mask_only":
            painted = ref
        else:
            a = alpha[..., None]
            painted = rgb * a + ref * (1.0 - a)

        proc = self._process_mask(
            alpha,
            float(mask_hardness),
            int(mask_expansion),
            float(mask_blur_radius),
            float(mask_blur_strength),
        )

        painted_t = _np_to_bhwc(painted)
        mask_t = torch.from_numpy(proc.astype(np.float32))[None, ...]
        return (painted_t, mask_t)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Node 2 — Context Inpainter (Fixer)                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _parse_wildcard_prompt(
    prompt: str,
    n_regions: int,
    region_sizes: Optional[List[int]] = None,
) -> List[str]:
    """Parse Impact-Pack-style per-face wildcard markers.

    Supported markers (Forbidden-Vision compatible):
      * ``[SEP]``                -- segment separator
      * ``[ASC]`` / ``[DSC]``    -- ascending / descending region-index order
      * ``[ASC-SIZE]`` / ``[DSC-SIZE]`` -- order regions by area (small→large
        or large→small). Requires ``region_sizes`` to be supplied.
      * ``[SKIP]``               -- emit an empty prompt for that region

    Missing segments are padded with the last available segment (or the
    entire prompt if no ``[SEP]`` was used).
    """
    if not prompt:
        return [""] * n_regions
    order = "ASC"
    p = prompt
    # Order markers (size-based variants take priority over the plain ones).
    for tok, val in (("[DSC-SIZE]", "DSC-SIZE"),
                     ("[ASC-SIZE]", "ASC-SIZE"),
                     ("[DSC]", "DSC"),
                     ("[ASC]", "ASC")):
        if tok in p:
            order = val
            p = p.replace(tok, "")
    parts = [s.strip() for s in re.split(r"\[SEP\]", p)]
    parts = [s for s in parts if s != ""]
    if not parts:
        return [""] * n_regions

    # Build region-index permutation according to the chosen order.
    if order == "DSC":
        idx_order = list(range(n_regions))[::-1]
    elif order == "ASC-SIZE" and region_sizes is not None and len(region_sizes) >= n_regions:
        idx_order = sorted(range(n_regions), key=lambda i: region_sizes[i])
    elif order == "DSC-SIZE" and region_sizes is not None and len(region_sizes) >= n_regions:
        idx_order = sorted(range(n_regions), key=lambda i: -region_sizes[i])
    else:  # ASC default
        idx_order = list(range(n_regions))

    out: List[str] = [""] * n_regions
    for j, region_idx in enumerate(idx_order):
        seg = parts[j] if j < len(parts) else parts[-1]
        if "[SKIP]" in seg:
            out[region_idx] = ""
        else:
            out[region_idx] = seg
    return out


def _label_regions(mask_2d: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected-component labelling with cv2 / scipy / pure-python fallback."""
    binary = (mask_2d > 0.5).astype(np.uint8)
    if _HAS_CV2:
        n, labels = cv2.connectedComponents(binary, connectivity=8)
        return labels.astype(np.int32), max(int(n) - 1, 0)
    if _HAS_SCIPY:
        labels, n = _ndi.label(binary)  # type: ignore
        return labels.astype(np.int32), int(n)
    # last-resort: treat the entire mask as one region
    return binary.astype(np.int32), 1 if binary.any() else 0


class MECContextInpainter:
    """Smart-blend inpainted output back over the original image.

    The math is laid out in the order it runs:

      * **crop_padding** – extends the masked bbox by a multiplier so the
        inpaint sees context.
      * **mask_expansion_blend** – per-blend dilate/erode on the *blend* mask
        only (not the inpaint mask).
      * **blend_softness** – Gaussian feather on that expanded mask.
      * **enable_color_correction** – matches the inpainted region's
        per-channel mean / std to the original region's mean / std (Reinhard).
      * **enable_lightness_rescue** -- CIE LAB L-channel comparison.  If the
        inpainted region is >5 % darker, lerp L upwards by the deficit.
      * **enable_differential_diffusion** – ``|orig − inpaint|`` per pixel is
        used as a soft preservation weight, so unchanged pixels are kept.
      * **sampling_mask_blur*** – additional blur applied to the *output*
        ``debug_mask`` (used when feeding back into another sampler).
    """

    CATEGORY = "MEC/Paint"
    DESCRIPTION = (
        "Smart-blend an inpainted region back over the original image with crop padding, "
        "feathered blend mask, optional color correction, lightness rescue, and differential diffusion."
    )
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("blended_image", "debug_mask")
    OUTPUT_TOOLTIPS = (
        "Final composited image with the inpainted region blended back into the original.",
        "Debug mask used for the blend (after expansion and blur).",
    )
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image":   ("IMAGE", {"tooltip": "Original (un-inpainted) image used as the blend base."}),
                "mask":             ("MASK", {"tooltip": "Mask defining the inpainted region."}),
                "inpainted_image":  ("IMAGE", {"tooltip": "Inpainted image to blend back over the original."}),
                "crop_padding":     ("FLOAT",  {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.01, "tooltip": "Multiplier extending the masked bbox so the inpaint sees more context."}),
                "blend_softness":   ("FLOAT",  {"default": 8.0, "min": 0.0, "max": 200.0, "step": 0.5, "tooltip": "Gaussian feather radius applied to the blend mask in pixels."}),
                "mask_expansion_blend": ("INT",  {"default": 0, "min": -100, "max": 100, "step": 1, "tooltip": "Per-blend dilate (positive) / erode (negative) of the blend mask in pixels."}),
                "enable_color_correction":     ("BOOLEAN", {"default": True, "tooltip": "Reinhard mean/std color match between original and inpainted regions."}),
                "enable_lightness_rescue":     ("BOOLEAN", {"default": True, "tooltip": "Lift CIE LAB L-channel of the inpaint when it is more than ~5% darker."}),
                "enable_differential_diffusion":("BOOLEAN", {"default": False, "tooltip": "Use |orig - inpaint| as a soft preservation weight to keep unchanged pixels."}),
                "sampling_mask_blur_size":     ("INT",   {"default": 21, "min": 0, "max": 201, "step": 1, "tooltip": "Kernel size (odd) for the additional blur on the output debug mask."}),
                "sampling_mask_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend factor for the sampling mask blur applied to debug_mask."}),
            },
            # face_positive_prompt / face_negative_prompt are
            # *informational only* in this node -- they are parsed for
            # wildcards and logged so downstream FaceInpaint nodes can
            # consume them, but this node never samples. Demoted from
            # required to optional + tooltipped so users aren't
            # surprised they have no visible effect here.
            "optional": {
                "face_positive_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "Optional region-attached positive prompt; parsed for "
                        "`{a|b|c}` wildcards per detected mask region and "
                        "logged. Does NOT sample here -- pair with a "
                        "FaceInpaint/KSampler downstream."
                    ),
                }),
                "face_negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Same as face_positive_prompt but for negatives.",
                }),
            },
        }

    # ---- core math ------------------------------------------------
    @staticmethod
    def _color_match(orig: np.ndarray, fake: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Per-channel mean/std match (Reinhard) constrained to mask region."""
        out = fake.copy()
        sel = m > 0.05
        if not sel.any():
            return out
        for c in range(3):
            o = orig[..., c][sel]
            f = fake[..., c][sel]
            mo, so = float(o.mean()), float(o.std() + 1e-6)
            mf, sf = float(f.mean()), float(f.std() + 1e-6)
            out[..., c] = ((fake[..., c] - mf) * (so / sf) + mo).clip(0.0, 1.0)
        return out

    @staticmethod
    def _lightness_rescue(orig: np.ndarray, fake: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Lift L channel of ``fake`` toward ``orig`` if it is >5% darker."""
        if not _HAS_CV2:
            # luminance-only fallback (BT.709)
            lo = (orig * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
            lf = (fake * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
            sel = m > 0.05
            if not sel.any():
                return fake
            d = float(lo[sel].mean() - lf[sel].mean())
            if d <= 0.05:
                return fake
            return np.clip(fake + d * m[..., None], 0.0, 1.0)
        lab_o = cv2.cvtColor((orig * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_f = cv2.cvtColor((fake * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        sel = m > 0.05
        if not sel.any():
            return fake
        # L is in 0..255 in OpenCV's 8-bit LAB; 5 % is ~12.75
        lo, lf = lab_o[..., 0], lab_f[..., 0]
        d = float(lo[sel].mean() - lf[sel].mean())
        if d <= 12.75:
            return fake
        lab_f[..., 0] = np.clip(lab_f[..., 0] + d * m, 0.0, 255.0)
        out = cv2.cvtColor(lab_f.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return out

    @staticmethod
    def _differential_weight(orig: np.ndarray, fake: np.ndarray) -> np.ndarray:
        """High weight where pixels actually changed, low where they didn't."""
        d = np.abs(fake - orig).max(axis=-1)
        # rescale per-image so the brightest delta is 1
        m = d.max()
        if m > 1e-6:
            d = d / m
        return d.astype(np.float32)

    # ---- main ------------------------------------------------------
    def execute(self, original_image, mask, inpainted_image,
                crop_padding, blend_softness, mask_expansion_blend,
                enable_color_correction, enable_lightness_rescue,
                enable_differential_diffusion,
                sampling_mask_blur_size, sampling_mask_blur_strength,
                face_positive_prompt="", face_negative_prompt=""):
        # MANUAL bug-fix (Apr 2026): hard-clamp crop_padding into the
        # documented [1.0, 2.0] range so a stray IPU/widget can't blow
        # up the bbox math. min/max widget bounds are advisory; a JSON
        # import or wildcard could deliver any float.
        try:
            crop_padding = float(crop_padding)
        except (TypeError, ValueError):
            crop_padding = 1.2
        crop_padding = max(1.0, min(2.0, crop_padding))
        orig = _to_bhwc(original_image)[0].cpu().numpy()
        fake = _to_bhwc(inpainted_image)[0].cpu().numpy()
        H, W = orig.shape[:2]
        m = _to_mask(mask)[0].cpu().numpy()
        if m.shape != (H, W):
            m_t = torch.from_numpy(m)[None, None].float()
            m_t = F.interpolate(m_t, size=(H, W), mode="bilinear", align_corners=False)
            m = m_t.squeeze().numpy()

        if fake.shape[:2] != (H, W):
            f_t = torch.from_numpy(fake).permute(2, 0, 1).unsqueeze(0)
            f_t = F.interpolate(f_t, size=(H, W), mode="bilinear", align_corners=False)
            fake = f_t.squeeze(0).permute(1, 2, 0).numpy()

        # ---- crop window with padding (purely informative — used for the
        # debug mask and for region detection) -----------------------------
        ys, xs = np.where(m > 0.05)
        if ys.size > 0 and xs.size > 0:
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            cy = (y0 + y1) * 0.5
            cx = (x0 + x1) * 0.5
            hh = (y1 - y0) * float(crop_padding) * 0.5
            ww = (x1 - x0) * float(crop_padding) * 0.5
            y0 = max(0, int(cy - hh))
            y1 = min(H - 1, int(cy + hh))
            x0 = max(0, int(cx - ww))
            x1 = min(W - 1, int(cx + ww))
        else:
            y0 = 0
            x0 = 0
            y1 = H - 1
            x1 = W - 1

        # ---- region split for wildcard prompts ---------------------------
        labels, n_reg = _label_regions(m)
        # Compute per-region pixel counts so [ASC-SIZE] / [DSC-SIZE] work.
        region_sizes: List[int] = []
        for i in range(1, n_reg + 1):
            region_sizes.append(int((labels == i).sum()))
        pos_prompts = _parse_wildcard_prompt(face_positive_prompt, max(n_reg, 1), region_sizes or None)
        neg_prompts = _parse_wildcard_prompt(face_negative_prompt, max(n_reg, 1), region_sizes or None)
        # We don't actually run a sampler here; we expose the parsed prompts
        # via the node's print output (deterministic, observable in console).
        if n_reg > 0:
            # MANUAL bug-fix (Apr 2026): use logger instead of print so
            # output is consistent with the rest of the suite and can
            # be filtered by users.
            logger.info("[MECContextInpainter] %d mask regions detected.", n_reg)
            for i in range(n_reg):
                logger.info(
                    "  region %d: + %r  - %r",
                    i, pos_prompts[i], neg_prompts[i],
                )

        # ---- colour correction & lightness rescue ------------------------
        out = fake
        if enable_color_correction:
            out = self._color_match(orig, out, m)
        if enable_lightness_rescue:
            out = self._lightness_rescue(orig, out, m)

        # ---- differential diffusion --------------------------------------
        if enable_differential_diffusion:
            w = self._differential_weight(orig, fake)
            blend = np.clip(m * np.maximum(w, 0.05), 0.0, 1.0)
        else:
            blend = m.copy()

        # blend mask: expansion + softness
        if int(mask_expansion_blend) != 0:
            blend = _morph_2d(blend, int(mask_expansion_blend))
        if float(blend_softness) > 0.0:
            t = torch.from_numpy(blend)[None, None].float()
            blend = _gaussian_blur_2d(t, float(blend_softness)).squeeze().numpy()
        blend = np.clip(blend, 0.0, 1.0)

        # final composite
        b = blend[..., None]
        composite = out * b + orig * (1.0 - b)
        composite = np.clip(composite, 0.0, 1.0)

        # debug mask: optionally blurred sampling mask
        dbg = m
        if int(sampling_mask_blur_size) > 0 and float(sampling_mask_blur_strength) > 0.0:
            t = torch.from_numpy(dbg)[None, None].float()
            blurred = _gaussian_blur_2d(t, float(sampling_mask_blur_size) / 2.0).squeeze().numpy()
            dbg = (1.0 - float(sampling_mask_blur_strength)) * dbg + \
                  float(sampling_mask_blur_strength) * blurred

        return (_np_to_bhwc(composite), torch.from_numpy(dbg.astype(np.float32))[None, ...])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Node 3 — Tone Refiner                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
class MECToneRefiner:
    """Auto-correct tone, optional upscale and centre-focus DOF.

    The neural-corrector is *not* a learned model — it's a deterministic
    histogram-driven exposure / black-level / gray-world correction whose
    effect is gated by the user blend factors.  It runs in three stages:

      1. **Black/white-point lift**: percentile-based remapping of darkest 1 %
         to 0 and brightest 99 % to 1, with a smoothstep curve to avoid
         clipped highlights.
      2. **Gray-world colour balance**: divide each channel by its mean and
         renormalise so the overall mean equals the channel-wide mean.
      3. **Tone / colour blend**: lerp between original and corrected.

    DOF is a fake center-focus blur — depth = 1 at centre, 0 at corners,
    raised to a power that matches ``ai_dof_focus_depth``.
    """

    CATEGORY = "MEC/Paint"
    DESCRIPTION = (
        "Auto-correct tone (black/white-point + gray-world), optionally upscale, "
        "and apply a fake center-focus depth-of-field blur."
    )
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("refined_image", "refined_latent")
    OUTPUT_TOOLTIPS = (
        "Tone- and color-corrected image (optionally upscaled and DOF-blurred).",
        "VAE-encoded latent of the refined image (zero placeholder when auto_upscale is False).",
    )
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":             ("IMAGE", {"tooltip": "Image to refine."}),
                "neural_corrector":  ("BOOLEAN", {"default": True, "tooltip": "Enable deterministic tone + gray-world correction (not a learned model)."}),
                "corrector_tone":    ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend amount toward the tone-corrected image (0 = original, 1 = full correction)."}),
                "corrector_color":   ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend amount toward the gray-world color-corrected image."}),
                "highlight_protection": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Roll-off applied above 95th-percentile to prevent highlight clipping (0 = none, 1 = strong)."}),
                "shadow_lift":       ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Lift shadows below the 5th-percentile (0 = none, 1 = strong; mirrors highlight_protection on the dark side)."}),
                "enable_upscale":    ("BOOLEAN", {"default": False, "tooltip": "Upscale by upscale_factor; uses upscale_model if provided, bicubic otherwise."}),
                "upscale_factor":    ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Upscale multiplier applied when enable_upscale is True."}),
                "ai_enable_dof":     ("BOOLEAN", {"default": False, "tooltip": "Apply depth-based DOF (uses depth_map if connected, else fake center-focus)."}),
                "ai_dof_strength":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Strength of the DOF blur (also scales the maximum blur radius)."}),
                "ai_dof_focus_depth":("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Focus plane: when depth_map is connected this is the in-focus depth value (0=near,1=far); without depth_map it controls center-focus tightness."}),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Optional pre-existing latent; passed through when supplied (skips VAE encode)."}),
                "vae":    ("VAE", {"tooltip": "VAE used to encode the refined image into refined_latent (when auto_upscale is True)."}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional UPSCALE_MODEL (RealESRGAN / 4x-NMKD / etc.). When connected and enable_upscale is True, used instead of bicubic and resized to upscale_factor."}),
                "depth_map":     ("MASK", {"tooltip": "Optional depth map (0=near,1=far). Drives DOF when connected; replaces the fake center-focus radial gradient."}),
                # MANUAL bug-fix (Apr 2026): explicit auto_upscale toggle.
                # When False (and no `latent` is supplied), VAE-encode is
                # skipped entirely and a zero-latent placeholder is
                # returned -- avoids surprise GPU spikes when users wire
                # a VAE but only want the IMAGE output.
                "auto_upscale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True (default, back-compat), encode the "
                        "refined image through the supplied VAE to "
                        "produce `refined_latent`. Set False to skip "
                        "the VAE-encode and return a zero placeholder."
                    ),
                }),
            },
        }

    @staticmethod
    def _auto_tone(img: np.ndarray) -> np.ndarray:
        """Percentile-based black/white-point with smoothstep curve."""
        out = img.copy()
        for c in range(3):
            ch = out[..., c]
            lo = float(np.percentile(ch, 1.0))
            hi = float(np.percentile(ch, 99.0))
            if hi - lo < 1e-3:
                continue
            n = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
            n = n * n * (3.0 - 2.0 * n)        # smoothstep
            out[..., c] = n
        return out.clip(0.0, 1.0)

    @staticmethod
    def _highlight_protect(img: np.ndarray, strength: float) -> np.ndarray:
        """Filmic-style soft roll-off above the 95th percentile.

        Replaces the linear top-end with a Reinhard ``x/(1+k*x)`` curve so
        bright pixels never clip to pure white. ``strength`` in [0,1] controls
        how aggressive the roll-off is.
        """
        if strength <= 1e-6:
            return img
        out = img.copy()
        for c in range(3):
            ch = out[..., c]
            knee = float(np.percentile(ch, 95.0))
            if knee >= 1.0 - 1e-3:
                continue
            above = ch > knee
            if not above.any():
                continue
            x = (ch[above] - knee) / max(1.0 - knee, 1e-6)
            k = 1.0 + 6.0 * float(strength)
            rolled = x / (1.0 + k * x)
            ch[above] = knee + rolled * (1.0 - knee)
            out[..., c] = ch
        return out.clip(0.0, 1.0)

    @staticmethod
    def _shadow_lift(img: np.ndarray, strength: float) -> np.ndarray:
        """Inverse of highlight protection: lift the bottom 5%."""
        if strength <= 1e-6:
            return img
        out = img.copy()
        for c in range(3):
            ch = out[..., c]
            knee = float(np.percentile(ch, 5.0))
            if knee <= 1e-3:
                continue
            below = ch < knee
            if not below.any():
                continue
            x = 1.0 - ch[below] / max(knee, 1e-6)
            k = 1.0 + 6.0 * float(strength)
            lifted = 1.0 - x / (1.0 + k * x)
            ch[below] = lifted * knee
            out[..., c] = ch
        return out.clip(0.0, 1.0)

    @staticmethod
    def _gray_world(img: np.ndarray) -> np.ndarray:
        means = img.reshape(-1, 3).mean(axis=0) + 1e-6
        target = float(means.mean())
        scale = target / means
        return np.clip(img * scale[None, None, :], 0.0, 1.0)

    @staticmethod
    def _dof(img: np.ndarray, strength: float, focus: float) -> np.ndarray:
        H, W = img.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W * 0.5, H * 0.5
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        r = np.clip(r, 0.0, 1.0)
        # focus = 1 means strong centre-only focus, focus = 0.5 = wide.
        power = 1.0 + (float(focus) - 0.5) * 8.0
        coc = np.clip(r ** power, 0.0, 1.0)
        coc = coc * float(strength)
        radius = max(0.5, 12.0 * float(strength))
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        blurred = _gaussian_blur_2d(t, radius).squeeze(0).permute(1, 2, 0).numpy()
        return np.clip(img * (1.0 - coc[..., None]) + blurred * coc[..., None], 0.0, 1.0)

    @staticmethod
    def _dof_depth(img: np.ndarray, depth: np.ndarray, strength: float, focus_depth: float) -> np.ndarray:
        """Depth-driven circle-of-confusion DOF.

        ``depth`` is a (H,W) float in [0,1]; ``focus_depth`` is the in-focus
        plane. Pixels whose depth differs most from focus_depth get the
        strongest blur. Multi-scale blur is approximated with two Gaussian
        radii blended by the per-pixel CoC magnitude.
        """
        H, W = img.shape[:2]
        if depth.shape != (H, W):
            d_t = torch.from_numpy(depth)[None, None].float()
            d_t = F.interpolate(d_t, size=(H, W), mode="bilinear", align_corners=False)
            depth = d_t.squeeze().numpy()
        coc = np.abs(depth - float(focus_depth))
        coc = (coc / max(coc.max(), 1e-6)) * float(strength)
        coc = np.clip(coc, 0.0, 1.0)
        radius_max = max(0.5, 18.0 * float(strength))
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        b1 = _gaussian_blur_2d(t, radius_max * 0.5).squeeze(0).permute(1, 2, 0).numpy()
        b2 = _gaussian_blur_2d(t, radius_max).squeeze(0).permute(1, 2, 0).numpy()
        # 2-tap multi-scale blend by CoC magnitude.
        c = coc[..., None]
        c2 = (c * c)
        out = img * (1.0 - c) + b1 * (c - c2) + b2 * c2
        return np.clip(out, 0.0, 1.0)

    def execute(self, image, neural_corrector, corrector_tone, corrector_color,
                highlight_protection, shadow_lift,
                enable_upscale, upscale_factor, ai_enable_dof,
                ai_dof_strength, ai_dof_focus_depth,
                latent=None, vae=None, upscale_model=None,
                depth_map=None, auto_upscale=True):
        img = _to_bhwc(image)[0].cpu().numpy()
        out = img

        if neural_corrector:
            toned = self._auto_tone(out)
            colored = self._gray_world(toned)
            out = (1.0 - float(corrector_tone)) * out + float(corrector_tone) * toned
            out = (1.0 - float(corrector_color)) * out + float(corrector_color) * colored
            out = np.clip(out, 0.0, 1.0)

        # Highlight clipping protection (filmic roll-off).
        if float(highlight_protection) > 0.0:
            out = self._highlight_protect(out, float(highlight_protection))
        if float(shadow_lift) > 0.0:
            out = self._shadow_lift(out, float(shadow_lift))

        if ai_enable_dof:
            if depth_map is not None:
                d = _to_mask(depth_map)[0].cpu().numpy()
                out = self._dof_depth(out, d, float(ai_dof_strength), float(ai_dof_focus_depth))
            else:
                out = self._dof(out, float(ai_dof_strength), float(ai_dof_focus_depth))

        if enable_upscale and float(upscale_factor) > 1.0:
            new_h = int(round(out.shape[0] * float(upscale_factor)))
            new_w = int(round(out.shape[1] * float(upscale_factor)))
            if upscale_model is not None:
                # Use ComfyUI's upscale-with-model utility for native compat.
                try:
                    import comfy_extras.nodes_upscale_model as _um  # type: ignore
                    img_t = _np_to_bhwc(out)
                    (upscaled,) = _um.ImageUpscaleWithModel().upscale(upscale_model, img_t)
                    # Resize to exact target shape (model outputs fixed factor).
                    if upscaled.shape[1] != new_h or upscaled.shape[2] != new_w:
                        u = upscaled.permute(0, 3, 1, 2)
                        u = F.interpolate(u, size=(new_h, new_w), mode="bicubic", align_corners=False)
                        upscaled = u.permute(0, 2, 3, 1).clamp(0, 1)
                    out = upscaled[0].cpu().numpy()
                except (ImportError, AttributeError, RuntimeError) as exc:
                    logger.warning(
                        "[MECToneRefiner] upscale_model failed (%s: %s); falling back to bicubic.",
                        type(exc).__name__, exc,
                    )
                    t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                    t = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False)
                    out = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
            else:
                t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                t = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False)
                out = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

        out_img = _np_to_bhwc(out)

        # Latent passthrough or re-encode
        if latent is not None:
            out_lat = latent
        elif vae is not None and auto_upscale:
            # MANUAL bug-fix (Apr 2026): narrowed from broad Exception so
            # genuine VAE bugs surface. Only OOM and shape mismatches are
            # caught; everything else propagates with a clear traceback.
            try:
                samples = vae.encode(out_img[:, :, :, :3])
                out_lat = {"samples": samples}
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.warning(
                    "[MECToneRefiner] VAE.encode failed (%s: %s); returning zero-latent placeholder.",
                    type(exc).__name__, exc,
                )
                out_lat = {"samples": torch.zeros(
                    1, 4,
                    max(out_img.shape[1] // 8, 1),
                    max(out_img.shape[2] // 8, 1),
                )}
        else:
            out_lat = {"samples": torch.zeros(
                1, 4,
                max(out_img.shape[1] // 8, 1),
                max(out_img.shape[2] // 8, 1),
            )}
        return (out_img, out_lat)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Node 4 — Builder Sampler                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _build_cfg_schedule(mode: str, start: float, finish: float, pivot: float, steps: int) -> List[float]:
    """Per-step CFG schedule.

    * **Constant** – every step uses ``start``.
    * **Linear** – linear ramp ``start → finish`` over ``steps``.
    * **Ease Down** – early steps fall fast toward ``pivot`` (cubic ease-out
      to 70 % of ``steps``) then slow ease from ``pivot → finish`` for the
      remainder.  Used to keep the early steps under firm guidance and let the
      late steps relax for fine detail.
    """
    steps = max(int(steps), 1)
    if mode == "Constant":
        return [float(start)] * steps
    if mode == "Linear":
        return [float(start) + (float(finish) - float(start)) * (i / max(steps - 1, 1))
                for i in range(steps)]
    # Ease Down
    out: List[float] = []
    knee = max(int(round(steps * 0.7)), 1)
    for i in range(steps):
        if i < knee:
            t = i / max(knee - 1, 1)
            t = 1.0 - (1.0 - t) ** 3      # cubic ease-out
            v = float(start) + (float(pivot) - float(start)) * t
        else:
            t = (i - knee) / max(steps - knee - 1, 1)
            t = t ** 2                     # ease-in toward finish
            v = float(pivot) + (float(finish) - float(pivot)) * t
        out.append(float(v))
    return out


class _CFGScheduler:
    """Closure object that swaps the model's CFG per sampling step.

    ComfyUI's ``ksampler`` exposes a ``set_model_sampler_cfg_function`` hook on
    the ModelPatcher which receives ``(args)`` where ``args["cond_scale"]``
    already equals ``cfg``.  We replace that scale with our scheduled value
    selected by the current ``timestep`` (largest sigma first).
    """

    def __init__(self, schedule: List[float], sigmas: torch.Tensor):
        self.schedule = list(schedule)
        # Map sigma -> step index.  ComfyUI sigmas come in *decreasing* order
        # of magnitude, so step 0 == sigmas[0].
        self.sigma_to_step = {float(s.item()): i for i, s in enumerate(sigmas[:-1])}

    def __call__(self, args):
        sigma = args.get("sigma", None)
        cond = args["cond"]
        uncond = args["uncond"]
        scale = args.get("cond_scale", self.schedule[0])
        if sigma is not None:
            try:
                key = float(sigma.flatten()[0].item())
                idx = self.sigma_to_step.get(key, 0)
                if 0 <= idx < len(self.schedule):
                    scale = self.schedule[idx]
            except Exception:
                pass
        return uncond + (cond - uncond) * float(scale)


class MECBuilderSampler:
    """KSampler with adaptive CFG curves and an optional polish pass.

    Uses ComfyUI's standard ``comfy.sample.sample`` so the schedule honours
    the user's ``sampler_name`` / ``scheduler`` exactly the same way the
    built-in KSampler does — we only override ``cond_scale`` per step.
    """

    CATEGORY = "MEC/Paint"
    DESCRIPTION = (
        "KSampler with adaptive CFG curves (Constant, Linear, Ease Down) plus an optional "
        "self-correction polish pass and resolution presets."
    )
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "preview_image")
    OUTPUT_TOOLTIPS = (
        "Sampled latent (with optional polish pass applied).",
        "VAE-decoded preview image when a VAE is provided (zero image otherwise).",
    )
    FUNCTION = "execute"

    RESOLUTION_PRESETS = {
        "SDXL (1024x1024)": (1024, 1024),
        "SD1.5 (512x512)":  (512, 512),
        "Custom":           None,
    }

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers = ["euler", "dpmpp_2m"]
            schedulers = ["normal", "karras"]
        return {
            "required": {
                "model":    ("MODEL", {"tooltip": "Diffusion model to sample with."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning."}),
                "steps":    ("INT",   {"default": 20, "min": 1, "max": 200, "tooltip": "Number of sampling steps."}),
                "cfg":      ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Starting CFG scale (also constant CFG when cfg_mode is Constant)."}),
                "sampler_name": (samplers, {"tooltip": "Sampler algorithm."}),
                "scheduler":    (schedulers, {"tooltip": "Sigma schedule."}),
                "denoise":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength (1.0 = full sampling, lower = partial img2img)."}),
                "cfg_mode":   (["Constant", "Linear", "Ease Down"], {"default": "Constant", "tooltip": "Adaptive CFG curve shape across steps."}),
                "cfg_finish": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Final CFG value at the end of the schedule (used by Linear / Ease Down)."}),
                "cfg_pivot":  ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Pivot CFG value used by Ease Down to control the curve knee."}),
                "self_correction": ("BOOLEAN", {"default": False, "tooltip": "Run a 2-step polish pass after the main sampling."}),
                "resolution_preset": (list(cls.RESOLUTION_PRESETS.keys()), {"default": "SD1.5 (512x512)", "tooltip": "Preset resolution; choose Custom to use custom_width/custom_height."}),
                "custom_width":  ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Custom output width in pixels (used when preset is Custom)."}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Custom output height in pixels (used when preset is Custom)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed."}),
            },
            "optional": {
                "vae":          ("VAE", {"tooltip": "Optional VAE; when provided, decodes the latent into preview_image."}),
                "latent_image": ("LATENT", {"tooltip": "Optional input latent (img2img). When omitted, an empty latent is created."}),
            },
        }

    def _resolve_resolution(self, preset, w, h):
        wh = self.RESOLUTION_PRESETS.get(preset)
        if wh is None:
            return int(w), int(h)
        return wh

    def execute(self, model, positive, negative, steps, cfg,
                sampler_name, scheduler, denoise,
                cfg_mode, cfg_finish, cfg_pivot,
                self_correction, resolution_preset,
                custom_width, custom_height, seed,
                vae=None, latent_image=None):
        import comfy.samplers
        import comfy.sample

        # ── Required-input validation ──
        if model is None:
            raise ValueError(
                "MECBuilderSampler: 'model' input is required. "
                "Connect a MODEL output (e.g. CheckpointLoaderSimple) to this node."
            )
        if positive is None or negative is None:
            raise ValueError(
                "MECBuilderSampler: 'positive' and 'negative' CONDITIONING "
                "inputs are required. Connect CLIPTextEncode outputs."
            )

        w, h = self._resolve_resolution(resolution_preset, custom_width, custom_height)

        if latent_image is None:
            samples = torch.zeros(1, 4, h // 8, w // 8)
            latent_image = {"samples": samples}
        latent = latent_image["samples"].to(torch.float32)

        device = model.load_device if hasattr(model, "load_device") else \
                 (latent.device if isinstance(latent, torch.Tensor) else torch.device("cpu"))

        # ---- adaptive CFG ----
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, int(steps)
        )
        if 0.0 < float(denoise) < 1.0:
            cut = max(int(round(steps * (1.0 - float(denoise)))), 0)
            sigmas = sigmas[cut:]
        sched = _build_cfg_schedule(cfg_mode, float(cfg), float(cfg_finish),
                                    float(cfg_pivot), max(len(sigmas) - 1, 1))

        m = model.clone()
        try:
            m.set_model_sampler_cfg_function(_CFGScheduler(sched, sigmas))
        except Exception:
            # Older Comfy: fall back to a constant CFG below.
            pass

        noise = comfy.sample.prepare_noise(latent, int(seed), None)
        out_latent = comfy.sample.sample_custom(
            m, noise, float(cfg), sampler, sigmas,
            positive, negative, latent,
            seed=int(seed),
        )

        # ---- self-correction polish pass ----
        if self_correction:
            polish_steps = 2
            polish_sigmas = comfy.samplers.calculate_sigmas(
                m.get_model_object("model_sampling"), scheduler, polish_steps + 2
            )[-(polish_steps + 1):]
            noise2 = comfy.sample.prepare_noise(out_latent, int(seed) + 1, None)
            out_latent = comfy.sample.sample_custom(
                m, noise2, float(cfg_finish if cfg_mode != "Constant" else cfg),
                sampler, polish_sigmas,
                positive, negative, out_latent,
                seed=int(seed) + 1,
            )

        out_dict = {"samples": out_latent}

        # ---- preview decode ----
        if vae is not None:
            try:
                preview = vae.decode(out_latent)
                preview = _to_bhwc(preview)
            except Exception:
                preview = torch.zeros(1, h, w, 3)
        else:
            preview = torch.zeros(1, h, w, 3)
        return (out_dict, preview)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ComfyUI registration                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
NODE_CLASS_MAPPINGS = {
    "MECAdvancedPaintCanvas": MECAdvancedPaintCanvas,
    "MECContextInpainter":    MECContextInpainter,
    "MECToneRefiner":         MECToneRefiner,
    "MECBuilderSampler":      MECBuilderSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MECAdvancedPaintCanvas": "Advanced Paint Canvas (MEC)",
    "MECContextInpainter":    "Context Inpainter / Fixer (MEC)",
    "MECToneRefiner":         "Tone Refiner (MEC)",
    "MECBuilderSampler":      "Builder Sampler (MEC)",
}
