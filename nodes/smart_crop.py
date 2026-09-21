"""Smart Image Crop / Stitch — mask-driven crop for detail work.

Ported from ComfyUI-Smart-Image-Crop-and-Stitch, which is itself a rework of
lquesada's Inpaint-CropAndStitch. This pack already carries that lineage as
InpaintCropProMEC / InpaintStitchProMEC, which are video-aware and carry a
canvas. These two are the STILLS path, and they do several things the other
pair does not:

  * a very light grey mask still counts. Masks arriving from a paint tool, a
    matte, or a JPEG round-trip are rarely a clean 1.0, and an exact test
    finds nothing and silently crops the whole frame.
  * holes get filled. A subject mask with gaps regenerates the gaps as
    background, which is the classic hole-in-the-face result.
  * the crop is taken at the TARGET aspect. Crop first and resize after, and
    the region is squashed before the model ever sees it.
  * dimensions snap to a VAE-friendly multiple, and the snap direction is
    chosen so an upscale never lands below the minimum it was asked for.

Two deliberate differences from upstream:

  * the stitcher info is a STITCHER, not a bare DICT, so it cannot be wired
    into an unrelated dict input and fail somewhere far away.
  * OpenCV is optional. Upstream degrades quietly to a torch path with
    different behaviour; here the torch path is the same algorithm, and the
    report says which ran, because "my holes did not fill" is otherwise
    impossible to diagnose from the outside.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import cv2
except Exception:  # noqa: BLE001 - optional, and its absence is reported
    cv2 = None

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None

CATEGORY = "C2C/Inpaint"

# A mask value this low is still a mask. Paint tools, mattes and anything that
# has been through a JPEG produce soft edges and stray low values; testing for
# exactly 1.0, or even > 0.5, throws away a real selection and crops the whole
# frame instead, which looks like the node ignoring the mask entirely.
MASK_EPS = 1.0 / 255.0

DIVISORS = [8, 16, 32, 64, 112, 128, 256]
NO_MASK_MODES = ["Bypass", "Resize Full Image", "Crop Full Image"]
BLEND_MODES = ["Box Feather", "Mask Feather", "Hard Paste"]
RESOLUTION_MODES = ["Automatic", "Manual"]


class SmartCropError(ValueError):
    pass


# ── geometry ────────────────────────────────────────────────────────────────

def snap_up(value: float, divisor: int) -> int:
    return max(divisor, int(math.ceil(float(value) / divisor) * divisor))


def snap_down(value: float, divisor: int) -> int:
    return max(divisor, int(math.floor(float(value) / divisor) * divisor))


def snap_nearest(value: float, divisor: int) -> int:
    return max(divisor, int(round(float(value) / divisor) * divisor))


def to_bhwc(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[0] in (1, 3, 4):
            return x.movedim(0, -1).unsqueeze(0)
        if x.shape[-1] in (1, 3, 4):
            return x.unsqueeze(0)
    return x


def mask_to_bhw(m: torch.Tensor) -> torch.Tensor:
    if m.ndim == 2:
        return m.unsqueeze(0)
    if m.ndim == 3 and m.shape[-1] == 1:
        return m.squeeze(-1)
    if m.ndim == 4 and m.shape[-1] == 1:
        return m.squeeze(-1).squeeze(1)
    return m


def mask_has_pixels(mask_2d: torch.Tensor, thresh: float = MASK_EPS) -> bool:
    return bool(torch.any(mask_2d > thresh).item())


def bbox_from_mask(mask_2d: torch.Tensor, thresh: float = MASK_EPS):
    """Tight bounds of everything above the threshold, or None."""
    idx = (mask_2d > thresh).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return None
    ys, xs = idx[:, 0], idx[:, 1]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def target_size(crop_w: float, crop_h: float, *, mode: str, min_res: int,
                max_res: int, manual_w: int, manual_h: int,
                divisor: int) -> tuple[int, int]:
    """The size the crop is resampled to before the model sees it.

    Automatic keeps the crop's aspect where it can and clamps it where it
    cannot, because a region far off square would otherwise demand a size that
    is both below the minimum on one axis and above the maximum on the other.

    The snap DIRECTION follows the reason for scaling: scaling up to reach a
    minimum snaps up, so the result is not left under the minimum it was asked
    for; scaling down to obey a maximum snaps down, for the same reason in
    reverse. Rounding to nearest in both cases quietly violates whichever
    bound was the point.
    """
    if mode == "Manual":
        return snap_up(manual_w, divisor), snap_up(manual_h, divisor)

    lo = snap_up(min_res, divisor)
    hi = max(lo, snap_down(max_res, divisor))
    aspect = float(crop_w) / float(crop_h)
    limit = hi / float(lo)

    if aspect > limit:
        return hi, lo
    if aspect < 1.0 / limit:
        return lo, hi

    shortest = min(crop_w, crop_h)
    longest = max(crop_w, crop_h)
    if shortest < lo:
        scale = lo / shortest
        return snap_up(crop_w * scale, divisor), snap_up(crop_h * scale, divisor)
    if longest > hi:
        scale = hi / longest
        return snap_down(crop_w * scale, divisor), snap_down(crop_h * scale, divisor)
    return snap_nearest(crop_w, divisor), snap_nearest(crop_h, divisor)


def widen_to_aspect(w: float, h: float, target_w: int, target_h: int):
    """Grow the crop (never shrink it) until its aspect matches the target.

    The crop has to be taken at the target's shape. Take it tight and resize
    afterwards and the region arrives at the model squashed - a face becomes a
    wide face, and the model faithfully generates a wide face.

    Growing rather than cropping to fit, because shrinking would cut off part
    of the very region the mask asked for.
    """
    target_ar = float(target_w) / float(target_h)
    current_ar = float(w) / float(h)
    if current_ar > target_ar:
        return w, w / target_ar
    return h * target_ar, h


def edge_pad(image_bhwc: torch.Tensor, x: int, y: int, w: int, h: int):
    """Crop, extending past the frame edge by repeating the edge pixel.

    A crop near the border legitimately runs outside the image. Zero-padding
    there puts a black band in the model's input and it generates content to
    match it; repeating the edge is wrong too, but wrong in a way the model
    ignores.
    """
    B, H, W, C = image_bhwc.shape
    pad_l, pad_t = max(0, -x), max(0, -y)
    pad_r, pad_b = max(0, (x + w) - W), max(0, (y + h) - H)

    read_x, read_y = max(0, x), max(0, y)
    read_w, read_h = w - pad_l - pad_r, h - pad_t - pad_b

    canvas = torch.zeros((B, h, w, C), device=image_bhwc.device,
                         dtype=image_bhwc.dtype)
    if read_w <= 0 or read_h <= 0:
        return canvas, (pad_l, pad_t)

    canvas[:, pad_t:pad_t + read_h, pad_l:pad_l + read_w, :] = \
        image_bhwc[:, read_y:read_y + read_h, read_x:read_x + read_w, :]
    if pad_t > 0:
        canvas[:, :pad_t, :, :] = canvas[:, pad_t:pad_t + 1, :, :]
    if pad_b > 0:
        canvas[:, -pad_b:, :, :] = canvas[:, -pad_b - 1:-pad_b, :, :]
    if pad_l > 0:
        canvas[:, :, :pad_l, :] = canvas[:, :, pad_l:pad_l + 1, :]
    if pad_r > 0:
        canvas[:, :, -pad_r:, :] = canvas[:, :, -pad_r - 1:-pad_r, :]
    return canvas, (pad_l, pad_t)


def resize_img(img_bhwc: torch.Tensor, tw: int, th: int,
               mode: str = "bicubic") -> torch.Tensor:
    """bicubic going up, area coming down — area avoids the aliasing a
    bicubic downscale leaves on hair and fine detail."""
    t = img_bhwc.movedim(-1, 1)
    align = None if mode in ("nearest", "area") else False
    t = F.interpolate(t, size=(th, tw), mode=mode, align_corners=align)
    return t.movedim(1, -1)


def resize_mask(mask_bhw: torch.Tensor, tw: int, th: int) -> torch.Tensor:
    t = F.interpolate(mask_bhw.unsqueeze(1), size=(th, tw), mode="nearest")
    return t.squeeze(1)


# ── mask shaping ────────────────────────────────────────────────────────────

def morph_close(mask_bhw: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    """Dilate then erode: closes gaps without growing the outline."""
    if kernel < 3:
        return mask_bhw
    pad = kernel // 2
    m = mask_bhw.unsqueeze(1)
    m = F.max_pool2d(m, kernel_size=kernel, stride=1, padding=pad)
    m = -F.max_pool2d(-m, kernel_size=kernel, stride=1, padding=pad)
    return m.squeeze(1)


def grow_or_shrink(mask_bhw: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels == 0:
        return mask_bhw
    kernel = abs(int(pixels)) * 2 + 1
    if kernel < 3:
        return mask_bhw
    pad = kernel // 2
    m = mask_bhw.unsqueeze(1)
    if pixels > 0:
        m = F.max_pool2d(m, kernel_size=kernel, stride=1, padding=pad)
    else:
        m = -F.max_pool2d(-m, kernel_size=kernel, stride=1, padding=pad)
    return (m.squeeze(1) > 0).float()


def fill_holes_torch(mask_bhw: torch.Tensor, iterations: int = 64) -> torch.Tensor:
    """Fill ENCLOSED holes without OpenCV.

    Flood the outside from the border and keep whatever the flood never
    reached. Implemented as a dilation of the background repeatedly masked
    back against the not-mask, which is a flood fill written in max-pools.

    `iterations` bounds the work: a hole wider than 2*iterations pixels is
    left open rather than letting a big frame run for thousands of passes. It
    is reported, not hidden.
    """
    solid = (mask_bhw > MASK_EPS).float()
    background = 1.0 - solid

    outside = torch.zeros_like(background)
    outside[:, 0, :] = background[:, 0, :]
    outside[:, -1, :] = background[:, -1, :]
    outside[:, :, 0] = background[:, :, 0]
    outside[:, :, -1] = background[:, :, -1]

    for _ in range(max(1, iterations)):
        grown = F.max_pool2d(outside.unsqueeze(1), kernel_size=3, stride=1,
                             padding=1).squeeze(1)
        grown = grown * background          # the flood cannot cross the mask
        if torch.equal(grown, outside):
            break
        outside = grown

    # anything that is background but the flood never reached is enclosed
    return torch.clamp(solid + (background - outside), 0.0, 1.0)


def fill_holes_cv(mask_bhw: torch.Tensor) -> torch.Tensor:
    """The same thing via OpenCV's floodFill, which has no iteration bound."""
    if cv2 is None or np is None:
        return fill_holes_torch(mask_bhw)
    solid = (mask_bhw > MASK_EPS).float()
    arr = solid.detach().cpu().numpy().astype(np.uint8)
    out = []
    for frame in arr:
        u8 = frame * 255
        padded = np.pad(u8, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        cv2.floodFill(padded, None, (0, 0), 255)
        holes = (padded[1:-1, 1:-1] == 0).astype(np.uint8) * 255
        out.append(np.maximum(u8, holes))
    stacked = np.stack(out, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stacked).to(device=mask_bhw.device,
                                        dtype=mask_bhw.dtype)


def prepare_mask(mask: torch.Tensor, *, patch_holes: bool,
                 grow_pixels: int) -> tuple[torch.Tensor, str]:
    """Solidify, close, fill, grow. Returns the mask and which path ran."""
    raw = mask_to_bhw(mask).contiguous().float()
    solid = (raw > MASK_EPS).float()
    backend = "OpenCV" if cv2 is not None else "torch"
    if patch_holes:
        solid = morph_close(solid, kernel=9)
        solid = fill_holes_cv(solid) if cv2 is not None else fill_holes_torch(solid)
    return grow_or_shrink(solid, grow_pixels), backend


# ── blending ────────────────────────────────────────────────────────────────

def gaussian_kernel1d(sigma: float, size: int) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(mask_bhw: torch.Tensor, blur_px: int = 16) -> torch.Tensor:
    if blur_px <= 0:
        return mask_bhw
    _, h, w = mask_bhw.shape
    if h <= 1 or w <= 1:
        return mask_bhw

    sigma = float(blur_px) * 0.33
    size = 2 * int(3.0 * sigma + 0.5) + 1
    size = min(size if size % 2 else size + 1, max(3, 2 * min(h, w) - 1))
    if size % 2 == 0:
        size -= 1
    sigma = max(0.1, min(sigma, size / 6.0))

    k = gaussian_kernel1d(sigma, size).to(mask_bhw.device, dtype=mask_bhw.dtype)
    x = mask_bhw.unsqueeze(1)
    pad = size // 2
    pad_mode = "reflect" if pad < h and pad < w else "replicate"
    x = F.pad(x, (pad, pad, 0, 0), mode=pad_mode)
    x = F.conv2d(x, k.view(1, 1, 1, size))
    x = F.pad(x, (0, 0, pad, pad), mode=pad_mode)
    x = F.conv2d(x, k.view(1, 1, size, 1))
    return torch.clamp(x.squeeze(1), 0.0, 1.0)


def color_match(image_bhwc: torch.Tensor, reference_bhwc: torch.Tensor,
                amount: float = 0.0,
                mask_bhw: torch.Tensor | None = None) -> torch.Tensor:
    """Pull the crop's mean and spread toward the plate's, inside the mask.

    A generated crop is usually slightly off the plate in exposure or cast,
    and a feathered seam turns that into a visible halo rather than hiding it.
    Weighted by the blend mask so the statistics come from the region actually
    being pasted, not from the feathered surround.
    """
    amount = float(max(0.0, min(1.0, amount)))
    if amount <= 0.0:
        return image_bhwc

    img, ref = image_bhwc.float(), reference_bhwc.float()
    eps = 1e-5
    weights = None
    if mask_bhw is not None:
        w = torch.clamp(mask_bhw.float(), 0.0, 1.0).unsqueeze(-1)
        if float(w.sum().item()) > eps:
            weights = w

    if weights is None:
        img_mean = img.mean(dim=(1, 2), keepdim=True)
        ref_mean = ref.mean(dim=(1, 2), keepdim=True)
        img_std = img.std(dim=(1, 2), keepdim=True).clamp_min(eps)
        ref_std = ref.std(dim=(1, 2), keepdim=True).clamp_min(eps)
    else:
        denom = weights.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
        img_mean = (img * weights).sum(dim=(1, 2), keepdim=True) / denom
        ref_mean = (ref * weights).sum(dim=(1, 2), keepdim=True) / denom
        img_var = (((img - img_mean) ** 2) * weights).sum(dim=(1, 2), keepdim=True) / denom
        ref_var = (((ref - ref_mean) ** 2) * weights).sum(dim=(1, 2), keepdim=True) / denom
        img_std = torch.sqrt(img_var).clamp_min(eps)
        ref_std = torch.sqrt(ref_var).clamp_min(eps)

    matched = (img - img_mean) * (ref_std / img_std) + ref_mean
    return torch.clamp(torch.lerp(img, matched, amount), 0.0, 1.0).to(image_bhwc.dtype)


def box_feather_mask(h: int, w: int, feather: int, device, dtype) -> torch.Tensor:
    """A soft-edged rectangle for blending a whole crop back in."""
    m = torch.ones((1, h, w), dtype=dtype, device=device)
    if feather > 0:
        margin = min(feather // 2, max(0, min(h, w) // 2 - 1))
        if margin > 0:
            m[:, :margin, :] = 0
            m[:, -margin:, :] = 0
            m[:, :, :margin] = 0
            m[:, :, -margin:] = 0
        m = gaussian_blur(m, blur_px=feather)
    return m


# ── the nodes ───────────────────────────────────────────────────────────────

def _stitcher(bypass: bool, size: tuple[int, int], *, full_resize: bool = False,
              **lists) -> dict:
    base = {"bypass": bypass, "original_size": size,
            "full_image_resize": full_resize,
            "x": [], "y": [], "w": [], "h": [],
            "padL": [], "padT": [], "target_w": [], "target_h": []}
    base.update(lists)
    return base


class SmartImageCropMEC:
    """Mask-driven crop for stills: find the subject, crop at the right shape,
    resample to a size the VAE is happy with."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "resolution_mode": (RESOLUTION_MODES, {"default": "Automatic"}),
                "max_resolution": ("INT", {
                    "default": 2048, "min": 256, "max": 16384, "step": 64,
                    "tooltip": "Upper limit on the longest side. A crop larger "
                               "than this is scaled DOWN to fit."}),
                "min_resolution": ("INT", {
                    "default": 768, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Lower limit on the shortest side. A crop "
                               "smaller than this is scaled UP, which is the "
                               "whole point of cropping - a 200px face gets "
                               "the model's full attention."}),
                "manual_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "manual_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "mask_grow_pixels": ("INT", {
                    "default": 32, "min": -1024, "max": 1024, "step": 8,
                    "tooltip": "Grow (or shrink) the mask before measuring the "
                               "crop, so the model sees context around the "
                               "region rather than a tight cut-out."}),
                "patch_mask_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Close gaps and fill enclosed holes. A subject "
                               "mask with holes regenerates those holes as "
                               "background - the hole-in-the-face result."}),
                "no_mask_mode": (NO_MASK_MODES, {
                    "default": "Bypass",
                    "tooltip": "What to do when the mask is empty. Bypass "
                               "passes the image through untouched, which is "
                               "the safe default for a batch where only some "
                               "frames have a mask."}),
                "force_divisibility": (DIVISORS, {
                    "default": 128,
                    "tooltip": "Snap output dimensions to this multiple. A VAE "
                               "silently pads anything else, which shifts the "
                               "crop by a few pixels and breaks the stitch."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STITCHER", "IMAGE", "STRING")
    RETURN_NAMES = ("crop_image", "crop_mask", "stitcher", "preview_overlay", "info")
    FUNCTION = "crop"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Crop what the mask points at, at the aspect it will be "
                   "generated in, scaled to a size the VAE accepts.")

    @classmethod
    def IS_CHANGED(cls, image, mask, **kw):
        import hashlib
        h = hashlib.md5()
        h.update(image.detach().cpu().numpy().tobytes())
        h.update(mask.detach().cpu().numpy().tobytes())
        h.update(repr(sorted(kw.items())).encode())
        return h.hexdigest()

    def crop(self, image, mask, resolution_mode, max_resolution, min_resolution,
             manual_width, manual_height, mask_grow_pixels, patch_mask_holes,
             no_mask_mode, force_divisibility):
        if min_resolution > max_resolution:
            raise SmartCropError(
                f"min_resolution ({min_resolution}) is above max_resolution "
                f"({max_resolution}). Nothing can satisfy both.")

        img = to_bhwc(image).contiguous()
        msk, backend = prepare_mask(mask, patch_holes=patch_mask_holes,
                                    grow_pixels=mask_grow_pixels)
        B, H, W, _ = img.shape
        divisor = int(force_divisibility)

        notes = []
        if patch_mask_holes and cv2 is None:
            notes.append(
                "OpenCV is not installed, so holes were filled with the torch "
                "flood fill. Same algorithm, but it gives up on holes wider "
                "than ~128px - install opencv-python if a large hole survives.")

        if not any(mask_has_pixels(msk[b % msk.shape[0]]) for b in range(B)):
            return self._no_mask(img, msk, no_mask_mode, max_resolution,
                                 divisor, notes)

        # ONE target size for the whole batch.
        #
        # A batch is a single tensor, so every frame's crop must come out the
        # same size. Sizing each frame from its own mask looks correct frame
        # by frame and then fails at torch.cat with a shape error that says
        # nothing about masks - and on the days every mask happens to be the
        # same size it does not fail at all, which hides it until it matters.
        # The size comes from the LARGEST region, so nothing is under-resolved.
        boxes = [bbox_from_mask(msk[b % msk.shape[0]]) for b in range(B)]
        sized = [bb for bb in boxes if bb is not None]
        ref_w = max(bb[2] for bb in sized)
        ref_h = max(bb[3] for bb in sized)
        tw, th = target_size(ref_w, ref_h, mode=resolution_mode,
                             min_res=min_resolution, max_res=max_resolution,
                             manual_w=manual_width, manual_h=manual_height,
                             divisor=divisor)

        crops, crop_masks = [], []
        sx, sy, sw, sh, spl, spt, stw, sth = [], [], [], [], [], [], [], []
        overlay = img.clone()
        green = torch.tensor([0.0, 1.0, 0.0], device=img.device, dtype=img.dtype)
        red = torch.tensor([1.0, 0.0, 0.0], device=img.device, dtype=img.dtype)
        grew = 0
        unmasked = 0

        for b in range(B):
            mi = b % msk.shape[0]
            bbox = boxes[b]
            if bbox is None:
                # No mask on this frame, but another frame in the batch has
                # one. Centre the same window on the frame: dropping it would
                # silently shorten the batch, and passing it through at full
                # size cannot be concatenated with the others.
                unmasked += 1
                bbox = [0, 0, W, H]
            x, y, w, h = bbox
            cx, cy = x + w / 2.0, y + h / 2.0

            new_w, new_h = widen_to_aspect(w, h, tw, th)
            if new_w > w + 0.5 or new_h > h + 0.5:
                grew += 1

            crop_x, crop_y = int(cx - new_w / 2), int(cy - new_h / 2)
            crop_w, crop_h = max(1, int(new_w)), max(1, int(new_h))

            canvas, (pad_l, pad_t) = edge_pad(img[b:b + 1], crop_x, crop_y,
                                              crop_w, crop_h)
            up = tw > crop_w or th > crop_h
            crops.append(resize_img(canvas, tw, th,
                                    mode="bicubic" if up else "area"))

            mask_canvas, _ = edge_pad(msk[mi:mi + 1].unsqueeze(-1), crop_x,
                                      crop_y, crop_w, crop_h)
            crop_masks.append(resize_mask(mask_canvas[..., 0], tw, th))

            sx.append(crop_x); sy.append(crop_y)
            sw.append(crop_w); sh.append(crop_h)
            spl.append(pad_l); spt.append(pad_t)
            stw.append(tw); sth.append(th)

            region = overlay[b, y:y + h, x:x + w, :]
            tint = msk[mi, y:y + h, x:x + w].unsqueeze(-1)
            overlay[b, y:y + h, x:x + w, :] = torch.lerp(region, red, tint * 0.35)
            dx, dy = max(0, crop_x), max(0, crop_y)
            dw, dh = min(W - dx, crop_w), min(H - dy, crop_h)
            if dw > 0 and dh > 0:
                overlay[b, dy:dy + 2, dx:dx + dw, :] = green
                overlay[b, dy + dh - 2:dy + dh, dx:dx + dw, :] = green
                overlay[b, dy:dy + dh, dx:dx + 2, :] = green
                overlay[b, dy:dy + dh, dx + dw - 2:dx + dw, :] = green

        stitcher = _stitcher(False, (W, H), x=sx, y=sy, w=sw, h=sh,
                             padL=spl, padT=spt, target_w=stw, target_h=sth)

        info = [f"Cropped {len(sx)} frame(s). Hole filling: {backend}."]
        if stw:
            info.append(f"Output {stw[0]}x{sth[0]} from a {sw[0]}x{sh[0]} region "
                        f"(scale {stw[0] / max(1, sw[0]):.2f}x).")
        widths = {bb[2] for bb in sized}
        heights = {bb[3] for bb in sized}
        if len(sized) > 1 and (len(widths) > 1 or len(heights) > 1):
            info.append(
                "The masks in this batch are different sizes. One output size "
                f"({tw}x{th}) was used for all of them, taken from the largest, "
                "because a batch is a single tensor. Smaller regions are "
                "scaled up further than they would be on their own.")
        if unmasked:
            info.append(
                f"{unmasked} frame(s) had no mask. They were centre-cropped to "
                "the same window so the batch keeps its length.")
        if grew:
            info.append(
                f"{grew} crop(s) were widened to match the output aspect. "
                "Cropping tight and resizing after would squash the region "
                "before the model ever saw it.")
        info.extend(notes)
        return (torch.cat(crops, dim=0), torch.cat(crop_masks, dim=0),
                stitcher, overlay, "\n".join(info))

    def _no_mask(self, img, msk, mode, max_resolution, divisor, notes):
        B, H, W, _ = img.shape
        passthrough = torch.cat([msk[b % msk.shape[0]:b % msk.shape[0] + 1]
                                 for b in range(B)], dim=0)
        head = "The mask is empty. "

        if mode == "Resize Full Image":
            if W >= H:
                tw = snap_down(max_resolution, divisor)
                th = snap_up(tw * H / W, divisor)
            else:
                th = snap_down(max_resolution, divisor)
                tw = snap_up(th * W / H, divisor)
            resized = resize_img(img, tw, th,
                                 mode="bicubic" if tw > W else "area")
            ones = torch.ones((B, th, tw), device=img.device, dtype=msk.dtype)
            st = _stitcher(False, (W, H), full_resize=True,
                           x=[0] * B, y=[0] * B, w=[W] * B, h=[H] * B,
                           padL=[0] * B, padT=[0] * B,
                           target_w=[tw] * B, target_h=[th] * B)
            return (resized, ones, st, img.clone(),
                    "\n".join([head + f"Resized the whole frame to {tw}x{th}."] + notes))

        if mode == "Crop Full Image":
            ones = torch.ones((B, H, W), device=img.device, dtype=msk.dtype)
            st = _stitcher(False, (W, H),
                           x=[0] * B, y=[0] * B, w=[W] * B, h=[H] * B,
                           padL=[0] * B, padT=[0] * B,
                           target_w=[W] * B, target_h=[H] * B)
            return (img, ones, st, img.clone(),
                    "\n".join([head + "Using the whole frame as the crop."] + notes))

        return (img, passthrough, _stitcher(True, (W, H)), img.clone(),
                "\n".join([head + "Bypassing: the image passes through "
                           "untouched and the stitcher will do nothing."] + notes))


class SmartImageStitcherMEC:
    """Put the processed crop back where it came from."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "stitcher": ("STITCHER",),
                "feather_pixels": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
                "blend_mode": (BLEND_MODES, {
                    "default": "Box Feather",
                    "tooltip": "Box Feather softens the crop's rectangular "
                               "edge. Mask Feather blends along the mask "
                               "itself, which keeps untouched detail outside "
                               "the subject. Hard Paste shows the seam and is "
                               "for checking alignment."}),
                "resize_full_image_output": (
                    ["Restore Original Size", "Keep Resized Image"],
                    {"default": "Restore Original Size",
                     "tooltip": "Only applies when the crop node ran in "
                                "'Resize Full Image' mode."}),
                "enable_color_match": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pull the crop's exposure and cast toward the "
                               "plate before blending. A feathered seam turns "
                               "a small mismatch into a visible halo instead "
                               "of hiding it."}),
                "color_match_amount": ("FLOAT", {"default": 0.35, "min": 0.0,
                                                 "max": 1.0, "step": 0.01}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "stitch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Blend the processed crop back into the original frame."

    @classmethod
    def IS_CHANGED(cls, original_image, processed_image, stitcher, **kw):
        import hashlib
        h = hashlib.md5()
        h.update(original_image.detach().cpu().numpy().tobytes())
        h.update(processed_image.detach().cpu().numpy().tobytes())
        h.update(repr(stitcher.get("x") if isinstance(stitcher, dict) else None).encode())
        h.update(repr(sorted((k, str(v)) for k, v in kw.items())).encode())
        return h.hexdigest()

    def stitch(self, original_image, processed_image, stitcher, feather_pixels,
               blend_mode="Box Feather",
               resize_full_image_output="Restore Original Size",
               enable_color_match=False, color_match_amount=0.35, mask=None):
        # clone: the input can be an inference tensor, which cannot be written to
        base = to_bhwc(original_image).contiguous().clone()
        processed = to_bhwc(processed_image).contiguous()

        if not isinstance(stitcher, dict):
            raise SmartCropError(
                "The stitcher input is not a stitcher. Wire it from Smart Image "
                "Crop's 'stitcher' output.")
        if stitcher.get("bypass", False):
            return (base, "Stitcher is in bypass (the crop found no mask). "
                          "The original image passes through.")

        missing = [k for k in ("x", "y", "w", "h") if k not in stitcher]
        if missing:
            raise SmartCropError(
                f"The stitcher is missing {', '.join(missing)}. It did not come "
                "from Smart Image Crop.")

        if stitcher.get("full_image_resize", False) and \
                resize_full_image_output == "Keep Resized Image":
            return (processed, "Kept the resized full image at its new size.")

        mask_input = None
        note = ""
        if blend_mode == "Mask Feather":
            if mask is None:
                # Silently falling back would look like the feather doing
                # nothing, so it is stated.
                blend_mode = "Box Feather"
                note = ("Mask Feather was selected but no mask is connected, so "
                        "Box Feather was used instead. Wire the crop node's "
                        "crop_mask in to blend along the subject.")
            else:
                mask_input = mask_to_bhw(mask).contiguous()

        sx, sy, sw, sh = stitcher["x"], stitcher["y"], stitcher["w"], stitcher["h"]
        if not sx:
            return (base, "The stitcher holds no regions; nothing to do.")

        B_base = base.shape[0]
        B_proc = processed.shape[0]
        if B_proc == 0:
            return (base, "The processed image is empty; nothing to stitch.")

        pasted = 0
        for b in range(min(B_base, len(sx))):
            x, y, w, h = sx[b], sy[b], sw[b], sh[b]
            if w <= 0 or h <= 0:
                continue
            patch = processed[b % B_proc: b % B_proc + 1]
            # area on the way back down: the model returned a larger crop than
            # the slot it goes into, and bicubic would alias the fine detail
            # that was the reason for cropping in the first place.
            patch_resized = resize_img(patch, w, h, mode="area")

            if blend_mode == "Mask Feather" and mask_input is not None:
                mi = b % mask_input.shape[0]
                m = resize_mask(mask_input[mi:mi + 1], w, h)
                if feather_pixels > 0:
                    m = gaussian_blur(m, blur_px=feather_pixels)
            elif blend_mode == "Hard Paste":
                m = torch.ones((1, h, w), dtype=torch.float32, device=base.device)
            else:
                m = box_feather_mask(h, w, feather_pixels, base.device,
                                     torch.float32)

            if enable_color_match and color_match_amount > 0:
                reference, _ = edge_pad(base[b:b + 1], x, y, w, h)
                patch_resized = color_match(patch_resized, reference,
                                            amount=color_match_amount,
                                            mask_bhw=m)

            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(base.shape[2], x + w), min(base.shape[1], y + h)
            px0, py0 = max(0, -x), max(0, -y)
            vw, vh = x1 - x0, y1 - y0
            if vw <= 0 or vh <= 0:
                continue

            p = patch_resized[0, py0:py0 + vh, px0:px0 + vw, :]
            mm = m.unsqueeze(-1)[0, py0:py0 + vh, px0:px0 + vw, :].to(p.dtype)
            base[b, y0:y1, x0:x1, :] = p * mm + base[b, y0:y1, x0:x1, :] * (1.0 - mm)
            pasted += 1

        info = [f"Stitched {pasted} region(s) with {blend_mode.lower()}"
                f"{f', feather {feather_pixels}px' if feather_pixels else ''}."]
        if enable_color_match and color_match_amount > 0:
            info.append(f"Colour matched to the plate at {color_match_amount:.2f}.")
        if note:
            info.append(note)
        return (base, "\n".join(info))


NODE_CLASS_MAPPINGS = {
    "SmartImageCropMEC": SmartImageCropMEC,
    "SmartImageStitcherMEC": SmartImageStitcherMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartImageCropMEC": "Smart Image Crop",
    "SmartImageStitcherMEC": "Smart Image Stitch",
}
