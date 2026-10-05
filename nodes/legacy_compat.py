"""Deprecated node classes kept so saved workflows load (issue #11, ledger L2.02).

Behaviour copied verbatim from commit 71b56e8^ (parent bf187018). New graphs
should use the unified / core replacements named in each class DESCRIPTION.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from .utils import normalize_bbox as _norm_bbox

_LEGACY_CATEGORY = "\U0001f43a C2C/\U0001f9f0 Core/Legacy"


class MaskPreviewOverlay:
    """Generate a preview image with the mask overlaid on the source image.
    Supports custom overlay colour, edge highlight, bbox drawing, and
    side-by-side comparison."""

    DEPRECATED = True

    DISPLAY_MODES = ["overlay", "mask_only", "side_by_side", "checkerboard", "edge_highlight"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "display_mode": (cls.DISPLAY_MODES, {"default": "overlay"}),
                "overlay_color_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_color_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "opacity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_width": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1,
                                      "tooltip": "Edge contour width in pixels"}),
                "show_bbox": ("BOOLEAN", {"default": False, "tooltip": "Draw bounding box of mask region"}),
                "bbox_color_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_color_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "bbox": ("BBOX", {"tooltip": "External bbox to draw (overrides auto-detect)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = _LEGACY_CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Deprecated — kept so old workflows load. Closest current node: MaskOpsMEC preview output."
    )

    def preview(self, image, mask, display_mode, overlay_color_r, overlay_color_g,
                overlay_color_b, opacity, edge_width, show_bbox,
                bbox_color_r, bbox_color_g, bbox_color_b, bbox=None):

        B, H, W, C = image.shape
        m = mask.clone().to(image.device)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[0] < B:
            if m.shape[0] == 1:
                m = m.expand(B, -1, -1).clone()
            else:
                repeats = (B + m.shape[0] - 1) // m.shape[0]
                m = m.repeat(repeats, 1, 1)[:B]
        if m.shape[1] != H or m.shape[2] != W:
            m = F.interpolate(m.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False).squeeze(1)

        color = torch.tensor([overlay_color_r, overlay_color_g, overlay_color_b],
                             device=image.device, dtype=image.dtype)
        bbox_clr = torch.tensor([bbox_color_r, bbox_color_g, bbox_color_b],
                                device=image.device, dtype=image.dtype)

        out = image.clone()

        if display_mode == "side_by_side":
            out = torch.zeros(B, H, W * 2, 3, device=image.device, dtype=image.dtype)

        for i in range(B):
            frame = image[i].clone()
            mi = m[i]

            if display_mode == "overlay":
                mask_3d = mi.unsqueeze(-1)
                overlay = color.view(1, 1, 3).expand(H, W, 3)
                frame = frame * (1 - mask_3d * opacity) + overlay * mask_3d * opacity

            elif display_mode == "mask_only":
                frame = mi.unsqueeze(-1).expand(H, W, 3)

            elif display_mode == "side_by_side":
                mask_vis = mi.unsqueeze(-1).expand(H, W, 3)
                frame_rgb = frame[:, :, :3] if frame.shape[-1] > 3 else frame
                frame = torch.cat([frame_rgb, mask_vis], dim=1)

            elif display_mode == "checkerboard":
                checker = self._checkerboard(H, W, 16, image.device)
                bg = checker.unsqueeze(-1).expand(H, W, 3) * 0.5 + 0.25
                mask_3d = mi.unsqueeze(-1)
                frame = frame * mask_3d + bg * (1 - mask_3d)

            elif display_mode == "edge_highlight":
                edge = self._detect_edge(mi, edge_width)
                edge_3d = edge.unsqueeze(-1)
                overlay = color.view(1, 1, 3).expand(H, W, 3)
                frame = frame * (1 - edge_3d) + overlay * edge_3d

            if display_mode not in ("edge_highlight", "side_by_side") and edge_width > 0:
                edge = self._detect_edge(mi, edge_width)
                edge_3d = edge.unsqueeze(-1)
                overlay_edge = color.view(1, 1, 3).expand(H, W, 3)
                frame = frame * (1 - edge_3d) + overlay_edge * edge_3d

            if show_bbox:
                box = self._get_bbox(mi, bbox)
                if box is not None:
                    frame = self._draw_bbox(frame, box, bbox_clr, 2)

            if display_mode == "side_by_side":
                out[i] = frame
            else:
                out[i] = frame

        return (out.clamp(0, 1),)

    @staticmethod
    def _detect_edge(mask, width):
        if width <= 0:
            return torch.zeros_like(mask)
        m = mask.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, 2 * width + 1, 2 * width + 1, device=mask.device)
        dilated = F.conv2d(F.pad(m, (width, width, width, width), mode="constant", value=0),
                           kernel, padding=0)
        dilated = (dilated > 0.5).float()
        eroded = F.conv2d(F.pad(m, (width, width, width, width), mode="constant", value=1),
                          kernel, padding=0)
        total = kernel.numel()
        eroded = (eroded >= total).float()
        edge = (dilated - eroded).squeeze(0).squeeze(0).clamp(0, 1)
        return edge

    @staticmethod
    def _checkerboard(h, w, size, device):
        y = torch.arange(h, device=device) // size
        x = torch.arange(w, device=device) // size
        return ((y.unsqueeze(1) + x.unsqueeze(0)) % 2).float()

    @staticmethod
    def _get_bbox(mask, external_bbox):
        if external_bbox is not None:
            return external_bbox
        coords = torch.nonzero(mask > 0.5, as_tuple=False)
        if coords.shape[0] == 0:
            return None
        y_min = int(coords[:, 0].min())
        y_max = int(coords[:, 0].max())
        x_min = int(coords[:, 1].min())
        x_max = int(coords[:, 1].max())
        return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

    @staticmethod
    def _draw_bbox(frame, bbox, color, thickness):
        x, y, bw, bh = bbox
        H, W, C = frame.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W - 1, x + bw), min(H - 1, y + bh)
        t = thickness
        frame[y1:y1 + t, x1:x2, :] = color
        frame[max(0, y2 - t):y2, x1:x2, :] = color
        frame[y1:y2, x1:x1 + t, :] = color
        frame[y1:y2, max(0, x2 - t):x2, :] = color
        return frame


class MaskCompositeAdvanced:
    """Combine two masks using various boolean / blending operations with
    optional per-mask invert and output threshold."""

    DEPRECATED = True

    OPERATIONS = ["union", "intersect", "subtract", "xor", "blend", "min", "max", "difference"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "operation": (cls.OPERATIONS, {"default": "union"}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Blend ratio (only for 'blend' mode). 0=all A, 1=all B"}),
                "invert_a": ("BOOLEAN", {"default": False}),
                "invert_b": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Binarize output (0 = keep soft)"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "composite"
    CATEGORY = _LEGACY_CATEGORY
    DESCRIPTION = (
        "Deprecated — kept so old workflows load. Closest current node: core MaskComposite "
        "(fewer operations)."
    )

    def composite(self, mask_a, mask_b, operation, blend_factor,
                  invert_a, invert_b, threshold):
        a = mask_a.clone()
        b = mask_b.clone()

        if a.dim() == 2:
            a = a.unsqueeze(0)
        if b.dim() == 2:
            b = b.unsqueeze(0)

        if a.shape[0] != b.shape[0]:
            target_b = max(a.shape[0], b.shape[0])
            if a.shape[0] < target_b:
                a = a.expand(target_b, -1, -1).clone()
            if b.shape[0] < target_b:
                b = b.expand(target_b, -1, -1).clone()

        if a.shape[1:] != b.shape[1:]:
            b = F.interpolate(b.unsqueeze(1), size=a.shape[1:],
                              mode="bilinear", align_corners=False).squeeze(1)

        if invert_a:
            a = 1.0 - a
        if invert_b:
            b = 1.0 - b

        if operation == "union":
            out = torch.max(a, b)
        elif operation == "intersect":
            out = torch.min(a, b)
        elif operation == "subtract":
            out = (a - b).clamp(0, 1)
        elif operation == "xor":
            out = ((a > 0.5).float() + (b > 0.5).float()) % 2
        elif operation == "blend":
            out = a * (1 - blend_factor) + b * blend_factor
        elif operation == "min":
            out = torch.min(a, b)
        elif operation == "max":
            out = torch.max(a, b)
        elif operation == "difference":
            out = torch.abs(a - b)
        else:
            out = torch.max(a, b)

        if threshold > 0:
            out = (out >= threshold).float()

        return (out.clamp(0, 1),)


class MaskMath:
    """Perform mathematical operations on masks for fine-grained control."""

    DEPRECATED = True

    OPERATIONS = [
        "add_scalar", "multiply_scalar", "power", "invert",
        "clamp", "remap_range", "quantize", "threshold_hysteresis",
        "gamma", "contrast", "abs_diff_from_value",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (cls.OPERATIONS, {"default": "invert"}),
                "value_a": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                      "tooltip": "Primary parameter (meaning depends on operation)"}),
                "value_b": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                      "tooltip": "Secondary parameter (meaning depends on operation)"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "compute"
    CATEGORY = _LEGACY_CATEGORY
    DESCRIPTION = "Deprecated — kept so old workflows load. No unified successor."

    def compute(self, mask, operation, value_a, value_b):
        m = mask.clone()

        if operation == "add_scalar":
            m = m + value_a
        elif operation == "multiply_scalar":
            m = m * value_a
        elif operation == "power":
            m = m.clamp(0, 1).pow(max(0.01, value_a))
        elif operation == "invert":
            m = 1.0 - m
        elif operation == "clamp":
            m = m.clamp(value_a, value_b)
        elif operation == "remap_range":
            # Remap [value_a, value_b] → [0, 1]
            low, high = min(value_a, value_b), max(value_a, value_b)
            if high - low > 1e-6:
                m = (m - low) / (high - low)
            m = m.clamp(0, 1)
        elif operation == "quantize":
            levels = max(2, int(value_a))
            m = (m * (levels - 1)).round() / (levels - 1)
        elif operation == "threshold_hysteresis":
            # value_a = low threshold, value_b = high threshold
            high_mask = (m >= value_b).float()
            low_mask = (m >= value_a).float()
            # Simple connected-component-free hysteresis: dilate high into low
            kernel = torch.ones(1, 1, 3, 3, device=m.device)
            expanded = high_mask
            max_iters = min(500, int(max(m.shape[-2], m.shape[-1]) * 0.1))
            for _ in range(max(1, max_iters)):
                if expanded.dim() == 2:
                    expanded = expanded.unsqueeze(0).unsqueeze(0)
                elif expanded.dim() == 3:
                    expanded = expanded.unsqueeze(1)
                expanded = F.conv2d(F.pad(expanded, (1, 1, 1, 1), mode="constant", value=0),
                                    kernel, padding=0)
                expanded = (expanded > 0.5).float()
                if mask.dim() == 2:
                    expanded = expanded.squeeze(0).squeeze(0)
                else:
                    expanded = expanded.squeeze(1)
                expanded = expanded * low_mask
                if torch.equal(expanded, high_mask):
                    break
                high_mask = expanded
            m = expanded
        elif operation == "gamma":
            gamma = max(0.01, value_a)
            m = m.clamp(0, 1).pow(1.0 / gamma)
        elif operation == "contrast":
            # value_a = contrast factor, value_b = midpoint
            m = (m - value_b) * value_a + value_b
        elif operation == "abs_diff_from_value":
            m = torch.abs(m - value_a)

        return (m.clamp(0, 1),)


class BBoxFromMask:
    """Extract the tight bounding box of non-zero pixels in a mask."""

    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                                    "tooltip": "Uniform padding around the detected bbox"}),
                "padding_x": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                                      "tooltip": "Extra horizontal padding (added to both sides)"}),
                "padding_y": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                                      "tooltip": "Extra vertical padding (added to both sides)"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("BBOX", "INT", "INT", "INT", "INT", "STRING",)
    RETURN_NAMES = ("bbox", "x", "y", "width", "height", "bbox_str",)
    FUNCTION = "extract"
    CATEGORY = _LEGACY_CATEGORY
    DESCRIPTION = "Deprecated — kept so old workflows load. No unified successor."

    def extract(self, mask, padding, padding_x, padding_y, threshold):
        m = mask
        if m.dim() == 3:
            m = m[0]
        binary = (m >= threshold).float()
        coords = torch.nonzero(binary, as_tuple=False)

        if coords.shape[0] == 0:
            h, w = m.shape
            bbox = [0, 0, int(w), int(h)]
            return (bbox, 0, 0, int(w), int(h), json.dumps(bbox))

        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item())
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item())

        total_px = padding + padding_x
        total_py = padding + padding_y
        h, w = m.shape
        x_min = max(0, x_min - total_px)
        y_min = max(0, y_min - total_py)
        x_max = min(w - 1, x_max + total_px)
        y_max = min(h - 1, y_max + total_py)

        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        bbox = [x_min, y_min, bw, bh]
        return (bbox, x_min, y_min, bw, bh, json.dumps(bbox))


class BBoxToMask:
    """Convert a bounding box to a filled rectangular mask."""

    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX",),
                "image_width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "image_height": ("INT", {"default": 512, "min": 1, "max": 16384}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"
    CATEGORY = _LEGACY_CATEGORY
    DESCRIPTION = "Deprecated — kept so old workflows load. No unified successor."

    def convert(self, bbox, image_width, image_height, reference_image=None):
        if reference_image is not None:
            _, h, w, _ = reference_image.shape
            image_height, image_width = int(h), int(w)
        x, y, bw, bh = _norm_bbox(bbox)
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)
        mask = torch.zeros(1, image_height, image_width, dtype=torch.float32)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + bw)
        y2 = min(image_height, y + bh)
        mask[0, y1:y2, x1:x2] = 1.0
        return (mask,)


class BBoxPad:
    """Add asymmetric padding to a bounding box with clamping."""

    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX",),
                "pad_left": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "pad_right": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "pad_top": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "pad_bottom": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "image_width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "image_height": ("INT", {"default": 512, "min": 1, "max": 16384}),
            },
        }

    RETURN_TYPES = ("BBOX", "STRING",)
    RETURN_NAMES = ("bbox", "bbox_str",)
    FUNCTION = "pad"
    CATEGORY = _LEGACY_CATEGORY
    DESCRIPTION = "Deprecated — kept so old workflows load. No unified successor."

    def pad(self, bbox, pad_left, pad_right, pad_top, pad_bottom, image_width, image_height):
        x, y, bw, bh = _norm_bbox(bbox)
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)
        x1 = max(0, x - pad_left)
        y1 = max(0, y - pad_top)
        x2 = min(image_width, x + bw + pad_right)
        y2 = min(image_height, y + bh + pad_bottom)
        out = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
        return (out, json.dumps(out))


class BBoxCrop:
    """Crop an image and its mask to a bounding box region."""

    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox": ("BBOX",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX",)
    RETURN_NAMES = ("cropped_image", "cropped_mask", "bbox",)
    FUNCTION = "crop"
    CATEGORY = _LEGACY_CATEGORY
    DESCRIPTION = (
        "Deprecated — kept so old workflows load. Closest current node: SmartImageCropMEC."
    )

    def crop(self, image, bbox, mask=None):
        x, y, bw, bh = _norm_bbox(bbox)
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)
        B, H, W, C = image.shape
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + bw)
        y2 = min(H, y + bh)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, min(1, W), min(1, H)
        cropped = image[:, y1:y2, x1:x2, :]
        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(0)
            if m.shape[1] != H or m.shape[2] != W:
                m = F.interpolate(m.unsqueeze(1), size=(H, W),
                                  mode="bilinear", align_corners=False).squeeze(1)
            cropped_mask = m[:, y1:y2, x1:x2]
        else:
            cropped_mask = torch.ones(B, y2 - y1, x2 - x1, dtype=torch.float32, device=image.device)
        return (cropped, cropped_mask, [x1, y1, x2 - x1, y2 - y1])


NODE_CLASS_MAPPINGS = {
    "MaskPreviewOverlay": MaskPreviewOverlay,
    "MaskCompositeAdvanced": MaskCompositeAdvanced,
    "MaskMath": MaskMath,
    "BBoxFromMask": BBoxFromMask,
    "BBoxToMask": BBoxToMask,
    "BBoxPad": BBoxPad,
    "BBoxCrop": BBoxCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskPreviewOverlay": "Mask Preview Overlay (MEC) (legacy)",
    "MaskCompositeAdvanced": "Mask Composite Advanced (MEC) (legacy)",
    "MaskMath": "Mask Math (MEC) (legacy)",
    "BBoxFromMask": "BBox From Mask (MEC) (legacy)",
    "BBoxToMask": "BBox To Mask (MEC) (legacy)",
    "BBoxPad": "BBox Pad (MEC) (legacy)",
    "BBoxCrop": "BBox Crop (MEC) (legacy)",
}
