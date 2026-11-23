"""pixel_aspect.py — anamorphic / non-square Pixel Aspect Ratio round-trip.

AI models (and ComfyUI itself) assume SQUARE pixels. Anamorphic plates —
e.g. a 4448x3840 scan with PAR 1.7266 that displays as 2:1 in Nuke — read as
a squarish 1.158:1 image here, so generations come back distorted once the
lens squeeze is reapplied. Industry pipeline (Nuke Reformat semantics):

  PARDesqueezeMEC   plate → square pixels (stretch width by PAR, or squash
                    height) BEFORE any AI step, emitting a par_info JSON
                    that records exactly how to undo it.
  PARResqueezeMEC   AI output + par_info → back to the original pixel
                    dimensions for delivery back into Nuke, PAR untouched.

4448x3840 @ PAR 1.7266 → desqueeze(stretch_width) → 7680x3840 (true 2:1
square-pixel frame) → AI → resqueeze → 4448x3840 again. Lossless contract on
geometry: resqueeze restores the exact original W×H.

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ._is_changed_util import hash_args_and_kwargs
try:                                                       # A28 layout declarations (display only)
    from ._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

# Common industry PARs. "custom" uses the pixel_aspect float widget.
PAR_PRESETS: Dict[str, float] = {
    "custom": 0.0,
    "square 1.0": 1.0,
    "anamorphic 2x (2.0)": 2.0,
    "anamorphic 1.8x (1.8)": 1.8,
    "ARRI 4448x3840→2:1 (1.7266)": 1.7266,
    "anamorphic 1.5x (1.5)": 1.5,
    "anamorphic 1.33x (1.33)": 1.33,
    "NTSC DV (0.9091)": 0.9091,
    "PAL DV (1.0940)": 1.0940,
}


def _resize(img: torch.Tensor, w: int, h: int, filt: str) -> torch.Tensor:
    x = img.permute(0, 3, 1, 2)
    kw = {} if filt in ("nearest", "area") else {"align_corners": False}
    out = F.interpolate(x, size=(h, w), mode=filt, **kw)
    if filt == "bicubic":
        # bicubic rings past its inputs: keep it inside the input's own range. A22 L9.136: a 0..1 clamp on every
        # filter cut scene-linear EXR highlights (and negatives) on the PAR round trip.
        out = out.clamp(float(x.amin()), float(x.amax()))
    return out.permute(0, 2, 3, 1)


def _resolve_par(preset: str, pixel_aspect: float) -> float:
    v = PAR_PRESETS.get(preset, 0.0)
    par = v if v > 0 else float(pixel_aspect)
    if par <= 0:
        raise ValueError("pixel_aspect must be > 0 (use the preset or the custom float).")
    return par


# ---- L9.173 (A25 spec 84 / 85): lens metadata, overscan planning, round-trip validation --------------------------
_PAR_KEYS = ("pixel_aspect", "par", "squeeze", "anamorphic_squeeze", "lens_squeeze")


def _parse_lens_metadata(text: str) -> Dict[str, Any]:
    """Camera / lens metadata as a dict: a JSON object, or `key: value` / `key=value` lines (or `;`-separated).
    Numbers become floats. Empty text -> {}. Anything else is an error that says how to write it."""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"PAR Desqueeze: lens_metadata starts with '{{' but is not valid JSON ({exc}).") from exc
        if not isinstance(data, dict):
            raise ValueError("PAR Desqueeze: lens_metadata JSON must be an object, e.g. {\"lens\": \"Cooke 2x\"}.")
        return data
    out: Dict[str, Any] = {}
    for part in text.replace(";", "\n").splitlines():
        part = part.strip()
        if not part:
            continue
        sep = ":" if (":" in part and ("=" not in part or part.index(":") < part.index("="))) else "="
        if sep not in part:
            raise ValueError(f"PAR Desqueeze: lens_metadata line {part!r} is not 'key: value' or 'key=value' "
                             f"(or give a JSON object).")
        k, v = part.split(sep, 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"PAR Desqueeze: lens_metadata line {part!r} has no key.")
        try:
            out[k] = float(v.rstrip("xX")) if v and v.rstrip("xX").replace(".", "", 1).lstrip("-").isdigit() else v
        except ValueError:
            out[k] = v
    return out


def _overscan_plan(w: int, h: int, percent: float, multiple: int) -> Dict[str, int]:
    """Crop-safe overscan around a w x h frame: `percent` of each side's length added on both sides, then the
    padded size rounded UP to `multiple` (the extra split evenly, odd pixel to the right / bottom)."""
    pct = max(0.0, float(percent)) / 100.0
    mult = max(1, int(multiple))
    px, py = int(math.ceil(w * pct - 1e-9)), int(math.ceil(h * pct - 1e-9))
    pw, ph = w + 2 * px, h + 2 * py
    ew, eh = (-pw) % mult, (-ph) % mult
    return {"left": px + ew // 2, "right": px + ew - ew // 2, "top": py + eh // 2, "bottom": py + eh - eh // 2,
            "padded_width": pw + ew, "padded_height": ph + eh}


# A28 (docs/audit/A28/PARAM_CNP_B.md, PARDesqueezeMEC): pixel_aspect is read only for the "custom" preset
# (_resolve_par: preset value 0). Display only.
_DESQUEEZE_LAYOUT = {"pixel_aspect": {"c2c_show_if": {"par_preset": ["custom"]}}}


class PARDesqueezeMEC:
    DESCRIPTION = ("Convert an anamorphic plate to SQUARE pixels before AI processing "
                   "(Nuke: 4448x3840 @ PAR 1.7266 → 7680x3840 at 2:1). Wire par_info "
                   "into PARResqueezeMEC to restore the original geometry losslessly.")
    CATEGORY = "MEC/Plate"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "par_info", "overscan_mask")
    OUTPUT_TOOLTIPS = ("The frames with square pixels (wider with stretch_width, shorter with squash_height).",
                       "What was done, as JSON - connect it to PAR Resqueeze to undo it exactly after processing.",
                       "1 on the overscan border this node added (edge-extended pixels), 0 on the real picture; all 0 "
                       "without overscan. Use it to let the AI step repaint only the border.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({"required": {
            "image": ("IMAGE", {"tooltip": "Frames shot with non-square pixels (anamorphic lens, DV, HDV ...)."}),
            "par_preset": (tuple(PAR_PRESETS.keys()),
                           {"default": "ARRI 4448x3840→2:1 (1.7266)", "tooltip": "Camera / format the frames come from; sets the pixel aspect ratio. custom = type it in pixel_aspect."}),
            "pixel_aspect": ("FLOAT", {"default": 1.7266, "min": 0.1, "max": 4.0,
                             "step": 0.0001,
                             "tooltip": "Used when par_preset = custom. PAR>1 = pixels wider than tall."}),
            "method": (("stretch_width", "squash_height"), {"default": "stretch_width",
                       "tooltip": "stretch_width keeps every scanline (recommended); "
                                  "squash_height keeps the pixel count low."}),
            "filter": (("bicubic", "bilinear", "nearest", "area"), {"default": "bicubic",
                       "tooltip": "How pixels are resampled: bicubic (sharp, default), bilinear (soft), nearest (no new values; for masks / IDs), area (best when shrinking)."}),
        },
            # L9.173 (A25 spec 84): appended, optional; the defaults give exactly the old image and par_info
            "optional": {
            "lens_metadata": ("STRING", {"default": "", "multiline": True,
                              "tooltip": "Camera / lens facts to carry through the round trip, as 'key: value' lines "
                                         "or a JSON object (e.g. camera: ARRI Alexa 35, lens: Cooke 2x, squeeze: 2). "
                                         "Stored in par_info and given back by PAR Resqueeze's metadata. A squeeze / "
                                         "pixel_aspect value that disagrees with the PAR used is reported as a "
                                         "warning. Empty = nothing added (as before)."}),
            "overscan_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                 "tooltip": "Add this much border (percent of each side, edge-extended) around the "
                                            "square-pixel frame so an AI step has room at the edges. PAR Resqueeze "
                                            "crops it off again before restoring the plate size. 0 = none (as "
                                            "before)."}),
            "overscan_multiple": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1,
                                  "tooltip": "Round the padded frame UP to a multiple of this (e.g. 16 or 64 for a "
                                             "model's size rule); the extra goes into the border. 1 = no rounding "
                                             "(as before)."}),
        }}, _DESQUEEZE_LAYOUT)

    def execute(self, image: torch.Tensor, par_preset: str, pixel_aspect: float,
                method: str, filter: str, lens_metadata: str = "", overscan_percent: float = 0.0,
                overscan_multiple: int = 1):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        b, h0, w0, c = image.shape
        par = _resolve_par(par_preset, pixel_aspect)
        if abs(par - 1.0) < 1e-6:
            out, nw, nh = image, w0, h0
        elif method == "squash_height":
            nw, nh = w0, max(1, round(h0 / par))
            out = _resize(image, nw, nh, filter)
        else:
            nw, nh = max(1, round(w0 * par)), h0
            out = _resize(image, nw, nh, filter)
        meta: Dict[str, Any] = {
            "orig_width": w0, "orig_height": h0, "pixel_aspect": par,
            "method": method, "filter": filter,
            "square_width": nw, "square_height": nh,
            "display_aspect": round((w0 * par) / h0, 4),
        }
        lens = _parse_lens_metadata(lens_metadata)
        if lens:
            meta["lens"] = lens
            warnings = []
            for k in _PAR_KEYS:
                v = lens.get(k)
                if isinstance(v, (int, float)) and v > 0 and abs(float(v) - par) / par > 0.005:
                    warnings.append(f"lens_metadata {k}={float(v):g} differs from the PAR used ({par:g})")
            if warnings:
                meta["warnings"] = warnings
        mask = torch.zeros((out.shape[0], nh, nw), dtype=torch.float32, device=out.device)
        plan = _overscan_plan(nw, nh, overscan_percent, overscan_multiple)
        l, r, t, btm = plan["left"], plan["right"], plan["top"], plan["bottom"]
        if l or r or t or btm:
            x = out.permute(0, 3, 1, 2)
            out = F.pad(x, (l, r, t, btm), mode="replicate").permute(0, 2, 3, 1)
            mask = torch.ones((out.shape[0], out.shape[1], out.shape[2]), dtype=torch.float32, device=out.device)
            mask[:, t:t + nh, l:l + nw] = 0.0
            meta["overscan"] = dict(plan, percent=float(overscan_percent), multiple=int(overscan_multiple))
        return (out, json.dumps(meta), mask)


class PARResqueezeMEC:
    DESCRIPTION = ("Return an AI-processed square-pixel frame to the plate's original "
                   "pixel dimensions (PAR metadata is reapplied in Nuke). Wire "
                   "par_info from PARDesqueezeMEC — the round trip restores the "
                   "exact original W×H even if the AI changed resolution.")
    CATEGORY = "MEC/Plate"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING", "MASK")
    RETURN_NAMES = ("image", "info", "pixel_aspect", "metadata", "difference")
    OUTPUT_TOOLTIPS = ("The frames back at the plate's original width x height.", "What was restored, as JSON.",
                       "The plate's original pixel aspect ratio from par_info, to wire into a writer / format.",
                       "The plate's metadata restored as JSON: size, pixel aspect, display aspect, a Nuke format "
                       "line, the lens metadata given to PAR Desqueeze, the overscan removed and, with reference "
                       "wired, the round-trip check.",
                       "With reference wired: |result - reference| per pixel (mean of the channels, clipped to "
                       "0..1). All 0 without a reference.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "The processed square-pixel frames from after PAR Desqueeze."}),
            "par_info": ("STRING", {"default": "", "forceInput": True,
                                    "tooltip": "PAR Desqueeze's par_info output: the original size and aspect."}),
            "filter": (("bicubic", "bilinear", "nearest", "area"), {"default": "bicubic",
                       "tooltip": "How pixels are resampled: bicubic (sharp, default), bilinear (soft), nearest (no new values; for masks / IDs), area (best when shrinking)."}),
        },
            # L9.173 (A25 spec 85): appended, optional
            "optional": {
            "reference": ("IMAGE", {"tooltip": "Optional: the original plate (what went into PAR Desqueeze). The "
                                               "result is checked against it - size, frame count, largest / mean "
                                               "difference, PSNR - in metadata, and the difference output shows "
                                               "where they differ."}),
        }}

    def execute(self, image: torch.Tensor, par_info: str, filter: str, reference=None):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        try:
            meta: Dict[str, Any] = json.loads(par_info or "{}")
            w = int(meta["orig_width"])
            h = int(meta["orig_height"])
        except Exception as exc:
            raise ValueError(
                "par_info is not valid PARDesqueezeMEC output — wire the desqueeze "
                f"node's par_info here ({exc})."
            ) from exc
        info: Dict[str, Any] = {"restored_width": w, "restored_height": h,
                                "pixel_aspect": meta.get("pixel_aspect", 1.0)}
        removed = None
        osc = meta.get("overscan")
        if isinstance(osc, dict):              # L9.173: crop the Desqueeze overscan off first (scaled to this size)
            try:
                pw, ph = float(osc["padded_width"]), float(osc["padded_height"])
                sx, sy = image.shape[2] / pw, image.shape[1] / ph
                x0, x1 = int(round(float(osc["left"]) * sx)), int(round((pw - float(osc["right"])) * sx))
                y0, y1 = int(round(float(osc["top"]) * sy)), int(round((ph - float(osc["bottom"])) * sy))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"par_info has a broken overscan entry ({exc}); wire PAR Desqueeze's par_info "
                                 f"unchanged.") from exc
            if not (0 <= x0 < x1 <= image.shape[2] and 0 <= y0 < y1 <= image.shape[1]):
                raise ValueError(f"PAR Resqueeze: the overscan crop {x0}:{x1} x {y0}:{y1} does not fit the "
                                 f"{image.shape[2]}x{image.shape[1]} input.")
            image = image[:, y0:y1, x0:x1, :]
            removed = {"kept_box_xyxy": [x0, y0, x1, y1], "scale": [round(sx, 6), round(sy, 6)]}
            info["overscan_removed"] = removed
        out = image if (image.shape[2] == w and image.shape[1] == h) \
            else _resize(image, w, h, filter)
        par = float(meta.get("pixel_aspect", 1.0))
        md: Dict[str, Any] = {"orig_width": w, "orig_height": h, "pixel_aspect": par,
                              "display_aspect": meta.get("display_aspect", round(w * par / h, 4) if h else None),
                              "method": meta.get("method"),
                              "nuke_format": f"{w} {h} 0 0 {w} {h} {par:g} c2c_plate"}
        for k in ("lens", "warnings"):
            if k in meta:
                md[k] = meta[k]
        if removed:
            md["overscan_removed"] = removed
        diff = torch.zeros(out.shape[:3], dtype=torch.float32, device=out.device)
        if reference is not None:
            md["validation"], diff = _validate_round_trip(out, reference)
        return (out, json.dumps(info), par, json.dumps(md), diff)


def _validate_round_trip(out: torch.Tensor, ref: torch.Tensor):
    """Compare the restored frames with the original plate: geometry first, then pixels (frames paired from 0)."""
    if ref.ndim == 3:
        ref = ref.unsqueeze(0)
    zeros = torch.zeros(out.shape[:3], dtype=torch.float32, device=out.device)
    v: Dict[str, Any] = {"result_size": [int(out.shape[2]), int(out.shape[1])],
                         "reference_size": [int(ref.shape[2]), int(ref.shape[1])],
                         "result_frames": int(out.shape[0]), "reference_frames": int(ref.shape[0])}
    v["geometry_ok"] = v["result_size"] == v["reference_size"]
    v["frame_count_ok"] = v["result_frames"] == v["reference_frames"]
    if not v["geometry_ok"]:
        v["note"] = "sizes differ: no pixel comparison (did the par_info come from this plate?)"
        return v, zeros
    n = min(int(out.shape[0]), int(ref.shape[0]))
    c = min(int(out.shape[3]), int(ref.shape[3]))
    a = out[:n, ..., :c].float()
    b = ref[:n, ..., :c].to(a.device).float()
    d = (a - b).abs()
    mse = float((d ** 2).mean())
    peak = max(1.0, float(b.amax()) - min(0.0, float(b.amin())))
    v.update({"frames_compared": n, "max_abs_diff": round(float(d.amax()), 6),
              "mean_abs_diff": round(float(d.mean()), 6),
              "psnr_db": 99.0 if mse <= 1e-12 else round(10.0 * math.log10(peak * peak / mse), 3),
              "pixels_over_1_255_pct": round(float((d.amax(-1) > 1.0 / 255.0).float().mean()) * 100.0, 4)})
    diff = zeros.clone()
    diff[:n] = d.mean(-1).clamp(0.0, 1.0)
    return v, diff


NODE_CLASS_MAPPINGS = {
    "PARDesqueezeMEC": PARDesqueezeMEC,
    "PARResqueezeMEC": PARResqueezeMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PARDesqueezeMEC": "PAR Desqueeze (C2C, anamorphic → square px)",
    "PARResqueezeMEC": "PAR Resqueeze (C2C, back to plate)",
}
