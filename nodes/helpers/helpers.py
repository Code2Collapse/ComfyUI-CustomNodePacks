"""C2C Helpers — 9 utility nodes.  See package __init__ for the index."""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import torch

from .._is_changed_util import hash_args_and_kwargs

log = logging.getLogger("C2C.helpers")

_ANY = "STRING,INT,FLOAT,BOOLEAN,IMAGE,MASK,LATENT,CONDITIONING,MODEL,CLIP,VAE,SHOTLIST"


class _AnyType(str):
    """Wildcard input/output type — equals any other type for combo matching."""
    def __ne__(self, other):  # pragma: no cover - matched by ComfyUI internals
        return False

ANY = _AnyType("*")


# ─────────────────────────── 1. Batch Range ──────────────────────────────
class ImageBatchSliceMEC:
    """Batch Range (L7.65 P23): a [start:end:step] range, a split point, or one frame. Image Batch Split and Video
    Frame Extractor were merged in; their saved workflows migrate here (mode + values + outputs)."""

    MODES = ("range", "split at index", "split at ratio", "first frame", "middle frame", "last frame",
             "frame at index")
    CATEGORY = "C2C/Helpers"
    FUNCTION = "slice"
    RETURN_TYPES = ("IMAGE", "INT", "IMAGE", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("images", "frame_count", "remainder", "remainder_count", "total_frames", "is_video")
    OUTPUT_TOOLTIPS = (
        "The selected frames (the first part when splitting).",
        "How many frames 'images' holds.",
        "Split modes: the frames after the split point. Empty in the other modes.",
        "How many frames 'remainder' holds.",
        "How many frames came in.",
        "True when more than one frame came in.",
    )
    DESCRIPTION = (
        "Pick frames from an IMAGE batch: a start:end:step range (negative values count from the end, step=2 keeps "
        "every second frame), a split at a frame index or a fraction, or a single frame (first, middle, last or by "
        "index)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "start":  ("INT", {"default": 0,    "min": -100000, "max": 100000,
                               "tooltip": "Range mode: first frame. Negative counts from the end."}),
            "end":    ("INT", {"default": -1,   "min": -100000, "max": 100000,
                               "tooltip": "Range mode: exclusive end. -1 = end of batch."}),
            "step":   ("INT", {"default": 1,    "min": 1,        "max": 1024,
                               "tooltip": "Range mode: keep every Nth frame."}),
        }, "optional": {
            # appended after the original three so saved Image Batch Slice values keep their positions
            "mode":        (list(cls.MODES), {"default": "range",
                            "tooltip": "What to pick: a start:end:step range, a split (at a frame index or a "
                                       "fraction), or one frame (first, middle, last or by index)."}),
            "split_index": ("INT",   {"default": 1, "min": 0, "max": 100000,
                                      "tooltip": "'split at index': frames before this index go to 'images'."}),
            "split_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "'split at ratio': this fraction of the batch goes to 'images'."}),
            "frame_index": ("INT",   {"default": 0, "min": 0, "max": 999999,
                                      "tooltip": "'frame at index': 0-based, clamped to the last frame."}),
        }}

    def slice(self, images, start, end, step, mode="range", split_index=1, split_ratio=0.5, frame_index=0):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Batch Range expects an IMAGE batch")
        b = images.shape[0]
        rest = images[:0]
        if mode == "range":
            s = start if start >= 0 else max(0, b + start)
            e = end if end >= 0 else b + end + 1
            s = max(0, min(b, s)); e = max(0, min(b, e))
            out = images[s:e:max(1, step)].contiguous()
        elif mode in ("split at index", "split at ratio"):
            if mode == "split at ratio":
                cut = max(0, min(b, int(round(b * split_ratio))))
            else:
                cut = max(0, min(b, int(split_index)))
            out, rest = images[:cut].contiguous(), images[cut:].contiguous()
        elif mode in ("first frame", "middle frame", "last frame", "frame at index"):
            idx = {"first frame": 0, "middle frame": b // 2, "last frame": b - 1}.get(mode, min(frame_index, b - 1))
            out = images[idx].unsqueeze(0)
        else:
            raise ValueError(f"Batch Range: unknown mode {mode!r}; choose one of {', '.join(self.MODES)}")
        return (out, int(out.shape[0]), rest, int(rest.shape[0]), int(b), b > 1)


# ─────────────────────────── 2. Mask Batch Combine ───────────────────────
class MaskBatchCombineMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "combine"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    DESCRIPTION = (
        "Combine two MASK batches with one of: union (max), intersect (min), "
        "diff (A - B), xor, add (clamp), subtract (clamp). Sizes must match."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask_a": ("MASK",),
            "mask_b": ("MASK",),
            "op":     (["union", "intersect", "diff", "xor", "add", "subtract"],
                       {"default": "union"}),
        }}

    def combine(self, mask_a, mask_b, op):
        a = mask_a if isinstance(mask_a, torch.Tensor) else torch.zeros(1, 1, 1)
        b = mask_b if isinstance(mask_b, torch.Tensor) else torch.zeros_like(a)
        if a.shape != b.shape:
            raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
        if op == "union":     out = torch.maximum(a, b)
        elif op == "intersect": out = torch.minimum(a, b)
        elif op == "diff":    out = torch.clamp(a - b, 0.0, 1.0)
        elif op == "xor":     out = torch.clamp(torch.abs(a - b), 0.0, 1.0)
        elif op == "add":     out = torch.clamp(a + b, 0.0, 1.0)
        elif op == "subtract": out = torch.clamp(a - b, 0.0, 1.0)
        else: out = a
        return (out.contiguous(),)


# ─────────────────────────── 3. Seed List ────────────────────────────────
class SeedListMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "build"
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("first_seed", "csv_all_seeds")
    DESCRIPTION = (
        "Generate N deterministic seeds from a base seed. Modes: "
        "increment (base, base+1, …), hash (sha256 of base+index, "
        "useful for de-correlated samples), random (mt19937 stream)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            "count":     ("INT", {"default": 4, "min": 1, "max": 1024}),
            "mode":      (["increment", "hash", "random"], {"default": "increment"}),
        }}

    def build(self, base_seed, count, mode):
        seeds: list[int] = []
        if mode == "increment":
            seeds = [(base_seed + i) & 0xFFFFFFFF for i in range(count)]
        elif mode == "hash":
            for i in range(count):
                h = hashlib.sha256(f"{base_seed}-{i}".encode()).digest()
                seeds.append(int.from_bytes(h[:4], "big"))
        else:  # random
            import random
            r = random.Random(base_seed)
            seeds = [r.randint(0, 0xFFFFFFFF) for _ in range(count)]
        csv = ",".join(str(s) for s in seeds)
        return (int(seeds[0]), csv)


# ─────────────────────────── 4. Conditional Switch ───────────────────────
class ConditionalSwitchMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "pick"
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("out",)
    DESCRIPTION = (
        "Return value_true when condition is True, else value_false. "
        "Both inputs are wildcard (*) so it works for any type."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "condition":   ("BOOLEAN", {"default": True}),
            "value_true":  (ANY,),
            "value_false": (ANY,),
        }}

    def pick(self, condition, value_true, value_false):
        return (value_true if bool(condition) else value_false,)


# ─────────────────────────── 5. Text Template ────────────────────────────
class TextTemplateMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "format"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    DESCRIPTION = (
        "Substitute {a}{b}{c}{d} placeholders in a template string. "
        "Useful for building prompts from per-shot variables."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "template": ("STRING", {"multiline": True,
                "default": "a {a} of {b}, {c}, {d}"}),
            "a": ("STRING", {"default": "", "multiline": False}),
            "b": ("STRING", {"default": "", "multiline": False}),
            "c": ("STRING", {"default": "", "multiline": False}),
            "d": ("STRING", {"default": "", "multiline": False}),
        }}

    def format(self, template, a, b, c, d):
        try:
            out = template.format(a=a, b=b, c=c, d=d)
        except (KeyError, IndexError, ValueError):
            # Fall back to manual replace so unbalanced braces don't crash.
            out = (template
                   .replace("{a}", a).replace("{b}", b)
                   .replace("{c}", c).replace("{d}", d))
        return (out,)


# ─────────────────────────── 6. Number Lerp ──────────────────────────────
class NumberLerpMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "lerp"
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("float", "int")
    DESCRIPTION = (
        "Linear / smoothstep / cosine interpolation between two values. "
        "t is clamped to [0,1]."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "a": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.0001}),
            "b": ("FLOAT", {"default": 1.0, "min": -1e9, "max": 1e9, "step": 0.0001}),
            "t": ("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0,  "step": 0.001}),
            "curve": (["linear", "smoothstep", "cosine"], {"default": "linear"}),
        }}

    def lerp(self, a, b, t, curve):
        t = max(0.0, min(1.0, float(t)))
        if curve == "smoothstep":
            tt = t * t * (3.0 - 2.0 * t)
        elif curve == "cosine":
            import math
            tt = (1.0 - math.cos(t * math.pi)) * 0.5
        else:
            tt = t
        v = a + (b - a) * tt
        return (float(v), int(round(v)))


def _snap(v: int, m: int, direction: str) -> int:
    """Round v to a multiple of m (never below m): down, up or nearest."""
    if direction == "down":
        return max(m, (v // m) * m)
    if direction == "up":
        return max(m, ((v + m - 1) // m) * m)
    return max(m, int(round(v / m)) * m)


# ─────────────────────────── 7. Size ─────────────────────────────────────
class AspectPresetMEC:
    """Size (L7.65 P22): a preset aspect ratio scaled to a long edge, or a custom width x height, snapped to a
    multiple. Dimensions Snap was merged in; its saved workflows migrate here as preset 'Custom'."""

    CUSTOM = "Custom (width x height)"
    PRESETS = {
        "1:1 square":      (1.0,    1.0),
        "16:9 landscape":  (16.0,   9.0),
        "9:16 portrait":   (9.0,    16.0),
        "4:3 landscape":   (4.0,    3.0),
        "3:4 portrait":    (3.0,    4.0),
        "21:9 ultrawide":  (21.0,   9.0),
        "2.39:1 cinema":   (2.39,   1.0),
        "Wan 480p land":   (832.0,  480.0),
        "Wan 480p port":   (480.0,  832.0),
        "Wan 720p land":   (1280.0, 720.0),
        "Wan 720p port":   (720.0,  1280.0),
    }
    CATEGORY = "C2C/Helpers"
    FUNCTION = "pick"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    DESCRIPTION = (
        "Width and height for a render: a common aspect ratio scaled to a long edge (Wan 480p / 720p presets give "
        "Wan's native sizes), or your own width x height - snapped to a multiple of N, because Wan, Flux and SDXL "
        "need sizes divisible by 8 / 16 / 64."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset":   ([*cls.PRESETS.keys(), cls.CUSTOM], {"default": "16:9 landscape"}),
            "base":     ("INT", {"default": 1024, "min": 64, "max": 8192,
                "tooltip": "Long edge target (ignored for Wan presets and Custom)."}),
            "multiple": ("INT", {"default": 64, "min": 1, "max": 256}),
        }, "optional": {
            # appended after the original three so saved Aspect Preset values keep their positions
            "width":     ("INT", {"default": 1024, "min": 8, "max": 16384, "tooltip": "Custom: width before snapping."}),
            "height":    ("INT", {"default": 1024, "min": 8, "max": 16384, "tooltip": "Custom: height before snapping."}),
            "direction": (["down", "nearest", "up"], {"default": "down",
                          "tooltip": "Snap to the multiple below, the nearest one, or the one above."}),
        }}

    def pick(self, preset, base, multiple, width=1024, height=1024, direction="down"):
        m = max(1, int(multiple))
        if preset == self.CUSTOM:
            w, h = int(width), int(height)
        elif preset not in self.PRESETS:
            raise ValueError(
                f"[Size] Unknown preset '{preset}'. "
                f"Valid presets: {[*self.PRESETS.keys(), self.CUSTOM]}"
            )
        else:
            w_r, h_r = self.PRESETS[preset]
            if preset.startswith("Wan"):
                w, h = int(w_r), int(h_r)
            else:
                long_edge = base
                if w_r >= h_r:
                    w = long_edge
                    h = int(round(long_edge * h_r / w_r))
                else:
                    h = long_edge
                    w = int(round(long_edge * w_r / h_r))
        return (int(_snap(w, m, direction)), int(_snap(h, m, direction)))


# ─────────────────────────── 8. Probe ────────────────────────────────────
def _image_stats(t: torch.Tensor) -> tuple[str, float, float, float]:
    mean   = float(t.mean().item())
    std    = float(t.std().item())
    mn, mx = float(t.min().item()), float(t.max().item())
    # Fraction of pixels brighter than 0.95.
    bright = float((t > 0.95).float().mean().item()) * 100.0
    report = (f"shape={tuple(t.shape)} dtype={t.dtype} "
              f"mean={mean:.4f} std={std:.4f} "
              f"min={mn:.4f} max={mx:.4f} bright>{0.95:.2f}={bright:.1f}%")
    return report, mean, std, bright


def _mask_coverage(mask: torch.Tensor, threshold: float) -> tuple[str, float, float, float]:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Probe: the mask input must be a MASK tensor")
    t = mask
    if t.ndim == 2:
        t_b = t.unsqueeze(0)
    elif t.ndim == 3:
        t_b = t
    else:
        t_b = t.reshape(-1, *t.shape[-2:])
    per = (t_b > float(threshold)).float().mean(dim=(-2, -1)) * 100.0
    cmin = float(per.min().item())
    cmax = float(per.max().item())
    cmean = float(per.mean().item())
    report = (f"frames={t_b.shape[0]} thr={threshold:.2f} "
              f"coverage mean={cmean:.2f}% min={cmin:.2f}% max={cmax:.2f}%")
    return report, cmean, cmin, cmax


class ImageStatsProbeMEC:
    """Probe (L7.65 P24): reports on whatever is wired in - image, mask and/or latent - and passes each through.
    Mask Area Probe and VAE Latent Inspector were merged in; every one of their outputs kept its place here (owner
    2026-10-10: keep all outputs)."""

    CATEGORY = "C2C/Helpers"
    FUNCTION = "probe"
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT", "FLOAT",
                    "MASK", "FLOAT", "FLOAT", "FLOAT",
                    "LATENT", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "report", "mean", "std", "bright_pct",
                    "mask", "coverage_mean_pct", "coverage_min_pct", "coverage_max_pct",
                    "latent", "info_json", "verdict", "nan_count", "inf_count")
    OUTPUT_TOOLTIPS = (
        "The image, unchanged.",
        "One line per wired input (image stats, mask coverage, latent verdict).",
        "Image mean (0 when no image is wired).",
        "Image standard deviation.",
        "Percent of image pixels brighter than 0.95.",
        "The mask, unchanged.",
        "Mean per-frame mask coverage in percent (pixels above threshold).",
        "Lowest per-frame coverage in percent.",
        "Highest per-frame coverage in percent.",
        "The latent, unchanged.",
        "Latent shape, dtype, per-channel stats, NaN / Inf counts and verdict as JSON.",
        "Latent verdict: healthy / low_contrast / saturated / corrupt.",
        "NaN elements in the latent.",
        "Inf elements in the latent.",
    )
    DESCRIPTION = (
        "Pass-through probe for debugging: wire an image, a mask and/or a latent and read their statistics - "
        "black or blown-out frames, how much of each frame a mask covers, NaN / Inf or saturated latents."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {
            "images":    ("IMAGE",),
            "mask":      ("MASK",),
            "latent":    ("LATENT", {"tooltip": "ComfyUI LATENT dict (must contain 'samples')."}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Mask: a pixel counts as covered above this value."}),
            "fail_on_corrupt": ("BOOLEAN", {"default": False,
                                            "tooltip": "Latent: stop the run when it holds NaN or Inf."}),
        }}

    def probe(self, images=None, mask=None, latent=None, threshold=0.5, fail_on_corrupt=False):
        if images is None and mask is None and latent is None:
            raise ValueError("Probe: connect an image, a mask or a latent.")
        lines = []
        mean = std = bright = 0.0
        cmean = cmin = cmax = 0.0
        info_json, verdict, nan_count, inf_count = "", "", 0, 0
        if images is not None:
            r, mean, std, bright = _image_stats(images)
            log.info("[ImageStatsProbe] %s", r)
            lines.append(r)
        if mask is not None:
            r, cmean, cmin, cmax = _mask_coverage(mask, threshold)
            log.info("[MaskAreaProbe] %s", r)
            lines.append(r)
        if latent is not None:
            from ..vae_latent_inspector import inspect_latent
            info_json, verdict, nan_count, inf_count = inspect_latent(latent, fail_on_corrupt)
            lines.append(f"latent verdict={verdict} NaN={nan_count} Inf={inf_count}")
        return (images, "\n".join(lines), mean, std, bright, mask, cmean, cmin, cmax,
                latent, info_json, verdict, nan_count, inf_count)


# ─────────────────────────── 9. Execution Timer ─────────────────────────
class ExecutionTimerMEC:
    """
    Wallclock timer node.  Place between two stages of a workflow: the
    elapsed seconds since the previous tick are measured and returned as
    a string, while the wildcard payload is passed through unchanged.

    Because ComfyUI executes nodes lazily, ``label`` is used as a stable
    cache key so two timer instances don't clobber each other.
    """
    _last: dict[str, float] = {}

    CATEGORY = "C2C/Helpers"
    FUNCTION = "tick"
    RETURN_TYPES = (ANY, "STRING", "FLOAT")
    RETURN_NAMES = ("passthrough", "report", "elapsed_seconds")
    DESCRIPTION = (
        "Stopwatch: returns seconds since the *previous* execution of the "
        "same label. First tick returns 0. Passes input through unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "passthrough": (ANY,),
            "label":       ("STRING", {"default": "stage_a", "multiline": False}),
            "reset":       ("BOOLEAN", {"default": False,
                "tooltip": "If true, this tick is treated as the first."}),
        }}

    @classmethod
    def IS_CHANGED(cls, *args, **kw):
        return hash_args_and_kwargs(*args, **kw)

    def tick(self, passthrough, label, reset):
        now = time.perf_counter()
        key = str(label or "")
        prev = self._last.get(key)
        elapsed = 0.0 if (reset or prev is None) else now - prev
        self._last[key] = now
        report = f"[{key}] +{elapsed:.4f}s"
        return (passthrough, report, float(elapsed))
