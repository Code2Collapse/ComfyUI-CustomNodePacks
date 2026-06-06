"""C2C Helpers — 12 utility nodes.  See package __init__ for the index."""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import torch

log = logging.getLogger("C2C.helpers")

_ANY = "STRING,INT,FLOAT,BOOLEAN,IMAGE,MASK,LATENT,CONDITIONING,MODEL,CLIP,VAE,SHOTLIST"


class _AnyType(str):
    """Wildcard input/output type — equals any other type for combo matching."""
    def __ne__(self, other):  # pragma: no cover - matched by ComfyUI internals
        return False

ANY = _AnyType("*")


# ─────────────────────────── 1. Image Batch Slice ────────────────────────
class ImageBatchSliceMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "slice"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    DESCRIPTION = (
        "Extract a [start:end:step] range from an IMAGE batch. "
        "Negative end values count back from the end. step=2 keeps every "
        "second frame, etc."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "start":  ("INT", {"default": 0,    "min": -100000, "max": 100000}),
            "end":    ("INT", {"default": -1,   "min": -100000, "max": 100000,
                               "tooltip": "Exclusive. -1 = end of batch."}),
            "step":   ("INT", {"default": 1,    "min": 1,        "max": 1024}),
        }}

    def slice(self, images, start, end, step):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("ImageBatchSlice expects IMAGE tensor")
        b = images.shape[0]
        s = start if start >= 0 else max(0, b + start)
        e = end if end >= 0 else b + end + 1
        s = max(0, min(b, s)); e = max(0, min(b, e))
        out = images[s:e:max(1, step)].contiguous()
        return (out, int(out.shape[0]))


# ─────────────────────────── 2. Image Batch Split ────────────────────────
class ImageBatchSplitMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "split"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("first_part", "remainder", "first_count", "remainder_count")
    DESCRIPTION = (
        "Split an IMAGE batch into two pieces at a frame index OR "
        "by a fractional ratio (0.0–1.0)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "mode":   (["index", "ratio"], {"default": "index"}),
            "index":  ("INT",   {"default": 1, "min": 0, "max": 100000}),
            "ratio":  ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    def split(self, images, mode, index, ratio):
        b = images.shape[0]
        if mode == "ratio":
            cut = max(0, min(b, int(round(b * ratio))))
        else:
            cut = max(0, min(b, int(index)))
        a, r = images[:cut].contiguous(), images[cut:].contiguous()
        return (a, r, int(a.shape[0]), int(r.shape[0]))


# ─────────────────────────── 3. Mask Batch Combine ───────────────────────
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


# ─────────────────────────── 4. Seed List ────────────────────────────────
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


# ─────────────────────────── 5. Conditional Switch ───────────────────────
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


# ─────────────────────────── 6. Text Template ────────────────────────────
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


# ─────────────────────────── 7. Number Lerp ──────────────────────────────
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


# ─────────────────────────── 8. Dimensions Snap ──────────────────────────
class DimensionsSnapMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "snap"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    DESCRIPTION = (
        "Round (w,h) to the nearest multiple of N (default 64). "
        "Wan/Flux/SDXL all require dimensions divisible by 8/16/64 — "
        "this prevents shape-mismatch crashes at sample time."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width":     ("INT", {"default": 1024, "min": 8, "max": 16384}),
            "height":    ("INT", {"default": 1024, "min": 8, "max": 16384}),
            "multiple":  ("INT", {"default": 64,   "min": 1, "max": 256}),
            "direction": (["nearest", "down", "up"], {"default": "down"}),
        }}

    def snap(self, width, height, multiple, direction):
        m = max(1, int(multiple))
        def _snap(v):
            if direction == "down":    return max(m, (v // m) * m)
            if direction == "up":      return max(m, ((v + m - 1) // m) * m)
            return max(m, int(round(v / m)) * m)
        return (int(_snap(width)), int(_snap(height)))


# ─────────────────────────── 9. Aspect Preset ────────────────────────────
class AspectPresetMEC:
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
        "Common aspect ratios scaled to a base resolution and snapped to "
        "a multiple. Wan 480p/720p presets emit native Wan target sizes "
        "directly."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset":   (list(cls.PRESETS.keys()), {"default": "16:9 landscape"}),
            "base":     ("INT", {"default": 1024, "min": 64, "max": 8192,
                "tooltip": "Long edge target (ignored for Wan presets)."}),
            "multiple": ("INT", {"default": 64, "min": 1, "max": 256}),
        }}

    def pick(self, preset, base, multiple):
        w_r, h_r = self.PRESETS[preset]
        m = max(1, int(multiple))
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
        w = max(m, (w // m) * m)
        h = max(m, (h // m) * m)
        return (int(w), int(h))


# ─────────────────────────── 10. Image Stats Probe ──────────────────────
class ImageStatsProbeMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "probe"
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "report", "mean", "std", "bright_pct")
    DESCRIPTION = (
        "Passthrough: returns the input image unchanged plus a stats report. "
        "Useful for debugging black-frame / over-bright generations."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    def probe(self, images):
        t = images
        mean   = float(t.mean().item())
        std    = float(t.std().item())
        mn, mx = float(t.min().item()), float(t.max().item())
        # Fraction of pixels brighter than 0.95.
        bright = float((t > 0.95).float().mean().item()) * 100.0
        report = (f"shape={tuple(t.shape)} dtype={t.dtype} "
                  f"mean={mean:.4f} std={std:.4f} "
                  f"min={mn:.4f} max={mx:.4f} bright>{0.95:.2f}={bright:.1f}%")
        log.info("[ImageStatsProbe] %s", report)
        return (t, report, mean, std, bright)


# ─────────────────────────── 11. Mask Area Probe ─────────────────────────
class MaskAreaProbeMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "probe"
    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("mask", "report", "coverage_mean_pct", "coverage_min_pct", "coverage_max_pct")
    DESCRIPTION = (
        "Passthrough: returns the mask unchanged plus a coverage report. "
        "Per-frame coverage = (mask > threshold).mean()."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask":      ("MASK",),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    def probe(self, mask, threshold):
        if not isinstance(mask, torch.Tensor):
            raise ValueError("MaskAreaProbe expects MASK tensor")
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
        log.info("[MaskAreaProbe] %s", report)
        return (mask, report, cmean, cmin, cmax)


# ─────────────────────────── 12. Execution Timer ────────────────────────
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
        h = hashlib.md5()
        for v in args:
            if hasattr(v, 'cpu'):
                h.update(v.cpu().numpy().tobytes())
            elif isinstance(v, str):
                h.update(v.encode())
            else:
                h.update(str(v).encode())
        for k, v in sorted(kw.items()):
            h.update(k.encode())
            if hasattr(v, 'cpu'):
                h.update(v.cpu().numpy().tobytes())
            elif isinstance(v, str):
                h.update(v.encode())
            else:
                h.update(str(v).encode())
        return h.hexdigest()

    def tick(self, passthrough, label, reset):
        now = time.perf_counter()
        key = str(label or "")
        prev = self._last.get(key)
        elapsed = 0.0 if (reset or prev is None) else now - prev
        self._last[key] = now
        report = f"[{key}] +{elapsed:.4f}s"
        return (passthrough, report, float(elapsed))
