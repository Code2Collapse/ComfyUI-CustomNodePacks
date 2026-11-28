"""C2C Helpers — 9 utility nodes.  See package __init__ for the index."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any

import torch

from .._is_changed_util import hash_args_and_kwargs
from . import _helper_features as _hf                      # L9.170: the logic behind the appended A25 features
try:                                                      # A28 layout declarations (display only)
    from .._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '..', '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

log = logging.getLogger("C2C.helpers")

_ANY = "STRING,INT,FLOAT,BOOLEAN,IMAGE,MASK,LATENT,CONDITIONING,MODEL,CLIP,VAE,SHOTLIST"


class _AnyType(str):
    """Wildcard input/output type — equals any other type for combo matching."""
    def __ne__(self, other):  # pragma: no cover - matched by ComfyUI internals
        return False

ANY = _AnyType("*")


# ─────────────────────────── 1. Batch Range ──────────────────────────────

# A28 (docs/audit/A28/PARAM_CNP_B.md, Helpers): which controls show when. Display only - every input keeps its name,
# order, type and default and reaches the node's function as before. Replaces the c2c_mode_widgets.js rules for
# Image Batch Slice and Size (one visibility owner per node).
_SLICE_LAYOUT = {
    "start": {"c2c_show_if": {"mode": ["range"]}},
    "end": {"c2c_show_if": {"mode": ["range"]}},
    "step": {"c2c_show_if": {"mode": ["range"]}},
    "split_index": {"c2c_show_if": {"mode": ["split at index"]}},
    "split_ratio": {"c2c_show_if": {"mode": ["split at ratio"]}},
    "frame_index": {"c2c_show_if": {"mode": ["frame at index"]}},
    # L9.170
    "ranges": {"c2c_show_if": {"mode": ["ranges"]}},
    "fps": {"c2c_show_if": {"mode": ["ranges", "resample to fps"]}},
    "target_fps": {"c2c_show_if": {"mode": ["resample to fps"]}},
}


def _aspect_layout(cls) -> dict:
    """Size: base = the ratio presets only; multiple / direction = ratio presets + Custom (the Wan presets return
    their native size before snapping, pick()); width / height = Custom only."""
    ratio = [k for k in cls.PRESETS if not k.startswith("Wan")]
    return {
        "base": {"c2c_show_if": {"preset": ratio}},
        "multiple": {"c2c_show_if": {"preset": ratio + [cls.CUSTOM]}},
        "direction": {"c2c_show_if": {"preset": ratio + [cls.CUSTOM]}},
        "width": {"c2c_show_if": {"preset": [cls.CUSTOM]}},
        "height": {"c2c_show_if": {"preset": [cls.CUSTOM]}},
        # L9.170: production / profile controls live in a folded section
        "model_profile": {"c2c_section": "Profiles"},
        "custom_profiles": {"c2c_section": "Profiles"},
        "profile": {"c2c_section": "Profiles"},
        "overscan_pct": {"c2c_section": "Profiles"},
    }


_PROBE_LAYOUT = {
    "threshold": {"c2c_show_if_linked": ["mask"]},          # read only for a mask (_mask_coverage)
    "fail_on_corrupt": {"c2c_show_if_linked": ["latent"]},  # read only for a latent (inspect_latent)
    # L9.170: analysis controls, folded
    "scope": {"c2c_section": "Analysis", "c2c_show_if_linked": ["images"]},
    "temporal_check": {"c2c_section": "Analysis", "c2c_show_if_linked": ["images", "mask"]},
    "anomaly_threshold": {"c2c_section": "Analysis", "c2c_show_if_linked": ["images", "mask"]},
}


class ImageBatchSliceMEC:
    """Batch Range (L7.65 P23): a [start:end:step] range, a split point, or one frame. Image Batch Split and Video
    Frame Extractor were merged in; their saved workflows migrate here (mode + values + outputs)."""

    MODES = ("range", "split at index", "split at ratio", "first frame", "middle frame", "last frame",
             "frame at index", "ranges", "resample to fps")        # L9.170: the last two appended
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
        "every second frame), a split at a frame index or a fraction, a single frame (first, middle, last or by "
        "index), a list of ranges in frames, seconds or timecodes, or the batch re-sampled to another frame rate."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({"required": {
            "images": ("IMAGE", {"tooltip": "The frames to pick from (an IMAGE batch)."}),
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
            # L9.170 (A25 5.4 #13 / 4.1 #2-#3): disjoint ranges, times and timecodes, re-sampling to a frame rate
            "ranges":      ("STRING", {"default": "0..9", "multiline": False,
                                       "tooltip": "'ranges' mode: comma-separated ranges, ends included, e.g. "
                                                  "'0..9, 20..29/2, -5..'. '/N' keeps every Nth frame, a negative "
                                                  "number counts from the end, an open end runs to the last frame. "
                                                  "Positions may be seconds ('1.5s..3s') or timecodes "
                                                  "('00:00:02:00..00:00:03:12', HH:MM:SS:FF) when fps is set. "
                                                  "'images' gets the frames in the order listed, 'remainder' every "
                                                  "frame not picked."}),
            "fps":         ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                                      "tooltip": "Frame rate of the incoming batch. 'ranges': needed only for "
                                                 "seconds / timecodes. 'resample to fps': the source rate."}),
            "target_fps":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                                      "tooltip": "'resample to fps': the new frame rate. Each output frame is the "
                                                 "source frame nearest its time, so the clip keeps its duration "
                                                 "(e.g. 30 -> 24 drops frames, 24 -> 48 repeats them)."}),
        }}, _SLICE_LAYOUT)

    def slice(self, images, start, end, step, mode="range", split_index=1, split_ratio=0.5, frame_index=0,
              ranges="0..9", fps=0.0, target_fps=0.0):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Batch Range expects an IMAGE batch")
        b = images.shape[0]
        rest = images[:0]
        if b == 0 and mode in ("first frame", "middle frame", "last frame", "frame at index", "ranges",
                               "resample to fps"):
            raise ValueError(f"Batch Range: the IMAGE batch is empty, so mode {mode!r} has no frame to pick.")
        if mode == "ranges":
            picks = _hf.parse_frame_ranges(ranges, b, float(fps))
            idx = [i for seg in picks for i in seg]
            chosen = set(idx)
            left = [i for i in range(b) if i not in chosen]
            out = images[torch.tensor(idx, dtype=torch.long)].contiguous()
            rest = images[torch.tensor(left, dtype=torch.long)].contiguous() if left else images[:0]
            return (out, int(out.shape[0]), rest, int(rest.shape[0]), int(b), b > 1)
        if mode == "resample to fps":
            idx = _hf.resample_indices(b, float(fps), float(target_fps))
            out = images[torch.tensor(idx, dtype=torch.long)].contiguous()
            return (out, int(out.shape[0]), rest, 0, int(b), b > 1)
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
        "diff (A - B), xor, add (clamp), subtract (clamp), subtract (soft, A*(1-B)). Sizes must match. diff and "
        "subtract are both clamp(A - B) (0.5 - 0.5 = 0); subtract (soft) is A x (1 - B), the soft cut Mask Combine "
        "uses (0.5, 0.5 -> 0.25)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask_a": ("MASK",),
            "mask_b": ("MASK",),
            # audit 2026-11-27 (masking M-12): the soft subtract is APPENDED, so saved values keep their meaning
            "op":     (["union", "intersect", "diff", "xor", "add", "subtract", "subtract (soft, A*(1-B))"],
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
        elif op == "subtract (soft, A*(1-B))": out = torch.clamp(a * (1.0 - b), 0.0, 1.0)
        else: out = a
        return (out.contiguous(),)


# ─────────────────────────── 3. Seed List ────────────────────────────────
class SeedListMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "build_all"          # L9.170: build() keeps its old 2-value return for Python callers
    RETURN_TYPES = ("INT", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("first_seed", "csv_all_seeds", "seeds", "seed_map_json", "manifest_json")
    OUTPUT_IS_LIST = (False, False, True, False, False)
    OUTPUT_TOOLTIPS = (
        "The first seed of the list.", "Every seed, comma separated (e.g. for a seed sweep).",
        "Every seed as a list: a node wired here runs once per seed.",
        "Per-shot seed map as JSON {shot id: [seed of frame 0, frame 1, ...]} (count seeds per shot). A shot's seeds "
        "depend only on base_seed and the shot's id, so reordering or adding shots never changes them. '{}' when "
        "shot_ids is empty.",
        "Reproducibility manifest (JSON): the recipe (base seed, count, mode, algorithm, shot ids), every seed and "
        "seed map, and run metadata (prompt fingerprint, node id, Python / torch / ComfyUI versions, UTC time).",
    )
    DESCRIPTION = (
        "Generate N deterministic seeds from a base seed. Modes: "
        "increment (base, base+1, …), hash (sha256 of base+index, "
        "useful for de-correlated samples), random (mt19937 stream). Optional per-shot seed maps and a "
        "reproducibility manifest."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                  "tooltip": "The seed every other seed is derived from."}),
            "count":     ("INT", {"default": 4, "min": 1, "max": 1024, "tooltip": "How many seeds to make."}),
            "mode":      (["increment", "hash", "random"], {"default": "increment",
                          "tooltip": "increment: base, base+1, ...; hash: sha256 of base and index (unrelated seeds); "
                                     "random: a seeded random stream. All are repeatable for the same base."}),
        }, "optional": {
            # L9.170 (A25 5.4 #16.1): per-shot / per-frame seed planning
            "shot_ids":  ("STRING", {"default": "", "multiline": True,
                          "tooltip": "Optional shot ids (comma or one per line, e.g. 'sh010, sh020'). Each shot gets "
                                     "its own 'count' seeds (one per frame index) in seed_map_json, made with the same "
                                     "mode from a base derived from base_seed and the shot id. Empty: no map."}),
        }, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    def build(self, base_seed, count, mode, **kw):
        """The pre-L9.170 contract: (first_seed, csv_all_seeds)."""
        return self.build_all(base_seed, count, mode, **kw)[:2]

    def build_all(self, base_seed, count, mode, shot_ids="", prompt=None, unique_id=None):
        if mode not in ("increment", "hash", "random"):
            raise ValueError(f"Seed List: unknown mode {mode!r}; choose increment, hash or random.")
        seeds = _hf.seeds_for(int(base_seed), int(count), mode)
        csv = ",".join(str(s) for s in seeds)
        shots = _hf.parse_shot_ids(shot_ids)
        seed_map = {s: _hf.seeds_for(_hf.shot_base(int(base_seed), s), int(count), mode) for s in shots}
        manifest = self._manifest(base_seed, count, mode, shots, seeds, seed_map, prompt, unique_id)
        return (int(seeds[0]), csv, [int(s) for s in seeds], json.dumps(seed_map), manifest)

    @staticmethod
    def _manifest(base_seed, count, mode, shots, seeds, seed_map, prompt, unique_id) -> str:
        """L9.170 (A25 5.4 #16.2): everything needed to rebuild these seeds, plus where and when they were made."""
        import platform
        from datetime import datetime, timezone
        try:
            from comfyui_version import __version__ as comfy_version  # present in ComfyUI >= 0.3
        except Exception:
            comfy_version = None
        algo = {"increment": "(base + i) mod 2^32",
                "hash": "first 4 bytes (big-endian) of sha256('<base>-<i>')",
                "random": "Python random.Random(base).randint(0, 2^32-1), i times"}[mode]
        return json.dumps({
            "node": "SeedListMEC", "recipe_version": 1,
            "recipe": {"base_seed": int(base_seed), "count": int(count), "mode": mode, "algorithm": algo,
                       "shot_ids": shots,
                       "shot_base": "first 4 bytes (big-endian) of sha256('<base>:shot:<shot id>')"},
            "seeds": [int(s) for s in seeds], "seed_map": seed_map,
            "run": {"prompt_sha256": _hf.prompt_fingerprint(prompt), "node_id": unique_id,
                    "python": platform.python_version(), "torch": torch.__version__, "comfyui": comfy_version,
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        }, indent=1)


# ─────────────────────────── 4. Conditional Switch ───────────────────────
class ConditionalSwitchMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "pick"
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("out",)
    OUTPUT_TOOLTIPS = ("value_true or value_false, whichever the condition chose.",)
    DESCRIPTION = (
        "Return value_true when condition is True, else value_false. "
        "Both inputs are wildcard (*) so it works for any type. Optional: decide by a typed comparison "
        "(compare_a <operator> compare_b), or route one of up to five branches by index, with an explicit fallback."
    )
    SOURCES = ("condition", "compare", "index")
    BRANCHES = ("value_false", "value_true", "value_2", "value_3", "value_4")   # index 0..4 (False = 0, True = 1)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "condition":   ("BOOLEAN", {"default": True, "tooltip": "On: pass value_true. Off: pass value_false."}),
            # L9.196: lazy - only the chosen branch is evaluated (both upstream branches always ran before)
            "value_true":  (ANY, {"lazy": True, "tooltip": "Passed on when condition is on (only this branch runs)."}),
            "value_false": (ANY, {"lazy": True, "tooltip": "Passed on when condition is off (only this branch runs)."}),
        }, "optional": {
            # L9.170 (A25 5.4 #11): condition builder + multi-branch routing, appended after the saved widgets
            "source":     (list(cls.SOURCES), {"default": "condition",
                           "tooltip": "What picks the branch. condition: the switch above. compare: compare_a "
                                      "<operator> compare_b (true -> value_true, false -> value_false). index: the "
                                      "branch number below (0 value_false, 1 value_true, 2-4 value_2..value_4)."}),
            "index":      ("INT", {"default": 1, "min": -1, "max": 99,
                           "tooltip": "source 'index': 0 = value_false, 1 = value_true, 2..4 = value_2..value_4. A "
                                      "number without a wired branch uses 'fallback' (an error when that is not "
                                      "wired either)."}),
            "compare_a":  (ANY, {"tooltip": "source 'compare': the value to test (any type: number, text, boolean, "
                                            "a tensor for 'is empty')."}),
            "operator":   (list(_hf.COMPARE_OPERATORS), {"default": "==",
                           "tooltip": "source 'compare': how compare_a is tested against compare_b. contains / "
                                      "starts with / ends with / matches regex compare as text; 'is empty' / 'is not "
                                      "empty' ignore compare_b."}),
            "compare_b":  ("STRING", {"default": "", "multiline": False,
                           "tooltip": "source 'compare': the value compare_a is tested against, typed as compare_as "
                                      "says."}),
            "compare_as": (list(_hf.COMPARE_AS), {"default": "auto",
                           "tooltip": "source 'compare': read both sides as numbers, text or booleans (true/false, "
                                      "yes/no, on/off, 1/0). auto: from compare_a's type. A side that cannot be read "
                                      "that way stops the run with a message."}),
            "value_2":    (ANY, {"lazy": True, "tooltip": "source 'index': branch 2 (only the chosen branch runs)."}),
            "value_3":    (ANY, {"lazy": True, "tooltip": "source 'index': branch 3 (only the chosen branch runs)."}),
            "value_4":    (ANY, {"lazy": True, "tooltip": "source 'index': branch 4 (only the chosen branch runs)."}),
            "fallback":   (ANY, {"lazy": True,
                           "tooltip": "Used when the chosen branch is not wired or the index is out of range. Not "
                                      "wired: that case stops the run with a message instead."}),
        }}

    def _route(self, condition, kw: dict) -> tuple[str, str]:
        """(input name to pass on, why). kw holds the connected optional inputs (core leaves unwired ones out)."""
        source = kw.get("source", "condition")
        if source == "condition":
            idx, why = int(bool(condition)), f"condition is {bool(condition)}"
        elif source == "compare":
            r, why = _hf.evaluate_comparison(kw.get("compare_a"), kw.get("operator", "=="),
                                             kw.get("compare_b", ""), kw.get("compare_as", "auto"))
            idx = int(r)
        elif source == "index":
            idx = int(kw.get("index", 1))
            why = f"index is {idx}"
        else:
            raise ValueError(f"Conditional Switch: unknown source {source!r}; choose one of {', '.join(self.SOURCES)}.")
        key = self.BRANCHES[idx] if 0 <= idx < len(self.BRANCHES) else None
        wired = key in ("value_true", "value_false") or (key is not None and key in kw)
        if wired:
            return key, f"{why} -> {key}"
        if "fallback" in kw:
            return "fallback", f"{why} -> {key or 'no branch'} not wired -> fallback"
        raise ValueError(f"Conditional Switch: {why}, which picks {key or 'no branch (use 0-4)'}, but that input is "
                         f"not wired and 'fallback' is not wired either. Wire it, or wire 'fallback'.")

    def check_lazy_status(self, condition, value_true=None, value_false=None, **kw):
        return [self._route(condition, kw)[0]]

    def pick(self, condition, value_true=None, value_false=None, **kw):
        key, why = self._route(condition, kw)
        if kw.get("source", "condition") != "condition":
            log.info("[ConditionalSwitch] %s", why)
        return ({"value_true": value_true, "value_false": value_false}.get(key, kw.get(key)),)


# ─────────────────────────── 5. Text Template ────────────────────────────
class TextTemplateMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "format_all"         # L9.170: format() keeps its old 1-value return for Python callers
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "report")
    OUTPUT_TOOLTIPS = ("The template with its placeholders filled in.",
                       "JSON: the engine, every variable's final value (after schema defaults and type checks) and, "
                       "for the extended engine, which conditional sections were kept or dropped.")
    DESCRIPTION = (
        "Substitute {a}{b}{c}{d} placeholders in a template string. "
        "Useful for building prompts from per-shot variables. Optional: more named variables, a typed schema "
        "(defaults, required fields, validation) and an extended engine with {?name}...{/name} sections and "
        "{name|upper} filters - plain text substitution, no code is ever run."
    )
    ENGINES = ("format", "extended")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "template": ("STRING", {"multiline": True,
                "default": "a {a} of {b}, {c}, {d}",
                "tooltip": "Text with {a} {b} {c} {d} placeholders; each is replaced by the input of that name."}),
            "a": ("STRING", {"default": "", "multiline": False, "tooltip": "Replaces {a} in the template."}),
            "b": ("STRING", {"default": "", "multiline": False, "tooltip": "Replaces {b} in the template."}),
            "c": ("STRING", {"default": "", "multiline": False, "tooltip": "Replaces {c} in the template."}),
            "d": ("STRING", {"default": "", "multiline": False, "tooltip": "Replaces {d} in the template."}),
        }, "optional": {
            # L9.170 (A25 5.4 #17): typed schema + conditional sections, appended after the saved widgets
            "variables": ("STRING", {"default": "", "multiline": True,
                          "tooltip": "More named variables, one 'name = value' per line (e.g. 'style = anime'); use "
                                     "them as {style}. They may also override a-d. '#' lines are comments."}),
            "schema":    ("STRING", {"default": "", "multiline": True,
                          "tooltip": "Optional typed schema, one per line: 'name: text|int|float|bool|choice(x|y) "
                                     "[= default] [required]', e.g. 'steps: int = 20' or 'style: choice(photo|anime) "
                                     "required'. Empty values take the default; a missing required value or a wrong "
                                     "type stops the run with a list of the problems."}),
            "engine":    (list(cls.ENGINES), {"default": "format",
                          "tooltip": "format: Python-style {name} substitution as before (unknown or broken braces "
                                     "are left as they are). extended: also {?name}...{/name} (kept when name is set "
                                     "and not false/0/no/off), {?name=value}...{/name}, {!name}...{/name} (kept when "
                                     "empty), filters {name|upper|lower|title|capitalize|strip|default:x|prefix:x|"
                                     "suffix:x|truncate:N}; '{{' '}}' are literal braces; an unknown name is an "
                                     "error."}),
        }}

    def format(self, template, a, b, c, d, **kw):
        """The pre-L9.170 contract: (text,)."""
        return self.format_all(template, a, b, c, d, **kw)[:1]

    def format_all(self, template, a, b, c, d, variables="", schema="", engine="format"):
        values = {"a": a, "b": b, "c": c, "d": d}
        values.update(_hf.parse_variables(variables))
        sch = _hf.parse_schema(schema)
        if sch:
            values, problems = _hf.apply_schema(values, sch)
            if problems:
                raise ValueError("Text Template: schema check failed: " + "; ".join(problems) + ".")
        notes: list[str] = []
        if engine == "extended":
            out, notes = _hf.render_extended(template, {k: str(v) for k, v in values.items()})
        elif engine == "format":
            try:
                out = template.format(**values)
            except (KeyError, IndexError, ValueError):
                # Fall back to manual replace so unbalanced braces don't crash.
                out = template
                for k, v in values.items():
                    out = out.replace("{" + k + "}", v)
        else:
            raise ValueError(f"Text Template: unknown engine {engine!r}; choose format or extended.")
        report = json.dumps({"engine": engine, "variables": values, "notes": notes})
        return (out, report)


# ─────────────────────────── 6. Number Lerp ──────────────────────────────
class NumberLerpMEC:
    CATEGORY = "C2C/Helpers"
    FUNCTION = "lerp"
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("float", "int")
    OUTPUT_TOOLTIPS = ("The interpolated value.", "The interpolated value, rounded to the nearest whole number.")
    DESCRIPTION = (
        "Linear / smoothstep / cosine interpolation between two values. "
        "t is clamped to [0,1]."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "a": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.0001, "tooltip": "The value at t = 0."}),
            "b": ("FLOAT", {"default": 1.0, "min": -1e9, "max": 1e9, "step": 0.0001, "tooltip": "The value at t = 1."}),
            "t": ("FLOAT", {"default": 0.5, "min": 0.0,  "max": 1.0,  "step": 0.001,
                            "tooltip": "How far from a to b (0..1)."}),
            "curve": (["linear", "smoothstep", "cosine", "hold"], {"default": "linear",
                      "tooltip": "linear: straight; smoothstep and cosine ease in and out near a and b; hold: keeps "
                                 "the earlier value until the next key (steps)."}),
        }, "optional": {
            # L9.170 (A25 5.4 #15): keyframe curves and time-domain evaluation, appended after the saved widgets
            "keyframes": ("STRING", {"default": "", "multiline": False,
                          "tooltip": "Optional keyframes 'position:value', comma separated, e.g. '0:0, 0.5:1, 1:0' "
                                     "(positions in the time_mode's unit: 0..1, frames or seconds). When set, a and b "
                                     "are ignored and curve shapes each segment; before the first / after the last "
                                     "key the value holds."}),
            "time_mode": (list(_hf.TIME_MODES), {"default": "normalised t",
                          "tooltip": "normalised t: evaluate at t (0..1) as before. frame / seconds: evaluate at "
                                     "'time' instead; without keyframes t = time / duration."}),
            "time":      ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.001,
                          "tooltip": "frame / seconds mode: where to evaluate (a frame number or seconds)."}),
            "duration":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e9, "step": 0.001,
                          "tooltip": "frame / seconds mode: length of one cycle in the same unit. Needed without "
                                     "keyframes; with keyframes 0 = up to the last key."}),
            "loop":      (list(_hf.LOOP_MODES), {"default": "clamp",
                          "tooltip": "frame / seconds mode: clamp holds the end value, loop restarts every cycle, "
                                     "ping-pong runs forward then back."}),
        }}

    def lerp(self, a, b, t, curve, keyframes="", time_mode="normalised t", time=0.0, duration=0.0, loop="clamp"):
        keys = _hf.parse_keyframes(keyframes) if str(keyframes or "").strip() else None
        if time_mode == "normalised t":
            pos = max(0.0, min(1.0, float(t)))
        elif time_mode in ("frame", "seconds"):
            period = float(duration)
            if period <= 0 and keys:
                period = keys[-1][0]
            if period <= 0 and not keys:
                raise ValueError(f"Number Lerp: time_mode {time_mode!r} without keyframes needs duration above 0.")
            pos = _hf.wrap_time(float(time), period, loop)
            if not keys:
                pos = max(0.0, min(1.0, pos / period))
        else:
            raise ValueError(f"Number Lerp: unknown time_mode {time_mode!r}; choose one of "
                             f"{', '.join(_hf.TIME_MODES)}.")
        if keys:
            v = _hf.eval_keyframes(keys, pos, curve)
        else:
            v = a + (b - a) * _hf.ease(pos, curve)
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
    OUTPUT_TOOLTIPS = ("Width in pixels, snapped to the multiple.", "Height in pixels, snapped to the multiple.")
    DESCRIPTION = (
        "Width and height for a render: a common aspect ratio scaled to a long edge (Wan 480p / 720p presets give "
        "Wan's native sizes), or your own width x height - snapped to a multiple of N, because Wan, Flux and SDXL "
        "need sizes divisible by 8 / 16 / 64."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({"required": {
            "preset":   ([*cls.PRESETS.keys(), cls.CUSTOM], {"default": "16:9 landscape",
                         "tooltip": "An aspect ratio scaled to 'base', a Wan native size, or Custom (width x height "
                                    "below)."}),
            "base":     ("INT", {"default": 1024, "min": 64, "max": 8192,
                "tooltip": "Long edge target (ignored for Wan presets and Custom)."}),
            "multiple": ("INT", {"default": 64, "min": 1, "max": 256,
                "tooltip": "Snap width and height to this multiple. Wan presets ignore it: their native sizes "
                           "(832x480, 1280x720) are already what Wan needs (multiples of 16)."}),
        }, "optional": {
            # appended after the original three so saved Aspect Preset values keep their positions
            "width":     ("INT", {"default": 1024, "min": 8, "max": 16384, "tooltip": "Custom: width before snapping."}),
            "height":    ("INT", {"default": 1024, "min": 8, "max": 16384, "tooltip": "Custom: height before snapping."}),
            "direction": (["down", "nearest", "up"], {"default": "down",
                          "tooltip": "Snap to the multiple below, the nearest one, or the one above."}),
            # L9.170 (A25 5.4 #10, 4.4 #44): model profiles, named profiles, overscan - appended, folded section
            "model_profile":   (list(_hf.MODEL_PROFILES), {"default": _hf.MODEL_PROFILE_NONE,
                                "tooltip": "Snap to what a model family needs instead of 'multiple': SD 1.5 / SDXL "
                                           "x64, SD3 / Flux / Qwen Image x16, Wan / HunyuanVideo x16, LTX-Video x32 "
                                           "(VAE downscale x patch size). Wan presets are already Wan-native and stay "
                                           "as they are."}),
            "custom_profiles": ("STRING", {"default": "", "multiline": True,
                                "tooltip": "Your own named sizes, one per line: 'Name = 1920x1080' (exact size, then "
                                           "snapped) or 'Name = 2.39:1' (ratio scaled to 'base'). Copy this text "
                                           "between workflows to share a project's profiles. '#' starts a comment."}),
            "profile":         ("STRING", {"default": "", "multiline": False,
                                "tooltip": "Name of a line in custom_profiles to use instead of 'preset'. Empty: "
                                           "use 'preset'. An unknown name stops the run and lists the names."}),
            "overscan_pct":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                "tooltip": "Render this many percent larger than the picked size (overscan margin to "
                                           "crop back later), then snap. 0 = off. Also applies to Wan presets (then "
                                           "snapped to x16 or the model profile)."}),
        }}, _aspect_layout(cls))

    def pick(self, preset, base, multiple, width=1024, height=1024, direction="down",
             model_profile=_hf.MODEL_PROFILE_NONE, custom_profiles="", profile="", overscan_pct=0.0):
        if model_profile not in _hf.MODEL_PROFILES:
            raise ValueError(f"[Size] Unknown model_profile {model_profile!r}. Valid: {list(_hf.MODEL_PROFILES)}")
        prof_m = _hf.MODEL_PROFILES[model_profile]
        m = prof_m or max(1, int(multiple))
        over = 1.0 + max(0.0, float(overscan_pct or 0.0)) / 100.0
        name = str(profile or "").strip()
        if name:
            table = _hf.parse_size_profiles(custom_profiles)
            if name not in table:
                raise ValueError(f"[Size] profile {name!r} is not in custom_profiles. Names: {list(table) or 'none'}")
            kind, pa, pb = table[name]
            if kind == "size":
                w, h = int(round(pa)), int(round(pb))
            elif pa >= pb:
                w, h = int(base), int(round(base * pb / pa))
            else:
                w, h = int(round(base * pa / pb)), int(base)
            return (int(_snap(int(round(w * over)), m, direction)), int(_snap(int(round(h * over)), m, direction)))
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
                # L9.189: returned exactly. Snapping them to the default multiple of 64 gave 832x448 / 1280x704,
                # not the native sizes the description promises (Wan needs multiples of 16, which these are).
                if over == 1.0:
                    return (int(w_r), int(h_r))
                wm = prof_m or 16                                   # L9.170 overscan: keep Wan's x16 rule
                return (int(_snap(int(round(w_r * over)), wm, direction)),
                        int(_snap(int(round(h_r * over)), wm, direction)))
            else:
                long_edge = base
                if w_r >= h_r:
                    w = long_edge
                    h = int(round(long_edge * h_r / w_r))
                else:
                    h = long_edge
                    w = int(round(long_edge * w_r / h_r))
        if over != 1.0:
            w, h = int(round(w * over)), int(round(h * over))
        return (int(_snap(w, m, direction)), int(_snap(h, m, direction)))


# ─────────────────────────── 8. Probe ────────────────────────────────────
def _image_stats(t: torch.Tensor) -> tuple[str, float, float, float]:
    # audit 2026-11-27: t.std() and (t > 0.95).float() each made a full-size float copy (+4 GB on a 124-frame 2K
    # clip). Same numbers, 8 frames at a time with float64 sums.
    # agy review: the one-pass (s2 - n*mean^2) variance cancels catastrophically on bright constant HDR plates (it
    # went negative / NaN); chunks are now merged with Chan et al.'s parallel update (per-chunk mean and M2).
    n, mean, m2, n_bright = 0, 0.0, 0.0, 0
    mn, mx = float("inf"), float("-inf")
    t = t.reshape(1, -1) if t.ndim == 0 else t
    for i in range(0, int(t.shape[0]), 8):
        c = t[i:i + 8].double()
        nb = c.numel()
        if nb == 0:
            continue
        mb = float(c.mean())
        m2b = float(((c - mb) ** 2).sum())
        delta = mb - mean
        tot = n + nb
        mean += delta * nb / tot
        m2 += m2b + delta * delta * n * nb / tot
        n = tot
        mn, mx = min(mn, float(c.min())), max(mx, float(c.max()))
        n_bright += int(torch.count_nonzero(c > 0.95))
    if n == 0:
        raise ValueError("Probe: the image is empty (0 values); connect an image with at least one pixel.")
    std = math.sqrt(max(0.0, m2 / (n - 1))) if n > 1 else float("nan")
    # Fraction of pixels brighter than 0.95.
    bright = n_bright / n * 100.0
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
    FUNCTION = "probe_all"          # L9.170: probe() keeps its old 14-value return for Python callers
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT", "FLOAT",
                    "MASK", "FLOAT", "FLOAT", "FLOAT",
                    "LATENT", "STRING", "STRING", "INT", "INT",
                    "IMAGE", "STRING", "INT", "INT", "INT", "INT")                      # L9.170: appended
    RETURN_NAMES = ("images", "report", "mean", "std", "bright_pct",
                    "mask", "coverage_mean_pct", "coverage_min_pct", "coverage_max_pct",
                    "latent", "info_json", "verdict", "nan_count", "inf_count",
                    "scope", "frame_stats_json", "anomaly_count", "width", "height", "frame_count")
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
        "The chosen scope of the whole image batch (histogram, luma waveform or RGB parade; 256 px high, log "
        "density; long clips and large frames are sampled). A black 64x64 picture when scope is 'none'.",
        "temporal_check on: JSON with per-frame stats (luma, channel means, clipped %, NaN, Inf, mask mean) and the "
        "flagged frames with the reason. '' when off.",
        "temporal_check on: how many frames were flagged (0 when off).",
        "Width of the image (or mask) in pixels; 0 when only a latent is wired.",
        "Height of the image (or mask) in pixels; 0 when only a latent is wired.",
        "Frames in the image batch (else the mask, else the latent: batch x latent frames for a 5-D video latent).",
    )
    DESCRIPTION = (
        "Pass-through probe for debugging: wire an image, a mask and/or a latent and read their statistics - "
        "black or blown-out frames, how much of each frame a mask covers, NaN / Inf or saturated latents."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({"required": {}, "optional": {
            "images":    ("IMAGE", {"tooltip": "An image to measure (passed on unchanged)."}),
            "mask":      ("MASK", {"tooltip": "A mask to measure coverage of (passed on unchanged)."}),
            "latent":    ("LATENT", {"tooltip": "ComfyUI LATENT dict (must contain 'samples')."}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Mask: a pixel counts as covered above this value."}),
            "fail_on_corrupt": ("BOOLEAN", {"default": False,
                                            "tooltip": "Latent: stop the run when it holds NaN or Inf."}),
            # L9.170 (A25 5.4 #14, 4.1 #15): scopes, temporal anomalies, size outputs
            "scope":     (list(_hf.SCOPES), {"default": "none",
                          "tooltip": "Draw a scope of the image batch on the 'scope' output: histogram (R, G, B), "
                                     "waveform (luma per column) or rgb parade."}),
            "temporal_check": ("BOOLEAN", {"default": False,
                               "tooltip": "Measure every frame and flag sudden exposure, colour, clipping or mask "
                                          "(alpha) jumps and frames with NaN / Inf (frame_stats_json, "
                                          "anomaly_count)."}),
            "anomaly_threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001,
                                  "tooltip": "temporal_check: a frame is flagged when luma, a channel mean or the "
                                             "mask mean moves more than this (0..1 scale) from the previous frame, "
                                             "or the clipped share moves more than this fraction."}),
        }}, _PROBE_LAYOUT)

    def probe(self, images=None, mask=None, latent=None, threshold=0.5, fail_on_corrupt=False, **kw):
        """The pre-L9.170 contract: the first 14 outputs."""
        return self.probe_all(images, mask, latent, threshold, fail_on_corrupt, **kw)[:14]

    def probe_all(self, images=None, mask=None, latent=None, threshold=0.5, fail_on_corrupt=False, scope="none",
                  temporal_check=False, anomaly_threshold=0.05):
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
        # L9.170 additions: never touch the outputs above
        if scope not in _hf.SCOPES:
            raise ValueError(f"Probe: unknown scope {scope!r}; choose one of {', '.join(_hf.SCOPES)}.")
        if scope != "none" and images is not None:
            with torch.no_grad():
                scope_img = _hf.render_scope(images, scope)
        else:
            if scope != "none":
                lines.append(f"scope {scope}: no image wired")
            scope_img = torch.zeros(1, 64, 64, 3)
        stats_json, anomaly_count = "", 0
        if temporal_check and (images is not None or mask is not None):
            with torch.no_grad():
                rows = _hf.frame_stats(images, mask)
            flagged = _hf.find_anomalies(rows, float(anomaly_threshold))
            anomaly_count = len(flagged)
            stats_json = json.dumps({"threshold": float(anomaly_threshold), "anomalies": flagged, "frames": rows})
            lines.append(f"temporal: {anomaly_count} flagged frame(s)"
                         + (f" {[a['frame'] for a in flagged][:20]}" if flagged else ""))
        src = images if images is not None else None
        if src is not None:
            fh, fw, fc = int(src.shape[1]), int(src.shape[2]), int(src.shape[0])
        elif mask is not None:
            m3 = mask.reshape(-1, *mask.shape[-2:])
            fh, fw, fc = int(m3.shape[1]), int(m3.shape[2]), int(m3.shape[0])
        else:
            s = latent.get("samples") if isinstance(latent, dict) else None
            fh, fw, fc = 0, 0, int(s.shape[0]) if isinstance(s, torch.Tensor) and s.ndim >= 1 else 0
            if isinstance(s, torch.Tensor) and s.ndim == 5:       # audit 2026-11-27: [B,C,T,H,W] reported 1
                fc = int(s.shape[0]) * int(s.shape[2])
        return (images, "\n".join(lines), mean, std, bright, mask, cmean, cmin, cmax,
                latent, info_json, verdict, nan_count, inf_count,
                scope_img, stats_json, anomaly_count, fw, fh, fc)


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
    _hist: dict[str, list[float]] = {}                     # L9.170: stage -> measured seconds (newest last)

    CATEGORY = "C2C/Helpers"
    FUNCTION = "tick_all"           # L9.170: tick() keeps its old 3-value return for Python callers
    RETURN_TYPES = (ANY, "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("passthrough", "report", "elapsed_seconds", "stats_json")
    OUTPUT_TOOLTIPS = ("The input, unchanged.", "'[label] +seconds' for this tick.",
                       "Seconds since the previous tick with the same label (0 on the first).",
                       "JSON: this stage's kept measurements (count, last, median, p95, min, max, total), RAM / VRAM "
                       "in use now, and a tree of every stage grouped by '/' in the labels (e.g. 'render/decode' "
                       "adds to 'render'). First ticks and resets are not measurements and are not kept.")
    DESCRIPTION = (
        "Stopwatch: returns seconds since the *previous* execution of the "
        "same label. First tick returns 0. Passes input through unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "passthrough": (ANY, {"tooltip": "Anything: passed on unchanged; the timer ticks when it arrives."}),
            "label":       ("STRING", {"default": "stage_a", "multiline": False,
                            "tooltip": "Timers with the same label measure from each other's ticks."}),
            "reset":       ("BOOLEAN", {"default": False,
                "tooltip": "If true, this tick is treated as the first."}),
        }, "optional": {
            # L9.170 (A25 5.4 #12): stage timing between two labels + kept measurements with median / p95 / memory
            "since_label":  ("STRING", {"default": "", "multiline": False,
                             "tooltip": "Measure from the last tick of THIS other label instead of this label's own "
                                        "last tick: put a timer 'start' before a stage and one with since_label "
                                        "'start' after it to time exactly that stage (stage name 'start->label'). "
                                        "Empty: as before."}),
            "history_size": ("INT", {"default": 50, "min": 1, "max": 10000,
                             "tooltip": "How many measurements of this stage are kept (newest) for median / p95 in "
                                        "stats_json. Kept in memory until ComfyUI restarts."}),
        }}

    @classmethod
    def IS_CHANGED(cls, *args, **kw):
        return hash_args_and_kwargs(*args, **kw)

    def tick(self, passthrough, label, reset, **kw):
        """The pre-L9.170 contract: (passthrough, report, elapsed_seconds)."""
        return self.tick_all(passthrough, label, reset, **kw)[:3]

    def tick_all(self, passthrough, label, reset, since_label="", history_size=50):
        # audit 2026-11-27: GPU kernels of the previous node run asynchronously - wait for them, or the time of a GPU
        # stage is booked to whatever stage happens to synchronise next.
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - timing must never fail the run
            pass
        now = time.perf_counter()
        key = str(label or "")
        src = str(since_label or "").strip() or key
        prev = self._last.get(src)
        elapsed = 0.0 if (reset or prev is None) else now - prev
        self._last[key] = now
        stage = key if src == key else f"{src}->{key}"
        report = f"[{stage}] +{elapsed:.4f}s"
        hist = self._hist.setdefault(stage, [])
        if not (reset or prev is None):
            hist.append(float(elapsed))
            del hist[:-max(1, int(history_size))]
        stats = {"stage": stage, "elapsed": round(float(elapsed), 6), "measured": not (reset or prev is None),
                 "stats": _hf.stage_stats(hist), "memory": _hf.memory_snapshot(), "tree": _hf.stage_tree(self._hist)}
        return (passthrough, report, float(elapsed), json.dumps(stats))
