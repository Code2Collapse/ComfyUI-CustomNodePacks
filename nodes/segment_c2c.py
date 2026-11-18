"""Segment (C2C) - the one segmentation node of owner card P04 (L7.123). Design: docs/research/
L7.123_segment_node_design.md in the C2C workspace.

Slice 1 (2026-11-19): target "objects" on ComfyUI's own SAM 3 / 3.1 (the same cached detector as the Mask + Matting
viewer's Find, nodes/c2c_viewer/detect.py, so one model is in memory for both): words through SAM 3's detector
("person", "person:3, dog" - several objects of one word need "word:N", a plain word returns the strongest match),
else clicks and boxes through the SAM decoder.

Slice 2 (2026-11-19): a clip is tracked - the objects found on `seed_frame` are followed forward to the end and
backward to frame 0 with ONE tracking state each way (core SAM 3 forward_video), so they keep their identity and
order; `info` carries the union matte's edge stability (L7.127). `video = per frame` keeps the slice-1 behaviour.
Later slices append targets (face parts, clothes, semantic classes, everything) and the SAMRefiner refine stage;
every addition is appended so saved values never move.

Apache-2.0, C2C.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

log = logging.getLogger("C2C.Segment")

TARGETS = ["objects"]          # appended to in later slices (face parts, clothes, semantic classes, everything)


def _parse_picks(raw: str, W: int, H: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple]]:
    """picks JSON -> (positive points, negative points, boxes) in frame pixels.
    {"positive": [[x, y], ...], "negative": [[x, y], ...], "boxes": [[x0, y0, x1, y1], ...]}; an empty string = none."""
    if not raw or not str(raw).strip():
        return [], [], []
    try:
        d = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Segment (C2C): 'picks' is not valid JSON ({exc}). Expected "
                         '{"positive": [[x, y]], "negative": [[x, y]], "boxes": [[x0, y0, x1, y1]]} in pixels.') from exc
    if not isinstance(d, dict):
        raise ValueError("Segment (C2C): 'picks' must be a JSON object with positive / negative / boxes lists.")

    def pts(key):
        out = []
        for p in d.get(key) or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(min(max(p[0], 0), W - 1)), float(min(max(p[1], 0), H - 1))))
        return out

    boxes = []
    for b in d.get("boxes") or []:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            x0, y0, x1, y1 = (float(v) for v in b[:4])
            x0, x1 = sorted((min(max(x0, 0), W), min(max(x1, 0), W)))
            y0, y1 = sorted((min(max(y0, 0), H), min(max(y1, 0), H)))
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                boxes.append((x0, y0, x1, y1))
    return pts("positive"), pts("negative"), boxes


def _to_hw(m, H: int, W: int) -> np.ndarray:
    m = np.asarray(m, np.float32)
    if m.shape != (H, W):
        m = np.asarray(torch.nn.functional.interpolate(torch.from_numpy(m)[None, None], size=(H, W), mode="bilinear",
                                                       align_corners=False)[0, 0])
    return (m > 0.5).astype(np.float32)


def _to_hw_stack(t, H: int, W: int) -> np.ndarray:
    t = torch.as_tensor(t).float()
    if not t.shape[0]:
        return np.zeros((0, H, W), np.float32)
    return np.stack([_to_hw(t[n].numpy(), H, W) for n in range(t.shape[0])])


def _bbox_of(m: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(m > 0.5)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class SegmentC2C:
    DESCRIPTION = ("Segment (C2C): find objects by words, clicks or boxes with ComfyUI's own SAM 3 / 3.1 and get one "
                   "mask per object plus their union and boxes. Words: 'person', 'person:3, dog' (word:N for several "
                   "of one kind). Slice 1 of the unified segmentation node (more targets follow).")
    CATEGORY = "C2C/Segmentation"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK", "MASK", "BBOX", "STRING")
    RETURN_NAMES = ("mask", "masks", "bboxes", "info")
    OUTPUT_TOOLTIPS = (
        "Union of every object found, one mask per frame (B,H,W).",
        "One mask per object found on the first frame (N,H,W), strongest first.",
        "One [x0, y0, x1, y1] box per object of the first frame, in pixels.",
        "JSON: model, what was asked, objects with scores and boxes per frame, time, and notes.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .c2c_viewer.detect import model_choices
            models = model_choices()
        except Exception as exc:  # noqa: BLE001 - the node must still load to say what is missing
            log.warning("[Segment] SAM checkpoint list unavailable: %s", exc)
            models = ["(no SAM 3 checkpoint)"]
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "A still or a clip (B,H,W,C)."}),
                "model": (models, {"tooltip": "SAM 3 / 3.1 checkpoint (ComfyUI's own SAM 3, the same file the Mask + "
                                              "Matting viewer uses; it is loaded once and shared)."}),
                "target": (TARGETS, {"default": "objects",
                                     "tooltip": "What to segment. objects: things named by words, or picked by clicks "
                                                "and boxes."}),
                "prompt": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "Words, e.g. 'person', 'person:3, dog'. With words, clicks and boxes "
                                                 "are ignored in this slice. Empty: use clicks / boxes."}),
                "picks": ("STRING", {"default": "", "multiline": True,
                                     "tooltip": "Clicks and boxes in pixels: {\"positive\": [[x, y]], \"negative\": "
                                                "[[x, y]], \"boxes\": [[x0, y0, x1, y1]]}. One object per box (the "
                                                "clicks go with every box), else one object from the clicks."}),
                "max_objects": ("INT", {"default": 20, "min": 1, "max": 200,
                                        "tooltip": "Keep at most this many objects per frame (strongest first)."}),
            },
            # slice 2 (appended): one tracking state over a clip
            "optional": {
                "video": (["track whole clip", "per frame"], {
                    "default": "track whole clip",
                    "tooltip": "A clip: find the objects on seed_frame and follow them through every frame (forward "
                               "and backward, one tracking state, objects keep their order), or segment each frame on "
                               "its own."}),
                "seed_frame": ("INT", {"default": 0, "min": 0, "max": 100000,
                                       "tooltip": "The frame the objects are found on before tracking (pick a frame "
                                                  "where they are clearly visible)."}),
            },
        }

    def execute(self, image, model, target, prompt, picks, max_objects, video="track whole clip", seed_frame=0):
        from .c2c_viewer.detect import get_detector, resolve_model
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("Segment (C2C) expects an IMAGE tensor [B,H,W,C].")
        if target not in TARGETS:
            raise ValueError(f"Segment (C2C): target {target!r} is not built yet (this version has {TARGETS}).")
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        text = str(prompt or "").strip()
        pos, neg, boxes = _parse_picks(picks, W, H)
        if not text and not pos and not boxes:
            raise ValueError("Segment (C2C): say what to segment - type words in 'prompt' (e.g. 'person') or give "
                             "clicks / boxes in 'picks'.")
        tag, path = resolve_model(model)
        det = get_detector(tag, path)
        t0 = time.time()
        # float frames in 0..1, as the viewer's Find passes them (a uint8 frame broke the decoder's box / click path)
        frames = image[..., :3].detach().float().clamp(0, 1).cpu().numpy()
        terms = [t.strip() for t in text.split(",") if t.strip()]      # "person:3, dog" -> one detection per term
        notes: List[str] = []
        if text and (pos or boxes):
            notes.append("Words were given, so clicks and boxes were not used.")

        def detect(b: int) -> List[Dict[str, Any]]:
            if terms:
                objs = [{**o, "label": t} for t in terms for o in det.detect(frames[b], t, [], [], [])]
            else:
                objs = [{**o, "label": "picked"} for o in det.detect(frames[b], "", pos, neg, boxes)]
            objs = sorted(objs, key=lambda o: -float(o.get("score", 0.0)))[: int(max_objects)]
            for o in objs:
                o["mask"] = _to_hw(o["mask"], H, W)
            return objs

        want_track = B > 1 and str(video) == "track whole clip"
        tracked = False
        with torch.no_grad():               # never inference_mode: outputs must stay mutable downstream
            if want_track and callable(getattr(det, "track", None)):
                k = min(max(int(seed_frame), 0), B - 1)
                seed = detect(k)
                obj = np.zeros((B, len(seed), H, W), np.float32)
                if seed:
                    init = torch.from_numpy(np.stack([o["mask"] for o in seed]))
                    obj[k] = init.numpy()
                    if k < B - 1:                               # one tracking state forward to the end ...
                        fwd = det.track(image[k:], init)
                        for t in range(1, B - k):
                            obj[k + t] = _to_hw_stack(fwd.frame(t), H, W)
                    if k > 0:                                   # ... and one backward to frame 0
                        bwd = det.track(image[: k + 1].flip(0), init)
                        for t in range(1, k + 1):
                            obj[k - t] = _to_hw_stack(bwd.frame(t), H, W)
                tracked = True
                meta = [(o.get("label", ""), round(float(o.get("score", 0.0)), 4)) for o in seed]
                per_frame = [[{"label": meta[n][0], "score": meta[n][1], "box": _bbox_of(obj[b, n]),
                               "area": int(obj[b, n].sum())} for n in range(len(seed))] for b in range(B)]
                union = obj.max(axis=1) if seed else np.zeros((B, H, W), np.float32)
                first = [{"mask": obj[0, n], "box": _bbox_of(obj[0, n])} for n in range(len(seed))]
                notes.append(f"Tracked {len(seed)} object(s) through the whole clip from frame {k} (forward and "
                             "backward, one tracking state).")
            else:
                if want_track:
                    notes.append("This model cannot track, so each frame was segmented on its own (objects can swap "
                                 "order between frames).")
                elif B > 1:
                    notes.append("video = per frame: each frame was segmented on its own.")
                union = np.zeros((B, H, W), np.float32)
                first, per_frame = [], []
                for b in range(B):
                    rows = []
                    for o in detect(b):
                        m = o["mask"]
                        union[b] = np.maximum(union[b], m)
                        box = [int(v) for v in (o.get("box") or _bbox_of(m))]
                        rows.append({"label": o.get("label", ""), "score": round(float(o.get("score", 0.0)), 4),
                                     "box": box, "area": int(m.sum())})
                        if b == 0:
                            first.append({"mask": m, "box": box})
                    per_frame.append(rows)
        if not first:
            notes.append("Nothing was found" + (f" for '{text}'" if text else "") + ".")
        edge = None
        if B > 1 and union.any():
            try:                                                # L7.127: how steady the matte edge is
                from .temporal_consistency_checker import _edge_stability
                vals = [_edge_stability(union[b], union[b + 1], frames[b], frames[b + 1], 0.5)[0]
                        for b in range(B - 1)]
                edge = {"mean": round(float(np.mean(vals)), 4), "max": round(float(np.max(vals)), 4)}
            except Exception as exc:  # noqa: BLE001 - a missing OpenCV only drops the number
                log.debug("[Segment] edge stability skipped: %s", exc)
        masks = (torch.from_numpy(np.stack([o["mask"] for o in first])) if first
                 else torch.zeros((1, H, W), dtype=torch.float32))
        info = {"model": tag, "target": target, "prompt": text, "clicks": len(pos), "negative_clicks": len(neg),
                "boxes": len(boxes), "frames": B, "tracked": tracked, "objects_first_frame": len(first),
                "per_frame": per_frame, "edge_stability": edge, "seconds": round(time.time() - t0, 3), "notes": notes}
        return (torch.from_numpy(union), masks, [o["box"] for o in first], json.dumps(info))


NODE_CLASS_MAPPINGS = {"SegmentC2C": SegmentC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"SegmentC2C": "Segment (C2C)"}
