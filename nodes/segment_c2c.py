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

TARGETS = ["objects", "face parts", "clothes"]   # appended to in later slices (semantic classes, everything)
# slice 4: the parsing targets run the Semantic Segment parsers (SegFormer face / clothes, nodes/semantic_segment.py)
_PARSERS = {"face parts": ("segformer_face", "skin,hair"), "clothes": ("segformer_clothes", "upper_clothes")}


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


_FRAMES_FORMAT = ('"frames" in picks must be an object of frame index -> picks, e.g. {"frames": {"12": {"positive": '
                  '[[x, y]], "negative": [[x, y]], "boxes": [[x0, y0, x1, y1]]}}}')


def _parse_frame_picks(raw: str, W: int, H: int, B: int) -> Dict[int, Tuple[List, List, List]]:
    """L9.172: the optional "frames" object of picks -> {frame: (positive, negative, boxes)}, each parsed with the
    top-level rules. No "frames" key (every older picks string) = {}. A non-integer key, a frame outside the clip or
    an entry with no positive click and no box -> ValueError."""
    if not raw or not str(raw).strip():
        return {}
    d = json.loads(raw)                                    # _parse_picks already refused bad JSON / a non-object
    fr = d.get("frames") if isinstance(d, dict) else None
    if fr is None:
        return {}
    if not isinstance(fr, dict):
        raise ValueError(f"Segment (C2C): {_FRAMES_FORMAT}.")
    out: Dict[int, Tuple[List, List, List]] = {}
    for key, v in fr.items():
        k = str(key).strip()
        if not k.isdigit():
            raise ValueError(f"Segment (C2C): picks 'frames' key {key!r} is not a frame number. {_FRAMES_FORMAT}.")
        f = int(k)
        if f >= B:
            raise ValueError(f"Segment (C2C): picks 'frames' has frame {f}, but the clip has {B} frame(s) "
                             f"(0..{B - 1}).")
        if not isinstance(v, dict):
            raise ValueError(f"Segment (C2C): picks 'frames' entry {key!r} must be an object. {_FRAMES_FORMAT}.")
        p, n, bx = _parse_picks(json.dumps(v), W, H)
        if not p and not bx:
            raise ValueError(f"Segment (C2C): picks 'frames' entry {key!r} needs at least one positive click or one "
                             "box.")
        out[f] = (p, n, bx)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    u = float(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / u if u else 0.0


def _match_greedy(tracked: np.ndarray, dets: List[np.ndarray]) -> List[Tuple[int, int]]:
    """Greedy IoU matching, highest first, IoU > 0 required -> [(tracked object n, detection j)]."""
    pairs = sorted(((_iou(tracked[n], d), n, j) for n in range(tracked.shape[0]) for j, d in enumerate(dets)),
                   key=lambda p: -p[0])
    used_n, used_j, out = set(), set(), []
    for v, n, j in pairs:
        if v <= 0:
            break
        if n in used_n or j in used_j:
            continue
        used_n.add(n)
        used_j.add(j)
        out.append((n, j))
    return out


def _centroid(m: np.ndarray):
    ys, xs = np.nonzero(m > 0.5)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def _best_redetection(ref: np.ndarray, cands: List[np.ndarray]):
    """recover_lost: the candidate with the highest IoU against `ref`, else the nearest centroid within twice ref's
    box diagonal; None when neither."""
    best, best_v = None, 0.0
    for c in cands:
        v = _iou(ref, c)
        if v > best_v:
            best, best_v = c, v
    if best is not None:
        return best
    rc = _centroid(ref)
    if rc is None:
        return None
    x0, y0, x1, y1 = _bbox_of(ref)
    reach = 2.0 * float(np.hypot(x1 - x0, y1 - y0))
    near, near_d = None, None
    for c in cands:
        cc = _centroid(c)
        if cc is None:
            continue
        dist = float(np.hypot(cc[0] - rc[0], cc[1] - rc[1]))
        if dist <= reach and (near_d is None or dist < near_d):
            near, near_d = c, dist
    return near


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


def _refine(core, frame_hwc: np.ndarray, mask: np.ndarray):
    from .segment_refine import refine as _samrefine
    out, log = _samrefine(core, frame_hwc, mask)
    return (out > 0.5).astype(np.float32), log


def _bbox_of(m: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(m > 0.5)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class SegmentC2C:
    DESCRIPTION = ("Segment (C2C): find objects by words, clicks or boxes with ComfyUI's own SAM 3 / 3.1 and get one "
                   "mask per object plus their union and boxes. Words: 'person', 'person:3, dog' (word:N for several "
                   "of one kind). Slice 1 of the unified segmentation node (more targets follow). "
                   "picks may carry per-frame clicks / boxes ({\"frames\": {\"12\": {...}}}): per frame they replace "
                   "the top-level picks on that frame; when tracking, frames after seed_frame are corrections the "
                   "tracker restarts from. recover_lost re-finds tracked objects (named by words) where they "
                   "vanish. face parts / clothes accept groups such as '@lips' or '@head, -hair'.")
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
                                                "and boxes (SAM 3). face parts / clothes: the classes named in the "
                                                "prompt, one mask each (face / clothes parser; model is not used)."}),
                "prompt": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "objects: words, e.g. 'person', 'person:3, dog' (with words, clicks "
                                                 "and boxes are ignored; empty: use clicks / boxes). face parts: "
                                                 "classes such as 'skin, hair, l_eye, mouth' (empty = skin, hair). "
                                                 "clothes: 'upper_clothes, pants, dress' (empty = upper_clothes). "
                                                 "face parts / clothes also take groups: '@lips', '@eyes', '@head, "
                                                 "-hair' ('-' removes), plus your own from groups_json."}),
                "picks": ("STRING", {"default": "", "multiline": True,
                                     "tooltip": "Clicks and boxes in pixels: {\"positive\": [[x, y]], \"negative\": "
                                                "[[x, y]], \"boxes\": [[x0, y0, x1, y1]]}. One object per box (the "
                                                "clicks go with every box), else one object from the clicks. "
                                                "Optional per-frame picks: \"frames\": {\"12\": {\"positive\": "
                                                "[[x, y]], \"boxes\": [...]}}. per frame (or a still): that frame "
                                                "uses its own picks (not with words). track whole clip: the seed "
                                                "frame's entry finds the objects; a later frame's entry is a "
                                                "correction - the detection there replaces the tracked mask it "
                                                "overlaps most and tracking restarts from it."}),
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
                # slice 3 (appended): SAMRefiner-style refinement, opt-in until measured better
                "refine": ("BOOLEAN", {"default": False,
                                       "tooltip": "EXPERIMENTAL: refine each object's mask once more with SAM, from "
                                                  "prompts mined out of the mask itself (SAMRefiner, ICLR 2025). "
                                                  "Off by default until it is measured better on your kind of shot; "
                                                  "info reports what each pass changed."}),
                # slice 4 (appended): the parsing targets' model download consent
                "allow_download": ("BOOLEAN", {"default": False,
                                               "tooltip": "face parts / clothes: allow downloading the parsing model "
                                                          "when it is missing (otherwise a model-free stand-in runs "
                                                          "and info says so)."}),
                # L9.172 (A25): appended after A28; the defaults run exactly the old code
                "recover_lost": ("BOOLEAN", {"default": False,
                                             "tooltip": "track whole clip: where a tracked object shrinks below 5 % "
                                                        "of its size on seed_frame, look for it again on that frame "
                                                        "with its words and put back the detection that overlaps "
                                                        "(or sits nearest to) its last good mask. Objects picked by "
                                                        "clicks cannot be looked for again (info says so). Ignored "
                                                        "with video = per frame."}),
                "groups_json": ("STRING", {"default": "", "multiline": True,
                                           "tooltip": "face parts / clothes: your own class groups, used as '@name' "
                                                      "in prompt: {\"my_face\": [\"@eyes\", \"nose\", \"@lips\"]}. "
                                                      "They win over the built-in groups of the same name. Empty = "
                                                      "built-ins only. Not used for objects."}),
            },
        }

    def execute(self, image, model, target, prompt, picks, max_objects, video="track whole clip", seed_frame=0,
                refine=False, allow_download=False, recover_lost=False, groups_json=""):
        from .c2c_viewer.detect import get_detector, resolve_model
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("Segment (C2C) expects an IMAGE tensor [B,H,W,C].")
        if target not in TARGETS:
            raise ValueError(f"Segment (C2C): target {target!r} is not built yet (this version has {TARGETS}).")
        if target in _PARSERS:
            return self._parse(image, target, prompt, bool(allow_download), groups_json or "")
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        text = str(prompt or "").strip()
        pos, neg, boxes = _parse_picks(picks, W, H)
        fpicks = _parse_frame_picks(picks, W, H, B)            # L9.172: {} for every older picks string
        if not text and not pos and not boxes and not fpicks:
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

        def detect(b: int, fp=None) -> List[Dict[str, Any]]:
            if fp is not None:                                  # L9.172: this frame's own clicks / boxes
                objs = [{**o, "label": "picked"} for o in det.detect(frames[b], "", fp[0], fp[1], fp[2])]
            elif terms:
                objs = [{**o, "label": t} for t in terms for o in det.detect(frames[b], t, [], [], [])]
            elif not pos and not boxes:                         # only per-frame picks, none for this frame
                return []
            else:
                objs = [{**o, "label": "picked"} for o in det.detect(frames[b], "", pos, neg, boxes)]
            objs = sorted(objs, key=lambda o: -float(o.get("score", 0.0)))[: int(max_objects)]
            for o in objs:
                o["mask"] = _to_hw(o["mask"], H, W)
                if refine and can_refine:
                    o["mask"], o["refine"] = _refine(det, frames[b], o["mask"])
                    refine_log.append({"frame": b, "label": o.get("label", ""), **o["refine"]})
            return objs

        can_refine = callable(getattr(det, "_prepare", None)) and hasattr(det, "SIDE")
        refine_log: List[Dict[str, Any]] = []
        if refine and not can_refine:
            notes.append("refine needs ComfyUI's own SAM 3 (core 0.39+); this model path cannot refine, so it was "
                         "skipped.")

        want_track = B > 1 and str(video) == "track whole clip"
        tracked = False
        corrections: List[Dict[str, Any]] = []
        recovery = {"objects": []} if recover_lost else None
        with torch.no_grad():               # never inference_mode: outputs must stay mutable downstream
            if want_track and callable(getattr(det, "track", None)):
                k = min(max(int(seed_frame), 0), B - 1)
                seed = detect(k, fpicks.get(k))
                later = sorted(f for f in fpicks if f > k)
                earlier = sorted(f for f in fpicks if f < k)
                if earlier:
                    notes.append(f"picks for frame(s) {earlier} were not used: corrections run forward from "
                                 f"seed_frame ({k}); move seed_frame earlier.")
                obj = np.zeros((B, len(seed), H, W), np.float32)
                if seed:
                    init = torch.from_numpy(np.stack([o["mask"] for o in seed]))
                    obj[k] = init.numpy()
                    seed_init = init
                    cursor = k
                    for f in later:                             # L9.172: corrections - track up to f, fix f, go on
                        seg = det.track(image[cursor:f + 1], init)
                        for t in range(1, f - cursor + 1):
                            obj[cursor + t] = _to_hw_stack(seg.frame(t), H, W)
                        found = detect(f, fpicks[f])
                        pairs = _match_greedy(obj[f], [o["mask"] for o in found])
                        for n, j in pairs:
                            obj[f, n] = found[j]["mask"]
                        corrections.append({"frame": f, "detections": len(found), "matched": len(pairs)})
                        if len(found) > len(pairs):
                            notes.append(f"Correction on frame {f}: {len(found) - len(pairs)} detection(s) overlapped "
                                         "no tracked object and were dropped (corrections fix existing objects; "
                                         "new ones need the seed frame).")
                        init = torch.from_numpy(obj[f].copy())
                        cursor = f
                    if cursor < B - 1:                          # one tracking state forward to the end ...
                        fwd = det.track(image[cursor:], init)
                        for t in range(1, B - cursor):
                            obj[cursor + t] = _to_hw_stack(fwd.frame(t), H, W)
                    if k > 0:                                   # ... and one backward to frame 0
                        bwd = det.track(image[: k + 1].flip(0), seed_init)
                        for t in range(1, k + 1):
                            obj[k - t] = _to_hw_stack(bwd.frame(t), H, W)
                elif later:
                    notes.append(f"Nothing was found on seed_frame {k}, so the corrections on frame(s) {later} had "
                                 "no tracked object to fix.")
                if recover_lost:                                # L9.172: re-find objects where tracking lost them
                    recovery = self._recover(det, frames, obj, seed, k, terms, notes, H, W)
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
                if fpicks and terms:
                    notes.append("Words were given, so the per-frame picks in 'frames' were not used.")
                if recover_lost:
                    notes.append("recover_lost works only when a clip is tracked (video = track whole clip on a "
                                 "model that can track); here every frame was segmented on its own, so it was "
                                 "skipped.")
                if fpicks and not terms and not pos and not boxes:
                    bare = [b for b in range(B) if b not in fpicks]
                    if bare:
                        notes.append(f"Frame(s) {bare} have no picks (no top-level clicks / boxes and no 'frames' "
                                     "entry), so nothing was segmented there.")
                union = np.zeros((B, H, W), np.float32)
                first, per_frame = [], []
                for b in range(B):
                    rows = []
                    for o in detect(b, None if terms else fpicks.get(b)):
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
                "per_frame": per_frame, "edge_stability": edge, "refine": refine_log if refine else None,
                "seconds": round(time.time() - t0, 3), "notes": notes}
        if fpicks:                                              # L9.172: keys only when the feature is used
            if tracked:
                info["corrections"] = corrections
            else:
                info["frame_picks"] = sorted(fpicks) if not terms else []
        if recover_lost:
            info["recovery"] = recovery
        return (torch.from_numpy(union), masks, [o["box"] for o in first], json.dumps(info))

    @staticmethod
    def _recover(det, frames, obj, seed, k, terms, notes, H, W):
        """L9.172 recover_lost: object n is lost on frame t when its area < 5 % of its area on the seed frame k. An
        object named by words is looked for again on each lost frame (det.detect with its word) and replaced by the
        detection that overlaps its nearest good mask most (earlier first, else later) - or, with no overlap, the
        nearest centroid within 2x that mask's box diagonal. Changes `obj` in place; returns the report."""
        B = obj.shape[0]
        report = {"objects": []}
        term_set = set(terms)
        for n, o in enumerate(seed):
            label = str(o.get("label", ""))
            seed_area = float(obj[k, n].sum())
            lost = [t for t in range(B) if t != k and seed_area > 0 and float(obj[t, n].sum()) < 0.05 * seed_area]
            recovered: List[int] = []
            if lost and label in term_set:
                bad = set(lost)
                for t in lost:
                    ref_t = next((s for s in range(t - 1, -1, -1) if s not in bad), None)
                    if ref_t is None:
                        ref_t = next((s for s in range(t + 1, B) if s not in bad), None)
                    if ref_t is None:
                        continue
                    cands = [_to_hw(d["mask"], H, W) for d in det.detect(frames[t], label, [], [], [])]
                    pick = _best_redetection(obj[ref_t, n], cands)
                    if pick is not None:
                        obj[t, n] = pick
                        recovered.append(t)
                        bad.discard(t)
            elif lost:
                notes.append(f"recover_lost: object {n} ({label or 'picked'}) is lost on frame(s) {lost}, but objects "
                             "picked by clicks / boxes cannot be looked for again without words; add a correction "
                             "for that frame in picks 'frames'.")
            report["objects"].append({"label": label, "lost_frames": lost, "recovered_frames": recovered})
        return report


def _parse_target(self, image, target, prompt, allow_download, groups_json=""):
    """face parts / clothes: one mask per named class (the prompt, comma separated; empty = a default set) from the
    Semantic Segment parser, their union, boxes from the first frame. Unknown class names are refused in words.
    L9.172: '@group' / '-x' in the prompt are expanded (nodes/_semantic_groups.py, plus groups_json) first."""
    from ._semantic_groups import expand_classes, parse_user_groups, uses_groups
    from .semantic_segment import CLOTHES_CLASSES, FACE_CLASSES, SemanticSegmentMEC
    model_name, default = _PARSERS[target]
    valid = [c for c in (FACE_CLASSES if target == "face parts" else CLOTHES_CLASSES) if c != "background"]
    raw = str(prompt or "").strip() or default
    groups_info = None
    if uses_groups(raw):
        classes, excluded, used = expand_classes(raw, valid, "face" if target == "face parts" else "clothes",
                                                 groups_json)
        groups_info = {"used": used, "excluded": excluded}
        if not classes:
            raise ValueError(f"Segment (C2C) {target}: nothing is left of {raw!r} after the exclusions; name at "
                             "least one class or group that is not removed.")
    else:
        parse_user_groups(groups_json)                     # a broken groups_json is reported even when unused
        classes = [c.strip() for c in raw.split(",") if c.strip()]
    bad = [c for c in classes if c not in valid]
    if bad:
        raise ValueError(f"Segment (C2C) {target}: unknown class name(s) {bad}. Use any of: {', '.join(valid)}.")
    t0 = time.time()
    parser = SemanticSegmentMEC()
    B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    per_class, parser_info = [], None
    with torch.no_grad():
        for c in classes:
            m, info_c = parser.parse(image, model_name, c, 0.5, False, True, allow_download)
            m = m.float()
            if m.dim() == 2:
                m = m.unsqueeze(0)
            per_class.append((c, (m > 0.5).float()))
            parser_info = parser_info or info_c
    union = torch.stack([m for _c, m in per_class]).amax(0)
    first = [(c, m[0].numpy()) for c, m in per_class]
    per_frame = [[{"label": c, "score": 1.0, "box": _bbox_of(m[b].numpy()), "area": int(m[b].sum())}
                  for c, m in per_class] for b in range(B)]
    notes = []
    try:
        pi = json.loads(parser_info) if isinstance(parser_info, str) else (parser_info or {})
    except (TypeError, ValueError):
        pi = {"raw": str(parser_info)[:300]}
    if isinstance(pi, dict) and any("model-free" in str(v) or "needs_the_model" in str(k) for k, v in pi.items()):
        notes.append("The parsing model is not installed, so a model-free stand-in ran (approximate); turn on "
                     "allow_download or install it.")
    info = {"target": target, "parser": model_name, "classes": classes, "frames": B, "tracked": False,
            "objects_first_frame": len(first), "per_frame": per_frame, "parser_info": pi,
            "seconds": round(time.time() - t0, 3), "notes": notes}
    if groups_info is not None:                            # L9.172: only when a group / exclusion was used
        info["groups"] = groups_info
    masks = torch.stack([torch.from_numpy(m) for _c, m in first]) if first else torch.zeros((1, H, W))
    return (union, masks, [_bbox_of(m) for _c, m in first], json.dumps(info))


SegmentC2C._parse = _parse_target

# L9.172 (A25 spec items 49/52): inputs appended after the A28 audit, in order
APPENDED_AFTER_A28 = ("recover_lost", "groups_json")


NODE_CLASS_MAPPINGS = {"SegmentC2C": SegmentC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"SegmentC2C": "Segment (C2C)"}
