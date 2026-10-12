"""Model-free masking: GrabCut, DIS optical flow and a guided filter (L7.91, owner 2026-10-12).

What a fresh install can do with no model weights, no third_party/ and no internet. Every result made here is an
APPROXIMATION and says so: callers label it ``approximate`` and never with a model's name ("never claim SAM/SEC
inference succeeded when a fallback was used"). The code is this pack's own (OpenCV + NumPy + torch).

- ``grabcut_segment``  one frame, seeded by points / a box / a mask (cv2.grabCut, at <= 512 px).
- ``ClassicVideoTracker``  the protocol the ONYX pipeline uses for the SAM 3.1 video model (init_state, add_prompt,
  propagate_in_video, reset_state): keyframes by GrabCut, the frames between by the previous mask warped with
  DIS optical flow and re-seeded GrabCut.
- ``guided_matte``  alpha from a trimap: guided filter (He et al.) in the unknown band.

Kill switch: C2C_MODEL_FALLBACK=off makes the callers raise the original error instead of falling back.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger("MEC.MaskMatting.ModelFree")

LABEL_SEGMENT = "model-free: GrabCut + DIS optical flow (approximate, no model)"
LABEL_MATTE = "model-free: guided filter (approximate, no model)"
TEXT_NEEDS_MODEL = ("Text prompts need a model (SAM 3 / SAM 3.1 or LocateAnything), and none is installed here. "
                    "The model-free mode takes points, a box or a mask.")

_WORK_SIDE = 512
_MISSING_RX = re.compile(
    r"not (present|found|installed|available|vendored)|missing|no such file|failed to load|cannot find|"
    r"could not (find|load|locate)|couldn'?t (connect|find)|can'?t load|cannot load|unable to load|"
    r"offline|download disabled|is required|"
    r"no ([a-z0-9. ]+ )?(model|weights?|checkpoint)",
    re.IGNORECASE)


def fallback_allowed() -> bool:
    """False when the user switched model-free fallbacks off (env C2C_MODEL_FALLBACK=off|0|false)."""
    return os.environ.get("C2C_MODEL_FALLBACK", "").strip().lower() not in ("off", "0", "false", "no")


def is_missing_model_error(exc: BaseException) -> bool:
    """A model, its package or its weights are absent - the case a model-free result may stand in for. A bug
    (TypeError, a shape error, ...) is not, and must surface."""
    if isinstance(exc, (ImportError, FileNotFoundError)):
        return True
    if isinstance(exc, (RuntimeError, ValueError, OSError)):
        return bool(_MISSING_RX.search(str(exc)))
    return False


def describe_fallback(wanted: str, exc: BaseException) -> Dict[str, str]:
    msg = " ".join(str(exc).split())
    return {"wanted": str(wanted), "why": (msg[:400] or type(exc).__name__)}


# ── one frame ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _cv2():
    import cv2
    return cv2


def _disc(mask: np.ndarray, x: float, y: float, r: int, value: int) -> None:
    _cv2().circle(mask, (int(round(x)), int(round(y))), max(1, int(r)), int(value), -1)


def _keep_seeded_components(fg: np.ndarray, seeds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Only the connected parts of ``fg`` that contain a seed: GrabCut likes to add look-alike patches elsewhere."""
    cv2 = _cv2()
    n, lab = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    if n <= 2 or not seeds:
        return fg
    h, w = fg.shape
    keep = set()
    for x, y in seeds:
        xi, yi = min(w - 1, max(0, int(round(x)))), min(h - 1, max(0, int(round(y))))
        if lab[yi, xi] > 0:
            keep.add(int(lab[yi, xi]))
    if not keep:
        return fg
    return np.isin(lab, list(keep))


def grabcut_segment(
    rgb_u8: np.ndarray,
    points_pos: Iterable[Tuple[float, float]] = (),
    points_neg: Iterable[Tuple[float, float]] = (),
    box: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[np.ndarray] = None,
    *,
    iterations: int = 5,
) -> Tuple[np.ndarray, float]:
    """One frame by GrabCut. Pixel coordinates; ``mask`` (H,W) seeds an approximate region. Returns the mask
    (H,W) float32 0/1 and an AGREEMENT score: the IoU between the seed region and the result (how little GrabCut had
    to change the seed) - a sanity measure, not a probability."""
    cv2 = _cv2()
    H, W = rgb_u8.shape[:2]
    s = min(1.0, _WORK_SIDE / float(max(H, W)))
    w, h = max(2, int(round(W * s))), max(2, int(round(H * s)))
    small = cv2.resize(rgb_u8, (w, h), interpolation=cv2.INTER_AREA) if s < 1.0 else rgb_u8.copy()
    small = np.ascontiguousarray(small[..., :3].astype(np.uint8))
    pos = [(x * s, y * s) for x, y in points_pos]
    neg = [(x * s, y * s) for x, y in points_neg]
    short = min(h, w)
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    seeds: List[Tuple[float, float]] = list(pos)

    if mask is not None:
        m = cv2.resize((np.asarray(mask) > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        if m.sum() < 1:
            return np.zeros((H, W), np.float32), 0.0
        k = max(1, int(round(short * 0.02)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        inner = cv2.erode(m, ker)
        outer = cv2.dilate(m, ker, iterations=3)
        gc[:] = cv2.GC_BGD
        gc[outer > 0] = cv2.GC_PR_BGD
        gc[m > 0] = cv2.GC_PR_FGD
        gc[inner > 0] = cv2.GC_FGD
        seed_region = m > 0
        if not seeds:
            ys, xs = np.nonzero(inner if inner.any() else m)
            seeds = [(float(x), float(y)) for x, y in zip(xs[:: max(1, len(xs) // 16)], ys[:: max(1, len(ys) // 16)])]
    elif box is not None:
        x0, y0, x1, y1 = [float(v) * s for v in box]
        x0, x1 = sorted((max(0.0, x0), min(w - 1.0, x1)))
        y0, y1 = sorted((max(0.0, y0), min(h - 1.0, y1)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("The box is too small to segment - draw a larger box.")
        gc[:] = cv2.GC_BGD
        gc[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1] = cv2.GC_PR_FGD
        seed_region = gc == cv2.GC_PR_FGD
        if not seeds:
            seeds = [((x0 + x1) / 2.0, (y0 + y1) / 2.0)]
    elif pos:
        # A point says "this", not "how big": grow a first guess by colour from each point (flood fill on a
        # blurred copy), cap it if it leaks over most of the frame, and let GrabCut settle the edge.
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        region = np.zeros((h, w), bool)
        for x, y in pos:
            ff = np.zeros((h + 2, w + 2), np.uint8)
            xi, yi = min(w - 1, max(0, int(round(x)))), min(h - 1, max(0, int(round(y))))
            cv2.floodFill(blur.copy(), ff, (xi, yi), (0, 0, 0), (14, 14, 14), (14, 14, 14),
                          4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
            grown = ff[1:-1, 1:-1] > 0
            if grown.mean() > 0.6:                       # leaked: fall back to a neighbourhood
                grown = np.zeros((h, w), np.uint8)
                _disc(grown, x, y, int(short * 0.15), 1)
                grown = grown > 0
            region |= grown
        gc[region] = cv2.GC_PR_FGD
        band = max(1, int(round(short * 0.02)))
        border = np.zeros((h, w), bool)
        border[:band], border[-band:], border[:, :band], border[:, -band:] = True, True, True, True
        gc[border & ~region] = cv2.GC_BGD               # the frame edge is background unless the subject touches it
        seed_region = region
    else:
        raise ValueError("Nothing to segment: give a point, a box or a mask.")

    for x, y in pos:
        _disc(gc, x, y, 3, cv2.GC_FGD)
    for x, y in neg:
        _disc(gc, x, y, max(3, int(short * 0.02)), cv2.GC_BGD)

    if not ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).any():
        return np.zeros((H, W), np.float32), 0.0
    if not ((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD)).any():
        fg = np.ones((h, w), bool)
    else:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, gc, None, bgd, fgd, int(max(1, iterations)), cv2.GC_INIT_WITH_MASK)
        except cv2.error as exc:                         # degenerate colour statistics: keep the seed
            logger.warning("[ModelFree] GrabCut gave up (%s); keeping the seed region.", exc)
        fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    fg = _keep_seeded_components(fg, seeds)
    inter = float(np.logical_and(fg, seed_region).sum())
    union = float(np.logical_or(fg, seed_region).sum())
    score = inter / union if union > 0 else 0.0
    if s < 1.0:
        fg = cv2.resize(fg.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
    return fg.astype(np.float32), float(score)


# ── video: the ONYX tracker protocol ──────────────────────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    u = float(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / u if u > 0 else 0.0


class ClassicVideoTracker:
    """Stands in for the SAM 3.1 video model inside ``_onyx_pipeline.run_onyx_pipeline``.

    ``add_prompt`` takes the same relative points and labels (1 positive, 0 negative, 2/3 box corners). Masks come
    back at frame resolution; ``out_probs`` is the IoU between the flow-warped previous mask and the re-seeded
    result - a drift measure (low = the subject moved or changed more than flow could follow)."""

    MODEL_FREE = True

    def init_state(self, resource_path=None, offload_video_to_cpu: bool = False, **_kw) -> dict:
        frames = []
        for p in resource_path or []:
            a = np.asarray(p.convert("RGB") if hasattr(p, "convert") else p)
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
            frames.append(np.ascontiguousarray(a[..., :3]))
        if not frames:
            raise ValueError("No frames to track.")
        return {"frames": frames, "keys": {}, "flow": {}}

    def add_prompt(self, state: dict, frame_idx: int, points=None, point_labels=None, obj_id: int = 1,
                   rel_coordinates: bool = True, **_kw) -> None:
        frames = state["frames"]
        fi = min(len(frames) - 1, max(0, int(frame_idx)))
        H, W = frames[fi].shape[:2]
        sx, sy = (W, H) if rel_coordinates else (1.0, 1.0)
        pos, neg, corners = [], [], {}
        for (x, y), lab in zip(points or [], point_labels or []):
            px, py = float(x) * sx, float(y) * sy
            lab = int(lab)
            if lab == 1:
                pos.append((px, py))
            elif lab == 0:
                neg.append((px, py))
            elif lab in (2, 3):
                corners[lab] = (px, py)
        box = None
        if 2 in corners and 3 in corners:
            (x0, y0), (x1, y1) = corners[2], corners[3]
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        mask, score = grabcut_segment(frames[fi], pos, neg, box)
        state["keys"].setdefault(int(obj_id), {})[fi] = (mask, score)

    def _flow_back(self, state: dict, cur: int, prev: int) -> np.ndarray:
        """Per pixel of frame ``cur``: where it was in ``prev`` (work resolution), DIS optical flow."""
        key = (cur, prev)
        if key not in state["flow"]:
            cv2 = _cv2()
            a, b = state["frames"][cur], state["frames"][prev]
            H, W = a.shape[:2]
            s = min(1.0, _WORK_SIDE / float(max(H, W)))
            size = (max(8, int(round(W * s))), max(8, int(round(H * s))))
            ga = cv2.cvtColor(cv2.resize(a, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            gb = cv2.cvtColor(cv2.resize(b, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            state["flow"] = {key: dis.calc(ga, gb, None)}       # one pair at a time: frames can be many
        return state["flow"][key]

    def _warp(self, state: dict, mask: np.ndarray, cur: int, prev: int) -> np.ndarray:
        cv2 = _cv2()
        flow = self._flow_back(state, cur, prev)
        h, w = flow.shape[:2]
        H, W = mask.shape
        small = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        warped = cv2.remap(small, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (cv2.resize(warped, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)

    def propagate_in_video(self, state: dict, start_frame_idx: int = 0, max_frame_num_to_track=None,
                           reverse: bool = False, **_kw) -> Iterator[Tuple[int, dict]]:
        frames, keys = state["frames"], state["keys"]
        B = len(frames)
        start = min(B - 1, max(0, int(start_frame_idx)))
        n = int(max_frame_num_to_track) if max_frame_num_to_track else B
        order = range(start, max(-1, start - n), -1) if reverse else range(start, min(B, start + n))
        obj_ids = sorted(keys)
        prev_masks: Dict[int, Optional[np.ndarray]] = {o: None for o in obj_ids}
        prev_fi: Optional[int] = None
        for fi in order:
            masks, probs = [], []
            for o in obj_ids:
                if fi in keys[o]:
                    m, sc = keys[o][fi]
                elif prev_masks[o] is not None and prev_fi is not None and prev_masks[o].any():
                    warped = self._warp(state, prev_masks[o], fi, prev_fi)
                    m, _agree = grabcut_segment(frames[fi], mask=warped, iterations=3)
                    sc = _iou(warped, m)
                else:
                    m, sc = np.zeros(frames[fi].shape[:2], np.float32), 0.0
                prev_masks[o] = m
                masks.append(m > 0.5)
                probs.append(float(sc))
            prev_fi = fi
            yield fi, {"out_obj_ids": list(obj_ids),
                       "out_binary_masks": np.stack(masks) if masks else np.zeros((0,) + frames[fi].shape[:2], bool),
                       "out_probs": probs}

    def reset_state(self, state: dict) -> None:
        state.clear()


class ClassicSamPredictor:
    """The SAM 2 image-predictor API (``set_image`` / ``predict``) over GrabCut, for the model-free SAM_MODEL the
    SAM Model Loader hands out when no SAM model is installed (L7.98). Pixel coordinates; labels 1 = positive,
    0 = negative. One mask per call - no invented alternatives - with the GrabCut agreement as its score.
    ``mask_input`` (the low-res logits of an earlier pass) seeds the next pass, as SAM's refinement loop expects."""

    MODEL_FREE = True

    def __init__(self):
        self._img: Optional[np.ndarray] = None

    def set_image(self, img_np) -> None:
        a = np.asarray(img_np)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if float(a.max()) <= 1.0 else a.astype(np.uint8)
        self._img = np.ascontiguousarray(a[..., :3])

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True,
                return_logits=False, normalize_coords=True, **_kw):
        if self._img is None:
            raise RuntimeError("set_image() first")
        cv2 = _cv2()
        H, W = self._img.shape[:2]
        pos: List[Tuple[float, float]] = []
        neg: List[Tuple[float, float]] = []
        if point_coords is not None and len(point_coords):
            pc = np.asarray(point_coords, np.float64).reshape(-1, 2)
            pl = (np.asarray(point_labels).reshape(-1) if point_labels is not None else np.ones(len(pc), int))
            for (x, y), lab in zip(pc, pl):
                if int(lab) == 1:
                    pos.append((float(x), float(y)))
                elif int(lab) == 0:
                    neg.append((float(x), float(y)))
        b = tuple(float(v) for v in np.asarray(box, np.float64).reshape(-1)[:4]) if box is not None else None
        seed = None
        if mask_input is not None:
            mi = np.asarray(mask_input, np.float32)
            mi = mi.reshape((-1,) + mi.shape[-2:])[0]
            seed = cv2.resize((mi > 0).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        if seed is not None and seed.any():
            m, score = grabcut_segment(self._img, pos, neg, None, mask=seed)
        else:
            m, score = grabcut_segment(self._img, pos, neg, b)
        low = (cv2.resize(m, (256, 256), interpolation=cv2.INTER_LINEAR) * 20.0 - 10.0)[None].astype(np.float32)
        masks = m[None].astype(np.float32) if return_logits else (m[None] > 0.5)
        return masks, np.array([score], np.float32), low


def model_free_sam(wanted: str, exc: BaseException) -> dict:
    """The SAM_MODEL the loader hands out when the chosen SAM model cannot be loaded: no weights inside - every SAM
    node that receives it runs GrabCut and labels its result model-free / approximate."""
    return {"model": None, "model_type": "model-free", "device": "cpu", "original_device": "cpu",
            "offload_to_cpu": False, "dtype": torch.float32, "model_path": "", "load_method": "model-free",
            "model_free": True, "label": LABEL_SEGMENT, "fallback_from": describe_fallback(wanted, exc)}


def text_needs_model(*_a, **_kw):
    """The ONYX ``seg_one_fn`` slot when no text-capable model is installed (only text keyframes call it)."""
    raise ValueError(TEXT_NEEDS_MODEL)


# ── matte ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def guided_matte(image_bhwc: torch.Tensor, trimap_bhw: torch.Tensor, *, radius: int = 8,
                 epsilon: float = 1e-4) -> torch.Tensor:
    """Alpha from a trimap (1 = subject, 0 = background, between = unknown): a guided filter of the trimap's
    0 / 0.5 / 1 guess, used only inside the unknown band. Approximate - it follows the image's edges, it does not
    know what hair is."""
    from . import _vfx
    img = image_bhwc.float()
    tri = trimap_bhw.float().to(img.device)
    if tri.ndim == 2:
        tri = tri.unsqueeze(0)
    H, W = tri.shape[-2:]
    r = int(max(1, min(int(radius), (min(H, W) - 1) // 2)))
    fg = tri >= 0.75
    unknown = (tri > 0.25) & ~fg
    guess = fg.float() + 0.5 * unknown.float()
    if min(H, W) < 3:
        return guess.clamp(0, 1)
    soft = _vfx.guided_refine(img[..., :3] if img.shape[-1] >= 3 else img, guess, radius=r, epsilon=float(epsilon))
    return torch.where(unknown, soft, fg.float()).clamp(0, 1)
