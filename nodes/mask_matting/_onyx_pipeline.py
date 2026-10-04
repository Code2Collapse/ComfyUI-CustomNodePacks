"""ONYX-style SAM3.1 video session + tiled ViTMatte pipeline for MaskOpsMEC."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from .utils import (
    free_vram,
    morph_dilate,
    morph_erode,
    parse_bbox,
    parse_points,
    to_mask,
)

logger = logging.getLogger("MEC.MaskOps.Onyx")

MAX_OBJECTS = 16
_TILE_OVERLAP = 128
_TILE_MARGIN = 32


@dataclass
class KeyframeSpec:
    frame: int
    points_pos: List[Tuple[float, float]] = field(default_factory=list)
    points_neg: List[Tuple[float, float]] = field(default_factory=list)
    box: Optional[Tuple[int, int, int, int]] = None
    text: Optional[str] = None
    mask_source: Optional[Literal["input"]] = None


@dataclass
class ObjectSpec:
    id: int
    label: str = ""
    color: str = ""
    enabled: bool = True
    opacity: float = 1.0
    keyframes: List[KeyframeSpec] = field(default_factory=list)


@dataclass
class TrackingSpec:
    direction: Literal["forward", "backward", "both"] = "forward"
    start_frame: int = 0
    max_frames: int = 0


@dataclass
class SceneSpec:
    version: int
    objects: List[ObjectSpec]
    tracking: TrackingSpec


@dataclass
class LegacyPrompts:
    frame_annotation: int
    tracking_direction: str
    start_frame: int
    max_frames_to_track: int
    positive_coords: str
    negative_coords: str
    pos_bbox: Any
    neg_bbox: Any
    normal_bbox: Any
    text_prompt: str


@dataclass
class OnyxResult:
    mask_coarse: torch.Tensor
    alpha_combined: torch.Tensor
    object_alphas: torch.Tensor
    object_coarse: torch.Tensor
    trimap_combined: torch.Tensor
    trimaps_per_object: torch.Tensor
    segmenter_score: float
    objects_meta: List[Dict[str, Any]]
    pipeline_info: Dict[str, Any]


def _clamp_frame(f: int, B: int) -> int:
    return max(0, min(B - 1, int(f)))


def _norm_direction(raw: str) -> Literal["forward", "backward", "both"]:
    s = (raw or "forward").strip().lower()
    if s in ("both", "bidirectional"):
        return "both"
    if s == "backward":
        return "backward"
    return "forward"


def _parse_keyframe(raw: dict, *, B: int, H: int, W: int, obj_id: int) -> KeyframeSpec:
    if "frame" not in raw:
        raise ValueError(f"Object {obj_id}: each keyframe needs a frame index.")
    frame = int(raw["frame"])
    if frame < 0 or frame >= B:
        raise ValueError(
            f"Object {obj_id}: keyframe frame {frame} is out of range "
            f"(clip has {B} frame{'s' if B != 1 else ''}, valid 0..{B - 1})."
        )
    kf = KeyframeSpec(frame=frame)
    pts = raw.get("points") or {}
    if isinstance(pts, dict):
        for p in pts.get("positive") or []:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                raise ValueError(f"Object {obj_id}: positive points must be [[x,y],...].")
            kf.points_pos.append((float(p[0]), float(p[1])))
        for p in pts.get("negative") or []:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                raise ValueError(f"Object {obj_id}: negative points must be [[x,y],...].")
            kf.points_neg.append((float(p[0]), float(p[1])))
    box = raw.get("box")
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"Object {obj_id}: box must be [x0,y0,x1,y1] in pixels.")
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Object {obj_id}: box on frame {frame} is empty or inverted.")
        kf.box = (
            max(0, min(W - 1, x0)),
            max(0, min(H - 1, y0)),
            max(0, min(W, x1)),
            max(0, min(H, y1)),
        )
    text = raw.get("text")
    if text is not None and str(text).strip():
        kf.text = str(text).strip()
    mask_src = raw.get("mask")
    if mask_src is not None:
        if str(mask_src).lower() != "input":
            raise ValueError(
                f"Object {obj_id}: mask keyframe value must be \"input\" "
                f"(got {mask_src!r})."
            )
        kf.mask_source = "input"
    if not (kf.points_pos or kf.points_neg or kf.box or kf.text or kf.mask_source):
        raise ValueError(
            f"Object {obj_id}: keyframe on frame {frame} has no prompt "
            f"(add points, box, text, or mask:\"input\")."
        )
    return kf


def parse_scene_prompts(raw: str, *, B: int, H: int, W: int) -> SceneSpec:
    if not raw or not str(raw).strip():
        raise ValueError("scene_prompts is empty — provide JSON or use legacy prompt sockets.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"scene_prompts is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("scene_prompts must be a JSON object.")
    version = int(data.get("version", 0))
    if version != 1:
        raise ValueError(f"scene_prompts version must be 1 (got {version}).")
    objs_raw = data.get("objects")
    if not isinstance(objs_raw, list) or not objs_raw:
        raise ValueError("scene_prompts needs at least one object in objects[].")
    if len(objs_raw) > MAX_OBJECTS:
        raise ValueError(
            f"Too many objects ({len(objs_raw)}). ONYX supports up to {MAX_OBJECTS}."
        )
    seen_ids: set[int] = set()
    objects: List[ObjectSpec] = []
    for o in objs_raw:
        if not isinstance(o, dict):
            raise ValueError("Each entry in objects[] must be an object.")
        oid = int(o.get("id", 0))
        if oid < 1 or oid > MAX_OBJECTS:
            raise ValueError(f"Object id must be 1..{MAX_OBJECTS} (got {oid}).")
        if oid in seen_ids:
            raise ValueError(f"Duplicate object id {oid}.")
        seen_ids.add(oid)
        opacity = float(o.get("opacity", 1.0))
        if opacity < 0.0 or opacity > 1.0:
            raise ValueError(f"Object {oid}: opacity must be 0..1 (got {opacity}).")
        kfs_raw = o.get("keyframes") or []
        if not isinstance(kfs_raw, list) or not kfs_raw:
            raise ValueError(f"Object {oid} has no keyframes — add at least one prompt.")
        keyframes = [_parse_keyframe(k, B=B, H=H, W=W, obj_id=oid) for k in kfs_raw]
        objects.append(ObjectSpec(
            id=oid,
            label=str(o.get("label") or ""),
            color=str(o.get("color") or ""),
            enabled=bool(o.get("enabled", True)),
            opacity=opacity,
            keyframes=keyframes,
        ))
    tr = data.get("tracking") or {}
    if not isinstance(tr, dict):
        tr = {}
    tracking = TrackingSpec(
        direction=_norm_direction(str(tr.get("direction", "forward"))),
        start_frame=max(0, int(tr.get("start_frame", 0))),
        max_frames=max(0, int(tr.get("max_frames", 0))),
    )
    return SceneSpec(version=1, objects=objects, tracking=tracking)


def legacy_to_scene(
    *,
    B: int,
    H: int,
    W: int,
    legacy: LegacyPrompts,
    external_mask: Optional[torch.Tensor],
) -> SceneSpec:
    pos_pts, _ = parse_points(legacy.positive_coords or "")
    neg_a, neg_b = parse_points(legacy.negative_coords or "")
    neg_pts = neg_a + neg_b
    bbox = parse_bbox(legacy.pos_bbox) or parse_bbox(legacy.normal_bbox)
    text = (legacy.text_prompt or "").strip() or None
    frame = _clamp_frame(legacy.frame_annotation, B)
    kf = KeyframeSpec(frame=frame)
    kf.points_pos = [(float(x), float(y)) for x, y in pos_pts]
    kf.points_neg = [(float(x), float(y)) for x, y in neg_pts]
    if bbox is not None:
        kf.box = bbox
    if text:
        kf.text = text
    if external_mask is not None:
        kf.mask_source = "input"
    if not (kf.points_pos or kf.points_neg or kf.box or kf.text or kf.mask_source):
        raise ValueError(
            "No prompts found — wire points, bbox, text, or external_mask, "
            "or provide scene_prompts JSON."
        )
    return SceneSpec(
        version=1,
        objects=[ObjectSpec(id=1, label="object 1", enabled=True, opacity=1.0, keyframes=[kf])],
        tracking=TrackingSpec(
            direction=_norm_direction(legacy.tracking_direction),
            start_frame=max(0, int(legacy.start_frame)),
            max_frames=max(0, int(legacy.max_frames_to_track)),
        ),
    )


def resolve_scene(
    scene_raw: str,
    *,
    B: int,
    H: int,
    W: int,
    legacy: LegacyPrompts,
    external_mask: Optional[torch.Tensor],
) -> SceneSpec:
    if scene_raw and str(scene_raw).strip():
        return parse_scene_prompts(scene_raw, B=B, H=H, W=W)
    return legacy_to_scene(B=B, H=H, W=W, legacy=legacy, external_mask=external_mask)


def band_px(H: int, W: int, band_scale: float) -> int:
    return max(3, int(round(float(band_scale) * max(H, W) * 0.004)))


def _interior_points(mask_hw: np.ndarray, n: int = 3) -> List[Tuple[float, float]]:
    m = (mask_hw > 0.5).astype(np.uint8)
    if m.sum() < 1:
        return []
    try:
        from scipy import ndimage
        dist = ndimage.distance_transform_edt(m)
        pts: List[Tuple[float, float]] = []
        work = dist.copy()
        for _ in range(n):
            flat = int(work.argmax())
            if work.flat[flat] <= 0:
                break
            y, x = divmod(flat, work.shape[1])
            pts.append((float(x), float(y)))
            work[max(0, y - 5):y + 6, max(0, x - 5):x + 6] = 0
        if pts:
            return pts
    except Exception:
        pass
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return []
    cx, cy = float(xs.mean()), float(ys.mean())
    return [(cx, cy)]


def mask_to_tracker_prompt(
    mask_hw: np.ndarray,
    W: int,
    H: int,
    *,
    add_negatives: bool = True,
) -> Tuple[List[List[float]], List[int]]:
    m = (mask_hw > 0.5).astype(np.float32)
    if m.sum() < 1:
        raise ValueError("Mask seed is empty — draw a larger region.")
    ys, xs = np.where(m > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    points: List[List[float]] = [
        [x0 / max(W, 1), y0 / max(H, 1)],
        [x1 / max(W, 1), y1 / max(H, 1)],
    ]
    labels: List[int] = [2, 3]
    for px, py in _interior_points(m, 3):
        points.append([px / max(W, 1), py / max(H, 1)])
        labels.append(1)
    if add_negatives:
        dil = morph_dilate(torch.from_numpy(m).unsqueeze(0), 3)[0].numpy()
        ring = np.clip(dil - m, 0, 1)
        ry, rx = np.where(ring > 0.5)
        if len(rx) > 0:
            step = max(1, len(rx) // 3)
            for i in range(0, min(len(rx), 3 * step), step):
                points.append([float(rx[i]) / max(W, 1), float(ry[i]) / max(H, 1)])
                labels.append(0)
    return points, labels


def keyframe_to_tracker_prompt(
    kf: KeyframeSpec,
    W: int,
    H: int,
) -> Tuple[List[List[float]], List[int]]:
    points: List[List[float]] = []
    labels: List[int] = []
    if kf.box is not None:
        x0, y0, x1, y1 = kf.box
        points.extend([
            [x0 / max(W, 1), y0 / max(H, 1)],
            [x1 / max(W, 1), y1 / max(H, 1)],
        ])
        labels.extend([2, 3])
    for px, py in kf.points_pos:
        points.append([px / max(W, 1), py / max(H, 1)])
        labels.append(1)
    for px, py in kf.points_neg:
        points.append([px / max(W, 1), py / max(H, 1)])
        labels.append(0)
    if not points:
        raise ValueError(f"Keyframe on frame {kf.frame} has no point or box prompts.")
    return points, labels


def _frames_to_pil(img_bhwc: torch.Tensor) -> List[Any]:
    from PIL import Image
    out = []
    for i in range(img_bhwc.shape[0]):
        u8 = (img_bhwc[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(u8))
    return out


def _mask_frame(external_mask: Optional[torch.Tensor], frame: int, B: int, H: int, W: int) -> np.ndarray:
    if external_mask is None:
        raise ValueError(
            f"Keyframe on frame {frame} uses mask:\"input\" but no external_mask is connected."
        )
    em = to_mask(external_mask)
    if em.ndim == 2:
        m = em
    elif em.shape[0] == 1:
        m = em[0]
    else:
        fi = _clamp_frame(frame, em.shape[0])
        m = em[fi]
    if m.shape[-2:] != (H, W):
        # A rough prompt mask (ONYX's "mask prompt") is often drawn on a proxy
        # or comes from another resolution; it only seeds the tracker, so it is
        # scaled onto the plate rather than refused.
        import torch.nn.functional as F
        m = F.interpolate(m.float()[None, None], size=(H, W), mode="bilinear",
                          align_corners=False)[0, 0]
    return m.cpu().numpy().astype(np.float32)


def _output_id_list(val) -> List[int]:
    if val is None:
        return []
    if isinstance(val, torch.Tensor):
        return [int(x) for x in val.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in val]


def _output_prob_at(probs, j: int) -> Optional[float]:
    if probs is None:
        return None
    if isinstance(probs, torch.Tensor):
        return float(probs[j].detach().cpu().item())
    p = probs[j]
    return float(p.item()) if hasattr(p, "item") else float(p)


def _output_mask_at(masks, j: int) -> np.ndarray:
    if isinstance(masks, torch.Tensor):
        return masks[j].detach().cpu().numpy()
    m = masks[j]
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    return np.asarray(m)


def seed_all_objects(
    model,
    state,
    scene: SceneSpec,
    *,
    img_bhwc: torch.Tensor,
    seg_one_fn: Callable,
    external_mask: Optional[torch.Tensor],
) -> List[Dict[str, Any]]:
    B, H, W, _ = img_bhwc.shape
    calls: List[Dict[str, Any]] = []
    items: List[Tuple[int, KeyframeSpec]] = []
    for obj in scene.objects:
        if not obj.enabled:
            continue
        for kf in obj.keyframes:
            items.append((obj.id, kf))
    items.sort(key=lambda t: (t[1].frame, t[0]))
    for obj_id, kf in items:
        if kf.text:
            pos = [(float(x), float(y)) for x, y in kf.points_pos]
            neg = [(float(x), float(y)) for x, y in kf.points_neg]
            bbox = kf.box
            frame_np = img_bhwc[kf.frame].cpu().numpy()
            mask_hw, _score = seg_one_fn(
                frame_np, pos, neg, bbox, None, kf.text,
            )
            fg_count = int(np.sum(np.asarray(mask_hw) > 0.5))
            if fg_count < 1:
                raise ValueError(
                    f"Object {obj_id}: '{kf.text}' found nothing on frame {kf.frame} - "
                    f"try other words or add a point/box."
                )
            if kf.points_pos or kf.points_neg or kf.box:
                pts, lbls = keyframe_to_tracker_prompt(kf, W, H)
                kind = "text+points/box→tracker"
            else:
                pts, lbls = mask_to_tracker_prompt(mask_hw, W, H)
                kind = "text→mask→tracker"
        elif kf.mask_source == "input":
            mask_hw = _mask_frame(external_mask, kf.frame, B, H, W)
            pts, lbls = mask_to_tracker_prompt(mask_hw, W, H)
            kind = "mask→tracker"
        else:
            pts, lbls = keyframe_to_tracker_prompt(kf, W, H)
            kind = "points/box"
        model.add_prompt(
            state,
            frame_idx=int(kf.frame),
            points=pts,
            point_labels=lbls,
            obj_id=int(obj_id),
            rel_coordinates=True,
        )
        calls.append({
            "obj_id": obj_id,
            "frame": int(kf.frame),
            "kind": kind,
            "n_points": len(pts),
            "labels": list(lbls),
        })
    return calls


def _enabled_objects(scene) -> List[ObjectSpec]:
    """Enabled objects of a SceneSpec, or of a plain list of ObjectSpec
    (combine_alphas / pack_object_alphas are handed `scene.objects`)."""
    objs = scene.objects if isinstance(scene, SceneSpec) else list(scene or [])
    return [o for o in objs if o.enabled]


def _keyframe_bounds(scene: SceneSpec) -> Tuple[int, int]:
    frames = [kf.frame for o in _enabled_objects(scene) for kf in o.keyframes]
    if not frames:
        return 0, 0
    return min(frames), max(frames)


def _max_track_forward(start: int, B: int, max_frames: int) -> int:
    if max_frames > 0:
        return max_frames
    return max(1, B - start)


def _max_track_backward(start: int, max_frames: int) -> int:
    if max_frames > 0:
        return max_frames
    return max(1, start + 1)


def propagate_objects(
    model,
    state,
    scene: SceneSpec,
    *,
    B: int,
    H: int,
    W: int,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    enabled = _enabled_objects(scene)
    if not enabled:
        raise ValueError("No enabled objects to propagate.")
    O = len(enabled)
    id_to_idx = {o.id: i for i, o in enumerate(enabled)}
    coarse = torch.zeros((O, B, H, W), dtype=torch.float32)
    scores: Dict[int, float] = {o.id: 0.0 for o in enabled}
    earliest, latest = _keyframe_bounds(scene)
    start_fwd = max(earliest, scene.tracking.start_frame)
    start_fwd = _clamp_frame(start_fwd, B)
    start_bwd = _clamp_frame(latest, B)
    direction = scene.tracking.direction
    max_f = scene.tracking.max_frames

    def _ingest(frame_idx: int, outputs: dict) -> None:
        obj_ids = _output_id_list(outputs.get("out_obj_ids"))
        masks = outputs.get("out_binary_masks")
        probs = outputs.get("out_probs")
        if masks is None:
            return
        for j, oid in enumerate(obj_ids):
            oid = int(oid)
            if oid not in id_to_idx:
                continue
            oi = id_to_idx[oid]
            m = torch.from_numpy(_output_mask_at(masks, j).astype(np.float32))
            if tuple(m.shape) != (H, W):
                # SAM3 hands masks back at video resolution; anything else
                # (a low-res tracker output) is resampled onto the plate
                # rather than crashing the whole run.
                import torch.nn.functional as F
                m = (F.interpolate(m[None, None], size=(H, W), mode="bilinear",
                                   align_corners=False)[0, 0] > 0.5).float()
            coarse[oi, frame_idx] = m
            sc = _output_prob_at(probs, j)
            if sc is not None:
                scores[oid] = max(scores.get(oid, 0.0), sc)

    if direction in ("forward", "both"):
        mf = _max_track_forward(start_fwd, B, max_f)
        for fi, outputs in model.propagate_in_video(
            state,
            start_frame_idx=start_fwd,
            max_frame_num_to_track=mf,
            reverse=False,
        ):
            _ingest(int(fi), outputs if isinstance(outputs, dict) else {})

    if direction in ("backward", "both"):
        mb = _max_track_backward(start_bwd, max_f)
        for fi, outputs in model.propagate_in_video(
            state,
            start_frame_idx=start_bwd,
            max_frame_num_to_track=mb,
            reverse=True,
        ):
            _ingest(int(fi), outputs if isinstance(outputs, dict) else {})

    return coarse, scores


def trimap_resaware(coarse_obhw: torch.Tensor, band_scale: float) -> torch.Tensor:
    O, B, H, W = coarse_obhw.shape
    band = band_px(H, W, band_scale)
    flat = (coarse_obhw > 0.5).float().reshape(O * B, 1, H, W)
    fg = morph_erode(flat, band).reshape(O, B, H, W)
    outer = morph_dilate(flat, band).reshape(O, B, H, W)
    unknown = (outer - fg).clamp(0, 1)
    return (fg + unknown * 0.5).clamp(0, 1)


def combined_coarse_mask(coarse_obhw: torch.Tensor, objects: Sequence[ObjectSpec]) -> torch.Tensor:
    O, B, H, W = coarse_obhw.shape
    out = torch.zeros((B, H, W), dtype=torch.float32)
    enabled = _enabled_objects(list(objects))
    for i, _obj in enumerate(enabled):
        out = torch.maximum(out, coarse_obhw[i])
    return (out > 0.5).float()


def _unknown_bbox(trimap_hw: torch.Tensor, margin: int) -> Tuple[int, int, int, int]:
    unk = (trimap_hw > 0.25) & (trimap_hw < 0.75)
    if not bool(unk.any()):
        ys, xs = torch.where(trimap_hw > 0.5)
        if len(xs) == 0:
            return 0, 0, trimap_hw.shape[1], trimap_hw.shape[0]
    else:
        ys, xs = torch.where(unk)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(trimap_hw.shape[0], int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(trimap_hw.shape[1], int(xs.max()) + 1 + margin)
    return x0, y0, x1, y1


def _feather_weights(h: int, w: int, overlap: int) -> torch.Tensor:
    yy = torch.linspace(0, 1, h).view(-1, 1).expand(h, w)
    xx = torch.linspace(0, 1, w).view(1, -1).expand(h, w)
    wy = torch.ones(h, 1)
    wx = torch.ones(1, w)
    if overlap > 0 and h > overlap * 2:
        ramp = torch.linspace(0, 1, overlap)
        wy[:overlap, 0] = ramp
        wy[-overlap:, 0] = ramp.flip(0)
    if overlap > 0 and w > overlap * 2:
        ramp = torch.linspace(0, 1, overlap)
        wx[0, :overlap] = ramp
        wx[0, -overlap:] = ramp.flip(0)
    return (wy * wx).clamp(min=1e-6)


def _matte_crop(matter, frame_u8, tri_u8, crop_h: int, crop_w: int) -> torch.Tensor:
    """Run matter on a crop; returns alpha (crop_h, crop_w)."""
    img_t = torch.from_numpy(frame_u8).float().div(255.0).unsqueeze(0)
    tri_t = torch.from_numpy(tri_u8).float().div(255.0).unsqueeze(0)
    out = matter.matte(img_t, torch.zeros_like(tri_t), trimap=tri_t)
    alpha = out["alpha"][0]
    if alpha.shape[-2:] != (crop_h, crop_w):
        import torch.nn.functional as F
        alpha = F.interpolate(
            alpha.unsqueeze(0).unsqueeze(0),
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return alpha.clamp(0, 1)


def refine_frame_tiled(
    matter,
    frame_hwc: torch.Tensor,
    trimap_hw: torch.Tensor,
    coarse_hw: torch.Tensor,
    *,
    matte_tile: int,
    tile_overlap: int = _TILE_OVERLAP,
    margin: int = _TILE_MARGIN,
) -> torch.Tensor:
    H, W = trimap_hw.shape
    x0, y0, x1, y1 = _unknown_bbox(trimap_hw, margin)
    if x1 <= x0 or y1 <= y0:
        return coarse_hw.clamp(0, 1)
    crop_h, crop_w = y1 - y0, x1 - x0
    frame_u8 = (frame_hwc.cpu().numpy() * 255).astype(np.uint8)
    tri_crop = (trimap_hw[y0:y1, x0:x1].numpy() * 255).astype(np.uint8)
    frame_crop = frame_u8[y0:y1, x0:x1]
    alpha_crop = torch.zeros((crop_h, crop_w), dtype=torch.float32)
    weight_crop = torch.zeros((crop_h, crop_w), dtype=torch.float32)
    tile = max(64, int(matte_tile))
    step = max(1, tile - tile_overlap)
    for y in range(0, crop_h, step):
        for x in range(0, crop_w, step):
            ty0, tx0 = y, x
            ty1 = min(crop_h, ty0 + tile)
            tx1 = min(crop_w, tx0 + tile)
            if ty1 <= ty0 or tx1 <= tx0:
                continue
            sub_tri = tri_crop[ty0:ty1, tx0:tx1]
            sub_tri_f = sub_tri.astype(np.float32)
            has_unknown = ((sub_tri_f > 10.0) & (sub_tri_f < 245.0)).any()
            if not has_unknown:
                continue
            sub_frame = frame_crop[ty0:ty1, tx0:tx1]
            sub_alpha = _matte_crop(matter, sub_frame, sub_tri, ty1 - ty0, tx1 - tx0)
            w = _feather_weights(ty1 - ty0, tx1 - tx0, tile_overlap)
            alpha_crop[ty0:ty1, tx0:tx1] += sub_alpha * w
            weight_crop[ty0:ty1, tx0:tx1] += w
    weight_crop = weight_crop.clamp(min=1e-6)
    alpha_crop = (alpha_crop / weight_crop).clamp(0, 1)
    full = coarse_hw.clone().float()
    full[y0:y1, x0:x1] = alpha_crop
    full = torch.where(trimap_hw >= 0.99, torch.ones_like(full), full)
    full = torch.where(trimap_hw <= 0.01, torch.zeros_like(full), full)
    return full.clamp(0, 1)


def refine_object_tiled(
    matter,
    img_bhwc: torch.Tensor,
    trimap_obhw: torch.Tensor,
    coarse_obhw: torch.Tensor,
    *,
    matte_tile: int,
) -> torch.Tensor:
    O, B, H, W = trimap_obhw.shape
    out = torch.zeros_like(coarse_obhw)
    for o in range(O):
        for b in range(B):
            out[o, b] = refine_frame_tiled(
                matter,
                img_bhwc[b],
                trimap_obhw[o, b],
                coarse_obhw[o, b],
                matte_tile=matte_tile,
            )
    return out.clamp(0, 1)


def stabilise_alpha_unknown(
    alpha_obhw: torch.Tensor,
    trimap_obhw: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    if not enabled or alpha_obhw.ndim != 4 or alpha_obhw.shape[1] < 3:
        return alpha_obhw
    from ._auto_quality import temporal_alpha_median
    out = alpha_obhw.clone()
    O = alpha_obhw.shape[0]
    for o in range(O):
        med = temporal_alpha_median(alpha_obhw[o])
        unk = (trimap_obhw[o] > 0.25) & (trimap_obhw[o] < 0.75)
        out[o] = torch.where(unk, med, alpha_obhw[o])
    return out.clamp(0, 1)


def combine_alphas(
    alpha_obhw: torch.Tensor,
    objects: Sequence[ObjectSpec],
) -> torch.Tensor:
    enabled = _enabled_objects(list(objects))
    if not enabled:
        return torch.zeros(alpha_obhw.shape[1:])
    B, H, W = alpha_obhw.shape[1], alpha_obhw.shape[2], alpha_obhw.shape[3]
    combined = torch.zeros((B, H, W), dtype=torch.float32)
    for i, obj in enumerate(enabled):
        weighted = alpha_obhw[i] * float(obj.opacity)
        combined = torch.maximum(combined, weighted)
    return combined.clamp(0, 1)


def pack_object_alphas(alpha_obhw: torch.Tensor, objects: Sequence[ObjectSpec]) -> torch.Tensor:
    planes: List[torch.Tensor] = []
    enabled = _enabled_objects(list(objects))
    for i, _obj in enumerate(enabled):
        for b in range(alpha_obhw.shape[1]):
            planes.append(alpha_obhw[i, b])
    if not planes:
        return torch.zeros((0, alpha_obhw.shape[2], alpha_obhw.shape[3]))
    return torch.stack(planes, 0)


def ensure_sam31_video_model(
    *,
    model_name: str,
    device: str,
    precision: str,
    attention: str,
    offload: str,
    auto_download: bool,
):
    from .segmenters.sam31_backend import SAM31Segmenter, _have_vendor
    from .segmenters import get_segmenter_cls
    from .node import _resolve_model_choice

    if not _have_vendor():
        raise RuntimeError(
            "SAM 3.1 library is missing from this install. Reinstall "
            "ComfyUI-CustomNodePacks (expected third_party/sam3_lib)."
        )
    cls = get_segmenter_cls("sam3.1")
    if cls is None:
        raise RuntimeError("SAM 3.1 segmenter is not registered in this install.")
    resolved = _resolve_model_choice(
        model_name or "(auto)", cls.MODELS_KEY, auto_download=bool(auto_download),
    )
    seg = cls(
        model_name=resolved,
        device=device,
        precision=precision,
        attention=attention,
        offload=offload,
    )
    seg.load()
    if seg._video_predictor is None or seg._video_predictor.model is None:
        raise RuntimeError("SAM 3.1 video model failed to load.")
    return seg, seg._video_predictor.model


def run_onyx_pipeline(
    img_bhwc: torch.Tensor,
    *,
    scene_raw: str,
    legacy: LegacyPrompts,
    matter_key: str,
    matter_model: str,
    matter_cls,
    sam31_model_name: str,
    device: str,
    precision: str,
    attention: str,
    offload: str,
    auto_download: bool,
    band_scale: float,
    matte_tile: int,
    temporal_stabilise: bool,
    post_refine: str,
    external_mask: Optional[torch.Tensor],
    external_trimap: Optional[torch.Tensor],
    video_model=None,
    seg_one_fn=None,
    matter_instance=None,
) -> OnyxResult:
    B, H, W, _ = img_bhwc.shape
    legacy_prompts = legacy
    scene = resolve_scene(
        scene_raw, B=B, H=H, W=W, legacy=legacy_prompts, external_mask=external_mask,
    )
    enabled = _enabled_objects(scene.objects)
    if not enabled:
        raise ValueError("No enabled objects in scene.")

    pipeline_info: Dict[str, Any] = {"pipeline": "onyx", "version": 1}
    seg_inst = None
    model = video_model
    if model is None:
        seg_inst, model = ensure_sam31_video_model(
            model_name=sam31_model_name,
            device=device,
            precision=precision,
            attention=attention,
            offload=offload,
            auto_download=auto_download,
        )
    if seg_one_fn is None:
        if seg_inst is None:
            seg_inst, _ = ensure_sam31_video_model(
                model_name=sam31_model_name,
                device=device,
                precision=precision,
                attention=attention,
                offload=offload,
                auto_download=auto_download,
            )
        seg_one_fn = seg_inst._seg_one

    pils = _frames_to_pil(img_bhwc)
    state = model.init_state(
        resource_path=pils,
        offload_video_to_cpu=(B >= 8),
    )
    try:
        seed_calls = seed_all_objects(
            model, state, scene,
            img_bhwc=img_bhwc,
            seg_one_fn=seg_one_fn,
            external_mask=external_mask,
        )
        pipeline_info["seed_calls"] = seed_calls
        coarse, obj_scores = propagate_objects(
            model, state, scene, B=B, H=H, W=W,
        )
    finally:
        try:
            model.reset_state(state)
        except Exception as exc:
            logger.warning("[ONYX] reset_state failed: %s", exc)
        free_vram()

    trimaps = trimap_resaware(coarse, band_scale)
    mask_combined = combined_coarse_mask(coarse, scene.objects)
    if external_trimap is not None:
        trimap_combined = to_mask(external_trimap).float().clamp(0, 1)
        if trimap_combined.shape[0] == 1 and B > 1:
            trimap_combined = trimap_combined.expand(B, -1, -1)
    else:
        trimap_combined = trimap_resaware(
            mask_combined.unsqueeze(0), band_scale,
        )[0]

    mat_key = (matter_key or "none").split("  [")[0].strip()
    if matter_instance is not None:
        matter = matter_instance
    elif mat_key not in ("none", ""):
        if matter_cls is None:
            raise RuntimeError(f"Unknown matter backend '{matter_key}'.")
        from .node import _resolve_model_choice as _rmc
        matter = matter_cls(
            model_name=_rmc(matter_model or "(auto)", matter_cls.MODELS_KEY,
                            auto_download=bool(auto_download)),
            device=device,
            precision=precision,
            attention=attention,
            offload=offload,
        )
        matter.load()
    else:
        matter = None

    if matter is not None:
        alpha_o = refine_object_tiled(
            matter, img_bhwc, trimaps, coarse, matte_tile=int(matte_tile),
        )
    else:
        alpha_o = coarse.clone()
        if post_refine in ("guided", "crf+guided"):
            from . import _vfx
            for o in range(alpha_o.shape[0]):
                for b in range(B):
                    a = alpha_o[o, b].unsqueeze(0)
                    img_b = img_bhwc[b:b + 1]
                    alpha_o[o, b] = _vfx.guided_refine(img_b, a, radius=8, epsilon=1e-4)[0]

    alpha_o = stabilise_alpha_unknown(alpha_o, trimaps, enabled=bool(temporal_stabilise))
    alpha_combined = combine_alphas(alpha_o, scene.objects)
    object_alphas = pack_object_alphas(alpha_o, scene.objects)
    mean_score = float(sum(obj_scores.values()) / max(len(obj_scores), 1))

    objects_meta = [
        {
            "id": o.id,
            "label": o.label,
            "color": o.color,
            "enabled": o.enabled,
            "opacity": o.opacity,
            "score": float(obj_scores.get(o.id, 0.0)),
        }
        for o in scene.objects
    ]
    pipeline_info["objects"] = objects_meta
    pipeline_info["tracking"] = {
        "direction": scene.tracking.direction,
        "start_frame": scene.tracking.start_frame,
        "max_frames": scene.tracking.max_frames,
        "earliest_keyframe": _keyframe_bounds(scene)[0],
        "latest_keyframe": _keyframe_bounds(scene)[1],
    }
    enabled_objs = _enabled_objects(scene.objects)
    pipeline_info["trimap_band_px"] = band_px(H, W, band_scale)
    pipeline_info["trimaps_per_object"] = [
        {"id": o.id, "shape": list(trimaps[i].shape)}
        for i, o in enumerate(enabled_objs)
    ]

    return OnyxResult(
        mask_coarse=mask_combined,
        alpha_combined=alpha_combined,
        object_alphas=object_alphas,
        object_coarse=coarse,
        trimap_combined=trimap_combined,
        trimaps_per_object=trimaps,
        segmenter_score=mean_score,
        objects_meta=objects_meta,
        pipeline_info=pipeline_info,
    )
