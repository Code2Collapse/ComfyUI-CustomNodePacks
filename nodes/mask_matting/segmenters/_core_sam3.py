"""ComfyUI's own SAM 3 / 3.1 for Mask + Matting and its viewer (L7.114 Detect, L7.125 tracking).

Core 0.39+ ships the model (comfy/ldm/sam3), its text encoder (comfy/text_encoders/sam3_clip) and the SAM3 nodes
(comfy_extras/nodes_sam3.py). Everything here CALLS core at run time; none of core's code is copied (core is GPL-3.0,
this pack Apache-2.0). Design and the measurements behind the choices: docs/design/core_sam3_tracking.md.

- CoreSAM3: one loaded checkpoint (MODEL + CLIP from one Comfy-Org file):
  - detect(): words on one frame, every match;
  - segment_union(): the cascade's per-frame contract;
  - mask_from_prompts(): a seed mask from clicks / a box;
  - track(): forward video tracking from first-frame masks.
- CoreVideoTracker: the tracker protocol ONYX speaks (init_state / add_prompt / add_mask / propagate_in_video /
  reset_state), on top of track().
- get() / release(): one checkpoint in memory at a time, shared by the viewer and the matte.

Kill switch: C2C_SAM_CORE=0 keeps Mask + Matting off this path (available(respect_kill=True)); the viewer's Detect
does not read it (L7.114 asked for core SAM 3 there).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger("C2C.CoreSAM3")

SIDE = 1008                     # SAM 3's input size (core resizes every frame to it)
THRESHOLD = 0.5                 # core SAM3_Detect's default
REFINE = 2                      # core SAM3_Detect's default refine_iterations
MAX_OBJECTS = 32
RECOMMENDED = "sam3.1_multiplex_fp16.safetensors"
RECOMMENDED_URL = "https://huggingface.co/Comfy-Org/sam3.1"
KILL_ENV = "C2C_SAM_CORE"


def available(respect_kill: bool = True) -> bool:
    """ComfyUI ships SAM 3 (core 0.39+), and (for Mask + Matting) C2C_SAM_CORE is not 0."""
    if respect_kill and os.environ.get(KILL_ENV, "").strip() == "0":
        return False
    try:
        import comfy.ldm.sam3.detector  # noqa: F401
        import comfy.text_encoders.sam3_clip  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _hint() -> str:
    return f"Use the Comfy-Org file {RECOMMENDED} ({RECOMMENDED_URL})."


class TrackResult:
    """What forward_video returned, read one frame at a time (masks stay bit-packed / low-res until asked for)."""

    def __init__(self, res, n_obj: int, size: Tuple[int, int]):
        self.res, self.n_obj, self.size = res, int(n_obj), size
        self.scores: List[float] = [1.0] * self.n_obj
        if isinstance(res, dict):
            sc = list(res.get("scores") or [])
            self.scores = [float(s) for s in sc[:self.n_obj]] + [1.0] * max(0, self.n_obj - len(sc))

    def frame(self, i: int) -> torch.Tensor:
        """[n_obj, H, W] float 0/1 on the CPU for the i-th tracked frame."""
        import torch.nn.functional as F
        H, W = self.size
        res = self.res
        if isinstance(res, dict):
            packed = res.get("packed_masks")
            if packed is None:
                return torch.zeros(self.n_obj, H, W)
            from comfy.ldm.sam3.tracker import unpack_masks
            m = unpack_masks(packed[i])[:self.n_obj].float()
        else:                                            # SAM 3 (not multiplex): logits [N, O, h, w]
            m = (res[i, :self.n_obj] > 0).float()
        if m.shape[0] < self.n_obj:
            m = torch.cat([m, torch.zeros(self.n_obj - m.shape[0], *m.shape[1:], device=m.device)], 0)
        m = m.detach().to("cpu", torch.float32)
        if tuple(m.shape[-2:]) != (H, W):
            m = (F.interpolate(m[None], size=(H, W), mode="bilinear", align_corners=False)[0] > 0.5).float()
        return m


class CoreSAM3:
    """One SAM 3 / 3.1 checkpoint loaded through core.

    Words: the detector head finds every match above THRESHOLD (strongest first, at most MAX_OBJECTS), each mask
    refined by core's SAM decoder pass on the object's box (core's SAM3_Detect does the same, refine_iterations=2).
    Clicks and boxes: the SAM decoder, one object per box (the clicks go with every box), else one object from the
    clicks; REFINE passes feed the mask back with the same prompts. A decoder mask has no detector score, so its
    score is the mean probability inside the mask."""

    SIDE = SIDE
    THRESHOLD = THRESHOLD
    REFINE = REFINE

    def __init__(self, tag: str, path: str):
        import comfy.sd as comfy_sd
        import folder_paths
        try:
            model, clip, _vae, _cv = comfy_sd.load_checkpoint_guess_config(
                path, output_vae=False, output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"))
        except Exception as exc:  # noqa: BLE001 - say which file, and which file works
            raise RuntimeError(f"ComfyUI could not read {tag} as a SAM 3 checkpoint ({exc}). {_hint()}") from exc
        net = getattr(getattr(model, "model", None), "diffusion_model", None)
        if net is None or not hasattr(net, "forward_segment") or clip is None:
            raise RuntimeError(f"{tag} is not a SAM 3 checkpoint (ComfyUI read it as another kind of model). "
                               f"{_hint()}")
        self.model, self.clip = model, clip
        self.path = path
        self._words: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}

    # ── helpers ──
    def _encode(self, text: str):
        if text not in self._words:
            out = self.clip.encode_from_tokens(self.clip.tokenize(text), return_dict=True)
            self._words[text] = (out["cond"], out.get("attention_mask"))
        return self._words[text]

    def _prepare(self, frame_hwc: np.ndarray):
        import comfy.model_management as mm
        import comfy.utils as comfy_utils
        img = torch.from_numpy(np.ascontiguousarray(frame_hwc[..., :3], dtype=np.float32))[None]   # [1, H, W, 3]
        mm.load_model_gpu(self.model)
        device, dtype = mm.get_torch_device(), self.model.model.get_dtype()
        frame = comfy_utils.common_upscale(img.movedim(-1, 1), self.SIDE, self.SIDE, "bilinear",
                                           crop="disabled").to(device=device, dtype=dtype)
        return self.model.model.diffusion_model, frame, img[0], device, dtype

    # ── one frame ──
    def detect(self, frame_hwc: np.ndarray, text: str, pos, neg, boxes) -> List[Dict[str, Any]]:
        """Every object on one frame: words through the detector, else clicks / boxes through the SAM decoder.
        [{"mask": HxW float, "box": [x0, y0, x1, y1] px or None, "score": float}]."""
        H, W = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
        words = self._encode(text) if text else None                 # the text encoder first: it is a separate model
        net, frame, image_hwc, device, dtype = self._prepare(frame_hwc)
        if words is not None:
            return self._detect_words(net, frame, image_hwc, words, H, W, device, dtype)
        return self._segment_prompts(net, frame, pos, neg, boxes, H, W, device, dtype)

    def _detect_words(self, net, frame, image_hwc, words, H, W, device, dtype) -> List[Dict[str, Any]]:
        emb, att = words
        emb = emb.to(device=device, dtype=dtype)
        att = att.to(device) if att is not None else torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64,
                                                                   device=device)
        res = net(frame, text_embeddings=emb, text_mask=att, boxes=None, threshold=self.THRESHOLD, orig_size=(H, W))
        probs = res["scores"][0].float().sigmoid()
        keep = probs > self.THRESHOLD
        kept_boxes, kept_scores, kept_masks = res["boxes"][0][keep].float().cpu(), probs[keep].cpu(), res["masks"][0][keep]
        try:
            from comfy_extras.nodes_sam3 import _refine_mask       # core's own refinement (run time, not copied)
        except Exception:  # noqa: BLE001 - a core without it: the detector's own mask
            _refine_mask = None
        out = []
        for i in kept_scores.argsort(descending=True)[:MAX_OBJECTS].tolist():
            coarse = kept_masks[i]
            if _refine_mask is not None:
                m = _refine_mask(net, image_hwc, coarse, kept_boxes[i], H, W, device, dtype, self.REFINE)[0]
            else:
                m = (coarse > 0).float()
            box = kept_boxes[i].tolist()
            out.append({"mask": m.float().cpu().numpy(), "score": float(kept_scores[i]),
                        "box": [max(0, int(box[0])), max(0, int(box[1])), min(W, int(round(box[2]))),
                                min(H, int(round(box[3])))]})
        return out

    def _segment_prompts(self, net, frame, pos, neg, boxes, H, W, device, dtype) -> List[Dict[str, Any]]:
        import torch.nn.functional as F
        s = self.SIDE
        points = None
        if pos or neg:
            coords = [[x / W * s, y / H * s] for x, y in list(pos) + list(neg)]
            points = {"point_coords": torch.tensor([coords], dtype=dtype, device=device),
                      "point_labels": torch.tensor([[1] * len(pos) + [0] * len(neg)], dtype=torch.int32,
                                                   device=device)}
        corners = [torch.tensor([[[x0 / W * s, y0 / H * s], [x1 / W * s, y1 / H * s]]], dtype=dtype, device=device)
                   for x0, y0, x1, y1 in boxes] or [None]
        out = []
        for box in corners:
            logit = net.forward_segment(frame, point_inputs=points, box_inputs=box)
            for _ in range(max(0, self.REFINE - 1)):
                logit = net.forward_segment(frame, point_inputs=points, box_inputs=box, mask_inputs=logit)
            logit = F.interpolate(logit.float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            m = (logit > 0).float()
            if m.max() <= 0:
                continue
            out.append({"mask": m.cpu().numpy(), "box": None,
                        "score": float(torch.sigmoid(logit)[m > 0].mean())})
        return out

    def mask_from_prompts(self, frame_hwc: np.ndarray, pos, neg, box) -> Tuple[np.ndarray, float]:
        """One seed mask from clicks and / or one box (the SAM decoder). Empty when SAM finds nothing."""
        H, W = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
        objs = self.detect(frame_hwc, "", list(pos or []), list(neg or []), [tuple(box)] if box is not None else [])
        if not objs:
            return np.zeros((H, W), np.float32), 0.0
        return np.asarray(objs[0]["mask"], np.float32), float(objs[0]["score"])

    def segment_union(self, frame_hwc: np.ndarray, pos, neg, bbox, neg_bbox, text) -> Tuple[np.ndarray, float]:
        """The cascade segmenter's per-frame contract (sam31_backend._seg_one): one HxW mask + a score.
        Words: every match; clicks / a box, when given with words, choose among the matches (a match holding a
        positive click or centred in the box; a match holding a negative click is dropped), all matches when none is
        chosen. No words: the SAM decoder on the clicks and the box. neg_bbox has no counterpart in core's decoder:
        with words it drops the matches centred in it, without words its own mask is subtracted (as the sam3 package
        backend does)."""
        H, W = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
        pos, neg = list(pos or []), list(neg or [])
        if text and str(text).strip():
            objs = self.detect(frame_hwc, str(text).strip(), [], [], [])

            def holds(o, pt):
                x, y = int(min(W - 1, max(0, pt[0]))), int(min(H - 1, max(0, pt[1])))
                return np.asarray(o["mask"])[y, x] > 0.5

            def centred(o, b):
                bx = o.get("box")
                if not bx or b is None:
                    return False
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]

            objs = [o for o in objs if not any(holds(o, p) for p in neg) and not centred(o, neg_bbox)]
            chosen = [o for o in objs if any(holds(o, p) for p in pos) or centred(o, bbox)] if (pos or bbox) else []
            objs = chosen or objs
        else:
            objs = self.detect(frame_hwc, "", pos, neg, [tuple(bbox)] if bbox is not None else [])
        if not objs:
            return np.zeros((H, W), np.float32), 0.0
        m = np.max(np.stack([np.asarray(o["mask"], np.float32) for o in objs]), axis=0)
        if neg_bbox is not None and not (text and str(text).strip()):
            neg_m, _ = self.mask_from_prompts(frame_hwc, [], [], neg_bbox)
            m = np.clip(m - neg_m, 0.0, 1.0)
        return m, max(float(o.get("score", 0.0)) for o in objs)

    # ── video ──
    def track(self, frames_bhwc: torch.Tensor, init_masks: torch.Tensor) -> TrackResult:
        """Track objects forward through ``frames_bhwc`` [N, H, W, 3] from their masks on its FIRST frame
        (``init_masks`` [O, H, W]). Core keeps the frames on the CPU and moves one at a time."""
        import comfy.model_management as mm
        import comfy.utils as comfy_utils
        N, H, W = int(frames_bhwc.shape[0]), int(frames_bhwc.shape[1]), int(frames_bhwc.shape[2])
        O = int(init_masks.shape[0])
        mm.load_model_gpu(self.model)
        device, dtype = mm.get_torch_device(), self.model.model.get_dtype()
        net = self.model.model.diffusion_model
        frames_in = frames_bhwc[..., :3].detach().to("cpu", torch.float32).movedim(-1, 1).contiguous()
        init = (init_masks > 0.5).float().unsqueeze(1).to(device=device, dtype=dtype)
        res = net.forward_video(images=frames_in, initial_masks=init, pbar=comfy_utils.ProgressBar(N),
                                text_prompts=None, new_det_thresh=self.THRESHOLD, max_objects=O, detect_interval=1,
                                target_device=device, target_dtype=dtype)
        return TrackResult(res, O, (H, W))

    def unload(self):
        self.model = self.clip = None                                 # core's model management holds weak refs only
        self._words.clear()
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ── one checkpoint in memory, shared by the viewer and the matte ──────────────────────────────────────────────────

_lock = threading.Lock()
_current: Dict[str, Any] = {}


def release() -> None:
    with _lock:
        h = _current.pop("handle", None)
        _current.clear()
    if h is not None:
        try:
            h.unload()
        except Exception:  # noqa: BLE001
            pass


def get(tag: str, path: str) -> CoreSAM3:
    """The loaded checkpoint at ``path`` (loading it, and releasing another one, when needed)."""
    real = os.path.normcase(os.path.realpath(path))
    with _lock:
        h = _current.get("handle")
        if h is not None and _current.get("path") == real:
            return h
    release()
    h = CoreSAM3(tag, path)
    with _lock:
        _current.update(handle=h, path=real)
    try:
        from ..._c2c_memguard import register
        register("c2c_core_sam3", release, lambda: bool(_current))
    except Exception:  # noqa: BLE001
        pass
    return h


# ── ONYX's tracker protocol ──────────────────────────────────────────────────────────────────────────────────────

class CoreVideoTracker:
    """sam3_lib's video-predictor protocol (as ONYX uses it) on core SAM 3.

    Each object's keyframes seed it: points / a box through the SAM decoder, or a mask given with add_mask. An
    object is tracked from each of its keyframes up to its next keyframe (or the range end); objects that share a
    segment are tracked together (SAM 3.1 tracks up to 16 at once). Backward = the same on the reversed frames."""

    def __init__(self, core: CoreSAM3):
        self.core = core
        self.model = self                                  # ONYX checks seg._video_predictor.model

    # protocol
    def init_state(self, resource_path=None, offload_video_to_cpu: bool = False, frames: Optional[torch.Tensor] = None):
        if frames is None:
            frames = torch.from_numpy(np.stack([np.asarray(p.convert("RGB"), np.float32) / 255.0
                                                for p in resource_path]))
        return {"frames": frames.detach().to("cpu", torch.float32), "kf": {}, "seeds": {}, "calls": []}

    def add_prompt(self, state, frame_idx, points=None, point_labels=None, obj_id=0, rel_coordinates=True, **_kw):
        state["kf"].setdefault(int(obj_id), {})[int(frame_idx)] = (
            "prompt", [list(map(float, p)) for p in (points or [])], [int(v) for v in (point_labels or [])],
            bool(rel_coordinates))

    def add_mask(self, state, frame_idx, mask, obj_id=0):
        state["kf"].setdefault(int(obj_id), {})[int(frame_idx)] = ("mask", np.asarray(mask, np.float32))

    def reset_state(self, state):
        state.clear()

    # seeds
    def _seed(self, state, obj_id: int, f: int) -> np.ndarray:
        key = (obj_id, f)
        if key in state["seeds"]:
            return state["seeds"][key]
        frames = state["frames"]
        H, W = int(frames.shape[1]), int(frames.shape[2])
        entry = state["kf"][obj_id][f]
        if entry[0] == "mask":
            m = entry[1]
            if m.shape != (H, W):
                import torch.nn.functional as F
                m = F.interpolate(torch.from_numpy(m)[None, None], size=(H, W), mode="bilinear",
                                  align_corners=False)[0, 0].numpy()
            m = (m > 0.5).astype(np.float32)
        else:
            _, pts, lbls, rel = entry
            sx, sy = (W, H) if rel else (1.0, 1.0)
            px = [(p[0] * sx, p[1] * sy) for p in pts]
            pos = [p for p, l in zip(px, lbls) if l == 1]
            neg = [p for p, l in zip(px, lbls) if l == 0]
            corners = [p for p, l in zip(px, lbls) if l in (2, 3)]
            box = None
            if len(corners) >= 2:
                xs, ys = [c[0] for c in corners], [c[1] for c in corners]
                box = (min(xs), min(ys), max(xs), max(ys))
            m, _s = self.core.mask_from_prompts(frames[f].numpy(), pos, neg, box)
        state["seeds"][key] = m
        return m

    def propagate_in_video(self, state, start_frame_idx: int = 0, max_frame_num_to_track: Optional[int] = None,
                           reverse: bool = False, **_kw) -> Iterator[Tuple[int, Dict[str, Any]]]:
        frames = state["frames"]
        B = int(frames.shape[0])
        start = max(0, min(B - 1, int(start_frame_idx)))
        n = int(max_frame_num_to_track) if max_frame_num_to_track else B
        order = list(range(start, max(-1, start - n), -1)) if reverse else list(range(start, min(B, start + n)))
        pos_of = {f: i for i, f in enumerate(order)}
        # segments: (pos_start, pos_end) -> [obj ids]
        groups: Dict[Tuple[int, int], List[int]] = {}
        for oid, kfs in state["kf"].items():
            ps = sorted(pos_of[f] for f in kfs if f in pos_of)
            for i, p0 in enumerate(ps):
                p1 = ps[i + 1] if i + 1 < len(ps) else len(order)
                groups.setdefault((p0, p1), []).append(oid)
        runs = []                                         # (p0, p1, [oids], TrackResult)
        for (p0, p1), oids in sorted(groups.items()):
            idx = order[p0:p1]
            seeds = torch.from_numpy(np.stack([self._seed(state, oid, order[p0]) for oid in oids]))
            sub = frames[idx]                             # the frames in tracking order (reversed for backward)
            res = self.core.track(sub, seeds)
            runs.append((p0, p1, oids, res))
            state["calls"].append({"frames": [idx[0], idx[-1]], "objects": list(oids), "reverse": bool(reverse)})
        for p, f in enumerate(order):
            ids, masks, probs = [], [], []
            for p0, p1, oids, res in runs:
                if p0 <= p < p1:
                    fm = res.frame(p - p0)
                    for j, oid in enumerate(oids):
                        ids.append(oid)
                        masks.append(fm[j].numpy())
                        probs.append(res.scores[j] if j < len(res.scores) else 1.0)
            if not ids:
                continue
            yield f, {"out_obj_ids": ids, "out_binary_masks": np.stack(masks), "out_probs": probs}
