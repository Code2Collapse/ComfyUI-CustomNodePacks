"""
VideoMaskEditorMEC
==================
Full-stack interactive video mask editor.

The actual painting is performed in the browser (see js/video_mask_editor.js).
This node:
  1. Hosts a server-side per-session store of pinned-keyframe masks
     (each pinned keyframe is a single grayscale PNG kept in RAM).
  2. At execute() time, reconstructs the per-frame mask batch:
       - frames that the user pinned use the pinned mask directly,
       - non-pinned frames are tweened from the surrounding pinned
         keyframes using one of: distance-transform interpolation
         (shape-coherent), linear alpha blend, or hold-nearest.
  3. Optionally feathers + thresholds the final masks.

INPUTS (required):
  image          IMAGE   (B,H,W,3) — the video batch; needed to know B/H/W
  session_id     STRING  unique UUID; set by the JS extension when the
                          node is first created. Used to key the server-
                          side session dict so multiple instances don't
                          collide.
  tween_mode     COMBO   {"distance_transform","linear","hold"}
  feather        FLOAT   gaussian sigma in px applied to the final masks
  threshold      FLOAT   binarize after tween (0 = no binarize)

INPUTS (optional):
  input_mask     MASK    fallback mask (used for any frame that has no
                          pinned/tween value because the user hasn't
                          set any keyframes). Shape (B|1,H,W).

OUTPUTS:
  mask           MASK    (B,H,W) float32 in [0,1]
  info           STRING  JSON: {session, n_keyframes, frames, source}

VRAM tier: 0 (CPU-only).
"""

from __future__ import annotations

import io
import json
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False

try:
    from PIL import Image  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "VideoMaskEditorMEC requires Pillow. Install with: pip install pillow"
    ) from e


# ─── In-memory session store ─────────────────────────────────────────
# {session_id: {"keyframes": {frame_int: np.uint8 mask (H,W)},
#               "shape": (H,W), "ts": float, "lock": threading.Lock}}
_SESSIONS: Dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()
_SESSION_TTL_SEC = 60 * 60 * 6  # 6h
_SESSION_MAX = 64


def _gc_sessions() -> None:
    """Drop sessions older than TTL or trim to MAX (oldest first)."""
    now = time.time()
    with _SESSIONS_LOCK:
        # TTL.
        stale = [sid for sid, s in _SESSIONS.items()
                 if now - s.get("ts", now) > _SESSION_TTL_SEC]
        for sid in stale:
            _SESSIONS.pop(sid, None)
        # Cap.
        if len(_SESSIONS) > _SESSION_MAX:
            ordered = sorted(_SESSIONS.items(), key=lambda kv: kv[1].get("ts", 0))
            for sid, _ in ordered[:len(_SESSIONS) - _SESSION_MAX]:
                _SESSIONS.pop(sid, None)


def _get_session(session_id: str, create: bool = False) -> Optional[dict]:
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(session_id)
        if s is None and create:
            s = {
                "keyframes": {},
                "shape": None,
                "ts": time.time(),
                "lock": threading.Lock(),
            }
            _SESSIONS[session_id] = s
        if s is not None:
            s["ts"] = time.time()
        return s


# ─── PNG codec helpers ───────────────────────────────────────────────
def _png_to_mask(png_bytes: bytes) -> np.ndarray:
    """Decode a grayscale PNG into a uint8 (H,W) array."""
    im = Image.open(io.BytesIO(png_bytes))
    if im.mode != "L":
        im = im.convert("L")
    return np.array(im, dtype=np.uint8)


def _mask_to_png(arr: np.ndarray) -> bytes:
    """Encode a uint8 (H,W) mask to PNG bytes."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


# ─── Tween implementations ───────────────────────────────────────────
def _resize_to(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a uint8 mask to (H,W). Uses cv2 if available, else PIL."""
    th, tw = target_hw
    if mask.shape == (th, tw):
        return mask
    if _HAS_CV2:
        return cv2.resize(mask, (tw, th), interpolation=cv2.INTER_LINEAR)
    im = Image.fromarray(mask, mode="L").resize((tw, th), Image.BILINEAR)
    return np.array(im, dtype=np.uint8)


def _signed_dt(mask: np.ndarray) -> np.ndarray:
    """Signed distance transform: positive inside, negative outside, in px.
    Robust to all-zero / all-one masks.
    """
    if not _HAS_CV2:
        # Fallback (slow but correct): just use the mask directly.
        m = (mask > 127).astype(np.float32)
        return (m - 0.5) * 100.0
    binm = (mask > 127).astype(np.uint8)
    if binm.max() == 0:
        # All background — uniform large negative.
        return np.full(mask.shape, -1e3, dtype=np.float32)
    if binm.min() == 1:
        # All foreground — uniform large positive.
        return np.full(mask.shape, 1e3, dtype=np.float32)
    inside = cv2.distanceTransform(binm, cv2.DIST_L2, 3).astype(np.float32)
    outside = cv2.distanceTransform(1 - binm, cv2.DIST_L2, 3).astype(np.float32)
    return inside - outside


def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (cx, cy) of a binary mask, or None if empty."""
    binm = mask > 127
    if not binm.any():
        return None
    ys, xs = np.where(binm)
    return float(xs.mean()), float(ys.mean())


def _translate(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a uint8 mask. cv2 if available, else numpy roll fallback."""
    if dx == 0 and dy == 0:
        return mask
    if _HAS_CV2:
        H, W = mask.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            mask, M, (W, H), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    # Integer roll fallback.
    out = np.zeros_like(mask)
    idx = int(round(dx))
    idy = int(round(dy))
    H, W = mask.shape[:2]
    src_x0 = max(0, -idx)
    src_y0 = max(0, -idy)
    dst_x0 = max(0, idx)
    dst_y0 = max(0, idy)
    w = max(0, W - abs(idx))
    h = max(0, H - abs(idy))
    if w > 0 and h > 0:
        out[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = \
            mask[src_y0:src_y0 + h, src_x0:src_x0 + w]
    return out


def _tween_dt(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Distance-transform tween with centroid pre-alignment.

    Standard signed-DT lerp fails when the two keyframe shapes don't
    overlap (the lerped SDT goes negative everywhere between them and
    you get an empty middle frame). Fix: translate both keyframes
    toward the interpolated centroid first, lerp THOSE DTs, then we
    get a proper "moving shape" interpolation.
    """
    ca = _centroid(a)
    cb = _centroid(b)
    if ca is not None and cb is not None:
        # Target centroid = lerp.
        tcx = ca[0] * (1.0 - t) + cb[0] * t
        tcy = ca[1] * (1.0 - t) + cb[1] * t
        a_aligned = _translate(a, tcx - ca[0], tcy - ca[1])
        b_aligned = _translate(b, tcx - cb[0], tcy - cb[1])
    else:
        a_aligned = a
        b_aligned = b
    da = _signed_dt(a_aligned)
    db = _signed_dt(b_aligned)
    d = da * (1.0 - t) + db * t
    # 1px transition band for sub-pixel edge.
    soft = np.clip(d + 0.5, 0.0, 1.0)
    return (soft * 255.0).astype(np.uint8)


def _tween_linear(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t).astype(np.uint8)


def _tween_hold(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a if t < 0.5 else b


_TWEEN_FUNCS = {
    "distance_transform": _tween_dt,
    "linear": _tween_linear,
    "hold": _tween_hold,
}


# ─── aiohttp route registration ──────────────────────────────────────
def register_routes(server) -> None:
    """Bind /mec/video_mask_editor/* routes."""
    try:
        from aiohttp import web
    except Exception:
        print("[MEC.VideoMaskEditor] aiohttp not available; routes disabled.")
        return

    routes = server.routes

    @routes.post("/mec/video_mask_editor/init")
    async def init_session(request):
        sid = request.rel_url.query.get("session", "").strip()
        if not sid:
            return web.json_response({"error": "session required"}, status=400)
        try:
            body = await request.json()
            h = int(body.get("h"))
            w = int(body.get("w"))
        except Exception:
            return web.json_response({"error": "invalid body"}, status=400)
        if h <= 0 or w <= 0 or h > 8192 or w > 8192:
            return web.json_response({"error": "invalid shape"}, status=400)
        s = _get_session(sid, create=True)
        with s["lock"]:
            if s["shape"] != (h, w):
                # Shape change: drop any existing keyframes.
                s["keyframes"].clear()
                s["shape"] = (h, w)
        _gc_sessions()
        return web.json_response({"ok": True, "shape": [h, w]})

    @routes.post("/mec/video_mask_editor/keyframe")
    async def set_keyframe(request):
        sid = request.rel_url.query.get("session", "").strip()
        frame_s = request.rel_url.query.get("frame", "")
        if not sid or not frame_s.isdigit():
            return web.json_response({"error": "session+frame required"}, status=400)
        frame = int(frame_s)
        if frame < 0 or frame > 100000:
            return web.json_response({"error": "frame out of range"}, status=400)
        data = await request.read()
        if not data or len(data) > 32 * 1024 * 1024:
            return web.json_response({"error": "empty or too-large payload"}, status=400)
        try:
            arr = _png_to_mask(data)
        except Exception as e:
            return web.json_response({"error": f"png decode: {e}"}, status=400)
        s = _get_session(sid, create=True)
        with s["lock"]:
            if s["shape"] is None:
                s["shape"] = arr.shape
            elif s["shape"] != arr.shape:
                arr = _resize_to(arr, s["shape"])
            s["keyframes"][frame] = arr
        return web.json_response({
            "ok": True, "frame": frame, "n_keyframes": len(s["keyframes"]),
        })

    @routes.delete("/mec/video_mask_editor/keyframe")
    async def del_keyframe(request):
        sid = request.rel_url.query.get("session", "").strip()
        frame_s = request.rel_url.query.get("frame", "")
        if not sid or not frame_s.isdigit():
            return web.json_response({"error": "session+frame required"}, status=400)
        s = _get_session(sid)
        if s is None:
            return web.json_response({"ok": True})
        with s["lock"]:
            s["keyframes"].pop(int(frame_s), None)
            n = len(s["keyframes"])
        return web.json_response({"ok": True, "n_keyframes": n})

    @routes.get("/mec/video_mask_editor/state")
    async def get_state(request):
        sid = request.rel_url.query.get("session", "").strip()
        if not sid:
            return web.json_response({"error": "session required"}, status=400)
        s = _get_session(sid)
        if s is None:
            return web.json_response({"keyframes": [], "shape": None})
        with s["lock"]:
            return web.json_response({
                "keyframes": sorted(s["keyframes"].keys()),
                "shape": list(s["shape"]) if s["shape"] else None,
                "n_keyframes": len(s["keyframes"]),
            })

    @routes.get("/mec/video_mask_editor/keyframe")
    async def get_keyframe(request):
        sid = request.rel_url.query.get("session", "").strip()
        frame_s = request.rel_url.query.get("frame", "")
        if not sid or not frame_s.isdigit():
            return web.json_response({"error": "session+frame required"}, status=400)
        s = _get_session(sid)
        if s is None:
            return web.Response(status=404)
        with s["lock"]:
            arr = s["keyframes"].get(int(frame_s))
        if arr is None:
            return web.Response(status=404)
        return web.Response(body=_mask_to_png(arr), content_type="image/png")

    @routes.post("/mec/video_mask_editor/clear")
    async def clear_session(request):
        sid = request.rel_url.query.get("session", "").strip()
        if not sid:
            return web.json_response({"error": "session required"}, status=400)
        s = _get_session(sid)
        if s is not None:
            with s["lock"]:
                s["keyframes"].clear()
        return web.json_response({"ok": True})

    print("[MEC.VideoMaskEditor] routes registered (/mec/video_mask_editor/*)")


# ─── Mask reconstruction (used by execute) ───────────────────────────
def _reconstruct_batch(
    n_frames: int,
    target_hw: Tuple[int, int],
    keyframes: Dict[int, np.ndarray],
    tween_mode: str,
    fallback: Optional[np.ndarray],
) -> Tuple[np.ndarray, str]:
    """Build (n_frames,H,W) uint8 mask batch.
    Returns (batch, source_tag).
    """
    H, W = target_hw
    out = np.zeros((n_frames, H, W), dtype=np.uint8)
    tween = _TWEEN_FUNCS.get(tween_mode, _tween_dt)

    if not keyframes:
        if fallback is not None:
            fb = _resize_to(fallback, target_hw)
            for i in range(n_frames):
                out[i] = fb
            return out, "fallback_only"
        return out, "empty"

    # Ensure all keyframes match target shape.
    kfs: Dict[int, np.ndarray] = {}
    for f, m in keyframes.items():
        kfs[int(f)] = _resize_to(m, target_hw)
    sorted_f = sorted(kfs.keys())
    first_f, last_f = sorted_f[0], sorted_f[-1]

    for i in range(n_frames):
        if i in kfs:
            out[i] = kfs[i]
            continue
        if i < first_f:
            out[i] = kfs[first_f]
            continue
        if i > last_f:
            out[i] = kfs[last_f]
            continue
        # Find surrounding keyframes.
        lo = first_f
        hi = last_f
        for f in sorted_f:
            if f <= i:
                lo = f
            if f >= i:
                hi = f
                break
        if lo == hi:
            out[i] = kfs[lo]
            continue
        t = (i - lo) / float(hi - lo)
        out[i] = tween(kfs[lo], kfs[hi], float(t))
    return out, f"keyframes={len(keyframes)},tween={tween_mode}"


def _gaussian_feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to a uint8 (...,H,W) mask. Returns float32 in [0,1]."""
    if sigma <= 0:
        return mask.astype(np.float32) / 255.0
    if not _HAS_CV2:
        return mask.astype(np.float32) / 255.0
    k = int(2 * round(sigma * 2.0) + 1)
    k = max(3, k | 1)
    if mask.ndim == 3:
        out = np.empty(mask.shape, dtype=np.float32)
        for i in range(mask.shape[0]):
            out[i] = cv2.GaussianBlur(mask[i], (k, k), sigma).astype(np.float32) / 255.0
        return out
    return cv2.GaussianBlur(mask, (k, k), sigma).astype(np.float32) / 255.0


# ─── ComfyUI Node ────────────────────────────────────────────────────
class VideoMaskEditorMEC:
    """Pinpoint per-frame mask editing across an entire video batch."""

    VRAM_TIER = 0
    CATEGORY = "C2C/VideoMask"
    DESCRIPTION = (
        "Open the in-browser video mask editor (brush / erase / fill / "
        "lasso / onion-skin) to pin per-frame keyframes, then tweens "
        "non-keyframed frames using distance-transform interpolation."
    )
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input video batch (B,H,W,3). Drives B/H/W.",
                }),
                "session_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Auto-set by the editor UI. Don't edit by hand.",
                }),
                "tween_mode": (
                    ["distance_transform", "linear", "hold"],
                    {"default": "distance_transform",
                     "tooltip": "How to interpolate non-keyframed frames."},
                ),
                "feather": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 32.0, "step": 0.1,
                    "tooltip": "Gaussian feather radius (px). 0 = none.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Binarize after tween. 0 = keep soft mask.",
                }),
            },
            "optional": {
                "input_mask": ("MASK", {
                    "tooltip": "Fallback mask used if no keyframes are set.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        session_id: str,
        tween_mode: str,
        feather: float,
        threshold: float,
        input_mask: Optional[torch.Tensor] = None,
    ):
        if image is None or image.ndim != 4:
            raise ValueError("image must be IMAGE tensor (B,H,W,C)")
        B, H, W, _C = image.shape
        sid = (session_id or "").strip()
        if not sid:
            raise ValueError(
                "VideoMaskEditorMEC: session_id is empty. Open the node's "
                "in-browser editor at least once so it can assign a UUID."
            )

        # Pull keyframes from the session store.
        s = _get_session(sid)
        kfs: Dict[int, np.ndarray] = {}
        if s is not None:
            with s["lock"]:
                kfs = dict(s["keyframes"])

        # Fallback mask: take first slice of input_mask, if any.
        fallback: Optional[np.ndarray] = None
        if input_mask is not None and input_mask.numel() > 0:
            fm = input_mask
            if fm.ndim == 3:
                fm0 = fm[0]
            elif fm.ndim == 2:
                fm0 = fm
            else:
                fm0 = fm.reshape(-1, fm.shape[-2], fm.shape[-1])[0]
            fb = (fm0.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            fallback = fb

        batch_u8, src_tag = _reconstruct_batch(
            n_frames=B,
            target_hw=(H, W),
            keyframes=kfs,
            tween_mode=tween_mode,
            fallback=fallback,
        )

        # Feather + threshold + cast to float.
        mask_f = _gaussian_feather(batch_u8, float(feather))
        if threshold > 0:
            mask_f = (mask_f >= float(threshold)).astype(np.float32)
        mask_t = torch.from_numpy(mask_f.astype(np.float32))

        info = json.dumps({
            "session": sid[:8] + "…" if len(sid) > 8 else sid,
            "n_keyframes": len(kfs),
            "frames": B,
            "shape": [H, W],
            "tween_mode": tween_mode,
            "feather": float(feather),
            "threshold": float(threshold),
            "source": src_tag,
        }, separators=(",", ":"))
        return (mask_t, info)


NODE_CLASS_MAPPINGS = {"VideoMaskEditorMEC": VideoMaskEditorMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoMaskEditorMEC": "Video Mask Editor"}
