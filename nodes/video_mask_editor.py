"""
VideoMaskEditorMEC
==================
Full-stack interactive video mask editor.

The actual painting is performed in the browser (see js/video_mask_editor.js).
This node:
  1. Hosts a server-side per-session store of pinned-keyframe masks
     (each pinned keyframe is a single grayscale PNG kept in RAM).
  2. At execute() time, reconstructs the per-frame mask batch:
       - a painted (pinned) frame uses its own mask,
       - a frame BETWEEN two painted frames follows tween_mode: distance-transform
         shape interpolation, linear blend, hold the previous one, or none,
       - a frame before the first / after the last painted frame follows
         outside_keys: empty (default) or hold the nearest painted frame,
       - every frame no rule fills takes input_mask's own frame (if connected).
     Before 2026-10-23 the first / last keyframe was always held over the whole
     clip, so one painted frame masked every frame (owner report, ledger L9.35).
  3. Optionally feathers + thresholds the final masks.
  4. Keyframes are written to input/c2c_video_masks/<session>/ as they are
     pinned, so a server restart no longer loses them (L9.16).

INPUTS (required):
  image          IMAGE   (B,H,W,3) — the video batch; needed to know B/H/W
  session_id     STRING  unique UUID; set by the JS extension when the
                          node is first created. Used to key the server-
                          side session dict so multiple instances don't
                          collide.
  tween_mode     COMBO   {"distance_transform","linear","hold","none"} (between painted frames)
  feather        FLOAT   gaussian sigma in px applied to the final masks
  threshold      FLOAT   binarize after tween (0 = no binarize)

INPUTS (optional):
  outside_keys   COMBO   {"empty","hold"} before the first / after the last painted frame
  input_mask     MASK    per-frame base mask, used on every frame the painted keyframes
                          do not fill. Shape (B|1,H,W); a shorter batch repeats its last frame.

OUTPUTS:
  mask           MASK    (B,H,W) float32 in [0,1]
  info           STRING  JSON: {session, n_keyframes, frames, source}

VRAM tier: 0 (CPU-only).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ._is_changed_util import hash_args_and_kwargs

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


# ─── Disk persistence (L9.16) ────────────────────────────────────────
# One grayscale PNG per painted frame in input/c2c_video_masks/<session>/ (C2C_VME_DIR overrides the root, the tests
# use it). Written when a keyframe is pinned, read back the first time a session is used after a restart.
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$")


def _disk_root() -> Optional[Path]:
    env = os.environ.get("C2C_VME_DIR", "").strip()
    if env:
        return Path(env)
    try:
        import folder_paths
        return Path(folder_paths.get_input_directory()) / "c2c_video_masks"
    except Exception:  # noqa: BLE001 - no ComfyUI: RAM only
        return None


def _disk_dir(session_id: str) -> Optional[Path]:
    root = _disk_root()
    if root is None or not _SESSION_RE.match(session_id or ""):
        return None
    return root / session_id


def _disk_put(session_id: str, frame: int, arr: np.ndarray) -> None:
    d = _disk_dir(session_id)
    if d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"kf_{int(frame):06d}.png.tmp"
        tmp.write_bytes(_mask_to_png(arr))
        os.replace(tmp, d / f"kf_{int(frame):06d}.png")
    except OSError as exc:
        print(f"[MEC.VideoMaskEditor] could not save keyframe {frame} of {session_id[:8]}: {exc}")


def _disk_delete(session_id: str, frame: Optional[int] = None) -> None:
    d = _disk_dir(session_id)
    if d is None or not d.is_dir():
        return
    for f in ([d / f"kf_{int(frame):06d}.png"] if frame is not None else list(d.glob("kf_*.png"))):
        try:
            f.unlink()
        except OSError:
            pass


def _disk_load(session_id: str) -> Dict[int, np.ndarray]:
    d = _disk_dir(session_id)
    out: Dict[int, np.ndarray] = {}
    if d is None or not d.is_dir():
        return out
    for f in sorted(d.glob("kf_*.png")):
        m = re.match(r"kf_(\d+)\.png$", f.name)
        if not m:
            continue
        try:
            out[int(m.group(1))] = _png_to_mask(f.read_bytes())
        except Exception as exc:  # noqa: BLE001
            print(f"[MEC.VideoMaskEditor] skipped unreadable keyframe {f}: {exc}")
    return out


def _get_session(session_id: str, create: bool = False) -> Optional[dict]:
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(session_id)
        if s is None and not create:
            saved = _disk_load(session_id)
            if saved:
                create = True
        else:
            saved = None
        if s is None and create:
            if saved is None:
                saved = _disk_load(session_id)
            s = {
                "keyframes": dict(saved),
                "shape": next(iter(saved.values())).shape if saved else None,
                "ts": time.time(),
                "lock": threading.Lock(),
            }
            _SESSIONS[session_id] = s
        if s is not None:
            s["ts"] = time.time()
        return s


def session_digest(session_id: str) -> str:
    """Content digest of a session's pinned keyframes.

    The editor's whole state lives in this server-side store, not in any widget, so ComfyUI's
    cache key never saw a keyframe edit and a downstream preview kept the old masks until some
    other widget changed (issue #11.4). IS_CHANGED includes this digest: editing, adding or
    deleting a keyframe re-runs the node; re-pinning an identical mask does not. Per-keyframe
    hashes are cached against the array object, so a queue with unchanged keyframes costs a dict
    walk, not a re-hash."""
    s = _SESSIONS.get(session_id) if session_id else None
    if s is None:
        return "no-session"
    with s["lock"]:
        cache = s.setdefault("kf_digest", {})
        parts = []
        for frame in sorted(s["keyframes"]):
            arr = s["keyframes"][frame]
            hit = cache.get(frame)
            if hit is None or hit[0] is not arr:
                hit = (arr, hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest())
                cache[frame] = hit
            parts.append(f"{frame}:{tuple(arr.shape)}:{hit[1]}")
        for frame in [f for f in cache if f not in s["keyframes"]]:
            cache.pop(frame, None)
    return hashlib.sha1("|".join(parts).encode("ascii")).hexdigest()


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
            # A new frame size keeps the painted keyframes: they are resized to the batch at execute time. (They used
            # to be dropped here, silently, whenever the preview size differed.)
            s["shape"] = (h, w)
        _gc_sessions()
        return web.json_response({"ok": True, "shape": [h, w], "keyframes": sorted(s["keyframes"].keys())})

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
        _disk_put(sid, frame, arr)
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
        _disk_delete(sid, int(frame_s))
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
        _disk_delete(sid)
        return web.json_response({"ok": True})

    @routes.get("/mec/video_mask_editor/effective")
    async def get_effective(request):
        """The mask the node will output for one frame (painted, interpolated, held, input mask or empty), as an
        8-bit PNG at most `max` px on the long side, plus how it was made in the X-C2C-Source header."""
        q = request.rel_url.query
        sid = q.get("session", "").strip()
        try:
            frame = int(q.get("frame", "-1"))
            n = int(q.get("n", "0"))
            max_px = int(q.get("max", "0"))
        except ValueError:
            return web.json_response({"error": "frame, n and max must be integers"}, status=400)
        tween = q.get("tween", "distance_transform")
        outside = q.get("outside", "empty")
        if not sid or frame < 0 or n <= frame:
            return web.json_response({"error": "session, frame and n (> frame) required"}, status=400)
        s = _get_session(sid)
        kfs: Dict[int, np.ndarray] = {}
        base = None
        shape = None
        if s is not None:
            with s["lock"]:
                kfs = dict(s["keyframes"])
                base = s.get("input_mask")
                shape = s.get("shape")
        if shape is None:
            shape = next(iter(kfs.values())).shape if kfs else None
        if shape is None:
            return web.Response(status=204)
        arr, src = _frame_mask(frame, tuple(shape), kfs, sorted(kfs), tween, outside, _base_getter(base, tuple(shape)))
        if arr is None:
            return web.Response(status=204, headers={"X-C2C-Source": src})
        if max_px > 0 and max(arr.shape) > max_px:
            k = max_px / float(max(arr.shape))
            arr = _resize_to(arr, (max(1, round(arr.shape[0] * k)), max(1, round(arr.shape[1] * k))))
        return web.Response(body=_mask_to_png(arr), content_type="image/png", headers={"X-C2C-Source": src})

    print("[MEC.VideoMaskEditor] routes registered (/mec/video_mask_editor/*)")


# ─── Mask reconstruction (used by execute) ───────────────────────────
TWEEN_MODES = ["distance_transform", "linear", "hold", "none"]
OUTSIDE_MODES = ["empty", "hold"]


def _base_getter(input_mask, target_hw: Tuple[int, int]):
    """frame -> uint8 (H, W) of input_mask's own frame (a shorter batch repeats its last frame), or None."""
    if input_mask is None:
        return lambda i: None
    m = input_mask
    if m.ndim == 2:
        m = m[None]
    n = int(m.shape[0])

    def get(i):
        fm = m[min(int(i), n - 1)].detach().float().cpu().clamp(0, 1).numpy()
        return _resize_to((fm * 255.0 + 0.5).astype(np.uint8), target_hw)
    return get


def _frame_mask(i: int, target_hw, kfs: Dict[int, np.ndarray], sorted_f: List[int], tween_mode: str,
                outside: str, base) -> Tuple[Optional[np.ndarray], str]:
    """(uint8 mask or None, how it was made) for frame i: painted / interpolated / held / input / empty."""
    if i in kfs:
        return _resize_to(kfs[i], target_hw), "painted"
    if sorted_f:
        first_f, last_f = sorted_f[0], sorted_f[-1]
        if first_f < i < last_f:
            lo = max(f for f in sorted_f if f < i)
            hi = min(f for f in sorted_f if f > i)
            if tween_mode == "hold":
                return _resize_to(kfs[lo], target_hw), f"held from {lo}"
            if tween_mode != "none":
                tween = _TWEEN_FUNCS.get(tween_mode, _tween_dt)
                t = (i - lo) / float(hi - lo)
                return tween(_resize_to(kfs[lo], target_hw), _resize_to(kfs[hi], target_hw), float(t)), \
                    f"interpolated {lo}-{hi}"
        elif outside == "hold":
            near = first_f if i < first_f else last_f
            return _resize_to(kfs[near], target_hw), f"held from {near}"
    b = base(i)
    if b is not None:
        return b, "input mask"
    return None, "empty"


def _reconstruct_batch(
    n_frames: int,
    target_hw: Tuple[int, int],
    keyframes: Dict[int, np.ndarray],
    tween_mode: str,
    fallback=None,
    outside: str = "empty",
) -> Tuple[np.ndarray, str]:
    """Build the (n_frames, H, W) uint8 mask batch frame by frame. `fallback` is the input MASK tensor (per-frame
    base), a single uint8 array (old callers: one frame for all) or None. Returns (batch, source summary)."""
    H, W = target_hw
    out = np.zeros((n_frames, H, W), dtype=np.uint8)
    if isinstance(fallback, np.ndarray):
        fb = _resize_to(fallback, target_hw)
        base = lambda i: fb  # noqa: E731
    else:
        base = _base_getter(fallback, target_hw)
    kfs = {int(f): m for f, m in keyframes.items()}
    sorted_f = sorted(kfs)
    counts: Dict[str, int] = {}
    for i in range(n_frames):
        arr, src = _frame_mask(i, target_hw, kfs, sorted_f, tween_mode, outside, base)
        kind = src.split(" ")[0]
        counts[kind] = counts.get(kind, 0) + 1
        if arr is not None:
            out[i] = arr
    summary = ",".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return out, f"keyframes={len(kfs)},tween={tween_mode},outside={outside}:{summary}"


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
                    TWEEN_MODES,
                    {"default": "distance_transform",
                     "tooltip": "Frames BETWEEN two painted frames: distance_transform morphs the shape, linear "
                                "blends, hold keeps the previous painted mask, none leaves them unpainted (frame "
                                "by frame)."},
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
                    "tooltip": "Per-frame base mask (e.g. a tracked mask): every frame your keyframes do not "
                               "fill keeps its own frame of this mask.",
                }),
                "outside_keys": (OUTSIDE_MODES, {
                    "default": "empty",
                    "tooltip": "Frames before the first / after the last painted frame: empty (the mask stays on "
                               "the frames you painted) or hold the nearest painted mask.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls, image, session_id, tween_mode, feather, threshold,
                   input_mask=None, outside_keys="empty", **kwargs):
        return hash_args_and_kwargs(
            image, session_id, tween_mode, feather, threshold, input_mask, outside_keys,
            session_digest((session_id or "").strip()),   # keyframe edits (#11.4)
            **kwargs,
        )

    def execute(
        self,
        image: torch.Tensor,
        session_id: str,
        tween_mode: str,
        feather: float,
        threshold: float,
        input_mask: Optional[torch.Tensor] = None,
        outside_keys: str = "empty",
    ):
        if image is None or not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("VideoMaskEditorMEC: image must be IMAGE tensor [B,H,W,C]")
        if input_mask is not None and (
            not isinstance(input_mask, torch.Tensor)
            or input_mask.ndim not in (2, 3)
        ):
            raise ValueError(
                "VideoMaskEditorMEC: input_mask must be MASK tensor [H,W] or [B,H,W]"
            )
        with torch.no_grad():
            return self._execute_impl(
                image, session_id, tween_mode, feather, threshold, input_mask, outside_keys,
            )

    def _execute_impl(
        self,
        image: torch.Tensor,
        session_id: str,
        tween_mode: str,
        feather: float,
        threshold: float,
        input_mask: Optional[torch.Tensor] = None,
        outside_keys: str = "empty",
    ):
        B, H, W, _C = image.shape
        sid = (session_id or "").strip()
        if not sid:
            raise ValueError(
                "VideoMaskEditorMEC: session_id is empty. Open the node's "
                "in-browser editor at least once so it can assign a UUID."
            )

        # Pull keyframes from the session store (RAM, else the saved PNGs).
        s = _get_session(sid, create=True)
        with s["lock"]:
            kfs = dict(s["keyframes"])
            if s["shape"] is None:
                s["shape"] = (H, W)
            # the editor's effective-mask preview shows input_mask frames on unpainted frames
            s["input_mask"] = input_mask if input_mask is not None and input_mask.numel() > 0 else None

        # input_mask is a per-frame base (it used to be collapsed to its frame 0 for every frame).
        base = input_mask if input_mask is not None and input_mask.numel() > 0 else None
        if base is not None and base.ndim > 3:
            base = base.reshape(-1, base.shape[-2], base.shape[-1])

        batch_u8, src_tag = _reconstruct_batch(
            n_frames=B,
            target_hw=(H, W),
            keyframes=kfs,
            tween_mode=tween_mode,
            fallback=base,
            outside=outside_keys if outside_keys in OUTSIDE_MODES else "empty",
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
            "outside_keys": outside_keys,
            "painted_frames": sorted(kfs),
            "feather": float(feather),
            "threshold": float(threshold),
            "source": src_tag,
        }, separators=(",", ":"))
        return (mask_t, info)


# L7.65 P01: no longer registered - this class is the engine of the Image Mask Editor's 'video' mode (its keyframe store and routes stay) (saved workflows
# migrate on load, nodes/_legacy_replacements.json); C2C_LEGACY_MASK_EDITORS=1 registers it again
# (nodes/image_mask_editor/legacy.py).
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
