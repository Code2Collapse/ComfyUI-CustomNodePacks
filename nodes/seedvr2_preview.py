"""Live preview for the numz SeedVR2 upscaler (L7.57) - their code unchanged.

numz/ComfyUI-SeedVR2_VideoUpscaler (Apache-2.0) runs four module-level phases that its node imports by name:
encode -> upscale -> decode -> post-process (src/core/generation_phases.py). Decode writes each batch into
ctx['final_video'] (values in [-1, 1]), records (write_start, write_end, ...) in ctx['decode_batch_info'] and then
calls progress_callback(i, n, 1, "Phase 3: Decoding"). Post-process maps to [0, 1], drops prepended frames and
releases ctx['input_images'].

We replace two attributes of their node module at runtime - decode_all_batches and post-process - with wrappers:
  * decode: after each batch, the batch's last frame and the input frame at that index go to the node's widget
    (event "c2c.seedvr2.preview"), at most every 0.4 s and always for the last batch;
  * post-process: the input frames are read BEFORE theirs runs (it releases them), the final frames after; both are
    kept as a small JPEG strip for scrubbing (event "c2c.seedvr2.done", GET /c2c/seedvr2/frame).
Any error in our code stops previewing for the run and leaves theirs untouched. Kill switch: C2C_SEEDVR2_PREVIEW=0.

Stage 2 (L7.72, live tiles): their VAE class (src/models/video_vae_v3/modules/attn_video_vae.py, VideoAutoencoderKL)
decodes a tiled frame in tiled_decode, which calls self.slicing_decode(tile_latent) once per tile, tile_latent being a
slice VIEW z[:, :, :, y0:y1, x0:x1] of its input. We wrap both class methods: tiled_decode marks "a tiled decode of
z is running" (thread-local), slicing_decode returns their tile untouched and, while that mark is set, sends the tile's
last frame (downscaled to the <= 640 px frame canvas) with its output-space rectangle ("c2c.seedvr2.tile"). The
rectangle comes from their tile_debug record (debug.decode_tile_boundaries) when present, else from the view's storage
offset inside z - their loop is never copied. Non-tiled decodes (no mark, or the tile is the whole latent) pass through.
Design: docs/research/seedvr2_preview.md (work area).
"""
from __future__ import annotations

import base64
import functools
import inspect
import io
import logging
import os
import sys
import threading
import time
from collections import OrderedDict

log = logging.getLogger("C2C.seedvr2_preview")

_TARGET_SUFFIX = "src.interfaces.video_upscaler"
_PHASES = ("encode_all_batches", "upscale_all_batches", "decode_all_batches", "postprocess_all_batches")
_MARK = "__c2c_seedvr2_preview__"
LIVE_MAX_SIDE = 640
STRIP_MAX_SIDE = 480
STRIP_MAX_FRAMES = 240
STRIP_BUDGET_BYTES = 48 * 1024 * 1024
LIVE_MIN_INTERVAL_S = 0.4
JPEG_QUALITY = 82

_VAE_SUFFIX = "src.models.video_vae_v3.modules.attn_video_vae"
TILE_MAX_RUNS = 8                     # per-run tile state kept (throttle clock, broken flag, decode counter)

_LOCK = threading.Lock()
_STRIPS: "OrderedDict[str, dict]" = OrderedDict()   # key -> {"after": [bytes], "before": [...], "alpha": [...], "idx": [...]}
_PATCHED: set = set()
_ROUTES = False


def enabled() -> bool:
    """L3.01: the same switch as the front end's "SeedVR2 preview" setting; C2C_SEEDVR2_PREVIEW still wins."""
    try:
        from . import _c2c_settings as _S
        return bool(_S.get("c2c.seedvr2.preview"))
    except Exception:  # noqa: BLE001
        return os.environ.get("C2C_SEEDVR2_PREVIEW", "1").strip() not in ("0", "false", "off", "no")


# ── finding and wrapping their module ─────────────────────────────────────────────────────────────────────────────
def _candidate_modules():
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == _TARGET_SUFFIX or name.endswith("." + _TARGET_SUFFIX)):
            continue
        if hasattr(mod, "SeedVR2VideoUpscaler") and all(callable(getattr(mod, p, None)) for p in _PHASES):
            yield name, mod


def _takes_progress_callback(fn) -> bool:
    try:
        return "progress_callback" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def install() -> int:
    """Wrap every loaded SeedVR2 node module once. Returns how many modules are wrapped now (0 = not installed)."""
    if not enabled():
        return 0
    n = 0
    for name, mod in _candidate_modules():
        dec, post = mod.decode_all_batches, mod.postprocess_all_batches
        if getattr(dec, _MARK, False) and getattr(post, _MARK, False):
            n += 1
            continue
        if not (_takes_progress_callback(dec) and _takes_progress_callback(post)):
            log.warning("[C2C] SeedVR2 preview: %s changed its phase signatures; preview off for it.", name)
            continue
        mod.decode_all_batches = _wrap_decode(dec)
        mod.postprocess_all_batches = _wrap_postprocess(post)
        _PATCHED.add(name)
        n += 1
        log.info("[C2C] SeedVR2 preview attached to %s", name)
    try:
        install_tiles()
    except Exception as exc:  # noqa: BLE001 - stage 2 never stops stage 1
        log.debug("SeedVR2 tile preview: %s", exc)
    return n


def _on_prompt(json_data):
    try:
        prompt = (json_data or {}).get("prompt") or {}
        if any(isinstance(v, dict) and v.get("class_type") == "SeedVR2VideoUpscaler" for v in prompt.values()):
            install()
    except Exception as exc:  # noqa: BLE001 - must never block a prompt
        log.debug("SeedVR2 preview on_prompt: %s", exc)
    return json_data


def register() -> bool:
    """Called from the pack's __init__: install now (if SeedVR2 is already loaded), on every prompt that uses it, and
    register the frame route."""
    if not enabled():
        return False
    install()
    try:
        from server import PromptServer
        inst = PromptServer.instance
        inst.add_on_prompt_handler(_on_prompt)
        _register_routes(inst)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("SeedVR2 preview: no server (%s)", exc)
        return False


# ── one run ───────────────────────────────────────────────────────────────────────────────────────────────────────
class _Run:
    def __init__(self):
        try:
            from comfy_execution.utils import get_executing_context
            ec = get_executing_context()
        except Exception:  # noqa: BLE001
            ec = None
        self.node = str(ec.node_id) if ec is not None else ""
        self.prompt_id = str(ec.prompt_id) if ec is not None else ""
        # -inf, not 0.0: time.monotonic() counts from boot, so with 0.0 the FIRST batch was dropped whenever the
        # machine had been up for less than LIVE_MIN_INTERVAL_S (measured 2026-11-03: uptime 2926 s, test interval 1 h)
        self.last_sent = float("-inf")
        self.broken = False

    @property
    def key(self) -> str:
        return f"{self.prompt_id}:{self.node}"

    def fail(self, where: str, exc: BaseException) -> None:
        if not self.broken:
            self.broken = True
            log.warning("[C2C] SeedVR2 preview stopped for this run (%s): %s", where, exc)


def _bind(fn, args, kwargs):
    sig = inspect.signature(fn)
    ba = sig.bind(*args, **kwargs)
    return ba


def _wrap_decode(orig):
    @functools.wraps(orig)
    def decode_all_batches(*args, **kwargs):
        run = _Run()
        try:
            install_tiles()                           # their VAE module is surely loaded by now (load order)
        except Exception as exc:  # noqa: BLE001
            log.debug("SeedVR2 tile preview: %s", exc)
        try:
            ba = _bind(orig, args, kwargs)
        except TypeError:
            return orig(*args, **kwargs)              # let theirs raise its own error
        theirs = ba.arguments.get("progress_callback")
        ctx = ba.arguments.get("ctx")

        def progress(current, total, frames, phase):
            if theirs is not None:
                theirs(current, total, frames, phase)
            if run.broken or not str(phase).startswith("Phase 3"):
                return
            try:
                _send_live(run, ctx, int(current), int(total))
            except Exception as exc:  # noqa: BLE001
                run.fail("decode", exc)

        ba.arguments["progress_callback"] = progress
        return orig(*ba.args, **ba.kwargs)

    setattr(decode_all_batches, _MARK, True)
    return decode_all_batches


def _wrap_postprocess(orig):
    @functools.wraps(orig)
    def postprocess_all_batches(*args, **kwargs):
        run = _Run()
        before = None
        prepend = 0
        try:
            ba = _bind(orig, args, kwargs)
            ctx = ba.arguments.get("ctx")
            prepend = int(ba.arguments.get("prepend_frames") or 0)
            src = ctx.get("input_images") if isinstance(ctx, dict) else None
            if src is not None:
                before = _strip_jpegs(src[prepend:], STRIP_MAX_SIDE, rgb_range=(0.0, 1.0))
        except Exception as exc:  # noqa: BLE001
            run.fail("post-process (input)", exc)
        result = orig(*args, **kwargs)
        if run.broken:
            return result
        try:
            out = result if isinstance(result, dict) else (args[0] if args else kwargs.get("ctx"))
            final = out.get("final_video") if isinstance(out, dict) else None
            if final is not None and getattr(final, "ndim", 0) == 4 and final.shape[0] > 0:
                _store_and_announce(run, final, before)
        except Exception as exc:  # noqa: BLE001
            run.fail("post-process (output)", exc)
        return result

    setattr(postprocess_all_batches, _MARK, True)
    return postprocess_all_batches


# ── frames → JPEG ─────────────────────────────────────────────────────────────────────────────────────────────────
def _encoder():
    try:
        from . import _c2c_preview_encode as enc
    except ImportError:  # loaded top-level (tests)
        import _c2c_preview_encode as enc
    return enc


def _small(frame_hwc, max_side: int, lo: float, hi: float):
    """[H,W,C] tensor (any device/dtype, values lo..hi) → uint8 numpy [h,w,C] with the long side <= max_side.
    L9.169: through the shared preview encoder (area downscale on the frame's device, rounded uint8)."""
    x = frame_hwc.detach()
    if x.ndim != 3:
        raise ValueError(f"frame shape {tuple(x.shape)}")
    if (float(lo), float(hi)) != (0.0, 1.0):
        x = (x.float() - lo) / (hi - lo)                    # one frame
    frames, why = _encoder().preview_frames(x.unsqueeze(0), long_edge=max_side)
    if frames is None:
        raise RuntimeError(why)
    return frames[0][..., :1] if int(x.shape[-1]) == 1 else frames[0]


def _jpeg(arr) -> bytes:
    from PIL import Image

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _strip_jpegs(frames_thwc, max_side: int, rgb_range=(0.0, 1.0), channels=slice(0, 3)):
    n = int(frames_thwc.shape[0])
    idx = list(range(n)) if n <= STRIP_MAX_FRAMES else [round(i * (n - 1) / (STRIP_MAX_FRAMES - 1))
                                                         for i in range(STRIP_MAX_FRAMES)]
    # every frame goes through _small (one frame in memory at a time; the single point a preview failure is caught)
    return idx, [_jpeg(_small(frames_thwc[i][..., channels], max_side, *rgb_range)) for i in idx]


def _send(event: str, data: dict) -> None:
    from server import PromptServer

    PromptServer.instance.send_sync(event, data)


def _send_live(run: _Run, ctx, current: int, total: int) -> None:
    now = time.monotonic()
    if current < total and now - run.last_sent < LIVE_MIN_INTERVAL_S:
        return
    info = ctx.get("decode_batch_info") or []
    final = ctx.get("final_video")
    if not info or final is None:
        return
    idx = int(info[-1][1]) - 1
    after = _jpeg(_small(final[idx][..., :3], LIVE_MAX_SIDE, -1.0, 1.0))   # decode writes [-1, 1]
    payload = {"node": run.node, "prompt_id": run.prompt_id, "batch": current, "batches": total,
               "frame": idx, "total_frames": int(final.shape[0]), "after": _data_url(after)}
    src = ctx.get("input_images")
    if src is not None and idx < int(src.shape[0]):
        payload["before"] = _data_url(_jpeg(_small(src[idx][..., :3], LIVE_MAX_SIDE, 0.0, 1.0)))
    run.last_sent = now
    _send("c2c.seedvr2.preview", payload)


def _store_and_announce(run: _Run, final, before) -> None:
    idx, after = _strip_jpegs(final, STRIP_MAX_SIDE)
    alpha = None
    if final.shape[-1] == 4:
        _, alpha = _strip_jpegs(final, STRIP_MAX_SIDE, channels=slice(3, 4))
    b_idx, b_jpegs = before if before else (None, None)
    entry = {"after": after, "idx": idx, "alpha": alpha,
             "before": b_jpegs if b_idx == idx else None,
             "w": int(final.shape[2]), "h": int(final.shape[1])}
    entry["bytes"] = sum(len(j) for k in ("after", "before", "alpha") for j in (entry[k] or []))
    with _LOCK:
        _STRIPS[run.key] = entry
        _STRIPS.move_to_end(run.key)
        while len(_STRIPS) > 3 or (len(_STRIPS) > 1 and sum(e["bytes"] for e in _STRIPS.values()) > STRIP_BUDGET_BYTES):
            _STRIPS.popitem(last=False)
    _send("c2c.seedvr2.done", {"node": run.node, "prompt_id": run.prompt_id, "key": run.key, "count": len(idx),
                               "frames": idx, "before": entry["before"] is not None, "alpha": alpha is not None,
                               "width": entry["w"], "height": entry["h"]})


def strip_frame(key: str, i: int, which: str):
    with _LOCK:
        e = _STRIPS.get(key)
        frames = e.get(which) if e else None
        if not frames or not (0 <= i < len(frames)):
            return None
        return frames[i]


def _register_routes(inst) -> None:
    global _ROUTES
    if _ROUTES:
        return
    from aiohttp import web

    async def _frame(request):
        q = request.rel_url.query
        try:
            i = int(q.get("i", "0"))
        except ValueError:
            return web.json_response({"error": "i must be a frame number"}, status=400)
        which = q.get("which", "after")
        if which not in ("after", "before", "alpha"):
            return web.json_response({"error": "which must be after, before or alpha"}, status=400)
        data = strip_frame(q.get("key", ""), i, which)
        if data is None:
            return web.json_response({"error": "That preview is no longer kept; run the node again."}, status=404)
        return web.Response(body=data, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    inst.routes.get("/c2c/seedvr2/frame")(_frame)
    _ROUTES = True


# ── stage 2 (L7.72): live tiles from their tiled VAE decode ───────────────────────────────────────────────────────
_TLS = threading.local()                           # .active = [record of the tiled_decode running on this thread]
_TILE_RUNS: "OrderedDict[str, dict]" = OrderedDict()   # run key -> {"last": t, "broken": bool, "seq": n}
_TILE_PATCHED: set = set()


def _vae_classes():
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == _VAE_SUFFIX or name.endswith("." + _VAE_SUFFIX)):
            continue
        for obj in list(vars(mod).values()):
            if isinstance(obj, type) and "tiled_decode" in obj.__dict__ and "slicing_decode" in obj.__dict__:
                yield name, obj


def _first_param(fn):
    try:
        ps = [p for p in inspect.signature(fn).parameters.values()]
    except (TypeError, ValueError):
        return None
    return ps[1].name if len(ps) > 1 else None


def install_tiles() -> int:
    """Wrap tiled_decode / slicing_decode of every loaded SeedVR2 VAE class once. Returns how many are wrapped."""
    if not enabled():
        return 0
    n = 0
    for name, cls in _vae_classes():
        td, sd = cls.__dict__["tiled_decode"], cls.__dict__["slicing_decode"]
        if getattr(td, _MARK, False) and getattr(sd, _MARK, False):
            n += 1
            continue
        if not (callable(td) and callable(sd) and _first_param(td) == "z" and _first_param(sd) == "z"):
            log.warning("[C2C] SeedVR2 tile preview: %s.%s changed its decode signatures; tiles off for it.",
                        name, cls.__name__)
            continue
        cls.tiled_decode = _wrap_tiled_decode(td)
        cls.slicing_decode = _wrap_slicing_decode(sd)
        _TILE_PATCHED.add(f"{name}.{cls.__name__}")
        n += 1
        log.info("[C2C] SeedVR2 tile preview attached to %s.%s", name, cls.__name__)
    return n


def _tile_state(run: _Run) -> dict:
    with _LOCK:
        st = _TILE_RUNS.get(run.key)
        if st is None:
            st = _TILE_RUNS[run.key] = {"last": float("-inf"), "broken": False, "seq": 0}
            while len(_TILE_RUNS) > TILE_MAX_RUNS:
                _TILE_RUNS.popitem(last=False)
        return st


def _wrap_tiled_decode(orig):
    @functools.wraps(orig)
    def tiled_decode(self, *args, **kwargs):
        rec = None
        try:
            if enabled():
                z = args[0] if args else kwargs.get("z")
                if z is not None and getattr(z, "ndim", 0) in (4, 5):
                    run = _Run()
                    st = _tile_state(run)
                    if not st["broken"]:
                        st["seq"] += 1
                        dbg = getattr(self, "debug", None)
                        bounds = getattr(dbg, "decode_tile_boundaries", None) if dbg else None
                        rec = {"vae": self, "z": z if z.ndim == 5 else z.unsqueeze(2), "run": run, "st": st,
                               "seq": st["seq"], "tile": 0, "grid": [],
                               "bounds": bounds if isinstance(bounds, list) else None,
                               "bounds_seen": len(bounds) if isinstance(bounds, list) else 0}
        except Exception as exc:  # noqa: BLE001
            log.debug("SeedVR2 tile preview (start): %s", exc)
            rec = None
        if rec is None:
            return orig(self, *args, **kwargs)
        stack = getattr(_TLS, "active", None)
        if stack is None:
            stack = _TLS.active = []
        stack.append(rec)
        try:
            return orig(self, *args, **kwargs)
        finally:
            stack.pop()

    setattr(tiled_decode, _MARK, True)
    return tiled_decode


def _wrap_slicing_decode(orig):
    @functools.wraps(orig)
    def slicing_decode(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        stack = getattr(_TLS, "active", None)
        if not stack:
            return out                                   # not inside a tiled decode: nothing of ours runs
        rec = stack[-1]
        if rec["vae"] is not self or rec["st"]["broken"]:
            return out
        try:
            _on_tile(rec, args[0] if args else kwargs.get("z"), out)
        except Exception as exc:  # noqa: BLE001
            rec["st"]["broken"] = True
            rec["run"].fail("tile", exc)
        return out

    setattr(slicing_decode, _MARK, True)
    return slicing_decode


def _tile_origin(z5, t):
    """(y, x) of the slice view t = z5[:, :, :, y:y+h, x:x+w] in latent cells, or None if t is not such a view."""
    if getattr(t, "ndim", 0) != 5 or tuple(t.shape[:3]) != tuple(z5.shape[:3]):
        return None
    if t.untyped_storage().data_ptr() != z5.untyped_storage().data_ptr():
        return None
    if (t.stride(3), t.stride(4)) != (z5.stride(3), z5.stride(4)):
        return None
    d = int(t.storage_offset()) - int(z5.storage_offset())
    sh, sw = int(z5.stride(3)), int(z5.stride(4))
    H, W, h, w = int(z5.shape[3]), int(z5.shape[4]), int(t.shape[3]), int(t.shape[4])
    if d < 0 or sh <= 0 or sw <= 0:
        return None
    if sh >= sw:
        y, r = divmod(d, sh)
        x = r // sw
    else:
        x, r = divmod(d, sw)
        y = r // sh
    if y * sh + x * sw != d or y + h > H or x + w > W:
        return None
    return y, x


def _on_tile(rec, tile_latent, decoded) -> None:
    z5, vae = rec["z"], rec["vae"]
    H, W = int(z5.shape[3]), int(z5.shape[4])
    if tile_latent is None or getattr(decoded, "ndim", 0) != 5:
        return
    if int(tile_latent.shape[-2]) == H and int(tile_latent.shape[-1]) == W:
        return                                           # the whole latent = their non-tiled path
    sf = getattr(vae, "spatial_downsample_factor", None)
    if not isinstance(sf, int) or sf <= 0:
        sf = max(1, int(decoded.shape[-2]) // max(1, int(tile_latent.shape[-2])))
    fw, fh = W * sf, H * sf
    rect = None
    bounds = rec["bounds"]
    if bounds is not None and len(bounds) > rec["bounds_seen"]:
        b = bounds[-1]                                   # their tile_debug record of this tile (output pixels)
        rec["bounds_seen"] = len(bounds)
        rect = (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))
    if rect is None:
        org = _tile_origin(z5, tile_latent)
        if org is None:
            return
        y, x = org
        rect = (x * sf, y * sf, int(tile_latent.shape[-1]) * sf, int(tile_latent.shape[-2]) * sf)
    rec["tile"] += 1
    rec["grid"].append(list(rect))
    x0, y0, tw, th = rect
    last = x0 + tw >= fw and y0 + th >= fh                 # their raster order ends at the bottom-right corner
    st, run = rec["st"], rec["run"]
    now = time.monotonic()
    if not last and now - st["last"] < LIVE_MIN_INTERVAL_S:
        return
    scale = min(1.0, LIVE_MAX_SIDE / float(max(fw, fh)))
    side = max(1, int(round(max(tw, th) * scale)))
    frame = decoded[0, :3, -1].permute(1, 2, 0)          # the tile's last frame, HWC, still on their device
    img = _jpeg(_small(frame, side, -1.0, 1.0))         # VAE output is [-1, 1]
    st["last"] = now
    _send("c2c.seedvr2.tile", {"node": run.node, "prompt_id": run.prompt_id, "seq": rec["seq"], "tile": rec["tile"],
                               "x": x0, "y": y0, "w": tw, "h": th, "width": fw, "height": fh, "last": bool(last),
                               "grid": [list(g) for g in rec["grid"]], "image": _data_url(img)})
