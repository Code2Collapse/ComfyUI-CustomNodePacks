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

_LOCK = threading.Lock()
_STRIPS: "OrderedDict[str, dict]" = OrderedDict()   # key -> {"after": [bytes], "before": [...], "alpha": [...], "idx": [...]}
_PATCHED: set = set()
_ROUTES = False


def enabled() -> bool:
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
        self.last_sent = 0.0
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
def _small(frame_hwc, max_side: int, lo: float, hi: float):
    """[H,W,C] tensor (any device/dtype, values lo..hi) → uint8 numpy [h,w,C] with the long side <= max_side."""
    import torch
    import torch.nn.functional as F

    x = frame_hwc.detach()
    if x.ndim != 3:
        raise ValueError(f"frame shape {tuple(x.shape)}")
    x = x.float()
    x = ((x - lo) / (hi - lo)).clamp(0, 1)
    h, w = int(x.shape[0]), int(x.shape[1])
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        x = F.interpolate(x.permute(2, 0, 1).unsqueeze(0), size=(nh, nw), mode="area")[0].permute(1, 2, 0)
    return (x.cpu().numpy() * 255.0 + 0.5).astype("uint8")


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
