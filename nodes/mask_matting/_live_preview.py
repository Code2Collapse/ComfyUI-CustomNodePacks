"""Live preview of Mask + Matting while it runs (L7.73, the masking half of L7.57).

While MaskOps works through a clip, the Matte viewer on the node shows the matte of the newest finished frame and a
count "frame i / N", labelled by pass (segmentation, matting). When the run ends the node's normal result replaces it.

Event ``c2c.mask_matting.live`` (PromptServer.send_sync, to the client that queued the prompt):
    {"node": "12:5", "display_node": "12:5", "prompt_id": "...", "stage": "segmentation" | "matting",
     "label": "classic", "frame": 41, "done": 42, "total": 120, "source_width": 1920, "source_height": 1080,
     "width": 640, "height": 360, "image": "data:image/jpeg;base64,..."}
"image"/"width"/"height" are absent on a count-only event (a backend whose per-frame matte is not reachable).

Budgets: at most one event per MIN_INTERVAL_S (fake-clock tested), at most one frame per event (the newest finished),
long side <= MAX_SIDE (<= the viewer's proxy size). The frame is area-downscaled on its own device, so the only extra
device work is one proxy-size frame moved to the CPU per event; frames between events are only referenced, never
copied.

How partial results are reached without touching the backends (their files stay as they are): for the duration of one
pass, ``segmentation`` / ``matting`` / ``onyx`` below wrap call boundaries that already exist, the way
nodes/seedvr2_preview.py wraps numz's phases, and restore them afterwards:
  * the backend module's ``interruptible_range`` (every per-frame loop of the SAM 3 / SAM 3.1 / salient / ViTMatte /
    RVM / MatAnyone / BGMv2 backends) -> the frame count;
  * the instance's ``_seg_one`` (SAM 3 / SAM 3.1 per-frame path) and ``_matte_multi_frame`` (ViTMatte) -> that
    frame's matte;
  * ``_classic.ClassicVideoTracker.propagate_in_video`` (model-free segmenter) and, in ONYX, the tracker handed to
    ``_onyx_pipeline.propagate_objects`` (SAM 3.1 / core SAM 3 / model-free) -> each tracked frame's mask;
  * ONYX's ``refine_frame_tiled`` (the matter, one object-frame per call) -> each matted frame.

Only inside ComfyUI's executor (comfy_execution.utils.get_executing_context): a direct Python call gets nothing
wrapped. The node id is the executing node's id ("12:5" inside a subgraph), as in seedvr2_preview.py.
Any error in this module stops previewing for that run (logged once) and never reaches the node's result.
Kill switch: C2C_MASK_LIVE_PREVIEW=0 (also off when the Matte viewer is off, C2C_VIEWER=0).
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, Iterator, Optional

log = logging.getLogger("MEC.MaskOps.live")

EVENT = "c2c.mask_matting.live"
MIN_INTERVAL_S = 0.5
JPEG_QUALITY = 82
#: long side of a live frame; <= the viewer's proxy (c2c_viewer/_common.MAX_SIDE = 960, pinned by
#: tests/test_l773_live_preview.py). Not imported from there: that package's __init__ loads the helper nodes.
MAX_SIDE = 640

_RUN: Dict[str, Any] = {"key": None, "live": None}     # the current run's state (one node executes at a time)


def enabled() -> bool:
    """The C2C setting 'Live matte while running' (C2C_MASK_LIVE_PREVIEW=0 still wins), and the Matte viewer on."""
    try:
        from .. import _c2c_settings as _S
        on = bool(_S.get("c2c.matte.livePreview"))
    except Exception:  # noqa: BLE001 - no settings registry: the env variable alone
        on = os.environ.get("C2C_MASK_LIVE_PREVIEW", "1").strip().lower() not in ("0", "false", "off", "no")
    if not on:
        return False
    try:
        from ._picks import viewer_enabled
        return bool(viewer_enabled())
    except Exception:  # noqa: BLE001
        return os.environ.get("C2C_VIEWER", "").strip() != "0"


def _context():
    try:
        from comfy_execution.utils import get_executing_context
        return get_executing_context()
    except Exception:  # noqa: BLE001 - an older core: no live preview
        return None


def _server_send(event: str, data: dict) -> None:
    from server import PromptServer
    inst = PromptServer.instance
    inst.send_sync(event, data, getattr(inst, "client_id", None))


def _display_node(node_id: str) -> str:
    try:  # core sets last_node_id to the display id right before it runs the node (execution.py)
        from server import PromptServer
        d = getattr(PromptServer.instance, "last_node_id", None)
        return str(d) if d not in (None, "") else node_id
    except Exception:  # noqa: BLE001
        return node_id


class LivePreview:
    """One run of one node: the throttle and the fail-once switch."""

    def __init__(self, node: str, prompt_id: str, *, display_node: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic, send: Optional[Callable[[str, dict], None]] = None):
        self.node = str(node)
        self.prompt_id = str(prompt_id)
        self.display_node = str(display_node) if display_node else self.node
        self.clock = clock
        self.send = send or _server_send
        # -inf, not 0.0: time.monotonic() counts from boot (seedvr2_preview.py, measured 2026-11-03)
        self.last_sent = float("-inf")
        self.broken = False
        self.sent = 0

    def fail(self, where: str, exc: BaseException) -> None:
        if not self.broken:
            self.broken = True
            log.warning("[C2C] Mask + Matting live preview stopped for this run (%s): %s", where, exc)

    def due(self) -> bool:
        return not self.broken and self.clock() - self.last_sent >= MIN_INTERVAL_S

    def report(self, stage: str, done: int, total: int, *, frame_index: Optional[int] = None, frame: Any = None,
               label: str = "") -> bool:
        """O(1) unless an event is due. ``frame``: the newest finished frame's matte ([H,W] or [O,H,W], tensor or
        numpy, any device) or a zero-argument callable returning it. Returns True when an event was sent."""
        if not self.due():
            return False
        try:
            now = self.clock()
            payload: Dict[str, Any] = {
                "node": self.node, "display_node": self.display_node, "prompt_id": self.prompt_id,
                "stage": str(stage), "label": str(label or ""), "done": int(done), "total": int(total),
                "frame": int(frame_index) if frame_index is not None else max(0, int(done) - 1),
            }
            if callable(frame):
                frame = frame()
            if frame is not None:
                u8, (sw, sh) = proxy_u8(frame)
                payload.update(image=_data_url(u8), width=int(u8.shape[1]), height=int(u8.shape[0]),
                               source_width=int(sw), source_height=int(sh))
            self.send(EVENT, payload)
            self.last_sent = now
            self.sent += 1
            return True
        except Exception as exc:  # noqa: BLE001 - the preview never touches the node's result
            self.fail(stage, exc)
            return False


def current() -> Optional[LivePreview]:
    """The live preview of the executing node (None: off, or not inside ComfyUI's executor)."""
    if not enabled():
        return None
    ec = _context()
    if ec is None:
        return None
    try:
        key = (str(ec.prompt_id), str(ec.node_id))
        if _RUN["key"] != key or _RUN["live"] is None:
            _RUN["key"] = key
            _RUN["live"] = LivePreview(key[1], key[0], display_node=_display_node(key[1]))
        return _RUN["live"]
    except Exception as exc:  # noqa: BLE001
        log.debug("live preview context: %s", exc)
        return None


# ── one frame -> a small grey JPEG ────────────────────────────────────────────────────────────────────────────────
def proxy_u8(frame: Any, max_side: int = MAX_SIDE):
    """The matte of one frame -> (uint8 [h, w] with max(h, w) <= max_side, (source w, source h)). [O,H,W] = the union
    of the objects. Downscaled on the frame's own device; only the proxy goes to the CPU."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.asarray(frame)) if not isinstance(frame, torch.Tensor) else frame.detach()
    while x.ndim > 2:
        x = x.amax(0) if x.shape[0] > 1 else x[0]
    if x.ndim != 2 or x.numel() == 0:
        raise ValueError(f"matte frame shape {tuple(x.shape)}")
    h, w = int(x.shape[0]), int(x.shape[1])
    s = min(1.0, float(max_side) / float(max(w, h)))
    tw, th = max(1, int(round(w * s))), max(1, int(round(h * s)))
    with torch.no_grad():
        y = x.float()
        if (tw, th) != (w, h):
            y = F.interpolate(y[None, None], size=(th, tw), mode="area")[0, 0]
        y = (y.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
    return y, (w, h)


def _data_url(u8) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(u8, "L").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── taps (installed for one pass, always restored) ────────────────────────────────────────────────────────────────
class _Patches:
    def __init__(self):
        self._undo = []

    def set(self, obj, name: str, value) -> None:
        had = name in getattr(obj, "__dict__", {})
        old = obj.__dict__[name] if had else None
        setattr(obj, name, value)
        self._undo.append((obj, name, had, old))

    def restore(self) -> None:
        while self._undo:
            obj, name, had, old = self._undo.pop()
            try:
                if had:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)
            except Exception as exc:  # noqa: BLE001
                log.warning("[C2C] live preview could not restore %s.%s: %s", obj, name, exc)


class _Tap:
    """One pass: the newest finished frame and the count."""

    def __init__(self, live: LivePreview, stage: str, total: int, label: str):
        self.live, self.stage, self.total, self.label = live, stage, max(1, int(total)), label
        self.latest = None          # (frame index or None, matte reference)
        self.seen = set()

    def frame_done(self, index: Optional[int], matte=None, done: Optional[int] = None) -> None:
        try:
            if done is None:
                if index is not None:
                    self.seen.add(int(index))
                done = len(self.seen)
            self.live.report(self.stage, min(done, self.total), self.total, frame_index=index, frame=matte,
                             label=self.label)
        except Exception as exc:  # noqa: BLE001
            self.live.fail(self.stage, exc)

    # interruptible_range(n, label) -> the same indices; frame i is done when the loop asks for i + 1
    def wrap_range(self, orig):
        tap = self

        def interruptible_range(n, *a, **k):
            for i in orig(n, *a, **k):
                yield i
                lat, tap.latest = tap.latest, None
                tap.frame_done(i, lat)
        return interruptible_range

    # a per-frame call returning that frame's matte (pick(result) -> the matte)
    def wrap_frame_call(self, orig, pick):
        tap = self

        def call(*a, **k):
            out = orig(*a, **k)
            try:
                tap.latest = pick(out)
            except Exception:  # noqa: BLE001
                tap.latest = None
            return out
        return call

    # a tracker's propagate_in_video generator: (frame index, {"out_binary_masks": [O,H,W], ...})
    def wrap_propagation(self, gen) -> Iterator:
        try:
            for item in gen:
                try:
                    fi, out = item
                    masks = out.get("out_binary_masks") if isinstance(out, dict) else None
                    self.frame_done(int(fi), masks)
                except Exception as exc:  # noqa: BLE001
                    self.live.fail(self.stage, exc)
                yield item
        finally:
            close = getattr(gen, "close", None)
            if close is not None:
                close()


def _range_module(inst):
    mod = sys.modules.get(type(inst).__module__)
    return mod if mod is not None and callable(getattr(mod, "interruptible_range", None)) else None


def _label(inst) -> str:
    return str(getattr(inst, "KEY", "") or type(inst).__name__)


class _pass:
    """Run ``build(patches)`` (install taps) on enter, restore on exit. Any error while installing = no preview for
    the pass, same run. ``after(result)``: optional, called by ``within`` with the wrapped call's result."""

    def __init__(self, build, after=None):
        self.build, self.after, self.patches = build, after, _Patches()

    def __enter__(self):
        try:
            self.build(self.patches)
        except Exception as exc:  # noqa: BLE001
            self.patches.restore()
            self.after = None
            log.warning("[C2C] Mask + Matting live preview off for this pass: %s", exc)
        return self

    def __exit__(self, *exc):
        self.patches.restore()
        return False


def segmentation(seg_inst, total: int):
    """Around one segmenter call of the cascade pipeline."""
    def build(p: _Patches):
        live = current()
        if live is None or live.broken:
            return
        tap = _Tap(live, "segmentation", total, _label(seg_inst))
        mod = _range_module(seg_inst)
        if mod is not None:
            p.set(mod, "interruptible_range", tap.wrap_range(mod.interruptible_range))
        one = getattr(seg_inst, "_seg_one", None)
        if callable(one):
            p.set(seg_inst, "_seg_one", tap.wrap_frame_call(one, lambda r: r[0]))
        from . import _classic
        cls = _classic.ClassicVideoTracker
        orig = cls.__dict__.get("propagate_in_video")
        if orig is not None:
            def propagate_in_video(self, *a, **k):
                return tap.wrap_propagation(orig(self, *a, **k))
            p.set(cls, "propagate_in_video", propagate_in_video)
    return _pass(build)


def matting(mat_inst, total: int):
    """Around one matter call of the cascade pipeline. A matter with no per-frame loop to tap (the guided filter: one
    batch call) reports once when it returns: its last frame, "N / N"."""
    state: Dict[str, Any] = {}

    def build(p: _Patches):
        live = current()
        if live is None or live.broken or mat_inst is None:
            return
        tap = _Tap(live, "matting", total, _label(mat_inst))
        state["tap"] = tap
        mod = _range_module(mat_inst)
        if mod is not None:
            p.set(mod, "interruptible_range", tap.wrap_range(mod.interruptible_range))
        one = getattr(mat_inst, "_matte_multi_frame", None)
        if callable(one):
            p.set(mat_inst, "_matte_multi_frame", tap.wrap_frame_call(one, lambda r: r[0]))

    def after(result):
        tap = state.get("tap")
        if tap is None or tap.seen or tap.live.broken:
            return
        alpha = result.get("alpha") if isinstance(result, dict) else None
        if alpha is None or getattr(alpha, "ndim", 0) < 2:
            return
        last = alpha[-1] if alpha.ndim >= 3 else alpha
        tap.frame_done(tap.total - 1, last, done=tap.total)
    return _pass(build, after)


class _TrackerTap:
    """The tracker ONYX propagates with, with its propagate_in_video tapped (everything else passes through)."""

    def __init__(self, model, tap: _Tap):
        self.__dict__["_c2c_model"] = model
        self.__dict__["_c2c_tap"] = tap

    def propagate_in_video(self, *a, **k):
        return self._c2c_tap.wrap_propagation(self._c2c_model.propagate_in_video(*a, **k))

    def __getattr__(self, name):
        return getattr(self.__dict__["_c2c_model"], name)


def onyx(total: int):
    """Around run_onyx_pipeline: the tracking pass and the matter pass (per object-frame)."""
    def build(p: _Patches):
        live = current()
        if live is None or live.broken:
            return
        from . import _onyx_pipeline as op
        seg = _Tap(live, "segmentation", total, "onyx tracker")
        orig_prop = op.propagate_objects

        def propagate_objects(model, *a, **k):
            seg.label = _label(model)
            return orig_prop(_TrackerTap(model, seg), *a, **k)
        p.set(op, "propagate_objects", propagate_objects)

        mat = _Tap(live, "matting", total, "onyx matter")
        orig_obj, orig_frame = op.refine_object_tiled, op.refine_frame_tiled
        shape = {"O": 1, "B": max(1, int(total)), "k": 0}

        def refine_object_tiled(matter, img_bhwc, trimap_obhw, *a, **k):
            try:
                shape.update(O=int(trimap_obhw.shape[0]), B=int(trimap_obhw.shape[1]), k=0)
                mat.label = _label(matter)
                mat.total = max(1, shape["B"])
            except Exception:  # noqa: BLE001
                pass
            return orig_obj(matter, img_bhwc, trimap_obhw, *a, **k)

        def refine_frame_tiled(*a, **k):
            out = orig_frame(*a, **k)
            try:
                b = shape["k"] % shape["B"]
                shape["k"] += 1
                # one object-frame per call; the count is frames of the object being matted
                mat.frame_done(b, out, done=b + 1)
            except Exception as exc:  # noqa: BLE001
                live.fail("matting", exc)
            return out
        p.set(op, "refine_object_tiled", refine_object_tiled)
        p.set(op, "refine_frame_tiled", refine_frame_tiled)
    return _pass(build)


def within(cm, fn):
    """``fn`` run inside the context manager ``cm`` (keeps the call sites in node.py one-line edits)."""
    def run(*a, **k):
        with cm:
            out = fn(*a, **k)
            after = getattr(cm, "after", None)
            if after is not None:
                try:
                    after(out)
                except Exception as exc:  # noqa: BLE001 - never the node's result
                    log.warning("[C2C] Mask + Matting live preview (end of pass): %s", exc)
            return out
    return run
