"""Inpaint Crop Pro live preview (L7.99 slice 2, owner A11): the preview follows a changed crop setting without a run.

Each run of the node keeps, per node, what the crop's geometry needs: the image size, the masks exactly as the node
measured them (bit-packed when every value is 0 or 1, float32 otherwise) and the preview frame. When a setting
changes, ``POST /c2c/inpaint_crop/plan`` recomputes the windows with the node's own geometry functions
(``InpaintCropProMEC.plan_windows``) and draws the same picture the node draws after a run. Nothing is estimated: a
request the cache cannot answer exactly (no run yet, another preview frame, an entry too big to keep) returns
``{"stale": true, "reason": ...}`` and the picture of the last run stays.

Memory: one shared budget (128 MB) over all nodes, least recently used out first; an entry larger than the budget
is not kept. Design: docs/research/l799_inpaint_crop_stitch.md (D4.2).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("MEC.InpaintCropLive")

BUDGET_BYTES = 128 * 1024 * 1024
_LOCK = threading.Lock()
_ENTRIES: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


_CHUNK = 8 * 1024 * 1024                     # elements per packing step (bounded transient memory on any device)
_BITS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int32)


def _pack(t: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    """Exact compact copy: 1 bit per pixel when every value is 0 or 1 (packed on the tensor's own device in chunks,
    so a binary 81-frame 1080p mask costs ~21 MB and never a full float copy), float32 otherwise."""
    if t is None:
        return None
    t = t.detach()
    flat = t.reshape(-1)
    binary = all(bool(torch.logical_or(c == 0, c == 1).all()) for c in flat.split(_CHUNK))
    if binary:
        parts = []
        w = _BITS.to(flat.device)
        for c in flat.split(_CHUNK):                         # _CHUNK is a multiple of 8: only the last part pads
            c = c.to(torch.int32)
            pad = (-c.numel()) % 8
            if pad:
                c = torch.cat([c, c.new_zeros(pad)])
            parts.append((c.view(-1, 8) * w).sum(1).to(torch.uint8).cpu().numpy())
        return {"kind": "bits", "shape": tuple(t.shape), "data": np.concatenate(parts) if parts else np.zeros(0, np.uint8)}
    if t.numel() * 4 > BUDGET_BYTES:
        raise MemoryError("soft mask too large for the live preview")
    return {"kind": "f32", "shape": tuple(t.shape), "data": t.float().cpu().contiguous().numpy().copy()}


def _unpack(p: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    if p is None:
        return None
    if p["kind"] == "bits":
        n = int(np.prod(p["shape"]))
        return torch.from_numpy(np.unpackbits(p["data"])[:n].reshape(p["shape"]).astype(np.float32))
    return torch.from_numpy(p["data"].copy())


def _nbytes(entry: Dict[str, Any]) -> int:
    total = int(entry["frame"].nbytes)
    for k in ("mask", "opt"):
        if entry[k] is not None:
            total += int(entry[k]["data"].nbytes)
    return total


def store(node_id: Optional[str], *, shape_bhw, mask, opt_mask, frame: torch.Tensor, frame_index: int,
          settings: Dict[str, Any]) -> bool:
    """Keep what the preview needs from this run (the masks as the geometry reads them). False = not kept."""
    if not node_id:
        return False
    try:
        fr = frame[..., :3].detach().float().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        entry = {"shape": tuple(int(v) for v in shape_bhw), "mask": _pack(mask), "opt": _pack(opt_mask),
                 "frame": fr, "frame_index": int(frame_index), "settings": dict(settings)}
    except Exception as exc:  # noqa: BLE001 - the live preview must never fail a run (incl. too large)
        logger.debug("[InpaintCropLive] not cached: %s", exc)
        with _LOCK:
            _ENTRIES.pop(str(node_id), None)                   # never answer with an older run's masks
        return False
    size = _nbytes(entry)
    with _LOCK:
        _ENTRIES.pop(str(node_id), None)
        if size > BUDGET_BYTES:
            return False
        _ENTRIES[str(node_id)] = entry
        while sum(_nbytes(e) for e in _ENTRIES.values()) > BUDGET_BYTES and len(_ENTRIES) > 1:
            _ENTRIES.popitem(last=False)
    return True


def get(node_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        e = _ENTRIES.get(str(node_id))
        if e is not None:
            _ENTRIES.move_to_end(str(node_id))
        return e


def clear() -> None:
    with _LOCK:
        _ENTRIES.clear()


def render(node_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """The preview for the cached run of ``node_id`` with ``values`` changed. Same numbers and picture as a run."""
    from .inpaint_suite import (InpaintCropProMEC, _crop_preview_lines, _draw_crop_preview, _edge_replicate_pad,
                                _resize_image_alg)
    from ._video_comparer_io import save_preview_png

    entry = get(node_id)
    if entry is None:
        return {"stale": True, "reason": "Run the node once; the preview then follows the crop settings."}
    known = set(entry["settings"])
    s = {**entry["settings"], **{k: v for k, v in (values or {}).items() if k in known}}
    B, H, W = entry["shape"]
    k = max(0, min(int(s.get("preview_frame", 0) or 0), B - 1))
    if k != entry["frame_index"]:
        return {"stale": True, "reason": f"Run the node to see frame {k}."}
    res = InpaintCropProMEC.plan_windows((B, H, W), _unpack(entry["mask"]), _unpack(entry["opt"]), **s)
    plan, geo = res["plan"], res["geo"]
    frame = torch.from_numpy(entry["frame"]).float().div(255.0).unsqueeze(0)          # [1,H,W,3]
    if geo["preresize"] is not None:
        nw, nh, alg = geo["preresize"]
        frame = _resize_image_alg(frame, nh, nw, alg)
    if geo["extend_pads"] is not None:
        frame = _edge_replicate_pad(frame, *geo["extend_pads"])
    fr = plan["frames"][k if len(plan["frames"]) > 1 else 0]
    x, y, w, h = fr["window"]
    pads = fr["padding"]
    canvas = _edge_replicate_pad(frame, pads["top"], pads["bottom"], pads["left"], pads["right"])
    cx, cy = x + pads["left"], y + pads["top"]
    crop = canvas[:, cy:cy + h, cx:cx + w, :]
    ow, oh = plan["output_size"]
    if (ow, oh) != (w, h):
        alg = s["upscale_algorithm"] if (ow > w or oh > h) else s["downscale_algorithm"]
        crop = _resize_image_alg(crop, oh, ow, alg)
    pic = _draw_crop_preview(frame[0], geo["bbox_mask"][k], fr["window"], crop[0],
                             _crop_preview_lines(plan, fr, k, B) + ["(live - follows the settings; run to update the crop)"])
    saved = save_preview_png(pic, "inpaint_crop_preview")
    return {"images": [saved], "plan": plan} if saved else {"stale": True, "reason": "Preview could not be saved."}


def register_routes(server: Any) -> None:
    from aiohttp import web

    @server.routes.post("/c2c/inpaint_crop/plan")
    async def _plan(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "invalid JSON"}, status=400)
        node_id = str((body or {}).get("node_id") or "")
        values = (body or {}).get("values") or {}
        if not node_id or not isinstance(values, dict):
            return web.json_response({"error": "node_id and values are required"}, status=400)
        try:
            out = await asyncio.get_running_loop().run_in_executor(None, render, node_id, values)
        except Exception as exc:  # noqa: BLE001 - plain message, never a traceback in the UI
            logger.warning("[InpaintCropLive] preview failed: %s", exc)
            return web.json_response({"stale": True, "reason": f"Preview not updated: {exc}"})
        return web.json_response(out)
