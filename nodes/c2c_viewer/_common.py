"""Shared pieces of the Matte viewer's helper nodes (L7.114): proxy sizes, frame sampling, temp files for /view.

The helper nodes run inside ComfyUI's own queue (the viewer posts small helper prompts to /prompt); what they hand the
browser are temp files (ComfyUI serves them through /view and clears the temp folder on start) and a small JSON in the
node's `ui` output under the key "c2c_viewer". Budgets (design_matte_segment_viewer.md 5): proxies at most MAX_SIDE px
on the long side, at most MAX_FRAMES frames per set (every k-th frame of a longer clip).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SUBFOLDER = "c2c_viewer"
MAX_SIDE = 960
MAX_FRAMES = 240
UI_KEY = "c2c_viewer"
#: frontend 1.53 hides dev_only node definitions from the node library, the search box and the Add Node menu unless
#: Comfy.DevMode is on (settingStore: registerNodeDef sets skip_list; nodeDefStore filter core.dev_only); the server
#: sends the flag from the class attribute DEV_ONLY (server.py node_info). Registered = a helper prompt validates.
HIDDEN_CATEGORY = "MaskEditControl/Viewer helpers"


def temp_root() -> str:
    import folder_paths
    return folder_paths.get_temp_directory()


def out_dir(tag: str) -> str:
    d = os.path.join(temp_root(), SUBFOLDER, tag)
    os.makedirs(d, exist_ok=True)
    return d


def file_ref(tag: str, filename: str) -> Dict[str, str]:
    """What the browser passes to /view?filename=..&subfolder=..&type=temp."""
    return {"filename": filename, "subfolder": f"{SUBFOLDER}/{tag}", "type": "temp"}


def proxy_size(width: int, height: int, max_side: int = MAX_SIDE) -> Tuple[int, int]:
    s = min(1.0, float(max_side) / float(max(width, height)))
    return max(1, int(round(width * s))), max(1, int(round(height * s)))


def sample_indices(n: int, max_frames: int = MAX_FRAMES) -> Tuple[List[int], int]:
    """Every frame of a short clip, every k-th of a long one (the last frame always included). Returns (idx, k)."""
    n = max(0, int(n))
    if n <= max_frames:
        return list(range(n)), 1
    k = int(math.ceil(n / float(max_frames)))
    idx = list(range(0, n, k))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx, k


def resize_image(frame_hwc: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    """One frame (H,W,C) -> uint8 (h,w,3) at ``size`` = (w, h), area-averaged."""
    w, h = size
    x = frame_hwc[..., :3].detach().float().cpu().clamp(0, 1).permute(2, 0, 1).unsqueeze(0)
    if (x.shape[-1], x.shape[-2]) != (w, h):
        x = F.interpolate(x, size=(h, w), mode="area")
    return (x[0].permute(1, 2, 0).numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def resize_mask(mask_hw: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    x = mask_hw.detach().float().cpu().clamp(0, 1)[None, None]
    if (x.shape[-1], x.shape[-2]) != (w, h):
        x = F.interpolate(x, size=(h, w), mode="area")
    return (x[0, 0].numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def save_jpeg(rgb_u8: np.ndarray, path: str, quality: int = 85) -> None:
    from PIL import Image
    Image.fromarray(rgb_u8, "RGB").save(path, "JPEG", quality=quality)


def save_alpha_png(a_u8: np.ndarray, path: str) -> None:
    """White with the mask in the ALPHA channel ("LA"): the browser composites it with destination-in."""
    from PIL import Image
    la = np.stack([np.full_like(a_u8, 255), a_u8], -1)
    Image.fromarray(la, "LA").save(path, "PNG", optimize=False, compress_level=3)


def ui(payload: Dict[str, Any]) -> Dict[str, Any]:
    """A helper's output: nothing for the graph, one JSON for the viewer (a list: ComfyUI concatenates ui lists)."""
    return {"ui": {UI_KEY: [payload]}, "result": ()}


def frame_index(frame: int, frames: int, what: str = "frame") -> int:
    f = int(frame)
    if f < 0 or f >= frames:
        raise ValueError(f"The viewer asked for {what} {f}, but the plate has {frames} frame"
                         f"{'s' if frames != 1 else ''}. Load the plate again in the viewer.")
    return f


def normalised_points(raw: Any, width: int, height: int) -> Tuple[List[Tuple[float, float]],
                                                                 List[Tuple[float, float]]]:
    """[{x, y, label}] in 0..1 -> (positive, negative) in pixels."""
    import json
    if not raw:
        return [], []
    data = json.loads(raw) if isinstance(raw, str) else raw
    pos, neg = [], []
    for p in data or []:
        x = min(width - 1.0, max(0.0, float(p["x"]) * width))
        y = min(height - 1.0, max(0.0, float(p["y"]) * height))
        (pos if int(p.get("label", 1)) != 0 else neg).append((x, y))
    return pos, neg


def normalised_boxes(raw: Any, width: int, height: int) -> List[Tuple[int, int, int, int]]:
    import json
    if not raw:
        return []
    data = json.loads(raw) if isinstance(raw, str) else raw
    out = []
    for b in data or []:
        x0, y0, x1, y1 = (float(v) for v in b[:4])
        box = (int(round(x0 * width)), int(round(y0 * height)), int(round(x1 * width)), int(round(y1 * height)))
        if box[2] > box[0] and box[3] > box[1]:
            out.append(box)
    return out


def bbox_of(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def split_terms(text: str) -> Sequence[str]:
    return [t.strip() for t in str(text or "").split(",") if t.strip()]
