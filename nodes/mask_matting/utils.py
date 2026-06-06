"""Shared utilities for MaskMattingMEC backends."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

logger = logging.getLogger("MEC.MaskMatting")


# ──────────────────────────────────────────────────────────────────────
# Folder registration. Each backend gets its own subfolder under
# ComfyUI/models/ so users can drop weights in once and forget.
# ──────────────────────────────────────────────────────────────────────
_BACKEND_FOLDERS: Dict[str, str] = {
    "sam2":           "sam2",
    "sam3":           "sam3",
    "sam3.1":         "sam3.1",
    "sec":            "sec",
    "vitmatte":       "vitmatte",
    "rvm":            "rvm",
    "matanyone":      "matanyone",
    "birefnet":       "BiRefNet",
    "rmbg":           "RMBG",
    "grounding-dino": "grounding-dino",
    "videomama":      "videomama",
    "inspyrenet":     "inspyrenet",
    "cutie":          "cutie",
    "dis":            "dis",
    "xmem":           "xmem",
    "bgmattingv2":    "bgmattingv2",
    "dino":           "dino",
}


def _register_backend_folders() -> None:
    """Tell ``folder_paths`` about each backend's models dir.

    Safe to call multiple times; uses ``add_model_folder_path`` if available
    and falls back to mutating ``folder_names_and_paths`` directly.
    """
    models_root = folder_paths.models_dir if hasattr(folder_paths, "models_dir") else os.path.join(
        os.path.dirname(folder_paths.__file__), "models"
    )
    for key, sub in _BACKEND_FOLDERS.items():
        abs_dir = os.path.join(models_root, sub)
        try:
            os.makedirs(abs_dir, exist_ok=True)
        except OSError:
            pass
        try:
            folder_paths.add_model_folder_path(key, abs_dir, is_default=False)
        except (AttributeError, TypeError):
            # older ComfyUI: mutate the registry directly
            try:
                table = folder_paths.folder_names_and_paths
                exts = {".pt", ".pth", ".safetensors", ".bin", ".onnx"}
                if key in table:
                    paths, e = table[key]
                    if abs_dir not in paths:
                        paths.append(abs_dir)
                    table[key] = (paths, e | exts)
                else:
                    table[key] = ([abs_dir], exts)
            except Exception:
                pass


_register_backend_folders()


# ──────────────────────────────────────────────────────────────────────
# Quiet HF/transformers progress bars in the ComfyUI console. The default
# unicode block characters render as garbage in cmd.exe ("â-^â-^…") and
# spam the log. Disable them once at import time.
# ──────────────────────────────────────────────────────────────────────
def _quiet_progress_bars() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from huggingface_hub.utils import disable_progress_bars  # type: ignore
        disable_progress_bars()
    except Exception:
        pass
    try:
        # transformers ships its own tqdm wrapper for from_pretrained etc.
        # This is the source of the "Loading weights: 100%|ΓûêΓûê...|" garbage
        # that mojibakes inside cmd.exe. Disable just transformers' bar so
        # sampler / other progress bars keep working.
        from transformers.utils import logging as _tlog  # type: ignore
        _tlog.disable_progress_bar()
        _tlog.set_verbosity_warning()
    except Exception:
        pass


_quiet_progress_bars()


# ──────────────────────────────────────────────────────────────────────
# Curated downloadable presets per backend. Shown in the model dropdown
# as ``[preset:<key>] <filename>``; ``auto_download=True`` fetches them
# on demand into the backend's models folder.
# ──────────────────────────────────────────────────────────────────────
_PRESETS: Dict[str, List[Dict[str, str]]] = {
    "sam2": [
        {"name": "sam2.1_hiera_tiny.safetensors",
         "url":  "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2.1_hiera_tiny.safetensors"},
        {"name": "sam2.1_hiera_small.safetensors",
         "url":  "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2.1_hiera_small.safetensors"},
        {"name": "sam2.1_hiera_base_plus.safetensors",
         "url":  "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2.1_hiera_base_plus.safetensors"},
        {"name": "sam2.1_hiera_large.safetensors",
         "url":  "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2.1_hiera_large.safetensors"},
    ],
    "sam3": [
        {"name": "sam3.safetensors",
         "url":  "https://huggingface.co/facebook/sam3/resolve/main/sam3.safetensors"},
    ],
    "sam3.1": [
        # Comfy-Org reupload (extra-dependency-free re-implementation, single-file).
        {"name": "sam3.1_multiplex_fp16.safetensors",
         "url":  "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"},
        {"name": "sam3.1_multiplex_fp32.safetensors",
         "url":  "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex.safetensors"},
        # 1038lab reupload — same architecture, native key format. Used by comfyui-sam3.
        {"name": "sam3.pt",
         "url":  "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"},
    ],
    "sec": [
        {"name": "SeC-4B.safetensors",
         "url":  "https://huggingface.co/OpenIXCLab/SeC-4B/resolve/main/model.safetensors"},
    ],
    "vitmatte": [
        {"name": "vitmatte-small-composition-1k.safetensors",
         "url":  "https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/model.safetensors"},
        {"name": "vitmatte-base-composition-1k.safetensors",
         "url":  "https://huggingface.co/hustvl/vitmatte-base-composition-1k/resolve/main/model.safetensors"},
    ],
    "rvm": [
        {"name": "rvm_mobilenetv3.pth",
         "url":  "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"},
        {"name": "rvm_resnet50.pth",
         "url":  "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth"},
    ],
    "matanyone": [
        # Primary: GitHub release (public, no token). This is the original
        # weight file shipped with v1.0.0 and used by the vendored
        # MatAnyone2 hugging_face/app.py.
        {"name": "matanyone.pth",
         "url":  "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"},
        # Alt: HF mirror — same weights re-published as safetensors on the
        # public PeiqingYang/MatAnyone repo (141 MB, 'model.safetensors').
        # Saved under the matanyone backend root with its original HF name.
        {"name": "model.safetensors",
         "url":  "https://huggingface.co/PeiqingYang/MatAnyone/resolve/main/model.safetensors"},
    ],
    "birefnet": [
        {"name": "BiRefNet-general-epoch_244.pth",
         "url":  "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/BiRefNet-general-epoch_244.pth"},
    ],
    "rmbg": [
        {"name": "RMBG-2.0.safetensors",
         "url":  "https://huggingface.co/briaai/RMBG-2.0/resolve/main/model.safetensors"},
    ],
    "grounding-dino": [
        {"name": "groundingdino_swint_ogc.pth",
         "url":  "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"},
    ],
    "inspyrenet": [
        {"name": "InSPyReNet_SwinB.pth",
         "url":  "https://huggingface.co/Plachta/InSPyReNet/resolve/main/InSPyReNet_SwinB.pth"},
    ],
}


def list_backend_presets(key: str) -> List[Dict[str, str]]:
    return list(_PRESETS.get(key, []))


def get_preset_url(key: str, name: str) -> Optional[str]:
    for p in _PRESETS.get(key, []):
        if p.get("name") == name:
            return p.get("url")
    return None


def download_preset(key: str, name: str, *, force: bool = False) -> str:
    """Download preset ``name`` for backend ``key`` into its models folder."""
    url = get_preset_url(key, name)
    if not url:
        raise ValueError(f"No preset '{name}' registered for backend '{key}'.")
    dst_dir = backend_first_root(key)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst) and not force:
        return dst
    logger.info("[MaskMatting] downloading preset %s/%s from %s", key, name, url)
    tmp = dst + ".part"
    try:
        if "huggingface.co" in url:
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
                _, after = url.split("huggingface.co/", 1)
                parts = after.split("/resolve/")
                repo = parts[0]
                rel = parts[1].split("/", 1)[1] if "/" in parts[1] else parts[1]
                cached = hf_hub_download(repo_id=repo, filename=rel)
                import shutil
                shutil.copy2(cached, dst)
                return dst
            except Exception as e:
                logger.warning("[MaskMatting] hf_hub_download failed (%s); falling back to urllib", e)
        import urllib.request
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dst)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return dst


_WEIGHT_EXTS = (".pt", ".pth", ".safetensors", ".bin", ".onnx", ".ckpt")
_SKIP_FRAGMENTS = (".cache", "/cache/", "\\cache\\", ".metadata", ".lock",
                   ".gitignore", ".gitattributes", "incomplete", "/.locks/",
                   "\\.locks\\", "/snapshots/", "\\snapshots\\", "blobs/")


def list_backend_files(key: str) -> List[str]:
    """Return real installed weight filenames for the given backend key.

    Filters out HF cache subdirs, ``.metadata``/``.lock`` siblings,
    ``.gitignore`` files, etc. Only keeps files whose extension matches a
    known weight format.
    """
    try:
        raw = list(folder_paths.get_filename_list(key))
    except Exception:
        return []
    out: List[str] = []
    for f in raw:
        low = f.lower().replace("\\", "/")
        if any(s in low for s in _SKIP_FRAGMENTS):
            continue
        if not low.endswith(_WEIGHT_EXTS):
            continue
        out.append(f)
    return out


def resolve_backend_weight(key: str, name: str) -> Optional[str]:
    """Resolve ``name`` against backend folder ``key``; returns absolute path or None."""
    try:
        return folder_paths.get_full_path(key, name)
    except Exception:
        return None


def backend_first_root(key: str) -> str:
    """Return the first registered root folder for backend ``key``."""
    try:
        roots = folder_paths.get_folder_paths(key)
        if roots:
            return roots[0]
    except Exception:
        pass
    return os.path.join(folder_paths.models_dir, _BACKEND_FOLDERS.get(key, key))


# ──────────────────────────────────────────────────────────────────────
# Tensor helpers
# ──────────────────────────────────────────────────────────────────────
def to_bhwc(image: torch.Tensor) -> torch.Tensor:
    """Coerce IMAGE input to (B,H,W,C) float in [0,1]."""
    t = image
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.shape[-1] not in (1, 3, 4) and t.shape[1] in (1, 3, 4):
        t = t.permute(0, 2, 3, 1).contiguous()
    return t.clamp(0, 1).float()


def to_mask(mask: torch.Tensor) -> torch.Tensor:
    """Coerce MASK input to (B,H,W) float in [0,1]."""
    m = mask
    if m.ndim == 4:
        if m.shape[1] == 1:
            m = m.squeeze(1)
        elif m.shape[-1] == 1:
            m = m.squeeze(-1)
        else:
            m = m.mean(dim=1)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    return m.clamp(0, 1).float()


def np_to_bhwc(img_np: np.ndarray) -> torch.Tensor:
    if img_np.ndim == 3:
        img_np = img_np[None, ...]
    return torch.from_numpy(np.ascontiguousarray(img_np.astype(np.float32))).clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────
# Prompt parsing — supports both KJ-Nodes and MEC point schemas.
# ──────────────────────────────────────────────────────────────────────
def parse_points(points_json: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return (positive_points, negative_points) in pixel coords.

    Accepts either:
      * KJ format: ``{"positive": [[x,y],...], "negative": [[x,y],...]}``
      * MEC format: ``[{"x":N,"y":N,"label":1|0}, ...]``
    """
    pos: List[Tuple[float, float]] = []
    neg: List[Tuple[float, float]] = []
    if not points_json:
        return pos, neg
    try:
        data = json.loads(points_json) if isinstance(points_json, str) else points_json
    except (json.JSONDecodeError, TypeError):
        return pos, neg
    if isinstance(data, dict) and ("positive" in data or "negative" in data):
        for p in data.get("positive", []) or []:
            try:
                pos.append((float(p[0]), float(p[1])))
            except (IndexError, TypeError, ValueError):
                continue
        for p in data.get("negative", []) or []:
            try:
                neg.append((float(p[0]), float(p[1])))
            except (IndexError, TypeError, ValueError):
                continue
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                x = float(item.get("x", item.get("X", 0)))
                y = float(item.get("y", item.get("Y", 0)))
            except (TypeError, ValueError):
                continue
            label = int(item.get("label", item.get("type", 1) or 1))
            (pos if label != 0 else neg).append((x, y))
    return pos, neg


def parse_bbox(bbox: Any) -> Optional[Tuple[int, int, int, int]]:
    """Return (x0,y0,x1,y1) or None. Accepts BBOX list, JSON string, dict."""
    if bbox is None:
        return None
    try:
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
    except json.JSONDecodeError:
        return None
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            return int(x), int(y), int(x + w), int(y + h)
        if all(k in bbox for k in ("x0", "y0", "x1", "y1")):
            return tuple(int(bbox[k]) for k in ("x0", "y0", "x1", "y1"))
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        a, b, c, d = bbox[0], bbox[1], bbox[2], bbox[3]
        # Heuristic: xywh if c,d look like sizes (positive, smaller than 4096)
        try:
            if isinstance(a, (list, tuple)):  # nested list of bboxes
                a, b, c, d = a[0], a[1], a[2], a[3]
            return int(a), int(b), int(c), int(d)
        except (TypeError, ValueError):
            return None
    return None


# ──────────────────────────────────────────────────────────────────────
# Morphological primitives (reflect-padded to prevent boundary artifacts)
# ──────────────────────────────────────────────────────────────────────
def _morph_pool(x: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """Reflect-padded morphological pool for a (B,1,H,W) tensor.

    Uses ``reflect`` padding instead of zero-padding to prevent erosion
    from eating into mask regions that touch the image boundary.
    """
    if k <= 0:
        return x
    size = 2 * int(k) + 1
    pad = int(k)
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    if mode == "max":
        return F.max_pool2d(x_padded, kernel_size=size, stride=1, padding=0)
    return 1.0 - F.max_pool2d(
        1.0 - x_padded, kernel_size=size, stride=1, padding=0,
    )


def morph_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological erosion with reflect padding.  Accepts (B,H,W) or (B,1,H,W)."""
    m = to_mask(mask)
    squeezed = m.unsqueeze(1)
    out = _morph_pool(squeezed, radius, "min")
    return out.squeeze(1).clamp(0, 1)


def morph_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological dilation with reflect padding.  Accepts (B,H,W) or (B,1,H,W)."""
    m = to_mask(mask)
    squeezed = m.unsqueeze(1)
    out = _morph_pool(squeezed, radius, "max")
    return out.squeeze(1).clamp(0, 1)


def morph_open(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological opening (erode → dilate).  Removes small bright spots."""
    return morph_dilate(morph_erode(mask, radius), radius)


def morph_close(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological closing (dilate → erode).  Fills small dark holes."""
    return morph_erode(morph_dilate(mask, radius), radius)


def morph_gradient(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological gradient (dilate − erode).  Highlights edges."""
    return (morph_dilate(mask, radius) - morph_erode(mask, radius)).clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────
# Trimap generation
# ──────────────────────────────────────────────────────────────────────
def mask_to_trimap(mask: torch.Tensor, dilate: int = 8, erode: int = 8) -> torch.Tensor:
    """Build a 3-band trimap: 0=bg, 0.5=unknown, 1=fg from a binary mask.

    Uses reflect-padded morphological ops so boundary regions are handled
    correctly even when the subject touches the image edge.
    """
    m = to_mask(mask)
    b = (m > 0.5).float().unsqueeze(1)  # (B,1,H,W)

    fg = _morph_pool(b, int(erode), "min")       # eroded core
    outer = _morph_pool(b, int(dilate), "max")   # dilated outer
    unknown = (outer - fg).clamp(0, 1)
    trimap = fg + unknown * 0.5
    return trimap.squeeze(1).clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────
# Interrupt-aware iteration (per project permanent rules)
# ──────────────────────────────────────────────────────────────────────
def interruptible_range(n: int, label: str = "frame"):
    """Yield 0..n-1, raising on user interrupt and printing ETA each step."""
    import time
    try:
        import comfy.model_management as mm
    except Exception:
        mm = None
    t0 = time.time()
    for i in range(n):
        if mm is not None:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                raise
        yield i
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (n - i - 1) / max(rate, 1e-6)
        if n > 1:
            logger.info(
                "[MaskMatting] %s %d/%d — %.2fs elapsed — ETA %.2fs — %.2f it/s",
                label, i + 1, n, elapsed, eta, rate,
            )


# ──────────────────────────────────────────────────────────────────────
# VRAM cleanup helper (for try/finally blocks)
# ──────────────────────────────────────────────────────────────────────
def free_vram(unload_models: bool = False) -> None:
    import gc
    try:
        import comfy.model_management as mm
        if unload_models:
            mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# Subject-aware trimap presets
# ──────────────────────────────────────────────────────────────────────
SUBJECT_PRESETS: Dict[str, Dict[str, int]] = {
    "custom":     {"dilate": 8,  "erode": 8,  "edge": 4},
    "hair":       {"dilate": 24, "erode": 4,  "edge": 12},
    "fur":        {"dilate": 32, "erode": 4,  "edge": 16},
    "cloth":      {"dilate": 6,  "erode": 6,  "edge": 3},
    "skin_face":  {"dilate": 4,  "erode": 4,  "edge": 2},
    "hard_edge":  {"dilate": 2,  "erode": 2,  "edge": 1},
    "soft_glow":  {"dilate": 16, "erode": 2,  "edge": 8},
}


def apply_subject_preset(name: str, dilate: int, erode: int, edge: int) -> Tuple[int, int, int]:
    if name == "custom" or name not in SUBJECT_PRESETS:
        return dilate, erode, edge
    p = SUBJECT_PRESETS[name]
    return p["dilate"], p["erode"], p["edge"]


# ──────────────────────────────────────────────────────────────────────
# Bbox helpers
# ──────────────────────────────────────────────────────────────────────
def bbox_from_mask(mask_2d: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_2d > 0.05)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, mask_2d.shape[1] - 1, mask_2d.shape[0] - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_to_json(b: Tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = b
    return json.dumps({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)})
