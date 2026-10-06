"""Smart selection backends for Image Mask Editor (SAM + ViTMatte refine).

Route handlers call the handle_* functions with plain Python values; tests call them
directly without aiohttp. All model work runs on a dedicated single-worker executor.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from . import store

log = logging.getLogger("C2C.ImageMaskEditor.Smart")

_MP_LIMIT = 16_000_000
_MAX_PNG_BYTES = 64 * 1024 * 1024
_MAX_DIM = 16_384
_FRAME_KEY_RE = re.compile(r"^[a-f0-9]{40}$")
_IDLE_SECONDS = 600

_BACKENDS: Dict[str, Any] = {}
_LOCK = threading.RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ime-smart")

_embeddings: Dict[str, dict] = {}
_models: Dict[str, dict] = {}
_editor_models: Dict[str, str] = {}
_last_seq: Dict[Tuple[str, str], int] = {}

_loop = None
_idle_handle = None


def get_executor() -> ThreadPoolExecutor:
    return _EXECUTOR


def init_loop(loop) -> None:
    global _loop
    _loop = loop


def register_backend(kind: str, obj: Any) -> None:
    """Replace a backend (tests, the testbed stub pack, other packs).

    sam:   load(model_name) -> handle; embed(handle, rgb_u8 HxWx3) -> embedding handle (a backend that needs
           the image later keeps it in this handle - the server keeps no frame copy); predict(embedding,
           coords Nx2 | None, labels N | None, box 4 | None) -> (mask u8 HxW, score); release(handle).
    matte: available() -> (ok, message); matte(rgb_u8 HxWx3, trimap f32 HxW in {0, 0.5, 1}) -> alpha f32 HxW.
    """
    with _LOCK:
        _BACKENDS[kind] = obj


def get_backend(kind: str) -> Any:
    with _LOCK:
        if kind not in _BACKENDS:
            if kind == "sam":
                _BACKENDS[kind] = _DefaultSamBackend()
            elif kind == "matte":
                _BACKENDS[kind] = _DefaultMatteBackend()
            else:
                raise KeyError(f"unknown backend kind: {kind}")
        return _BACKENDS[kind]


def reset_state_for_tests() -> None:
    """Clear all cached state (tests only)."""
    global _idle_handle
    with _LOCK:
        _embeddings.clear()
        _models.clear()
        _editor_models.clear()
        _last_seq.clear()
        if _idle_handle is not None:
            try:
                _idle_handle.cancel()
            except Exception:
                pass
            _idle_handle = None
        for kind in list(_BACKENDS.keys()):
            be = _BACKENDS[kind]
            if hasattr(be, "reset"):
                be.reset()


def _soft_empty_cache() -> None:
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _arm_idle_timer() -> None:
    # The handlers run on the worker thread, and asyncio's call_later / cancel are not thread-safe:
    # re-arm on the loop thread.
    if _loop is None or _loop.is_closed():
        return

    def _fire():
        _EXECUTOR.submit(_idle_cleanup)

    def _rearm():
        global _idle_handle
        if _idle_handle is not None:
            try:
                _idle_handle.cancel()
            except Exception:
                pass
        _idle_handle = _loop.call_later(_IDLE_SECONDS, _fire)

    try:
        _loop.call_soon_threadsafe(_rearm)
    except RuntimeError:        # loop closed during shutdown
        pass


def _idle_cleanup() -> None:
    with _LOCK:
        for eid in list(_embeddings.keys()):
            _drop_embedding_unlocked(eid)
        for name in list(_models.keys()):
            if _models[name]["refcount"] <= 0:
                _release_model_unlocked(name)
    _soft_empty_cache()


def _touch_activity() -> None:
    _arm_idle_timer()


def _decode_png_l(png_bytes: bytes) -> np.ndarray:
    if not png_bytes or len(png_bytes) > _MAX_PNG_BYTES:
        raise ValueError("empty or too-large PNG payload")
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes))
    if im.mode != "L":
        im = im.convert("L")
    arr = np.array(im, dtype=np.uint8)
    h, w = arr.shape[:2]
    if h > _MAX_DIM or w > _MAX_DIM:
        raise ValueError(f"image dimensions exceed {_MAX_DIM}")
    return arr


def _decode_png_rgb(png_bytes: bytes) -> np.ndarray:
    if not png_bytes or len(png_bytes) > _MAX_PNG_BYTES:
        raise ValueError("empty or too-large PNG payload")
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes))
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    h, w = arr.shape[:2]
    if h > _MAX_DIM or w > _MAX_DIM:
        raise ValueError(f"image dimensions exceed {_MAX_DIM}")
    if h * w > _MP_LIMIT:
        raise ValueError(f"image exceeds {_MP_LIMIT} pixels")
    return arr


def _encode_png_l(arr: np.ndarray) -> bytes:
    from PIL import Image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


def _parse_meta_json(raw: Union[str, bytes, dict]) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _check_seq(editor_id: str, kind: str, seq: int) -> Optional[Tuple[int, dict, dict]]:
    """Return error response if superseded, else None."""
    key = (editor_id, kind)
    with _LOCK:
        prev = _last_seq.get(key, -1)
        if seq < prev:
            return 409, {"superseded": True}, {}
        _last_seq[key] = seq
    return None


def _acquire_model(model_name: str) -> Any:
    with _LOCK:
        entry = _models.get(model_name)
        if entry is None:
            backend = get_backend("sam")
            handle = backend.load(model_name)
            entry = {"handle": handle, "refcount": 0}
            _models[model_name] = entry
        entry["refcount"] += 1
        return entry["handle"]


def _release_model_for_editor(editor_id: str) -> None:
    with _LOCK:
        model_name = _editor_models.pop(editor_id, None)
        if not model_name:
            return
        entry = _models.get(model_name)
        if entry is None:
            return
        entry["refcount"] = max(0, entry["refcount"] - 1)
        if entry["refcount"] <= 0:
            _release_model_unlocked(model_name)


def _release_model_unlocked(model_name: str) -> None:
    entry = _models.pop(model_name, None)
    if entry is None:
        return
    try:
        get_backend("sam").release(entry["handle"])
    except Exception as exc:
        log.debug("SAM model release failed: %s", exc)


def _drop_embedding_unlocked(editor_id: str) -> None:
    _embeddings.pop(editor_id, None)
    model_name = _editor_models.get(editor_id)
    if model_name:
        entry = _models.get(model_name)
        if entry is not None:
            entry["refcount"] = max(0, entry["refcount"] - 1)
            if entry["refcount"] <= 0:
                _release_model_unlocked(model_name)
        _editor_models.pop(editor_id, None)


def _validate_sam_meta(meta: dict) -> Tuple[str, str, int, int, int, str, list, Optional[list], list]:
    eid = store.validate_editor_id(str(meta.get("editor_id", "")))
    frame_key = str(meta.get("frame_key", "")).strip().lower()
    if not _FRAME_KEY_RE.fullmatch(frame_key):
        raise ValueError("invalid frame_key")
    try:
        seq = int(meta.get("seq", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("seq required") from exc
    if seq < 0:
        raise ValueError("seq required")
    try:
        w = int(meta["width"])
        h = int(meta["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("width and height required") from exc
    if w <= 0 or h <= 0 or w > _MAX_DIM or h > _MAX_DIM:
        raise ValueError("invalid width or height")
    if w * h > _MP_LIMIT:
        raise ValueError(f"image exceeds {_MP_LIMIT} pixels")
    model = str(meta.get("model", "")).strip()
    if not model:
        raise ValueError("model required")
    points_raw = meta.get("points") or []
    if not isinstance(points_raw, list):
        raise ValueError("points must be a list")
    points: list = []
    for pt in points_raw:
        if not isinstance(pt, (list, tuple)) or len(pt) < 3:
            raise ValueError("each point must be [x, y, label]")
        x, y, lab = float(pt[0]), float(pt[1]), int(pt[2])
        if x < 0 or y < 0 or x >= w or y >= h:
            raise ValueError("point outside frame")
        if lab not in (0, 1):
            raise ValueError("point label must be 0 or 1")
        points.append([x, y, lab])
    box = meta.get("box")
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("box must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if x1 < x0 or y1 < y0:
            raise ValueError("box must be ordered")
        box = [x0, y0, x1, y1]
    return eid, frame_key, seq, w, h, model, points, box


def handle_sam(meta_raw: Union[str, bytes, dict], image_png: Optional[bytes] = None):
    """Run SAM predict. Returns (status, body, headers)."""
    _touch_activity()
    try:
        meta = _parse_meta_json(meta_raw)
        eid, frame_key, seq, w, h, model, points, box = _validate_sam_meta(meta)
    except ValueError as exc:
        return 400, {"error": str(exc)}, {}

    superseded = _check_seq(eid, "sam", seq)
    if superseded:
        return superseded

    cache_key = (frame_key, model)
    need_embed = False
    with _LOCK:
        emb = _embeddings.get(eid)
        if emb is None or emb.get("key") != cache_key or emb.get("w") != w or emb.get("h") != h:
            need_embed = True

    if need_embed:
        if not image_png:
            return 409, {"need_image": True}, {}
        try:
            rgb = _decode_png_rgb(image_png)
        except ValueError as exc:
            return 400, {"error": str(exc)}, {}
        rh, rw = rgb.shape[:2]
        if rh != h or rw != w:
            return 400, {"error": "image size does not match width/height in meta"}, {}
        backend = get_backend("sam")
        err = getattr(backend, "runtime_error", lambda _m: None)(model)
        if err:
            return 400, {"error": err}, {}
        try:
            with _LOCK:
                old_model = _editor_models.get(eid)
                if old_model and old_model != model:
                    entry = _models.get(old_model)
                    if entry is not None:
                        entry["refcount"] = max(0, entry["refcount"] - 1)
                        if entry["refcount"] <= 0:
                            _release_model_unlocked(old_model)
                if _editor_models.get(eid) != model:
                    model_handle = _acquire_model(model)
                    _editor_models[eid] = model
                else:
                    entry = _models.get(model)
                    model_handle = entry["handle"] if entry else _acquire_model(model)
            backend = get_backend("sam")
            emb_handle = backend.embed(model_handle, rgb)
        except Exception as exc:
            log.exception("SAM embed failed")
            return 400, {"error": str(exc)}, {}
        with _LOCK:
            _embeddings[eid] = {
                "key": cache_key,
                "frame_key": frame_key,
                "model": model,
                "w": w,
                "h": h,
                "emb": emb_handle,
            }

    with _LOCK:
        emb = _embeddings.get(eid)
        if emb is None:
            return 409, {"need_image": True}, {}

    labels = [int(p[2]) for p in points]
    coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32) if points else None
    label_arr = np.array(labels, dtype=np.int32) if labels else None
    box_np = np.array(box, dtype=np.float32) if box else None

    try:
        backend = get_backend("sam")
        mask_u8, score = backend.predict(
            emb["emb"], coords, label_arr, box_np,
        )
    except Exception as exc:
        log.exception("SAM predict failed")
        return 400, {"error": str(exc)}, {}

    if mask_u8.shape[0] != h or mask_u8.shape[1] != w:
        return 400, {"error": "SAM returned unexpected mask size"}, {}

    png = _encode_png_l(mask_u8)
    return 200, png, {"X-IME-Score": str(float(score)), "Content-Type": "image/png"}


def _validate_refine_meta(meta: dict) -> Tuple[str, int, list]:
    eid = store.validate_editor_id(str(meta.get("editor_id", "")))
    try:
        seq = int(meta.get("seq", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("seq required") from exc
    if seq < 0:
        raise ValueError("seq required")
    bbox = meta.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if x1 < x0 or y1 < y0:
        raise ValueError("bbox must be ordered")
    return eid, seq, [x0, y0, x1, y1]


def _build_trimap(mask_u8: np.ndarray, band_u8: np.ndarray) -> np.ndarray:
    tri = np.zeros(mask_u8.shape, dtype=np.float32)
    tri[(band_u8 > 0)] = 0.5
    tri[(band_u8 == 0) & (mask_u8 >= 128)] = 1.0
    return tri


def handle_refine(
    meta_raw: Union[str, bytes, dict],
    image_png: bytes,
    mask_png: bytes,
    band_png: bytes,
):
    """Run ViTMatte on a crop. Returns (status, body, headers)."""
    _touch_activity()
    try:
        meta = _parse_meta_json(meta_raw)
        eid, seq, _bbox = _validate_refine_meta(meta)
    except ValueError as exc:
        return 400, {"error": str(exc)}, {}

    superseded = _check_seq(eid, "refine", seq)
    if superseded:
        return superseded

    try:
        rgb = _decode_png_rgb(image_png)
        mask = _decode_png_l(mask_png)
        band = _decode_png_l(band_png)
    except ValueError as exc:
        return 400, {"error": str(exc)}, {}

    h, w = rgb.shape[:2]
    if mask.shape != (h, w) or band.shape != (h, w):
        return 400, {"error": "image, mask and band must share the same size"}, {}
    if h * w > _MP_LIMIT:
        return 400, {"error": f"crop exceeds {_MP_LIMIT} pixels"}, {}

    if not np.any(band > 0):
        return 200, mask_png, {"Content-Type": "image/png"}

    matte = get_backend("matte")
    ok, msg = matte.available()
    if not ok:
        return 409, {"error": msg, "need_weights": True}, {}

    tri = _build_trimap(mask, band)
    try:
        alpha = matte.matte(rgb, tri)
    except Exception as exc:
        log.exception("refine matte failed")
        return 400, {"error": str(exc)}, {}

    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return 200, _encode_png_l(alpha_u8), {"Content-Type": "image/png"}


def handle_release(editor_id: str):
    _touch_activity()
    try:
        eid = store.validate_editor_id(editor_id)
    except ValueError as exc:
        return 400, {"error": str(exc)}, {}
    with _LOCK:
        _drop_embedding_unlocked(eid)
        for kind in ("sam", "refine"):
            _last_seq.pop((eid, kind), None)
    return 200, {"ok": True}, {}


def _sam_model_files() -> List[str]:
    """Same list building as SAMModelLoaderMEC.INPUT_TYPES, minus its combo placeholder.

    The loader node needs a non-empty combo, so it lists "(place model in models/sams/ ...)" when nothing is
    installed. The editor has its own "choose a model" entry; carried over, that placeholder became a
    selectable "model" whose first click failed (seen in the L7.38 slice 2b e2e screenshot)."""
    from ..sam_model_loader import SAMModelLoaderMEC, _DOWNLOAD_REGISTRY

    model_files: List[str] = []
    try:
        import folder_paths
        for key in ("sams", "sam2", "sam3"):
            if key in folder_paths.folder_names_and_paths:
                try:
                    model_files += folder_paths.get_filename_list(key)
                except Exception:
                    pass
        SAMModelLoaderMEC._scan_extra_paths(model_files)
    except Exception:
        pass
    model_files = [n for n in model_files if not n.startswith("(")]
    for name in _DOWNLOAD_REGISTRY:
        if name not in model_files:
            model_files.append(f"[download] {name}")
    return sorted(set(model_files))


def _is_installed_model(name: str) -> bool:
    if name.startswith("[download]"):
        return False
    if name.startswith("("):
        return False
    try:
        from ..sam_model_loader import SAMModelLoaderMEC
        SAMModelLoaderMEC._resolve_path(name)
        return True
    except Exception:
        return False


def _installed_size(name: str) -> Optional[int]:
    try:
        from ..sam_model_loader import SAMModelLoaderMEC
        path = SAMModelLoaderMEC._resolve_path(name)
        if path and os.path.isfile(path):
            return os.path.getsize(path)
    except Exception:
        pass
    return None


def list_sam_models() -> dict:
    names = _sam_model_files()
    installed = [n for n in names if _is_installed_model(n)]
    download = [n for n in names if n not in installed]
    ordered = installed + download
    # No installed model -> no default: a "[download]" entry is fetched only when the user picks it.
    default = ""
    if installed:
        sized = [(n, _installed_size(n) or 0) for n in installed]
        sized.sort(key=lambda t: (t[1], t[0]))
        default = sized[0][0]
    backend = get_backend("sam")
    reason = getattr(backend, "runtime_error", lambda _m: None)(ordered[0] if ordered else "")
    unavailable = {}
    if reason:
        unavailable = {name: reason for name in ordered}
        default = ""
    result = {"models": ordered, "default": default}
    if unavailable:
        result["unavailable"] = unavailable
    return result


def handle_sam_models():
    return 200, list_sam_models(), {}


# ── Default SAM backend ───────────────────────────────────────────────


class _DefaultSamBackend:
    """Wraps SAMModelLoaderMEC + nodes.utils SAM predict helpers."""

    def __init__(self):
        self._load_cache: Dict[str, Any] = {}

    def reset(self) -> None:
        self._load_cache.clear()

    def runtime_error(self, model_name: str) -> Optional[str]:
        import importlib.util
        if importlib.util.find_spec("sam2") is not None:
            return None
        return (
            "SAM 2.1 needs the 'sam2' Python package, which this ComfyUI does not have. "
            "Install it with: pip install git+https://github.com/facebookresearch/sam2.git "
            "- then restart ComfyUI."
        )

    def load(self, model_name: str):
        if model_name in self._load_cache:
            return self._load_cache[model_name]
        import torch
        from ..sam_model_loader import SAMModelLoaderMEC

        device = "cpu"
        try:
            import comfy.model_management as mm
            device = str(mm.get_torch_device().type)
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "float16" if device == "cuda" else "float32"
        wrapper = SAMModelLoaderMEC().load(
            model_name,
            "auto",
            device,
            True,  # prefer offload_to_cpu (RAM rule)
            dtype,
        )[0]
        self._load_cache[model_name] = wrapper
        return wrapper

    def embed(self, model_wrapper: dict, rgb_u8: np.ndarray):
        from ..utils import get_sam_predictor

        model = model_wrapper.get("model")
        model_type = model_wrapper.get("model_type", "sam2.1")
        predictor = get_sam_predictor(model, model_type, rgb_u8)
        if predictor is None:
            raise RuntimeError("No compatible SAM predictor for this model.")
        return {"predictor": predictor, "model_wrapper": model_wrapper}

    def predict(self, emb_handle, point_coords, point_labels, box):
        from ..sam_model_loader import move_to_inference_device, restore_device
        from ..utils import sam_predict

        predictor = emb_handle["predictor"]
        model_wrapper = emb_handle["model_wrapper"]
        kwargs = {"multimask_output": False}
        if point_coords is not None and len(point_coords):
            kwargs["point_coords"] = point_coords
            kwargs["point_labels"] = point_labels
        if box is not None:
            kwargs["box"] = box
        try:
            move_to_inference_device(model_wrapper)
            masks_np, scores, _ = sam_predict(predictor, model_wrapper, **kwargs)
        finally:
            restore_device(model_wrapper)
        if masks_np is None or scores is None:
            raise RuntimeError("SAM predict returned no mask.")
        mask = masks_np[0] if masks_np.ndim == 3 else masks_np
        if mask.dtype != np.uint8:
            mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        score = float(scores[0] if hasattr(scores, "__len__") else scores)
        return mask, score

    def release(self, model_wrapper) -> None:
        for key, val in list(self._load_cache.items()):
            if val is model_wrapper:
                del self._load_cache[key]
                break


# ── Default matte backend ─────────────────────────────────────────────


def _vitmatte_local_dir() -> Optional[str]:
    try:
        from ..mask_matting.utils import backend_first_root
    except Exception:
        return None

    def _is_hf_model_dir(p: str) -> bool:
        return (
            isinstance(p, str)
            and os.path.isdir(p)
            and os.path.isfile(os.path.join(p, "preprocessor_config.json"))
            and os.path.isfile(os.path.join(p, "config.json"))
        )

    candidates: list = []
    try:
        root = backend_first_root("vitmatte")
        if root and os.path.isdir(root):
            if _is_hf_model_dir(root):
                candidates.append(root)
            for entry in sorted(os.listdir(root)):
                sub = os.path.join(root, entry)
                if _is_hf_model_dir(sub):
                    candidates.append(sub)
    except Exception:
        pass
    return candidates[0] if candidates else None


class _DefaultMatteBackend:
    def __init__(self):
        self._matter = None
        self._model_dir: Optional[str] = None

    def reset(self) -> None:
        self._matter = None
        self._model_dir = None

    def available(self) -> Tuple[bool, str]:
        if _vitmatte_local_dir() is None:
            return False, (
                "ViTMatte weights not found. Place a HuggingFace model folder "
                "(config.json + preprocessor_config.json) under ComfyUI/models/vitmatte/."
            )
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, "Install transformers: pip install transformers"
        return True, ""

    def _ensure_matter(self):
        if self._matter is not None:
            return
        model_dir = _vitmatte_local_dir()
        if model_dir is None:
            raise RuntimeError("ViTMatte weights not found locally.")
        import torch
        from ..mask_matting.matters.vitmatte_backend import ViTMatteMatter

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._matter = ViTMatteMatter(model_name=model_dir, device=device, precision="fp32")
        self._model_dir = model_dir

    def matte(self, rgb_u8: np.ndarray, trimap_f32: np.ndarray) -> np.ndarray:
        import torch
        from ..mask_matting.utils import np_to_bhwc

        self._ensure_matter()
        h, w = rgb_u8.shape[:2]
        img_f = rgb_u8.astype(np.float32) / 255.0
        img_t = np_to_bhwc(img_f)
        coarse = np.zeros((h, w), dtype=np.float32)
        coarse[trimap_f32 >= 0.75] = 1.0
        coarse_t = torch.from_numpy(coarse).unsqueeze(0)
        tri_t = torch.from_numpy(trimap_f32).unsqueeze(0)
        out = self._matter.matte(
            img_t,
            coarse_t,
            trimap=tri_t,
            edge_radius=4,
            tile_size=512,
            tile_overlap=64,
            tile_batch=4,
        )
        alpha = out["alpha"][0].cpu().numpy().astype(np.float32)
        return alpha
