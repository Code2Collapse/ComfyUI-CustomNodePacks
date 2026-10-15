"""Viewer helper "Detect" (L7.114): one frame + text / points / boxes -> every object found, each with its own mask.

The viewer's Find box sends comma-separated words ("person, dog"); a click without Find sends points. Each object
comes back as a PNG mask at proxy size (the browser builds its hit map from it), a box, a label and a score.

Model slot (owner answer round 5: "whichever is more stronger"): the SAM checkpoints in models/sam3.1/ and models/sam3/
(the folder ComfyUI-SAM3 and comfyui-rmbg also use) are listed strongest first - SAM 3.1 Multiplex, then SAM 3.1,
then SAM 3. With none on the machine the list holds one entry, "MISSING: <file>", and the run stops with where to
put which file. Nothing is downloaded here. Detection runs through CNP's own SAM code (segmenters/sam31_backend.py;
segmenters/sam3_backend.py for a SAM 3 file when the sam3 package is installed).
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import _common as C

logger = logging.getLogger("C2C.Viewer.Detect")

RECOMMENDED = "sam3.1_multiplex_fp16.safetensors"
MISSING_PREFIX = "MISSING: "
MAX_OBJECTS = 32
SAM_FOLDERS = ("sam3.1", "sam3")


def _rank(name: str) -> Tuple[int, int, str]:
    n = name.lower().replace("\\", "/").rsplit("/", 1)[-1]
    is31 = "3.1" in n or "sam31" in n or "3_1" in n
    strength = 3 if (is31 and "multiplex" in n) else 2 if is31 else 1
    return (-strength, 1 if "fp32" in n else 0, n)          # fp16 first: same model, half the memory


def sam_checkpoints() -> List[Tuple[str, str]]:
    """[(tag, absolute path)] strongest first; tag = "<folder key>/<file>" as the model slot shows it."""
    from ..mask_matting.utils import list_backend_files, resolve_backend_weight
    seen, out = set(), []
    for key in SAM_FOLDERS:
        for f in list_backend_files(key):
            path = resolve_backend_weight(key, f)
            if not path or not os.path.isfile(path):
                continue
            real = os.path.normcase(os.path.realpath(path))
            if real in seen:                                 # the same file reachable through both folders
                continue
            seen.add(real)
            out.append((f"{key}/{f}", path))
    out.sort(key=lambda t: _rank(t[0]))
    return out


def missing_info() -> Dict[str, str]:
    from ..mask_matting.utils import backend_first_root, get_preset_url
    return {"file": RECOMMENDED, "folder": os.path.abspath(backend_first_root("sam3.1")),
            "url": get_preset_url("sam3.1", RECOMMENDED) or "https://huggingface.co/Comfy-Org/sam3.1"}


def model_choices() -> List[str]:
    try:
        found = [t for t, _p in sam_checkpoints()]
    except Exception as exc:  # noqa: BLE001 - a broken folder list must not take the node definition down
        logger.warning("[C2C viewer] SAM checkpoint scan failed: %s", exc)
        found = []
    return found or [MISSING_PREFIX + RECOMMENDED]


def missing_sentence(info: Optional[Dict[str, str]] = None) -> str:
    i = info or missing_info()
    return (f"No SAM 3 / 3.1 checkpoint is installed, so the viewer cannot find objects. Put {i['file']} in "
            f"{i['folder']} (download: {i['url']}), then press R in ComfyUI to refresh the node list. The matte "
            "itself still runs without it (model-free segmenter, approximate).")


def resolve_model(choice: str) -> Tuple[str, str]:
    """A model-slot value -> (tag, path). MISSING or a file that is gone: the sentence above."""
    if not choice or str(choice).startswith(MISSING_PREFIX):
        raise ValueError(missing_sentence())
    for tag, path in sam_checkpoints():
        if tag == choice:
            return tag, path
    raise ValueError(f"The detection model {choice} is no longer on this machine. Press R in ComfyUI to refresh the "
                     "list and pick another one.")


# ── the detector (cached: Find after Find does not reload SAM) ──────────────────────────────────────────────────

class _SegmenterDetector:
    """detect(frame, text, pos, neg, boxes) over a CNP segmenter: its per-object output when it has one, else the
    union mask as one object."""

    def __init__(self, seg):
        self.seg = seg

    def detect(self, frame_hwc: np.ndarray, text: str, pos, neg, boxes) -> List[Dict[str, Any]]:
        if hasattr(self.seg, "detect_objects"):
            return self.seg.detect_objects(frame_hwc, text=text, pos=pos, neg=neg, boxes=boxes)
        out = self.seg.segment(torch.from_numpy(frame_hwc)[None].float(), mode="auto", positive_points=pos,
                               negative_points=neg, bbox=boxes[0] if boxes else None, text_prompt=text)
        m = out["mask"][0].float().cpu().numpy()
        return [{"mask": m, "box": None, "score": float(out.get("score", 0.0))}] if m.max() > 0.5 else []

    def unload(self):
        try:
            self.seg.unload()
        except Exception:  # noqa: BLE001
            pass


def _default_factory(tag: str, path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if device == "cuda" else "fp32"
    from ..mask_matting.segmenters.sam31_backend import SAM31Segmenter, _have_vendor, _vendor_root
    if _have_vendor():
        seg = SAM31Segmenter(model_name=path, device=device, precision=precision, attention="auto", offload="none")
        seg.load()
        return _SegmenterDetector(seg)
    from ..mask_matting.segmenters.sam3_backend import SAM3Segmenter, _have_sam3
    if tag.startswith("sam3/") and _have_sam3():
        seg = SAM3Segmenter(model_name=tag.split("/", 1)[1], device=device, precision=precision, attention="auto",
                            offload="none")
        seg.load()
        return _SegmenterDetector(seg)
    raise RuntimeError(
        f"The SAM 3.1 library is not installed (looked for it in {_vendor_root()}; it is not shipped with "
        "ComfyUI-CustomNodePacks) and the sam3 Python package is not installed, so the viewer cannot find objects "
        f"with {tag}. The matte itself still runs (model-free segmenter, approximate).")


#: tests replace this with a fake; signature (tag, path) -> object with detect(...) and unload()
DETECTOR_FACTORY: Callable[[str, str], Any] = _default_factory
_lock = threading.Lock()
_loaded: Dict[str, Any] = {}


def _release() -> None:
    with _lock:
        for d in _loaded.values():
            try:
                d.unload()
            except Exception:  # noqa: BLE001
                pass
        _loaded.clear()


def get_detector(tag: str, path: str):
    with _lock:
        det = _loaded.get(path)
        if det is not None:
            return det
    det = DETECTOR_FACTORY(tag, path)
    with _lock:
        for other in list(_loaded):                     # one detection model in memory at a time
            try:
                _loaded.pop(other).unload()
            except Exception:  # noqa: BLE001
                pass
        _loaded[path] = det
    try:
        from .._c2c_memguard import register
        register("c2c_viewer_detect", _release, lambda: bool(_loaded))
    except Exception:  # noqa: BLE001
        pass
    return det


class C2CViewerDetect:
    DEV_ONLY = True
    CATEGORY = C.HIDDEN_CATEGORY
    DESCRIPTION = ("Viewer helper (hidden): finds objects on one frame (text, points or boxes) for the Mask + "
                   "Matting viewer and writes one mask per object. Not meant to be added by hand.")
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        choices = model_choices()
        missing = choices[0].startswith(MISSING_PREFIX)
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The plate."}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "tooltip": "Frame to look at."}),
                "model": (choices, {
                    "default": choices[0],
                    "tooltip": "SAM checkpoint for Find and clicks, strongest first (SAM 3.1 Multiplex > 3.1 > 3), "
                               "from models/sam3.1 and models/sam3.",
                    # the viewer shows MISSING with the folder and the link (no route needed: it reads this)
                    "c2c_missing": missing_info() if missing else None,
                }),
            },
            "optional": {
                "text": ("STRING", {"default": "", "tooltip": "What to find, comma-separated (person, dog)."}),
                "points": ("STRING", {"default": "", "tooltip": 'Clicks: [{"x":0.4,"y":0.5,"label":1}] in 0..1.'}),
                "boxes": ("STRING", {"default": "", "tooltip": "Boxes: [[x0,y0,x1,y1]] in 0..1."}),
                "max_side": ("INT", {"default": C.MAX_SIDE, "min": 64, "max": 2048, "step": 16}),
            },
        }

    def execute(self, image, frame, model, text="", points="", boxes="", max_side=C.MAX_SIDE):
        from ..mask_matting._stage_cache import fingerprint
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        f = C.frame_index(frame, B)
        tag, path = resolve_model(model)
        pos, neg = C.normalised_points(points, W, H)
        bxs = C.normalised_boxes(boxes, W, H)
        terms = C.split_terms(text)
        if not terms and not (pos or neg or bxs):
            raise ValueError("Type what to find (for example: person) or click the subject, then try again.")
        det = get_detector(tag, path)
        frame_np = image[f].detach().float().cpu().numpy()
        objects: List[Dict[str, Any]] = []
        with torch.no_grad():
            for term in terms:
                for o in det.detect(frame_np, term, [], [], []):
                    objects.append({**o, "label": term})
            if pos or neg or bxs:
                for o in det.detect(frame_np, "", pos, neg, bxs):
                    objects.append({**o, "label": o.get("label") or "clicked"})
        objects = [o for o in objects if np.asarray(o["mask"]).max() > 0.5]
        objects.sort(key=lambda o: -float(o.get("score", 0.0)))
        objects = objects[:MAX_OBJECTS]

        pw, ph = C.proxy_size(W, H, int(max_side))
        key = fingerprint(image[f])[:12] + "_" + hashlib.sha1(
            f"{tag}|{text}|{points}|{boxes}|{pw}".encode("utf-8")).hexdigest()[:8]
        dtag = f"detect_{key}"
        d = C.out_dir(dtag)
        out = []
        for k, o in enumerate(objects):
            m = torch.from_numpy(np.asarray(o["mask"], dtype=np.float32))
            name = f"o{k:02d}.png"
            C.save_alpha_png(C.resize_mask(m, (pw, ph)), os.path.join(d, name))
            box = o.get("box") or C.bbox_of(np.asarray(o["mask"]))
            out.append({"id": k, "label": str(o.get("label") or ""), "score": round(float(o.get("score", 0.0)), 4),
                        "box": [round(box[0] / W, 5), round(box[1] / H, 5), round(box[2] / W, 5),
                                round(box[3] / H, 5)] if box else None,
                        "area": round(float((m > 0.5).float().mean()), 6), "mask": C.file_ref(dtag, name)})
        message = ""
        if not out:
            what = f"'{text.strip()}'" if terms else "the click"
            message = f"Nothing found for {what} on frame {f}. Try other words, or click the subject."
        return C.ui({"kind": "detect", "frame": f, "width": W, "height": H, "proxy": [pw, ph], "model": tag,
                     "objects": out, "message": message})
